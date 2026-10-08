"""Trainval-only quality audit. Review flags never become training exclusions.

The policy is written before any image is opened. Images and sheets are written
only below the external data directory; the manifest and raw files are immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from roastml.decode import decode_image
from roastml.features import box_mask
from roastml.paths import data_dir
from roastml.segment import SegConfig, background_white_balance, segment
from tools.imghash import phash_array, thumb_array
from tools.index_sources import read_yolo_names, yolo_label_path
from tools.make_manifest import map_label

ML_DIR = Path(__file__).resolve().parent.parent
FOLDS = ("ontoum224", "rf_robusta", "rf_boos", "agtron")
TARGETS = ("light", "medium", "dark")
POLICY = {
    "id": "qa_20261008_v1", "seed": 20261006,
    "scope": "All manifest split=trainval rows; no split=test image opened.",
    "hard_exclusion": ["image byte read/decode failure", "mandatory Agtron ROI missing or invalid after EXIF"],
    "review_only": ["resolution", "blur", "clipping/exposure", "compression", "annotation mismatch/invalid bbox", "WB guard", "pipeline mask mismatch", "near duplicates", "non-target/ambiguous labels"],
    "label_rule": "No relabeling, no exclusion by absolute color, predicted label or model error. No source is added to/removed from the reference R2 pool.",
    "blur": {"long_side": 256, "texture_std_min": 8, "ratio_fine_coarse_max": 0.8,
             "minimum_laplacian_var_by_original_short_side": {"below128": 8, "128to511": 12, "512plus": 18},
             "meaning": "Conservative review heuristic combining original resolution, texture and multiscale detail; not a calibrated focus classifier."},
    "clipping": {"black_rgb_max": 2, "white_rgb_min": 253, "review_fraction": 0.15,
                 "meaning": "Measured in bean bbox interior when available; review only, especially dark roast."},
    "compression": {"jpeg_mean_quantizer_min": 35, "bytes_per_original_pixel_max": 0.25},
    "near_duplicate": {"phash_hamming_max": 6, "gray64_mad_max": 0.03,
                       "meaning": "Pair candidates; low-texture/same-scene matches require visual confirmation, never automatic exclusions."},
    "pipeline": {"selected_pixels_in_bbox_review_below": 0.5, "diagnostic_bbox_only": True},
    "wb": "Use shared runtime guard; Agtron ROI deliberately disables background WB because bean pile is not a neutral reference. paper_path is not loaded.",
    "holdout": "QA hard exclusions only intersect outer-training masks. Every reference holdout image remains unchanged.",
    "learned_thresholds": "None. Any future learned quality threshold must be fitted in the training fold only.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_trainval(manifest: Path) -> tuple[list[dict], dict]:
    """Read split metadata but return only trainval paths to the image loader."""
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [dict(r) for r in rows if r["split"] == "trainval"]
    if len({r["path"] for r in selected}) != len(selected):
        raise ValueError("trainval manifest paths are not unique")
    frozen = [r for r in rows if r["split"] == "test"]
    md5_train = {r["md5"] for r in selected if r.get("md5")}
    group_train = {r["group"] for r in selected}
    hashes = np.asarray([int(r["phash"], 16) for r in selected if r.get("phash")], dtype=np.uint64)
    return selected, {"rows": len(rows), "split_counts_metadata_only": dict(Counter(r["split"] for r in rows)),
                      "cross_split_exact_md5_metadata_matches": sum(r.get("md5") in md5_train for r in frozen),
                      "cross_split_group_metadata_matches": sum(r["group"] in group_train for r in frozen),
                      "cross_split_phash_metadata_candidates": sum(int(np.count_nonzero(np.bitwise_count(hashes ^ np.uint64(int(r["phash"], 16))) <= 6)) for r in frozen if r.get("phash")),
                      "cross_split_pixel_comparison": "Not performed: frozen-test images are forbidden."}


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes data directory")
    return path


def crop_roi(decoded, row: dict) -> np.ndarray:
    rgb = np.asarray(decoded.image)
    roi = row.get("roi", "")
    if row["source"] == "agtron" and not roi:
        raise ValueError("mandatory_roi_missing")
    if roi:
        box = tuple(int(v) for v in roi.split())
        if len(box) != 4:
            raise ValueError("roi_requires_four_coordinates")
        x0, y0, x1, y1 = box
        w, h = decoded.orig_size
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"roi_outside_exif_image:{box}:{w}x{h}")
        # Match current baseline's decode scaling; report original coordinates too.
        scale = decoded.scale
        xa, ya, xb, yb = [int(round(v * scale)) for v in box]
        rgb = rgb[ya:yb, xa:xb]
        if min(rgb.shape[:2]) < 1:
            raise ValueError("roi_empty_after_decode")
    return rgb


def read_boxes_checked(path: Path) -> tuple[list, list[str]]:
    label_path = yolo_label_path(path)
    if not label_path.is_file():
        return [], ["bbox_annotation_missing"]
    boxes, issues = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("bbox_fields")
            cls = int(fields[0])
            cx, cy, w, h = [float(v) for v in fields[1:]]
            if not np.isfinite([cx, cy, w, h]).all() or w <= 0 or h <= 0:
                raise ValueError("bbox_nonfinite_or_nonpositive")
            if not (0 <= cx - w / 2 <= cx + w / 2 <= 1 and 0 <= cy - h / 2 <= cy + h / 2 <= 1):
                issues.append("bbox_outside_image")
            boxes.append((cls, cx, cy, w, h))
        except (ValueError, TypeError):
            issues.append("bbox_parse_invalid")
    return boxes, sorted(set(issues))


def detail_metrics(rgb: np.ndarray, original_short_side: float) -> dict:
    """Resolution/texture-aware multiscale detail; never uses roast class/color."""
    h, w = rgb.shape[:2]
    side = POLICY["blur"]["long_side"]
    scale = min(1, side / max(h, w))
    small = cv2.resize(rgb, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    coarse = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    coarse_lap = float(cv2.Laplacian(coarse, cv2.CV_32F).var())
    texture = float(gray.std())
    ratio = lap / max(coarse_lap, 1e-6)
    threshold = 8 if original_short_side < 128 else 12 if original_short_side < 512 else 18
    return {"laplacian_var_256": lap, "coarse_laplacian_var": coarse_lap,
            "fine_coarse_ratio": ratio, "texture_std_256": texture,
            "blur_review": texture >= 8 and lap < threshold and ratio < 0.8,
            "blur_laplacian_threshold_for_resolution": threshold,
            "low_texture_review": texture < 8}


def flag(record: dict, name: str, reason: str) -> None:
    if name not in record["flags"]:
        record["flags"].append(name)
    record["reasons"].append(f"{name}: {reason}")


def audit_one(root: Path, row: dict, names: dict) -> tuple[dict, np.ndarray | None]:
    if row["split"] != "trainval":
        raise ValueError("QA loader refuses non-trainval image")
    rec = {**row, "flags": [], "reasons": [], "hard_exclude": False, "hard_exclude_reasons": [],
           "reference_model_pool": row["source"] in FOLDS and row["label"] in TARGETS,
           "n_boxes": 0, "annotation_issues": [], "pipeline_error": ""}
    path = safe_path(root, row["path"])
    try:
        raw = path.read_bytes()
        rec["bytes"] = len(raw)
        rec["md5_actual"] = hashlib.md5(raw).hexdigest()
        if rec["md5_actual"] != row["md5"]:
            flag(rec, "manifest_hash_mismatch", "actual bytes differ from reference md5; do not overwrite reference")
        with Image.open(io.BytesIO(raw)) as header:
            rec["header_size"] = list(header.size)
            try:
                rec["exif_orientation"] = int(header.getexif().get(274, 1))
            except (ValueError, TypeError):
                rec["exif_orientation"] = 1
                flag(rec, "exif_orientation_invalid", "malformed orientation; runtime fallback remains usable")
            rec["format"] = header.format
            q = getattr(header, "quantization", None)
            rec["jpeg_mean_quantizer"] = float(np.mean([v for table in q.values() for v in table])) if q else None
        decoded = decode_image(raw)
        rec["exif_size"] = list(decoded.orig_size)
        rec["decoded_size"] = list(decoded.image.size)
        rec["bytes_per_original_pixel"] = len(raw) / (decoded.orig_size[0] * decoded.orig_size[1])
    except Exception as error:
        rec["hard_exclude"] = True
        rec["hard_exclude_reasons"] = [f"decode_or_read_failed:{type(error).__name__}:{error}"]
        flag(rec, "decode_failed", rec["hard_exclude_reasons"][0])
        return rec, None
    try:
        rgb = crop_roi(decoded, row)
    except Exception as error:
        rec["hard_exclude"] = True
        rec["hard_exclude_reasons"] = [f"mandatory_roi_invalid:{error}"]
        flag(rec, "roi_invalid", str(error))
        return rec, None
    rec["view_size"] = [int(rgb.shape[1]), int(rgb.shape[0])]
    if rec["exif_orientation"] not in (1, 2, 3, 4, 5, 6, 7, 8):
        flag(rec, "exif_orientation_invalid", "orientation outside supported 1..8")
    if row.get("roi"):
        coords = list(map(int, row["roi"].split()))
        original_short = min(coords[2] - coords[0], coords[3] - coords[1])
        rec["mandatory_roi_valid_after_exif"] = True
    else:
        original_short = min(decoded.orig_size)
    rec["original_view_short_side"] = original_short
    if original_short < 128:
        flag(rec, "low_resolution", "original inference view shorter side below 128 px; review only")
    boxes = []
    if row["source"] in ("rf_boos", "rf_robusta"):
        boxes, issues = read_boxes_checked(path)
        rec["annotation_issues"] = issues
        rec["n_boxes"] = len(boxes)
        for issue in issues:
            flag(rec, issue, "annotation diagnostic; no raw annotation is edited")
        if boxes:
            dims = [min(b[3] * decoded.orig_size[0], b[4] * decoded.orig_size[1]) for b in boxes]
            rec["bean_bbox_short_px_median"] = float(np.median(dims))
            rec["bean_bbox_short_px_min"] = float(min(dims))
            if np.median(dims) < 32:
                flag(rec, "small_bean_bbox", "median bbox shorter side below 32 original px")
            try:
                observed = "|".join(sorted(set(names[row["source"]][b[0]] for b in boxes)))
                observed_label, why = map_label(row["source"], observed)
                rec["annotation_label_orig"] = observed
                if observed_label != row["label"]:
                    flag(rec, "label_annotation_mismatch", f"manifest={row['label']}; annotation mapping={observed_label}; {why}")
            except (IndexError, KeyError) as error:
                flag(rec, "annotation_class_invalid", str(error))
    if row["label"] in ("mixed", "empty", "green"):
        flag(rec, f"non_target_{row['label']}", "reference warning/diagnostic label; no relabeling")
    elif row["label"] not in TARGETS:
        flag(rec, "unknown_label", "human decision required")
    if "Maw" in row["label_orig"]:
        flag(rec, "ambiguous_maw", "original Maw class has no roast definition; remains mixed as decided")
    focus_rgb, focus_short = focus_region(rgb, boxes, decoded.orig_size, original_short)
    rec["focus_measurement_region"] = "median_area_bbox" if boxes else "inference_view"
    rec["focus_original_short_side"] = focus_short
    metrics = detail_metrics(focus_rgb, focus_short)
    rec.update(metrics)
    if metrics["blur_review"]:
        flag(rec, "low_multiscale_detail", "low detail relative to resolution/texture; blur review, no exclusion")
    if metrics["low_texture_review"]:
        flag(rec, "low_texture", "focus cannot be inferred reliably from low texture")
    gray64 = thumb_array(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    thumb = (gray64 * 255).round().astype(np.uint8)
    rec["qa_view_phash"] = f"{phash_array(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)):016x}"
    if rec["jpeg_mean_quantizer"] is not None and rec["jpeg_mean_quantizer"] >= 35 and rec["bytes_per_original_pixel"] < 0.25:
        flag(rec, "compression_review", "large JPEG quantizer combined with low bytes/pixel")
    cfg = SegConfig()
    try:
        seg = segment(rgb, cfg, find_beans=not bool(row.get("roi")))
        rec["pipeline_mode"] = seg.mode
        rec["pipeline_notes"] = seg.notes
        rec["pipeline_wb_applied"] = seg.wb_applied
        rec["border_uniform"] = seg.border_uniform
        rec["pipeline_mask_fraction"] = float(seg.pixel_mask.mean())
        if not seg.wb_applied:
            flag(rec, "wb_not_applied", "ROI has no neutral reference" if row.get("roi") else "shared background WB guard rejects reference")
        if seg.mode == "full":
            flag(rec, "pipeline_full_fallback", "full-image fallback retained; no exclusion")
        if boxes:
            truth = box_mask(seg.pixel_mask.shape, [b[1:] for b in boxes], shrink=0)
            inside = float(np.count_nonzero(seg.pixel_mask & truth) / max(np.count_nonzero(seg.pixel_mask), 1))
            rec["selected_pixels_in_bbox"] = inside
            rec["bbox_pixels_selected"] = float(np.count_nonzero(seg.pixel_mask & truth) / max(np.count_nonzero(truth), 1))
            if inside < 0.5:
                flag(rec, "pipeline_background_selection", "less than half selected pixels lie in annotation bbox")
            diag_wb = background_white_balance(cv2.resize(rgb, (seg.lab.shape[1], seg.lab.shape[0])).astype(np.float32) / 255, cfg)
            rec["bbox_diagnostic_wb_applied"] = diag_wb.applied
            if diag_wb.applied != seg.wb_applied:
                flag(rec, "wb_train_pipeline_mismatch", "shared guard disagreement")
        px_mask = box_mask(rgb.shape[:2], [b[1:] for b in boxes], shrink=0.2) if boxes else np.ones(rgb.shape[:2], bool)
        px = rgb[px_mask]
        if not px.size:
            px = rgb.reshape(-1, 3)
        rec["clipping_measurement_region"] = "bbox_shrink_20pct" if boxes else "inference_view"
        rec["black_clip_fraction"] = float((px.max(axis=1) <= 2).mean())
        rec["white_clip_fraction"] = float((px.min(axis=1) >= 253).mean())
        rec["gray_p01"] = float(np.percentile(px.mean(axis=1), 1))
        rec["gray_p99"] = float(np.percentile(px.mean(axis=1), 99))
        for name in ("black", "white"):
            if rec[f"{name}_clip_fraction"] > 0.15:
                flag(rec, f"{name}_clipping_review", "clipping review only; darkness never excludes dark roast")
    except Exception as error:
        rec["pipeline_error"] = f"{type(error).__name__}:{error}"
        flag(rec, "pipeline_diagnostic_error", rec["pipeline_error"])
    return rec, thumb


def focus_region(rgb, boxes, original_size, original_short):
    """Focus is measured on a representative bean bbox, not its white backdrop."""
    if not boxes:
        return rgb, original_short
    b = sorted(boxes, key=lambda b: b[3] * b[4])[len(boxes) // 2]
    _, cx, cy, bw, bh = b
    h, w = rgb.shape[:2]
    xa, xb = max(0, int((cx-bw/2)*w)), min(w, max(1, int(np.ceil((cx+bw/2)*w))))
    ya, yb = max(0, int((cy-bh/2)*h)), min(h, max(1, int(np.ceil((cy+bh/2)*h))))
    patch = rgb[ya:yb, xa:xb]
    if min(patch.shape[:2]) < 2:
        return rgb, original_short
    return patch, min(bw * original_size[0], bh * original_size[1])


def find_near_duplicates(records: list[dict], thumbs: list) -> list[dict]:
    indices = [i for i, r in enumerate(records) if r.get("qa_view_phash") and thumbs[i] is not None]
    hashes = np.asarray([int(records[i]["qa_view_phash"], 16) for i in indices], dtype=np.uint64)
    pairs = []
    for a, ia in enumerate(indices):
        distances = np.bitwise_count(hashes[a + 1:] ^ hashes[a])
        for offset in np.flatnonzero(distances <= 6):
            ib = indices[a + 1 + int(offset)]
            mad = float(np.abs(thumbs[ia].astype(np.float32) - thumbs[ib]).mean() / 255)
            if mad >= 0.03:
                continue
            ra, rb = records[ia], records[ib]
            pairs.append({"a": ra["path"], "b": rb["path"], "source_a": ra["source"], "source_b": rb["source"],
                          "label_a": ra["label"], "label_b": rb["label"], "same_group": ra["group"] == rb["group"],
                          "phash_hamming": int(distances[offset]), "mad64": mad,
                          "exact_md5": ra.get("md5_actual") == rb.get("md5_actual")})
    adjacency = Counter(path for p in pairs for path in (p["a"], p["b"]))
    cross = set(path for p in pairs if p["source_a"] != p["source_b"] for path in (p["a"], p["b"]))
    conflict = set(path for p in pairs if p["label_a"] != p["label_b"] for path in (p["a"], p["b"]))
    for rec in records:
        rec["near_duplicate_candidate_count"] = adjacency[rec["path"]]
        if rec["path"] in adjacency:
            flag(rec, "near_duplicate_candidate", "pHash+MAD candidate, may share scene or low texture; review only")
        if rec["path"] in cross:
            flag(rec, "cross_source_duplicate_candidate", "manual visual confirmation required before altering data")
        if rec["path"] in conflict:
            flag(rec, "near_duplicate_label_ambiguity", "similar image with different labels; human decision required, no relabeling")
    return pairs


def summarize(records: list[dict], pairs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in records:
        by[(r["source"], r["label"])].append(r)
    by_path = {r["path"]: r for r in records}
    # Near-duplicate connected components reduce declared groups conservatively.
    # They are diagnostics, not a claim of statistical independence.
    parents = {r["group"]: r["group"] for r in records}
    def representative(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    for pair in pairs:
        if pair["source_a"] == pair["source_b"]:
            a, b = representative(by_path[pair["a"]]["group"]), representative(by_path[pair["b"]]["group"])
            parents[a] = b
    strata = []
    for (source, label), rows in sorted(by.items()):
        item = {"source": source, "label": label, "rows": len(rows), "manifest_groups": len({r['group'] for r in rows}),
                "near_duplicate_connected_group_components": len({representative(r['group']) for r in rows}),
                "independence_verified": False,
                "hard_exclude": sum(r["hard_exclude"] for r in rows), "flags": dict(Counter(f for r in rows for f in r["flags"]))}
        for field in ("original_view_short_side", "bean_bbox_short_px_median", "laplacian_var_256", "fine_coarse_ratio", "texture_std_256", "black_clip_fraction", "white_clip_fraction", "selected_pixels_in_bbox", "border_uniform"):
            values = [r[field] for r in rows if field in r and r[field] is not None]
            if values:
                item[field] = dict(zip(("min", "p25", "median", "p75", "max"), map(float, np.percentile(values, [0, 25, 50, 75, 100]))))
        strata.append(item)
    exact = defaultdict(list)
    for r in records:
        if r.get("md5_actual"):
            exact[r["md5_actual"]].append(r["path"])
    return {"audited_trainval": len(records), "reference_model_pool": sum(r["reference_model_pool"] for r in records),
            "hard_excluded_total": sum(r["hard_exclude"] for r in records),
            "hard_excluded_reference_pool": sum(r["hard_exclude"] and r["reference_model_pool"] for r in records),
            "flag_counts": dict(Counter(f for r in records for f in r["flags"])), "source_class": strata,
            "exact_md5_duplicate_groups": [paths for paths in exact.values() if len(paths) > 1],
            "near_duplicate_pairs": len(pairs), "near_duplicate_images": len({p[k] for p in pairs for k in ("a", "b")}),
            "cross_source_near_duplicate_pairs": sum(p["source_a"] != p["source_b"] for p in pairs),
            "near_duplicate_different_label_pairs": sum(p["label_a"] != p["label_b"] for p in pairs),
            "near_duplicate_different_group_pairs": sum(not p["same_group"] for p in pairs),
            "limitations": ["Flags are conservative diagnostics, not validated defect classifications.",
                            "All near duplicate candidates require human review; low-texture scenes can match accidentally.",
                            "Manifest groups are declared units, not a proof of independent coffee samples. Agtron has five phone groups but related roast samples.",
                            "No labels were changed; mixed/green/empty remain non-target rows.",
                            "No frozen-test pixels or paper_path images were loaded."]}


def add_provenance(summary: dict, records: list[dict], root: Path) -> None:
    summary["preprocessing_source_sha256"] = {name: sha256(ML_DIR / "roastml" / name) for name in ("decode.py", "segment.py", "features.py")}
    summary["trainval_actual_md5_listing_sha256"] = hashlib.sha256(json.dumps([(r["path"], r.get("md5_actual")) for r in records], separators=(",", ":")).encode()).hexdigest()
    digest = hashlib.sha256()
    for r in records:
        if r["source"] not in ("rf_boos", "rf_robusta"):
            continue
        label = yolo_label_path(safe_path(root, r["path"]))
        digest.update(r["path"].encode())
        digest.update(label.read_bytes() if label.is_file() else b"MISSING")
    summary["trainval_annotation_listing_sha256"] = digest.hexdigest()


def build_inner_groups(root: Path, result_dir: Path) -> dict:
    """Conservative near-duplicate connected components for INNER splits only.

    Preserve manifest groups for the exact reference video cap and preserve
    outer LOSO holdouts. A candidate link is sufficient to avoid inner leakage;
    this does not declare a duplicate, exclude a photo or modify any label.
    """
    root, result_dir = root.resolve(), result_dir.resolve()
    if not result_dir.is_relative_to((ML_DIR / "results").resolve()) or not result_dir.name.startswith("qa_"):
        raise ValueError("QA report destination must be ML/results/qa_*")
    cache = root / "cache" / result_dir.name
    records = json.loads((cache / "ledger.json").read_text(encoding="utf-8"))
    with (cache / "near_duplicate_pairs.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    if any(r["split"] != "trainval" for r in records):
        raise ValueError("inner-group guard refuses non-trainval rows")
    by_path = {r["path"]: r for r in records}
    source_by_group = defaultdict(set)
    for r in records:
        source_by_group[r["group"]].add(r["source"])
    if any(len(sources) != 1 for sources in source_by_group.values()):
        raise ValueError("one manifest group spans multiple sources")
    parents = {group: group for group in source_by_group}
    def representative(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    for p in pairs:
        if p["source_a"] != p["source_b"]:
            continue
        ga, gb = by_path[p["a"]]["group"], by_path[p["b"]]["group"]
        a, b = representative(ga), representative(gb)
        parents[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for group in sorted(parents):
        members[representative(group)].append(group)
    mapping, components = {}, []
    for rep, groups in sorted(members.items()):
        source = next(iter(source_by_group[rep]))
        component = source + ":qa_inner:" + hashlib.sha256("\n".join(groups).encode()).hexdigest()[:16]
        for group in groups:
            mapping[group] = component
        linked = [r for r in records if r["group"] in set(groups)]
        components.append({"inner_group": component, "source": source, "manifest_groups": groups,
                           "n_rows": len(linked), "labels": dict(Counter(r["label"] for r in linked))})
    residual = [p for p in pairs if p["source_a"] == p["source_b"] and mapping[by_path[p["a"]]["group"]] != mapping[by_path[p["b"]]["group"]]]
    if residual:
        raise AssertionError("near duplicate candidate remains across inner-group components")
    policy = {"policy_id": "qa_inner_candidate_components_20261008_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Within-source union of manifest groups connected by fixed QA pHash+MAD candidate pairs.",
              "thresholds": POLICY["near_duplicate"], "qa_policy_sha256": sha256(result_dir / "policy.json"),
              "rule": "Apply only to inner train/validation splitting. Compute R2 video cap with original manifest groups first. Never merge sources, modify labels, exclude images or alter outer holdouts.",
              "interpretation": "Conservative grouping guard; candidates are not all visually confirmed duplicates, and components are not proven independent samples."}
    write_json(result_dir / "inner_group_policy.json", policy)
    payload = {"policy_id": policy["policy_id"], "manifest_sha256": sha256(root / "manifest.csv"),
               "policy_sha256": sha256(result_dir / "inner_group_policy.json"), "qa_policy_sha256": policy["qa_policy_sha256"],
               "near_duplicate_pairs_sha256": sha256(cache / "near_duplicate_pairs.csv"), "ledger_sha256": sha256(cache / "ledger.json"),
               "group_to_inner_group": mapping, "components": components, "manifest_groups": len(mapping),
               "inner_group_components": len(components), "multi_manifest_group_components": sum(len(c["manifest_groups"]) > 1 for c in components),
               "residual_within_source_candidate_cross_component_pairs": len(residual),
               "unmerged_cross_source_candidate_pairs": sum(p["source_a"] != p["source_b"] for p in pairs),
               "rule": policy["rule"], "interpretation": policy["interpretation"]}
    write_json(cache / "inner_groups.json", payload)
    write_json(result_dir / "inner_groups_summary.json", {k: v for k, v in payload.items() if k not in ("group_to_inner_group", "components")})
    return payload


def enrich_ledger(root: Path, result_dir: Path, workers: int = 4) -> dict:
    """Improve bbox focus diagnostics without repeating expensive segmentation.

    The existing policy/primary run provenance remain intact. No hard exclusion,
    label, split, group or reference mask changes during this diagnostic pass.
    """
    start = time.perf_counter()
    root, result_dir = root.resolve(), result_dir.resolve()
    if not result_dir.is_relative_to((ML_DIR / "results").resolve()) or not result_dir.name.startswith("qa_"):
        raise ValueError("QA report destination must be ML/results/qa_*")
    cache_dir = root / "cache" / result_dir.name
    records = json.loads((cache_dir / "ledger.json").read_text(encoding="utf-8"))
    prior = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    if prior["manifest_sha256_before"] != sha256(root / "manifest.csv"):
        raise ValueError("reference manifest changed after primary QA run")
    with (cache_dir / "near_duplicate_pairs.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    for p in pairs:
        p["same_group"] = p["same_group"] == "True"
        p["exact_md5"] = p["exact_md5"] == "True"
    def enrich(r):
        if r["split"] != "trainval":
            raise ValueError("enrichment refuses frozen-test image")
        if not r.get("n_boxes") or r["hard_exclude"]:
            r["focus_measurement_region"] = "inference_view"
            r["focus_original_short_side"] = r.get("original_view_short_side")
            return r
        path = safe_path(root, r["path"])
        boxes, _ = read_boxes_checked(path)
        decoded = decode_image(path.read_bytes())
        rgb = crop_roi(decoded, r)
        focus_rgb, focus_short = focus_region(rgb, boxes, decoded.orig_size, r["original_view_short_side"])
        r["full_view_focus_metrics_reference"] = {k: r[k] for k in detail_metrics(rgb, r["original_view_short_side"])}
        r.update(detail_metrics(focus_rgb, focus_short))
        r["focus_measurement_region"] = "median_area_bbox"
        r["focus_original_short_side"] = focus_short
        r["flags"] = [f for f in r["flags"] if f not in ("low_multiscale_detail", "low_texture")]
        r["reasons"] = [v for v in r["reasons"] if not v.startswith(("low_multiscale_detail:", "low_texture:"))]
        if r["blur_review"]:
            flag(r, "low_multiscale_detail", "low detail within median-area bean bbox, considering bean resolution and texture")
        if r["low_texture_review"]:
            flag(r, "low_texture", "focus is ambiguous within low-texture bean bbox")
        return r
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(enrich, records))
    summary = {**prior, **summarize(records, pairs)}
    _, metadata = load_trainval(root / "manifest.csv")
    summary["manifest_metadata"] = metadata
    summary["focus_diagnostic_enrichment"] = {"original_audit_script_sha256": prior["script_sha256"],
        "enrichment_script_sha256": sha256(Path(__file__)), "wall_seconds": time.perf_counter() - start,
        "reason": "Measure focus within median-area annotated bean bbox to account for bean pixel resolution and texture, without changing preregistered thresholds or masks."}
    add_provenance(summary, records, root)
    write_json(cache_dir / "ledger.json", records)
    write_csv(cache_dir / "ledger.csv", records)
    index = make_sheets(root, records, root / "contact_sheets" / result_dir.name)
    summary["contact_sheets"] = {"directory": str(root / "contact_sheets" / result_dir.name), "sheets": len(index), "examples": sum(len(r["examples"]) for r in index)}
    write_csv(result_dir / "source_class_counts.csv", [{k: v for k, v in r.items() if k in ('source','label','rows','manifest_groups','near_duplicate_connected_group_components','independence_verified','hard_exclude')} for r in summary["source_class"]])
    write_json(result_dir / "summary.json", summary)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def make_sheets(root: Path, records: list[dict], out: Path, max_each: int = 16) -> list[dict]:
    """Every source x label gets typical and flagged examples, with exact path index."""
    out.mkdir(parents=True, exist_ok=True)
    strata = defaultdict(list)
    for r in records:
        strata[(r["source"], r["label"])].append(r)
    rng = random.Random(POLICY["seed"])
    index = []
    noncritical = {"wb_not_applied", "near_duplicate_candidate", "non_target_mixed", "non_target_green", "non_target_empty", "ambiguous_maw"}
    for (source, label), rows in sorted(strata.items()):
        normal = [r for r in rows if not (set(r["flags"]) - noncritical) and not r["hard_exclude"]]
        flagged = [r for r in rows if set(r["flags"]) - noncritical or r["hard_exclude"]]
        if not normal:
            normal = sorted(rows, key=lambda r: len(set(r["flags"]) - noncritical))[:max_each]
            normal_kind = "typical_all_have_review_flags"
        else:
            normal_kind = "normal_no_critical_review_flags"
        rng.shuffle(normal)
        # Cover different flag types before filling randomly.
        rng.shuffle(flagged)
        covered, diverse = set(), []
        for r in flagged:
            if set(r["flags"]) - covered:
                diverse.append(r)
                covered.update(r["flags"])
        selected_flags = (diverse + [r for r in flagged if r not in diverse])[:max_each]
        for kind, selection, meaning in (("normal", normal[:max_each], normal_kind), ("flagged", selected_flags, "review_only")):
            tile_w, tile_h, cols = 368, 244, 4
            n = max(1, len(selection))
            sheet = Image.new("RGB", (tile_w * cols, 32 + tile_h * ((n + cols - 1) // cols)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 6), f"{source} / {label} / {kind} / {meaning}; raw view | mask red + bbox cyan", fill="black")
            entries = []
            for i, r in enumerate(selection):
                x, y = (i % cols) * tile_w, 32 + (i // cols) * tile_h
                draw.text((x + 3, y + 3), f"#{i+1} {Path(r['path']).name[:46]}", fill="black")
                draw.text((x + 3, y + 16), f"group: {r['group'].split(':')[-1][:45]}", fill="black")
                draw.text((x + 3, y + 29), ','.join(r["flags"])[:54] or "no review flags", fill="darkred" if r["flags"] else "black")
                try:
                    if r["split"] != "trainval":
                        raise ValueError("non-trainval sheet image")
                    decoded = decode_image(safe_path(root, r["path"]).read_bytes())
                    rgb = crop_roi(decoded, r)
                    im = Image.fromarray(rgb)
                    im.thumbnail((178, 178))
                    sheet.paste(im, (x + 3 + (178 - im.width)//2, y + 52 + (178-im.height)//2))
                    seg = segment(rgb, SegConfig(), find_beans=not bool(r.get("roi")))
                    overlay = cv2.resize(rgb, (seg.lab.shape[1], seg.lab.shape[0])).astype(np.float32)
                    overlay[seg.pixel_mask] = 0.65 * overlay[seg.pixel_mask] + np.array([255, 0, 0]) * 0.35
                    oi = Image.fromarray(overlay.astype(np.uint8))
                    od = ImageDraw.Draw(oi)
                    if r["source"] in ("rf_boos", "rf_robusta"):
                        boxes, _ = read_boxes_checked(safe_path(root, r["path"]))
                        w, h = oi.size
                        for _, cx, cy, bw, bh in boxes:
                            od.rectangle([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], outline="cyan", width=2)
                    oi.thumbnail((178, 178))
                    sheet.paste(oi, (x + 187 + (178-oi.width)//2, y + 52 + (178-oi.height)//2))
                except Exception as error:
                    draw.text((x + 3, y + 55), f"UNUSABLE: {str(error)[:48]}", fill="red")
                entries.append({"tile": i + 1, "path": r["path"], "group": r["group"], "flags": r["flags"]})
            if not selection:
                draw.text((4, 60), "No flagged examples in this stratum.", fill="black")
            filename = f"{source}_{label}_{kind}.jpg"
            sheet.save(out / filename, quality=90)
            index.append({"source": source, "label": label, "kind": kind, "meaning": meaning,
                          "sheet": str(out / filename), "examples": entries})
    write_json(out / "index.json", index)
    return index


def run(root: Path, result_dir: Path, workers: int = 4, sheets: bool = True) -> dict:
    start = time.perf_counter()
    root, result_dir = root.resolve(), result_dir.resolve()
    if not result_dir.is_relative_to((ML_DIR / "results").resolve()) or not result_dir.name.startswith("qa_"):
        raise ValueError("QA report destination must be ML/results/qa_*")
    cache_dir = root / "cache" / result_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    policy = {**POLICY, "preregistered_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(result_dir / "policy.json", policy)  # before opening any image
    rows, manifest_meta = load_trainval(root / "manifest.csv")
    before_hash = sha256(root / "manifest.csv")
    names = {s: read_yolo_names(root / "raw" / s / "data.yaml") for s in ("rf_boos", "rf_robusta")}
    print(f"QA: {len(rows)} trainval rows; policy registered before image reads; workers={workers}", flush=True)
    cv2.setNumThreads(1)
    def worker(r):
        return audit_one(root, r, names)
    records, thumbs = [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (rec, thumb) in enumerate(pool.map(worker, rows), 1):
            records.append(rec)
            thumbs.append(thumb)
            if i % 500 == 0:
                print(f"QA: {i}/{len(rows)} images audited", flush=True)
    pairs = find_near_duplicates(records, thumbs)
    summary = summarize(records, pairs)
    summary.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "manifest": str(root / "manifest.csv"),
                    "manifest_sha256_before": before_hash, "manifest_sha256_after": sha256(root / "manifest.csv"),
                    "policy_sha256": sha256(result_dir / "policy.json"), "script_sha256": sha256(Path(__file__)),
                    "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__},
                    "manifest_metadata": manifest_meta, "ledger": str(cache_dir / "ledger.csv"),
                    "new_qa_mask_changes_reference_training": bool(summary["hard_excluded_reference_pool"]),
                    "visual_review": {"status": "pending", "scope": "generated stratified trainval contact sheets; flags do not imply defects"}})
    add_provenance(summary, records, root)
    write_json(cache_dir / "ledger.json", records)
    write_csv(cache_dir / "ledger.csv", records)
    write_csv(cache_dir / "near_duplicate_pairs.csv", pairs)
    write_json(result_dir / "train_exclusion_mask.json", {"policy_id": POLICY["id"], "manifest_sha256": before_hash,
               "scope": "training only; intersect reference R2 mask; preserve holdouts",
               "exclusions": {r["path"]: r["hard_exclude"] for r in records},
               "excluded_paths": [r["path"] for r in records if r["hard_exclude"]]})
    print(f"QA: hard excluded={summary['hard_excluded_total']} reference pool={summary['hard_excluded_reference_pool']}; near pairs={len(pairs)}", flush=True)
    write_csv(result_dir / "source_class_counts.csv", [{k: v for k, v in r.items() if k in ('source','label','rows','manifest_groups','near_duplicate_connected_group_components','independence_verified','hard_exclude')} for r in summary["source_class"]])
    if sheets:
        index = make_sheets(root, records, root / "contact_sheets" / result_dir.name)
        summary["contact_sheets"] = {"directory": str(root / "contact_sheets" / result_dir.name), "sheets": len(index), "examples": sum(len(r["examples"]) for r in index)}
    summary["audit_wall_seconds"] = time.perf_counter() - start
    write_json(result_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("audited_trainval", "reference_model_pool", "hard_excluded_total", "near_duplicate_pairs", "cross_source_near_duplicate_pairs", "audit_wall_seconds")}), flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--result-dir", type=Path, default=ML_DIR / "results" / "qa_audit_20261008")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-contact-sheets", action="store_true")
    ap.add_argument("--enrich-existing-ledger", action="store_true", help="Supplement previous run's bbox focus diagnostics; retain original provenance and masks.")
    ap.add_argument("--build-inner-groups", action="store_true", help="Conservative candidate connected components for inner splitting only; no new exclusions.")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.build_inner_groups:
        mapping = build_inner_groups(args.data_dir or data_dir(), args.result_dir)
        print(json.dumps({k: mapping[k] for k in ('manifest_groups','inner_group_components','multi_manifest_group_components','residual_within_source_candidate_cross_component_pairs')}), flush=True)
    elif args.enrich_existing_ledger:
        summary = enrich_ledger(args.data_dir or data_dir(), args.result_dir, args.workers)
        print(json.dumps({"enrichment": summary["focus_diagnostic_enrichment"], "flag_counts": summary["flag_counts"]}), flush=True)
    else:
        run(args.data_dir or data_dir(), args.result_dir, args.workers, not args.no_contact_sheets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
