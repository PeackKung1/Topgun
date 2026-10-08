"""Controlled B1 preprocessing ablation: legacy scalar ROI vs actual x/y scale.

Only Agtron trainval rows are decoded. Every 3395-image outer holdout and exact
2393-row R2 training mask remain fixed. C=0.01 is fixed, not retuned. Artifacts
are separate candidates and never overwrite models/current or reference caches.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import PIL

from roastml.decode import decode_image
from roastml.features import FEATURES_ALL, FEATURE_SETS, image_features, select
from roastml.linear_model import LinearSoftmax, spec_from_sklearn
from roastml.paths import data_dir
from roastml.rgb_views import crop_roi
from roastml.segment import SegConfig
from tools.compare_tabular import calibration
from tools.train_baseline import CLASSES, FOLDS, SEED, attach_boxes, fold_table, load_rows, make_pipe, row_key, video_cap_mask

ML_DIR = Path(__file__).resolve().parents[1]
PLAN = {
    "id": "roi_scaling_ablation_20261008_v1", "seed": SEED,
    "arms": {"legacy_scalar": "ROI coordinate x and y both use decoded width/original EXIF width",
             "actual_xy": "Shared roastml.rgb_views.crop_roi uses decoded/original width and height independently"},
    "model": "B1 LogisticRegression C=0.01 class_weight=balanced + training-only StandardScaler",
    "feature_set": "Lab_hist", "class_order": CLASSES, "outer_folds": FOLDS,
    "train_mask": "Exact original R2 cap 10 frames/video seed20261006; manifest group unchanged",
    "holdout": "All3395 reference trainval images unchanged; frozen test never opened",
    "view": "Agtron mandatory ROI after EXIF, no WB; others unchanged reference pipeline/train bbox features",
    "selection": "No tuning, epochs, threshold change, label change, QA cleaning or model selection in this ablation",
    "limitations": "A geometry correction can change a few feature values; this ablation does not establish a cleaning or CNN effect. Pi/E2E unverified.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legacy_roi(decoded, roi):
    """Reproduce the reference crop exactly, without changing baseline code."""
    # Validate first to prevent a reference reproduction from exposing labels.
    crop_roi(decoded, roi)
    xa, ya, xb, yb = [int(round(int(v) * decoded.scale)) for v in roi.split()]
    rgb = np.asarray(decoded.image)[ya:yb, xa:xb]
    if min(rgb.shape[:2]) < 1:
        raise ValueError("legacy ROI empty after decode")
    return rgb


def extract_updated(root, row, config):
    if row["split"] != "trainval":
        raise ValueError("ROI ablation refuses frozen-test image")
    if row["source"] != "agtron" or not row.get("roi"):
        raise ValueError("Only mandatory Agtron ROI rows may be decoded in this ablation")
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes data directory")
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != row["md5"]:
        raise ValueError(f"manifest bytes hash mismatch: {row['path']}")
    decoded = decode_image(raw)
    old = legacy_roi(decoded, row["roi"])
    new = crop_roi(decoded, row["roi"])
    # Re-extract BOTH arms on affected rows so the cache is not assumed correct.
    cfg = SegConfig.from_dict(config)
    before = image_features(old, cfg, find_beans=False)
    after = image_features(new, cfg, find_beans=False)
    if before.seg.wb_applied or after.seg.wb_applied:
        raise AssertionError("Agtron ROI must not use bean pile as white reference")
    values = list(map(int, row["roi"].split()))
    legacy_bounds = [round(v * decoded.scale) for v in values]
    ow, oh = decoded.orig_size
    dw, dh = decoded.image.size
    xy_bounds = [round(values[0]*dw/ow), round(values[1]*dh/oh), round(values[2]*dw/ow), round(values[3]*dh/oh)]
    return {"path": row["path"], "group": row["group"], "label": row["label"], "roi": row["roi"],
            "orig_size_after_exif": list(decoded.orig_size), "decoded_size": list(decoded.image.size),
            "legacy_bounds": legacy_bounds, "actual_xy_bounds": xy_bounds,
            "legacy_crop_size": [old.shape[1], old.shape[0]], "actual_xy_crop_size": [new.shape[1], new.shape[0]],
            "geometry_changed": legacy_bounds != xy_bounds, "x_legacy_reextracted": before.x.tolist(),
            "x_actual_xy": after.x.tolist(), "feature_abs_delta_max": float(np.max(np.abs(after.x-before.x))),
            "wb_applied": False}


def fixed_loso(rows, train_features, pipeline_features, cap, out_dir, arm):
    labels = np.array([r["label"] for r in rows])
    source = np.array([r["source"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    paths = np.array([r["path"] for r in rows])
    probabilities = np.zeros((len(rows), 3), np.float64)
    fold_runs = {}
    names = FEATURE_SETS["Lab_hist"]
    for fold in FOLDS:
        train = cap & (source != fold)
        holdout = source == fold
        t0 = time.perf_counter()
        pipe = make_pipe(0.01).fit(select(train_features[train], names), labels[train])
        spec = spec_from_sklearn(pipe, names, CLASSES)
        probabilities[holdout] = LinearSoftmax(spec).proba(pipeline_features[holdout])
        dump(out_dir / f"{arm}_{fold}_model.json", spec)
        fold_runs[fold] = {"train_rows": int(train.sum()), "train_groups": len(set(groups[train])),
                           "holdout_rows": int(holdout.sum()), "holdout_groups": len(set(groups[holdout])),
                           "train_paths": paths[train].tolist(), "holdout_paths": paths[holdout].tolist(),
                           "training_seconds": time.perf_counter() - t0,
                           "model_sha256": sha256(out_dir / f"{arm}_{fold}_model.json")}
    prediction = np.array(CLASSES)[probabilities.argmax(1)]
    table = fold_table(labels, prediction, source)
    csv_write(out_dir / f"{arm}_predictions.csv", [{"path": r["path"], "source": r["source"], "label": r["label"],
              "group": r["group"], "prediction": pred, **{f"p_{c}": float(p[k]) for k, c in enumerate(CLASSES)}} for r, pred, p in zip(rows, prediction, probabilities)])
    for fold in list(FOLDS) + ["pooled"]:
        with (out_dir / f"{arm}_confusion_{fold}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true/pred"] + CLASSES)
            writer.writerows([[c] + values for c, values in zip(CLASSES, table[fold]["cm"])])
    return {"metrics": table, "fold_runs": fold_runs, "calibration": calibration(labels, probabilities),
            "worst_fold": min(FOLDS, key=lambda f: table[f]["macro_f1"])}, probabilities


def run(root, out_dir, workers=4):
    import sklearn
    t0 = time.perf_counter()
    root, out_dir = root.resolve(), out_dir.resolve()
    if not out_dir.is_relative_to((ML_DIR / "results").resolve()):
        raise ValueError("ablation outputs must be under ML/results")
    out_dir.mkdir(parents=True, exist_ok=False)
    dump(out_dir / "plan.json", {**PLAN, "preregistered_at_utc": datetime.now(timezone.utc).isoformat()})
    reference_path = ML_DIR / "results" / "baseline_loso.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    config = reference["seg_config"]
    rows = load_rows(root)
    attach_boxes(root, rows)
    if len(rows) != 3395 or reference["n_images"] != len(rows):
        raise ValueError("reference holdout population changed")
    cache_path = root / "cache" / reference["feature_cache"]
    cache_hash_before = sha256(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    entries = [cache[row_key(r)] for r in rows]
    if any(e["err"] for e in entries):
        raise ValueError("reference contains errors; do not silently change holdouts")
    X_train = np.array([e["x_train"] for e in entries], np.float64)
    X_pipeline = np.array([e["x"] for e in entries], np.float64)
    group = np.array([r["group"] for r in rows])
    paths = np.array([r["path"] for r in rows])
    cap = video_cap_mask(group, paths, seed=SEED)
    if int(cap.sum()) != 2393:
        raise ValueError("exact reference R2 training cap changed")
    manifest_before = sha256(root / "manifest.csv")
    agtron = [r for r in rows if r["source"] == "agtron"]
    if len(agtron) != 561:
        raise ValueError("Agtron trainval population changed")
    print(f"ROI scaling: {len(agtron)} Agtron trainval images; fixed R2 training {int(cap.sum())}; holdouts {len(rows)}", flush=True)
    cv2.setNumThreads(1)
    def work(r):
        return extract_updated(root, r, config)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        updates = list(pool.map(work, agtron))
    update_by_path = {u["path"]: u for u in updates}
    X_train_xy, X_pipeline_xy = X_train.copy(), X_pipeline.copy()
    reference_feature_delta = []
    for i, row in enumerate(rows):
        if row["path"] not in update_by_path:
            continue
        update = update_by_path[row["path"]]
        reference_feature_delta.append(float(np.max(np.abs(np.array(update["x_legacy_reextracted"]) - X_pipeline[i]))))
        X_train_xy[i] = update["x_actual_xy"]
        X_pipeline_xy[i] = update["x_actual_xy"]
    if max(reference_feature_delta) > 1e-10:
        raise ValueError(f"legacy cache does not reproduce shared preprocessing: maxdiff={max(reference_feature_delta)}")
    cache_out = root / "cache" / out_dir.name
    cache_out.mkdir(parents=True, exist_ok=True)
    dump(cache_out / "agtron_feature_updates.json", {"plan_id": PLAN["id"], "manifest_sha256": manifest_before,
         "reference_cache_sha256": cache_hash_before, "feature_names": FEATURES_ALL, "seg_config": config, "rows": updates})
    geometry = [{k: v for k, v in u.items() if k not in ("x_legacy_reextracted", "x_actual_xy")} for u in updates]
    csv_write(out_dir / "agtron_geometry_feature_deltas.csv", [{k: json.dumps(v) if isinstance(v, list) else v for k, v in u.items()} for u in geometry])
    legacy, old_p = fixed_loso(rows, X_train, X_pipeline, cap, out_dir, "legacy_scalar")
    score = legacy["metrics"]["mean"]["macro_f1"]
    expected = reference["by_run"]["R2"]["B1"]["folds"]["pipeline"]["mean"]["macro_f1"]
    if abs(score - expected) > 1e-12:
        raise ValueError(f"legacy reference not reproduced: actual={score} expected={expected}")
    updated, new_p = fixed_loso(rows, X_train_xy, X_pipeline_xy, cap, out_dir, "actual_xy")
    candidate_dir = ML_DIR / "models" / "candidates" / out_dir.name
    candidate_dir.mkdir(parents=True, exist_ok=False)
    fit_start = time.perf_counter()
    pipe = make_pipe(0.01).fit(select(X_train_xy[cap], FEATURE_SETS["Lab_hist"]), np.array([r["label"] for r in rows])[cap])
    spec = spec_from_sklearn(pipe, FEATURE_SETS["Lab_hist"], CLASSES)
    full_p = LinearSoftmax(spec).proba(X_pipeline_xy)
    sklearn_p = pipe.predict_proba(select(X_pipeline_xy, FEATURE_SETS["Lab_hist"]))[:, [list(pipe.classes_).index(c) for c in CLASSES]]
    parity_error = float(np.max(np.abs(full_p-sklearn_p)))
    if parity_error > 1e-10:
        raise AssertionError("NumPy/sklearn export probability parity failed")
    dump(candidate_dir / "model.json", spec)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__, "sklearn": sklearn.__version__}
    card = {"backend": "b1_linear", "name": "b1-Lab_hist-C0.01-R2-roi-actualxy", "model_file": "model.json",
            "low_conf_threshold": 0.0, "seg_config": config, "feature_set": "Lab_hist", "C": 0.01, "run": "R2_actual_xy",
            "roi_scaling": "actual_xy", "roi_scaling_note": "shared crop_roi validates after EXIF and uses actual decoded width and height ratios",
            "train": {"split": "trainval", "rows": int(cap.sum()), "groups": len(set(group[cap])), "sources": FOLDS,
                      "counts": dict(Counter(r["label"] for r, keep in zip(rows, cap) if keep)), "training_mask_sha256": hashlib.sha256("\n".join(paths[cap]).encode()).hexdigest()},
            "loso_mean_macro_f1": {"pipeline": updated["metrics"]["mean"]["macro_f1"]}, "loso_folds": FOLDS,
            "manifest_sha256": manifest_before, "versions": versions, "preprocessing_ablation": PLAN,
            "data_license_note": "Trained with Agtron Unknown license; internal use; model files must not be committed",
            "numpy_sklearn_probability_abs_error_max": parity_error, "training_seconds": time.perf_counter()-fit_start,
            "status": {"notebook_export_verified": True, "pi_latency_verified": False, "fw_e2e_verified": False, "frozen_test_evaluated": False}}
    dump(candidate_dir / "model_card.json", card)
    changes = old_p.argmax(1) != new_p.argmax(1)
    csv_write(out_dir / "prediction_delta_by_image.csv", [{"path": r["path"], "source": r["source"], "label": r["label"],
              "legacy_prediction": CLASSES[old_p[i].argmax()], "actual_xy_prediction": CLASSES[new_p[i].argmax()],
              "label_changed": bool(changes[i]), "prob_abs_delta_max": float(np.max(np.abs(old_p[i]-new_p[i])))} for i, r in enumerate(rows)])
    report = {"plan": PLAN, "seed": SEED, "versions": versions, "reference_report": str(reference_path), "reference_report_sha256": sha256(reference_path),
              "reference_cache": str(cache_path), "reference_cache_sha256_before": cache_hash_before, "reference_cache_sha256_after": sha256(cache_path),
              "manifest_sha256_before": manifest_before, "manifest_sha256_after": sha256(root / "manifest.csv"),
              "source_code_sha256": {str(p): sha256(p) for p in [Path(__file__), ML_DIR / "roastml/rgb_views.py", ML_DIR / "roastml/decode.py", ML_DIR / "roastml/segment.py", ML_DIR / "roastml/features.py"]},
              "seg_config": config, "n_holdout_images": len(rows), "r2_train_rows": int(cap.sum()),
              "training_mask_sha256": card["train"]["training_mask_sha256"], "agtron_rows": len(updates),
              "geometry_changed_images": sum(u["geometry_changed"] for u in updates),
              "geometry_changed_groups": len({u["group"] for u in updates if u["geometry_changed"]}),
              "features_changed_images": sum(u["feature_abs_delta_max"] > 1e-12 for u in updates),
              "feature_abs_delta_max": max(u["feature_abs_delta_max"] for u in updates),
              "legacy_reextraction_vs_cache_max": max(reference_feature_delta), "arms": {"legacy_scalar": legacy, "actual_xy": updated},
              "mean_macro_f1_delta_actual_minus_legacy": updated["metrics"]["mean"]["macro_f1"] - score,
              "prediction_label_changes": int(changes.sum()), "prediction_label_changes_by_source": dict(Counter(r["source"] for r, change in zip(rows, changes) if change)),
              "probability_abs_delta_max": float(np.max(np.abs(new_p-old_p))), "probability_abs_delta_mean": float(np.mean(np.abs(new_p-old_p))),
              "candidate": {"directory": str(candidate_dir), "model_sha256": sha256(candidate_dir / "model.json"), "model_bytes": (candidate_dir / "model.json").stat().st_size,
                            "numpy_sklearn_probability_abs_error_max": parity_error},
              "wall_seconds": time.perf_counter()-t0, "interpretation": "Separate preprocessing ablation. Updated D1 uses actual_xy shared RGB; compare it to actual_xy B1 if geometry changes. No QA cleaning effect is identified when hard exclusion masks are equal."}
    csv_write(out_dir / "compare.csv", [{"arm": arm, "mean_macro_f1": result["metrics"]["mean"]["macro_f1"], "pooled_macro_f1": result["metrics"]["pooled"]["macro_f1"],
              "mean_accuracy": result["metrics"]["mean"]["acc"], "pooled_accuracy": result["metrics"]["pooled"]["acc"],
              "pooled_cross_step_rate": result["metrics"]["pooled"]["cross_step_rate"], "ece": result["calibration"]["ece"]} for arm, result in report["arms"].items()])
    dump(out_dir / "report.json", report)
    print(json.dumps({k: report[k] for k in ('geometry_changed_images','features_changed_images','feature_abs_delta_max','mean_macro_f1_delta_actual_minus_legacy','prediction_label_changes','wall_seconds')}), flush=True)
    print(f"B1 legacy={score:.15f} actual_xy={updated['metrics']['mean']['macro_f1']:.15f}; candidate={candidate_dir}", flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--result-dir", type=Path, default=ML_DIR / "results" / "roi_scaling_ablation_20261008")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(args.data_dir or data_dir(), args.result_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
