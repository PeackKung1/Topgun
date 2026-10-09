"""ตรวจตัวนับเมล็ดของ BeanLinearBackend เทียบระหว่าง 2 เวอร์ชันโค้ด (trainval เท่านั้น · ไม่แตะ split=test)

ใช้รันโค้ดเวอร์ชันไหนก็ได้: ตั้ง PYTHONPATH ให้ชี้ ML/ ของเวอร์ชันนั้น (เช่นจาก `git archive <rev> ML`)
แล้วรันไฟล์นี้ — สคริปต์ import แค่ `roastml` จึงใช้ได้กับโค้ดเก่า/ใหม่โดยไม่ต้องแก้

  run      ต่อภาพ: n_beans, mode ของ segment, contour-only count, estimated, warnings, bbox → predictions_<tag>.csv
  latency  bytes→dict ผ่าน roastml.api.load (ต้นฉบับ + JPEG ≤1600 px) → predictions_latency_<tag>.csv
  compare  สรุปสองเวอร์ชัน → summary.json

ชุดภาพ (ตัดสินก่อนรัน):
  count GT   : rf_boos ทุกภาพ trainval (light/medium/dark/mixed) · n_gt = จำนวน box YOLO (box รายเมล็ด)
  single bean: ontoum224 light/medium/dark (ชุดภาพเมล็ดเดี่ยว 224 px → GT = 1 โดยนิยามของชุด) + rf_boos ที่ n_gt == 1
  empty      : rf_hendi label=empty (ภาพในเครื่องคั่วไม่มีเมล็ด) → GT = 0
  green      : ontoum224/rf_hendi/rf_robusta label=green (มีเมล็ดดิบ — ไม่ใช่ภาพว่าง รายงานแยก)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASSES = ("light", "medium", "dark")
SETS = {
    "rf_boos": ("light", "medium", "dark", "mixed"),
    "ontoum224": ("light", "medium", "dark", "green"),
    "rf_hendi": ("empty", "green"),
    "rf_robusta": ("green",),
}
LAT_SOURCES = ("ontoum224", "rf_boos", "rf_hendi", "rf_robusta")  # agtron ต้องใช้ ROI → ไม่อยู่ในชุด latency


def manifest_rows(root: Path) -> list[dict]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "trainval"]  # test แช่แข็ง: ไม่โหลดเลย
    for r in rows:
        p = (root / r["path"]).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError(f"manifest path escapes data root: {r['path']}")
    return rows


def yolo_count(img_path: Path) -> int:
    """จำนวน box ใน <split>/labels/<stem>.txt (โครง Roboflow YOLO) · ไม่มีไฟล์ → error (ห้ามเดาเป็น 0)"""
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not lbl.is_file():
        raise FileNotFoundError(f"no YOLO label for {img_path}")
    return sum(1 for line in lbl.read_text(encoding="utf-8").splitlines() if len(line.split()) >= 5)


def load_backend(model: Path):
    from roastml.bean_backend import BeanLinearBackend
    card = json.loads((model / "model_card.json").read_text(encoding="utf-8"))
    return BeanLinearBackend(model, card)


def cmd_run(args) -> None:
    import roastml
    import roastml.bean_backend as bb
    from roastml.decode import decode_image

    root = args.data_dir.resolve()
    backend = load_backend(args.model)
    rec: dict = {}
    orig_segment, orig_split = bb.segment, bb.split_beans

    def seg_spy(*a, **k):
        s = orig_segment(*a, **k)
        rec["mode"], rec["n_regions"] = s.mode, s.n_regions
        return s

    def split_spy(seg, cfg=None, **k):
        res = orig_split(seg, cfg, **k)
        # contour-only = พฤติกรรมเดิมก่อน PR (ไม่ส่ง rgb) — โค้ดเก่าไม่มี kwarg rgb จึงเท่ากับผลจริง
        contour = res if not k else orig_split(seg, cfg)
        rec["contour_only"] = len(contour) if contour else ""
        return res

    bb.segment, bb.split_beans = seg_spy, split_spy
    rows = [r for r in manifest_rows(root) if r["label"] in SETS.get(r["source"], ())]
    out = []
    t_start = time.perf_counter()
    for i, r in enumerate(rows):
        rec.clear()
        p = root / r["path"]
        img = decode_image(p.read_bytes())
        o = backend.predict(img)
        boxes = [b["bbox"] for b in o.beans]
        out.append({
            "path": r["path"], "source": r["source"], "label": r["label"],
            "n_gt": yolo_count(p) if r["source"] == "rf_boos" else "",
            "mode": rec.get("mode", ""), "n_regions": rec.get("n_regions", ""),
            "n_beans": "" if o.n_beans is None else o.n_beans,
            "contour_only": rec.get("contour_only", ""),
            "estimated": "bean_count_estimated" in o.warnings,
            "no_beans_detected": "no_beans_detected" in o.warnings,
            "warnings": "|".join(o.warnings), "bboxes": json.dumps(boxes),
            "img_w": img.image.width, "img_h": img.image.height,
        })
        if (i + 1) % 500 == 0:
            print(f"{args.tag}: {i + 1}/{len(rows)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"predictions_{args.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    meta = {"tag": args.tag, "rev": args.rev, "roastml_file": roastml.__file__, "n": len(out),
            "seconds": time.perf_counter() - t_start, "python": platform.python_version(), "numpy": np.__version__}
    (args.out / f"run_{args.tag}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))


def web1600(raw: bytes, quality: int = 85) -> bytes:
    with Image.open(io.BytesIO(raw)) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        rgb.thumbnail((1600, 1600))
        buf = io.BytesIO(); rgb.save(buf, "JPEG", quality=quality)
        return buf.getvalue()


def latency_sample(root: Path, per_source: int, seed: int, dense_paths: list[str]) -> list[dict]:
    rows = [r for r in manifest_rows(root) if r["source"] in LAT_SOURCES]
    rng = random.Random(seed)
    picked = []
    for s in LAT_SOURCES:
        pool = sorted((r for r in rows if r["source"] == s), key=lambda r: r["path"])
        picked += rng.sample(pool, min(per_source, len(pool)))
    by_path = {r["path"]: r for r in rows}
    picked += [by_path[p] for p in dense_paths if p in by_path]
    return picked


def cmd_latency(args) -> None:
    import roastml
    from roastml.api import load

    root = args.data_dir.resolve()
    dense = json.loads(args.dense.read_text(encoding="utf-8")) if args.dense else []
    rows = latency_sample(root, args.per_source, args.seed, dense)
    blobs = []
    for r in rows:
        raw = (root / r["path"]).read_bytes()
        blobs.append((r, "original", raw)); blobs.append((r, "web1600", web1600(raw)))
    pred = load(args.model)
    for _, _, raw in blobs[:10]:
        pred.predict_bytes(raw)  # warmup
    out = []
    for rep in range(args.repeat):
        for r, kind, raw in blobs:
            t0 = time.perf_counter(); res = pred.predict_bytes(raw); ms = (time.perf_counter() - t0) * 1000
            out.append({"path": r["path"], "source": r["source"], "kind": kind, "rep": rep, "ms": ms,
                        "n_beans": res["n_beans"], "status": res["status"]})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"predictions_latency_{args.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    meta = {"tag": args.tag, "rev": args.rev, "roastml_file": roastml.__file__, "n_images": len(rows),
            "repeat": args.repeat, "seed": args.seed, "per_source": args.per_source, "host": platform.processor()}
    (args.out / f"latency_{args.tag}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))


# ------------------------------------------------------------------ compare
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dup_pairs(boxes, thr=0.7) -> int:
    return sum(1 for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if iou(boxes[i], boxes[j]) > thr)


def read_pred(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def nb(r) -> int | None:
    return None if r["n_beans"] == "" else int(r["n_beans"])


def count_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    gt = np.array([int(r["n_gt"]) for r in rows], float)
    pr = np.array([nb(r) or 0 for r in rows], float)  # null → 0 (นับไม่ได้ = ผิดเท่าจำนวนจริง)
    err = np.abs(pr - gt)
    return {"n": len(rows), "MAE_null_as_0": round(float(err.mean()), 3),
            "MAPE": round(float(np.mean(err / np.maximum(gt, 1))), 3),
            "null_rate": round(float(np.mean([nb(r) is None for r in rows])), 3),
            "over_rate": round(float(np.mean(pr > gt)), 3), "under_rate": round(float(np.mean(pr < gt)), 3),
            "gt_median": float(np.median(gt)), "gt_max": int(gt.max())}


def hist(vals) -> dict:
    out: dict[str, int] = {}
    for v in vals:
        k = "null" if v is None else (str(v) if v <= 5 else (">5" if v <= 20 else ">20"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def summarize(P: dict[str, dict]) -> dict:
    rows = list(P.values())
    boos = [r for r in rows if r["source"] == "rf_boos"]
    s = {"count_rf_boos": {"all": count_block(boos)}}
    for m in ("beans", "pile", "full"):
        s["count_rf_boos"][f"mode_{m}"] = count_block([r for r in boos if r["mode"] == m])
    s["count_rf_boos"]["mixed_only"] = count_block([r for r in boos if r["label"] == "mixed"])
    one_boos = [r for r in boos if r["n_gt"] == "1"]
    one_ont = [r for r in rows if r["source"] == "ontoum224" and r["label"] in CLASSES]
    s["single_bean"] = {
        "rf_boos_gt1": {"n": len(one_boos), "n_beans_hist": hist(nb(r) for r in one_boos)},
        "ontoum224": {"n": len(one_ont), "n_beans_hist": hist(nb(r) for r in one_ont),
                      "exactly_1_rate": round(float(np.mean([nb(r) == 1 for r in one_ont])), 3),
                      "mean_n_beans_null_as_0": round(float(np.mean([nb(r) or 0 for r in one_ont])), 2)},
    }
    dups = {k: [r for r in rows if r["bboxes"] not in ("", "[]")] for k in ("all",)}["all"]
    per_img = [(r, dup_pairs(json.loads(r["bboxes"]))) for r in dups]
    s["dup_boxes_iou_gt_0.7"] = {
        "images_with_boxes": len(per_img), "images_with_dup": sum(d > 0 for _, d in per_img),
        "dup_pairs_total": sum(d for _, d in per_img),
        "by_source": {src: {"images_with_dup": sum(d > 0 for r, d in per_img if r["source"] == src),
                            "pairs": sum(d for r, d in per_img if r["source"] == src)}
                      for src in sorted({r["source"] for r, _ in per_img})}}
    empty = [r for r in rows if r["label"] == "empty"]
    green = [r for r in rows if r["label"] == "green"]
    s["empty_rf_hendi"] = {"n": len(empty), "n_beans_gt0": sum((nb(r) or 0) > 0 for r in empty),
                           "n_beans_eq0_or_null_rate": round(float(np.mean([(nb(r) or 0) == 0 for r in empty])), 3),
                           "no_beans_detected": sum(r["no_beans_detected"] == "True" for r in empty),
                           "mode": hist_str(r["mode"] for r in empty)}
    s["green_not_empty"] = {src: {"n": len(g), "n_beans_gt0": sum((nb(r) or 0) > 0 for r in g),
                                  "no_beans_detected": sum(r["no_beans_detected"] == "True" for r in g)}
                            for src in sorted({r["source"] for r in green})
                            for g in [[r for r in green if r["source"] == src]]}
    s["estimated_rate_by_source"] = {src: round(float(np.mean([r["estimated"] == "True" for r in rows if r["source"] == src])), 3)
                                     for src in sorted({r["source"] for r in rows})}
    return s


def hist_str(vals) -> dict:
    out: dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out


def lat_summary(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for kind in ("original", "web1600"):
        ms = np.array([float(r["ms"]) for r in rows if r["kind"] == kind])
        out[kind] = {"n_calls": int(ms.size), "p50": round(float(np.percentile(ms, 50)), 1),
                     "p95": round(float(np.percentile(ms, 95)), 1), "max": round(float(ms.max()), 1)}
    return out


def cmd_compare(args) -> None:
    old, new = read_pred(args.out / f"predictions_{args.old}.csv"), read_pred(args.out / f"predictions_{args.new}.csv")
    if set(old) != set(new):
        raise SystemExit("image sets differ between runs")
    both = [(old[p], new[p]) for p in sorted(old)]
    if any(o["mode"] != n["mode"] for o, n in both):
        raise SystemExit("segment mode differs between versions (segment.py should be unchanged)")
    replaced = [(o, n) for o, n in both if n["estimated"] == "True" and nb(o) is not None]
    changed = [(o, n) for o, n in both if nb(o) != nb(n)]
    lost_nbd = [(o, n) for o, n in both if o["no_beans_detected"] == "True" and n["no_beans_detected"] != "True"]
    boos_rep = [(o, n) for o, n in replaced if o["source"] == "rf_boos"]

    def err_delta(pairs):
        if not pairs:
            return None
        e_o = [abs((nb(o) or 0) - int(o["n_gt"])) for o, _ in pairs]
        e_n = [abs((nb(n) or 0) - int(n["n_gt"])) for _, n in pairs]
        return {"n": len(pairs), "MAE_old": round(float(np.mean(e_o)), 3), "MAE_new": round(float(np.mean(e_n)), 3),
                "new_worse": int(sum(b > a for a, b in zip(e_o, e_n))), "new_better": int(sum(b < a for a, b in zip(e_o, e_n)))}

    rep = {
        "rule": "trainval only; split=test never loaded; same model dir for both versions",
        "runs": {t: json.loads((args.out / f"run_{t}.json").read_text(encoding="utf-8")) for t in (args.old, args.new)},
        args.old: summarize(old), args.new: summarize(new),
        "edge_replaced_contour": {"images": len(replaced),
                                  "by_source": hist_str(o["source"] for o, _ in replaced),
                                  "rf_boos_error_on_replaced": err_delta(boos_rep)},
        "n_beans_changed": {"images": len(changed), "by_source": hist_str(o["source"] for o, _ in changed),
                            "null_to_value": sum(nb(o) is None and nb(n) is not None for o, n in changed)},
        "no_beans_detected_lost": {"images": len(lost_nbd), "by_source_label": hist_str(f"{o['source']}/{o['label']}" for o, _ in lost_nbd)},
    }
    for t in (args.old, args.new):
        lp = args.out / f"predictions_latency_{t}.csv"
        if lp.is_file():
            rep.setdefault("latency_ms", {})[t] = lat_summary(lp)
            rep.setdefault("latency_runs", {})[t] = json.loads((args.out / f"latency_{t}.json").read_text(encoding="utf-8"))
    (args.out / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--rev", required=True, help="git rev ของโค้ดที่รัน (บันทึกเท่านั้น)")
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    lat = sub.choices["latency"]
    lat.add_argument("--per-source", type=int, default=40)
    lat.add_argument("--seed", type=int, default=20261009)
    lat.add_argument("--repeat", type=int, default=3)
    lat.add_argument("--dense", type=Path, help="JSON list ของ path ภาพหนาแน่นที่จะรวมในชุด latency")
    c = sub.add_parser("compare")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--old", default="old")
    c.add_argument("--new", default="new")
    args = ap.parse_args(argv)
    {"run": cmd_run, "latency": cmd_latency, "compare": cmd_compare}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
