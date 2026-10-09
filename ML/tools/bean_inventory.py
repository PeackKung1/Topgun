"""Inventory ข้อมูลรายเมล็ด ต่อ source (trainval เท่านั้น · ไม่โหลด split=test)

นิยาม (ตัดสินก่อนรัน):
- ภาพระดับเดียว = label ∈ light/medium/dark (ไม่รวม mixed/green/empty)
- "แยกเมล็ดได้" = segment(find_beans=True) ได้ mode=beans และตัวนับ contour (ไม่ใช้ edge) ได้ ≥ 1 เมล็ด
  และไม่มี blob ที่พื้นที่ bbox > 2.5× median ของภาพนั้น (= ไม่มีกลุ่มเมล็ดแตะกัน)
  rf_boos มี GT box รายเมล็ด → ต้องนับได้เท่ากับ GT ด้วย
- จำนวนเมล็ด: rf_boos = จำนวน GT box · source อื่น = จำนวน contour ในภาพที่แยกได้ (ไม่มี GT รายเมล็ด)
- ขนาดเมล็ด = sqrt(w*h) ของ bbox (px ในภาพ decode ≤1600 px) และสัดส่วนต่อด้านสั้นของภาพ
- agtron: ใช้ ROI crop บังคับ แล้วลอง find_beans=True (pipeline จริงใช้ find_beans=False)
- rf_robusta: box YOLO เป็นกรอบพื้นที่ ไม่ใช่รายเมล็ด → รายงานจำนวน box แยก ไม่นับเป็นเมล็ด
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from roastml.beans import split_beans
from roastml.decode import decode_image
from roastml.rgb_views import crop_roi
from roastml.segment import SegConfig, segment

CLASSES = ("light", "medium", "dark")
SOURCES = ("ontoum224", "rf_boos", "rf_robusta", "rf_hendi", "agtron")


def yolo_boxes(img_path: Path) -> list[tuple[float, float, float, float]]:
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not lbl.is_file():
        raise FileNotFoundError(f"no YOLO label for {img_path}")
    return [tuple(float(v) for v in line.split()[3:5]) for line in lbl.read_text(encoding="utf-8").splitlines()
            if len(line.split()) >= 5]  # (w, h) สัมพัทธ์


def pct(vals) -> dict | None:
    if not len(vals):
        return None
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return {"p10": round(float(p10), 3), "median": round(float(p50), 3), "p90": round(float(p90), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="อ่าน seg_config จาก model_card.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.data_dir.resolve()
    cfg = SegConfig.from_dict(json.loads((args.model / "model_card.json").read_text(encoding="utf-8")).get("seg_config"))
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == "trainval" and r["source"] in SOURCES and r["label"] in CLASSES]
    t0 = time.perf_counter()
    per = []
    for r in rows:
        p = (root / r["path"]).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"path escapes data root: {r['path']}")
        img = decode_image(p.read_bytes())
        rgb = crop_roi(img, r["roi"]) if r["source"] == "agtron" else np.asarray(img.image)
        H, W = rgb.shape[:2]
        seg = segment(rgb, cfg, find_beans=True)
        beans = split_beans(seg, cfg) if seg.mode == "beans" else None
        sizes = [float(np.sqrt(b.bbox[2] * b.bbox[3])) for b in (beans or [])]
        areas = np.array([b.bbox[2] * b.bbox[3] for b in (beans or [])], float)
        cluster = bool(areas.size and (areas > 2.5 * np.median(areas)).any())
        rec = {"source": r["source"], "label": r["label"], "mode": seg.mode, "n_contour": len(beans or []),
               "cluster": cluster, "short_side": min(H, W), "sizes": sizes, "n_gt": None, "gt_sizes": []}
        if r["source"] in ("rf_boos", "rf_robusta"):
            bx = yolo_boxes(p)
            rec["n_gt"] = len(bx)
            rec["gt_sizes"] = [float(np.sqrt(w * W * h * H)) for w, h in bx]
        sep = seg.mode == "beans" and rec["n_contour"] >= 1 and not cluster
        if r["source"] == "rf_boos":
            sep = sep and rec["n_contour"] == rec["n_gt"]
        rec["separable"] = sep
        per.append(rec)

    out = {"definition": __doc__.split("นิยาม (ตัดสินก่อนรัน):")[1].strip(), "frozen_test_loaded": False,
           "seg_config_from": str(args.model / "model_card.json"), "sources": {}}
    for s in SOURCES:
        sel = [x for x in per if x["source"] == s]
        sep = [x for x in sel if x["separable"]]
        if s == "rf_boos":
            beans_total = int(sum(x["n_gt"] for x in sep))
            sz = [v for x in sep for v in x["gt_sizes"]]
            rel = [v / x["short_side"] for x in sep for v in x["gt_sizes"]]
            size_src = "GT box"
        else:
            beans_total = int(sum(x["n_contour"] for x in sep))
            sz = [v for x in sep for v in x["sizes"]]
            rel = [v / x["short_side"] for x in sep for v in x["sizes"]]
            size_src = "contour bbox"
        out["sources"][s] = {
            "single_level_images": len(sel),
            "separable_images": len(sep),
            "separable_by_label": {c: sum(x["label"] == c for x in sep) for c in CLASSES},
            "beans_total_in_separable": beans_total,
            "bean_size_px": pct(sz), "bean_size_rel_short_side": pct(rel), "size_source": size_src,
            "mode_counts": {m: sum(x["mode"] == m for x in sel) for m in ("beans", "pile", "full")},
            "images_with_cluster_blob": sum(x["cluster"] for x in sel),
            "image_short_side_px": pct([x["short_side"] for x in sel]),
        }
        if s == "rf_boos":
            out["sources"][s]["gt_boxes_all_single_level"] = int(sum(x["n_gt"] for x in sel))
        if s == "rf_robusta":
            out["sources"][s]["yolo_region_boxes_per_image"] = pct([x["n_gt"] for x in sel])
            out["sources"][s]["note"] = "YOLO boxes = กรอบพื้นที่ (ไม่ใช่รายเมล็ด)"
    out["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["sources"], indent=1))  # ascii: console Windows เป็น cp1252
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
