"""ประเมิน detector เมล็ด + สีรายกรอบ บนภาพ dev จริง ผ่านเส้นทางเดียวกับ FW (ภาพเต็มเฟรม ไม่มี ROI)

    python -m tools.eval_det_beans --detector models/yolo_bean_20261010/holdout_boos/bean.onnx \
        --b1-folds <dir ที่มี <fold>/model.json> --out results/yolo_bean_20261010/eval_holdout_boos

นับ   : rf_boos (GT = จำนวนกรอบ YOLO) · ontoum224 (GT = 1) · rf_hendi Empty ครึ่ง report (GT = 0) · agtron/rf_robusta (ไม่มี GT → รายงานการกระจาย)
จำแนก : paired ราย fold — "B1 broadcast" (ทุกเมล็ดได้ label ของภาพ) เทียบ "det+B1 ต่อกรอบ" บนกรอบชุดเดียวกัน
        น้ำหนัก B1 ของแต่ละ fold ไม่เคยเห็น source นั้น (โมเดลราย fold ของ P1) · ถ่วง 1/(เมล็ดต่อภาพ)
ปน   : rf_boos Mixed — error ต่อภาพ = Σ|pred_c − gt_c|
ใช้เฉพาะ DevData (trainval ที่ไม่ใช่ group ที่ freeze) · ไม่เลือก threshold ใด ๆ จากผลนี้
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from roastml.decode import decode_image
from roastml.detector import BeanDetector, DetConfig
from roastml.features import pixel_stats
from roastml.linear_model import LinearSoftmax
from roastml.paths import data_dir
from roastml.perbean import box_bean_features, group_predictions
from roastml.segment import SegConfig, prepare_image, segment
from tools.index_sources import read_yolo_boxes, read_yolo_names
from tools.perbean_protocol import FOLDS, GATES, DevData, declare, dump
from tools.train_perbean import CLASSES, weighted_metrics
from tools.tune_counter import bootstrap

_state = {}


def init_worker(root, detector, det_cfg):
    import cv2
    cv2.setNumThreads(1)
    _state.update(data=DevData(root), det=BeanDetector(detector, DetConfig.from_dict(det_cfg)), seg=SegConfig())


def evaluate_row(r):
    data, det, sc = _state["data"], _state["det"], _state["seg"]
    data.assert_safe(r)
    rgb = np.asarray(decode_image((data.root / r["path"]).read_bytes()).image)
    boxes, scores = det.detect(rgb)
    rec = {"path": r["path"], "source": r["source"], "group": r["group"], "label": r["label"], "n": int(len(boxes)),
           "size": [int(rgb.shape[1]), int(rgb.shape[0])], "boxes": boxes.tolist(), "scores": np.round(scores, 3).tolist()}
    if r["source"] == "rf_boos":
        gt = read_yolo_boxes(data.root / r["path"])
        rec["gt"] = len(gt)
        rec["gt_classes"] = [b[0] for b in gt]
    elif r["source"] == "ontoum224":
        rec["gt"] = 1
    elif r["source"] == "rf_hendi":
        rec["gt"] = 0
    if r["source"] in FOLDS and r["label"] in CLASSES + ["mixed"]:
        prep = prepare_image(rgb, sc)
        seg = segment(rgb, sc, prepared=prep)
        rec["image_x"] = pixel_stats(seg.lab[seg.pixel_mask])
        work_boxes = boxes.astype(np.float64) * prep.scale
        rec["X"] = box_bean_features(prep.wb.lab, work_boxes, sc)                             # สูตรที่ใช้เทรน B1 (rect, shrink 0.2)
        rec["Xe"] = box_bean_features(prep.wb.lab, work_boxes, sc, shrink=0.1, shape="ellipse")  # ตัวเลือกสำหรับ tools.select_bean_labels
        rec["Xm"] = (box_bean_features(prep.wb.lab, work_boxes, sc, shape="mask", mask=seg.pixel_mask)
                     if seg.mode != "full" else rec["X"])                                          # เหมือน det_backend
        rec["seg_mode"] = seg.mode
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--b1-folds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="จำกัดภาพต่อ source (ทดสอบเร็ว)")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    data = DevData(data_dir())
    det_cfg = DetConfig(conf=args.conf, threads=1, tile=args.tile).to_dict()
    declare(args.out, "det-beans-eval", {
        "selection": "none; conf/NMS predeclared (ultralytics defaults, max_det 1000)", "path": "FW path: full decoded frame, no ROI, WB as served",
        "count_gt": "rf_boos YOLO boxes; ontoum224 = 1; rf_hendi Empty report-half = 0; agtron/rf_robusta have no per-bean GT",
        "class": "paired per fold vs B1 broadcast on the same boxes; B1 fold weights never saw the fold source; weight 1/n per image",
        "note": args.note, "limit_per_source": args.limit}, {"detector": str(args.detector), "det_config": det_cfg}, data)
    _, empty_report = data.empty_split()
    rows = [r for r in data.rows if r["source"] in FOLDS and r["label"] in CLASSES + ["mixed"]] + empty_report
    rows = sorted(rows, key=lambda r: r["path"])
    if args.limit:
        rng = np.random.default_rng(20261010)
        rows = [r for s in (*FOLDS, "rf_hendi") for r in
                (lambda sub: [sub[i] for i in sorted(rng.choice(len(sub), min(args.limit, len(sub)), replace=False))])([r for r in rows if r["source"] == s])]
    if args.workers == 1:
        init_worker(data.root, args.detector, det_cfg)
        records = [evaluate_row(r) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(data.root, args.detector, det_cfg)) as pool:
            records = []
            for i, rec in enumerate(pool.map(evaluate_row, rows, chunksize=8)):
                records.append(rec)
                if (i + 1) % 400 == 0:
                    print(f"evaluated {i + 1}/{len(rows)}", flush=True)

    # ---------------- count
    def count_block(rr):
        if not rr:
            return {"n": 0}
        err = np.array([abs(r["n"] - r["gt"]) for r in rr], float)
        return {"n": len(rr), "MAE": bootstrap(err), "exact_rate": float((err == 0).mean()),
                "over_rate": float(np.mean([r["n"] > r["gt"] for r in rr])), "under_rate": float(np.mean([r["n"] < r["gt"] for r in rr]))}

    boos = [r for r in records if r["source"] == "rf_boos"]
    count = {"rf_boos_all": count_block(boos),
             "rf_boos_by_group": {g: count_block([r for r in boos if r["group"] == g]) for g in sorted({r["group"] for r in boos})},
             "ontoum224": count_block([r for r in records if r["source"] == "ontoum224"]),
             "empty_report_half": {"n": sum(r["source"] == "rf_hendi" for r in records),
                                   "zero_rate": bootstrap([float(r["n"] == 0) for r in records if r["source"] == "rf_hendi"])}}
    dev_flat = {r["manifest_path"] if "manifest_path" in r else r["path"]: int(r["n_total"]) for r in data.count_dev if r.get("n_total") and r["scene"] == "flat"}
    count["count_dev_flat"] = bootstrap([abs(r["n"] - dev_flat[r["path"]]) for r in records if r["path"] in dev_flat])
    for s in ("agtron", "rf_robusta"):
        v = np.array([r["n"] for r in records if r["source"] == s])
        if len(v):
            count[f"{s}_n_distribution(no GT)"] = {"images": len(v), "p5_p25_p50_p75_p95": np.percentile(v, [5, 25, 50, 75, 95]).tolist(),
                                                   "zero": int((v == 0).sum()), "one": int((v == 1).sum()), "max": int(v.max())}

    # ---------------- per-bean class, paired per fold
    METHODS = ("B1 broadcast", "det+B1 per box", "det+B1+group")
    folds, pooled = {}, {k: np.zeros((3, 3)) for k in METHODS}
    group_cfg = {h: json.loads((args.b1_folds / h / "model_card.json").read_text(encoding="utf-8"))["group_config"] for h in FOLDS}

    def bean_labels(b1, r, cfg):
        """label ต่อกรอบ 2 แบบ: argmax ต่อกรอบ · จัดกลุ่มความสว่าง (gap เลือกจาก train fold ของ P1 ไม่ได้ดูผลนี้)"""
        P = b1.proba(r["X"])
        grouped, _, info = group_predictions(r["X"], P, list(b1.classes), cfg)
        return np.array(b1.classes)[P.argmax(1)], np.array(b1.classes)[grouped], info["k"]

    for hold in FOLDS:
        b1 = LinearSoftmax.load(args.b1_folds / hold / "model.json")
        test = [r for r in records if r["source"] == hold and r["label"] in CLASSES]
        yt, ww, preds, split_single = [], [], {k: [] for k in METHODS}, 0
        for r in test:
            n = r["n"]
            if not n:
                continue
            image_pred = CLASSES[int(b1.proba(r["image_x"])[0].argmax())]
            per_box, grouped, k = bean_labels(b1, r, group_cfg[hold])
            split_single += int(k > 1)
            yt += [r["label"]] * n; ww += [1 / n] * n
            preds["B1 broadcast"] += [image_pred] * n; preds["det+B1 per box"] += list(per_box); preds["det+B1+group"] += list(grouped)
        if not yt:
            folds[hold] = {"images": len(test), "zero_detection_images": len(test)}
            continue
        m = {k: weighted_metrics(yt, preds[k], ww) for k in METHODS}
        for k in pooled:
            pooled[k] += np.array(m[k]["cm"])
        base = m["B1 broadcast"]["macro_f1"]
        folds[hold] = {"images": len(test), "zero_detection_images": sum(r["n"] == 0 for r in test), "paired": m,
                       "group_config": group_cfg[hold], "single_roast_images_split_by_group": split_single,
                       "delta": {k: m[k]["macro_f1"] - base for k in METHODS[1:]},
                       "noninferiority_pass": {k: bool(m[k]["macro_f1"] >= base - GATES["fold_f1_drop"]) for k in METHODS[1:]}}
    extreme = {k: float((cm[0, 2] + cm[2, 0]) / max(cm[0].sum() + cm[2].sum(), 1e-9)) for k, cm in pooled.items()}

    # ---------------- mixed (rf_boos Mixed, YOLO per-bean classes)
    names = read_yolo_names(data.root / "raw/rf_boos/data.yaml")
    mapping = [next(c for c in CLASSES if c in n.lower()) for n in names]
    b1 = LinearSoftmax.load(args.b1_folds / "rf_boos" / "model.json")
    mixed = []
    for r in records:
        if r["source"] != "rf_boos" or r["label"] != "mixed":
            continue
        gt = np.bincount([CLASSES.index(mapping[c]) for c in r["gt_classes"]], minlength=3)
        base = np.eye(3, dtype=int)[int(b1.proba(r["image_x"])[0].argmax())] * r["n"]
        if r["n"]:
            per_box, grouped, k = bean_labels(b1, r, group_cfg["rf_boos"])
            cand = np.bincount([CLASSES.index(c) for c in per_box], minlength=3)
            cand_g = np.bincount([CLASSES.index(c) for c in grouped], minlength=3)
        else:
            cand = cand_g = np.zeros(3, int)
            k = 0
        mixed.append({"path": r["path"], "gt": gt.tolist(), "n": r["n"], "B1 broadcast": int(np.abs(base - gt).sum()),
                      "det+B1 per box": int(np.abs(cand - gt).sum()), "det+B1+group": int(np.abs(cand_g - gt).sum()),
                      "pred": cand.tolist(), "pred_group": cand_g.tolist(), "group_k": k})
    mixed_summary = {k: bootstrap([m[k] for m in mixed]) for k in METHODS}

    gates = {"boos_mae": {"value": count["rf_boos_all"]["MAE"]["value"], "gate": GATES["boos_mae"]},
             "flat_mae_dev(n=%d)" % count["count_dev_flat"]["n"]: {"value": count["count_dev_flat"]["value"], "gate": GATES["flat_mae"]},
             "empty_zero_rate": {"value": count["empty_report_half"]["zero_rate"]["value"], "gate": GATES["empty_zero_rate"]},
             "cross_extreme_rate": {"value": {k: extreme[k] for k in METHODS[1:]}, "gate": GATES["cross_extreme_rate"]},
             "fold_noninferiority": {h: f.get("noninferiority_pass") for h, f in folds.items()},
             "touching_mape": "unmeasured (no GT)", "pile_mape": "unmeasured (no GT)", "pi_p95_ms": "unmeasured"}
    dump(args.out / "summary.json", {"detector": str(args.detector), "det_config": det_cfg, "note": args.note, "images": len(records),
                                     "count": count, "folds": folds, "pooled_light_dark_rate": extreme,
                                     "mixed_rf_boos": {"n": len(mixed), "error_per_image": mixed_summary}, "gates": gates})
    dump(args.out / "per_image.json", [{k: v for k, v in r.items() if k not in ("X", "Xe", "Xm", "image_x")} for r in records])
    dump(args.out / "mixed_per_image.json", mixed)
    # feature ต่อกรอบ (ตัวเลขล้วน) เก็บนอก git ไว้วิเคราะห์ซ้ำโดยไม่ต้องรัน detector ใหม่
    cache = data.root / "cache" / f"det_eval_{args.out.parent.name}_{args.out.name}.npz"
    keep = [i for i, r in enumerate(records) if "X" in r]
    np.savez_compressed(cache, paths=np.array([records[i]["path"] for i in keep]),
                        **{f"x{j}": records[i]["X"] for j, i in enumerate(keep)}, **{f"e{j}": records[i]["Xe"] for j, i in enumerate(keep)}, **{f"m{j}": records[i]["Xm"] for j, i in enumerate(keep)},
                        **{f"im{j}": records[i]["image_x"] for j, i in enumerate(keep)})
    print(json.dumps({"count": {k: (v if not isinstance(v, dict) or "MAE" not in v else {"n": v["n"], "MAE": v["MAE"]["value"], "exact": v["exact_rate"]})
                                for k, v in count.items() if k != "rf_boos_by_group"},
                      "folds": {h: ({n: round(m["macro_f1"], 4) for n, m in f["paired"].items()} | {"zero_det": f["zero_detection_images"]}) if "paired" in f else f
                                for h, f in folds.items()},
                      "light_dark": extreme, "mixed": {k: v["value"] for k, v in mixed_summary.items()}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
