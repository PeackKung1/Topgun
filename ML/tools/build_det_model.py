"""ประกอบโฟลเดอร์โมเดล det_b1_beans สำหรับ deploy: detector (ONNX) + B1 สี + model_card ที่เขียนผลวัดจริงและเกณฑ์ที่ไม่ผ่าน

    python -m tools.build_det_model --detector models/yolo_bean_20261010/final/bean.onnx \
        --feature-cache <perbean_features_*.npz> --eval results/.../eval_holdout_boos/summary.json \
        --eval-final results/.../eval_final/summary.json --selection results/.../label_selection.json --out models/det_b1_20261010

- B1: สูตรเดียวกับ B1 R2 (Lab_hist, C = 0.01, ไม่เกิน 10 เฟรม/วิดีโอ) เทรนจาก feature cache ของ P1
  = trainval ที่ไม่ใช่ group ที่ freeze ทั้ง 4 source · ไม่อ่านโมเดลเก่า · ไม่มีการเลือก hyperparameter
- ไม่เขียนทับโฟลเดอร์ที่มีอยู่ · models/ ไม่เข้า git (agtron license Unknown · Ultralytics AGPL-3.0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from roastml import __version__
from roastml.detector import DetConfig
from roastml.segment import SegConfig
from tools.perbean_protocol import FOLDS, GATES
from tools.train_perbean import CLASSES, fit_broadcast


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--feature-cache", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True, help="summary.json ของ detector ที่ไม่เคยเห็น rf_boos (หลักฐาน held-out)")
    ap.add_argument("--eval-final", type=Path, required=True, help="summary.json ของ detector ตัวที่ deploy")
    ap.add_argument("--selection", type=Path, required=True, help="label_selection.json จาก tools.select_bean_labels (วิธีให้ label รายเมล็ด)")
    ap.add_argument("--bench", type=Path, help="ผล bench บน notebook (ถ้ามี)")
    ap.add_argument("--name", default="det-yolo11n-b1-Lab_hist")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("ใช้โฟลเดอร์ใหม่ — ไม่เขียนทับโมเดลเดิม")
    meta = json.loads(args.feature_cache.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(args.feature_cache, allow_pickle=False)
    records = [{**r, "train_x": z[f"tr{i}"]} for i, r in enumerate(meta["records"])]
    single = [r for r in records if r["row"]["label"] in CLASSES and r["row"]["source"] in FOLDS]
    if any(r["row"]["split"] != "trainval" for r in single):
        raise ValueError("feature cache มีแถวที่ไม่ใช่ trainval")
    spec = fit_broadcast(single)
    ev, evf, sel = (json.loads(p.read_text(encoding="utf-8")) for p in (args.eval, args.eval_final, args.selection))
    chosen = sel["candidates"][sel["chosen"]]
    label_card = sel["chosen_card"]
    base = sel["candidates"]["broadcast"]
    folds = {h: {"images": ev["folds"][h]["images"], "zero_detection_images": ev["folds"][h]["zero_detection_images"],
                 "B1 broadcast": round(base["fold_macro_f1"][h], 4), "deployed": round(chosen["fold_macro_f1"][h], 4),
                 "delta": round(chosen["fold_delta"][h], 4), "noninferiority_pass": bool(chosen["fold_delta"][h] >= -GATES["fold_f1_drop"])}
             for h in FOLDS}
    failed = [f"per-bean macro-F1 non-inferiority on fold {h} ({f['delta']:+.4f} < -{GATES['fold_f1_drop']})"
              for h, f in folds.items() if not f["noninferiority_pass"]]
    if chosen["light_dark_rate"] > GATES["cross_extreme_rate"]:
        failed.append(f"light<->dark rate {chosen['light_dark_rate']:.4f} > {GATES['cross_extreme_rate']}")
    if ev["count"]["rf_boos_all"]["MAE"]["value"] > GATES["boos_mae"]:
        failed.append("rf_boos count MAE (held-out detector)")
    if evf["count"]["empty_report_half"]["zero_rate"]["value"] < GATES["empty_zero_rate"]:
        failed.append("Empty zero-rate (report half)")
    args.out.mkdir(parents=True)
    (args.out / "model.json").write_text(json.dumps(spec), encoding="utf-8")
    shutil.copyfile(args.detector, args.out / "bean.onnx")
    card = {
        "backend": "det_b1_beans", "name": args.name, "model_file": "model.json", "detector_file": "bean.onnx",
        "det_config": DetConfig().to_dict(), **label_card, "seg_config": SegConfig().to_dict(),
        "low_conf_threshold": 0.0, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "schema_version": 3,
        "feature_set": "Lab_hist", "C": 0.01,
        "train": {"b1": {"rows_single_roast": len(single), "sources": list(FOLDS), "video_cap": 10, "split": "trainval minus count-test frozen groups",
                         "counts": {c: sum(r["row"]["label"] == c for r in single) for c in CLASSES}},
                  "detector": "YOLO11n single class, synthetic copy-paste scenes (ontoum224 + rf_boos bean cutouts) + real rf_boos frames + rf_hendi Empty tune-half negatives"},
        "measured": {
            "count_heldout_detector(never saw rf_boos)": {k: ev["count"][k] for k in ("rf_boos_all", "empty_report_half", "count_dev_flat")},
            "count_deployed_detector": {k: evf["count"][k] for k in evf["count"] if k != "rf_boos_by_group"},
            "per_bean_paired_folds(heldout detector; B1 fold weights never saw the fold source)": folds,
            "pooled_light_dark_rate": {"B1 broadcast": base["light_dark_rate"], "deployed": chosen["light_dark_rate"]},
            "mixed_rf_boos_error_per_image(102 frames, 6 beans each)": {"B1 broadcast": base["mixed_error_per_image"],
                                                                         "deployed": chosen["mixed_error_per_image"]},
            "label_selection": {"chosen": sel["chosen"], "status": sel["status"], "selection_bias": sel["selection_bias"]}},
        "gates": {**{k: GATES[k] for k in ("flat_mae", "touching_mape", "pile_mape", "boos_mae", "empty_zero_rate", "fold_f1_drop", "cross_extreme_rate", "pi_p95_ms")},
                  "failed": failed, "unmeasured": ["touching MAPE (no GT)", "pile MAPE (no GT)", "Pi 5 p95", "count-test frozen", "split=test"]},
        "deployment_eligible": False,
        "limitations": [
            "detector เรียนจากฉากสังเคราะห์ (ตัดแปะเมล็ดจริง) — ภาพกอง/แตะกันจริงยังไม่มี GT ให้วัด",
            "นับเฉพาะเมล็ดที่ detector เห็น · เมล็ดที่ถูกบังไม่ถูกนับ · ภาพแน่นได้ count_method = estimated",
            "rf_robusta ใช้ประเมินการนับไม่ได้ (กรอบเป็นกรอบทั้งกอง ไม่ใช่รายเมล็ด)",
            "confidence = 0.999×สัดส่วนคลาสข้างมาก + 0.001×prob เฉลี่ย — ไม่ใช่ความมั่นใจของตัวจำแนก",
            "เมล็ดเล็กมากในภาพ (จาน/ถ้วยถ่ายไกล) ตรวจไม่เจอ · จานอาจถูกมองเป็นเมล็ดใหญ่ (มีกฎตัดกรอบภาชนะช่วยบางส่วน)",
            "วิธีให้ label รายเมล็ดถูกเลือกบน dev LOSO → ตัวเลขมีอคติด้านบวก",
            "ยังไม่ได้วัด latency/RAM บน Pi จริง · ยังไม่ได้ประเมิน frozen/test",
            "เทรนรวม agtron (license Unknown) — ห้าม commit · detector เทรนด้วย Ultralytics (AGPL-3.0)"],
        "versions": {"roastml": __version__},
        "result_sources": {"eval_heldout": {"path": str(args.eval), "sha256": sha(args.eval)},
                           "eval_final": {"path": str(args.eval_final), "sha256": sha(args.eval_final)},
                           "selection": {"path": str(args.selection), "sha256": sha(args.selection)},
                           "detector_sha256": sha(args.detector), "feature_cache": str(args.feature_cache)},
    }
    if args.bench:
        card["measured"]["notebook_bench"] = json.loads(args.bench.read_text(encoding="utf-8"))
        card["result_sources"]["bench"] = {"path": str(args.bench), "sha256": sha(args.bench)}
    (args.out / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "label_card": label_card, "failed_gates": failed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
