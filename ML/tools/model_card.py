"""Build a separate immutable card from measured results (adapted from 21ec097).

- ตรวจว่าโมเดลใน models/current ตรงกับ "selected" ใน baseline_loso.json (run / feature set / C) ไม่ตรง → error
- เพิ่ม: ตัวเลข LOSO (ก)/(ข) ราย fold + F1 รายคลาส + ผิดข้ามขั้น, ข้อจำกัด (ตัดสิน 8 ต.ค.), สถานะ agtron-test
- models/ อยู่ใน .gitignore (เทรนรวม agtron license Unknown) → ไฟล์ card ไม่ขึ้น git แต่สคริปต์นี้ขึ้น ทำซ้ำได้

ใช้: python -m tools.model_card --model-dir models/reference_b1_r2 --output results/<run>/reference_card.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
RESULTS = ML_DIR / "results"


def fold_summary(tab: dict, folds: list[str]) -> dict:
    out = {}
    for f in folds + ["mean", "pooled"]:
        if f not in tab:
            continue
        t = tab[f]
        row = {"macro_f1": round(t["macro_f1"], 4), "acc": round(t["acc"], 4),
               "cross_step_rate": round(t["cross_step_rate"], 4)}
        if "per_class" in t:
            row["n"] = t["n"]
            row["f1_per_class"] = {c: round(v["f1"], 4) for c, v in t["per_class"].items()}
        out[f] = row
    return out


def build_card(card: dict, rep: dict) -> dict:
    sel = rep["selected"]
    run, model = sel["run"], sel["model"]
    if (card.get("run"), card.get("feature_set"), card.get("C")) != (run, sel["feature_set"], sel["C"]):
        raise ValueError(f"โมเดลใน models/current ({card.get('run')}, {card.get('feature_set')}, {card.get('C')}) "
                         f"ไม่ตรงกับ selected ใน baseline_loso.json ({run}, {sel['feature_set']}, {sel['C']})")
    folds = list(card.get("loso_folds") or [])
    by_run = rep["by_run"]
    sel_tab = by_run[run][model]["folds"]
    no_agtron = [f for f in folds if f != "agtron"]

    def mean_excl_agtron(r: str) -> float:
        t = by_run[r][model]["folds"]["pipeline"]
        return round(float(np.mean([t[f]["macro_f1"] for f in no_agtron])), 4)

    ex = {r: mean_excl_agtron(r) for r in ("R1", "R2") if r in by_run}
    wb = rep.get("wb_guard", {}).get("rf_robusta", {})
    excluded = rep.get("excluded_sources", {})

    card = dict(card)
    card.update({
        "run": run,
        "run_description": rep["runs"][run],
        "selection": {"rule": rep["decision_rule"], "selected_by": sel["selected_by"],
                      "best_mean_pipeline_by_run": rep["decision"]["best_mean_pipeline_by_run"],
                      "r1_minus_r0": round(rep["decision"]["r1_minus_r0"], 4), "r1_kept": rep["decision"]["r1_kept"]},
        "loso": {
            "folds": folds,
            "eval_views": {"bbox": "(ก) train view: bbox robusta/boos · ROI agtron · smart crop ontoum224",
                           "pipeline": "(ข) smart-crop pipeline แบบที่ Pi เห็น (agtron = ROI)"},
            "bbox": fold_summary(sel_tab["bbox"], folds),
            "pipeline": fold_summary(sel_tab["pipeline"], folds),
            "agtron_by_value_pipeline": by_run[run][model]["agtron_by_value"],
        },
        "limitations": [
            "(a) fold agtron ถูกใช้ทำ error analysis (agtron 25 → light) ก่อนแก้ WB ใน box_features "
            "→ ตัวเลข agtron ใน LOSO มองโลกในแง่ดี",
            f"(b) ถ้าไม่นับ fold agtron, R1 ดีกว่า R2: mean (ข) 3 fold R1 = {ex.get('R1')} vs R2 = {ex.get('R2')} "
            "· กฎ auto (รวม agtron) เลือก R2",
            f"(c) rf_robusta {wb.get('box_wb_not_applied_pct')}% ของภาพไม่ผ่าน guard → ไม่ได้ทำ WB "
            "(พื้นหลังไม่สม่ำเสมอ/มีสี) ทั้งตอนเทรนและบน Pi",
            "(d) ตัด " + ", ".join(f"{k} ({v})" for k, v in excluded.items()) + " ออกจากการเทรนและ LOSO",
            "(e) ยังไม่มีตัวเลข agtron-test — รันครั้งเดียวตอน freeze ด้วย tools/eval_test.py เมื่อผู้ใช้สั่ง",
            "ข้อมูลทั้งหมดมาจากอินเทอร์เน็ต ไม่มีรูปถ่ายเอง (ml-spec ข้อ 5) · รูปจากมือถืออาจารย์อาจต่างจากทุก source",
            "ยังไม่ได้วัด latency/RAM บน Pi จริง (bench มีแต่ notebook)",
        ],
        "test": {"agtron_test": "ยังไม่รัน", "devices": "0:iPhone 12, 3:Motorola X4 (split=test แช่แข็ง)"},
        "card_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "card_source": "tools/model_card.py จาก results/baseline_loso.json " + rep.get("created", ""),
    })
    return card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=RESULTS / "baseline_loso.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    card_path = args.model_dir / "model_card.json"
    if args.output.exists() or args.output.resolve() == card_path.resolve():
        ap.error("output must be a new file; model artifacts remain immutable")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    rep = json.loads(args.results.read_text(encoding="utf-8"))
    test_status = card.get("test")  # ถ้า eval_test เคยรันแล้ว อย่าเขียนทับผล test
    if "selected" in rep and "by_run" in rep:
        new = build_card(card, rep)
    else:
        new = dict(card)
        new["measured_results"] = rep
        new["verification"] = {"pi_latency": "unverified", "fw_e2e": "unverified", "frozen_test": "not evaluated"}
        new["limitations"] = list(card.get("limitations", [])) + [
            "Internet datasets do not directly establish accuracy on the judging phone.",
            "Deployment selection requires measured Raspberry Pi p95 <=500 ms; notebook timing is insufficient.",
        ]
    if isinstance(test_status, dict) and test_status.get("agtron_test") not in (None, "ยังไม่รัน"):
        new["test"] = test_status
    from tools.bench_candidates import fingerprint
    new["artifact"] = fingerprint(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(new, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(f"→ {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
