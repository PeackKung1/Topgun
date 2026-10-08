"""Proposed keep/review/drop categories for trainval images (proposal only).

Criteria come from the user's request (2026-10-08): phone-camera photos,
one roast level per image, a single bean or several beans, plain background
or a clearly visible subject. Rules use only manifest/QA-ledger metadata and
visual review of the QA contact sheets - never model predictions, never
absolute colour across sources. Nothing is deleted; raw/ is untouched.
The CSV goes to $ROAST_DATA_DIR/cache (it lists image paths); ML/results gets
counts only. Applying it to training is a separate, user-approved ablation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tools.train_d1 import ML_DIR, dump, sha256

CONTRACT = {"light", "medium", "dark"}

RULES = {
    "drop_not_contract_class": "label is green/empty/mixed (not light/medium/dark): mixed roast or not roasted",
    "drop_roaster_interior": "rf_hendi: beans inside a roaster drum, dim light, machine parts dominate",
    "keep_phone_hd": "agtron: original phone photos (iPhone 11/8, Redmi 5A, Galaxy A72, Mi 9; ~12 MP), close-up bean pile, mandatory ROI",
    "keep_single_bean_plain": "ontoum224: one bean on plain white background (224 px only, not HD)",
    "review_color_cast_few_beans": "rf_boos: 3-6 beans on white, but video frames stretched to 640 px with a strong blue/grey cast; only 2 videos per class",
    "keep_multi_bean_clear": "rf_robusta album photo where segmentation found beans or a pile: many beans, clearly visible (416 px stretched, varied containers)",
    "review_cluttered_scene": "rf_robusta album photo where segmentation fell back to the full frame (coloured plate/cloth). "
                              "Contact sheet: most are clear multi-bean photos (keepable); a few red-checkered-cloth photos "
                              "look like greenish/mixed beans -> review those",
    "review_label_ambiguous_video": "rf_robusta 'Coffee-Test' video (one video for all labels): bowl from above, beans small; "
                                    "'light' frames look medium-brown; some frames show no bowl (hand/empty table/black bars) -> needs user decision",
}


def category(r: dict) -> str:
    src, label = r["source"], r["label"]
    if label not in CONTRACT:
        return "drop_not_contract_class"
    if src == "rf_hendi":
        return "drop_roaster_interior"
    if src == "agtron":
        return "keep_phone_hd"
    if src == "ontoum224":
        return "keep_single_bean_plain"
    if src == "rf_boos":
        return "review_color_cast_few_beans"
    if src == "rf_robusta":
        if ":video:" in r["group"]:
            return "review_label_ambiguous_video"
        return "review_cluttered_scene" if r["pipeline_mode"] == "full" else "keep_multi_bean_clear"
    raise ValueError(f"unknown source {src}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.data_dir.resolve()
    ledger = root / "cache/qa_audit_20261008/ledger.csv"
    with ledger.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "trainval"]
    out_csv = root / "cache/dataset_selection_20261008.csv"
    counts, pool = Counter(), Counter()
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "group", "category", "decision", "in_r2_reference_pool"])
        for r in rows:
            c = category(r)
            in_pool = r["reference_model_pool"] in ("True", "1")
            w.writerow([r["path"], r["source"], r["label"], r["group"], c, c.split("_")[0], in_pool])
            counts[(r["source"], r["label"], c)] += 1
            if in_pool:
                pool[c] += 1
    out = ML_DIR / "results/dataset_selection_20261008"
    dump(out / "summary.json", {
        "rules": RULES, "ledger_sha256": sha256(ledger), "selection_csv": str(out_csv), "selection_sha256": sha256(out_csv),
        "counts": [{"source": s, "label": l, "category": c, "n": n} for (s, l, c), n in sorted(counts.items())],
        "reference_pool_by_category": dict(pool), "applied_to_training": False,
        "note": "proposal only; no file deleted; frozen test not read"})
    for k, n in sorted(counts.items()):
        print(*k, n, sep="\t")
    print("R2 pool:", dict(pool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
