"""End-to-end runtime parity: roastml.api.load(dir).predict_bytes(bytes) versus
the offline probabilities recorded for the same images (trainval only).

The offline export verification feeds cached 224px views straight into ORT.
This tool instead goes through the real FW path: file bytes -> decode (EXIF,
<=1600 px) -> shared RGB views -> ONNX -> contract dict, and checks label and
probability agreement (the contract rounds probabilities to 4 decimals).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from roastml.api import load
from tools.bench_candidates import checked_image_path, fingerprint
from tools.train_baseline import CLASSES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="offline predictions CSV for this model")
    ap.add_argument("--prefix", default="fp32_prob_")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error("output exists; choose a new file")
    root = args.data_dir.resolve()
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = {r["path"]: r for r in csv.DictReader(f)}
    with args.predictions.open(newline="", encoding="utf-8") as f:
        offline = list(csv.DictReader(f))
    rows = random.Random(args.seed).sample(offline, min(args.n, len(offline)))
    predictor = load(args.model)
    diffs, agree, records = [], 0, []
    t0 = time.perf_counter()
    for r in rows:
        m = manifest[r["path"]]
        raw = checked_image_path(root, m).read_bytes()  # refuses anything outside split=trainval
        res = predictor.predict_bytes(raw, source=m["source"], roi=m.get("roi", ""))
        ref = np.array([float(r[args.prefix + c]) for c in CLASSES])
        got = np.array([res["probs"][c] for c in CLASSES])
        diffs.append(float(np.abs(got - ref).max()))
        agree += res["label"] == CLASSES[int(ref.argmax())]
        records.append({"path": r["path"], "source": m["source"], "status": res["status"], "label": res["label"],
                        "offline_label": CLASSES[int(ref.argmax())], "max_abs_prob_diff": diffs[-1]})
    report = {"model": fingerprint(args.model), "predictor_info": predictor.info(), "n": len(rows),
              "label_agreement": agree / len(rows), "max_abs_prob_diff": max(diffs),
              "mean_abs_prob_diff": float(np.mean(diffs)), "tolerance_note": "contract rounds probs to 4 decimals",
              "seconds": time.perf_counter() - t0, "records": records, "frozen_test_loaded": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("n", "label_agreement", "max_abs_prob_diff", "mean_abs_prob_diff")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
