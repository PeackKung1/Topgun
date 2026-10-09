"""Frozen evaluation, gated before data access; never run until user authorizes.

Adapted from commit 21ec097 with backend-neutral prediction and immutable outputs.
No threshold/view/model tuning arguments and no modification of model cards.
Example AFTER explicit authorization:
python -m tools.eval_test --confirm-freeze --model-dir <frozen model> --output-dir <new run dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from roastml.api import load
from roastml.contract import LABELS
from roastml.paths import data_dir
from tools.bench_candidates import fingerprint, runtime_provenance, versions
from tools.train_baseline import metrics

CLASSES = list(LABELS)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - half, center + half]


def load_test_rows(root: Path, *, confirmed: bool = False) -> list[dict]:
    if not confirmed:
        raise PermissionError("frozen test rows require explicit freeze confirmation")
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == "test" and r.get("label") in LABELS]
    if any(r.get("source") == "agtron" and not r.get("roi") for r in rows):
        raise ValueError("Agtron test rows require ROI metadata; full images are forbidden")
    return rows


def predict_row(predictor, root: Path, row: dict, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise PermissionError("frozen image access requires explicit freeze confirmation")
    if row.get("split") != "test":
        raise ValueError("evaluator only reads split=test after explicit authorization")
    if row.get("source") == "agtron" and not row.get("roi"):
        raise ValueError("Agtron requires ROI metadata before image access")
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest image path escapes data root")
    result = predictor.predict_dataset(path.read_bytes(), source=row["source"], roi=row.get("roi", ""))
    if result["label"] not in LABELS or result["probs"] is None:
        raise RuntimeError(f"model failed frozen image: {row['path']}: {result['status']}")
    return result


def calibration(y: np.ndarray, P: np.ndarray, bins: int = 10) -> dict:
    confidence, predicted = P.max(1), np.asarray(CLASSES)[P.argmax(1)]
    correct = predicted == y
    reliability, ece = [], 0.0
    for i in range(bins):
        mask = (confidence >= i / bins) & (confidence <= 1 if i == bins - 1 else confidence < (i + 1) / bins)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(acc - conf)
        reliability.append({"lo": i / bins, "hi": (i + 1) / bins, "n": n, "acc": acc, "confidence": conf})
    return {"ece": ece, "bins": reliability, "probability_precision": "FW contract rounded to four decimals"}


def breakdown(keys: list[str], y: np.ndarray, p: np.ndarray) -> dict:
    return {key: metrics(y[np.asarray(keys) == key], p[np.asarray(keys) == key]) for key in sorted(set(keys))}


def agtron_flash(root: Path, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise PermissionError("frozen metadata access requires explicit freeze confirmation")
    path = root / "raw" / "agtron" / "photos.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["name_coffee"]: row.get("flash", "unknown") for row in csv.DictReader(f, delimiter=";")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm-freeze", action="store_true")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args(argv)
    if not args.confirm_freeze:
        print("Frozen evaluation refused: requires explicit user freeze authorization and --confirm-freeze.")
        return 2
    if args.model_dir is None or args.output_dir is None:
        ap.error("--model-dir and a new --output-dir are required after authorization")
    if args.output_dir.exists():
        print("Frozen evaluation refused: immutable output directory already exists.")
        return 3
    # Reserve the run before reading test rows/images; failures leave an audit receipt.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"confirmed_freeze": True, "started_unix": time.time(), "status": "started",
               "model_dir": str(args.model_dir.resolve()), "model_card_modified": False}
    receipt_path = args.output_dir / "freeze_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        root = args.data_dir.resolve() if args.data_dir else data_dir()
        identity = fingerprint(args.model_dir)
        predictor = load(args.model_dir)
        rows = load_test_rows(root, confirmed=True)
        if not rows:
            raise ValueError("no frozen target rows")
        results = [predict_row(predictor, root, row, confirmed=True) for row in rows]
        y, pred = np.asarray([r["label"] for r in rows]), np.asarray([r["label"] for r in results])
        P = np.asarray([[r["probs"][c] for c in CLASSES] for r in results])
        P /= P.sum(1, keepdims=True)
        mt = metrics(y, pred)
        flash = agtron_flash(root, confirmed=True)
        report = {"created_unix": time.time(), "run_count": 1, "n": len(rows), "metrics": mt,
                  "accuracy_wilson95": wilson(int((y == pred).sum()), len(rows)), "calibration": calibration(y, P),
                  "by_source": breakdown([r["source"] for r in rows], y, pred),
                  "by_agtron_value": breakdown([r.get("label_orig", "") for r in rows], y, pred),
                  "by_device": breakdown([r.get("device", "unknown") for r in rows], y, pred),
                  "by_flash": breakdown([r.get("flash") or flash.get(Path(r["path"]).name, "unknown") for r in rows], y, pred),
                  "status_counts": dict(Counter(r["status"] for r in results)), "artifact": identity,
                  "manifest_sha256": hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest(),
                  "versions": versions(), "predictor_info": predictor.info(),
                  "runtime_provenance": runtime_provenance(),
                  "protocol": "runtime bytes→dict, mandatory Agtron post-EXIF ROI, no annotation boxes",
                  "elapsed_seconds": time.time() - receipt["started_unix"]}
        with (args.output_dir / "frozen_test.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        with (args.output_dir / "predictions.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "source", "label", "pred", "status", *CLASSES])
            writer.writeheader()
            for row, result in zip(rows, results):
                writer.writerow({"path": row["path"], "source": row["source"], "label": row["label"],
                                 "pred": result["label"], "status": result["status"], **result["probs"]})
        with (args.output_dir / "confusion.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred", *CLASSES])
            writer.writerows([[c, *counts] for c, counts in zip(CLASSES, mt["cm"])])
        receipt.update(status="completed", finished_unix=time.time(), artifact_sha256=identity["artifact_sha256"])
        print(json.dumps({"output_dir": str(args.output_dir), "n": len(rows), "accuracy": mt["acc"], "macro_f1": mt["macro_f1"]}))
        return 0
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        raise
    finally:
        # Only this newly reserved run's receipt is updated; outputs never overwrite another run.
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
