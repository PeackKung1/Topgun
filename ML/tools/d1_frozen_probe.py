"""D1 frozen-backbone probe, corrected protocol (v2), trainval only.

Why v2: train_d1's frozen head was trained full-batch with patience 2, so the
inner search stopped after 2-8 optimiser steps (train loss ~0.8-1.5, chance is
1.10). Its LOSO score measures an unconverged head, not the frozen features.

v2 fits a converged multinomial LogisticRegression (lbfgs) on the cached
MobileNetV3-Small embeddings (mean of full+smart views), scaler fit on the
training fold only, C and temperature chosen on the same group-guarded inner
validation as every other arm. The outer LOSO holdouts and the R2 cap are the
original ones. lbfgs is deterministic, so a single seed is reported and the
other seeds are checked for identical output.

Because the head is linear, logits(mean embedding) == mean of per-view logits,
so the result is exactly what the ONNX runtime (per-view softmax, weighted
log-prob pooling with equal weights) would produce.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from tools.compare_tabular import (
    TEMPERATURES, calibrate_prob, calibration, inner_split, load_inner_group_guard, nll,
)
from tools.train_baseline import CLASSES, FOLDS, SEED, fold_table, video_cap_mask
from tools.train_d1 import ML_DIR, dump, prepare_views, sha256

PLAN = {
    "id": "d1_frozen_probe_v2_20261008",
    "reason": "v1 frozen head unconverged (full-batch AdamW, patience 2, stopped after 2-8 steps)",
    "embeddings": "cached torchvision MobileNetV3-Small DEFAULT avgpool 576-d, mean of full+smart views",
    "model": "StandardScaler(train fold only) + LogisticRegression(lbfgs, class_weight=balanced, max_iter=5000)",
    "C_grid": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    "temperature_grid": TEMPERATURES,
    "selection": "inner group-guarded validation macro-F1, then calibrated NLL, then smaller C",
    "outer_folds": FOLDS, "cap_frames_per_video": 10, "training_cap_seed": SEED,
    "seeds_checked": [20261006, 20261007, 20261008],
    "low_conf_threshold": 0.0, "qa_arm": "QA mask identical to reference (no hard exclusions) -> not refit",
    "frozen_test": "never loaded",
}


def make_pipe(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))])


def probs(pipe, X: np.ndarray) -> np.ndarray:
    order = [list(pipe.classes_).index(c) for c in CLASSES]
    return pipe.predict_proba(X)[:, order]


def select(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    tr, va, split = inner_split(y, groups, seed)
    trials = []
    for C in PLAN["C_grid"]:
        pipe = make_pipe(C).fit(X[tr], y[tr])
        raw = probs(pipe, X[va])
        pred = np.array(CLASSES)[raw.argmax(1)]
        f1 = fold_table(y[va], pred, np.repeat("inner", len(va)), folds=["inner"])["inner"]["macro_f1"]
        losses = {str(t): nll(y[va], calibrate_prob(raw, t)) for t in TEMPERATURES}
        t = min(TEMPERATURES, key=lambda t: (losses[str(t)], abs(t - 1.0)))
        trials.append({"C": C, "validation_macro_f1": f1, "temperature": t, "validation_nll": losses[str(t)],
                       "n_iter": int(np.max(pipe[-1].n_iter_))})
    best = max(range(len(trials)), key=lambda i: (trials[i]["validation_macro_f1"], -trials[i]["validation_nll"], -i))
    return trials[best], {"split": {k: v for k, v in split.items() if not isinstance(v, list)}, "trials": trials, "selected": best}


def run(E, rows, cap, inner_groups, seed):
    y = np.array([r["label"] for r in rows])
    src = np.array([r["source"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    p_all = np.zeros((len(rows), 3))
    folds = {}
    for held in FOLDS:
        te = src == held
        tr = cap & ~te
        if set(grp[tr]) & set(grp[te]):
            raise AssertionError("outer group leakage")
        best, inner = select(E[tr], y[tr], inner_groups[tr], seed)
        pipe = make_pipe(best["C"]).fit(E[tr], y[tr])
        p_all[te] = calibrate_prob(probs(pipe, E[te]), best["temperature"])
        folds[held] = {"training_rows": int(tr.sum()), "training_groups": len(set(grp[tr])),
                       "holdout_rows": int(te.sum()), "holdout_groups": len(set(grp[te])),
                       "C": best["C"], "temperature": best["temperature"], "inner": inner}
    pred = np.array(CLASSES)[p_all.argmax(1)]
    table = fold_table(y, pred, src)
    for held in FOLDS:
        table[held]["calibration"] = calibration(y[src == held], p_all[src == held])
    table["pooled"]["calibration"] = calibration(y, p_all)
    table["mean"]["ece"] = float(np.mean([table[s]["calibration"]["ece"] for s in FOLDS]))
    return p_all, table, folds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ML_DIR / "results" / "d1_frozen_probe_v2_20261008")
    args = ap.parse_args(argv)
    root = args.data_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "plan.json", PLAN)  # written before any fit
    started = time.perf_counter()
    rows, _, info = prepare_views(root, workers=1)  # loads the cached views; rebuilds only if code/config changed
    emb_files = sorted((root / "cache").glob(f"d1_embeddings_{info['key']}_*.npy"))
    if len(emb_files) != 1:
        raise SystemExit(f"expected exactly one cached embedding file for view key {info['key']}: {emb_files}")
    E = np.load(emb_files[0])
    if E.shape[0] != len(rows) or not np.isfinite(E).all():
        raise ValueError("embedding cache does not match the 3395-row reference population")
    cap = video_cap_mask(np.array([r["group"] for r in rows]), np.array([r["path"] for r in rows]), seed=SEED)
    if int(cap.sum()) != 2393 or len(rows) != 3395:
        raise ValueError("reference population or R2 cap changed")
    inner_groups, guard = load_inner_group_guard(rows, root / "cache/qa_audit_20261008/inner_groups.json")
    import sklearn
    results, first = [], None
    for seed in PLAN["seeds_checked"]:
        p, table, folds = run(E, rows, cap, inner_groups, seed)
        if first is None:
            first = (p, table, folds)
        results.append({"seed": seed, "mean_macro_f1": table["mean"]["macro_f1"], "pooled_macro_f1": table["pooled"]["macro_f1"],
                        "identical_to_first_seed": bool(np.allclose(p, first[0], atol=1e-12))})
        print(f"seed {seed} mean macro-F1 {table['mean']['macro_f1']:.4f} pooled {table['pooled']['macro_f1']:.4f}", flush=True)
    p, table, folds = first
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "group", "label", "prediction", *[f"prob_{c}" for c in CLASSES]])
        for r, q in zip(rows, p):
            w.writerow([r["path"], r["source"], r["group"], r["label"], CLASSES[int(q.argmax())], *map(float, q)])
    dump(args.out / "report.json", {
        "plan": PLAN, "metrics": table, "folds": folds, "seed_checks": results,
        "worst_fold": min(FOLDS, key=lambda s: table[s]["macro_f1"]),
        "embedding_file": emb_files[0].name, "embedding_sha256": sha256(emb_files[0]),
        "view_cache_key": info["key"], "manifest_sha256": info["manifest_sha256"], "inner_group_guard": guard,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "tool_sha256": sha256(Path(__file__)), "seconds": time.perf_counter() - started})
    print(json.dumps({"out": str(args.out), "mean_macro_f1": table["mean"]["macro_f1"],
                      "per_fold": {s: table[s]["macro_f1"] for s in FOLDS}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
