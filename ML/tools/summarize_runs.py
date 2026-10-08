"""Aggregate every 2026-10-08 LOSO run from its per-image predictions (trainval only).

All arms must share the identical 3,395-image holdout population; the tool
refuses otherwise. Metrics are recomputed with the same fold_table/calibration
code for every arm, so B1, B2, MLP and D1 numbers are directly comparable.
Outputs: model_comparison.csv, per_fold.csv, confusion_<model>.csv, summary.json
and three PNGs (numbers only, no photos).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.compare_tabular import calibration
from tools.train_baseline import CLASSES, FOLDS, fold_table

ML_DIR = Path(__file__).resolve().parents[1]
R = ML_DIR / "results"
SEEDS = (20261006, 20261007, 20261008)

# name -> (list of prediction files, probability column prefix, description)
def _seeded(pattern: str, seeds=SEEDS):
    return [pattern.format(seed=s) for s in seeds]


MODELS = {
    "B1_R2_reference": (["tabular_reference_reproduction_20261008/predictions_B1_reference_20261006.csv"], "prob_",
                        "B1 Lab_hist LogReg C=0.01, original R2 protocol reproduction (fixed T=1)"),
    "B1": (_seeded("tabular_20261008_groupguard/predictions_B1_reference_{seed}.csv"), "prob_",
           "B1 + group-guarded inner temperature"),
    "B1_qa": (_seeded("tabular_20261008_groupguard/predictions_B1_qa_{seed}.csv"), "prob_",
              "B1 with QA training mask (identical to reference: 0 hard exclusions)"),
    "SVM_RBF": (_seeded("tabular_20261008_groupguard/predictions_SVM_RBF_reference_{seed}.csv"), "prob_",
                "B2 SVM-RBF on same 15 features"),
    "RandomForest": (_seeded("tabular_20261008_groupguard/predictions_RandomForest_reference_{seed}.csv"), "prob_",
                     "B2 RandomForest on same 15 features"),
    "MLP": (["tabular_20261008_groupguard/predictions_MLP_reference_20261006.csv"], "prob_",
            "small MLP on same 15 features (1 seed: not top-3)"),
    "D1_frozen_v1": (_seeded("d1_20261008_groupguard/D1_frozen_reference_{seed}_predictions.csv"), "p_",
                     "MobileNetV3-S frozen + linear head, v1 (head unconverged: 2-8 full-batch steps)"),
    "D1_frozen_v2": (["d1_frozen_probe_v2_20261008/predictions.csv"], "prob_",
                     "MobileNetV3-S frozen + converged LogReg probe (deterministic)"),
    "D1_finetune": (_seeded("d1_20261008_groupguard/D1_finetune_reference_{seed}_predictions.csv"), "p_",
                    "MobileNetV3-S fine-tune, aug none/mild, <=4 epochs"),
    "D1_finetune_medaug": (_seeded("d1_20261008_medaug/D1_finetune_reference_{seed}_predictions.csv"), "p_",
                           "MobileNetV3-S fine-tune, medium aug, <=6 epochs"),
    "B1_sel_drop_coffeetest": (_seeded("tabular_20261008_sel_drop_coffeetest/predictions_B1_qa_{seed}.csv"), "prob_",
                               "B1, training mask drops rf_robusta Coffee-Test video (user decision)"),
    "B1_sel_strict": (_seeded("tabular_20261008_sel_strict_keep_only/predictions_B1_qa_{seed}.csv"), "prob_",
                      "B1, training uses only keep_* categories (no rf_boos / Coffee-Test / robusta full-frame)"),
    "D1_finetune_sel_drop_coffeetest": (_seeded("d1_20261008_sel_drop_coffeetest/D1_finetune_qa_{seed}_predictions.csv"), "p_",
                                        "D1 fine-tune (groupguard protocol), Coffee-Test dropped from training"),
    "D1_finetune_sel_strict": (_seeded("d1_20261008_sel_strict_keep_only/D1_finetune_qa_{seed}_predictions.csv"), "p_",
                               "D1 fine-tune (groupguard protocol), keep_* categories only"),
    "D1_finetune_onnx_fp32":(["d1_export_folds_20261008/D1_finetune_pooled_predictions.csv"], "fp32_prob_",
                              "D1_finetune seed 20261006 exported ONNX FP32"),
    "D1_finetune_onnx_int8": (["d1_export_folds_20261008/D1_finetune_pooled_predictions.csv"], "int8_prob_",
                              "D1_finetune seed 20261006 static INT8 QDQ MinMax"),
}


def read_predictions(path: Path, prefix: str):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["path"])
    P = np.array([[float(r[prefix + c]) for c in CLASSES] for r in rows])
    if not np.isfinite(P).all() or np.abs(P.sum(1) - 1).max() > 1e-6:
        raise ValueError(f"{path}: probabilities invalid")
    return ([r["path"] for r in rows], np.array([r["label"] for r in rows]),
            np.array([r["source"] for r in rows]), P)


def evaluate(y, src, P):
    pred = np.array(CLASSES)[P.argmax(1)]
    table = fold_table(y, pred, src)
    for s in FOLDS:
        table[s]["ece"] = calibration(y[src == s], P[src == s])["ece"]
    table["pooled"]["calibration"] = calibration(y, P)
    return table


def plot(summary: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, blue = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})
    names = [n for n in summary if summary[n]["n_seeds"]]
    order = sorted(names, key=lambda n: summary[n]["mean_macro_f1_mean"])
    # 1) mean LOSO macro-F1 (single series -> one hue, no legend), std as whiskers
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.2))
    vals = [summary[n]["mean_macro_f1_mean"] for n in order]
    errs = [summary[n]["mean_macro_f1_std"] or 0 for n in order]
    ax.barh(order, vals, height=0.6, color=blue, xerr=errs, error_kw={"ecolor": muted, "lw": 1})
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=ink, fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("mean LOSO macro-F1 over 4 source folds (mean ± std across seeds)")
    ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=150); plt.close(fig)
    # 2) pooled confusion matrices, first seed, sequential blue
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    k = len(order); cols = 4; rows_ = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 2.6, rows_ * 2.6))
    for ax in np.ravel(axes): ax.axis("off")
    for ax, n in zip(np.ravel(axes), reversed(order)):
        cm = np.array(summary[n]["pooled_confusion_first_seed"], float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.axis("on"); ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if norm[i, j] > 0.55 else ink)
        ax.set_xticks(range(3), CLASSES, fontsize=7); ax.set_yticks(range(3), CLASSES, fontsize=7)
        ax.set_title(n, fontsize=8, color=ink); ax.set_xlabel("predicted", fontsize=7)
    fig.suptitle("Pooled LOSO confusion (rows = true label, row-normalised)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "confusion_pooled.png", dpi=150); plt.close(fig)
    # 3) reliability: B1 vs the best D1 arm (2 series, legend + direct labels)
    d1 = max([n for n in names if n.startswith("D1")], key=lambda n: summary[n]["mean_macro_f1_mean"])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--")
    for n, color in (("B1", "#2a78d6"), (d1, "#eb6834")):
        rel = [b for b in summary[n]["pooled_reliability_first_seed"] if b["n"] >= 20]
        ax.plot([b["confidence"] for b in rel], [b["accuracy"] for b in rel], color=color, lw=2, marker="o", ms=4,
                label=f"{n} (ECE {summary[n]['pooled_ece_first_seed']:.3f})")
    ax.set_xlabel("mean confidence (bins with n≥20)"); ax.set_ylabel("accuracy"); ax.set_xlim(0.3, 1); ax.set_ylim(0, 1)
    ax.grid(color=grid, lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "reliability_pooled.png", dpi=150); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=R / "summary_20261008")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reference_paths = None
    summary, per_fold = {}, []
    for name, (files, prefix, desc) in MODELS.items():
        present = [R / f for f in files if (R / f).is_file()]
        if not present:
            print(f"skip {name}: no prediction files"); continue
        tables = []
        for f in present:
            paths, y, src, P = read_predictions(f, prefix)
            if reference_paths is None:
                reference_paths = paths
            if paths != reference_paths:
                raise ValueError(f"{f}: holdout population differs from the reference arm")
            t = evaluate(y, src, P)
            tables.append((f, t))
            for s in FOLDS + ["pooled"]:
                m = t[s]
                per_fold.append({"model": name, "file": f.relative_to(R).as_posix(), "fold": s, "n": m["n"],
                                 "acc": m["acc"], "macro_f1": m["macro_f1"], "cross_step_rate": m["cross_step_rate"],
                                 **{f"f1_{c}": m["per_class"][c]["f1"] for c in CLASSES},
                                 **{f"recall_{c}": m["per_class"][c]["recall"] for c in CLASSES},
                                 "ece": m["ece"] if s != "pooled" else m["calibration"]["ece"]})
        mean_f1 = np.array([t["mean"]["macro_f1"] for _, t in tables])
        pooled_f1 = np.array([t["pooled"]["macro_f1"] for _, t in tables])
        first = tables[0][1]
        worst = min(FOLDS, key=lambda s: first[s]["macro_f1"])
        summary[name] = {
            "description": desc, "n_seeds": len(tables), "files": [f.relative_to(R).as_posix() for f, _ in tables],
            "file_sha256": [hashlib.sha256(f.read_bytes()).hexdigest() for f, _ in tables],
            "mean_macro_f1_mean": float(mean_f1.mean()),
            "mean_macro_f1_std": float(mean_f1.std(ddof=1)) if len(tables) > 1 else None,
            "pooled_macro_f1_mean": float(pooled_f1.mean()),
            "pooled_macro_f1_std": float(pooled_f1.std(ddof=1)) if len(tables) > 1 else None,
            "mean_acc_mean": float(np.mean([t["mean"]["acc"] for _, t in tables])),
            "pooled_acc_mean": float(np.mean([t["pooled"]["acc"] for _, t in tables])),
            "pooled_cross_step_rate_mean": float(np.mean([t["pooled"]["cross_step_rate"] for _, t in tables])),
            "pooled_ece_mean": float(np.mean([t["pooled"]["calibration"]["ece"] for _, t in tables])),
            "per_fold_macro_f1_first_seed": {s: first[s]["macro_f1"] for s in FOLDS},
            "worst_fold_first_seed": worst, "worst_fold_macro_f1_first_seed": first[worst]["macro_f1"],
            "pooled_confusion_first_seed": first["pooled"]["cm"],
            "pooled_reliability_first_seed": first["pooled"]["calibration"]["reliability"],
            "pooled_ece_first_seed": first["pooled"]["calibration"]["ece"],
        }
        with (args.out / f"confusion_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["true\\pred", *CLASSES])
            for c, row in zip(CLASSES, first["pooled"]["cm"]):
                w.writerow([c, *row])
    cols = ["model", "n_seeds", "mean_macro_f1_mean", "mean_macro_f1_std", "pooled_macro_f1_mean", "pooled_macro_f1_std",
            "mean_acc_mean", "pooled_acc_mean", "pooled_cross_step_rate_mean", "pooled_ece_mean",
            "worst_fold_first_seed", "worst_fold_macro_f1_first_seed", "description"]
    with (args.out / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for n, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean_macro_f1_mean"]):
            w.writerow({"model": n, **{k: s[k] for k in cols[1:]}})
    with (args.out / "per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0])); w.writeheader(); w.writerows(per_fold)
    (args.out / "summary.json").write_text(json.dumps({"n_holdout": len(reference_paths), "models": summary,
                                                        "frozen_test_loaded": False}, indent=2), encoding="utf-8")
    plot(summary, args.out)
    for n, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean_macro_f1_mean"]):
        std = f"±{s['mean_macro_f1_std']:.3f}" if s["mean_macro_f1_std"] is not None else ""
        print(f"{n:24s} seeds={s['n_seeds']} meanF1={s['mean_macro_f1_mean']:.4f}{std} pooledF1={s['pooled_macro_f1_mean']:.4f} "
              f"worst={s['worst_fold_first_seed']}:{s['worst_fold_macro_f1_first_seed']:.3f} ECE={s['pooled_ece_mean']:.3f} "
              f"xstep={s['pooled_cross_step_rate_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
