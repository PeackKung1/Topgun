"""Fixed-rule probability ensemble of B1 and image-level CNN arms on the LOSO holdouts (trainval only).

    python -m tools.ensemble_b1_cnn --out results/ensemble_b1_keras_<date> --cnn-run results/keras_cnn_<date> --inner-groups <json> --write-plan
    python -m tools.ensemble_b1_cnn --out results/ensemble_b1_keras_<date> --cnn-run results/keras_cnn_<date> --inner-groups <json>

The rule has no free parameter: equal-weight mean of the two members' class probabilities, argmax.
Nothing is fitted or chosen on any holdout image. B1 is refitted here with the existing
tools.compare_tabular code (reference feature cache + group-guarded inner temperature) and must
reproduce the tracked reference score before any ensemble is formed.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np

from roastml.paths import data_dir
from roastml.segment import SegConfig
from tools.compare_tabular import dump, load_cached, load_inner_group_guard, run_loso, sha256, write_csv
from tools.summarize_runs import evaluate, read_predictions
from tools.train_baseline import CLASSES, FOLDS, SEED, attach_boxes, extract_features, load_rows
from tools.train_keras_cnn import B1_MEAN_MACRO_F1, REFERENCES, SEEDS, references

ML_DIR = Path(__file__).resolve().parents[1]
CNN_ARMS = ("K1_control_frozen", "K1_control_finetune", "K2_new_finetune")
COLUMNS = ["model", "n_seeds", "mean_macro_f1_mean", "mean_macro_f1_std", "pooled_macro_f1_mean", "pooled_macro_f1_std",
           "mean_acc_mean", "pooled_acc_mean", "pooled_cross_step_rate_mean", "pooled_ece_mean",
           "worst_fold_first_seed", "worst_fold_macro_f1_first_seed", "beats_B1", "source"]


def build_plan(cnn_run: Path, inner_groups: Path, manifest: Path) -> dict:
    return {
        "protocol": "ensemble_b1_cnn_v1",
        "rule": "p = 0.5 * p_B1 + 0.5 * p_CNN per image, prediction = argmax; no weight, threshold or temperature is fitted",
        "members": {"B1": "tools.compare_tabular.run_loso('B1', 20261006, 'reference', R2 cap, group-guarded inner temperature); "
                          "deterministic, so the same B1 probabilities are paired with every CNN seed",
                    "CNN": {"run": cnn_run.name, "arms": list(CNN_ARMS), "seeds": SEEDS,
                            "plan_sha256": sha256(cnn_run / "plan.json")}},
        "ensembles": {f"E_B1+{a}": ["B1", a] for a in CNN_ARMS},
        "all_reported": "every ensemble listed here is reported; none is dropped or selected",
        "gate_before_ensembling": "B1 fixed-protocol refit must reproduce mean LOSO macro-F1 0.6381973706626417 within 1e-12; otherwise stop",
        "population": "the 3395 trainval images of tools.train_baseline.load_rows; split=test never read",
        "inner_groups": {"file": inner_groups.name, "directory": inner_groups.parent.name, "sha256": sha256(inner_groups)},
        "manifest_sha256": sha256(manifest),
        "metric": "tools.summarize_runs.evaluate (fold_table + calibration); primary = mean LOSO macro-F1, mean +- std(ddof=1) over the 3 CNN seeds",
        "decision_rule": {"wins_if": "mean LOSO macro-F1 (mean over 3 seeds) > B1", "B1_mean_macro_f1": B1_MEAN_MACRO_F1,
                          "otherwise": "B1 stays the deployed model"},
        "post_hoc_disclosure": "this idea was proposed after the holdout results of both members were seen (CNN ahead of B1 on rf_robusta "
                               "only). The rule itself has no tunable part, but the decision to try it used holdout information, so a win "
                               "here would be a hypothesis to confirm on new data, not a deploy decision by itself.",
        "diagnostics": "per fold: share of images where exactly one member is right, and the 'either member right' upper bound; descriptive only",
        "limits": "LOSO source-level convenience dataset; no Pi/E2E/frozen-test claim; inner groups and manifest are the regenerated copies "
                  "documented in the CNN run plan",
    }


def aggregate(name: str, tables: list[dict], source: str) -> tuple[dict, list[dict]]:
    mean_f1 = np.array([t["mean"]["macro_f1"] for t in tables]); pooled_f1 = np.array([t["pooled"]["macro_f1"] for t in tables])
    std = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else None  # noqa: E731
    first = tables[0]; worst = min(FOLDS, key=lambda s: first[s]["macro_f1"])
    row = {"model": name, "n_seeds": len(tables), "mean_macro_f1_mean": float(mean_f1.mean()), "mean_macro_f1_std": std(mean_f1),
           "pooled_macro_f1_mean": float(pooled_f1.mean()), "pooled_macro_f1_std": std(pooled_f1),
           "mean_acc_mean": float(np.mean([t["mean"]["acc"] for t in tables])),
           "pooled_acc_mean": float(np.mean([t["pooled"]["acc"] for t in tables])),
           "pooled_cross_step_rate_mean": float(np.mean([t["pooled"]["cross_step_rate"] for t in tables])),
           "pooled_ece_mean": float(np.mean([t["pooled"]["calibration"]["ece"] for t in tables])),
           "worst_fold_first_seed": worst, "worst_fold_macro_f1_first_seed": first[worst]["macro_f1"],
           "beats_B1": bool(mean_f1.mean() > B1_MEAN_MACRO_F1), "source": source}
    folds = []
    for s in FOLDS + ["pooled"]:
        f1 = np.array([t[s]["macro_f1"] for t in tables])
        folds.append({"model": name, "fold": s, "n": first[s]["n"], "n_seeds": len(tables), "macro_f1_mean": float(f1.mean()),
                      "macro_f1_std": std(f1), "acc_mean": float(np.mean([t[s]["acc"] for t in tables])),
                      "cross_step_rate_mean": float(np.mean([t[s]["cross_step_rate"] for t in tables])),
                      "ece_mean": float(np.mean([t[s]["ece"] if s != "pooled" else t[s]["calibration"]["ece"] for t in tables])),
                      "source": source})
    return row, folds


def refit_b1(root: Path, out: Path, inner_groups: Path, workers: int) -> tuple[Path, dict]:
    """B1 LOSO probabilities with the existing tabular code; stops unless the tracked reference score is reproduced."""
    reference = json.loads((ML_DIR / "results" / "baseline_loso.json").read_text(encoding="utf-8"))
    rows = load_rows(root); attach_boxes(root, rows)
    _, cache_name = extract_features(root, rows, SegConfig(), workers)
    if cache_name != reference["feature_cache"]:
        raise SystemExit(f"feature cache tag {cache_name} differs from the reference {reference['feature_cache']}; stop")
    rows, Xt, Xp, cap, cache_path, errors = load_cached(root, reference)
    validation_groups, guard = load_inner_group_guard(rows, inner_groups)
    fixed = run_loso("B1", SEED, "fixed_reference", cap, rows, Xt, Xp, out, fixed=True, validation_groups=validation_groups)
    reproduced = fixed["metrics"]["mean"]["macro_f1"]
    if abs(reproduced - reference["selected"]["mean_macro_f1_pipeline"]) > 1e-12:
        raise SystemExit(f"B1 reference reproduction failed: {reproduced} vs {reference['selected']['mean_macro_f1_pipeline']}; stop")
    b1 = run_loso("B1", SEED, "reference", cap, rows, Xt, Xp, out, validation_groups=validation_groups)
    with (ML_DIR / "results" / "summary_20261008" / "per_fold.csv").open(newline="", encoding="utf-8") as f:
        tracked = {r["fold"]: r for r in csv.DictReader(f) if r["model"] == "B1" and r["file"].endswith(f"_{SEED}.csv")}
    check = {s: {"macro_f1": b1["metrics"][s]["macro_f1"], "macro_f1_tracked": float(tracked[s]["macro_f1"]),
                 "ece": b1["metrics"][s]["calibration"]["ece"], "ece_tracked": float(tracked[s]["ece"]),
                 "temperature": b1["folds"][s]["temperature"]} for s in FOLDS}
    info = {"fixed_reference_mean_macro_f1": reproduced, "reference_mean_macro_f1": reference["selected"]["mean_macro_f1_pipeline"],
            "feature_cache": cache_path.name, "feature_cache_sha256": sha256(cache_path), "feature_errors": len(errors),
            "r2_train_rows": int(cap.sum()), "inner_group_guard": {k: guard[k] for k in ("sha256", "original_groups", "guarded_groups")},
            "per_fold_vs_tracked_B1": check,
            "max_abs_macro_f1_diff": max(abs(v["macro_f1"] - v["macro_f1_tracked"]) for v in check.values()),
            "max_abs_ece_diff": max(abs(v["ece"] - v["ece_tracked"]) for v in check.values())}
    return out / f"predictions_B1_reference_{SEED}.csv", info


def run(root: Path, out: Path, cnn_run: Path, inner_groups: Path, plan: dict, workers: int) -> dict:
    b1_file, b1_info = refit_b1(root, out, inner_groups, workers)
    paths, labels, source, Pb = read_predictions(b1_file, "prob_")
    b1_table = evaluate(labels, source, Pb)
    right_b = Pb.argmax(1) == np.array([CLASSES.index(c) for c in labels])
    compare, per_fold, diagnostics = [], [], {}
    for name, (_, arm) in plan["ensembles"].items():
        tables, member_tables, diag = [], [], []
        for seed in SEEDS:
            p, y, src, Pc = read_predictions(cnn_run / f"{arm}_{seed}_predictions.csv", "p_")
            if p != paths or (y != labels).any():
                raise ValueError(f"{arm} {seed}: holdout population differs from B1")
            P = 0.5 * Pb + 0.5 * Pc
            write_csv(out / f"{name}_{seed}_predictions.csv",
                      [{"path": paths[i], "source": source[i], "label": labels[i], "prediction": CLASSES[int(P[i].argmax())],
                        **{f"p_{c}": float(P[i, k]) for k, c in enumerate(CLASSES)}} for i in range(len(paths))])
            tables.append(evaluate(labels, source, P)); member_tables.append(evaluate(labels, source, Pc))
            right_c = Pc.argmax(1) == np.array([CLASSES.index(c) for c in labels])
            diag.append({s: {"only_B1_right": float((right_b & ~right_c)[m].mean()), "only_CNN_right": float((~right_b & right_c)[m].mean()),
                             "both_right": float((right_b & right_c)[m].mean()), "either_right_upper_bound_acc": float((right_b | right_c)[m].mean())}
                         for s, m in [(f, source == f) for f in FOLDS] + [("pooled", np.ones(len(paths), bool))]})
        for label, tabs, src_name in ((name, tables, f"{out.name} (this run)"), (arm, member_tables, f"{cnn_run.name} (member, recomputed)")):
            row, folds = aggregate(label, tabs, src_name); compare.append(row); per_fold += folds
        diagnostics[name] = {s: {k: float(np.mean([d[s][k] for d in diag])) for k in diag[0][s]} for s in diag[0]}
    row, folds = aggregate("B1 (refit here)", [b1_table], f"{out.name} (member, refit)"); row["beats_B1"] = None
    compare.append(row); per_fold += folds
    ref_overall, ref_folds = references()
    for ref in REFERENCES:
        r = ref_overall[ref]
        compare.append({**{k: (r[k] if k in ("model", "worst_fold_first_seed") else (float(r[k]) if r[k] != "" else None)) for k in COLUMNS[:12]},
                        "n_seeds": int(r["n_seeds"]), "beats_B1": None if ref == "B1" else False,
                        "source": "results/summary_20261008/model_comparison.csv"})
        for s in FOLDS + ["pooled"]:
            sub = [x for x in ref_folds if x["model"] == ref and x["fold"] == s]
            f1 = np.array([float(x["macro_f1"]) for x in sub])
            per_fold.append({"model": ref, "fold": s, "n": int(sub[0]["n"]), "n_seeds": len(sub), "macro_f1_mean": float(f1.mean()),
                             "macro_f1_std": float(f1.std(ddof=1)) if len(sub) > 1 else None,
                             "acc_mean": float(np.mean([float(x["acc"]) for x in sub])),
                             "cross_step_rate_mean": float(np.mean([float(x["cross_step_rate"]) for x in sub])),
                             "ece_mean": float(np.mean([float(x["ece"]) for x in sub])), "source": "results/summary_20261008/per_fold.csv"})
    write_csv(out / "compare.csv", [{k: r.get(k) for k in COLUMNS} for r in compare])
    write_csv(out / "compare_per_fold.csv", per_fold)
    winners = [r["model"] for r in compare if r["model"] in plan["ensembles"] and r["beats_B1"]]
    import sklearn
    report = {"plan": plan, "b1_refit": b1_info, "ensembles": {r["model"]: r for r in compare if r["model"] in plan["ensembles"]},
              "per_seed_mean_macro_f1": {}, "complementarity": diagnostics,
              "decision": {"B1_mean_macro_f1": B1_MEAN_MACRO_F1, "ensembles_beating_B1": winners,
                           "result": "an ensemble scores above B1 — post-hoc, confirm on new data before any deploy decision" if winners
                           else "no ensemble beats B1; B1 stays the deployed model"},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "tool_sha256": sha256(Path(__file__)), "frozen_test_loaded": False}
    for name in plan["ensembles"]:
        report["per_seed_mean_macro_f1"][name] = [
            evaluate(*read_predictions(out / f"{name}_{seed}_predictions.csv", "p_")[1:])["mean"]["macro_f1"] for seed in SEEDS]
    dump(out / "report.json", report)
    for r in compare:
        std = f"±{r['mean_macro_f1_std']:.4f}" if r["mean_macro_f1_std"] is not None else ""
        print(f"{r['model']:30s} seeds={r['n_seeds']} meanF1={r['mean_macro_f1_mean']:.4f}{std} pooledF1={r['pooled_macro_f1_mean']:.4f} "
              f"worst={r['worst_fold_first_seed']}:{r['worst_fold_macro_f1_first_seed']:.3f} ECE={r['pooled_ece_mean']:.3f} "
              f"xstep={r['pooled_cross_step_rate_mean']:.3f}", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cnn-run", type=Path, required=True)
    ap.add_argument("--inner-groups", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write-plan", action="store_true", help="write plan.json before any ensemble is computed, then exit")
    args = ap.parse_args(argv)
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    out, cnn_run = args.out.resolve(), args.cnn_run.resolve()
    plan = json.loads(json.dumps(build_plan(cnn_run, args.inner_groups, root / "manifest.csv")))
    if args.write_plan:
        if (out / "plan.json").exists():
            raise SystemExit("plan.json already exists; a plan is written once")
        out.mkdir(parents=True, exist_ok=True)
        dump(out / "plan.json", plan); print(f"plan written: {out / 'plan.json'}"); return 0
    if not (out / "plan.json").exists():
        raise SystemExit("plan.json missing: run --write-plan first")
    if json.loads((out / "plan.json").read_text(encoding="utf-8")) != plan:
        raise RuntimeError("plan.json differs from the plan this code and these inputs produce; use a new output directory")
    run(root, out, cnn_run, args.inner_groups, plan, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
