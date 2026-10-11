"""B1 (Lab_hist logistic regression) regularisation sweep: is C = 0.01 still the right value below the old grid?

    python -m tools.tune_b1_c --out results/b1_c_sweep_<date> --inner-groups <json> --write-plan
    python -m tools.tune_b1_c --out results/b1_c_sweep_<date> --inner-groups <json>

The 8 Oct baseline compared C in {0.01, 0.1, 1, 10} and kept the smallest; nothing below 0.01 was tried.
Two analyses on the reference feature cache and the exact R2 training cap, trainval only:
  sweep  : one fixed C for all four outer folds, scored on the LOSO holdouts (how the original C was chosen;
           picking the best C from this table is optimistic)
  nested : C chosen per outer fold on the group-guarded inner validation only (tools.compare_tabular.run_loso),
           so the holdout plays no part in the choice
Fitting, probabilities, inner selection and metrics are the existing helpers; only the C grid is new.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from roastml.paths import data_dir
from tools import compare_tabular
from tools.compare_tabular import dump, estimator_probs, fit_estimator, load_cached, load_inner_group_guard, run_loso, sha256, write_csv
from tools.summarize_runs import evaluate, read_predictions
from tools.train_baseline import CLASSES, FOLDS, SEED

ML_DIR = Path(__file__).resolve().parents[1]
SEEDS = [20261006, 20261007, 20261008]
GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
EXTENSION = [0.00003, 0.00001]
B1_C, B1_MEAN_MACRO_F1 = 0.01, 0.6381973706626417
KNOWN = {0.01: 0.6381973706626417, 0.1: 0.5168305830694876}  # results/baseline_loso.json, run R2, pipeline view


def build_plan(inner_groups: Path, manifest: Path) -> dict:
    return {
        "protocol": "b1_c_sweep_v1",
        "question": "does a C below the old grid minimum (0.01) improve B1's mean LOSO macro-F1?",
        "motivation": "results/baseline_loso.json (R2, pipeline view): C 0.01 -> 0.638, 0.1 -> 0.517, 1 -> 0.497, 10 -> 0.469; "
                      "the score rises monotonically as C shrinks and the grid stopped at 0.01",
        "model": "tools.train_baseline.make_pipe(C): StandardScaler + LogisticRegression(C, max_iter=5000, class_weight='balanced') on Lab_hist",
        "grid": GRID,
        "edge_rule": {"if": "the best sweep C is the smallest grid value", "then_add_once": EXTENSION,
                      "note": "declared here so an extension is not a post-hoc choice; no further extension"},
        "data": "reference feature cache (results/baseline_loso.json), exact R2 cap video_cap_mask(seed=20261006), 3395 trainval images, 4 outer folds",
        "sweep": "fixed C for all folds, temperature 1, scored on the LOSO holdouts; every C is reported",
        "nested": {"selection": "per outer fold on the inner validation: macro-F1, then calibrated NLL, then grid order (ascending C, so the smaller C wins ties)",
                   "inner_split": "tools.compare_tabular.inner_split with the group guard; seeds 20261006/20261007/20261008 change only the inner split",
                   "temperature": "chosen on the same inner validation from [0.5, 0.75, 1, 1.5, 2, 3, 5]"},
        "gate": {"C=0.01": KNOWN[0.01], "C=0.1": KNOWN[0.1], "tolerance": 1e-12, "on_failure": "stop before reporting anything"},
        "inner_groups": {"file": inner_groups.name, "directory": inner_groups.parent.name, "sha256": sha256(inner_groups)},
        "manifest_sha256": sha256(manifest),
        "metric": "tools.summarize_runs.evaluate (fold_table + calibration); primary = mean LOSO macro-F1",
        "decision_rule": {"primary": "nested: mean over 3 seeds of mean LOSO macro-F1 > B1 (0.6381973706626417)",
                          "secondary": "sweep under the original deploy rule: mean LOSO macro-F1 rounded to 2 decimals, then the smaller C",
                          "reading": "a sweep-only gain is selection on the holdouts (as the original C choice was) and needs the nested result or new data to back it"},
        "limits": "LOSO source-level convenience dataset; no Pi/E2E/frozen-test claim; inner groups and manifest are the regenerated copies "
                  "that reproduce the tracked B1 numbers exactly (results/ensemble_b1_keras_20261011)",
    }


def sweep_one(C: float, Xt, Xp, y, src, cap) -> tuple[np.ndarray, list[str]]:
    probs, warnings = np.zeros((len(y), len(CLASSES))), []
    for held in FOLDS:
        te = src == held
        pipe, fit = fit_estimator("B1", {"C": C}, SEED, Xt[~te & cap], y[~te & cap])
        probs[te] = estimator_probs(pipe, "B1", Xp[te])
        warnings += fit["warnings"]
    return probs, sorted(set(warnings))


def sweep_row(C: float, table: dict, warnings: list[str]) -> dict:
    return {"C": C, "mean_macro_f1": table["mean"]["macro_f1"], "mean_macro_f1_round2": round(table["mean"]["macro_f1"], 2),
            "pooled_macro_f1": table["pooled"]["macro_f1"], "mean_acc": table["mean"]["acc"], "pooled_acc": table["pooled"]["acc"],
            "pooled_cross_step_rate": table["pooled"]["cross_step_rate"], "pooled_ece_T1": table["pooled"]["calibration"]["ece"],
            **{f"macro_f1_{s}": table[s]["macro_f1"] for s in FOLDS},
            "worst_fold": min(FOLDS, key=lambda s: table[s]["macro_f1"]), "convergence_warnings": len(warnings)}


def run(root: Path, out: Path, inner_groups: Path, plan: dict) -> dict:
    reference = json.loads((ML_DIR / "results" / "baseline_loso.json").read_text(encoding="utf-8"))
    rows, Xt, Xp, cap, cache_path, errors = load_cached(root, reference)
    validation_groups, guard = load_inner_group_guard(rows, inner_groups)
    y = np.array([r["label"] for r in rows]); src = np.array([r["source"] for r in rows])

    grid, sweep = list(plan["grid"]), {}
    for C in grid:
        P, warnings = sweep_one(C, Xt, Xp, y, src, cap)
        sweep[C] = sweep_row(C, evaluate(y, src, P), warnings)
    for C, expected in KNOWN.items():
        if abs(sweep[C]["mean_macro_f1"] - expected) > plan["gate"]["tolerance"]:
            raise SystemExit(f"gate failed: C={C} gives {sweep[C]['mean_macro_f1']}, tracked {expected}; stop")
    best = max(grid, key=lambda c: (sweep[c]["mean_macro_f1"], -c))
    extended = best == min(grid)
    if extended:
        for C in plan["edge_rule"]["then_add_once"]:
            P, warnings = sweep_one(C, Xt, Xp, y, src, cap)
            sweep[C] = sweep_row(C, evaluate(y, src, P), warnings)
        grid = sorted(sweep)
    ordered = [sweep[c] for c in sorted(sweep)]
    write_csv(out / "sweep.csv", ordered)
    best = max(sweep, key=lambda c: (sweep[c]["mean_macro_f1"], -c))
    top = max(r["mean_macro_f1_round2"] for r in ordered)
    by_original_rule = min(c for c in sweep if sweep[c]["mean_macro_f1_round2"] == top)

    # nested: the existing run_loso with only the B1 grid replaced (ascending C = grid order for ties)
    compare_tabular.GRIDS["B1"] = [{"C": c} for c in sorted(sweep)]
    nested = []
    for seed in SEEDS:
        report = run_loso("B1", seed, "cnested", cap, rows, Xt, Xp, out, validation_groups=validation_groups)
        table = evaluate(*read_predictions(out / f"predictions_B1_cnested_{seed}.csv", "prob_")[1:])
        nested.append({"seed": seed, "mean_macro_f1": table["mean"]["macro_f1"], "pooled_macro_f1": table["pooled"]["macro_f1"],
                       "mean_acc": table["mean"]["acc"], "pooled_cross_step_rate": table["pooled"]["cross_step_rate"],
                       "pooled_ece": table["pooled"]["calibration"]["ece"],
                       **{f"macro_f1_{s}": table[s]["macro_f1"] for s in FOLDS},
                       **{f"C_{s}": report["folds"][s]["config"]["C"] for s in FOLDS},
                       **{f"T_{s}": report["folds"][s]["temperature"] for s in FOLDS},
                       "inner_macro_f1_by_C": {s: {str(t["config"]["C"]): t["validation_macro_f1"] for t in report["folds"][s]["inner"]["trials"]}
                                               for s in FOLDS}})
    write_csv(out / "nested.csv", [{k: v for k, v in r.items() if k != "inner_macro_f1_by_C"} for r in nested])
    scores = np.array([r["mean_macro_f1"] for r in nested])
    import sklearn
    report = {"plan": plan, "grid_evaluated": sorted(sweep), "grid_extended_by_edge_rule": extended,
              "gate": {str(c): sweep[c]["mean_macro_f1"] for c in KNOWN}, "sweep": ordered,
              "sweep_best": {"C": best, "mean_macro_f1": sweep[best]["mean_macro_f1"], "gain_over_B1": sweep[best]["mean_macro_f1"] - B1_MEAN_MACRO_F1,
                             "selected_by_original_deploy_rule": by_original_rule,
                             "original_rule_mean_macro_f1": sweep[by_original_rule]["mean_macro_f1"]},
              "nested": {"seeds": nested, "mean_macro_f1_mean": float(scores.mean()), "mean_macro_f1_std": float(scores.std(ddof=1)),
                         "beats_B1": bool(scores.mean() > B1_MEAN_MACRO_F1)},
              "decision": {"B1_mean_macro_f1": B1_MEAN_MACRO_F1, "B1_C": B1_C,
                           "primary_nested_beats_B1": bool(scores.mean() > B1_MEAN_MACRO_F1),
                           "secondary_sweep_rule_selects": by_original_rule},
              "provenance": {"feature_cache": cache_path.name, "feature_cache_sha256": sha256(cache_path), "feature_errors": len(errors),
                             "r2_train_rows": int(cap.sum()), "inner_group_guard": {k: guard[k] for k in ("sha256", "original_groups", "guarded_groups")},
                             "tool_sha256": sha256(Path(__file__)),
                             "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}},
              "frozen_test_loaded": False}
    dump(out / "report.json", report)
    for r in ordered:
        print(f"sweep C={r['C']:<8g} meanF1={r['mean_macro_f1']:.4f} pooledF1={r['pooled_macro_f1']:.4f} "
              + " ".join(f"{s}={r['macro_f1_' + s]:.3f}" for s in FOLDS) + f" xstep={r['pooled_cross_step_rate']:.3f}", flush=True)
    for r in nested:
        print(f"nested seed={r['seed']} meanF1={r['mean_macro_f1']:.4f} C per fold=" + str({s: r['C_' + s] for s in FOLDS}), flush=True)
    print(f"nested mean {scores.mean():.4f} ± {scores.std(ddof=1):.4f} | B1 {B1_MEAN_MACRO_F1:.4f} | sweep best C={best} "
          f"{sweep[best]['mean_macro_f1']:.4f} | original rule selects C={by_original_rule}", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inner-groups", type=Path, required=True)
    ap.add_argument("--write-plan", action="store_true", help="write plan.json before any fit, then exit")
    args = ap.parse_args(argv)
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    out = args.out.resolve()
    plan = json.loads(json.dumps(build_plan(args.inner_groups, root / "manifest.csv")))
    if args.write_plan:
        if (out / "plan.json").exists():
            raise SystemExit("plan.json already exists; a plan is written once")
        out.mkdir(parents=True, exist_ok=True)
        dump(out / "plan.json", plan); print(f"plan written: {out / 'plan.json'}"); return 0
    if not (out / "plan.json").exists():
        raise SystemExit("plan.json missing: run --write-plan first")
    if json.loads((out / "plan.json").read_text(encoding="utf-8")) != plan:
        raise RuntimeError("plan.json differs from the plan this code and these inputs produce; use a new output directory")
    run(root, out, args.inner_groups, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
