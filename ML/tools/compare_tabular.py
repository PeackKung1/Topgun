"""Controlled trainval-only tabular LOSO comparison, without reading image bytes.

The reference cache and the exact original R2 cap are mandatory. Hyperparameters
and temperature are selected only on a group-disjoint inner validation fold;
outer holdouts retain every reference image. sklearn is used only for training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from roastml.features import FEATURES_ALL, FEATURE_SETS, select
from roastml.linear_model import spec_from_sklearn
from roastml.paths import data_dir
from tools.train_baseline import (
    CLASSES, FOLDS, SEED, attach_boxes, fold_table, load_rows, make_pipe,
    row_key, video_cap_mask,
)

ML_DIR = Path(__file__).resolve().parents[1]
SEEDS = [20261006, 20261007, 20261008]
TEMPERATURES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
# This grid is written before any fit. A fixed epoch budget avoids sklearn's
# image-level internal early-stopping validation split.
GRIDS = {
    "B1": [{"C": 0.01}],
    "SVM_RBF": [{"C": c, "gamma": g} for c in (0.1, 1.0, 10.0) for g in ("scale", 0.01)],
    "RandomForest": [{"n_estimators": 64, "max_depth": d, "min_samples_leaf": m}
                     for d in (8, None) for m in (1, 4)],
    "MLP": [{"hidden": h, "alpha": a, "epochs": 200}
            for h in (16, 32) for a in (0.001, 0.1)],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def calibrate_prob(p: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.clip(p, 1e-15, 1.0)) / temperature)


def nll(y: np.ndarray, p: np.ndarray) -> float:
    idx = np.array([CLASSES.index(c) for c in y])
    return float(-np.log(np.clip(p[np.arange(len(y)), idx], 1e-15, 1.0)).mean())


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    truth = np.array([CLASSES.index(c) for c in y])
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == truth
    bin_id = np.minimum((conf * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for k in range(bins):
        m = bin_id == k
        count = int(m.sum())
        acc = float(correct[m].mean()) if count else None
        confidence = float(conf[m].mean()) if count else None
        if count:
            ece += count / len(y) * abs(acc - confidence)
        reliability.append({"bin": k, "lower": k / bins, "upper": (k + 1) / bins,
                            "n": count, "accuracy": acc, "confidence": confidence})
    onehot = np.eye(3)[truth]
    return {"ece": float(ece), "nll": nll(y, p),
            "brier": float(((p - onehot) ** 2).sum(axis=1).mean()), "reliability": reliability}


def inner_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """No image split fallback: search at most the five declared group folds."""
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("inner validation unavailable: fewer than two independent groups")
    k = min(5, len(unique))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    invalid = []
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        missing_train = sorted(set(CLASSES) - set(y[tr]))
        missing_val = sorted(set(CLASSES) - set(y[va]))
        if missing_train or missing_val:
            invalid.append({"fold": fold, "missing_train": missing_train, "missing_validation": missing_val})
            continue
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("group leakage")
        return tr, va, {"n_splits": k, "selected_fold": fold, "skipped_folds": invalid,
                        "n_train": len(tr), "n_validation": len(va),
                        "train_groups": sorted(set(groups[tr])), "validation_groups": sorted(set(groups[va]))}
    raise ValueError(f"inner validation unavailable: every group fold misses a class: {invalid}")


def make_estimator(family: str, cfg: dict, seed: int):
    if family == "B1":
        return make_pipe(cfg["C"])
    if family == "SVM_RBF":
        model = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", class_weight="balanced",
                    probability=False, decision_function_shape="ovr", random_state=seed)
    elif family == "RandomForest":
        model = RandomForestClassifier(**cfg, class_weight="balanced", random_state=seed, n_jobs=1)
    elif family == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(cfg["hidden"],), alpha=cfg["alpha"],
                              max_iter=cfg["epochs"], early_stopping=False, random_state=seed,
                              solver="adam", learning_rate_init=0.001, batch_size=64, tol=0.0,
                              n_iter_no_change=cfg["epochs"] + 1, activation="relu")
    else:
        raise ValueError(family)
    return Pipeline([("sc", StandardScaler()), ("clf", model)])


def fit_estimator(family: str, cfg: dict, seed: int, X: np.ndarray, y: np.ndarray):
    pipe = make_estimator(family, cfg, seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        kwargs = {"clf__sample_weight": compute_sample_weight("balanced", y)} if family == "MLP" else {}
        pipe.fit(X, y, **kwargs)
    return pipe, {"seconds": time.perf_counter() - t0,
                  "warnings": sorted(set(str(w.message) for w in seen))}


def estimator_probs(pipe, family: str, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    order = [list(pipe.classes_).index(c) for c in CLASSES]
    if family == "SVM_RBF":
        return softmax(pipe.decision_function(X)[:, order] / temperature)
    p = pipe.predict_proba(X)[:, order]
    return calibrate_prob(p, temperature) if temperature != 1.0 else p


def select_inner(family: str, X: np.ndarray, Xp: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    tr, va, split = inner_split(y, groups, seed)
    trials = []
    # Fixed B1 C reproduces the original classifier; nested B1 selects only T.
    for cfg in GRIDS[family]:
        pipe, fit = fit_estimator(family, cfg, seed, X[tr], y[tr])
        raw = estimator_probs(pipe, family, Xp[va])
        pred = np.array(CLASSES)[raw.argmax(axis=1)]
        score = fold_table(y[va], pred, np.repeat("inner", len(va)), folds=["inner"])["inner"]["macro_f1"]
        temp_losses = {str(t): nll(y[va], calibrate_prob(raw, t)) for t in TEMPERATURES}
        temp = min(TEMPERATURES, key=lambda t: (temp_losses[str(t)], abs(t - 1.0)))
        trials.append({"config": cfg, "validation_macro_f1": score, "temperature": temp,
                       "validation_nll": temp_losses[str(temp)], "temperature_nll_grid": temp_losses, "fit": fit})
    # Predeclared tie break: macro-F1, then calibrated NLL, then grid order.
    best = max(range(len(trials)), key=lambda i: (trials[i]["validation_macro_f1"], -trials[i]["validation_nll"], -i))
    return trials[best]["config"], trials[best]["temperature"], {"split": split, "trials": trials, "selected": best}


def load_cached(root: Path, reference: dict):
    rows = load_rows(root)  # filters split=trainval before touching annotations/cache
    attach_boxes(root, rows)
    cache_path = root / "cache" / reference["feature_cache"]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [r["path"] for r in rows if row_key(r) not in cache]
    if missing:
        raise ValueError(f"reference cache missing {len(missing)} rows; no implicit extraction: {missing[:3]}")
    errors = [{"path": r["path"], "error": cache[row_key(r)]["err"]} for r in rows if cache[row_key(r)]["err"]]
    rows = [r for r in rows if not cache[row_key(r)]["err"]]
    if len(rows) != reference["n_images"]:
        raise ValueError("reference holdout population changed")
    Xt = select(np.array([cache[row_key(r)]["x_train"] for r in rows]), FEATURE_SETS["Lab_hist"])
    Xp = select(np.array([cache[row_key(r)]["x"] for r in rows]), FEATURE_SETS["Lab_hist"])
    groups, paths = np.array([r["group"] for r in rows]), np.array([r["path"] for r in rows])
    cap = video_cap_mask(groups, paths, seed=SEED)
    if int(cap.sum()) != reference["r2_train_rows"]:
        raise ValueError("exact R2 reference mask changed")
    if not np.isfinite(Xt).all() or not np.isfinite(Xp).all():
        raise ValueError("nonfinite reference features")
    return rows, Xt, Xp, cap, cache_path, errors


def qa_keep(rows: list[dict], mask: dict) -> np.ndarray:
    # Missing paths are a QA-coverage failure, never silently considered clean.
    if set(r["path"] for r in rows) - set(mask):
        raise ValueError("QA exclusion ledger does not cover every holdout row")
    return np.array([not (mask[r["path"]].get("hard_exclude", False)
                         if isinstance(mask[r["path"]], dict) else bool(mask[r["path"]])) for r in rows])


def run_loso(family: str, seed: int, arm: str, mask: np.ndarray, rows: list[dict],
             Xt: np.ndarray, Xp: np.ndarray, out: Path, *, fixed: bool = False,
             validation_groups: np.ndarray | None = None) -> dict:
    y = np.array([r["label"] for r in rows])
    src = np.array([r["source"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    inner_groups = grp if validation_groups is None else validation_groups
    if inner_groups.shape != grp.shape:
        raise ValueError("validation groups must cover every original holdout row")
    all_probs, raw_probs = np.zeros((len(y), 3)), np.zeros((len(y), 3))
    folds = {}
    mask_rows = []
    fit_seconds = 0.0
    for held in FOLDS:
        te = src == held
        tr = ~te & mask
        if set(grp[tr]) & set(grp[te]):
            raise AssertionError("outer group leakage")
        if set(y[tr]) != set(CLASSES):
            raise ValueError(f"outer training classes missing: {held}")
        cfg, temperature = {"C": 0.01}, 1.0
        inner = None
        if not fixed:
            cfg, temperature, inner = select_inner(family, Xt[tr], Xp[tr], y[tr], inner_groups[tr], seed)
            inner["near_duplicate_group_guard"] = validation_groups is not None
        pipe, fit = fit_estimator(family, cfg, seed, Xt[tr], y[tr])
        fit_seconds += fit["seconds"] + (sum(t["fit"]["seconds"] for t in inner["trials"]) if inner else 0.0)
        raw_probs[te] = estimator_probs(pipe, family, Xp[te])
        all_probs[te] = estimator_probs(pipe, family, Xp[te], temperature)
        folds[held] = {"training_rows": int(tr.sum()), "training_counts": dict(Counter(y[tr])),
                       "training_groups": sorted(set(grp[tr])), "holdout_rows": int(te.sum()),
                       "holdout_groups": sorted(set(grp[te])), "config": cfg, "temperature": temperature,
                       "fit": fit, "inner": inner}
        for i, r in enumerate(rows):
            mask_rows.append({"outer_fold": held, "path": r["path"], "source": r["source"],
                              "label": r["label"], "group": r["group"], "train": bool(tr[i]),
                              "outer_holdout": bool(te[i])})
        print(f"{family} {arm} seed={seed} holdout={held} train={int(tr.sum())} cfg={cfg} T={temperature}", flush=True)
    name = f"{family}_{arm}_{seed}"
    predictions = []
    for r, p, raw in zip(rows, all_probs, raw_probs):
        predictions.append({"path": r["path"], "source": r["source"], "label": r["label"], "group": r["group"],
                            "prediction": CLASSES[int(p.argmax())], "confidence": float(p.max()),
                            **{f"prob_{c}": float(p[k]) for k, c in enumerate(CLASSES)},
                            **{f"uncalibrated_prob_{c}": float(raw[k]) for k, c in enumerate(CLASSES)}})
    write_csv(out / f"predictions_{name}.csv", predictions)
    write_csv(out / f"training_mask_{name}.csv", mask_rows)
    table = fold_table(y, np.array(CLASSES)[all_probs.argmax(axis=1)], src)
    for held in FOLDS:
        te = src == held
        table[held]["calibration"] = calibration(y[te], all_probs[te])
        table[held]["uncalibrated_calibration"] = calibration(y[te], raw_probs[te])
    table["pooled"]["calibration"] = calibration(y, all_probs)
    table["pooled"]["uncalibrated_calibration"] = calibration(y, raw_probs)
    table["mean"]["ece"] = float(np.mean([table[s]["calibration"]["ece"] for s in FOLDS]))
    report = {"family": family, "arm": arm, "seed": seed, "fixed_reference_protocol": fixed,
              "folds": folds, "metrics": table, "training_seconds": fit_seconds,
              "worst_fold": min(FOLDS, key=lambda s: table[s]["macro_f1"]),
              "predictions": f"predictions_{name}.csv", "training_mask": f"training_mask_{name}.csv"}
    dump(out / f"metrics_{name}.json", report)
    return report


def load_inner_group_guard(rows: list[dict], path: Path) -> tuple[np.ndarray, dict]:
    """Only validation grouping changes; cap and outer populations stay original."""
    guard = json.loads(path.read_text(encoding="utf-8"))
    mapping = guard["group_to_inner_group"]
    if set(r["group"] for r in rows) - set(mapping):
        raise ValueError("inner group guard must cover every reference manifest group")
    groups = np.array([mapping[r["group"]] for r in rows])
    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("inner groups must be stable nonempty strings")
    by_path = {r["path"]: groups[i] for i, r in enumerate(rows)}
    pairs_path = path.parent / "near_duplicate_pairs.csv"
    if not pairs_path.is_file():
        raise ValueError("near-duplicate grouping requires hash-linked pair evidence")
    if guard.get("near_duplicate_pairs_sha256") and guard["near_duplicate_pairs_sha256"] != sha256(pairs_path):
        raise ValueError("near-duplicate pair evidence hash does not match group guard")
    count = 0
    with pairs_path.open(newline="", encoding="utf-8") as f:
        for pair in csv.DictReader(f):
            if pair["a"] in by_path and pair["b"] in by_path:
                count += 1
                if by_path[pair["a"]] != by_path[pair["b"]]:
                    raise ValueError("residual near-duplicate pair crosses guarded groups")
    return groups, {"path": str(path), "sha256": sha256(path), "near_duplicate_pairs_sha256": sha256(pairs_path),
                    "manifest_sha256": guard.get("manifest_sha256"),
                    "reference_population_pairs_guarded": count, "residual_pairs_crossing_any_inner_split": 0,
                    "original_groups": len(set(r["group"] for r in rows)), "guarded_groups": len(set(groups)),
                    "policy": "conservative connected candidate groups; original R2 cap/outer holdouts unchanged"}


def export_spec(pipe, family: str, temperature: float) -> dict:
    names = FEATURE_SETS["Lab_hist"]
    if family == "B1":
        spec = spec_from_sklearn(pipe, names, CLASSES)
    else:
        scaler, clf = pipe.steps[0][1], pipe.steps[-1][1]
        spec = {"classes": CLASSES, "feature_names": names, "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist()}
        order = [list(clf.classes_).index(c) for c in CLASSES]
        if family == "MLP":
            layers = [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(clf.coefs_, clf.intercepts_)]
            layers[-1] = {"W": clf.coefs_[-1][:, order].tolist(), "b": clf.intercepts_[-1][order].tolist()}
            spec.update(type="mlp", layers=layers, activation="relu")
        elif family == "RandomForest":
            trees = []
            for tree in clf.estimators_:
                t = tree.tree_
                v = t.value[:, 0, :][:, order]
                v = v / v.sum(axis=1, keepdims=True)
                trees.append({"children_left": t.children_left.tolist(), "children_right": t.children_right.tolist(),
                              "feature": t.feature.tolist(), "threshold": t.threshold.tolist(), "value": v.tolist()})
            spec.update(type="forest", trees=trees)
        elif family == "SVM_RBF":
            spec.update(type="svm_rbf", estimator_classes=clf.classes_.tolist(),
                        support_vectors=clf.support_vectors_.tolist(), n_support=clf.n_support_.tolist(),
                        dual_coef=clf.dual_coef_.tolist(), intercept=clf.intercept_.tolist(), gamma=float(clf._gamma))
    spec["temperature"] = temperature
    return spec


def predict_export(spec: dict, X_all: np.ndarray) -> np.ndarray:
    """Standalone NumPy parity oracle for exported training artifacts."""
    idx = [FEATURES_ALL.index(n) for n in spec["feature_names"]]
    z = (np.asarray(X_all)[:, idx] - np.array(spec["scaler_mean"])) / np.array(spec["scaler_scale"])
    temperature = float(spec.get("temperature", 1.0))
    kind = spec["type"]
    if kind == "softmax":
        p = softmax(z @ np.array(spec["W"]).T + np.array(spec["b"]))
        return calibrate_prob(p, temperature) if temperature != 1.0 else p
    if kind == "mlp":
        for i, layer in enumerate(spec["layers"]):
            z = z @ np.array(layer["W"]) + np.array(layer["b"])
            if i < len(spec["layers"]) - 1:
                z = np.maximum(z, 0)
        p = softmax(z)
        return calibrate_prob(p, temperature) if temperature != 1.0 else p
    if kind == "forest":
        # sklearn trees compare float32 features even when fitted with float64.
        z = z.astype(np.float32)
        out = np.zeros((len(z), 3))
        for tree in spec["trees"]:
            left, right = np.array(tree["children_left"]), np.array(tree["children_right"])
            feature, threshold = np.array(tree["feature"]), np.array(tree["threshold"])
            node = np.zeros(len(z), int)
            while True:
                active = np.flatnonzero(left[node] != -1)
                if not len(active):
                    break
                cur = node[active]
                node[active] = np.where(z[active, feature[cur]] <= threshold[cur], left[cur], right[cur])
            out += np.array(tree["value"])[node]
        return calibrate_prob(out / len(spec["trees"]), temperature)
    if kind == "svm_rbf":
        sv, dual = np.array(spec["support_vectors"]), np.array(spec["dual_coef"])
        starts = np.r_[0, np.cumsum(spec["n_support"])]
        out = []
        order = [spec["estimator_classes"].index(c) for c in spec["classes"]]
        for offset in range(0, len(z), 256):
            batch = z[offset:offset + 256]
            distance = np.maximum((batch ** 2).sum(1)[:, None] + (sv ** 2).sum(1)[None] - 2 * batch @ sv.T, 0)
            kernel = np.exp(-spec["gamma"] * distance)
            votes, conf = np.zeros((len(batch), 3)), np.zeros((len(batch), 3))
            pair = 0
            for i in range(3):
                for j in range(i + 1, 3):
                    si, sj = slice(starts[i], starts[i + 1]), slice(starts[j], starts[j + 1])
                    dec = kernel[:, si] @ dual[j - 1, si] + kernel[:, sj] @ dual[i, sj] + spec["intercept"][pair]
                    votes[:, i] += dec >= 0
                    votes[:, j] += dec < 0
                    conf[:, i] += dec
                    conf[:, j] -= dec
                    pair += 1
            logits = votes + conf / (3 * (np.abs(conf) + 1))
            out.append(softmax(logits[:, order] / temperature))
        return np.concatenate(out)
    raise ValueError(f"unsupported type: {kind}")


def export_candidate(family: str, reports: list[dict], rows: list[dict], Xt: np.ndarray, Xp: np.ndarray,
                     mask: np.ndarray, run_id: str, arm: str, reference: dict, provenance: dict) -> dict:
    evidence = [r for r in reports if r["family"] == family and r["arm"] == arm and r["seed"] == SEED]
    configurations = [r["folds"][s]["config"] for r in evidence for s in FOLDS]
    key_counts = Counter(json.dumps(c, sort_keys=True) for c in configurations)
    cfg = json.loads(key_counts.most_common(1)[0][0])
    temperature = float(np.median([r["folds"][s]["temperature"] for r in evidence for s in FOLDS]))
    y = np.array([r["label"] for r in rows])
    pipe, fit = fit_estimator(family, cfg, SEED, Xt[mask], y[mask])
    spec = export_spec(pipe, family, temperature)
    # Expand selected vectors to the full runtime feature contract for parity.
    matrices = []
    for X in (Xt, Xp):
        full = np.zeros((len(X), len(FEATURES_ALL)))
        full[:, [FEATURES_ALL.index(n) for n in FEATURE_SETS["Lab_hist"]]] = X
        matrices.append(full)
    max_diff, agreement = 0.0, []
    for X, full in zip((Xt, Xp), matrices):
        native = estimator_probs(pipe, family, X, temperature)
        exported = predict_export(spec, full)
        max_diff = max(max_diff, float(np.abs(native - exported).max()))
        agreement.append(float((native.argmax(axis=1) == exported.argmax(axis=1)).mean()))
    if max_diff > 1e-10 or min(agreement) < 1.0:
        raise AssertionError(f"export parity failed {family}: {max_diff}, {agreement}")
    candidate = ML_DIR / "models" / "candidates" / f"tabular_{run_id}_{family}_{arm}"
    candidate.mkdir(parents=True, exist_ok=False)
    dump(candidate / "model.json", spec)
    card = {"backend": "tabular_json", "name": candidate.name, "model_file": "model.json",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "low_conf_threshold": 0.0, "seg_config": reference["seg_config"], "family": family,
            "feature_set": "Lab_hist", "config": cfg, "temperature": temperature,
            "train": {"split": "trainval", "rows": int(mask.sum()), "groups": len(set(r["group"] for r, keep in zip(rows, mask) if keep)),
                      "sources": FOLDS, "class_counts": dict(Counter(y[mask])), "r2_cap_seed": SEED,
                      "view": "reference R2 bbox train view; ROI agtron; inference pipeline view"},
            "selection": "modal config and median temperature from outer folds' group-disjoint inner validation; final refit full R2 pool",
            "calibration_limitation": "temperature estimated on inner held-out predictions then retained after full-pool refit",
            "provenance": provenance, "fit": fit, "numpy_export_parity_max_abs": max_diff,
            "numpy_export_label_agreement_train_pipeline": agreement,
            "data_license_note": "contains agtron Unknown license; internal use only, do not commit",
            "validation_status": {"trainval_export_parity": True, "pi_latency": False, "fw_e2e": False, "frozen_test": False},
            "limitations": ["No frozen-test evaluation", "No Pi latency measurement", "LOSO does not establish arbitrary phone accuracy",
                            "Top-three outer-LOSO selection is optimistic; frozen test remains reserved"]}
    dump(candidate / "model_card.json", card)
    return {"family": family, "arm": arm, "directory": str(candidate), "model_sha256": sha256(candidate / "model.json"),
            "model_bytes": (candidate / "model.json").stat().st_size, "max_abs_probability_difference": max_diff,
            "label_agreement": agreement}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--qa-mask", required=True, type=Path)
    parser.add_argument("--inner-groups", type=Path,
                        help="QA connected group guard for inner validation only; original cap stays unchanged")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    reference_path = ML_DIR / "results" / "baseline_loso.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference["selected"] != {"run": "R2", "model": "B1", "feature_set": "Lab_hist", "C": 0.01,
                                  "selected_by": "auto", "mean_macro_f1_pipeline": 0.6381973706626417,
                                  "mean_macro_f1_bbox": 0.6774059546653926}:
        raise ValueError("reference differs; explicitly review before comparing")
    rows, Xt, Xp, cap, cache_path, errors = load_cached(root, reference)
    qa = json.loads(args.qa_mask.read_text(encoding="utf-8"))
    if "exclusions" in qa:
        qa = qa["exclusions"]
    if "mask" in qa:
        qa = qa["mask"]
    clean = qa_keep(rows, qa)
    arms = {"reference": cap, "qa": cap & clean}
    validation_groups, guard_info = (load_inner_group_guard(rows, args.inner_groups) if args.inner_groups
                                      else (None, {"used": False, "limitation": "near-duplicate candidates may cross manifest groups"}))
    out = ML_DIR / "results" / f"tabular_{args.run_id}"
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"manifest_sha256": sha256(root / "manifest.csv"), "feature_cache": cache_path.name,
                  "feature_cache_sha256": sha256(cache_path), "reference_report_sha256": sha256(reference_path),
                  "qa_mask_sha256": sha256(args.qa_mask), "training_tool_sha256": sha256(Path(__file__)),
                  "inner_group_guard": guard_info,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    if guard_info.get("manifest_sha256") and guard_info["manifest_sha256"] != provenance["manifest_sha256"]:
        raise ValueError("inner group guard manifest differs from training manifest")
    predeclared = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "grids": GRIDS,
                  "temperature_grid": TEMPERATURES, "seeds": SEEDS, "r2_cap_seed": SEED,
                  "outer_folds": FOLDS, "inner": "first of five StratifiedGroupKFold validation folds containing all classes; no image fallback",
                  "inner_selection": "runtime pipeline-view macro-F1, then calibrated NLL, then grid order",
                  "final_export_selection": "modal selected config and median inner-validation temperature across outer folds at first seed",
                  "top3_selection": "highest seed-20261006 reference-mask mean outer macro-F1, then grid family order; optimistic selection explicitly disclosed",
                  "holdout_rows": len(rows), "original_cap_rows": int(cap.sum()), "qa_cap_rows": int(arms['qa'].sum()),
                  "qa_hard_excluded_rows": int((~clean).sum()), "preprocessing": reference["seg_config"],
                  "train_view": "reference R2 WB bbox/ROI/smart", "eval_view": "reference runtime pipeline/ROI",
                  "provenance": provenance, "cache_errors": errors, "frozen_test_loaded": False}
    dump(out / "predeclared_config.json", predeclared)
    write_csv(out / "r2_training_pool.csv", [{"path": r["path"], "source": r["source"], "label": r["label"],
              "group": r["group"], "r2_cap": bool(cap[i]), "qa_keep": bool(clean[i]),
              "qa_cap": bool(arms["qa"][i])} for i, r in enumerate(rows)])
    reports = []
    fixed = []
    for arm, mask in arms.items():
        r = run_loso("B1", SEED, f"fixed_{arm}", mask, rows, Xt, Xp, out, fixed=True,
                     validation_groups=validation_groups)
        fixed.append(r)
    reproduced = fixed[0]["metrics"]["mean"]["macro_f1"]
    if abs(reproduced - reference["selected"]["mean_macro_f1_pipeline"]) > 1e-12:
        raise AssertionError(f"reference reproduction failed: {reproduced}")
    for family in GRIDS:
        for arm, mask in arms.items():
            reports.append(run_loso(family, SEED, arm, mask, rows, Xt, Xp, out, validation_groups=validation_groups))
    ranking = sorted([r for r in reports if r["arm"] == "reference"],
                     key=lambda r: -r["metrics"]["mean"]["macro_f1"])
    top3 = [r["family"] for r in ranking[:3]]
    dump(out / "top3_selection.json", {"top3": top3, "seed": SEED, "optimistic_outer_selection": True,
                                      "ranking": [{"family": r["family"], "mean_macro_f1": r["metrics"]["mean"]["macro_f1"]} for r in ranking]})
    for seed in SEEDS[1:]:
        for family in top3:
            for arm, mask in arms.items():
                reports.append(run_loso(family, seed, arm, mask, rows, Xt, Xp, out, validation_groups=validation_groups))
    summary = []
    for family in GRIDS:
        for arm in arms:
            runs = [r for r in reports if r["family"] == family and r["arm"] == arm]
            row = {"family": family, "arm": arm, "n_seeds": len(runs)}
            for metric in ("macro_f1", "acc", "cross_step_rate", "ece"):
                for agg in ("mean", "pooled"):
                    values = [r["metrics"][agg]["calibration"]["ece"] if metric == "ece" and agg == "pooled"
                              else r["metrics"][agg][metric] for r in runs]
                    row[f"{agg}_{metric}_mean"] = float(np.mean(values))
                    row[f"{agg}_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(row)
    write_csv(out / "comparison.csv", summary)
    exports = []
    # Also export the rejected MLP for reproducible backend/parity checks; the
    # three-seed selection and comparison ranking remain unchanged.
    for family in GRIDS:
        for arm, mask in arms.items():
            exports.append(export_candidate(family, reports, rows, Xt, Xp, mask, args.run_id, arm, reference, provenance))
    final = {"provenance": provenance, "reference_reproduced": reproduced, "top3": top3, "comparison": summary,
             "fixed_B1_data_ablation": [{"arm": r["arm"], "metrics": r["metrics"]} for r in fixed],
             "exports": exports, "elapsed_seconds": time.perf_counter() - started,
             "limitations": ["Nested results include group validation and calibration; protocol differs from legacy chosen-config baseline",
                             "Single inner group validation has selection variance; outer holdouts contain every original image",
                             "Top-three selected using outer LOSO; confidence intervals are not independent-source population intervals",
                             "No frozen test images read", "No Pi measurement", "No ONNX dependency in training environment"],
             "status": {"tabular_comparison_complete": True, "numpy_export_parity_verified": True,
                        "pi_latency_verified": False, "fw_e2e_verified": False, "frozen_test_evaluated": False}}
    dump(out / "summary.json", final)
    print(json.dumps({"output": str(out), "top3": top3, "reference_reproduced": reproduced,
                      "elapsed_seconds": final["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
