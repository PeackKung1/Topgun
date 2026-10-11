"""Keras CNN LOSO for image-level roast level — the tools/train_d1.py protocol with only the network swapped.

    python -m tools.train_keras_cnn --run-dir results/keras_cnn_<date> --inner-groups <inner_groups.json> --write-plan
    python -m tools.train_keras_cnn --run-dir results/keras_cnn_<date> --inner-groups <inner_groups.json>
    python -m tools.train_keras_cnn --run-dir results/keras_cnn_<date> --summarize-only

Rows, views, video cap, inner split, selection rule, temperature grid, seeds, augmentation code and metrics are
imported from the existing helpers; nothing is re-implemented here except the Keras model and its training loop.
plan.json is written once before any fit and must stay byte-equivalent for every later invocation.
A rerun resumes finished (arm, seed, fold) units from ML/models/candidates/<run>/ (weights stay out of git).
Trainval only; split=test is never read.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from roastml.paths import data_dir
from tools.compare_tabular import TEMPERATURES, calibration, inner_split, nll, softmax
from tools.summarize_runs import evaluate, read_predictions
from tools.train_baseline import CLASSES, FOLDS, SEED, fold_table, load_rows, metrics, video_cap_mask
from tools.train_d1 import AUG_MEDIUM, CFG, SEEDS, augment_one, augmentation_params, choose_temperature, class_weights, csv_write, dump, prepare_views, sha256
from tools.train_d1 import PLAN as D1_PLAN

ML_DIR = Path(__file__).resolve().parents[1]
SUMMARY = ML_DIR / "results" / "summary_20261008"
REFERENCES = ("B1", "D1_finetune", "D1_finetune_medaug")
B1_MEAN_MACRO_F1 = 0.6381973706626417
SIZE, VIEWS = CFG.size, len(CFG.views)

# K2 changes one thing against K1: the fine-tune augmentation "mild" is replaced by "wbexp" =
# mild with only the two photometric ranges (exposure, linear red/blue gain) widened to the predeclared medium values.
AUG_WBEXP = {**D1_PLAN["augmentation_mild"], "exposure_EV": AUG_MEDIUM["exposure_EV"],
             "linear_red_blue_gain_log": AUG_MEDIUM["linear_red_blue_gain_log"]}
ARMS = {
    "K1_control_frozen": {"family": "frozen", "backbone": "MobileNetV3Small", "grid": D1_PLAN["frozen_head_grid"],
                          "max_epochs": D1_PLAN["frozen_max_epochs"]},
    "K1_control_finetune": {"family": "finetune", "backbone": "MobileNetV3Small", "grid": D1_PLAN["finetune_grid"],
                            "max_epochs": D1_PLAN["finetune_max_epochs"]},
    "K2_new_finetune": {"family": "finetune", "backbone": "MobileNetV3Small",
                        "grid": [{"lr": .0003, "augmentation": "none"}, {"lr": .0003, "augmentation": "wbexp"}],
                        "max_epochs": D1_PLAN["finetune_max_epochs"]},
}


def build_plan(inner_groups: Path, manifest: Path, deviations: list[str]) -> dict:
    keep = ("search_seed", "confirmation_seeds", "outer_folds", "training_cap_seed", "cap_frames_per_video", "classes",
            "rgb_view_config", "training_views", "agtron", "patience", "batch_images", "selection", "temperature_grid",
            "low_conf_threshold", "augmentation_mild", "forbidden_augmentation")
    return {
        "protocol": "keras_cnn_v1: tools/train_d1.py groupguard protocol, network swapped to keras.applications",
        **{k: D1_PLAN[k] for k in keep},
        "augmentation_wbexp": AUG_WBEXP,
        "arms": ARMS,
        "arm_rationale": {
            "K1_control_frozen": "control: same frozen-head grid as D1 groupguard (one full-batch AdamW step per epoch, <=40)",
            "K1_control_finetune": "control: same fine-tune grid as D1 groupguard; checks Keras result ~ PyTorch D1_finetune",
            "K2_new_finetune": "one new variable vs K1_control_finetune: 'mild' replaced by 'wbexp' (exposure_EV and "
                               "linear_red_blue_gain_log widened to the medium values; geometry, noise, blur and JPEG stay mild). "
                               "Reason: the 8 Oct evidence names strong colour cast as the D1 failure on rf_boos, and the medium "
                               "arm confounded photometric strength with crop 0.5 and JPEG q30. Same backbone, lr, epochs, grid size.",
        },
        "rows": "tools.train_baseline.load_rows(root); must be 3395 trainval images of 4 sources, 3 classes",
        "views": "tools.train_d1.prepare_views(root): full+smart 224 uint8 RGB; view logits averaged in training and prediction",
        "outer_train": "video_cap_mask(groups, paths, seed=20261006) & source != fold; holdout = every image of the fold source",
        "inner_split": "tools.compare_tabular.inner_split(labels[tr], validation_groups[tr], 20261006)",
        "inner_groups": {"file": inner_groups.name, "directory": inner_groups.parent.name, "sha256": sha256(inner_groups)},
        "manifest_sha256": sha256(manifest),
        "tuning": "seed 20261006 only; chosen config/epoch/temperature reused for seeds 20261006/20261007/20261008; "
                  "refit on the whole fold train; holdout never used for any choice",
        "seeding": "keras.utils.set_random_seed(seed) + tf.config.experimental.enable_op_determinism(); "
                   "numpy default_rng(seed) for batch order and augmentation (same call order as D1)",
        "model": {"backbone": "keras.applications.MobileNetV3Small(input_shape=(224,224,3), include_top=False, weights='imagenet', "
                              "pooling='avg', include_preprocessing=True)",
                  "preprocessing": "the backbone's own built-in Rescaling layer; views are fed as float32 0-255, "
                                   "no ImageNet mean/std and no second normalisation",
                  "head": "one Dense(3) on the 576-d pooled feature (Keras default init)",
                  "batchnorm": "running statistics frozen (backbone called with training=False); affine and conv weights trainable",
                  "optimizer": "keras.optimizers.AdamW(lr, weight_decay=0.01, epsilon=1e-8)",
                  "loss": "class-weighted cross entropy, weights n/(3*count) on the fit rows, weighted mean"},
        "metric": "tools.train_baseline.fold_table + tools.compare_tabular.calibration via tools.summarize_runs.evaluate; "
                  "primary = mean LOSO macro-F1 (mean of 4 folds, then mean +- std(ddof=1) over 3 seeds)",
        "decision_rule": {"wins_if": "mean LOSO macro-F1 (mean over 3 seeds) > B1", "B1_mean_macro_f1": B1_MEAN_MACRO_F1,
                          "otherwise": "B1 stays the deployed model", "references": list(REFERENCES),
                          "reference_source": "results/summary_20261008/model_comparison.csv and per_fold.csv"},
        "export": "only for an arm that beats B1: tf2onnx FP32, max |dp| vs Keras < 1e-5 and identical labels, then notebook p95",
        "deviations": deviations,
        "limits": "LOSO source-level convenience dataset; no Pi/E2E/frozen-test claim; every arm run is reported",
    }


def backend():
    import keras
    import tensorflow as tf
    return tf, keras


def build(tf, keras, name: str):
    base = getattr(keras.applications, name)(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet",
                                             pooling="avg", include_preprocessing=True)
    head = keras.layers.Dense(len(CLASSES), name="roast_head")
    head.build((None, base.output_shape[-1]))
    return base, head


def weight_provenance(name: str) -> dict:
    files = sorted((Path.home() / ".keras" / "models").glob("weights_mobilenet_v3_small*" if name == "MobileNetV3Small" else "*"))
    return {"backbone": name, "source": "keras.applications ImageNet weights (downloaded by Keras)",
            "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]}


def logits_fn(tf, base, head):
    @tf.function(input_signature=[tf.TensorSpec((None, VIEWS, SIZE, SIZE, 3), tf.float32)])
    def forward(x):
        z = head(base(tf.reshape(x, (-1, SIZE, SIZE, 3)), training=False))
        return tf.reduce_mean(tf.reshape(z, (-1, VIEWS, len(CLASSES))), axis=1)
    return forward


def predict_network(forward, images, ids, temperature=1.0):
    out = [forward(np.asarray(images[ids[s:s + 32]], np.float32)).numpy() for s in range(0, len(ids), 32)]
    return softmax(np.concatenate(out) / temperature)


def embeddings(tf, keras, root: Path, images, info: dict, name: str):
    path = root / "cache" / f"keras_embeddings_{info['key']}_{name}.npy"
    if path.exists():
        return np.load(path)
    keras.utils.set_random_seed(SEED)
    base, _ = build(tf, keras, name)
    pooled = tf.function(lambda x: base(x, training=False), input_signature=[tf.TensorSpec((None, SIZE, SIZE, 3), tf.float32)])
    out = []
    for start in range(0, len(images), 32):
        x = np.asarray(images[start:start + 32], np.float32)
        out.append(pooled(x.reshape(-1, SIZE, SIZE, 3)).numpy().reshape(len(x), VIEWS, -1).mean(1))
    arr = np.concatenate(out).astype(np.float32)
    np.save(path, arr)
    return arr


def selected_row(trace):
    return max(trace, key=lambda r: (r["macro_f1"], -r["nll"], -r["epoch"]))


def train_head(tf, keras, X, y, cfg, seed, epochs, plan, validation=None):
    """Linear head on frozen embeddings; like D1, one epoch = one full-batch AdamW step."""
    keras.utils.set_random_seed(seed)
    head = keras.layers.Dense(len(CLASSES)); head.build((None, X.shape[1]))
    optimiser = keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=.01, epsilon=1e-8)
    tx, ty = tf.constant(np.asarray(X, np.float32)), tf.constant(y, tf.int32)
    weights = tf.constant(class_weights(y))
    variables = head.trainable_variables

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ty, logits=head(tx))
            w = tf.gather(weights, ty)
            loss = tf.reduce_sum(ce * w) / tf.reduce_sum(w)
        optimiser.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    trace, best, best_rank, patience = [], None, (-1, -float("inf")), 0
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch, "train_loss": float(step())}
        if validation is not None:
            vx, vy = validation
            p = softmax(head(tf.constant(np.asarray(vx, np.float32))).numpy())
            row.update({"macro_f1": metrics(np.array(CLASSES)[vy], np.array(CLASSES)[p.argmax(1)])["macro_f1"],
                        "nll": nll(np.array(CLASSES)[vy], p)})
            rank = (row["macro_f1"], -row["nll"])
            if rank > best_rank: best_rank, best, patience = rank, [v.numpy() for v in variables], 0
            else: patience += 1
        trace.append(row)
        if validation is not None and patience >= plan["patience"]: break
    if validation is not None:
        for v, value in zip(variables, best): v.assign(value)
    kernel, bias = (v.numpy().astype(np.float64) for v in variables)
    return (kernel, bias), trace, (selected_row(trace) if validation is not None else {"epoch": epochs})


def head_probs(weights, X, temperature=1.0):
    kernel, bias = weights
    return softmax((np.asarray(X, np.float64) @ kernel + bias) / temperature)


def train_finetune(tf, keras, images, y, ids, cfg, seed, epochs, plan, name, validation=None):
    keras.utils.set_random_seed(seed)
    base, head = build(tf, keras, name)
    optimiser = keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=.01, epsilon=1e-8)
    weights = tf.constant(class_weights(y[ids]))
    variables = base.trainable_variables + head.trainable_variables
    forward = logits_fn(tf, base, head)

    @tf.function(input_signature=[tf.TensorSpec((None, VIEWS, SIZE, SIZE, 3), tf.float32), tf.TensorSpec((None,), tf.int32)])
    def step(x, target):
        with tf.GradientTape() as tape:
            # BN running statistics stay frozen (training=False); affine parameters and convolutions still fine-tune.
            z = head(base(tf.reshape(x, (-1, SIZE, SIZE, 3)), training=False))
            logits = tf.reduce_mean(tf.reshape(z, (-1, VIEWS, len(CLASSES))), axis=1)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
            w = tf.gather(weights, target)
            loss = tf.reduce_sum(ce * w) / tf.reduce_sum(w)
        optimiser.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    rng = np.random.default_rng(seed)
    aug = augmentation_params(plan, cfg["augmentation"])
    everything = base.weights + head.weights
    trace, best, best_rank, patience, started = [], None, (-1, -float("inf")), 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        shuffled = rng.permutation(ids)
        for start in range(0, len(ids), 32):
            batch = shuffled[start:start + 32]; pixels = np.array(images[batch], copy=True)
            if aug is not None:
                for i in range(len(pixels)):
                    for j in range(VIEWS): pixels[i, j] = augment_one(pixels[i, j], rng, aug)
            loss_sum += float(step(pixels.astype(np.float32), y[batch].astype(np.int32))) * len(batch)
        row = {"epoch": epoch, "train_loss": loss_sum / len(ids), "elapsed_seconds": time.perf_counter() - started}
        if validation is not None:
            p = predict_network(forward, images, validation)
            row.update({"macro_f1": metrics(np.array(CLASSES)[y[validation]], np.array(CLASSES)[p.argmax(1)])["macro_f1"],
                        "nll": nll(np.array(CLASSES)[y[validation]], p)})
            rank = (row["macro_f1"], -row["nll"])
            if rank > best_rank: best_rank, best, patience = rank, [v.numpy() for v in everything], 0
            else: patience += 1
        trace.append(row); print(f"fine {name} {seed} {cfg} epoch {epoch}/{epochs} {row}", flush=True)
        if validation is not None and patience >= plan["patience"]: break
    if validation is not None:
        for v, value in zip(everything, best): v.assign(value)
    return (base, head, forward), trace, (selected_row(trace) if validation is not None else {"epoch": epochs})


def release(keras):
    keras.backend.clear_session(); gc.collect()


def tune(tf, keras, arm, images, E, y, ti, vi, plan):
    """All inner trials of one outer fold (search seed only). Selection: macro-F1, then NLL, then grid order."""
    trials = []
    for cfg in arm["grid"]:
        if arm["family"] == "frozen":
            weights, trace, selected = train_head(tf, keras, E[ti], y[ti], cfg, SEED, arm["max_epochs"], plan, (E[vi], y[vi]))
            vp = head_probs(weights, E[vi])
        else:
            model, trace, selected = train_finetune(tf, keras, images, y, ti, cfg, SEED, arm["max_epochs"], plan, arm["backbone"], vi)
            vp = predict_network(model[2], images, vi)
            del model
        temperature, losses = choose_temperature(y[vi], vp)
        trials.append({"config": cfg, "trace": trace, "selected": selected, "temperature": temperature, "temperature_nll": losses})
        release(keras)
    best = max(range(len(trials)), key=lambda i: (trials[i]["selected"]["macro_f1"], -trials[i]["selected"]["nll"], -i))
    return trials, best


def run(root: Path, run_dir: Path, inner_groups_path: Path, plan: dict, limit: int | None = None) -> None:
    tf, keras = backend()
    tf.config.experimental.enable_op_determinism()
    started = time.perf_counter()
    rows, images, info = prepare_views(root)
    if len(rows) != 3395:
        raise SystemExit(f"load_rows returned {len(rows)} rows, expected 3395")
    y = np.array([CLASSES.index(r["label"]) for r in rows]); labels = np.array(CLASSES)[y]
    source = np.array([r["source"] for r in rows]); groups = np.array([r["group"] for r in rows])
    paths = np.array([r["path"] for r in rows])
    mapping = json.loads(inner_groups_path.read_text(encoding="utf-8"))["group_to_inner_group"]
    if set(groups) - set(mapping):
        raise ValueError("inner group mapping does not cover every training group")
    validation_groups = np.array([mapping[g] for g in groups])
    cap = video_cap_mask(groups, paths, seed=SEED)
    models_dir = ML_DIR / "models" / "candidates" / run_dir.name
    versions = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
                "keras_backend": keras.backend.backend(), "numpy": np.__version__, "opencv": cv2.__version__,
                "built_with_cuda": bool(tf.test.is_built_with_cuda()), "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
                "device": "CPU" if not tf.config.list_physical_devices("GPU") else "GPU", "platform": platform.platform(),
                "processor": platform.processor()}
    state = {"versions": versions, "data": {k: info[k] for k in ("key", "n", "shape", "manifest_sha256", "preprocessing_code_sha256")},
             "capped_training_rows": int(cap.sum()), "inner_groups_sha256": sha256(inner_groups_path),
             "training_code_sha256": sha256(Path(__file__)), "weights": {}, "arms": {}}
    for arm_name, arm in plan["arms"].items():
        state["weights"][arm["backbone"]] = None
        E = embeddings(tf, keras, root, images, info, arm["backbone"]) if arm["family"] == "frozen" else None
        release(keras)
        seeds = []
        for seed in SEEDS:
            probs = np.zeros((len(rows), len(CLASSES))); fold_runs = {}
            for fold in FOLDS:
                tr = np.flatnonzero(cap & (source != fold)); te = np.flatnonzero(source == fold)
                if limit: tr, te = tr[:: max(1, len(tr) // limit)], te[:: max(1, len(te) // limit)]
                selection_path = run_dir / f"{arm_name}_{fold}_selection.json"
                if not selection_path.exists():
                    if seed != SEED:
                        raise RuntimeError(f"{selection_path.name} missing: tuning must run with the search seed first")
                    it, iv, split = inner_split(labels[tr], validation_groups[tr], SEED)
                    ti, vi = tr[it], tr[iv]
                    trials, best = tune(tf, keras, arm, images, E, y, ti, vi, plan)
                    dump(selection_path, {"split": split, "trials": trials, "selected": best,
                                          "train_paths": paths[ti].tolist(), "validation_paths": paths[vi].tolist()})
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                chosen = selection["trials"][selection["selected"]]
                cfg, epochs, temperature = chosen["config"], chosen["selected"]["epoch"], chosen["temperature"]
                unit = models_dir / f"{arm_name}_{seed}_{fold}"
                if not (unit / "done.json").exists():
                    unit.mkdir(parents=True, exist_ok=True)
                    t0 = time.perf_counter()
                    if arm["family"] == "frozen":
                        weights, trace, _ = train_head(tf, keras, E[tr], y[tr], cfg, seed, epochs, plan)
                        p = head_probs(weights, E[te], temperature)
                        np.savez(unit / "head.npz", kernel=weights[0], bias=weights[1])
                    else:
                        model, trace, _ = train_finetune(tf, keras, images, y, tr, cfg, seed, epochs, plan, arm["backbone"])
                        p = predict_network(model[2], images, te, temperature)
                        np.savez(unit / "model.npz", **{f"{i:03d}": v.numpy() for i, v in enumerate(model[0].weights + model[1].weights)})
                        del model
                    np.save(unit / "holdout_probs.npy", p)
                    dump(unit / "done.json", {"arm": arm_name, "seed": seed, "fold": fold, "train_rows": len(tr),
                                              "train_groups": len(set(groups[tr])), "holdout_rows": len(te),
                                              "holdout_groups": len(set(groups[te])), "config": cfg, "epochs": epochs,
                                              "temperature": temperature, "train_seconds": time.perf_counter() - t0,
                                              "trace": trace, "class_order": CLASSES, "backbone": arm["backbone"]})
                    release(keras)
                    print(f"completed {arm_name} seed={seed} fold={fold}", flush=True)
                probs[te] = np.load(unit / "holdout_probs.npy")
                fold_runs[fold] = json.loads((unit / "done.json").read_text(encoding="utf-8"))
            if limit:
                continue
            pred = np.array(CLASSES)[probs.argmax(1)]
            order = np.argsort(paths)
            csv_write(run_dir / f"{arm_name}_{seed}_predictions.csv",
                      [{"path": rows[i]["path"], "source": rows[i]["source"], "group": rows[i]["group"], "label": rows[i]["label"],
                        "prediction": pred[i], **{f"p_{c}": float(probs[i, k]) for k, c in enumerate(CLASSES)}} for i in order])
            table = fold_table(labels, pred, source)
            seeds.append({"seed": seed, "metrics": table, "calibration": calibration(labels, probs), "fold_runs": fold_runs,
                          "worst_fold": min(FOLDS, key=lambda f: table[f]["macro_f1"])})
            dump(run_dir / "progress.json", {"arm": arm_name, "seed": seed, "mean_macro_f1": table["mean"]["macro_f1"],
                                             "seconds": time.perf_counter() - started})
        state["arms"][arm_name] = {"seeds": seeds}
    for name in state["weights"]:
        state["weights"][name] = weight_provenance(name)
    state["seconds_this_invocation"] = time.perf_counter() - started
    if not limit:
        dump(run_dir / "run_state.json", state)
        summarize(run_dir, plan)


def references() -> tuple[dict, list[dict]]:
    with (SUMMARY / "model_comparison.csv").open(newline="", encoding="utf-8") as f:
        overall = {r["model"]: r for r in csv.DictReader(f) if r["model"] in REFERENCES}
    with (SUMMARY / "per_fold.csv").open(newline="", encoding="utf-8") as f:
        per_fold = [r for r in csv.DictReader(f) if r["model"] in REFERENCES]
    if set(overall) != set(REFERENCES):
        raise ValueError("reference rows missing from results/summary_20261008/model_comparison.csv")
    return overall, per_fold


def summarize(run_dir: Path, plan: dict) -> dict:
    """Recompute every number from the prediction files with the 8 Oct summary code, then add the reference rows."""
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    ref_overall, ref_folds = references()
    columns = ["model", "n_seeds", "mean_macro_f1_mean", "mean_macro_f1_std", "pooled_macro_f1_mean", "pooled_macro_f1_std",
               "mean_acc_mean", "pooled_acc_mean", "pooled_cross_step_rate_mean", "pooled_ece_mean",
               "worst_fold_first_seed", "worst_fold_macro_f1_first_seed", "beats_B1", "source"]
    compare, per_fold, arms = [], [], {}
    for name in plan["arms"]:
        tables = []
        for seed in SEEDS:
            p, labels, source, P = read_predictions(run_dir / f"{name}_{seed}_predictions.csv", "p_")
            if len(p) != 3395:
                raise ValueError(f"{name} {seed}: predictions must cover the 3395-image holdout population")
            tables.append(evaluate(labels, source, P))
        mean_f1 = np.array([t["mean"]["macro_f1"] for t in tables]); pooled_f1 = np.array([t["pooled"]["macro_f1"] for t in tables])
        first = tables[0]; worst = min(FOLDS, key=lambda s: first[s]["macro_f1"])
        row = {"model": name, "n_seeds": len(tables), "mean_macro_f1_mean": float(mean_f1.mean()),
               "mean_macro_f1_std": float(mean_f1.std(ddof=1)), "pooled_macro_f1_mean": float(pooled_f1.mean()),
               "pooled_macro_f1_std": float(pooled_f1.std(ddof=1)),
               "mean_acc_mean": float(np.mean([t["mean"]["acc"] for t in tables])),
               "pooled_acc_mean": float(np.mean([t["pooled"]["acc"] for t in tables])),
               "pooled_cross_step_rate_mean": float(np.mean([t["pooled"]["cross_step_rate"] for t in tables])),
               "pooled_ece_mean": float(np.mean([t["pooled"]["calibration"]["ece"] for t in tables])),
               "worst_fold_first_seed": worst, "worst_fold_macro_f1_first_seed": first[worst]["macro_f1"],
               "beats_B1": bool(mean_f1.mean() > B1_MEAN_MACRO_F1), "source": f"{run_dir.name} (this run)"}
        compare.append(row)
        for s in FOLDS + ["pooled"]:
            ece = [t[s]["ece"] if s != "pooled" else t[s]["calibration"]["ece"] for t in tables]
            f1 = np.array([t[s]["macro_f1"] for t in tables])
            per_fold.append({"model": name, "fold": s, "n": first[s]["n"], "n_seeds": len(tables), "macro_f1_mean": float(f1.mean()),
                             "macro_f1_std": float(f1.std(ddof=1)), "acc_mean": float(np.mean([t[s]["acc"] for t in tables])),
                             "cross_step_rate_mean": float(np.mean([t[s]["cross_step_rate"] for t in tables])),
                             "ece_mean": float(np.mean(ece)), "source": f"{run_dir.name} (this run)"})
        arms[name] = {**row, "per_seed_mean_macro_f1": mean_f1.tolist(),
                      "per_fold_macro_f1_by_seed": {s: [t[s]["macro_f1"] for t in tables] for s in FOLDS},
                      "pooled_confusion_first_seed": first["pooled"]["cm"],
                      "selection": {f: {k: v for k, v in state["arms"][name]["seeds"][0]["fold_runs"][f].items()
                                        if k in ("config", "epochs", "temperature", "train_rows", "train_groups", "holdout_rows")}
                                    for f in FOLDS},
                      "train_seconds": {str(s["seed"]): sum(r["train_seconds"] for r in s["fold_runs"].values()) for s in state["arms"][name]["seeds"]}}
    for name in REFERENCES:
        r = ref_overall[name]
        compare.append({**{k: (r[k] if k in ("model", "worst_fold_first_seed") else (float(r[k]) if r[k] != "" else None))
                           for k in columns[:12]}, "n_seeds": int(r["n_seeds"]),
                        "beats_B1": False if name != "B1" else None, "source": "results/summary_20261008/model_comparison.csv"})
        for s in FOLDS + ["pooled"]:
            sub = [x for x in ref_folds if x["model"] == name and x["fold"] == s]
            f1 = np.array([float(x["macro_f1"]) for x in sub])
            per_fold.append({"model": name, "fold": s, "n": int(sub[0]["n"]), "n_seeds": len(sub), "macro_f1_mean": float(f1.mean()),
                             "macro_f1_std": float(f1.std(ddof=1)) if len(sub) > 1 else None,
                             "acc_mean": float(np.mean([float(x["acc"]) for x in sub])),
                             "cross_step_rate_mean": float(np.mean([float(x["cross_step_rate"]) for x in sub])),
                             "ece_mean": float(np.mean([float(x["ece"]) for x in sub])), "source": "results/summary_20261008/per_fold.csv"})
    csv_write(run_dir / "compare.csv", [{k: r.get(k) for k in columns} for r in compare])
    csv_write(run_dir / "compare_per_fold.csv", per_fold)
    winners = [n for n in plan["arms"] if arms[n]["beats_B1"]]
    report = {"plan": plan, "versions": state["versions"], "weights": state["weights"], "data": state["data"],
              "capped_training_rows": state["capped_training_rows"], "inner_groups_sha256": state["inner_groups_sha256"],
              "training_code_sha256": state["training_code_sha256"], "arms": arms,
              "references": {n: {k: ref_overall[n][k] for k in columns[1:12]} for n in REFERENCES},
              "decision": {"B1_mean_macro_f1": B1_MEAN_MACRO_F1, "arms_beating_B1": winners,
                           "result": "an arm beats B1; run the ONNX export checks" if winners else "no arm beats B1; B1 stays the deployed model"},
              "frozen_test_loaded": False}
    dump(run_dir / "report.json", report)
    for r in compare:
        std = f"±{r['mean_macro_f1_std']:.4f}" if r["mean_macro_f1_std"] is not None else ""
        print(f"{r['model']:22s} seeds={r['n_seeds']} meanF1={r['mean_macro_f1_mean']:.4f}{std} pooledF1={r['pooled_macro_f1_mean']:.4f} "
              f"worst={r['worst_fold_first_seed']}:{r['worst_fold_macro_f1_first_seed']:.3f} ECE={r['pooled_ece_mean']:.3f} "
              f"xstep={r['pooled_cross_step_rate_mean']:.3f}", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--inner-groups", type=Path, help="default: $ROAST_DATA_DIR/cache/qa_audit_20261008/inner_groups.json")
    ap.add_argument("--write-plan", action="store_true", help="write plan.json (before any fit) and exit")
    ap.add_argument("--deviation", action="append", default=[], help="recorded in plan.json; repeat for several")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--smoke-limit", type=int, help="code check only: ~N rows per fold train/holdout; run dir must be outside results/")
    args = ap.parse_args(argv)
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    run_dir = args.run_dir.resolve()
    plan_path = run_dir / "plan.json"
    if args.summarize_only:
        summarize(run_dir, json.loads(plan_path.read_text(encoding="utf-8"))); return 0
    inner = args.inner_groups or root / "cache/qa_audit_20261008/inner_groups.json"
    if not inner.exists():
        raise SystemExit(f"inner group file missing: {inner} — stop and ask; never tune on holdout images")
    if args.write_plan:
        if plan_path.exists():
            raise SystemExit(f"{plan_path} already exists; a plan is written once")
        run_dir.mkdir(parents=True, exist_ok=True)
        dump(plan_path, build_plan(inner, root / "manifest.csv", args.deviation)); print(f"plan written: {plan_path}"); return 0
    if not plan_path.exists():
        raise SystemExit("plan.json missing: run --write-plan before the first fit")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(build_plan(inner, root / "manifest.csv", plan["deviations"])))
    if current != plan:
        raise RuntimeError("plan.json differs from the plan this code and these inputs produce; create a new run directory")
    if args.smoke_limit and run_dir.is_relative_to((ML_DIR / "results").resolve()):
        raise SystemExit("--smoke-limit is a code check; use a run directory outside ML/results")
    run(root, run_dir, inner, plan, args.smoke_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
