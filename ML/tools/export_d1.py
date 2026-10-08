"""Export D1 checkpoints to FP32/static INT8 ONNX and verify trainval parity.

Fold calibration uses at most 128 original R2 training images, never holdouts.
Final calibration uses 200 training images. --export-final performs a fresh
group-disjoint inner selection before refitting all original R2 training rows.
No frozen-test image bytes are read; candidates/current are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from roastml.rgb_views import RGBViewConfig
from tools.compare_tabular import calibration, inner_split, load_inner_group_guard, nll, softmax
from tools.train_baseline import CLASSES, FOLDS, SEED, fold_table, video_cap_mask
from tools.train_d1 import (
    CFG, ML_DIR, PLAN, choose_temperature, dump, embeddings, official_model,
    predict_network, prepare_views, seed_all, sha256, train_finetune, train_head,
)

EXPORT_PLAN = {
    "opset": 17, "dynamic_batch": True, "input": "rgb", "output": "logits",
    "export_device": "cpu", "fold_calibration_max_images": 128,
    "final_calibration_images": 200, "calibration_seed": SEED,
    "calibration_sampling": "round-robin stratified source x class over original training cap, seeded shuffled paths",
    "quantization": {"format": "QDQ", "per_channel": True, "activation": "QUInt8", "weight": "QInt8",
                     "operators": ["Conv", "Gemm", "MatMul"], "method": "MinMax"},
    "verification": "every unchanged source holdout; all views; actual CPU Torch versus FP32 and INT8",
    "fp32_max_probability_error": 0.0002, "fp32_label_agreement": 1.0,
    "final_selection": "fresh connected-group inner validation using predeclared D1 grids; then full-cap refit",
    "final_probe": "first 200 seeded source-class training calibration images; diagnostic only, never test",
    "frozen_test": "never loaded", "pi_latency": "not measured by export verification",
}


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalise_pixels(pixels: np.ndarray, config: dict) -> np.ndarray:
    """N,V,H,W,C uint8 to N,V,C,H,W, identical to shared RGB view builder."""
    cfg = RGBViewConfig.from_dict(config)
    x = np.asarray(pixels, np.float32) / 255
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError("expected batch x views x height x width x RGB")
    x = (x - np.asarray(cfg.mean, np.float32)) / np.asarray(cfg.std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 1, 4, 2, 3))


def pool_logits(logits: np.ndarray, temperature: float, weights) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    w = np.asarray(weights, np.float64)
    if z.ndim != 3 or z.shape[2] != 3 or z.shape[1] != len(w) or not np.isfinite(z).all():
        raise ValueError("invalid N x views x classes logits")
    if temperature <= 0 or not np.isfinite(temperature) or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid temperature or view weights")
    p = softmax(z.reshape(-1, 3) / temperature).reshape(z.shape)
    return softmax(np.average(np.log(np.clip(p, 1e-12, 1)), axis=1, weights=w))


def calibration_ids(rows: list[dict], training_ids: np.ndarray, maximum: int, seed: int = SEED) -> np.ndarray:
    """Bounded, representative sampling using only declared training IDs."""
    if maximum < 1 or len(training_ids) < 1:
        raise ValueError("calibration needs training images")
    if len(set(int(i) for i in training_ids)) != len(training_ids):
        raise ValueError("training IDs must be unique")
    strata = {}
    for i in sorted(training_ids, key=lambda i: rows[int(i)]["path"]):
        r = rows[int(i)]
        strata.setdefault((r["source"], r["label"]), []).append(int(i))
    rng = np.random.default_rng(seed)
    queues = {k: list(rng.permutation(v)) for k, v in sorted(strata.items())}
    selected = []
    while len(selected) < min(maximum, len(training_ids)):
        for key in queues:
            if queues[key] and len(selected) < maximum:
                selected.append(queues[key].pop())
    return np.asarray(selected, int)


def ort_session(path: Path, threads: int = 1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def export_fp32(model, destination: Path, size: int = 224) -> dict:
    import onnx
    import torch
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    model = model.cpu().eval()
    sample = torch.zeros(1, 3, size, size)
    started = time.perf_counter()
    # Verified torch 2.14 legacy API avoids an uninstalled onnxscript dependency.
    torch.onnx.export(model, (sample,), str(destination), input_names=["rgb"], output_names=["logits"],
                      dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=EXPORT_PLAN["opset"], dynamo=False, external_data=False)
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    session = ort_session(destination)
    actual = session.run(["logits"], {"rgb": np.zeros((2, 3, size, size), np.float32)})[0]
    if actual.shape != (2, 3) or not np.isfinite(actual).all():
        raise ValueError("exported dynamic-batch graph does not return N x 3 logits")
    return {"sha256": sha256(destination), "bytes": destination.stat().st_size,
            "seconds": time.perf_counter() - started, "ir_version": graph.ir_version,
            "opset": {o.domain: o.version for o in graph.opset_import}, "dynamic_batch_verified": True}


def quantize_int8(fp32_path: Path, destination: Path, images: np.ndarray, ids: np.ndarray,
                  config: dict) -> dict:
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.samples = self.iterator()

        def iterator(self):
            for image_id in ids:
                views = normalise_pixels(images[int(image_id)][None], config)[0]
                for view in views:
                    yield {"rgb": view[None]}

        def get_next(self):
            return next(self.samples, None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    processed = destination.parent / "quantization_preprocessed.onnx"
    started = time.perf_counter()
    quant_pre_process(fp32_path, processed, skip_optimization=False, skip_symbolic_shape=False,
                      skip_onnx_shape=False, auto_merge=True)
    quantize_static(processed, destination, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=EXPORT_PLAN["quantization"]["operators"],
                    calibrate_method=CalibrationMethod.MinMax, calibration_providers=["CPUExecutionProvider"],
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "DedicatedQDQPair": True})
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    counts = Counter(n.op_type for n in graph.graph.node)
    if not counts["QuantizeLinear"] or not counts["DequantizeLinear"]:
        raise ValueError("INT8 graph lacks static QDQ nodes")
    return {"sha256": sha256(destination), "bytes": destination.stat().st_size,
            "seconds": time.perf_counter() - started, "node_counts": dict(counts),
            "calibration_images": len(ids), "calibration_views": len(ids) * len(config["views"]),
            "calibration_indices": ids.tolist()}


def compare_predictions(y: np.ndarray, native: np.ndarray, candidate: np.ndarray) -> dict:
    delta = np.abs(candidate - native)
    pred_native, pred_candidate = np.asarray(CLASSES)[native.argmax(1)], np.asarray(CLASSES)[candidate.argmax(1)]
    return {"rows": len(y), "max_abs_probability_difference": float(delta.max()),
            "mean_abs_probability_difference": float(delta.mean()),
            "label_agreement": float((pred_native == pred_candidate).mean()),
            "label_changes": int((pred_native != pred_candidate).sum()),
            "candidate_calibration": calibration(y, candidate), "native_calibration": calibration(y, native)}


def verify_models(model, fp32_path: Path, int8_path: Path, images: np.ndarray, ids: np.ndarray,
                  rows: list[dict], card: dict, output: Path) -> dict:
    import torch
    cfg = card["rgb_view_config"]
    temperature = float(card["temperature"])
    sessions = {"fp32": ort_session(fp32_path), "int8": ort_session(int8_path)}
    model = model.cpu().eval()
    probabilities = {"torch": [], "fp32": [], "int8": []}
    times = {k: 0.0 for k in probabilities}
    max_logits_diff = 0.0
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            batch_ids = ids[start:start + 16]
            x = normalise_pixels(images[batch_ids], cfg)
            b, v, c, h, w = x.shape
            tensor = np.ascontiguousarray(x.reshape(b * v, c, h, w))
            t0 = time.perf_counter()
            native = model(torch.from_numpy(tensor)).cpu().numpy()
            times["torch"] += time.perf_counter() - t0
            probabilities["torch"].append(pool_logits(native.reshape(b, v, 3), temperature, cfg["view_weights"]))
            for backend, session in sessions.items():
                t0 = time.perf_counter()
                actual = session.run(["logits"], {"rgb": tensor})[0]
                times[backend] += time.perf_counter() - t0
                if backend == "fp32":
                    max_logits_diff = max(max_logits_diff, float(np.abs(actual - native).max()))
                probabilities[backend].append(pool_logits(actual.reshape(b, v, 3), temperature, cfg["view_weights"]))
    probabilities = {k: np.concatenate(v) for k, v in probabilities.items()}
    labels = np.array([rows[int(i)]["label"] for i in ids])
    source = np.array([rows[int(i)]["source"] for i in ids])
    result = {"n": len(ids), "torch_vs_fp32": compare_predictions(labels, probabilities["torch"], probabilities["fp32"]),
              "torch_vs_int8": compare_predictions(labels, probabilities["torch"], probabilities["int8"]),
              "fp32_vs_int8": compare_predictions(labels, probabilities["fp32"], probabilities["int8"]),
              "max_fp32_logit_difference": max_logits_diff, "inference_seconds": times,
              "timing_scope": "cached RGB model forwards only; excludes bytes decode/views; not a Pi benchmark",
              "metrics": {k: fold_table(labels, np.array(CLASSES)[p.argmax(1)], source)
                          for k, p in probabilities.items()}, "frozen_test_loaded": False}
    predictions = []
    for position, image_id in enumerate(ids):
        row = rows[int(image_id)]
        prediction = {k: row[k] for k in ("path", "label", "source", "group")}
        for backend, p in probabilities.items():
            prediction[f"{backend}_prediction"] = CLASSES[int(p[position].argmax())]
            for j, cls in enumerate(CLASSES):
                prediction[f"{backend}_prob_{cls}"] = float(p[position, j])
        predictions.append(prediction)
    csv_write(output / "predictions.csv", predictions)
    result["fp32_probability_parity_verified"] = result["torch_vs_fp32"]["max_abs_probability_difference"] <= EXPORT_PLAN["fp32_max_probability_error"]
    result["fp32_label_parity_verified"] = result["torch_vs_fp32"]["label_agreement"] == 1.0
    dump(output / "verification.json", result)
    if not result["fp32_probability_parity_verified"] or not result["fp32_label_parity_verified"]:
        raise ValueError(f"FP32 export parity not established; evidence preserved in {output}")
    return result


def export_pair(model, checkpoint: Path, original_card: dict, rows: list[dict], images: np.ndarray,
                training: np.ndarray, evaluation: np.ndarray, output: Path, export_id: str,
                maximum_calibration: int) -> dict:
    if original_card["class_order"] != CLASSES:
        raise ValueError("checkpoint class order differs from training labels")
    cfg = original_card["rgb_view_config"]
    sample = calibration_ids(rows, training, maximum_calibration)
    if set(sample) & (set(evaluation) - set(training)):
        raise ValueError("holdout leaked into calibration")
    output.mkdir(parents=True, exist_ok=False)
    candidate_base = checkpoint / "exports" / export_id
    fp32_dir, int8_dir = candidate_base / "fp32", candidate_base / "int8"
    fp32_dir.mkdir(parents=True, exist_ok=False)
    int8_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    fp32 = export_fp32(model, fp32_dir / "model.onnx", cfg["size"])
    int8 = quantize_int8(fp32_dir / "model.onnx", int8_dir / "model.onnx", images, sample, cfg)
    calibration_paths = [rows[int(i)]["path"] for i in sample]
    training_paths = [rows[int(i)]["path"] for i in training]
    calibration_record = {"indices": sample.tolist(), "paths": calibration_paths, "rows": len(sample),
                          "source_class_counts": dict(Counter(f"{rows[int(i)]['source']}:{rows[int(i)]['label']}" for i in sample)),
                          "path_sha256": hashlib.sha256("\n".join(calibration_paths).encode()).hexdigest(),
                          "training_mask_sha256": hashlib.sha256("\n".join(training_paths).encode()).hexdigest(),
                          "training_rows": len(training), "holdout_calibration_overlap": 0}
    dump(output / "calibration.json", calibration_record)
    cards = []
    for kind, directory, info in [("fp32", fp32_dir, fp32), ("int8", int8_dir, int8)]:
        card = {**original_card, "backend": "onnx_rgb", "name": f"{original_card['name']}-{kind}",
                "model_file": "model.onnx", "output_kind": "logits", "input_name": "rgb", "output_name": "logits",
                "intra_op_num_threads": 1, "inter_op_num_threads": 1,
                "export": {"precision": kind, "info": info, "plan": EXPORT_PLAN,
                           "source_checkpoint_sha256": sha256(checkpoint / "model.pth"), "calibration": calibration_record},
                "status": {"pi_latency_verified": False, "fw_e2e_verified": False, "frozen_test_evaluated": False},
                "data_license_note": "includes agtron Unknown license; internal use only; never commit model artifacts"}
        dump(directory / "model_card.json", card)
        cards.append({"precision": kind, "directory": str(directory), "model_sha256": info["sha256"], "model_bytes": info["bytes"]})
    verification = verify_models(model, fp32_dir / "model.onnx", int8_dir / "model.onnx", images, evaluation, rows, original_card, output)
    result = {"source_checkpoint": str(checkpoint), "artifacts": cards, "calibration": calibration_record,
              "verification": verification, "seconds": time.perf_counter() - started}
    dump(output / "summary.json", result)
    return result


def final_train(family: str, rows: list[dict], images: np.ndarray, info: dict, root: Path,
                cap: np.ndarray, inner_groups: np.ndarray, device, checkpoint: Path) -> tuple[object, dict]:
    import torch
    labels = np.array([r["label"] for r in rows])
    y = np.array([CLASSES.index(c) for c in labels])
    ids = np.flatnonzero(cap)
    it, iv, split = inner_split(labels[ids], inner_groups[ids], SEED)
    ti, vi = ids[it], ids[iv]
    E, weights = embeddings(root, images, info, device)
    trials = []
    grid = PLAN["frozen_head_grid"] if family == "D1_frozen" else PLAN["finetune_grid"]
    maximum = PLAN["frozen_max_epochs"] if family == "D1_frozen" else PLAN["finetune_max_epochs"]
    started = time.perf_counter()
    for cfg in grid:
        if family == "D1_frozen":
            model, trace, selected = train_head(E[ti], y[ti], cfg, SEED, maximum, device, (E[vi], y[vi]))
            with torch.inference_mode():
                vp = softmax(model(torch.tensor(E[vi], device=device)).cpu().numpy())
        else:
            model, trace, selected = train_finetune(images, y, ti, cfg, SEED, maximum, device, vi)
            vp = predict_network(model, images, vi, device)
        temperature, losses = choose_temperature(y[vi], vp)
        trials.append({"config": cfg, "trace": trace, "selected": selected,
                       "temperature": temperature, "temperature_nll": losses})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    chosen = max(range(len(trials)), key=lambda i: (trials[i]["selected"]["macro_f1"], -trials[i]["selected"]["nll"], -i))
    best = trials[chosen]
    if family == "D1_frozen":
        head, trace, _ = train_head(E[ids], y[ids], best["config"], SEED, best["selected"]["epoch"], device)
        model = official_model().to(device)
        model.classifier = head
    else:
        model, trace, _ = train_finetune(images, y, ids, best["config"], SEED, best["selected"]["epoch"], device)
    checkpoint.mkdir(parents=True, exist_ok=False)
    torch.save(model.cpu().state_dict(), checkpoint / "model.pth")
    card = {"backend": "torch_training_only", "name": checkpoint.name, "family": family, "seed": SEED,
            "class_order": CLASSES, "rgb_view_config": CFG.to_dict(), "temperature": best["temperature"],
            "low_conf_threshold": 0.0, "weights": weights, "manifest_sha256": info["manifest_sha256"],
            "config": best["config"], "epochs": best["selected"]["epoch"], "train_rows": int(cap.sum()),
            "train_groups": len(set(r["group"] for i, r in enumerate(rows) if cap[i])),
            "train_mask_sha256": hashlib.sha256("\n".join(rows[int(i)]["path"] for i in ids).encode()).hexdigest(),
            "final_selection": {"group_split": split, "training_paths": [rows[int(i)]["path"] for i in ti],
                                "validation_paths": [rows[int(i)]["path"] for i in vi], "trials": trials,
                                "selected": chosen, "full_refit_trace": trace},
            "train_seconds": time.perf_counter() - started,
            "limitations": ["temperature retained from inner validation after full-pool refit", "no frozen-test evaluation"]}
    dump(checkpoint / "model_card.json", card)
    return model, card


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--inner-groups", required=True, type=Path)
    parser.add_argument("--export-folds", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--family", choices=["all", "frozen", "finetune"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="device for final training only; all exports run on CPU")
    parser.add_argument("--export-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args(argv)
    if not args.export_folds and not args.export_final:
        parser.error("choose --export-folds and/or --export-final")
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision
    torch.set_num_threads(2)
    started = time.perf_counter()
    root = args.data_dir
    rows, images, info = prepare_views(root, workers=1)
    if len(rows) != 3395:
        raise ValueError("D1 export must preserve the complete 3395-image reference population")
    cap = video_cap_mask(np.array([r["group"] for r in rows]), np.array([r["path"] for r in rows]), seed=SEED)
    if int(cap.sum()) != 2393:
        raise ValueError("original R2 training cap changed")
    inner_groups, guard = load_inner_group_guard(rows, args.inner_groups)
    if guard.get("manifest_sha256") != info["manifest_sha256"]:
        raise ValueError("group guard and RGB cache manifest disagree")
    out = ML_DIR / "results" / f"d1_export_{args.export_id}"
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                               "torchvision": torchvision.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
                  "manifest_sha256": info["manifest_sha256"], "rgb_view_cache_key": info["key"],
                  "rgb_code_sha256": info["rgb_code_sha256"], "inner_group_guard": guard,
                  "export_tool_sha256": sha256(Path(__file__)), "device": args.device,
                  "run_dir": str(args.run_dir), "r2_training_rows": int(cap.sum()), "holdout_rows": len(rows)}
    dump(out / "predeclared_export_plan.json", {"plan": EXPORT_PLAN, "provenance": provenance,
                                                "family": args.family, "export_folds": args.export_folds, "export_final": args.export_final})
    families = ["D1_frozen", "D1_finetune"] if args.family == "all" else ["D1_" + args.family]
    source = np.array([r["source"] for r in rows])
    records = []
    if args.export_folds:
        for family in families:
            for fold in FOLDS:
                checkpoint = ML_DIR / "models" / "candidates" / args.run_dir.name / f"{family}_reference_{SEED}_{fold}"
                card = json.loads((checkpoint / "model_card.json").read_text(encoding="utf-8"))
                training, evaluation = np.flatnonzero(cap & (source != fold)), np.flatnonzero(source == fold)
                mask_hash = hashlib.sha256("\n".join(rows[int(i)]["path"] for i in training).encode()).hexdigest()
                if card["train_rows"] != len(training) or card["train_mask_sha256"] != mask_hash or card["manifest_sha256"] != info["manifest_sha256"]:
                    raise ValueError("fold checkpoint does not match unchanged original R2 training mask")
                model = official_model(pretrained=False)
                model.load_state_dict(torch.load(checkpoint / "model.pth", map_location="cpu", weights_only=True))
                result = export_pair(model, checkpoint, card, rows, images, training, evaluation,
                                     out / f"{family}_{fold}", args.export_id, EXPORT_PLAN["fold_calibration_max_images"])
                records.append({"family": family, "fold": fold, "result": result})
                dump(out / "progress.json", {"records": records, "seconds": time.perf_counter() - started})
                print(f"export complete {family} {fold}", flush=True)
                del model
    if args.export_final:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        for family in families:
            checkpoint = ML_DIR / "models" / "candidates" / args.run_dir.name / f"{family}_final_reference_{args.export_id}"
            model, card = final_train(family, rows, images, info, root, cap, inner_groups, device, checkpoint)
            training = np.flatnonzero(cap)
            evaluation = calibration_ids(rows, training, EXPORT_PLAN["final_calibration_images"])
            result = export_pair(model, checkpoint, card, rows, images, training, evaluation,
                                 out / f"{family}_final", args.export_id, EXPORT_PLAN["final_calibration_images"])
            records.append({"family": family, "fold": "final_training_diagnostic", "result": result})
            dump(out / "progress.json", {"records": records, "seconds": time.perf_counter() - started})
            print(f"final export complete {family}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Pooled/mean scores come only from held-out folds, never the final-training diagnostic.
    pooled = {}
    for family in families:
        fold_records = [r for r in records if r["family"] == family and r["fold"] in FOLDS]
        if len(fold_records) == len(FOLDS):
            all_predictions = []
            for fold in FOLDS:
                with (out / f"{family}_{fold}" / "predictions.csv").open(newline="", encoding="utf-8") as f:
                    all_predictions.extend(csv.DictReader(f))
            y = np.array([r["label"] for r in all_predictions])
            src = np.array([r["source"] for r in all_predictions])
            pooled[family] = {}
            for backend in ("torch", "fp32", "int8"):
                p = np.array([[float(r[f"{backend}_prob_{c}"]) for c in CLASSES] for r in all_predictions])
                pooled[family][backend] = {"metrics": fold_table(y, np.array(CLASSES)[p.argmax(1)], src), "calibration": calibration(y, p)}
            csv_write(out / f"{family}_pooled_predictions.csv", all_predictions)
    report = {"provenance": provenance, "records": records, "loso": pooled, "seconds": time.perf_counter() - started,
              "status": {"onnx_export_verified": True, "static_int8_evaluated": True, "pi_latency_verified": False,
                         "fw_e2e_verified": False, "frozen_test_evaluated": False}}
    dump(out / "report.json", report)
    print(json.dumps({"output": str(out), "records": len(records), "seconds": report["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
