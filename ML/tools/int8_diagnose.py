"""Diagnose static INT8 accuracy loss of an exported D1 fold (trainval only).

export_d1's default QDQ/MinMax INT8 flipped up to 99% of labels relative to
FP32. This tool re-quantizes the SAME FP32 graph with the SAME training-only
calibration images under a few predeclared quantizer settings and measures
label agreement / probability error against FP32 on that fold's holdout.
Agreement with FP32 is the target (not holdout accuracy), so nothing here
selects a model on holdout labels. Artifacts go to a scratch directory under
the candidate's exports; nothing in current/ or reference is touched.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from tools.export_d1 import EXPORT_PLAN, normalise_pixels, ort_session, pool_logits
from tools.train_baseline import CLASSES
from tools.train_d1 import ML_DIR, dump, prepare_views, sha256

VARIANTS = {
    "default_minmax_u8s8": {"calibrate": "MinMax", "activation": "QUInt8", "per_channel": True, "reduce_range": False},
    "reduce_range": {"calibrate": "MinMax", "activation": "QUInt8", "per_channel": True, "reduce_range": True},
    "percentile_99_99": {"calibrate": "Percentile", "activation": "QUInt8", "per_channel": True, "reduce_range": False},
    "entropy": {"calibrate": "Entropy", "activation": "QUInt8", "per_channel": True, "reduce_range": False},
    "s8s8_symmetric": {"calibrate": "MinMax", "activation": "QInt8", "per_channel": True, "reduce_range": False},
    "weights_only_dynamic": {"dynamic": True},
}


def quantize(fp32: Path, out: Path, images, ids, cfg, v: dict) -> float:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    started = time.perf_counter()
    pre = out.parent / "pre.onnx"
    if not pre.exists():
        quant_pre_process(fp32, pre, skip_optimization=False, skip_symbolic_shape=False, skip_onnx_shape=False, auto_merge=True)
    if v.get("dynamic"):
        quantize_dynamic(pre, out, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"])
        return time.perf_counter() - started

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = (({"rgb": view[None]}) for i in ids for view in normalise_pixels(images[int(i)][None], cfg)[0])

        def get_next(self):
            return next(self.it, None)

    extra = {"ActivationSymmetric": v["activation"] == "QInt8", "WeightSymmetric": True, "DedicatedQDQPair": True}
    if v["calibrate"] == "Percentile":
        extra["CalibPercentile"] = 99.99
    quantize_static(pre, out, Reader(), quant_format=QuantFormat.QDQ, per_channel=v["per_channel"],
                    reduce_range=v["reduce_range"], activation_type=getattr(QuantType, v["activation"]),
                    weight_type=QuantType.QInt8, op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                    calibrate_method=getattr(CalibrationMethod, v["calibrate"]),
                    calibration_providers=["CPUExecutionProvider"], extra_options=extra)
    return time.perf_counter() - started


def run_probs(path: Path, images, ids, cfg, temperature) -> tuple[np.ndarray, float]:
    session = ort_session(path)
    out, t = [], 0.0
    for s in range(0, len(ids), 16):
        x = normalise_pixels(images[ids[s:s + 16]], cfg)
        b, v, c, h, w = x.shape
        t0 = time.perf_counter()
        z = session.run(["logits"], {"rgb": np.ascontiguousarray(x.reshape(b * v, c, h, w))})[0]
        t += time.perf_counter() - t0
        out.append(pool_logits(z.reshape(b, v, 3), temperature, cfg["view_weights"]))
    return np.concatenate(out), t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--export-results", type=Path, required=True, help="ML/results/d1_export_<id>")
    ap.add_argument("--folds", nargs="+", default=["rf_boos", "ontoum224"])
    ap.add_argument("--out", type=Path, default=ML_DIR / "results" / "d1_int8_diagnose_20261008")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "plan.json", {"variants": VARIANTS, "folds": args.folds, "target": "agreement with FP32 on holdout",
                                  "calibration": "the export's own training-only calibration ids", "export_plan": EXPORT_PLAN})
    import onnxruntime
    rows, images, info = prepare_views(args.data_dir.resolve(), workers=1)
    source = np.array([r["source"] for r in rows])
    summary = {"versions": {"python": platform.python_version(), "onnxruntime": onnxruntime.__version__},
               "cpu": platform.processor(), "folds": {}}
    for fold in args.folds:
        export = json.loads((args.export_results / f"D1_finetune_{fold}" / "summary.json").read_text(encoding="utf-8"))
        fp32 = Path([a for a in export["artifacts"] if a["precision"] == "fp32"][0]["directory"]) / "model.onnx"
        card = json.loads((fp32.parent / "model_card.json").read_text(encoding="utf-8"))
        cfg, temperature = card["rgb_view_config"], float(card["temperature"])
        calib = np.asarray(export["calibration"]["indices"], int)
        holdout = np.flatnonzero(source == fold)
        if set(calib) & set(holdout):
            raise ValueError("calibration overlaps holdout")
        ref, t_ref = run_probs(fp32, images, holdout, cfg, temperature)
        work = fp32.parent.parent / "int8_diagnose"
        work.mkdir(exist_ok=True)
        res = {"fp32_sha256": sha256(fp32), "fp32_seconds": t_ref, "n_holdout": len(holdout), "variants": {}}
        for name, v in VARIANTS.items():
            path = work / f"{name}.onnx"
            if path.exists():
                path.unlink()  # scratch artifact owned by this tool only
            q_seconds = quantize(fp32, path, images, calib, cfg, v)
            p, t = run_probs(path, images, holdout, cfg, temperature)
            agree = float((p.argmax(1) == ref.argmax(1)).mean())
            res["variants"][name] = {"label_agreement_vs_fp32": agree, "max_abs_prob_diff": float(np.abs(p - ref).max()),
                                     "mean_abs_prob_diff": float(np.abs(p - ref).mean()), "inference_seconds": t,
                                     "quantize_seconds": q_seconds, "bytes": path.stat().st_size, "sha256": sha256(path)}
            print(fold, name, f"agree={agree:.4f}", f"t={t:.1f}s", flush=True)
        summary["folds"][fold] = res
        dump(args.out / "report.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
