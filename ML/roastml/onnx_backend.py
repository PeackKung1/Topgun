"""ONNX RGB runtime using the same texture-preserving views as training."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .contract import LABELS, BackendOutput
from .decode import DecodedImage
from .rgb_views import RGBViewConfig, build_rgb_views, pool_view_probabilities
from .tabular_backend import model_file_path, softmax


class OnnxBackend:
    thread_safe = True  # ORT session calls are read-only; all view arrays are local.

    def __init__(self, model_dir: Path, card: dict):
        import onnxruntime as ort

        self.name = str(card.get("name") or "onnx_rgb")
        self.classes = list(card["class_order"])
        if len(self.classes) != 3 or set(self.classes) != set(LABELS):
            raise ValueError("class_order must contain each contract class exactly once")
        self.config = RGBViewConfig.from_dict(card["rgb_view_config"]).to_dict()
        self.output_kind = card.get("output_kind", "logits")
        if self.output_kind not in ("logits", "probabilities"):
            raise ValueError("output_kind must be logits or probabilities")
        self.temperature = float(card.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(card.get("intra_op_num_threads", 1))
        opts.inter_op_num_threads = int(card.get("inter_op_num_threads", 1))
        if opts.intra_op_num_threads < 1 or opts.inter_op_num_threads < 1:
            raise ValueError("runtime thread counts must be positive")
        artifact = model_file_path(model_dir, card.get("model_file", "model.onnx"))
        try:
            self.session = ort.InferenceSession(str(artifact), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:
            # ORT's InvalidProtobuf/Fail are direct Exception subclasses, so the
            # API's usual ValueError/RuntimeError catch alone cannot handle them.
            raise ValueError(f"ONNX session initialization failed: {type(exc).__name__}: {exc}") from exc
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError("ONNX model must have exactly one RGB input")
        self.input_name = card.get("input_name", inputs[0].name)
        self.output_name = card.get("output_name", outputs[0].name)
        if self.input_name != inputs[0].name or self.output_name not in [o.name for o in outputs]:
            raise ValueError("ONNX input/output name does not match card")
        shape = inputs[0].shape
        if inputs[0].type != "tensor(float)" or len(shape) != 4 or shape[1] != 3:
            raise ValueError("ONNX input must be float32 NCHW RGB")
        self.batch = shape[0] if isinstance(shape[0], int) else None
        if self.batch not in (None, 1):
            raise ValueError("ONNX input batch must be dynamic or one")
        self._card = dict(card)
        self.ort_version = ort.__version__

    def predict(self, img: DecodedImage) -> BackendOutput:
        return self.predict_profile(img)[0]

    def predict_with_metadata(self, img: DecodedImage, *, source="", roi="") -> BackendOutput:
        return self.predict_profile(img, source=source, roi=roi)[0]

    def predict_profile(self, img: DecodedImage, *, source="", roi="") -> tuple[BackendOutput, dict]:
        t0 = time.perf_counter()
        if roi and source != "agtron":
            raise ValueError("runtime ROI metadata is only supported for Agtron")
        X = build_rgb_views(img, self.config, source=source, roi=roi)
        t1 = time.perf_counter()
        if self.batch == 1:
            values = np.concatenate([self.session.run([self.output_name], {self.input_name: x[None]})[0] for x in X])
        else:
            values = np.asarray(self.session.run([self.output_name], {self.input_name: X})[0])
        if values.shape != (len(X), 3) or not np.isfinite(values).all():
            raise ValueError("ONNX output must be finite N×3")
        if self.output_kind == "probabilities":
            if np.any(values < 0) or np.any(values.sum(1) <= 0):
                raise ValueError("ONNX probabilities must be nonnegative and nonzero")
            logits = np.log(np.maximum(values / values.sum(1, keepdims=True), 1e-12))
        else:
            logits = values.astype(float)
        P = softmax(logits / self.temperature)
        p = pool_view_probabilities(P, self.config["view_weights"])
        t2 = time.perf_counter()
        return BackendOutput(probs={c: float(p[i]) for i, c in enumerate(self.classes)}), {
            "views": (t1 - t0) * 1000, "model": (t2 - t1) * 1000, "n_views": len(X),
        }

    def warmup(self, img):
        self.predict(img)

    def info(self):
        return {"backend": "onnx_rgb", "name": self.name, "classes": self.classes,
                "rgb_view_config": self.config, "output_kind": self.output_kind,
                "temperature": self.temperature, "onnxruntime": self.ort_version,
                "providers": self.session.get_providers()}
