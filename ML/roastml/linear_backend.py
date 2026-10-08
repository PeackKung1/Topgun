"""Backend ของ baseline B0/B1 (feature สี Lab + threshold / softmax) — numpy + opencv ล้วน ใช้บน Pi ได้

โฟลเดอร์โมเดล (สร้างโดย tools/train_baseline.py → ML/models/current/, gitignored):
  model_card.json : {"backend": "b1_linear" | "b0_threshold", "name", "model_file", "seg_config", "low_conf_threshold", ...}
  model.json      : spec ของ roastml.linear_model (LinearSoftmax / Threshold3)

ลำดับต่อภาพ: DecodedImage (≤1600 px, หมุน EXIF แล้ว) → segment/smart crop (≤800 px) → feature Lab → prob 3 คลาส
segment หา contour เมล็ดไม่เจอ (โหมด full) → ยังตอบเสมอ + warning "no_beans_detected"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import LABELS, BackendOutput
from .decode import DecodedImage
from .features import image_features
from .linear_model import LinearSoftmax, Threshold3
from .segment import SegConfig

BACKEND_TYPES = {"b1_linear": LinearSoftmax, "b0_threshold": Threshold3}


class LinearBackend:
    thread_safe = True  # ไม่มี state ที่เปลี่ยนระหว่าง predict

    def __init__(self, model_dir: Path, card: dict[str, Any]):
        kind = card.get("backend")
        if kind not in BACKEND_TYPES:
            raise ValueError(f"backend ไม่รองรับ: {kind!r}")
        spec_path = Path(model_dir) / card.get("model_file", "model.json")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.model = BACKEND_TYPES[kind](spec)
        if set(self.model.classes) != set(LABELS):
            raise ValueError(f"คลาสของโมเดล {self.model.classes} ไม่ตรง contract {LABELS}")
        self.cfg = SegConfig.from_dict(card.get("seg_config"))
        self.name = str(card.get("name") or kind)
        self.kind = kind
        self._card = card

    def predict(self, img: DecodedImage) -> BackendOutput:
        rgb = np.asarray(img.image)
        f = image_features(rgb, self.cfg)
        p = self.model.proba(f.x)[0]
        probs = {c: float(p[i]) for i, c in enumerate(self.model.classes)}
        warnings = ["no_beans_detected"] if f.seg.mode == "full" else []
        return BackendOutput(probs=probs, warnings=warnings)

    def warmup(self, img: DecodedImage) -> None:
        self.predict(img)

    def info(self) -> dict[str, Any]:
        keys = ("backend", "name", "feature_set", "C", "created", "loso_mean_macro_f1", "low_conf_threshold")
        return {k: self._card.get(k) for k in keys} | {"classes": list(self.model.classes)}
