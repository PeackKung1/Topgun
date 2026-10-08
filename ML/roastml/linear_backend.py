"""Backend ของ baseline B0/B1 (feature สี Lab + threshold / softmax) — numpy + opencv ล้วน ใช้บน Pi ได้

โฟลเดอร์โมเดล (สร้างโดย tools/train_baseline.py → ML/models/current/, gitignored):
  model_card.json : {"backend": "b1_linear" | "b0_threshold", "name", "model_file", "seg_config", "low_conf_threshold", ...}
  model.json      : spec ของ roastml.linear_model (LinearSoftmax / Threshold3)

ลำดับต่อภาพ: DecodedImage (≤1600 px, หมุน EXIF แล้ว) → segment/smart crop (≤800 px) → feature Lab → prob 3 คลาส
segment หา contour เมล็ดไม่เจอ (โหมด full) → ยังตอบเสมอ + warning "no_beans_detected"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .contract import LABELS, BackendOutput
from .decode import DecodedImage
from .features import pixel_stats
from .linear_model import LinearSoftmax, Threshold3
from .segment import SegConfig, segment

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
        return self.predict_profile(img)[0]

    def predict_with_metadata(self, img: DecodedImage, *, source: str = "", roi: str = "") -> BackendOutput:
        return self.predict_profile(img, source=source, roi=roi)[0]

    def predict_profile(self, img: DecodedImage, *, source: str = "", roi: str = "") -> tuple[BackendOutput, dict]:
        t0 = time.perf_counter()
        if source == "agtron":
            from .rgb_views import crop_roi
            rgb = crop_roi(img, roi)
        else:
            if roi:
                raise ValueError("ROI metadata is only supported for Agtron")
            rgb = np.asarray(img.image)
        if min(rgb.shape[:2]) < 8:
            # A decoded tiny image is valid input. Nearest expansion preserves
            # its colours and aspect ratio, then the usual full-view fallback
            # can still return a label. Normal trainval images are unaffected.
            height, width = rgb.shape[:2]
            scale = 8 / min(height, width)
            rgb = np.asarray(Image.fromarray(rgb).resize((max(8, round(width * scale)),
                                                         max(8, round(height * scale))), Image.Resampling.NEAREST))
        seg = segment(rgb, self.cfg, find_beans=source != "agtron")
        t1 = time.perf_counter()
        x = pixel_stats(seg.lab[seg.pixel_mask])
        t2 = time.perf_counter()
        p = self.model.proba(x)[0]
        t3 = time.perf_counter()
        probs = {c: float(p[i]) for i, c in enumerate(self.model.classes)}
        warnings = ["no_beans_detected"] if seg.mode == "full" else []
        return BackendOutput(probs=probs, warnings=warnings), {
            "views": (t1 - t0) * 1000, "features": (t2 - t1) * 1000,
            "model": (t3 - t2) * 1000, "n_views": 1,
        }

    def warmup(self, img: DecodedImage) -> None:
        self.predict(img)

    def info(self) -> dict[str, Any]:
        keys = ("backend", "name", "feature_set", "C", "created", "loso_mean_macro_f1", "low_conf_threshold")
        return {k: self._card.get(k) for k in keys} | {"classes": list(self.model.classes)}
