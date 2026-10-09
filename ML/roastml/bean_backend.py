"""B1 ระดับภาพ + นับเมล็ด/ระดับคั่วรายเมล็ด (ส่วนเสริม) · backend "b1_linear_beans"

status/label/probs/confidence = ผล B1 เดิมทุกค่าเสมอ (ขั้นตอนเดียวกับ LinearBackend.predict_profile · มี test parity)
ส่วนเสริม (ยังไม่ผ่านการตรวจ): n_beans, proportions, beans จาก split_beans + B1 ตัวเดียวกันจัดคลาสพิกเซลรายเมล็ด
- ไม่ส่ง warning mixed_roast และไม่ override label (ภาคผนวกเฟส 1: G4(b) false mixed ≈ 19% ตั้งแต่เฟส 0)
- ปิดได้ด้วย env ROAST_BEANS=0 (อ่านตอนโหลด) → ผลเหมือน b1_linear ทุก field
- G5: ขั้นนับเมล็ดล้ม/เกิน budget_ms → ผล B1 เดิม + n_beans=None, proportions=None, beans=[]
- per-bean acc ≈ 0.62 (in-sample บน rf_boos GT box) · ตรวจแล้วเฉพาะภาพเมล็ดน้อยบนพื้นขาว
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .beans import MAX_BLOBS, count_proportions, split_beans
from .contract import BackendOutput
from .decode import DecodedImage
from .features import pixel_stats
from .linear_backend import LinearBackend
from .segment import segment

log = logging.getLogger("roastml")

DEFAULT_BEAN_COUNTING = {"budget_ms": 100.0, "max_blobs": MAX_BLOBS}
VALIDATION_NOTE = ("experimental: edge-contour bean counts are estimates and have not been validated against dense-pile ground truth; "
                   "per-bean accuracy ~0.62 in-sample on rf_boos GT boxes; no mixed_roast warning and no label override")


class BeanLinearBackend(LinearBackend):
    thread_safe = True  # ไม่มี state ที่เปลี่ยนระหว่าง predict

    def __init__(self, model_dir: Path, card: dict[str, Any]):
        if card.get("backend") != "b1_linear_beans":
            raise ValueError("BeanLinearBackend ต้องใช้ backend b1_linear_beans")
        super().__init__(model_dir, {**card, "backend": "b1_linear"})
        self.kind = "b1_linear_beans"
        self._card = card
        bc = {**DEFAULT_BEAN_COUNTING, **(card.get("bean_counting") or {})}
        unknown = set(bc) - set(DEFAULT_BEAN_COUNTING)
        if unknown:
            raise ValueError(f"bean_counting ไม่รู้จัก key: {sorted(unknown)} (ไม่มี mixed/override)")
        self.enabled = os.environ.get("ROAST_BEANS", "1").strip() != "0"
        self.budget_ms = float(bc["budget_ms"])
        self.max_blobs = int(bc["max_blobs"])
        if not (0 < self.budget_ms <= 1000 and 1 <= self.max_blobs <= MAX_BLOBS):
            raise ValueError("bean_counting budget_ms/max_blobs ไม่ถูกต้อง")

    # --- เส้นทาง B1 (ขั้นตอนเดียวกับ LinearBackend.predict_profile) ---
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
        out = BackendOutput(probs=probs)
        # --- ส่วนเสริม: ล้มได้ แต่ห้ามกระทบผล B1 (G5) · แก้ได้แค่ n_beans/proportions/beans ---
        try:
            if self.enabled:
                self._add_beans(out, seg, rgb)
        except Exception:
            log.exception("bean counting failed; returning image-level B1 result")
            out.n_beans, out.proportions, out.beans = None, None, []
        if seg.mode == "full" and out.n_beans is None:
            out.warnings.append("no_beans_detected")
        t4 = time.perf_counter()
        return out, {"views": (t1 - t0) * 1000, "features": (t2 - t1) * 1000,
                     "model": (t3 - t2) * 1000, "beans": (t4 - t3) * 1000, "n_views": 1}

    def _add_beans(self, out: BackendOutput, seg, seg_rgb: np.ndarray) -> None:
        t0 = time.perf_counter()
        beans = split_beans(seg, self.cfg, rgb=seg_rgb)
        if not beans:
            return  # ภาพไม่เห็นขอบเมล็ดชัดพอ → n_beans None
        if any(bean.estimated for bean in beans):
            out.warnings.append("bean_count_estimated")
        if len(beans) > self.max_blobs:
            # G3 cap: ไม่จัดคลาสรายเมล็ด · contract กำหนด n_beans == len(beans) จึงคืน null ทั้งชุด
            log.info("bean count %d exceeds cap %d; bean fields left null", len(beans), self.max_blobs)
            return
        X = np.stack([pixel_stats(b.lab_pixels(seg.lab)) for b in beans])
        P = self.model.proba(X)
        if (time.perf_counter() - t0) * 1000 > self.budget_ms:
            log.warning("bean counting exceeded %.0f ms budget; dropping per-bean result", self.budget_ms)
            return
        classes = list(self.model.classes)
        labels = [classes[int(i)] for i in P.argmax(1)]
        proportions = count_proportions(labels, classes)
        out.n_beans = len(beans)
        out.proportions = proportions
        out.beans = [{"bbox": b.bbox, "label": lab, "conf": float(P[i].max())}
                     for i, (b, lab) in enumerate(zip(beans, labels))]

    def info(self) -> dict[str, Any]:
        base = super().info()
        base.update({"backend": "b1_linear_beans", "beans_enabled": self.enabled,
                     "bean_budget_ms": self.budget_ms, "max_blobs": self.max_blobs,
                     "n_beans_scope": "edge contours estimate visible beans in beans/pile/full modes; occluded beans cannot be inferred",
                     "bean_validation": VALIDATION_NOTE})
        return base
