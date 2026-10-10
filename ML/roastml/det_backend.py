"""Backend "det_b1_beans": detector หาเมล็ด (นับ + กรอบ) แล้วใช้สี Lab ของแต่ละกรอบตัดสินระดับคั่ว

model_card.json:
  {"backend": "det_b1_beans", "model_file": "model.json",          # B1 ระดับภาพ (fallback + label_mode broadcast)
   "detector_file": "bean.onnx", "det_config": {...},               # roastml.detector.DetConfig
   "bean_model_file": "bean_model.json" | ไม่ใส่ = ใช้ B1 ตัวเดียวกัน,
   "bean_label_mode": "per_box" | "image_prior" | "group" | "broadcast",
        # image_prior = prob ต่อกรอบ × (prob ระดับภาพ)^prior_weight · group = จัดกลุ่มความสว่าง (group_config) · broadcast = label ของภาพ
   "box_shape": "rect" | "ellipse" | "mask", "box_shrink": 0.2,     # พิกเซลที่ใช้คิดสีในแต่ละกรอบ (mask = พิกเซลของ segment ∩ กรอบ)
   "variants": {"balanced": {"label_th": "...", "prior_weight": 2.0}, "per_bean": {"label_th": "...", "prior_weight": 0.0}},
   "default_variant": "balanced",   # ตัวเลือกที่ผู้ใช้เลือกได้ต่อ request (เฉพาะ bean_label_mode = image_prior · ต่างกันแค่ prior_weight)
   "seg_config": {...}, "low_conf_threshold": ...}

ลำดับ: WB (ตัวเดียวกับ B1) → B1 ระดับภาพ → detector บนภาพ decode → feature ต่อกรอบ → softmax ต่อกรอบ → เสียงข้างมาก
ไม่เจอเมล็ด → n_beans = 0 + no_beans_detected และ label มาจาก B1 ระดับภาพ
ส่วนรายเมล็ดพัง → log แล้วถอยเป็นผล B1 ระดับภาพ (n_beans = null) — ไม่ทำให้ request ล้ม
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .contract import LABELS, BackendOutput
from .detector import BeanDetector, DetConfig
from .features import pixel_stats
from .linear_backend import LinearBackend
from .linear_model import LinearSoftmax
from .perbean import BOX_SHAPES, box_bean_features, fuse_image_prior, group_predictions, majority_probs
from .segment import prepare_image, segment

log = logging.getLogger("roastml")
LABEL_MODES = ("per_box", "image_prior", "group", "broadcast")
VALIDATION_NOTE = "experimental: detector trained on synthetic copy-paste scenes; pile/touching counts have no real GT"
DENSE_OVERLAP_SHARE = 0.3   # สัดส่วนกรอบที่ซ้อนกรอบอื่น > นี้ → ถือว่าแตะ/กอง → count_method = estimated
DENSE_IOU = 0.05


def overlap_share(boxes: np.ndarray) -> float:
    """สัดส่วนกรอบที่ IoU กับกรอบอื่นใด > DENSE_IOU (vectorized, n ≤ max_det)"""
    n = len(boxes)
    if n < 2:
        return 0.0
    b = boxes.astype(np.float64)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0), 0, None)
    inter = iw * ih
    area = b[:, 2] * b[:, 3]
    iou = inter / np.maximum(area[:, None] + area - inter, 1e-9)
    np.fill_diagonal(iou, 0.0)
    return float((iou.max(axis=1) > DENSE_IOU).mean())


class DetBeanBackend(LinearBackend):
    thread_safe = True

    def __init__(self, model_dir: Path, card: dict):
        if card.get("backend") != "det_b1_beans":
            raise ValueError("DetBeanBackend needs det_b1_beans")
        super().__init__(model_dir, {**card, "backend": "b1_linear"})
        self.kind, self._card = "det_b1_beans", card
        self.det_cfg = DetConfig.from_dict(card.get("det_config"))
        self.detector = BeanDetector(Path(model_dir) / card["detector_file"], self.det_cfg)
        self.bean_model = (LinearSoftmax.load(Path(model_dir) / card["bean_model_file"])
                           if card.get("bean_model_file") else self.model)
        if set(self.bean_model.classes) != set(LABELS):
            raise ValueError("bean model classes do not match contract")
        self.label_mode = card.get("bean_label_mode", "per_box")
        if self.label_mode not in LABEL_MODES:
            raise ValueError(f"bean_label_mode ไม่รองรับ: {self.label_mode!r}")
        self.box_shape, self.box_shrink = card.get("box_shape", "rect"), float(card.get("box_shrink", 0.2))
        if self.box_shape not in BOX_SHAPES or not 0 <= self.box_shrink < 0.5:
            raise ValueError("box_shape/box_shrink ไม่ถูกต้อง")
        self.prior_weight = float(card.get("prior_weight", 1.0))
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight ต้อง ≥ 0")
        self.variants, self.default_variant = self._read_variants(card)
        self.group_cfg = card.get("group_config")
        if self.label_mode == "group":   # ตรวจ config ตอน load ให้พังเร็ว
            group_predictions(np.zeros((1, 20)), np.ones((1, 3)) / 3, list(self.bean_model.classes), self.group_cfg)
        elif self.group_cfg is not None:
            raise ValueError("group_config ใช้ได้เฉพาะ bean_label_mode = group")
        self.enabled = os.environ.get("ROAST_BEANS", "1").strip() != "0"

    def _read_variants(self, card: dict) -> tuple[dict, str | None]:
        """ตรวจ variants ตอน load: ชื่อ a-z0-9_ · มีได้เฉพาะ label_th + prior_weight · default ต้องให้ค่าเท่ากับ card"""
        variants = card.get("variants")
        if variants is None:
            if card.get("default_variant") is not None:
                raise ValueError("default_variant ต้องมี variants")
            return {}, None
        if self.label_mode != "image_prior" or not isinstance(variants, dict) or not variants:
            raise ValueError("variants ใช้ได้เฉพาะ bean_label_mode = image_prior และต้องไม่ว่าง")
        out = {}
        for name, spec in variants.items():
            if not isinstance(name, str) or not name or len(name) > 32 or not name.replace("_", "").isalnum() or not name.isascii():
                raise ValueError(f"ชื่อ variant ไม่ถูกต้อง: {name!r}")
            if not isinstance(spec, dict) or set(spec) != {"label_th", "prior_weight"} or not isinstance(spec["label_th"], str):
                raise ValueError(f"variant {name}: ต้องมี label_th และ prior_weight เท่านั้น")
            weight = float(spec["prior_weight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"variant {name}: prior_weight ต้อง ≥ 0")
            out[name] = {"label_th": spec["label_th"], "prior_weight": weight}
        default = card.get("default_variant")
        if default not in out or out[default]["prior_weight"] != self.prior_weight:
            raise ValueError("default_variant ต้องอยู่ใน variants และมี prior_weight เท่ากับของ card")
        return out, default

    def predict_profile(self, img, *, source: str = "", roi: str = "", variant: str | None = None):
        start = time.perf_counter()
        if variant is not None and variant not in self.variants:
            raise ValueError(f"unknown variant: {variant!r}")
        prior_weight = self.variants[variant]["prior_weight"] if variant is not None else self.prior_weight
        offset = np.zeros(2, np.int64)
        if source == "agtron":
            from .rgb_views import crop_roi
            rgb = crop_roi(img, roi)
            x0, y0, _, _ = map(int, roi.split())
            offset = np.rint([x0 * img.image.width / img.orig_size[0], y0 * img.image.height / img.orig_size[1]]).astype(np.int64)
        else:
            if roi:
                raise ValueError("ROI metadata is only supported for Agtron")
            rgb = np.asarray(img.image)
        if min(rgb.shape[:2]) < 8:
            height, width = rgb.shape[:2]
            scale = 8 / min(height, width)
            rgb = np.asarray(Image.fromarray(rgb).resize((max(8, round(width * scale)), max(8, round(height * scale))),
                                                         Image.Resampling.NEAREST))
        prep = prepare_image(rgb, self.cfg, allow_wb=source != "agtron")
        wb_end = time.perf_counter()
        seg = segment(rgb, self.cfg, find_beans=source != "agtron", prepared=prep)
        seg_end = time.perf_counter()
        image_p = self.model.proba(pixel_stats(seg.lab[seg.pixel_mask]))[0]
        fallback_end = time.perf_counter()
        image_probs = {c: float(image_p[i]) for i, c in enumerate(self.model.classes)}
        out = BackendOutput(probs=dict(image_probs), warnings=["no_beans_detected"] if seg.mode == "full" else [])
        stages = {"WB": (wb_end - start) * 1000, "segment": (seg_end - wb_end) * 1000,
                  "fallback": (fallback_end - seg_end) * 1000, "count": 0.0, "features": 0.0,
                  "classify": 0.0, "group": 0.0, "n_views": 1}
        if not self.enabled:
            return out, stages
        try:
            t = time.perf_counter()
            boxes, scores = self.detector.detect(rgb)
            tc = time.perf_counter()
            stages["count"] = (tc - t) * 1000
            n = len(boxes)
            out.warnings = []
            if not n:
                out.n_beans, out.count_method, out.warnings = 0, "exact", ["no_beans_detected"]
                return out, stages
            classes = list(self.bean_model.classes)
            if self.label_mode == "broadcast":
                # ทุกเมล็ดได้ label ของภาพ · confidence ของภาพคงเป็นของ B1 จริง
                image_idx = int(np.argmax([image_probs[c] for c in classes]))
                labels = np.full(n, image_idx, np.int64)
                conf = np.full(n, image_probs[classes[image_idx]])
                tf = tp = time.perf_counter()
                out.proportions = {c: float(i == image_idx) for i, c in enumerate(classes)}
            else:
                use_mask = self.box_shape == "mask" and seg.mode != "full"   # full = segment ไม่เห็นเมล็ด → mask คือทั้งภาพ
                X = box_bean_features(prep.wb.lab, boxes.astype(np.float64) * prep.scale, self.cfg, shrink=self.box_shrink,
                                      shape=self.box_shape if use_mask or self.box_shape != "mask" else "rect",
                                      mask=seg.pixel_mask if use_mask else None)
                tf = time.perf_counter()
                P = self.bean_model.proba(X)
                if self.label_mode == "image_prior":
                    if list(self.model.classes) != classes:
                        raise ValueError("image/bean model class order differs")
                    P = fuse_image_prior(P, image_p, prior_weight)
                tp = time.perf_counter()
                labels, conf = P.argmax(1), P.max(1)
                if self.label_mode == "group":
                    labels, conf, _ = group_predictions(X, P, classes, self.group_cfg)
                out.probs, out.proportions = majority_probs(labels, P, classes)
            out.n_beans = n
            dense = n >= self.det_cfg.max_det or overlap_share(boxes) > DENSE_OVERLAP_SHARE
            out.count_method = "estimated" if dense else "exact"
            boxes = boxes.copy()
            boxes[:, :2] += offset
            out.beans = [{"bbox": boxes[i].tolist(), "label": classes[int(labels[i])], "conf": float(conf[i])}
                         for i in range(n)]
            if np.count_nonzero(np.bincount(labels, minlength=len(classes))) > 1:
                out.warnings.append("mixed_roast")
            if dense:
                out.warnings += ["bean_count_estimated", "count_visible_only"]
            stages.update(features=(tf - tc) * 1000, classify=(tp - tf) * 1000, group=(time.perf_counter() - tp) * 1000)
        except Exception:
            log.exception("detector bean path failed; returning B1 fallback")
            out = BackendOutput(probs=dict(image_probs), warnings=["no_beans_detected"])
        return out, stages

    def info(self):
        return super().info() | {"backend": self.kind, "beans_enabled": self.enabled, "bean_label_mode": self.label_mode,
                                "group_config": self.group_cfg, "prior_weight": self.prior_weight,
                                "variants": self.variants, "default_variant": self.default_variant,
                                "box_shape": self.box_shape, "box_shrink": self.box_shrink,
                                "det_config": self.det_cfg.to_dict(), "detector_input": self.detector.size,
                                "bean_validation": VALIDATION_NOTE,
                                "n_beans_scope": "visible beans found by the detector; occluded beans are not counted"}
