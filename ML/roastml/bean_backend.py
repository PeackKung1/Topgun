"""Visible bean counter and per-bean LR; exact FW keys, image majority label."""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .contract import BackendOutput, LABELS
from .counter import CountConfig, count_beans
from .features import pixel_stats
from .linear_backend import LinearBackend
from .linear_model import LinearSoftmax
from .perbean import count_features, group_predictions, majority_probs
from .segment import prepare_image, segment

log = logging.getLogger("roastml")
VALIDATION_NOTE = "experimental: counted pile/touching dev and real Pi latency gates remain required"

class BeanLinearBackend(LinearBackend):
    thread_safe = True

    def __init__(self, model_dir: Path, card: dict):
        if card.get("backend") != "b1_linear_beans":
            raise ValueError("BeanLinearBackend needs b1_linear_beans")
        super().__init__(model_dir, {**card, "backend": "b1_linear"})
        self.kind, self._card = "b1_linear_beans", card
        # Legacy cards load; the old 150 cap and time-based drop are obsolete.
        if set(card.get("bean_counting") or {})-{"budget_ms", "max_blobs"}:
            raise ValueError("unknown legacy bean_counting key")
        self.count_cfg = CountConfig.from_dict(card.get("count_config"))
        self.bean_model = LinearSoftmax.load(Path(model_dir)/card["bean_model_file"]) if card.get("bean_model_file") else self.model
        if set(self.bean_model.classes) != set(LABELS):
            raise ValueError("bean model classes do not match contract")
        self.group_cfg = card.get("group_config")
        if self.group_cfg is not None:
            group_predictions(np.zeros((1, 20)), np.ones((1, 3))/3, list(self.bean_model.classes), self.group_cfg)
        self.enabled = os.environ.get("ROAST_BEANS", "1").strip() != "0"

    def predict_profile(self, img, *, source: str = "", roi: str = ""):
        start = time.perf_counter()
        if source == "agtron":
            from .rgb_views import crop_roi
            rgb = crop_roi(img, roi)
        else:
            if roi:
                raise ValueError("ROI metadata is only supported for Agtron")
            rgb = np.asarray(img.image)
        if min(rgb.shape[:2]) < 8:
            height, width = rgb.shape[:2]
            scale = 8/min(height, width)
            rgb = np.asarray(Image.fromarray(rgb).resize((max(8, round(width*scale)), max(8, round(height*scale))), Image.Resampling.NEAREST))
        prep = prepare_image(rgb, self.cfg, allow_wb=source != "agtron")
        wb_end = time.perf_counter()
        seg = segment(rgb, self.cfg, find_beans=source != "agtron", prepared=prep)
        seg_end = time.perf_counter()
        image_x = pixel_stats(seg.lab[seg.pixel_mask])
        image_p = self.model.proba(image_x)[0]
        fallback_end = time.perf_counter()
        out = BackendOutput(probs={c: float(image_p[i]) for i,c in enumerate(self.model.classes)},
                            warnings=["no_beans_detected"] if seg.mode == "full" else [])
        stages = {"WB": (wb_end-start)*1000, "segment": (seg_end-wb_end)*1000,
                  "fallback": (fallback_end-seg_end)*1000, "count": 0.0, "features": 0.0,
                  "classify": 0.0, "group": 0.0, "n_views": 1}
        if not self.enabled:
            return out, stages
        try:
            t = time.perf_counter()
            result = count_beans(rgb, self.count_cfg, self.cfg, force_pile=source == "agtron", prepared=prep)
            tc = time.perf_counter()
            stages["count"] = (tc-t)*1000
            stages["count_method"] = result.method  # profiling only
            out.warnings = []
            if not result.n:
                out.n_beans = 0
                out.count_method = "exact"
                out.warnings = ["no_beans_detected"]
                return out, stages
            X = count_features(result)
            tf = time.perf_counter()
            P = self.bean_model.proba(X)
            tp = time.perf_counter()
            classes = list(self.bean_model.classes)
            labels, conf = P.argmax(1), P.max(1)
            if self.group_cfg is not None:
                labels, conf, _ = group_predictions(X, P, classes, self.group_cfg)
            out.probs, out.proportions = majority_probs(labels, P, classes)
            out.n_beans = result.n
            out.count_method = "estimated" if result.method in ("estimated", "split") else "exact"
            # Dataset ROIs are cropped in decoded coordinates. Public boxes
            # still reference the complete decoded image (as camera requests do).
            if source == "agtron":
                x0, y0, _, _ = map(int, roi.split())
                offset = np.rint([x0*img.image.width/img.orig_size[0], y0*img.image.height/img.orig_size[1]]).astype(int)
                result.bboxes[:, :2] += offset
            out.beans = [{"bbox": result.bboxes[i].tolist(), "label": classes[int(labels[i])], "conf": float(conf[i])}
                         for i in range(result.n)]
            if np.count_nonzero(np.bincount(labels, minlength=3)) > 1:
                out.warnings.append("mixed_roast")
            if out.count_method == "estimated":
                out.warnings.append("bean_count_estimated")
            stages["count_method"] = out.count_method
            if result.scene in ("touching", "pile") or "max_beans_cap" in result.notes or "visible_area_filter" in result.notes:
                out.warnings.append("count_visible_only")
            stages.update(features=(tf-tc)*1000, classify=(tp-tf)*1000, group=(time.perf_counter()-tp)*1000)
        except Exception:
            log.exception("bean path failed; returning B1 fallback")
            out = BackendOutput(probs={c: float(image_p[i]) for i,c in enumerate(self.model.classes)}, warnings=["no_beans_detected"])
        return out, stages

    def info(self):
        return super().info() | {"backend": self.kind, "beans_enabled": self.enabled,
                                "count_config": self.count_cfg.to_dict(), "group_config": self.group_cfg,
                                "bean_validation": VALIDATION_NOTE,
                                "n_beans_scope": "visible regions; occlusions are not reconstructed"}
