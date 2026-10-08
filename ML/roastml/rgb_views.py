"""Texture-preserving RGB views shared by D1 training, evaluation and ONNX.

Annotation crops are never part of ordinary inference. Agtron is a special
dataset diagnostic: its mandatory ROI is validated AFTER EXIF orientation.
No absolute-colour label rules, histogram equalisation or hue transforms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .decode import DecodedImage
from .segment import SegConfig, background_white_balance, segment, to_work


@dataclass(frozen=True)
class RGBViewConfig:
    size: int = 224
    views: tuple[str, ...] = ("full", "smart")
    view_weights: tuple[float, ...] = (0.5, 0.5)
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    pad_rgb: tuple[int, ...] = (124, 116, 104)
    seg_config: dict = field(default_factory=lambda: SegConfig().to_dict())

    def __post_init__(self):
        if not isinstance(self.size, int) or not 32 <= self.size <= 1024:
            raise ValueError("RGB size must be an integer in [32,1024]")
        if not self.views or any(v not in ("full", "smart") for v in self.views):
            raise ValueError("views must contain full/smart")
        weights = np.asarray(self.view_weights, dtype=float)
        if len(weights) != len(self.views) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("invalid view weights")
        if len(self.mean) != 3 or len(self.std) != 3 or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or min(self.std) <= 0:
            raise ValueError("invalid RGB normalisation")
        if len(self.pad_rgb) != 3 or any(not 0 <= x <= 255 for x in self.pad_rgb):
            raise ValueError("invalid padding RGB")

    @classmethod
    def from_dict(cls, value: dict | None):
        value = value or {}
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown RGB config: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def crop_roi(decoded: DecodedImage, roi: str) -> np.ndarray:
    """Validate original, oriented coordinates; map x/y with actual resize.

    Independent scales avoid one-pixel rounding errors on portrait JPEGs.
    An invalid mandatory crop must fail, never silently expose Agtron labels.
    """
    try:
        values = [int(v) for v in roi.split()]
    except (ValueError, AttributeError) as exc:
        raise ValueError("ROI must contain four integer coordinates") from exc
    if len(values) != 4:
        raise ValueError("ROI must contain four integer coordinates")
    x0, y0, x1, y1 = values
    ow, oh = decoded.orig_size
    if not (0 <= x0 < x1 <= ow and 0 <= y0 < y1 <= oh):
        raise ValueError(f"ROI outside oriented image: {values}, {decoded.orig_size}")
    w, h = decoded.image.size
    xa, xb = round(x0 * w / ow), round(x1 * w / ow)
    ya, yb = round(y0 * h / oh), round(y1 * h / oh)
    if not (0 <= xa < xb <= w and 0 <= ya < yb <= h):
        raise ValueError("ROI empty after decode resize")
    return np.asarray(decoded.image)[ya:yb, xa:xb].copy()


def resize_pad(rgb: np.ndarray, size: int, pad_rgb=(124, 116, 104)) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("empty or invalid RGB view")
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = np.empty((size, size, 3), np.uint8)
    out[:] = pad_rgb
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = (size - nw) // 2, (size - nh) // 2
    out[y:y+nh, x:x+nw] = resized
    return out


def rgb_view_images(decoded: DecodedImage, config: dict | RGBViewConfig | None = None,
                    *, source: str = "", roi: str = "") -> tuple[list[np.ndarray], dict]:
    cfg = config if isinstance(config, RGBViewConfig) else RGBViewConfig.from_dict(config)
    if source == "agtron" and not roi:
        raise ValueError("Agtron requires ROI after EXIF; full labelled image is forbidden")
    rgb = crop_roi(decoded, roi) if roi else np.asarray(decoded.image)
    if min(rgb.shape[:2]) < 8:
        # Segment's minimum size is 8; a valid tiny image still gets full views.
        images = [resize_pad(rgb, cfg.size, cfg.pad_rgb) for _ in cfg.views]
        return images, {"mode": "full", "wb_applied": False, "fallback": "tiny_image"}
    sc = SegConfig.from_dict(cfg.seg_config)
    work, _ = to_work(rgb, sc)
    allow_wb = source != "agtron" and not roi
    wb = background_white_balance(work.astype(np.float32) / 255, sc, allow=allow_wb)
    corrected = np.clip(np.rint(wb.f * 255), 0, 255).astype(np.uint8)
    # segment determines geometry only. The RGB texture is retained in corrected.
    seg = segment(rgb, sc, find_beans=not roi)
    x0, y0, x1, y1 = seg.crop_box
    smart = corrected[y0:y1, x0:x1]
    if not smart.size:
        smart = corrected
    images = [resize_pad(corrected if v == "full" else smart, cfg.size, cfg.pad_rgb) for v in cfg.views]
    return images, {"mode": seg.mode, "wb_applied": wb.applied, "crop_box": list(seg.crop_box),
                    "fallback": seg.mode != "beans"}


def build_rgb_views(decoded: DecodedImage, config: dict | RGBViewConfig | None = None,
                    *, source: str = "", roi: str = "") -> np.ndarray:
    cfg = config if isinstance(config, RGBViewConfig) else RGBViewConfig.from_dict(config)
    images, _ = rgb_view_images(decoded, cfg, source=source, roi=roi)
    x = np.asarray(images, np.float32) / 255
    x = (x - np.asarray(cfg.mean, np.float32)) / np.asarray(cfg.std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)


def pool_view_probabilities(probabilities: np.ndarray, weights) -> np.ndarray:
    """Normalised geometric mean (the spec's weighted log-prob pooling)."""
    p = np.asarray(probabilities, np.float64)
    w = np.asarray(weights, np.float64)
    if p.ndim != 2 or len(p) != len(w) or not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError("invalid per-view probabilities")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid view weights")
    p = p / p.sum(1, keepdims=True)
    z = np.average(np.log(np.clip(p, 1e-12, 1)), weights=w, axis=0)
    z -= z.max()
    q = np.exp(z)
    return q / q.sum()
