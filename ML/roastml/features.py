"""feature สีระดับภาพ (CIELAB) สำหรับ baseline B0/B1 — ใช้ฟังก์ชันเดียวกันทั้งตอนเทรนและบน Pi

ทำไม Lab: L* แยกความสว่างออกจากสี → ระดับคั่วคือการเลื่อนลงของ L* เป็นหลัก
ใช้ median/percentile แทน mean (ทน specular และเงาที่หลุดมา) + histogram L* 8 ช่อง
ข้อจำกัด (ml-spec ข้อ 2): แสง/white balance เปลี่ยนสีได้พอๆ กับระดับคั่ว → feature สีล้วนมีเพดาน
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .segment import SegConfig, SegResult, border_pixels, segment, to_lab

L_BINS = np.linspace(0.0, 100.0, 9)  # histogram L* 8 ช่อง
STAT_NAMES = ["L_med", "L_mean", "L_p10", "L_p25", "L_p75", "L_p90", "L_std", "L_iqr", "a_med", "b_med", "C_med", "hue"]
HIST_NAMES = [f"Lh{i}" for i in range(len(L_BINS) - 1)]
FEATURES_ALL = STAT_NAMES + HIST_NAMES

# ชุด feature ที่ train_baseline เทียบกัน (เลือกด้วย LOSO)
FEATURE_SETS = {
    "L1": ["L_med"],
    "L": ["L_med", "L_p10", "L_p90", "L_std"],
    "Lab": ["L_med", "L_p10", "L_p90", "L_std", "a_med", "b_med", "C_med"],
    "Lab_hist": ["L_med", "L_p10", "L_p90", "L_std", "a_med", "b_med", "C_med"] + HIST_NAMES,
    # B1-small: feature น้อย ทนโดเมนเปลี่ยนกว่า (ใช้คู่กับ C เล็ก = regularize แรง)
    "small": ["L_med", "a_med", "b_med", "L_iqr", "C_med"],
}

# feature ของ "ขอบภาพ" สำหรับ shortcut probe (ทายคลาสจากพื้นหลังอย่างเดียว)
BORDER_FRAC = 0.10
BORDER_NAMES = ["bL_med", "bL_p10", "bL_p90", "bL_std", "ba_med", "bb_med", "bC_med"]


def pixel_stats(lab_px: np.ndarray) -> np.ndarray:
    """(n, 3) Lab → vector ตาม FEATURES_ALL"""
    if len(lab_px) == 0:
        raise ValueError("ไม่มีพิกเซลให้คิด feature")
    L, a, b = lab_px[:, 0], lab_px[:, 1], lab_px[:, 2]
    p10, p25, p50, p75, p90 = np.percentile(L, [10, 25, 50, 75, 90])
    a_med, b_med = float(np.median(a)), float(np.median(b))
    hist = np.histogram(np.clip(L, 0, 100), bins=L_BINS)[0].astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    stats = [p50, float(L.mean()), p10, p25, p75, p90, float(L.std()), p75 - p25,
             a_med, b_med, float(np.median(np.hypot(a, b))), float(np.degrees(np.arctan2(b_med, a_med)))]
    return np.concatenate([np.asarray(stats, np.float64), hist])


@dataclass
class ImageFeatures:
    x: np.ndarray        # (len(FEATURES_ALL),)
    seg: SegResult
    ms: float


def image_features(rgb: np.ndarray, cfg: SegConfig | None = None, *, find_beans: bool = True) -> ImageFeatures:
    """rgb uint8 (h, w, 3) → feature ระดับภาพ จากพิกเซลที่ segment เลือก (beans = แกนในเมล็ด, pile/full = ตัดเงา/specular)"""
    t0 = time.perf_counter()
    seg = segment(rgb, cfg, find_beans=find_beans)
    x = pixel_stats(seg.lab[seg.pixel_mask])
    return ImageFeatures(x, seg, (time.perf_counter() - t0) * 1e3)


def border_features(rgb: np.ndarray, work_side: int = 400) -> np.ndarray:
    """สถิติ Lab ของแถบขอบภาพ (BORDER_FRAC) — ไม่ผ่าน WB · ใช้เฉพาะ shortcut probe"""
    import cv2

    h, w = rgb.shape[:2]
    s = min(1.0, work_side / max(h, w))
    if s < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    px = border_pixels(to_lab(rgb.astype(np.float32) / 255.0), BORDER_FRAC)
    L, a, b = px[:, 0], px[:, 1], px[:, 2]
    p10, p50, p90 = np.percentile(L, [10, 50, 90])
    return np.array([p50, p10, p90, L.std(), np.median(a), np.median(b), np.median(np.hypot(a, b))], np.float64)


def select(X: np.ndarray, names: list[str]) -> np.ndarray:
    idx = [FEATURES_ALL.index(n) for n in names]
    return X[..., idx]
