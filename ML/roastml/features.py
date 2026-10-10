"""feature สีระดับภาพ (CIELAB) สำหรับ baseline B0/B1 — ใช้ฟังก์ชันเดียวกันทั้งตอนเทรนและบน Pi

ทำไม Lab: L* แยกความสว่างออกจากสี → ระดับคั่วคือการเลื่อนลงของ L* เป็นหลัก
ใช้ median/percentile แทน mean (ทน specular และเงาที่หลุดมา) + histogram L* 8 ช่อง
ข้อจำกัด (ml-spec ข้อ 2): แสง/white balance เปลี่ยนสีได้พอๆ กับระดับคั่ว → feature สีล้วนมีเพดาน
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .segment import SegConfig, SegResult, background_white_balance, border_pixels, segment, to_lab, to_work

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


def _grouped_percentiles(vals: np.ndarray, starts: np.ndarray, counts: np.ndarray, qs) -> np.ndarray:
    """vals เรียงภายในกลุ่มแล้ว (กลุ่มต่อกัน) → percentile แบบ linear เหมือน np.percentile ต่อกลุ่ม · (n_group, len(qs))"""
    out = np.empty((len(counts), len(qs)))
    for j, q in enumerate(qs):
        pos = (counts - 1) * (q / 100.0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, counts - 1)
        frac = pos - lo
        out[:, j] = vals[starts + lo] * (1 - frac) + vals[starts + hi] * frac
    return out


def pixel_stats_grouped(lab_px: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """pixel_stats ของหลายเมล็ดพร้อมกัน (vectorized ไม่มี loop ต่อเมล็ด) — ผลเท่ากับเรียก pixel_stats ทีละกลุ่ม

    lab_px (m, 3) · labels (m,) int ใน 0..n-1 · ทุกกลุ่มต้องมี ≥ 1 พิกเซล → (n, len(FEATURES_ALL))
    """
    raw_labels = np.asarray(labels)
    if raw_labels.dtype.kind not in 'iu':
        raise ValueError('group labels must be integers')
    labels = raw_labels.astype(np.int64)
    lab_px = np.asarray(lab_px, np.float64)
    if n < 1 or lab_px.ndim != 2 or lab_px.shape[1] != 3 or labels.shape != (len(lab_px),):
        raise ValueError("invalid grouped pixels/labels shape")
    if not np.isfinite(lab_px).all() or np.any(labels < 0) or np.any(labels >= n):
        raise ValueError("invalid grouped pixel values/labels")
    counts = np.bincount(labels, minlength=n)
    if len(counts) != n or np.any(counts == 0):
        raise ValueError("ทุกเมล็ดต้องมีพิกเซล ≥ 1 และ label อยู่ใน 0..n-1")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    L, a, b = lab_px[:, 0], lab_px[:, 1], lab_px[:, 2]
    C = np.hypot(a, b)

    def sorted_within(v):
        # lexsort remains correct for every finite input, including extremes
        # used in parity tests; no hidden Lab-range assumption.
        return v[np.lexsort((v, labels))]

    pL = _grouped_percentiles(sorted_within(L), starts, counts, (10, 25, 50, 75, 90))
    a_med = _grouped_percentiles(sorted_within(a), starts, counts, (50,))[:, 0]
    b_med = _grouped_percentiles(sorted_within(b), starts, counts, (50,))[:, 0]
    C_med = _grouped_percentiles(sorted_within(C), starts, counts, (50,))[:, 0]
    mean = np.bincount(labels, L, n) / counts
    std = np.sqrt(np.maximum(np.bincount(labels, (L - mean[labels]) ** 2, n) / counts, 0.0))
    bins = np.clip(np.searchsorted(L_BINS, np.clip(L, 0, 100), side="right") - 1, 0, len(L_BINS) - 2)
    hist = np.bincount(labels * (len(L_BINS) - 1) + bins, minlength=n * (len(L_BINS) - 1))
    hist = hist.reshape(n, len(L_BINS) - 1).astype(np.float64) / counts[:, None]
    p10, p25, p50, p75, p90 = pL.T
    stats = np.column_stack([p50, mean, p10, p25, p75, p90, std, p75 - p25, a_med, b_med, C_med,
                             np.degrees(np.arctan2(b_med, a_med))])
    return np.concatenate([stats, hist], axis=1)


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


def box_mask(shape: tuple[int, int], boxes, shrink: float = 0.2) -> np.ndarray:
    """union ของ bbox (cx, cy, w, h สัมพัทธ์ 0..1) ที่หดเข้าแต่ละด้าน `shrink` ของความกว้าง/สูง → bool (h, w)"""
    h, w = shape
    m = np.zeros((h, w), bool)
    for cx, cy, bw, bh in boxes:
        x0, x1 = (cx - bw * (0.5 - shrink)) * w, (cx + bw * (0.5 - shrink)) * w
        y0, y1 = (cy - bh * (0.5 - shrink)) * h, (cy + bh * (0.5 - shrink)) * h
        xa, xb = max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)))
        ya, yb = max(0, int(np.floor(y0))), min(h, int(np.ceil(y1)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    return m


@dataclass
class BoxFeatures:
    x: np.ndarray
    mask: np.ndarray
    wb_applied: bool


def box_features(rgb: np.ndarray, boxes, cfg: SegConfig | None = None, shrink: float = 0.2,
                 *, white_balance: bool = True) -> BoxFeatures:
    """feature จากพิกเซลใน bbox (label YOLO) — ใช้ตอนเทรน robusta/boos เท่านั้น (บน Pi ไม่มี bbox)

    1. ย่อ + WB/exposure ด้วย segment.to_work / background_white_balance ตัวเดียวกับ pipeline
       พื้นหลัง = แถบขอบภาพ (ไม่ใช่ "นอก bbox"): ให้ภาพเดียวกันได้ guard และ gain ตรงกับที่ segment บน Pi คำนวณ
       ถ้า bbox แตะขอบจนขอบไม่สม่ำเสมอ guard ไม่ผ่านทั้งสองเส้นทางเหมือนกัน
    2. หด bbox เข้า `shrink` (ตัดมุมกล่องที่เป็นพื้นหลัง) แล้วตัด L* ตาม trim_lo..trim_hi (เงาร่อง/specular)
    white_balance=False = พฤติกรรมเดิมก่อน 8 ต.ค. (ไว้เทียบ R0)
    """
    cfg = cfg or SegConfig()
    if not boxes:
        raise ValueError("ไม่มี bbox")
    work, _ = to_work(rgb, cfg)
    wbr = background_white_balance(work.astype(np.float32) / 255.0, cfg, allow=white_balance)
    lab = wbr.lab
    m = box_mask(lab.shape[:2], boxes, shrink)
    if m.sum() < 20:
        m = box_mask(lab.shape[:2], boxes, 0.0)
    if not m.any():
        raise ValueError("bbox อยู่นอกภาพ")
    L = lab[..., 0]
    lo, hi = np.percentile(L[m], [cfg.trim_lo, cfg.trim_hi])
    core = m & (L >= lo) & (L <= hi)
    if core.sum() >= 20:
        m = core
    return BoxFeatures(pixel_stats(lab[m]), m, wbr.applied)


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
