"""Shared train/serve bean features and deterministic brightness grouping.

Runtime depends only on numpy/OpenCV. Group thresholds are selected using
synthetic mixtures of training-fold beans, never the held-out source.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

from .counter import CountConfig, count_beans
from .features import FEATURES_ALL, _grouped_percentiles, pixel_stats_grouped
from .segment import SegConfig, PreparedImage


def count_features(result):
    if not result.n:
        return np.zeros((0, len(FEATURES_ALL)), np.float64)
    return pixel_stats_grouped(result.lab[result.core], result.labels[result.core]-1, result.n)


BOX_SHAPES = ("rect", "ellipse", "mask")


def fuse_image_prior(P, image_probs, weight: float = 1.0):
    """prob ต่อเมล็ด × (prob ระดับภาพ)^weight แล้ว normalize — ภาพคั่วระดับเดียวไม่แกว่งตาม noise รายเมล็ด
    แต่เมล็ดที่หลักฐานชัดยังต่างจากภาพได้ (ภาพปนหลายระดับ) · weight = 0 → ไม่ใช้ prior"""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("prior weight ต้อง ≥ 0")
    Q = np.asarray(P, np.float64) * np.maximum(np.asarray(image_probs, np.float64), 1e-9) ** weight
    return Q / Q.sum(axis=1, keepdims=True)


def _paint_boxes(boxes, shape: str, shrink: float, h: int, w: int):
    """label image (0 = ไม่มี, i+1 = กรอบ i) · กรอบเล็กทับกรอบใหญ่ · loop ต่อกรอบ = วาด 1 รูปต่อกรอบ"""
    labels = np.zeros((h, w), np.int32)
    x0 = np.clip(np.floor(boxes[:, 0] + boxes[:, 2] * shrink), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(boxes[:, 1] + boxes[:, 3] * shrink), 0, h - 1).astype(np.int64)
    x1 = np.clip(np.ceil(boxes[:, 0] + boxes[:, 2] * (1 - shrink)), x0 + 1, w).astype(np.int64)
    y1 = np.clip(np.ceil(boxes[:, 1] + boxes[:, 3] * (1 - shrink)), y0 + 1, h).astype(np.int64)
    for i in np.argsort(-(boxes[:, 2] * boxes[:, 3]), kind="stable"):
        if shape == "ellipse":
            cv2.ellipse(labels, (int((x0[i] + x1[i] - 1) // 2), int((y0[i] + y1[i] - 1) // 2)),
                        (max(int((x1[i] - x0[i]) // 2), 0), max(int((y1[i] - y0[i]) // 2), 0)), 0, 0, 360, int(i + 1), -1)
        else:
            labels[y0[i]:y1[i], x0[i]:x1[i]] = i + 1
    return labels


def box_bean_features(lab, boxes, seg_config: SegConfig, shrink: float = 0.2, min_px: int = 20, shape: str = "rect",
                      mask=None):
    """feature สีต่อกรอบ (จาก detector) → (n, len(FEATURES_ALL)) · คิดทุกกรอบพร้อมกัน

    lab (h, w, 3) ภาพ work หลัง WB · boxes (n, 4) x, y, w, h ในพิกัดภาพ work
    shape "rect" (shrink 0.2) = สูตรเดียวกับ features.box_features ที่ใช้เทรน B1: กลางกรอบ + ตัด L* นอก trim_lo..trim_hi ต่อเมล็ด
    shape "ellipse" = วงรีในกรอบที่หดแล้ว (ตัดมุมที่เป็นพื้นหลัง/เมล็ดข้างเคียง) + trim เหมือนกัน
    shape "mask" = พิกเซลที่ segment() เลือกให้ B1 ระดับภาพ (mask) ∩ กรอบเต็ม → feature รายเมล็ดอยู่บนสเกลเดียวกับ feature ระดับภาพ
                   (ไม่ trim ซ้ำ · mask ผ่าน trim ระดับภาพแล้ว) · กรอบที่ได้ < min_px → ถอยไปใช้สูตร rect
    กรอบซ้อนกัน: กรอบเล็กทับกรอบใหญ่ · ทุกเมล็ดมีอย่างน้อย 1 พิกเซล (จุดกลางกรอบ)
    """
    if shape not in BOX_SHAPES or not 0 <= shrink < 0.5:
        raise ValueError("box feature: shape/shrink ไม่ถูกต้อง")
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    n = len(boxes)
    if not n:
        return np.zeros((0, len(FEATURES_ALL)), np.float64)
    h, w = lab.shape[:2]
    trim = np.ones(n, bool)   # เมล็ดที่ต้องตัด L* ต่อเมล็ด
    if shape == "mask":
        if mask is None or mask.shape != (h, w):
            raise ValueError("shape='mask' ต้องมี mask ขนาดเท่าภาพ work")
        full = _paint_boxes(boxes, "rect", 0.0, h, w)
        m = (full > 0) & mask
        px, g = lab[m].astype(np.float64), full[m].astype(np.int64) - 1
        ok = np.bincount(g, minlength=n) >= min_px
        trim = ~ok
        if not ok.all():   # เมล็ดที่ mask ไม่ครอบ (segment ไม่เห็น) → สูตร rect เฉพาะเมล็ดนั้น
            keep_ok = ok[g]
            px, g = px[keep_ok], g[keep_ok]
            core = _paint_boxes(boxes, "rect", shrink, h, w)
            mc = core > 0
            mc[mc] = ~ok[core[mc] - 1]
            px, g = np.concatenate([px, lab[mc].astype(np.float64)]), np.concatenate([g, core[mc].astype(np.int64) - 1])
    else:
        labels = _paint_boxes(boxes, shape, shrink, h, w)
        m = labels > 0
        px, g = lab[m].astype(np.float64), labels[m].astype(np.int64) - 1
    missing = np.flatnonzero(np.bincount(g, minlength=n) == 0)  # ถูกกรอบเล็กกว่าทับหมด → ใช้จุดกลางกรอบ
    if len(missing):
        cx = np.clip(np.rint(boxes[missing, 0] + boxes[missing, 2] / 2), 0, w - 1).astype(np.int64)
        cy = np.clip(np.rint(boxes[missing, 1] + boxes[missing, 3] / 2), 0, h - 1).astype(np.int64)
        px, g = np.concatenate([px, lab[cy, cx].astype(np.float64)]), np.concatenate([g, missing])
    counts = np.bincount(g, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    L = px[:, 0]
    lo_hi = _grouped_percentiles(L[np.lexsort((L, g))], starts, counts, (seg_config.trim_lo, seg_config.trim_hi))
    keep = ((L >= lo_hi[g, 0]) & (L <= lo_hi[g, 1])) | ~trim[g]
    keep |= (np.bincount(g[keep], minlength=n) < min_px)[g]  # เหลือน้อยเกิน → ใช้ทุกพิกเซลของกรอบนั้น
    return pixel_stats_grouped(px[keep], g[keep], n)


def bean_features(rgb, count_config: CountConfig, seg_config: SegConfig, *, force_pile=False,
                  prepared: PreparedImage | None = None):
    result = count_beans(rgb, count_config, seg_config, force_pile=force_pile, prepared=prepared)
    return result, count_features(result)


def brightness_groups(X, k):
    """1D k-means with deterministic quantile starts; no process-global RNG."""
    v = np.asarray(X)[:, FEATURES_ALL.index("L_med")]
    centers = np.percentile(v, (np.arange(k)+0.5)*100/k)
    for _ in range(30):
        labels = np.abs(v[:, None]-centers).argmin(1)
        if len(np.unique(labels)) < k:
            return None, None
        new = np.array([v[labels == j].mean() for j in range(k)])
        if np.allclose(new, centers, atol=1e-6):
            break
        centers = new
    order = np.argsort(centers)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[labels], centers[order]


def group_predictions(X, P, classes, config=None):
    """k=1..3; labels use group mean probabilities with strict dark→light order."""
    config = config or {}
    if set(config)-{"min_L_gap", "max_k"}:
        raise ValueError("unknown group_config key")
    gap = float(config.get("min_L_gap", 12.0))
    max_k = config.get("max_k", 3)
    if not np.isfinite(gap) or gap <= 0 or max_k not in (1, 2, 3):
        raise ValueError("invalid group_config")
    if not len(X):
        return np.array([], int), np.array([], float), {"k": 0}
    selected, k = np.zeros(len(X), int), 1
    for candidate in range(2, min(max_k, len(X))+1):
        ll, centers = brightness_groups(X, candidate)
        if ll is not None and np.min(np.diff(centers)) >= gap:
            selected, k = ll, candidate
    mean = np.array([P[selected == j].mean(0) for j in range(k)])
    ordered = [classes.index(c) for c in ("dark", "medium", "light")]
    # Enumerate ordered distinct class assignments (3, 3, 1 possibilities).
    choices = list(itertools.combinations(ordered, k))
    mapping = max(choices, key=lambda cols: sum(np.log(max(mean[j, c], 1e-12)) for j, c in enumerate(cols)))
    labels = np.asarray(mapping)[selected]
    # Preserve the specified .999 proportion + .001 mean-prob formula even
    # in the 1000-bean, one-vote-margin corner case. An ordered forced mapping
    # can otherwise let the mean probability overturn the count majority.
    counts = np.bincount(labels, minlength=len(classes))
    aggregate = .999*counts/len(labels) + .001*P.mean(0)
    if counts[aggregate.argmax()] < counts.max():
        best = int(P.mean(0).argmax())
        return np.full(len(X), best, int), np.full(len(X), P.mean(0)[best]), {"k": 1}
    confidence = np.array([mean[j, mapping[j]] for j in selected])
    return labels, confidence, {"k": k}


def majority_probs(labels, P, classes):
    n = len(labels)
    if not n:
        raise ValueError("empty beans need B1 fallback")
    counts = np.bincount(labels, minlength=len(classes))
    proportions = counts / n
    probs = 0.999 * proportions + 0.001 * P.mean(0)
    if counts[probs.argmax()] < counts.max():
        raise ValueError("mean probability overturns bean majority; unsupported count/mapping")
    return ({c: float(probs[i]) for i,c in enumerate(classes)},
            {c: float(proportions[i]) for i,c in enumerate(classes)})
