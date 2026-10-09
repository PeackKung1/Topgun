"""นับเมล็ด + แยกพิกเซลรายเมล็ด (ส่วนเสริม · numpy + OpenCV เท่านั้น)

ใช้ผลของ segment() ที่คำนวณไปแล้ว (ไม่แตะเส้นทาง B1) แล้วทำขั้น contour ซ้ำด้วยกฎเดียวกับ segment.py
→ ภาพ beans ใช้ contour; ถ้าเมล็ดแตะกันหรือเป็น pile/full ใช้ edge contours เป็นค่าประมาณ
- ตัวนับ edge ยังไม่ได้ตรวจเทียบ GT ของกองหนาแน่น และนับเมล็ดที่ถูกบังไม่ได้
- ตัวนับ = (a) contour เดิม: เฟส 0 บน rf_boos (GT รายเมล็ด) MAE 0.16, median rel error 0 · blob ที่เป็น
  เมล็ดติดกันนับเป็น 1 (ยังไม่แยก — watershed ไม่ได้ดีกว่าบนข้อมูลที่มี GT)
- bbox = [x, y, w, h] int เป็นพิกเซลของภาพที่ decode แล้ว (หลัง EXIF, ≤1600 px) ไม่ใช่ภาพ work
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .segment import SegConfig, SegResult, _odd, beanlike_mask, border_pixels

MAX_BLOBS = 150  # G3: เกินนี้ไม่จัดคลาสรายเมล็ด (คืนแค่จำนวน)


@dataclass
class Bean:
    bbox: list[int]          # x, y, w, h ในพิกัดภาพ decode
    work_xy: tuple[int, int] # มุมซ้ายบนของ core ในพิกัดภาพ work
    core: np.ndarray         # bool (bh, bw) พิกเซลแกนในของเมล็ด (กฎเดียวกับ segment.core_mask)
    estimated: bool = False  # True เมื่อแยกจากขอบภาพ ไม่ใช่ contour ที่แยกได้ตรงๆ

    def lab_pixels(self, lab: np.ndarray) -> np.ndarray:
        x, y = self.work_xy
        h, w = self.core.shape
        return lab[y:y + h, x:x + w][self.core]


def split_beans(
    seg: SegResult,
    cfg: SegConfig | None = None,
    *,
    rgb: np.ndarray | None = None,
) -> list[Bean] | None:
    """SegResult → รายการเมล็ด; ใช้ edge contours ช่วยนับภาพที่เมล็ดแตะกันได้"""
    cfg = cfg or SegConfig()
    lab = seg.lab
    h, w = lab.shape[:2]

    if seg.mode != "beans":
        beans = _edge_bean_candidates(seg, cfg, rgb) if rgb is not None else None
        return beans or None

    # สีพื้นหลัง: segment ใช้ median ของแถบขอบจาก lab หลัง WB (ทั้งกรณีทำ/ไม่ทำ WB) → คำนวณซ้ำได้ตรงตัว
    bg = np.median(border_pixels(lab, cfg.border_frac), axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(dist * 2, 0, 255).astype(np.uint8)
    t_otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dist > max(t_otsu / 2.0, cfg.min_contrast)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(h, w) * 0.006),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(h * w)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (cfg.min_area_frac <= a / img_area <= cfg.max_area_frac):
            continue
        sol = a / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        cand.append((c, a, sol, max(rw, rh) / max(min(rw, rh), 1.0)))
    if not cand:
        return None
    good = [a for _, a, sol, asp in cand if sol >= cfg.min_solidity and asp <= cfg.max_aspect]
    med = float(np.median(good if good else [a for _, a, _, _ in cand]))

    inv = 1.0 / seg.scale if seg.scale > 0 else 1.0
    beans: list[Bean] = []
    for c, a, sol, asp in cand:
        if a < cfg.min_rel_area * med:
            continue
        if a > cfg.cluster_area_ratio * med:
            if sol < cfg.cluster_min_solidity:
                continue
        elif sol < cfg.min_solidity or asp > cfg.max_aspect:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        rm = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(rm, [c - [x, y]], -1, 1, thickness=cv2.FILLED)
        dt = cv2.distanceTransform(rm, cv2.DIST_L2, 3)
        core = dt >= cfg.core_frac * dt.max()
        if core.sum() < 20:
            continue
        bbox = [int(round(x * inv)), int(round(y * inv)), max(1, int(round(bw * inv))), max(1, int(round(bh * inv)))]
        beans.append(Bean(bbox, (x, y), core))
    separated = beans or None
    if rgb is not None:
        edge_beans = _edge_bean_candidates(seg, cfg, rgb)
        if edge_beans:
            dense_split = (
                seg.n_regions <= 2
                and len(edge_beans) >= max(3, 3 * (len(separated) if separated else 0))
            ) or (
                len(edge_beans) >= 25
                and len(edge_beans) > 2 * (len(separated) if separated else 0)
            )
            if separated is None or dense_split:
                return edge_beans
    return separated


def _edge_bean_candidates(
    seg: SegResult,
    cfg: SegConfig,
    rgb: np.ndarray,
) -> list[Bean] | None:
    """Find individual visible bean outlines when foreground contours merge.

    This remains a CV heuristic: it uses a stable Canny threshold range and
    rejects edges outside the bean foreground, rather than inferring occluded
    beans. The caller uses it only when it yields more candidates than blobs.
    """
    h, w = seg.lab.shape[:2]
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    if seg.mode == "beans":
        bg = np.median(border_pixels(seg.lab, cfg.border_frac), axis=0)
        dist = np.linalg.norm(seg.lab - bg, axis=2)
        d8 = np.clip(dist * 2, 0, 255).astype(np.uint8)
        t_otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground = dist > max(t_otsu / 2.0, cfg.min_contrast)
    else:
        foreground = beanlike_mask(seg.lab)
    if not foreground.any():
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thresholds = ((40, 100), (50, 110), (60, 120), (70, 140), (80, 160), (90, 180))
    candidates: list[list[tuple[np.ndarray, tuple[int, int, int, int]]]] = []
    counts: list[int] = []
    min_area = max(20.0, cfg.min_area_frac * h * w * 1.25)
    max_area = cfg.max_area_frac * h * w * 0.09
    min_side = max(6, int(round(min(h, w) * 0.02)))
    for low, high in thresholds:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area <= area <= max_area):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if min(bw, bh) < min_side or max(bw, bh) / max(min(bw, bh), 1) > cfg.max_aspect:
                continue
            cx, cy = min(w - 1, x + bw // 2), min(h - 1, y + bh // 2)
            if not foreground[cy, cx]:
                continue
            found.append((contour, (x, y, bw, bh)))
        candidates.append(found)
        counts.append(len(found))

    # Pick the strongest edge setting in the stable high-threshold region.
    # This suppresses weak texture edges while keeping bean outlines.
    selected = len(thresholds) - 1
    for i in range(len(thresholds) - 2, -1, -1):
        if abs(counts[i + 1] - counts[i]) <= 0.15 * max(counts[i], counts[i + 1], 1):
            selected = i + 1
            break
    selected_candidates = candidates[selected]
    if not selected_candidates:
        return None

    inv = 1.0 / seg.scale if seg.scale > 0 else 1.0
    result: list[Bean] = []
    for _contour, (x, y, bw, bh) in selected_candidates:
        yy, xx = np.ogrid[:bh, :bw]
        rx, ry = max(bw * 0.34, 1.0), max(bh * 0.34, 1.0)
        core = (((xx + 0.5 - bw / 2) / rx) ** 2 + ((yy + 0.5 - bh / 2) / ry) ** 2 <= 1)
        core &= foreground[y:y + bh, x:x + bw]
        if core.sum() < 20:
            continue
        bbox = [int(round(x * inv)), int(round(y * inv)),
                max(1, int(round(bw * inv))), max(1, int(round(bh * inv)))]
        result.append(Bean(bbox, (x, y), core, estimated=True))
    return result or None


def count_proportions(labels: list[str], classes) -> dict[str, float]:
    n = len(labels)
    return {c: labels.count(c) / n for c in classes}

