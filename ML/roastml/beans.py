"""นับเมล็ด + แยกพิกเซลรายเมล็ด (ส่วนเสริม · numpy + OpenCV เท่านั้น)

ใช้ผลของ segment() ที่คำนวณไปแล้ว (ไม่แตะเส้นทาง B1) แล้วทำขั้น contour ซ้ำด้วยกฎเดียวกับ segment.py
→ ได้ blob ชุดเดียวกับ seg.n_regions (มี test ยืนยันว่าจำนวนตรงกัน)
- ทำงานเฉพาะ seg.mode == "beans" (พื้นหลังสม่ำเสมอ แยกเมล็ดได้) · pile/full → คืน None (ไม่ประมาณจำนวน:
  เฟส 0 ไม่มี GT ของกองหนาแน่น ตัวเลขประมาณจึงตรวจสอบไม่ได้)
- ตัวนับ = (a) contour เดิม: เฟส 0 บน rf_boos (GT รายเมล็ด) MAE 0.16, median rel error 0 · blob ที่เป็น
  เมล็ดติดกันนับเป็น 1 (ยังไม่แยก — watershed ไม่ได้ดีกว่าบนข้อมูลที่มี GT)
- bbox = [x, y, w, h] int เป็นพิกเซลของภาพที่ decode แล้ว (หลัง EXIF, ≤1600 px) ไม่ใช่ภาพ work
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .segment import SegConfig, SegResult, _odd, border_pixels

MAX_BLOBS = 150  # G3: เกินนี้ไม่จัดคลาสรายเมล็ด (คืนแค่จำนวน)


@dataclass
class Bean:
    bbox: list[int]          # x, y, w, h ในพิกัดภาพ decode
    work_xy: tuple[int, int] # มุมซ้ายบนของ core ในพิกัดภาพ work
    core: np.ndarray         # bool (bh, bw) พิกเซลแกนในของเมล็ด (กฎเดียวกับ segment.core_mask)

    def lab_pixels(self, lab: np.ndarray) -> np.ndarray:
        x, y = self.work_xy
        h, w = self.core.shape
        return lab[y:y + h, x:x + w][self.core]


def split_beans(seg: SegResult, cfg: SegConfig | None = None) -> list[Bean] | None:
    """SegResult → รายการเมล็ด · None = แยกเมล็ดไม่ได้ (โหมด pile/full)"""
    cfg = cfg or SegConfig()
    if seg.mode != "beans":
        return None
    lab = seg.lab
    h, w = lab.shape[:2]
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
    return beans or None


def count_proportions(labels: list[str], classes) -> dict[str, float]:
    n = len(labels)
    return {c: labels.count(c) / n for c in classes}

