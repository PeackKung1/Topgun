"""ตัวหาเมล็ดด้วย detector คลาสเดียว (YOLO export เป็น ONNX) — runtime ใช้แค่ onnxruntime + numpy + OpenCV

ภาพ decode (RGB) → letterbox เป็น imgsz → ONNX → ถอดกรอบ + NMS (numpy) → กรอบ [x, y, w, h] ในพิกัดภาพ decode
รูปแบบ output ที่รองรับ: (1, 4 + nc, N) ของ Ultralytics detect ที่ export แบบไม่มี NMS (cx, cy, w, h เป็น px ของ input · คะแนนผ่าน sigmoid แล้ว)
ภาพใหญ่ที่เมล็ดเล็ก: tile=True แบ่งภาพเป็นช่องซ้อนกันแล้วรวมด้วย NMS (ช้าลงตามจำนวนช่อง)
หลัง NMS ตัดกรอบที่ไม่ใช่เมล็ด 2 แบบ: กรอบภาชนะ (container_boxes) และรอยพิมพ์สีอ่อนบนป้าย/พื้นเรียบ (flat_marks)
ค่า conf / iou / max_det มาจาก model_card (ประกาศก่อนประเมิน) — ไม่ได้จูนบน test
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetConfig:
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 1000
    pre_nms_top: int = 4000   # จำกัดจำนวนผู้สมัครก่อน NMS (คุมเวลา)
    threads: int = 4          # intra-op threads ของ onnxruntime (Pi 5 มี 4 core)
    tile: bool = False        # แบ่งภาพ 2×2 ซ้อนกันเพิ่มจากภาพเต็ม
    tile_overlap: float = 0.2
    # ตัดกรอบ "ภาชนะ": กรอบที่มีจุดกลางของกรอบอื่น ≥ container_min อยู่ข้างใน และใหญ่ ≥ container_area_ratio × มัธยฐาน
    # (จาน/ถ้วยที่เมล็ดเล็กมากถูกมองเป็นเมล็ดใหญ่ 1 เมล็ด) · ค่าตั้งจากเหตุผล ไม่ได้จูน · container_min = 0 ปิด
    container_min: int = 5
    container_area_ratio: float = 4.0
    # ตัดกรอบ "รอยพิมพ์สีอ่อนบนพื้นเรียบ" (ตัวหนังสือขาวบนป้าย): รอบกรอบเป็นพื้นสว่างสีเดียว ≥ สัดส่วนนี้ แต่ในกรอบไม่มีเนื้อเมล็ด
    # ดู flat_marks · ตั้งจากภาพป้ายภาพเดียว (ตัวหนังสือ ≥ 0.79 · เมล็ดในกอง ≤ 0.58) ยังไม่ได้วัดกับ count test · 0 ปิด
    mark_flat_min: float = 0.65

    @classmethod
    def from_dict(cls, d: dict | None) -> "DetConfig":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"det_config ไม่รู้จัก key: {sorted(unknown)}")
        cfg = cls(**d)
        if not 0 < cfg.conf < 1 or not 0 < cfg.iou < 1 or not 0 <= cfg.tile_overlap < 0.5:
            raise ValueError("det_config: conf/iou/tile_overlap อยู่นอกช่วง")
        if isinstance(cfg.container_min, bool) or not isinstance(cfg.container_min, int) or cfg.container_min < 0 \
                or not np.isfinite(cfg.container_area_ratio) or cfg.container_area_ratio < 1:
            raise ValueError("det_config: container_min ≥ 0 และ container_area_ratio ≥ 1")
        if isinstance(cfg.mark_flat_min, bool) or not 0 <= cfg.mark_flat_min <= 1:
            raise ValueError("det_config: mark_flat_min อยู่นอกช่วง 0..1")
        for name in ("max_det", "pre_nms_top", "threads"):
            v = getattr(cfg, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"det_config: {name} ต้องเป็นจำนวนเต็ม ≥ 1")
        if cfg.max_det > 10000:
            raise ValueError("det_config: max_det ≤ 10000")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """ย่อ/ขยายคงสัดส่วนให้พอดี size×size เติมขอบเทา 114 → (ภาพ uint8, scale, pad_x, pad_y)"""
    h, w = rgb.shape[:2]
    s = min(size / h, size / w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, s, px, py


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float, max_det: int) -> np.ndarray:
    """greedy NMS · boxes (n, 4) x0, y0, x1, y1 · คืน index ที่เก็บ เรียงคะแนนมาก→น้อย"""
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(-scores, kind="stable")
    b = boxes[order].astype(np.float64)
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    alive = np.ones(len(b), bool)
    keep = []
    for i in range(len(b)):  # loop ต่อกรอบที่ "เก็บ" (≤ max_det) · งานข้างในเป็น vectorized
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        rest = np.flatnonzero(alive[i + 1:]) + i + 1
        if len(rest) == 0:
            break
        iw = np.minimum(b[i, 2], b[rest, 2]) - np.maximum(b[i, 0], b[rest, 0])
        ih = np.minimum(b[i, 3], b[rest, 3]) - np.maximum(b[i, 1], b[rest, 1])
        inter = np.maximum(iw, 0) * np.maximum(ih, 0)
        alive[rest[inter / np.maximum(area[i] + area[rest] - inter, 1e-9) > iou]] = False
    return order[np.asarray(keep, np.int64)]


def container_boxes(boxes: np.ndarray, min_inside: int, area_ratio: float) -> np.ndarray:
    """boxes (n, 4) x0, y0, x1, y1 → bool (n,) True = กรอบภาชนะ (ครอบจุดกลางกรอบอื่น ≥ min_inside และใหญ่กว่ามัธยฐานมาก)"""
    n = len(boxes)
    if not min_inside or n <= min_inside:
        return np.zeros(n, bool)
    b = boxes.astype(np.float64)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    inside = (cx[None, :] >= b[:, 0, None]) & (cx[None, :] <= b[:, 2, None]) & (cy[None, :] >= b[:, 1, None]) & (cy[None, :] <= b[:, 3, None])
    np.fill_diagonal(inside, False)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inside.sum(axis=1) >= min_inside) & (area >= area_ratio * np.median(area))


MARK_GRID = 32        # สุ่มพิกเซลเป็นตาราง 32×32 บนพื้นที่ 2× กรอบ → กรอบ = 16×16 ช่องกลาง · วงรอบ = ช่องที่เหลือ
MARK_DE = 10.0        # ΔE จากสีพื้น (มัธยฐานของวงรอบ) ที่ถือว่า "ไม่ใช่พื้น" — ค่าเดียวกับ SegConfig.bg_uniform_de
MARK_MIN_L = 50.0     # พื้นต้องสว่าง (กระดาษ/ป้าย) — กองเมล็ดคั่วเข้มที่มืดสม่ำเสมอไม่เข้าเงื่อนไข (เท่ากับ SegConfig.wb_min_L)
MARK_BLOB = 3         # เนื้อเมล็ด = หน้าต่าง 3×3 ช่อง (≈ 0.2 ของด้านกรอบ) ที่ทุกช่องต่างจากพื้นและไม่สว่างกว่าพื้น


def flat_marks(rgb: np.ndarray, boxes: np.ndarray, flat_min: float) -> np.ndarray:
    """rgb uint8 · boxes (n, 4) x0, y0, x1, y1 → bool (n,) True = รอยพิมพ์สีอ่อนบนพื้นเรียบ (ตัวหนังสือ/โลโก้) ไม่ใช่เมล็ด

    เมล็ดคั่วบนพื้นเรียบสว่างมีเนื้อที่ต่างจากพื้นและไม่สว่างกว่าพื้น · ตัวหนังสือสีอ่อนบนป้าย (ขาวบนน้ำตาล) สว่างกว่าพื้น
    ตัดเฉพาะเมื่อครบทุกข้อ: พื้นรอบกรอบสว่าง (L ≥ MARK_MIN_L) + เรียบ (≥ flat_min ของวงรอบอยู่ใน ΔE < MARK_DE จากมัธยฐาน)
    + ในกรอบไม่มีเนื้อเมล็ด · พื้นไม่เรียบ/มืด (กองเมล็ด ผ้า ไม้ลาย) → ไม่ตัด
    ไม่ครอบคลุม: ตัวหนังสือสีเข้มบนพื้นอ่อน (เส้นหนาพอ ๆ กับเมล็ดในกรอบ แยกด้วยสี/รูปทรงไม่ได้) และตัวหนังสือบนพื้นมืด
    """
    n = len(boxes)
    if not flat_min or not n:
        return np.zeros(n, bool)
    h, w = rgb.shape[:2]
    b = boxes.astype(np.float64)
    t = (np.arange(MARK_GRID) + 0.5) / MARK_GRID * 2 - 1   # -1..1 = 2× กรอบ รอบจุดกลาง
    xx = np.clip(np.rint((b[:, 0, None] + b[:, 2, None]) / 2 + t * (b[:, 2, None] - b[:, 0, None])), 0, w - 1).astype(np.int64)
    yy = np.clip(np.rint((b[:, 1, None] + b[:, 3, None]) / 2 + t * (b[:, 3, None] - b[:, 1, None])), 0, h - 1).astype(np.int64)
    patch = rgb.reshape(-1, 3)[(yy[:, :, None] * w + xx[:, None, :]).ravel()].reshape(n * MARK_GRID, MARK_GRID, 3)
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2Lab).reshape(n, MARK_GRID, MARK_GRID, 3)   # uint8: L × 2.55 · a, b + 128
    q = MARK_GRID // 4
    ring = np.ones((MARK_GRID, MARK_GRID), bool)
    ring[q:-q, q:-q] = False
    bg = np.median(lab[:, ring], axis=1).astype(np.float32)
    d = (lab.astype(np.float32) - bg[:, None, None]) * np.float32([100 / 255, 1, 1])
    far = np.einsum("nijc,nijc->nij", d, d) >= MARK_DE ** 2
    flat = 1 - far[:, ring].mean(axis=1) >= flat_min
    ink = (far & (d[..., 0] < MARK_DE / 2))[:, q:-q, q:-q]
    m = ink.shape[1] - MARK_BLOB + 1
    solid = np.logical_and.reduce([ink[:, i:i + m, j:j + m] for i in range(MARK_BLOB) for j in range(MARK_BLOB)]).any(axis=(1, 2))
    return flat & (bg[:, 0] * (100 / 255) >= MARK_MIN_L) & ~solid


def decode_output(out: np.ndarray, conf: float, channels_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(1, 4+nc, N) [channels_first, แบบ Ultralytics] หรือ (1, N, 4+nc) → (กรอบ x0y0x1y1 ในพิกัด input, คะแนน) เฉพาะที่ ≥ conf"""
    a = np.asarray(out)
    if a.ndim != 3 or a.shape[0] != 1:
        raise ValueError(f"รูปร่าง output ของ detector ไม่รองรับ: {a.shape}")
    a = a[0].T if channels_first else a[0]   # → (N, 4+nc)
    if a.shape[1] < 5:
        raise ValueError(f"output ต้องมี ≥ 5 ช่อง (cx, cy, w, h, score…): {a.shape}")
    scores = a[:, 4:].max(axis=1)
    sel = scores >= conf
    a, scores = a[sel], scores[sel]
    boxes = np.column_stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2])
    return boxes, scores


class BeanDetector:
    """โหลดครั้งเดียว · detect() เรียกพร้อมกันหลาย thread ได้ (InferenceSession.run เป็น thread-safe)"""

    def __init__(self, onnx_path: str | Path, cfg: DetConfig | None = None):
        import onnxruntime as ort  # import ที่นี่ → backend อื่นไม่ต้องมี onnxruntime

        self.cfg = cfg or DetConfig()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.cfg.threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError(f"detector ต้องรับ input (1, 3, S, S) ขนาดคงที่: {shape}")
        self.size = int(shape[2])
        oshape = self.session.get_outputs()[0].shape
        if len(oshape) != 3 or not isinstance(oshape[1], int) or oshape[1] < 5:
            raise ValueError(f"detector ต้องให้ output (1, 4+nc, N): {oshape}")

    def _run(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lb, s, px, py = letterbox(rgb, self.size)
        x = np.ascontiguousarray(lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        boxes, scores = decode_output(self.session.run(None, {self.input_name: x})[0], self.cfg.conf)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / s
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / s
        return boxes, scores

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb uint8 (h, w, 3) → (กรอบ int (n, 4) x, y, w, h ในพิกัดภาพ, คะแนน (n,)) เรียงคะแนนมาก→น้อย"""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"ต้องเป็นภาพ RGB: {rgb.shape}")
        h, w = rgb.shape[:2]
        boxes, scores = self._run(rgb)
        if self.cfg.tile and min(h, w) >= self.size:
            th, tw = int(h * (0.5 + self.cfg.tile_overlap / 2)), int(w * (0.5 + self.cfg.tile_overlap / 2))
            for y0 in (0, h - th):
                for x0 in (0, w - tw):
                    b, s = self._run(rgb[y0:y0 + th, x0:x0 + tw])
                    b[:, [0, 2]] += x0
                    b[:, [1, 3]] += y0
                    boxes, scores = np.concatenate([boxes, b]), np.concatenate([scores, s])
        if len(scores) > self.cfg.pre_nms_top:
            top = np.argsort(-scores, kind="stable")[:self.cfg.pre_nms_top]
            boxes, scores = boxes[top], scores[top]
        keep = nms(boxes, scores, self.cfg.iou, self.cfg.max_det)
        boxes, scores = boxes[keep], scores[keep]
        real = ~container_boxes(boxes, self.cfg.container_min, self.cfg.container_area_ratio)
        boxes, scores = boxes[real], scores[real]
        real = ~flat_marks(rgb, boxes, self.cfg.mark_flat_min)
        boxes, scores = boxes[real], scores[real]
        x0 = np.clip(np.floor(boxes[:, 0]), 0, w - 1)
        y0 = np.clip(np.floor(boxes[:, 1]), 0, h - 1)
        x1 = np.clip(np.ceil(boxes[:, 2]), 1, w)
        y1 = np.clip(np.ceil(boxes[:, 3]), 1, h)
        out = np.column_stack([x0, y0, np.maximum(x1 - x0, 1), np.maximum(y1 - y0, 1)]).astype(np.int64)
        return out, scores.astype(np.float64)
