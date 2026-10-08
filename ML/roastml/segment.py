"""หา foreground (เมล็ด) ด้วย classical CV → smart crop + เลือกพิกเซลสำหรับ feature สี (B0/B1)

ปรับจากโค้ดร่าง (claude-notes/draft_ml/roastml/segment.py) ให้ตรง ml-spec v3:
- ไม่มีผลลัพธ์ "ไม่พบเมล็ด" เป็นคำตอบ: segment ล้มเหลว → ใช้ภาพเต็ม (mode="full") เสมอ
- โหมด: "beans" (พื้นหลังสม่ำเสมอ แยกเมล็ดได้) · "pile" (ขอบภาพไม่ใช่พื้นหลัง แต่สีเหมือนกองเมล็ด)
        · "full" (หา contour รูปเมล็ดไม่เจอ → ทั้งภาพ + warning no_beans_detected)
  โหมด full เลือกพิกเซลที่ "ห่างจากสีพื้นหลัง (median ขอบภาพ)" — ไม่ใช่ตัด percentile
  (แก้ 8 ต.ค.: เดิมตัด L* ต่ำสุด 15% ทิ้ง บนกระดาษขาวที่มีเมล็ดไม่กี่เมล็ด ส่วนที่ถูกทิ้งคือตัวเมล็ด → ได้กระดาษล้วน)
- white balance จากพื้นหลังขาว (von Kries ใน linear RGB) เฉพาะเมื่อขอบภาพเป็นพื้นขาวสม่ำเสมอ

แนวคิดเดิมจากร่าง: ประมาณสีพื้นหลังจากขอบภาพ (median Lab) → foreground = ΔE เกิน threshold (Otsu + floor)
→ contour กรองด้วยรูปทรง → ใช้ "แกนใน" ของแต่ละเมล็ด (distance transform) ตัดขอบที่ปนพื้นหลัง/เงา
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass
class SegConfig:
    work_side: int = 800            # ย่อก่อนทำงาน → เวลาคงที่
    border_frac: float = 0.04       # ความหนาขอบที่ใช้ประมาณพื้นหลัง (สัดส่วนด้านสั้น)
    bg_uniform_de: float = 10.0     # ΔE ที่ถือว่า "เหมือนพื้นหลัง"
    bg_uniform_min: float = 0.6     # ขอบต้องเหมือนพื้นหลัง ≥ 60% ถึงเชื่อว่ามีพื้นหลัง
    white_balance: bool = True
    wb_min_L: float = 50.0          # พื้นหลังต้องสว่างพอถึงจะใช้เป็น white reference
    wb_max_chroma: float = 20.0     # และเกือบไม่มีสี
    wb_target: float = 0.9          # map พื้นหลังไปที่ sRGB 0.9
    wb_max_gain: float = 3.0
    min_contrast: float = 12.0      # floor ของ threshold ΔE (กัน Otsu จับ noise ในภาพเปล่า)
    min_area_frac: float = 0.0004
    max_area_frac: float = 0.9
    min_rel_area: float = 0.25      # เล็กกว่า 25% ของ median = เศษ/ฝุ่น
    min_solidity: float = 0.80
    max_aspect: float = 3.0
    cluster_area_ratio: float = 2.5  # ใหญ่กว่า 2.5× median = เมล็ดติดกัน (ยังใช้สีได้)
    cluster_min_solidity: float = 0.5
    core_frac: float = 0.25         # ใช้พิกเซลที่ลึกจากขอบ ≥ 25% ของความลึกสูงสุด
    min_core_px: int = 200          # พิกเซลแกนในรวมน้อยกว่านี้ → ถือว่าไม่เจอเมล็ด
    pile_beanlike_min: float = 0.6
    crop_margin: float = 0.05       # ขยาย bbox ของ smart crop (สัดส่วนด้านยาว)
    trim_lo: float = 15.0           # โหมด pile/full: ตัดพิกเซลมืดสุด (ร่องเงา) ...
    trim_hi: float = 98.0           # ... และสว่างสุด (specular) ตาม percentile ของ L*
    trim_center: float = 0.05       # โหมด pile/full: ตัดขอบภาพออกด้านละ 5%
    full_trim_lo: float = 2.0       # โหมด full: ภายใน foreground ตัด L* มืดสุด 2% (noise) ...
    full_trim_hi: float = 90.0      # ... และสว่างสุด 10% (specular + ขอบเงาจางบนพื้น)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SegConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegResult:
    mode: str                       # beans | pile | full
    lab: np.ndarray                 # float32 Lab ของภาพ work (หลัง WB) — L 0..100
    pixel_mask: np.ndarray          # bool (h, w) พิกเซลที่ใช้คิด feature สี
    crop_box: tuple[int, int, int, int]  # x0, y0, x1, y1 ในพิกัดภาพ work (smart crop)
    scale: float                    # work / input
    wb_applied: bool
    border_uniform: float
    n_regions: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- สี
def srgb_to_lin(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def to_lab(rgb_f: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] → Lab (L 0..100, a/b ≈ -127..127)"""
    return cv2.cvtColor(np.ascontiguousarray(rgb_f, dtype=np.float32), cv2.COLOR_RGB2Lab)


def border_pixels(arr: np.ndarray, frac: float) -> np.ndarray:
    h, w = arr.shape[:2]
    b = max(1, min(int(round(min(h, w) * frac)), (min(h, w) - 1) // 2))
    c = arr.shape[2]
    parts = [arr[:b], arr[-b:], arr[b:-b, :b], arr[b:-b, -b:]]
    return np.concatenate([p.reshape(-1, c) for p in parts if p.size], axis=0)


def beanlike_mask(lab: np.ndarray) -> np.ndarray:
    """สีที่ "เป็นไปได้" สำหรับเมล็ดกาแฟ (ดิบถึงคั่วเข้ม) — heuristic หยาบ ใช้แยก pile กับภาพอื่น"""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    colored = (hue > 15) & (hue < 125) & (C > 5)
    return (L < 82) & (colored | (L < 30))


def trimmed_mask(lab: np.ndarray, cfg: SegConfig) -> np.ndarray:
    """โหมด pile/full: พิกเซลกลางภาพ (ตัดขอบ) ที่ L* อยู่ระหว่าง percentile trim_lo..trim_hi"""
    h, w = lab.shape[:2]
    m = np.zeros((h, w), bool)
    y0, y1 = int(h * cfg.trim_center), max(int(h * (1 - cfg.trim_center)), int(h * cfg.trim_center) + 1)
    x0, x1 = int(w * cfg.trim_center), max(int(w * (1 - cfg.trim_center)), int(w * cfg.trim_center) + 1)
    L = lab[y0:y1, x0:x1, 0]
    lo, hi = np.percentile(L, [cfg.trim_lo, cfg.trim_hi])
    m[y0:y1, x0:x1] = (L >= lo) & (L <= hi)
    if not m.any():  # ภาพสีเดียวทั้งภาพ
        m[y0:y1, x0:x1] = True
    return m


def _center_box(h: int, w: int, frac: float) -> tuple[int, int, int, int]:
    y0, x0 = int(h * frac), int(w * frac)
    return y0, max(int(h * (1 - frac)), y0 + 1), x0, max(int(w * (1 - frac)), x0 + 1)


def bg_distance_mask(lab: np.ndarray, bg: np.ndarray, cfg: SegConfig) -> np.ndarray | None:
    """โหมด full: พิกเซลกลางภาพที่ ΔE จากสีพื้นหลังเกิน threshold (Otsu + floor) แล้วตัด L* สุดขั้ว
    คืน None ถ้าได้พิกเซลน้อยกว่า min_core_px (ภาพแทบไม่มีอะไรต่างจากขอบ)"""
    h, w = lab.shape[:2]
    y0, y1, x0, x1 = _center_box(h, w, cfg.trim_center)
    dist = np.linalg.norm(lab - bg, axis=2)
    t_otsu, _ = cv2.threshold(np.clip(dist * 2, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = np.zeros((h, w), bool)
    fg[y0:y1, x0:x1] = dist[y0:y1, x0:x1] > max(t_otsu / 2.0, cfg.min_contrast)
    if fg.sum() < cfg.min_core_px:
        return None
    L = lab[..., 0]
    lo, hi = np.percentile(L[fg], [cfg.full_trim_lo, cfg.full_trim_hi])
    m = fg & (L >= lo) & (L <= hi)
    return m if m.sum() >= cfg.min_core_px else fg


def _odd(n: float) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


# ---------------------------------------------------------------- main
def segment(rgb: np.ndarray, cfg: SegConfig | None = None, *, find_beans: bool = True) -> SegResult:
    """rgb uint8 (h, w, 3) → SegResult · ไม่ raise กับภาพปกติ ไม่มีทางได้ "ไม่พบเมล็ด"

    find_beans=False: ข้ามการหาเมล็ด ใช้ทั้งภาพ (โหมด pile) — ใช้กับ agtron ที่ crop ROI มาแล้ว
    """
    cfg = cfg or SegConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 8:
        raise ValueError(f"ต้องเป็นภาพ RGB ขนาด ≥ 8 px: {rgb.shape}")
    h0, w0 = rgb.shape[:2]
    s = min(1.0, cfg.work_side / max(h0, w0))
    work = cv2.resize(rgb, (max(1, round(w0 * s)), max(1, round(h0 * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    f = work.astype(np.float32) / 255.0
    h, w = f.shape[:2]
    full_box = (0, 0, w, h)
    notes: list[str] = []

    lab = to_lab(f)
    bpx = border_pixels(lab, cfg.border_frac)
    bg = np.median(bpx, axis=0)
    border_uniform = float(np.mean(np.linalg.norm(bpx - bg, axis=1) < cfg.bg_uniform_de))

    # --- white balance + exposure จากพื้นหลังขาว ---
    wb = False
    if (find_beans and cfg.white_balance and border_uniform >= cfg.bg_uniform_min
            and bg[0] >= cfg.wb_min_L and float(np.hypot(bg[1], bg[2])) <= cfg.wb_max_chroma):
        lin = srgb_to_lin(f)
        bg_lin = np.median(border_pixels(lin, cfg.border_frac), axis=0)
        target = float(srgb_to_lin(np.array(cfg.wb_target)))
        gains = np.clip(target / np.maximum(bg_lin, 1e-4), 1 / cfg.wb_max_gain, cfg.wb_max_gain)
        f = lin_to_srgb(np.clip(lin * gains, 0, 1)).astype(np.float32)
        lab = to_lab(f)
        bg = np.median(border_pixels(lab, cfg.border_frac), axis=0)
        wb = True

    def fallback(mode: str, note: str | None) -> SegResult:
        if note:
            notes.append(note)
        mask = bg_distance_mask(lab, bg, cfg) if mode == "full" else None
        if mask is None:  # pile หรือ full ที่หา foreground ไม่ได้ → ตัด percentile แบบเดิม
            mask = trimmed_mask(lab, cfg)
            if mode == "full":
                notes.append("full_trimmed")
        return SegResult(mode, lab, mask, full_box, s, wb, border_uniform, 0, notes)

    if not find_beans:
        return fallback("pile", None)

    # --- ขอบภาพไม่ใช่พื้นหลัง: กองเมล็ดเต็มเฟรม หรือภาพอื่น ---
    if border_uniform < cfg.bg_uniform_min:
        if float(beanlike_mask(lab).mean()) >= cfg.pile_beanlike_min:
            return fallback("pile", None)
        return fallback("full", "no_background")

    # --- foreground = ไกลจากสีพื้นหลัง ---
    dist = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(dist * 2, 0, 255).astype(np.uint8)
    t_otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t_otsu / 2.0, cfg.min_contrast)
    mask = (dist > thr).astype(np.uint8)
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
        return fallback("full", "no_contour")

    good = [a for _, a, sol, asp in cand if sol >= cfg.min_solidity and asp <= cfg.max_aspect]
    med = float(np.median(good if good else [a for _, a, _, _ in cand]))

    core_mask = np.zeros((h, w), bool)
    boxes = []
    for c, a, sol, asp in cand:
        if a < cfg.min_rel_area * med:
            continue
        if a > cfg.cluster_area_ratio * med:
            if sol < cfg.cluster_min_solidity:
                continue
        elif sol < cfg.min_solidity or asp > cfg.max_aspect:
            continue  # รูปทรงไม่ใช่เมล็ด
        x, y, bw, bh = cv2.boundingRect(c)
        rm = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(rm, [c - [x, y]], -1, 1, thickness=cv2.FILLED)
        dt = cv2.distanceTransform(rm, cv2.DIST_L2, 3)
        core = dt >= cfg.core_frac * dt.max()
        if core.sum() < 20:
            continue
        core_mask[y:y + bh, x:x + bw] |= core
        boxes.append((x, y, x + bw, y + bh))
    if not boxes or core_mask.sum() < cfg.min_core_px:
        return fallback("full", "no_beanlike_shape")

    bx = np.array(boxes)
    m = int(round(cfg.crop_margin * max(h, w)))
    crop = (max(0, int(bx[:, 0].min()) - m), max(0, int(bx[:, 1].min()) - m),
            min(w, int(bx[:, 2].max()) + m), min(h, int(bx[:, 3].max()) + m))
    return SegResult("beans", lab, core_mask, crop, s, wb, border_uniform, len(boxes), notes)
