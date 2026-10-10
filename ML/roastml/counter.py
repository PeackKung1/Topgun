"""ตัวนับเมล็ด v2 (OpenCV + numpy) — แทนตัวนับ edge-contour ของ PR #1 · ใช้ทั้งตอนเทรนและบน Pi

ลำดับต่อภาพ (ภาพ work ≤ work_side, WB ด้วย segment.background_white_balance ตัวเดียวกับ B1):
  1. ตัดสิน scene: ขอบภาพไม่สม่ำเสมอ + สีเหมือนเมล็ดเกือบทั้งภาพ → pile · ไม่งั้น → foreground จากพื้นหลัง
  2. foreground: ΔE (ถ่วง L* ด้วย fg_L_weight กันเงา) จากพื้นหลัง — พื้นสม่ำเสมอใช้ median ขอบภาพ
     ไม่สม่ำเสมอ (แสงไล่ระดับ/เงา) ใช้พื้นหลังเฉพาะที่ = median blur ขนาดใหญ่
  3. blob: ตัดเศษ + ตัดสิ่งที่สีไม่เหมือนเมล็ด (beanlike) · ขนาดอ้างอิง = median ของ blob เดี่ยว
     (ไม่มี blob เดี่ยว → peak ของ distance transform)
  4. blob เดี่ยว → 1 · blob ใหญ่ (แตะกัน) → touch_method: "area" = round(area/ref) แบ่งพิกเซลด้วย k-means
     ของพิกัด · "watershed" = marker จาก distance transform
  5. pile: ร่องมืดระหว่างเมล็ด (adaptive threshold) → distance transform → marker ขนาดสอดคล้องกับเมล็ด
     → นับได้เฉพาะเมล็ดที่เห็น (method = "estimated" → warning count_visible_only + bean_count_estimated)
ผล: label image (0 = ไม่ใช่เมล็ด, 1..n) + core mask สำหรับคิดสีรายเมล็ด + bbox ในพิกัดภาพ decode
parameter ทุกตัวอยู่ใน CountConfig (เก็บใน model_card.json["count_config"]) · ตั้งค่าบน dev เท่านั้น
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .segment import SegConfig, PreparedImage, prepare_image, beanlike_mask


@dataclass
class CountConfig:
    # --- scene ---
    pile_beanlike_min: float = 0.6     # ขอบไม่สม่ำเสมอ + beanlike ≥ นี้ → pile (กฎเดียวกับ segment)
    pile_fg_frac: float = 0.6          # foreground ครอบ > นี้ของภาพ → pile
    # --- foreground ---
    fg_L_weight: float = 0.5           # น้ำหนัก ΔL* ใน ΔE (< 1 = เงาเทาบนพื้นไม่กลายเป็นเมล็ด)
    fg_min_contrast: float = 12.0      # floor ของ threshold (ภาพว่างไม่ให้ Otsu จับ noise)
    bg_local_side: int = 96            # ย่อภาพก่อนประมาณพื้นหลังเฉพาะที่
    bg_local_kernel_frac: float = 0.45  # kernel median blur (สัดส่วนด้านสั้นของภาพย่อ)
    open_frac: float = 0.006           # morphology open/close (สัดส่วนด้านสั้น)
    # --- blob ---
    min_area_frac: float = 0.0002      # เล็กกว่านี้ของภาพ = ฝุ่น
    min_rel_area: float = 0.3          # เล็กกว่า min_rel_area × ref = เศษ
    min_visible_fraction: float = 0.5 # visible region / reference bean area (heuristic, needs real GT)
    min_beanlike: float = 0.5          # พื้นหลังไม่สม่ำเสมอ: สัดส่วนพิกเซลสีเหมือนเมล็ดใน blob ขั้นต่ำ
    min_beanlike_uniform: float = 0.0  # พื้นสม่ำเสมอ (WB แล้ว): ปิดไว้ — เมล็ดอมฟ้า/เทาไม่ผ่าน hue ของ beanlike
    min_ref_frac: float = 0.0          # พื้นที่เมล็ดอ้างอิงขั้นต่ำ (สัดส่วนภาพ) · น้อยกว่า = ไม่ใช่เมล็ด → 0
    cluster_min_solidity: float = 0.5  # blob ใหญ่ที่ solidity ต่ำกว่านี้ = ไม่ใช่กลุ่มเมล็ด (สาย/ขอบวัตถุ)
    single_min_solidity: float = 0.88
    single_max_aspect: float = 2.6
    single_max_ratio: float = 1.6      # area ≤ ratio × ref → 1 เมล็ด
    bean_aspect: float = 1.4           # ใช้แปลง peak DT (รัศมีด้านสั้น) เป็นพื้นที่: π r² × aspect
    touch_method: str = "watershed"    # "area" | "watershed"
    area_iterations: int = 30
    area_sample_limit: int = 4096
    area_assign_chunk: int = 4096
    area_epsilon: float = 0.5
    ref_radius_floor_frac: float = 0.4 # discard crumbs relative to deepest foreground
    metal_chroma_min: float = 8.0
    metal_colored_fraction: float = 0.3
    neutral_background_L: float = 50.0
    ws_peak_frac: float = 0.55         # marker ต้องมี DT ≥ frac × รัศมีอ้างอิง
    ws_separation: float = 1.0         # DT local-max kernel diameter / reference radius
    dt_min_peak_px: float = 2.0        # peak ของ DT ที่ต่ำกว่านี้ = noise (ใช้ตอนไม่มี blob เดี่ยว)
    # --- pile: ร่องมืดระหว่างเมล็ด → ภายในเมล็ด → ขนาดอ้างอิง (area-weighted) → DT marker ---
    pile_det_side: int = 400           # ย่อภาพก่อนหา (คุมเวลา)
    pile_block_frac: float = 0.10      # adaptive threshold block (สัดส่วนด้านสั้น)
    pile_gap_c: float = 4.0            # มืดกว่าค่าเฉลี่ยเฉพาะที่ ≥ c (L*) = ร่อง
    pile_blur_frac: float = 0.006      # blur ก่อน threshold (กันลายผิว/ร่องกลางเมล็ด)
    pile_ref_pct: float = 60.0         # รัศมีอ้างอิง = percentile ของ DT สูงสุดต่อชิ้นภายในเมล็ด
    pile_peak_frac: float = 0.45       # marker ต้องมี DT ≥ frac × รัศมีอ้างอิง
    pile_sep: float = 1.2              # marker ห่างกัน ≥ sep × รัศมีอ้างอิง
    pile_region_beanlike: bool = True  # จำกัดไว้ในพิกเซลสีเหมือนเมล็ด (ตัดโลหะ/กระดาษ) · ROI ที่รู้ว่าเป็นกอง = ไม่ใช้
    # --- core (พิกเซลคิดสี) ---
    core_frac: float = 0.35            # ลึกจากขอบเมล็ด ≥ frac × ความลึกสูงสุดของเมล็ดนั้น
    max_beans: int = 1000              # กันภาพ noise ระเบิด (เกิน → ตัดเหลือ blob ใหญ่สุด)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CountConfig":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"count_config ไม่รู้จัก key: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.touch_method not in ("area", "watershed"):
            raise ValueError(f"touch_method ไม่รองรับ: {cfg.touch_method!r}")
        for name, value in asdict(cfg).items():
            if isinstance(value, (int, float)) and (not np.isfinite(value) or value < 0):
                raise ValueError(f"invalid count_config {name}")
        if not isinstance(cfg.max_beans, int) or not 1 <= cfg.max_beans <= 10000:
            raise ValueError("max_beans must be an integer in [1,10000]")
        if not 0 < cfg.core_frac <= 1 or not 0 < cfg.ref_radius_floor_frac <= 1:
            raise ValueError("invalid core/reference radius fraction")
        if not 0.5 <= cfg.min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be in [.5,1]")
        for name in ('area_iterations','area_sample_limit','area_assign_chunk'):
            if not isinstance(getattr(cfg,name),int) or getattr(cfg,name) < 1:
                raise ValueError('invalid area solver configuration')
        if cfg.bg_local_side < 8 or cfg.pile_det_side < 8 or not 0 < cfg.bg_local_kernel_frac < 1:
            raise ValueError("invalid background/pile resolution")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CountResult:
    scene: str                    # flat | touching | pile | empty
    method: str                   # exact | split | estimated | none
    labels: np.ndarray            # int32 (h, w) ภาพ work · 0 = พื้นหลัง · 1..n = เมล็ด
    core: np.ndarray              # bool (h, w) พิกเซลคิดสี (อยู่ในเมล็ดใดเมล็ดหนึ่ง)
    n: int
    bboxes: np.ndarray            # (n, 4) int x, y, w, h ในพิกัดภาพ decode
    lab: np.ndarray               # float32 Lab ของภาพ work (หลัง WB)
    scale: float                  # work / decode
    wb_applied: bool
    ref_area: float = 0.0         # พื้นที่เมล็ดอ้างอิง (px ในภาพ work)
    notes: list[str] = field(default_factory=list)


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _relabel(a: np.ndarray) -> np.ndarray:
    """label ให้ต่อเนื่อง 0..k โดย 0 คงเป็น 0 เสมอ"""
    _, inv = np.unique(np.concatenate([[0], a.ravel()]), return_inverse=True)
    return inv[1:].reshape(a.shape).astype(np.int32)


def _lab_u8(lab: np.ndarray) -> np.ndarray:
    return np.dstack([np.clip(lab[..., 0] * 2.55, 0, 255), np.clip(lab[..., 1] + 128, 0, 255),
                      np.clip(lab[..., 2] + 128, 0, 255)]).astype(np.uint8)


def local_background(lab: np.ndarray, cfg: CountConfig) -> np.ndarray:
    """พื้นหลังเฉพาะที่: ย่อภาพ → median blur kernel ใหญ่ (ใหญ่กว่าเมล็ด) → ขยายกลับ · คืน Lab float32"""
    h, w = lab.shape[:2]
    s = cfg.bg_local_side / max(h, w)
    small = cv2.resize(_lab_u8(lab), (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    k = _odd(min(small.shape[:2]) * cfg.bg_local_kernel_frac)
    k = min(k, _odd(min(small.shape[:2]) - 2))
    bg = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.dstack([bg[..., 0] / 2.55, bg[..., 1] - 128, bg[..., 2] - 128])


def foreground(lab: np.ndarray, bg: np.ndarray, cfg: CountConfig) -> np.ndarray:
    """ΔE ถ่วง L* จากพื้นหลัง > max(Otsu/2, floor) → open/close → เติมรู (specular กลางเมล็ด)"""
    h, w = lab.shape[:2]
    d = lab - bg
    dist = np.sqrt((cfg.fg_L_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    t_otsu, _ = cv2.threshold(np.clip(dist * 2, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dist > max(t_otsu / 2.0, cfg.fg_min_contrast)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(h, w) * cfg.open_frac),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # เติมรู: พื้นที่ที่ไม่ต่อกับขอบภาพหลัง flood fill = รูในเมล็ด
    inv = (1 - mask).astype(np.uint8)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n > 1:
        x, y, ww, hh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
        touches = (x == 0) | (y == 0) | (x + ww >= w) | (y + hh >= h)
        holes = ~touches
        holes[0] = False
        mask[holes[lab_cc]] = 1
    return mask.astype(bool)


def _blob_shapes(cc: np.ndarray, stats: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """solidity + aspect ต่อ blob (loop ต่อ blob ไม่ใช่ต่อพิกเซล · ใช้ contour ของ OpenCV)"""
    sol, asp = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        sub = (cc[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        a = float(stats[i, cv2.CC_STAT_AREA])
        sol[i] = a / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        if len(c) >= 5:
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            asp[i] = max(rw, rh) / max(min(rw, rh), 1.0)
        else:
            asp[i] = 1.0
    return sol, asp


def _dt_peaks(dt: np.ndarray, sep: int, min_val: float) -> np.ndarray:
    """local maxima ของ distance transform (ห่างกัน ≥ ~sep px) · คืน marker int32 (0 = ไม่ใช่)"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(sep),) * 2)
    peak = (dt >= cv2.dilate(dt, k) - 1e-6) & (dt >= min_val)
    # plateau ติดกันนับเป็น marker เดียว
    _, markers = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
    return markers.astype(np.int32)


def _ref_area_from_dt(mask: np.ndarray, cfg: CountConfig) -> float:
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dt.max() <= 0:
        return 0.0
    markers = _dt_peaks(dt, max(3, dt.max()), cfg.dt_min_peak_px)
    vals = dt[(markers > 0) & (dt >= cfg.ref_radius_floor_frac * dt.max())]
    r = float(np.median(vals)) if vals.size else float(dt.max())
    return float(np.pi * r * r * cfg.bean_aspect)


def _split_area(sub: np.ndarray, k: int, cfg: CountConfig) -> np.ndarray:
    """Deterministic coordinate k-means; bounded arrays, no global RNG mutation."""
    ys, xs = np.nonzero(sub)
    out = np.zeros(sub.shape, np.int32)
    if k <= 1 or len(xs) < k:
        out[ys, xs] = 1
        return out
    pts = np.column_stack([xs, ys]).astype(np.float32)
    sample = pts[np.linspace(0,len(pts)-1,min(len(pts),max(k,cfg.area_sample_limit))).astype(int)]
    _, vectors = np.linalg.eigh(np.cov(sample.T))
    axis = vectors[:,-1]
    axis *= 1 if axis[np.abs(axis).argmax()] >= 0 else -1
    ordered = sample[np.argsort(sample @ axis,kind='stable')]
    centers = ordered[np.minimum(((np.arange(k)+.5)*len(ordered)/k).astype(int),len(ordered)-1)].copy()
    for _ in range(cfg.area_iterations):
        labels = ((sample[:,None]-centers)**2).sum(2).argmin(1)
        new = centers.copy()
        for j in range(k):
            if np.any(labels == j): new[j] = sample[labels == j].mean(0)
        if np.max(np.abs(new-centers)) <= cfg.area_epsilon:
            centers = new
            break
        centers = new
    for start in range(0,len(pts),cfg.area_assign_chunk):
        end = min(len(pts),start+cfg.area_assign_chunk)
        label = ((pts[start:end,None]-centers)**2).sum(2).argmin(1)+1
        out[ys[start:end],xs[start:end]] = label
    out = _relabel(out)
    return out


def _split_watershed(sub: np.ndarray, r_ref: float, cfg: CountConfig) -> np.ndarray:
    dt = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5)
    markers = _dt_peaks(dt, max(3.0, cfg.ws_separation * r_ref), cfg.ws_peak_frac * r_ref)
    if markers.max() <= 1:
        return sub.astype(np.int32)
    # cv2.watershed floods image gradients, not scalar terrain heights.
    # A flat foreground with a sharp exterior boundary keeps marker regions
    # balanced; using inverted DT as RGB makes a single basin dominate.
    img = cv2.cvtColor(sub.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    m = markers.copy()
    m[~sub] = 0
    bgm = m.max() + 1
    m[(~sub)] = bgm  # นอก blob = marker พื้นหลัง
    cv2.watershed(img, m)
    out = np.where(sub & (m > 0) & (m < bgm), m, 0).astype(np.int32)
    # OpenCV watershed can let its exterior marker enter the foreground.
    # All foreground pixels must belong to a bean; assign missing pixels to
    # their nearest foreground marker, instead of one dilation that loses area.
    out = _complete_regions(out, sub, markers)
    return _relabel(out)


def _complete_regions(labels: np.ndarray, region: np.ndarray, markers: np.ndarray) -> np.ndarray:
    missing = region & (labels == 0)
    if missing.any() and markers.max() > 0:
        _, nearest = cv2.distanceTransformWithLabels((markers == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_CCOMP)
        mapping = np.zeros(int(nearest.max())+1,np.int32)
        selected = markers > 0
        mapping[nearest[selected]] = markers[selected]
        labels[missing] = mapping[nearest[missing]]
    return labels


def _pile_labels(lab: np.ndarray, region: np.ndarray, cfg: CountConfig) -> tuple[np.ndarray, float]:
    """กอง: ร่องมืด (adaptive threshold บน L* ที่ blur) → ภายในเมล็ด → รัศมีอ้างอิงจาก component ใหญ่ (ถ่วงพื้นที่)
    → marker = peak ของ DT ที่ลึกพอและห่างกันพอ → watershed ภายในพื้นที่เมล็ด · นับได้เฉพาะเมล็ดที่เห็น"""
    h, w = lab.shape[:2]
    s = min(1.0, cfg.pile_det_side / max(h, w))
    hd, wd = max(8, round(h * s)), max(8, round(w * s))
    L8 = cv2.resize(np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8), (wd, hd), interpolation=cv2.INTER_AREA)
    reg = cv2.resize(region.astype(np.uint8), (wd, hd), interpolation=cv2.INTER_NEAREST).astype(bool)
    side = min(hd, wd)
    L8 = cv2.GaussianBlur(L8, (0, 0), max(0.5, cfg.pile_blur_frac * side))
    # THRESH_BINARY: พิกเซล > (mean - C) → ภายในเมล็ด · มืดกว่า mean เกิน c = ร่อง
    interior = cv2.adaptiveThreshold(L8, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                     _odd(side * cfg.pile_block_frac), cfg.pile_gap_c * 2.55)
    interior = (interior.astype(bool) & reg).astype(np.uint8)
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cc, cc = cv2.connectedComponents(interior, connectivity=4)
    if n_cc <= 1:
        return np.zeros((h, w), np.int32), 0.0
    dt = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    # รัศมีด้านสั้นอ้างอิง = percentile ของ "DT สูงสุดต่อชิ้น" (ชิ้นที่เมล็ดติดกันยังกว้างเท่าเมล็ดเดียว → ทนการติดกัน)
    mx = np.zeros(n_cc)
    np.maximum.at(mx, cc.ravel(), dt.ravel())
    mx = mx[1:][mx[1:] >= 2.0]
    if mx.size == 0:
        return np.zeros((h, w), np.int32), 0.0
    r_ref = float(np.percentile(mx, cfg.pile_ref_pct))
    ref_area = float(np.pi * r_ref * r_ref * cfg.bean_aspect)
    markers = _dt_peaks(dt, max(3.0, cfg.pile_sep * r_ref), max(1.0, cfg.pile_peak_frac * r_ref))
    if markers.max() == 0:
        return np.zeros((h, w), np.int32), 0.0
    m = markers.copy()
    bgm = int(m.max()) + 1
    m[~reg] = bgm
    img = cv2.cvtColor(np.clip(255 - dt * (255.0 / max(dt.max(), 1e-6)), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, m)
    out = np.where(interior.astype(bool) & (m > 0) & (m < bgm), m, 0).astype(np.int32)
    out = _complete_regions(out, interior.astype(bool), markers)
    out = _relabel(out)
    if s < 1:
        out = cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        out[~region] = 0
        out = _relabel(out)
    return out, float(ref_area / (s * s))


def _core_and_boxes(labels: np.ndarray, n: int, cfg: CountConfig, inv_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """core = ลึกจากขอบเมล็ด (รวมเส้นแบ่งระหว่างเมล็ด) ≥ core_frac × max ของเมล็ดนั้น · vectorized ต่อ label"""
    if n == 0:
        return np.zeros(labels.shape, bool), np.zeros((0, 4), np.int64)
    edge = np.zeros(labels.shape, bool)  # พิกเซลที่ label ต่างจากเพื่อนบ้าน = ขอบเมล็ด
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge[[0,-1],:] = True
    edge[:,[0,-1]] = True
    inside = ((labels > 0) & ~edge).astype(np.uint8)
    dt = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    lab_flat, dt_flat = labels.ravel(), dt.ravel()
    sel = lab_flat > 0
    mx = np.zeros(n + 1)
    np.maximum.at(mx, lab_flat[sel], dt_flat[sel])
    core = (labels > 0) & (dt >= cfg.core_frac * mx[labels]) & (dt > 0)
    # เมล็ดเล็กมากที่ core ว่าง → ใช้ทุกพิกเซลของเมล็ด
    has_core = np.bincount(labels[core], minlength=n + 1) > 0
    core |= (labels > 0) & ~has_core[labels]
    ys, xs = np.nonzero(labels)
    ll = labels[ys, xs]
    x0 = np.full(n + 1, np.iinfo(np.int64).max); y0 = x0.copy()
    x1 = np.full(n + 1, -1); y1 = x1.copy()
    np.minimum.at(x0, ll, xs); np.minimum.at(y0, ll, ys)
    np.maximum.at(x1, ll, xs); np.maximum.at(y1, ll, ys)
    b = np.column_stack([x0[1:], y0[1:], x1[1:] - x0[1:] + 1, y1[1:] - y0[1:] + 1]).astype(np.float64)
    b = np.rint(b * inv_scale).astype(np.int64)
    b[:, 2:] = np.maximum(b[:, 2:], 1)
    return core, b


def count_beans(rgb: np.ndarray, cfg: CountConfig | None = None, seg_cfg: SegConfig | None = None,
                *, force_pile: bool = False, prepared: PreparedImage | None = None) -> CountResult:
    """rgb uint8 (h, w, 3) ภาพ decode → CountResult · ไม่ raise กับภาพปกติ · ภาพว่าง → n = 0

    force_pile=True: ภาพที่รู้ว่าเป็นชั้นเมล็ดเต็มกรอบ (ROI ของ agtron ตอนเทรน/ประเมิน)
    """
    cfg = cfg or CountConfig()
    seg_cfg = seg_cfg or SegConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 8:
        raise ValueError(f"ต้องเป็นภาพ RGB ขนาด ≥ 8 px: {rgb.shape}")
    prepared = prepared or prepare_image(rgb, seg_cfg, allow_wb=not force_pile)
    s, wbr = prepared.scale, prepared.wb
    inv = 1.0 / s if s > 0 else 1.0
    lab = wbr.lab
    h, w = lab.shape[:2]
    notes: list[str] = []
    uniform = wbr.border_uniform >= seg_cfg.bg_uniform_min
    beanlike = beanlike_mask(lab)
    # Dark neutral metal is admitted by the coarse pile mask. Use positive
    # brown chroma evidence on textured/dark backgrounds, without rejecting
    # blue-tinted Boos beans on a valid bright white reference.
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    colored = (np.hypot(lab[..., 1], lab[..., 2]) >= cfg.metal_chroma_min) & (hue > 15) & (hue < 125)
    require_color = not uniform or wbr.bg[0] < cfg.neutral_background_L
    if require_color:
        beanlike &= colored

    def finish(scene: str, method: str, labels: np.ndarray, ref: float) -> CountResult:
        n = int(labels.max())
        if n and ref > 0:
            areas = np.bincount(labels.ravel(),minlength=n+1)
            keep = areas >= cfg.min_visible_fraction * ref
            keep[0] = False
            if not np.all(keep[1:]):
                labels = _relabel(np.where(keep[labels],labels,0))
                n = int(labels.max())
                notes.append('visible_area_filter')
        if n > cfg.max_beans:  # เก็บ max_beans เมล็ดใหญ่สุด
            areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
            keep = np.zeros(n + 1, bool)
            keep[1 + np.argsort(-areas)[:cfg.max_beans]] = True
            labels = np.where(keep[labels], labels, 0)
            labels = _relabel(labels)
            n = int(labels.max())
            notes.append("max_beans_cap")
            method = "estimated"
        core, boxes = _core_and_boxes(labels, n, cfg, inv)
        return CountResult(scene if n else "empty", method if n else "none", labels, core, n, boxes,
                           lab, s, wbr.applied, ref, notes)

    if force_pile or (not uniform and float(beanlike.mean()) >= cfg.pile_beanlike_min):
        region = np.ones((h, w), bool)
        if cfg.pile_region_beanlike and not force_pile:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(h, w) * 0.02),) * 2)
            region = cv2.morphologyEx(beanlike.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        labels, ref = _pile_labels(lab, region, cfg)
        return finish("pile", "estimated", labels, ref)

    bg = np.broadcast_to(wbr.bg.astype(np.float32), lab.shape) if uniform else local_background(lab, cfg)
    if not uniform:
        notes.append("local_background")
    fg = foreground(lab, bg, cfg)
    if fg.mean() > cfg.pile_fg_frac and float(beanlike[fg].mean() if fg.any() else 0) >= cfg.pile_beanlike_min:
        labels, ref = _pile_labels(lab, fg, cfg)
        return finish("pile", "estimated", labels, ref)

    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    bl = np.bincount(cc.ravel(), weights=beanlike.ravel().astype(np.float64), minlength=n_cc) / np.maximum(areas, 1)
    keep = np.zeros(n_cc, bool)
    keep[1:] = (areas[1:] >= cfg.min_area_frac * h * w) & (
        bl[1:] >= (cfg.min_beanlike_uniform if uniform else cfg.min_beanlike))
    if require_color:
        color_fraction = np.bincount(cc.ravel(), weights=colored.ravel(), minlength=n_cc) / np.maximum(areas, 1)
        keep &= color_fraction >= cfg.metal_colored_fraction
    if not keep.any():
        return finish("empty", "none", np.zeros((h, w), np.int32), 0.0)
    sol, asp = _blob_shapes(cc, stats, n_cc)
    single_shape = keep & (sol >= cfg.single_min_solidity) & (asp <= cfg.single_max_aspect)
    # Tiny regular dust must not define the size of an irregular real bean.
    dt = cv2.distanceTransform(np.isin(cc, np.flatnonzero(keep)).astype(np.uint8), cv2.DIST_L2, 5)
    depth = np.zeros(n_cc)
    np.maximum.at(depth, cc.ravel(), dt.ravel())
    single_shape &= depth >= cfg.ref_radius_floor_frac * depth.max()
    if single_shape.any():
        ref = float(np.median(areas[single_shape]))
        singles = single_shape & (areas <= cfg.single_max_ratio * ref)
        if singles.any():
            ref = float(np.median(areas[singles]))
    else:
        ref = _ref_area_from_dt(np.isin(cc, np.flatnonzero(keep)), cfg)
        notes.append("ref_from_dt")
    if ref <= 0 or ref < cfg.min_ref_frac * h * w:
        return finish("empty", "none", np.zeros((h, w), np.int32), 0.0)
    keep &= areas >= cfg.min_rel_area * ref
    keep &= ~((areas > cfg.single_max_ratio * ref) & (sol < cfg.cluster_min_solidity))  # สาย/ขอบวัตถุ
    keep &= (areas > cfg.single_max_ratio * ref) | ((sol >= cfg.cluster_min_solidity) & (asp <= cfg.single_max_aspect))
    r_ref = float(np.sqrt(ref / (np.pi * cfg.bean_aspect)))

    labels = np.zeros((h, w), np.int32)
    nxt, split_any = 0, False
    big = keep & (areas > cfg.single_max_ratio * ref)
    # blob เดี่ยว: relabel ทั้งหมดทีเดียว (vectorized)
    single_ids = np.flatnonzero(keep & ~big)
    lut = np.zeros(n_cc, np.int32)
    lut[single_ids] = np.arange(1, len(single_ids) + 1, dtype=np.int32)
    labels = lut[cc]
    nxt = len(single_ids)
    for i in np.flatnonzero(big):  # loop ต่อกลุ่มที่แตะกัน (ไม่ใช่ต่อเมล็ด)
        x, y, bw, bh = stats[i, :4]
        sub = cc[y:y + bh, x:x + bw] == i
        if cfg.touch_method == "area":
            part = _split_area(sub, min(cfg.max_beans, max(1, int(round(areas[i] / ref)))), cfg)
        else:
            part = _split_watershed(sub, r_ref, cfg)
        k = int(part.max())
        if k == 0:
            continue
        split_any |= k > 1
        view = labels[y:y + bh, x:x + bw]
        view[part > 0] = part[part > 0] + nxt
        nxt += k
    scene = "touching" if split_any else "flat"
    return finish(scene, "split" if split_any else "exact", labels, ref)
