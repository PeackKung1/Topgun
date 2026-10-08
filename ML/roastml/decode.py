"""bytes → รูป RGB ที่พร้อมเข้าโมเดล (ใช้ร่วมกันทั้ง train และ Pi)

ขั้นตอน: เช็คขนาด bytes → เปิดเฉพาะ format ที่อนุญาต → เช็คจำนวน pixel จาก header
→ JPEG ใช้ draft (decode ที่สเกลเล็กลง เร็ว + ใช้ RAM น้อย) → แก้ทิศตาม EXIF
→ แปลงเป็น RGB (โปร่งใส = พื้นขาว) → ย่อด้านยาวสุดให้ ≤ MAX_SIDE

เปิดไม่ได้ทุกกรณี หรือด้านสั้นหลังย่อ < MIN_SIDE → BadImageError (api แปลงเป็น status "bad_image")
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps

log = logging.getLogger("roastml")

MAX_SIDE = 1600                 # ด้านยาวสุดหลัง decode (ml-spec ข้อ 1)
MAX_BYTES = 32 * 1024 * 1024    # ไฟล์ใหญ่กว่านี้ไม่เปิด (FW ควรตั้ง MAX_CONTENT_LENGTH ไว้ใกล้เคียงกัน)
MAX_PIXELS = 80_000_000         # กัน decompression bomb (ต่ำกว่าเกณฑ์เตือนของ Pillow ~89 MP)
# ด้านสั้นขั้นต่ำหลังย่อ = ขั้นต่ำของ segment() · เล็กกว่านี้ไม่มีข้อมูลพอจำแนก → bad_image
# (เดิม B1 ขยายภาพแล้วตอบมั่นใจ เช่น ภาพขาว 1×1 → dark 0.94) · trainval ทั้งหมดด้านสั้น ≥ 224 px
MIN_SIDE = 8

# จำกัด format: กันไม่ให้ Pillow เรียก plugin ที่เสี่ยง เช่น EPS (เรียก Ghostscript)
ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "AVIF"]

try:  # HEIC จาก iPhone — ใช้ได้ถ้าติดตั้ง pillow-heif
    import pillow_heif

    pillow_heif.register_heif_opener()
    ALLOWED_FORMATS.append("HEIF")
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# EXIF orientation 5–8 = หมุน 90° → กว้าง/สูงสลับกัน
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}
_EXIF_ORIENTATION_TAG = 0x0112


class BadImageError(Exception):
    """เปิดไฟล์เป็นรูปไม่ได้ (ไฟล์ว่าง / ไม่ใช่รูป / เสีย / ใหญ่เกิน / เล็กเกิน)"""


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image          # RGB, ด้านยาวสุด ≤ MAX_SIDE
    orig_size: tuple[int, int]  # (w, h) ของรูปต้นฉบับหลังหมุนตาม EXIF — พิกัด bbox ของ beans อ้างอิงขนาดนี้
    format: str                 # format ที่ Pillow ตรวจเจอ เช่น "JPEG"

    @property
    def scale(self) -> float:
        """ขนาดที่ใช้ / ขนาดต้นฉบับ (≤ 1)"""
        return self.image.width / self.orig_size[0]


def decode_image(data: bytes, max_side: int = MAX_SIDE) -> DecodedImage:
    if not data:
        raise BadImageError("empty")
    if len(data) > MAX_BYTES:
        raise BadImageError(f"too_many_bytes:{len(data)}")

    try:
        img = Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS)
    except Image.DecompressionBombError as e:
        raise BadImageError("too_many_pixels") from e
    except Exception as e:  # UnidentifiedImageError, OSError, SyntaxError ฯลฯ
        raise BadImageError(f"cannot_identify:{type(e).__name__}") from e

    try:
        w, h = img.size
        if w <= 0 or h <= 0:
            raise BadImageError("zero_size")
        if w * h > MAX_PIXELS:
            raise BadImageError(f"too_many_pixels:{w}x{h}")

        orientation = _read_orientation(img)
        orig_size = (h, w) if orientation in _ROTATED_ORIENTATIONS else (w, h)
        fmt = img.format or "?"

        if fmt == "JPEG":
            img.draft("RGB", draft_target(w, h, max_side))
        img.load()  # decode จริงตรงนี้ — ไฟล์เสีย/ตัดท่อนจะ error ตรงนี้
    except BadImageError:
        raise
    except Exception as e:
        raise BadImageError(f"decode_failed:{type(e).__name__}") from e

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # EXIF เพี้ยนไม่ควรทำให้ทั้งรูปใช้ไม่ได้
        log.warning("exif_transpose failed; using image as-is", exc_info=True)

    try:
        img = _to_rgb(img)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
    except Exception as e:
        raise BadImageError(f"convert_failed:{type(e).__name__}") from e

    # เช็คหลังย่อ: ภาพยาวผอม (เช่น 4000×6) ถูก thumbnail จนด้านสั้น < MIN_SIDE ได้
    if min(img.size) < MIN_SIDE:
        raise BadImageError(f"too_small:{img.size[0]}x{img.size[1]}")

    return DecodedImage(image=img, orig_size=orig_size, format=fmt)


def draft_target(w: int, h: int, max_side: int) -> tuple[int, int]:
    """ขนาดที่ขอจาก JPEG draft: กรอบตามสัดส่วนภาพที่ด้านยาว = max_side

    Pillow เลือกสเกล 1/2, 1/4, 1/8 ที่ผลยัง ≥ ขนาดที่ขอทั้งสองด้าน
    เดิมขอ (max_side, max_side) → ภาพ 4032×3024 ด้านสั้น 3024 < 2×1600 จึงไม่ถูกย่อเลย
    (วัดด้วย Pillow 12.3.0, 8 ต.ค.: ขอตามสัดส่วนได้ 2016×1512 · load 40 → 27 ms)
    """
    long_side = max(w, h)
    if long_side <= max_side:
        return (w, h)
    return (max(1, w * max_side // long_side), max(1, h * max_side // long_side))


def _read_orientation(img: Image.Image) -> int:
    try:
        return int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1))
    except Exception:
        return 1


def _to_rgb(img: Image.Image) -> Image.Image:
    """แปลงเป็น RGB; ส่วนโปร่งใสวางบนพื้นขาว (ไม่ให้กลายเป็นดำ ซึ่งจะดูเหมือนคั่วเข้ม)"""
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img
