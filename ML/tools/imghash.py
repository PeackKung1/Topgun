"""md5 + perceptual hash (pHash แบบ DCT 64 bit) สำหรับหาภาพซ้ำ — ใช้ฝั่ง notebook เท่านั้น

pHash: grayscale → resize 32×32 (INTER_AREA) → DCT 2 มิติ → มุมซ้ายบน 8×8 → bit = ค่า > median ของ 64 ค่านั้น
ระยะ = Hamming distance (0 = เหมือน, ภาพต่างกันจริงมักได้ ~32)
อ่านภาพด้วย Pillow (cv2.imread เปิด path ภาษาไทย/ยูนิโค้ดบน Windows ไม่ได้)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_gray(path: Path) -> np.ndarray:
    """เปิดภาพ (หมุนตาม EXIF) → grayscale uint8"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("L"))


def phash_array(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    bits = low > np.median(low)
    return int(np.packbits(bits).view(">u8")[0])


def thumb_array(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """ภาพย่อ float32 [0,1] ไว้คำนวณ mean abs diff ยืนยันผล pHash"""
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hamming_matrix(hashes: np.ndarray) -> np.ndarray:
    """ระยะ Hamming ทุกคู่ของ array uint64 (n,) → (n, n) uint8"""
    h = hashes.astype(np.uint64)
    x = h[:, None] ^ h[None, :]
    bytes_ = x.view(np.uint8).reshape(len(h), len(h), 8)
    return np.unpackbits(bytes_, axis=-1).sum(axis=-1, dtype=np.uint8)
