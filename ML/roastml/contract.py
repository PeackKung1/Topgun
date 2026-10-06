"""ค่าคงที่และชนิดข้อมูลของ contract FW ↔ ML v2 (ml-spec ข้อ 4)

แยกไฟล์ไว้เพื่อให้ api.py และ backend ทุกตัว import ได้โดยไม่วนกัน
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .decode import DecodedImage

API_VERSION = "2"

LABELS = ("light", "medium", "dark")
LABEL_TH = {"light": "คั่วอ่อน", "medium": "คั่วกลาง", "dark": "คั่วเข้ม"}

STATUSES = ("ok", "low_confidence", "bad_image", "error")

WARNINGS = ("no_beans_detected", "mixed_roast", "blurry", "dark_image", "colored_light")
WARNING_TH = {
    "no_beans_detected": "ไม่เห็นเมล็ดชัดเจน ลองถ่ายให้เมล็ดอยู่กลางภาพและใกล้ขึ้น",
    "mixed_roast": "ในภาพมีเมล็ดหลายระดับคั่วปนกัน",
    "blurry": "ภาพเบลอ ลองถือกล้องให้นิ่งแล้วแตะโฟกัสที่เมล็ด",
    "dark_image": "ภาพมืด ลองถ่ายในที่สว่างขึ้น",
    "colored_light": "แสงมีสีเพี้ยน ลองถ่ายใต้แสงสีขาวหรือแสงธรรมชาติ",
}

# key ทุกตัวที่ต้องมีในผลลัพธ์ (ลำดับตาม spec)
RESULT_KEYS = (
    "status", "label", "label_th", "message_th", "confidence", "probs", "warnings",
    "n_beans", "proportions", "beans", "timing_ms", "model",
)
TIMING_KEYS = ("decode", "ml", "total")


@dataclass
class BackendOutput:
    """สิ่งที่ backend คืนให้ Predictor — Predictor เป็นคนตัดสิน status/label/ข้อความ"""

    probs: dict[str, float]                            # key = LABELS ครบ 3 ตัว
    warnings: list[str] = field(default_factory=list)  # ค่าใน WARNINGS เท่านั้น
    n_beans: int | None = None
    proportions: dict[str, float] | None = None        # key = LABELS, รวม = 1
    beans: list[dict[str, Any]] = field(default_factory=list)  # [{"bbox":[x,y,w,h], "label", "conf"}]


class Backend(Protocol):
    name: str
    thread_safe: bool  # False → Predictor ล็อกให้ทีละ request

    def predict(self, img: DecodedImage) -> BackendOutput:
        """raise BadImageError ได้ถ้ารูปใช้ไม่ได้, exception อื่น = status "error\""""
        ...

    def warmup(self, img: DecodedImage) -> None: ...

    def info(self) -> dict[str, Any]: ...
