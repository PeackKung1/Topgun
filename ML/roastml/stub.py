"""Backend ปลอมให้ FW ทำหน้าเว็บก่อนมีโมเดลจริง — load("stub")

- รูปที่ส่งมายัง decode จริง → ไฟล์เสียได้ bad_image จริง
- รูปที่เปิดได้: สุ่ม status ทุกค่า (รวม bad_image / error จำลอง) และ warnings ทุกค่า
- seed ได้ (ลำดับผลซ้ำได้ถ้าเรียกทีละครั้ง) · หน่วงเวลาได้ (คงที่หรือช่วงสุ่ม)
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any

from .contract import LABELS, STATUSES, WARNINGS, BackendOutput
from .decode import BadImageError, DecodedImage

# สัดส่วน status เริ่มต้น — ปรับได้ด้วย load("stub", status_weights={...})
DEFAULT_STATUS_WEIGHTS = {"ok": 0.6, "low_confidence": 0.2, "bad_image": 0.1, "error": 0.1}


class SimulatedFailure(Exception):
    """stub จำลอง error ภายใน (api log แบบไม่มี traceback)"""


class StubBackend:
    name = "stub"
    thread_safe = True

    def __init__(
        self,
        seed: int | None = None,
        delay_ms: float | tuple[float, float] = 0.0,
        status_weights: dict[str, float] | None = None,
        warning_rate: float = 0.3,
        beans_rate: float = 0.3,
    ) -> None:
        weights = dict(DEFAULT_STATUS_WEIGHTS if status_weights is None else status_weights)
        unknown = set(weights) - set(STATUSES)
        if unknown:
            raise ValueError(f"status_weights มี status ที่ไม่รู้จัก: {sorted(unknown)}")
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("status_weights ต้องไม่ติดลบ และรวมกันต้อง > 0")
        if isinstance(delay_ms, (tuple, list)):
            lo, hi = float(delay_ms[0]), float(delay_ms[1])
        else:
            lo = hi = float(delay_ms)
        if lo < 0 or hi < lo:
            raise ValueError("delay_ms ต้อง ≥ 0 และ (min, max) ต้อง min ≤ max")

        self._statuses = list(weights)
        self._weights = [weights[s] for s in self._statuses]
        self._delay = (lo, hi)
        self._warning_rate = float(warning_rate)
        self._beans_rate = float(beans_rate)
        self._seed = seed
        self._rng = random.Random(seed)
        self._lock = threading.Lock()  # random.Random ไม่ thread-safe สำหรับลำดับที่ seed ไว้

    def warmup(self, img: DecodedImage) -> None:
        pass  # ไม่แตะ RNG → ลำดับผลตาม seed เริ่มจาก request แรกจริง

    def info(self) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "delay_ms": list(self._delay),
            "status_weights": dict(zip(self._statuses, self._weights)),
            "warning_rate": self._warning_rate,
            "beans_rate": self._beans_rate,
        }

    def predict(self, img: DecodedImage) -> BackendOutput:
        # สุ่มทุกอย่างใน lock ครั้งเดียว แล้วค่อย sleep นอก lock (ให้ request ซ้อนกันได้จริง)
        with self._lock:
            rng = self._rng
            status = rng.choices(self._statuses, self._weights)[0]
            delay = rng.uniform(*self._delay)
            out = self._random_output(rng, status, img.orig_size)

        if delay > 0:
            time.sleep(delay / 1000.0)
        if status == "bad_image":
            raise BadImageError("stub: simulated bad_image")
        if status == "error":
            raise SimulatedFailure("stub: simulated error")
        return out

    def _random_output(
        self, rng: random.Random, status: str, orig_size: tuple[int, int]
    ) -> BackendOutput:
        # probs สอดคล้องกับ status ที่ threshold เริ่มต้น 0.6:
        # ok → ค่าสูงสุด 0.70–0.98 · low_confidence → 0.36–0.55
        top = rng.uniform(0.70, 0.98) if status == "ok" else rng.uniform(0.36, 0.55)
        label = rng.choice(LABELS)
        others = [lb for lb in LABELS if lb != label]
        rest = 1.0 - top
        # แบ่ง rest ให้อีก 2 คลาสโดยไม่มีคลาสไหน ≥ top (ทำได้เพราะ top > 1/3)
        a = rng.uniform(max(0.0, rest - top + 1e-3), min(rest, top - 1e-3))
        probs = {label: top, others[0]: a, others[1]: rest - a}

        warnings = [w for w in WARNINGS if rng.random() < self._warning_rate]

        beans: list[dict[str, Any]] = []
        n_beans = proportions = None
        if rng.random() < self._beans_rate:
            w, h = orig_size
            for _ in range(rng.randint(1, 12)):
                bw = max(1, int(w * rng.uniform(0.05, 0.2)))
                bh = max(1, int(h * rng.uniform(0.05, 0.2)))
                x = rng.randint(0, max(0, w - bw))
                y = rng.randint(0, max(0, h - bh))
                bean_label = label if rng.random() < 0.8 else rng.choice(LABELS)
                beans.append({
                    "bbox": [x, y, bw, bh],
                    "label": bean_label,
                    "conf": round(rng.uniform(0.4, 0.99), 4),
                })
            n_beans = len(beans)
            proportions = {lb: sum(b["label"] == lb for b in beans) / n_beans for lb in LABELS}

        return BackendOutput(
            probs=probs, warnings=warnings, n_beans=n_beans,
            proportions=proportions, beans=beans,
        )
