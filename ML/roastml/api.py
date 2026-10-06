"""Contract FW ↔ ML v2 (ml-spec ข้อ 4) — FW เรียกใช้ไฟล์นี้เท่านั้น

    from roastml.api import load
    predictor = load("stub")                  # หรือ load("ML/models/current") เมื่อมีโมเดล
    result = predictor.predict_bytes(raw)     # ไม่ raise · thread-safe · คืน dict ที่ json.dumps ได้

- `load()` raise ได้ (เรียกครั้งเดียวตอน start ให้พังเร็วถ้าตั้งค่าผิด)
- `predict_bytes()` ห้าม raise: ทุกความผิดพลาดกลายเป็น status "bad_image" หรือ "error"
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from . import __version__
from .contract import (
    API_VERSION, LABEL_TH, LABELS, RESULT_KEYS, STATUSES, WARNING_TH, WARNINGS,
    Backend, BackendOutput,
)
from .decode import HEIC_SUPPORTED, MAX_BYTES, MAX_SIDE, BadImageError, DecodedImage, decode_image
from .stub import SimulatedFailure, StubBackend

__all__ = [
    "load", "Predictor", "ModelLoadError",
    "LABELS", "LABEL_TH", "STATUSES", "WARNINGS", "WARNING_TH", "RESULT_KEYS",
]

log = logging.getLogger("roastml")

# ความมั่นใจต่ำกว่านี้ → low_confidence
# ค่าชั่วคราว: ค่าจริงจะเลือกด้วย LOSO แล้วเก็บใน model_card.json ของโมเดล (ห้ามใช้ test set เลือก)
DEFAULT_LOW_CONF_THRESHOLD = 0.6

MSG_BAD_IMAGE = "เปิดไฟล์รูปนี้ไม่ได้ กรุณาเลือกไฟล์รูป (JPG, PNG หรือ HEIC) ใหม่อีกครั้ง"
MSG_ERROR = "ระบบขัดข้องชั่วคราว กรุณาลองส่งรูปใหม่อีกครั้ง"


class ModelLoadError(RuntimeError):
    """โหลดโมเดลไม่ได้ (โฟลเดอร์ผิด / model_card เสีย / ยังไม่มี backend)"""


class Predictor:
    """ห่อ backend ให้ได้ผลตาม contract — decode, จับ exception, ตัดสิน status, จับเวลา"""

    def __init__(self, backend: Backend, low_conf_threshold: float = DEFAULT_LOW_CONF_THRESHOLD):
        if not 0.0 <= low_conf_threshold <= 1.0:
            raise ValueError("low_conf_threshold ต้องอยู่ใน [0, 1]")
        self.backend = backend
        self.low_conf_threshold = float(low_conf_threshold)
        # backend ที่ไม่ thread-safe → ให้ทำทีละ request (decode ยังขนานได้)
        self._backend_lock = None if getattr(backend, "thread_safe", False) else threading.Lock()
        self._load_ms: float | None = None

    # ------------------------------------------------------------------ public
    def predict_bytes(self, raw: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return self._predict(raw, t0)
        except Exception:  # ด่านสุดท้าย — ไม่ควรมาถึงตรงนี้
            log.exception("unexpected error in predict_bytes")
            try:
                return self._failure("error", t0, decode_ms=0.0)
            except Exception:
                return _hardcoded_error()

    def info(self) -> dict[str, Any]:
        try:
            backend_info = self.backend.info()
        except Exception:
            log.exception("backend.info() failed")
            backend_info = {"error": "backend.info() failed"}
        return {
            "api_version": API_VERSION,
            "roastml_version": __version__,
            "model": self.backend.name,
            "backend": backend_info,
            "labels": list(LABELS),
            "statuses": list(STATUSES),
            "warnings": list(WARNINGS),
            "low_conf_threshold": self.low_conf_threshold,
            "max_side": MAX_SIDE,
            "max_bytes": MAX_BYTES,
            "heic_supported": HEIC_SUPPORTED,
            "load_ms": self._load_ms,
            "python": platform.python_version(),
            "machine": platform.machine(),
        }

    def warmup(self) -> None:
        """decode รูปสังเคราะห์ 1 รูป + ให้ backend อุ่นเครื่อง (โหลด session, จัดสรร memory)"""
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (120, 80, 50)).save(buf, format="JPEG")
        img = decode_image(buf.getvalue())
        self.backend.warmup(img)

    # ----------------------------------------------------------------- internal
    def _predict(self, raw: Any, t0: float) -> dict[str, Any]:
        if raw is None:
            return self._failure("bad_image", t0, decode_ms=0.0)
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            # caller ส่งผิดชนิด (เช่น path เป็น str) = บั๊กฝั่งเรียก ไม่ใช่รูปเสีย
            log.error("predict_bytes expects bytes, got %s", type(raw).__name__)
            return self._failure("error", t0, decode_ms=0.0)

        try:
            img = decode_image(bytes(raw))
        except BadImageError as e:
            log.info("bad_image: %s", e)
            return self._failure("bad_image", t0, decode_ms=_ms_since(t0))
        t_decoded = time.perf_counter()
        decode_ms = (t_decoded - t0) * 1000.0

        try:
            out = self._run_backend(img)
            label, conf, probs = self._check_probs(out)
        except BadImageError as e:
            log.info("bad_image from backend: %s", e)
            return self._failure("bad_image", t0, decode_ms=decode_ms)
        except SimulatedFailure as e:
            log.info("%s", e)
            return self._failure("error", t0, decode_ms=decode_ms)
        except Exception:
            log.exception("backend %s failed", self.backend.name)
            return self._failure("error", t0, decode_ms=decode_ms)

        ml_ms = _ms_since(t_decoded)
        status = "ok" if conf >= self.low_conf_threshold else "low_confidence"
        warnings = _clean_warnings(out.warnings)
        return {
            "status": status,
            "label": label,
            "label_th": LABEL_TH[label],
            "message_th": _message(status, label, conf, warnings),
            "confidence": round(conf, 4),
            "probs": {k: round(v, 4) for k, v in probs.items()},
            "warnings": warnings,
            "n_beans": out.n_beans,
            "proportions": _clean_proportions(out.proportions),
            "beans": _clean_beans(out.beans),
            "timing_ms": _timing(decode_ms, ml_ms, _ms_since(t0)),
            "model": self.backend.name,
        }

    def _run_backend(self, img: DecodedImage) -> BackendOutput:
        if self._backend_lock is None:
            return self.backend.predict(img)
        with self._backend_lock:
            return self.backend.predict(img)

    @staticmethod
    def _check_probs(out: BackendOutput) -> tuple[str, float, dict[str, float]]:
        """ตรวจ probs จาก backend ให้ครบ 3 คลาส เป็นตัวเลขจริง ≥ 0 แล้ว normalize ให้รวม = 1"""
        probs = out.probs
        if not isinstance(probs, dict) or set(probs) != set(LABELS):
            raise ValueError(f"backend probs keys ไม่ตรง LABELS: {probs!r}")
        vals = {k: float(probs[k]) for k in LABELS}
        if any(not math.isfinite(v) or v < 0 for v in vals.values()):
            raise ValueError(f"backend probs มีค่าไม่ถูกต้อง: {vals!r}")
        total = sum(vals.values())
        if total <= 0:
            raise ValueError("backend probs รวมเป็น 0")
        vals = {k: v / total for k, v in vals.items()}
        label = max(LABELS, key=lambda k: vals[k])
        return label, vals[label], vals

    def _failure(self, status: str, t0: float, decode_ms: float) -> dict[str, Any]:
        """ผลสำหรับ bad_image / error — key ครบ แต่ label/probs เป็น null"""
        total = _ms_since(t0)
        return {
            "status": status,
            "label": None,
            "label_th": None,
            "message_th": MSG_BAD_IMAGE if status == "bad_image" else MSG_ERROR,
            "confidence": None,
            "probs": None,
            "warnings": [],
            "n_beans": None,
            "proportions": None,
            "beans": [],
            "timing_ms": _timing(decode_ms, max(0.0, total - decode_ms), total),
            "model": getattr(self.backend, "name", "?"),
        }


# ---------------------------------------------------------------------- load
def load(
    source: str | os.PathLike[str],
    *,
    low_conf_threshold: float | None = None,
    warmup: bool = True,
    **backend_kwargs: Any,
) -> Predictor:
    """โหลด predictor ครั้งเดียวตอน start

    load("stub", seed=1, delay_ms=150)          # หน่วงคงที่ 150 ms
    load("stub", delay_ms=(100, 400))           # หน่วงสุ่มในช่วง
    load("stub", status_weights={"ok": 1})      # ได้ ok ทุกครั้ง (ยกเว้นรูปเสียจริง)
    load("ML/models/current")                   # โมเดลจริง (ยังไม่มี backend → ModelLoadError)
    """
    t0 = time.perf_counter()
    if isinstance(source, str) and source.strip().lower() == "stub":
        backend: Backend = StubBackend(**backend_kwargs)
        threshold = DEFAULT_LOW_CONF_THRESHOLD
    else:
        backend, threshold = _load_model_dir(Path(source), backend_kwargs)
    if low_conf_threshold is not None:
        threshold = low_conf_threshold

    predictor = Predictor(backend, low_conf_threshold=threshold)
    if warmup:
        predictor.warmup()
    predictor._load_ms = round(_ms_since(t0), 1)
    log.info("roastml loaded model=%s in %.1f ms", backend.name, predictor._load_ms)
    return predictor


def _load_model_dir(path: Path, backend_kwargs: dict[str, Any]) -> tuple[Backend, float]:
    if not path.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์โมเดล: {path.resolve()}")
    card_path = path / "model_card.json"
    if not card_path.is_file():
        raise ModelLoadError(f"ไม่พบ {card_path}")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ModelLoadError(f"อ่าน {card_path} ไม่ได้: {e}") from e

    backend_type = card.get("backend")
    # TODO(M5/M6): ลงทะเบียน backend จริง (baseline B1, onnx) ที่นี่
    raise ModelLoadError(
        f"ยังไม่มี backend '{backend_type}' (จาก {card_path}) — ระหว่างนี้ใช้ load('stub')"
    )


# ------------------------------------------------------------------- helpers
def _ms_since(t: float) -> float:
    return (time.perf_counter() - t) * 1000.0


def _timing(decode_ms: float, ml_ms: float, total_ms: float) -> dict[str, float]:
    return {"decode": round(decode_ms, 1), "ml": round(ml_ms, 1), "total": round(total_ms, 1)}


def _clean_warnings(warnings: Any) -> list[str]:
    """เก็บเฉพาะ warning ที่อยู่ใน contract, ไม่ซ้ำ, ตามลำดับเดิม"""
    out: list[str] = []
    for w in warnings or []:
        if w in WARNINGS and w not in out:
            out.append(w)
        elif w not in WARNINGS:
            log.warning("dropping unknown warning %r", w)
    return out


def _clean_proportions(p: dict[str, float] | None) -> dict[str, float] | None:
    if p is None:
        return None
    return {k: round(float(p.get(k, 0.0)), 4) for k in LABELS}


def _clean_beans(beans: Any) -> list[dict[str, Any]]:
    return [
        {
            "bbox": [int(v) for v in b["bbox"]],
            "label": b["label"],
            "conf": round(float(b["conf"]), 4),
        }
        for b in beans or []
    ]


def _message(status: str, label: str, conf: float, warnings: list[str]) -> str:
    label_th = LABEL_TH[label]
    if status == "ok":
        msg = f"ผลวิเคราะห์: {label_th} (ความมั่นใจ {conf:.0%})"
    else:
        msg = (
            f"น่าจะเป็น{label_th} แต่ความมั่นใจต่ำ ({conf:.0%}) — "
            "ลองถ่ายใหม่ให้เห็นเมล็ดชัด ใกล้ขึ้น ในที่สว่างแสงสีขาว"
        )
    hints = [WARNING_TH[w] for w in warnings]
    if hints:
        msg += " · " + " · ".join(hints)
    return msg


def _hardcoded_error() -> dict[str, Any]:
    return {
        "status": "error", "label": None, "label_th": None, "message_th": MSG_ERROR,
        "confidence": None, "probs": None, "warnings": [], "n_beans": None,
        "proportions": None, "beans": [],
        "timing_ms": {"decode": 0.0, "ml": 0.0, "total": 0.0}, "model": "?",
    }
