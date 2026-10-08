"""ทดสอบ contract FW ↔ ML v2 (ml-spec ข้อ 4) ด้วย stub backend และ backend ทดสอบ"""

from __future__ import annotations

import inspect
import io
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from roastml import decode as decode_mod
from roastml.api import (
    LABELS, RESULT_KEYS, STATUSES, WARNINGS, ModelLoadError, Predictor, load,
)
from roastml.contract import BackendOutput, TIMING_KEYS
from roastml.decode import BadImageError, decode_image

SEED = 1234


# --------------------------------------------------------------- helpers
def img_bytes(size=(640, 480), fmt="JPEG", mode="RGB", color=(120, 80, 50), **save_kw) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt, **save_kw)
    return buf.getvalue()


def assert_valid_result(r: dict) -> None:
    """ตรวจ schema ตาม contract v2 ทุก field"""
    assert isinstance(r, dict)
    assert tuple(r) == RESULT_KEYS, f"keys ไม่ตรง: {list(r)}"
    assert r["status"] in STATUSES
    assert isinstance(r["message_th"], str) and r["message_th"]
    assert isinstance(r["model"], str) and r["model"]
    assert isinstance(r["warnings"], list) and set(r["warnings"]) <= set(WARNINGS)
    assert len(r["warnings"]) == len(set(r["warnings"]))
    assert isinstance(r["beans"], list)

    t = r["timing_ms"]
    assert tuple(t) == TIMING_KEYS
    assert all(isinstance(v, float) and v >= 0 for v in t.values())

    if r["status"] in ("bad_image", "error"):
        assert r["label"] is None and r["label_th"] is None
        assert r["confidence"] is None and r["probs"] is None
        assert r["warnings"] == [] and r["beans"] == []
        assert r["n_beans"] is None and r["proportions"] is None
    else:
        assert r["label"] in LABELS
        assert isinstance(r["label_th"], str) and r["label_th"]
        assert set(r["probs"]) == set(LABELS)
        assert all(0.0 <= v <= 1.0 for v in r["probs"].values())
        assert math.isclose(sum(r["probs"].values()), 1.0, abs_tol=1e-3)
        assert r["confidence"] == r["probs"][r["label"]] == max(r["probs"].values())

        if r["n_beans"] is None:
            assert r["proportions"] is None and r["beans"] == []
        else:
            assert r["n_beans"] == len(r["beans"]) >= 1
            assert set(r["proportions"]) == set(LABELS)
            assert math.isclose(sum(r["proportions"].values()), 1.0, abs_tol=1e-3)
            for b in r["beans"]:
                assert set(b) == {"bbox", "label", "conf"}
                assert len(b["bbox"]) == 4 and all(isinstance(v, int) for v in b["bbox"])
                assert b["label"] in LABELS and 0.0 <= b["conf"] <= 1.0

    # ต้องส่งเป็น JSON ได้ทันที (ไม่มี NaN / numpy type / tuple key)
    json.dumps(r, ensure_ascii=False, allow_nan=False)


@pytest.fixture
def stub():
    return load("stub", seed=SEED)


# --------------------------------------------------------------- schema / stub
def test_schema_every_field_many_calls(stub):
    raw = img_bytes()
    for _ in range(300):
        assert_valid_result(stub.predict_bytes(raw))


def test_fw_entry_takes_bytes_only_and_metadata_is_separate():
    # FW เรียก predict_bytes(raw) เท่านั้น — metadata ของ dataset อยู่ที่ predict_dataset
    assert list(inspect.signature(Predictor.predict_bytes).parameters) == ["self", "raw"]
    p = load("stub", seed=SEED, status_weights={"ok": 1.0})
    r = p.predict_dataset(img_bytes(), source="agtron", roi="0 0 10 10")  # stub ไม่รองรับ metadata
    assert_valid_result(r)
    assert r["status"] == "error"


def test_stub_covers_all_statuses_and_warnings(stub):
    raw = img_bytes()
    results = [stub.predict_bytes(raw) for _ in range(500)]
    assert {r["status"] for r in results} == set(STATUSES)
    assert set().union(*(r["warnings"] for r in results)) == set(WARNINGS)
    assert any(r["n_beans"] for r in results), "stub ควรสุ่ม beans ออกมาบ้าง"
    assert any(r["n_beans"] is None and r["status"] == "ok" for r in results)


def test_stub_seed_reproducible():
    raw = img_bytes()

    def run(seed):
        p = load("stub", seed=seed)
        return [
            {k: v for k, v in p.predict_bytes(raw).items() if k != "timing_ms"} for _ in range(50)
        ]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_stub_status_weights_force_ok():
    p = load("stub", seed=SEED, status_weights={"ok": 1.0})
    assert {p.predict_bytes(img_bytes())["status"] for _ in range(50)} == {"ok"}


def test_stub_delay_fixed_and_range():
    p = load("stub", seed=SEED, delay_ms=60, status_weights={"ok": 1.0})
    assert p.predict_bytes(img_bytes())["timing_ms"]["ml"] >= 55

    p = load("stub", seed=SEED, delay_ms=(20, 40), status_weights={"ok": 1.0})
    mls = [p.predict_bytes(img_bytes())["timing_ms"]["ml"] for _ in range(5)]
    assert all(m >= 15 for m in mls)


@pytest.mark.parametrize("kw", [
    {"status_weights": {"maybe": 1.0}},
    {"status_weights": {"ok": 0.0}},
    {"delay_ms": -1},
    {"delay_ms": (50, 10)},
])
def test_stub_rejects_bad_config(kw):
    with pytest.raises(ValueError):
        load("stub", **kw)


def test_bean_bboxes_inside_original_image():
    p = load("stub", seed=SEED, status_weights={"ok": 1.0}, beans_rate=1.0)
    w, h = 3000, 2000
    r = p.predict_bytes(img_bytes((w, h)))
    assert r["n_beans"]
    for x, y, bw, bh in (b["bbox"] for b in r["beans"]):
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h


# --------------------------------------------------------------- bad input → bad_image
def _truncated_jpeg() -> bytes:
    data = img_bytes((800, 600))
    return data[: len(data) // 3]


def _huge_dims_png() -> bytes:
    # 10000×10000 = 100 MP > MAX_PIXELS แต่ไฟล์ PNG โหมด 1-bit มีขนาดเล็ก
    return img_bytes((10_000, 10_000), fmt="PNG", mode="1", color=0)


BAD_INPUTS = {
    "none": None,
    "empty": b"",
    "text": "สวัสดี not an image".encode(),
    "random_bytes": bytes(range(256)) * 40,
    "pdf_header": b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n",
    "jpeg_magic_only": b"\xff\xd8\xff\xe0" + b"\x00" * 100,
    "truncated_jpeg": _truncated_jpeg(),
    "eps": b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n",
    "too_many_bytes": b"\xff\xd8" + b"\x00" * (decode_mod.MAX_BYTES + 1),
    "too_many_pixels": _huge_dims_png(),
    # ด้านสั้นหลังย่อ < MIN_SIDE: ไม่มีข้อมูลพอจำแนก (เดิมได้ label มั่นใจแบบไร้ความหมาย)
    "tiny_1x1": img_bytes((1, 1), fmt="PNG"),
    "tiny_7x7": img_bytes((7, 7), fmt="PNG"),
    "thin_after_resize_4000x6": img_bytes((4000, 6), fmt="PNG"),  # thumbnail → 1600×3
}


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")  # too_many_pixels ตั้งใจให้เกิด
@pytest.mark.parametrize("name", list(BAD_INPUTS))
def test_bad_inputs_return_bad_image_without_raising(name):
    # stub ที่ตอบ ok เสมอ → ถ้าได้ bad_image แปลว่ามาจาก decode จริง
    p = load("stub", seed=SEED, status_weights={"ok": 1.0})
    r = p.predict_bytes(BAD_INPUTS[name])
    assert_valid_result(r)
    assert r["status"] == "bad_image", name


@pytest.mark.parametrize("raw", ["path/to/img.jpg", 123, ["a"], io.BytesIO(b"x")])
def test_wrong_type_returns_error_without_raising(stub, raw):
    r = stub.predict_bytes(raw)
    assert_valid_result(r)
    assert r["status"] == "error"


def test_bytearray_and_memoryview_accepted():
    p = load("stub", seed=SEED, status_weights={"ok": 1.0})
    raw = img_bytes()
    assert p.predict_bytes(bytearray(raw))["status"] == "ok"
    assert p.predict_bytes(memoryview(raw))["status"] == "ok"


# --------------------------------------------------------------- good input หลายแบบ
GOOD_INPUTS = {
    "jpeg": lambda: img_bytes(),
    "png_rgba": lambda: img_bytes(fmt="PNG", mode="RGBA", color=(10, 20, 30, 0)),
    "png_gray": lambda: img_bytes(fmt="PNG", mode="L", color=100),
    "png_palette": lambda: img_bytes(fmt="PNG", mode="P", color=3),
    "gif": lambda: img_bytes(fmt="GIF", mode="P", color=1),
    "webp": lambda: img_bytes(fmt="WEBP"),
    "bmp": lambda: img_bytes(fmt="BMP"),
    "min_side_8x8": lambda: img_bytes((8, 8), fmt="PNG"),  # ขอบล่างของ MIN_SIDE ยังใช้ได้
    "large_jpeg_4000x3000": lambda: img_bytes((4000, 3000), quality=90),
}


@pytest.mark.parametrize("name", list(GOOD_INPUTS))
def test_good_inputs_are_not_bad_image(name):
    p = load("stub", seed=SEED, status_weights={"ok": 1.0})
    r = p.predict_bytes(GOOD_INPUTS[name]())
    assert_valid_result(r)
    assert r["status"] == "ok", name


def test_decode_rejects_short_side_below_min_after_resize():
    with pytest.raises(BadImageError, match="too_small:1600x3"):
        decode_image(img_bytes((4000, 6), fmt="PNG"))
    assert min(decode_image(img_bytes((8, 8), fmt="PNG")).image.size) == decode_mod.MIN_SIDE


def test_decode_downscales_large_image():
    d = decode_image(img_bytes((4000, 3000)))
    assert max(d.image.size) <= decode_mod.MAX_SIDE
    assert d.orig_size == (4000, 3000)
    assert d.image.mode == "RGB"


def test_decode_respects_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # หมุน 90° ตามเข็ม
    d = decode_image(img_bytes((400, 200), exif=exif.tobytes()))
    assert d.orig_size == (200, 400)
    assert d.image.size == (200, 400)


def test_decode_transparent_becomes_white_not_black():
    d = decode_image(img_bytes((10, 10), fmt="PNG", mode="RGBA", color=(0, 0, 0, 0)))
    assert d.image.getpixel((5, 5)) == (255, 255, 255)


def test_decode_raises_bad_image_error_directly():
    with pytest.raises(BadImageError):
        decode_image(b"")


def test_heic_if_supported():
    pytest.importorskip("pillow_heif")
    p = load("stub", seed=SEED, status_weights={"ok": 1.0})
    assert p.predict_bytes(img_bytes(fmt="HEIF"))["status"] == "ok"


# --------------------------------------------------------------- concurrency
def test_concurrent_8_threads():
    p = load("stub", seed=SEED, delay_ms=(1, 5))
    raw = img_bytes((1200, 900))
    barrier = threading.Barrier(8)
    n_per_thread = 25

    def worker(_):
        barrier.wait()  # ปล่อยพร้อมกันทั้ง 8 threads
        return [p.predict_bytes(raw) for _ in range(n_per_thread)]

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = [r for batch in ex.map(worker, range(8)) for r in batch]

    assert len(results) == 8 * n_per_thread
    for r in results:
        assert_valid_result(r)


def test_non_thread_safe_backend_is_serialized():
    """backend ที่ประกาศ thread_safe=False ต้องไม่ถูกเรียกซ้อนกัน"""

    class Serial:
        name = "serial-test"
        thread_safe = False

        def __init__(self):
            self.active = 0
            self.max_active = 0
            self._g = threading.Lock()

        def predict(self, img):
            with self._g:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            threading.Event().wait(0.005)
            with self._g:
                self.active -= 1
            return BackendOutput(probs={"light": 0.1, "medium": 0.8, "dark": 0.1})

        def warmup(self, img):
            pass

        def info(self):
            return {}

    b = Serial()
    p = Predictor(b)
    raw = img_bytes((100, 100))
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: p.predict_bytes(raw), range(40)))
    assert all(r["status"] == "ok" for r in results)
    assert b.max_active == 1


# --------------------------------------------------------------- backend พัง → error ไม่ raise
class FakeBackend:
    name = "fake"
    thread_safe = True

    def __init__(self, fn):
        self.fn = fn

    def predict(self, img):
        return self.fn(img)

    def warmup(self, img):
        pass

    def info(self):
        raise RuntimeError("info พัง")


def _raise(exc):
    def f(_img):
        raise exc
    return f


@pytest.mark.parametrize("fn", [
    _raise(RuntimeError("boom")),
    _raise(MemoryError()),
    lambda img: BackendOutput(probs={"light": 1.0}),                                   # key ไม่ครบ
    lambda img: BackendOutput(probs={"light": float("nan"), "medium": 0.5, "dark": 0.5}),
    lambda img: BackendOutput(probs={"light": -1.0, "medium": 0.5, "dark": 0.5}),
    lambda img: BackendOutput(probs={"light": 0.0, "medium": 0.0, "dark": 0.0}),
    lambda img: None,
    lambda img: BackendOutput(probs={"light": 0.2, "medium": 0.7, "dark": 0.1},
                              beans=[{"bbox": "เสีย"}]),                                # beans ผิดรูป
], ids=["runtime", "memory", "missing_key", "nan", "negative", "all_zero", "none", "bad_beans"])
def test_backend_failure_becomes_error(fn):
    p = Predictor(FakeBackend(fn))
    r = p.predict_bytes(img_bytes())
    assert_valid_result(r)
    assert r["status"] == "error"


def test_backend_bad_image_error_becomes_bad_image():
    p = Predictor(FakeBackend(_raise(BadImageError("unusable"))))
    assert p.predict_bytes(img_bytes())["status"] == "bad_image"


def test_low_confidence_still_has_label():
    flat = lambda img: BackendOutput(probs={"light": 0.3, "medium": 0.4, "dark": 0.3})
    r = Predictor(FakeBackend(flat), low_conf_threshold=0.6).predict_bytes(img_bytes())
    assert_valid_result(r)
    assert r["status"] == "low_confidence" and r["label"] == "medium"


def test_probs_are_normalized_and_unknown_warnings_dropped():
    fn = lambda img: BackendOutput(probs={"light": 2.0, "medium": 6.0, "dark": 2.0},
                                   warnings=["blurry", "blurry", "made_up"])
    r = Predictor(FakeBackend(fn)).predict_bytes(img_bytes())
    assert_valid_result(r)
    assert r["probs"] == {"light": 0.2, "medium": 0.6, "dark": 0.2}
    assert r["warnings"] == ["blurry"]


# --------------------------------------------------------------- load / info
def test_info_is_json_serializable(stub):
    info = stub.info()
    json.dumps(info, ensure_ascii=False, allow_nan=False)
    assert info["model"] == "stub" and info["api_version"] == "2"
    assert info["statuses"] == list(STATUSES)
    assert info["load_ms"] is not None


def test_info_survives_backend_info_failure():
    p = Predictor(FakeBackend(lambda img: None))
    json.dumps(p.info(), ensure_ascii=False)


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope")


def test_load_dir_without_card_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        load(tmp_path)


def test_load_dir_with_unknown_backend_raises(tmp_path):
    (tmp_path / "model_card.json").write_text('{"backend": "nope"}', encoding="utf-8")
    with pytest.raises(ModelLoadError):
        load(tmp_path)
