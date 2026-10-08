"""ทดสอบ backend B0/B1 ผ่าน api.load(<model dir>) ด้วยโมเดลสังเคราะห์ใน tmp_path"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image
from test_contract import assert_valid_result  # tests/ อยู่ใน sys.path ตอน pytest

from roastml.api import ModelLoadError, load
from roastml.features import FEATURE_SETS
from roastml.segment import SegConfig
from tools import train_baseline as tb


def jpeg(color, size=(640, 480)) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.clip(np.array(color, np.float32) + rng.normal(0, 10, (*size[::-1], 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def write_model(d, kind="b1_linear", spec=None, **card_extra):
    d.mkdir(parents=True, exist_ok=True)
    if spec is None:
        if kind == "b0_threshold":
            spec = tb.b0_spec((25.0, 40.0))
        else:  # B1-small ที่แยกตาม L_med ชัดๆ
            from roastml.features import FEATURES_ALL
            X = np.zeros((90, len(FEATURES_ALL)))
            X[:, FEATURES_ALL.index("L_med")] = np.r_[np.full(30, 15.0), np.full(30, 32.0), np.full(30, 50.0)]
            X += np.random.default_rng(1).normal(0, 1, X.shape)
            y = np.array(["dark"] * 30 + ["medium"] * 30 + ["light"] * 30)
            spec = tb.fit_b1(X, y, "small", 1.0)
    (d / "model.json").write_text(json.dumps(spec), encoding="utf-8")
    card = {"backend": kind, "name": f"test-{kind}", "model_file": "model.json",
            "seg_config": SegConfig().to_dict(), "low_conf_threshold": 0.0} | card_extra
    (d / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    return d


@pytest.mark.parametrize("kind", ["b1_linear", "b0_threshold"])
def test_load_and_predict(tmp_path, kind):
    p = load(write_model(tmp_path / "m", kind))
    dark, light = p.predict_bytes(jpeg((50, 32, 22))), p.predict_bytes(jpeg((170, 125, 85)))
    for r in (dark, light):
        assert_valid_result(r)
        assert r["status"] == "ok"  # threshold 0 → ตอบ argmax เสมอ ไม่มี low_confidence
        assert r["model"] == f"test-{kind}"
    assert dark["label"] == "dark" and light["label"] == "light"
    assert p.info()["backend"]["backend"] == kind


def test_low_conf_threshold_from_card(tmp_path):
    p = load(write_model(tmp_path / "m", "b1_linear", low_conf_threshold=0.99))
    r = p.predict_bytes(jpeg((105, 70, 45)))
    assert_valid_result(r)
    assert p.low_conf_threshold == 0.99 and r["status"] in ("ok", "low_confidence")


def test_blank_image_still_answers_with_warning(tmp_path):
    p = load(write_model(tmp_path / "m"))
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (240, 240, 240)).save(buf, "PNG")
    r = p.predict_bytes(buf.getvalue())
    assert_valid_result(r)
    assert r["label"] is not None and "no_beans_detected" in r["warnings"]


def test_bad_bytes_and_wrong_type(tmp_path):
    p = load(write_model(tmp_path / "m"))
    assert p.predict_bytes(b"not an image")["status"] == "bad_image"
    assert p.predict_bytes("path.jpg")["status"] == "error"


def test_load_errors(tmp_path):
    with pytest.raises(ModelLoadError):  # model.json หาย
        d = write_model(tmp_path / "a")
        (d / "model.json").unlink()
        load(d)
    with pytest.raises(ModelLoadError):  # คลาสไม่ตรง contract
        spec = tb.b0_spec((10, 20)) | {"classes": ["a", "b", "c"]}
        load(write_model(tmp_path / "b", "b0_threshold", spec=spec))
    with pytest.raises(ModelLoadError):  # backend ไม่รับ kwargs ของ stub
        load(write_model(tmp_path / "c"), seed=1)
    with pytest.raises(ModelLoadError):  # backend ไม่รู้จัก
        load(write_model(tmp_path / "d", "onnx_mobilenet", spec={}))


# ------------------------------------------------------------ decode: JPEG draft ตามสัดส่วนภาพ (แก้ 8 ต.ค.)
def test_draft_target_keeps_aspect():
    from roastml.decode import draft_target

    assert draft_target(4032, 3024, 1600) == (1600, 1200)
    assert draft_target(3024, 4032, 1600) == (1200, 1600)
    assert draft_target(800, 600, 1600) == (800, 600)


def test_decode_4032x3024_uses_draft_and_respects_max_side():
    from roastml.decode import MAX_SIDE, decode_image

    buf = io.BytesIO()
    Image.new("RGB", (4032, 3024), (100, 70, 50)).save(buf, "JPEG", quality=80)
    img = decode_image(buf.getvalue())
    assert max(img.image.size) == MAX_SIDE and img.orig_size == (4032, 3024)
    # ตรวจว่า draft ย่อได้จริง (ก่อน thumbnail) ด้วย API ของ Pillow ที่ติดตั้งอยู่
    im = Image.open(io.BytesIO(buf.getvalue()))
    im.draft("RGB", (1600, 1200))
    assert im.size == (2016, 1512)
