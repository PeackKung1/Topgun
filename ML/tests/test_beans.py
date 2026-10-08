"""นับเมล็ด/ระดับคั่วรายเมล็ด (backend b1_linear_beans) — ภาพสังเคราะห์ ไม่ใช้ sklearn/torch (รันบน Pi ได้)"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image
from test_contract import assert_valid_result
from test_linear_backend import write_model

from roastml import api
from roastml.api import ModelLoadError, load

LIGHT, MEDIUM, DARK = (170, 125, 85), (110, 72, 45), (50, 32, 22)
CORE = ("status", "label", "label_th", "message_th", "confidence", "probs", "warnings")


def beans_jpeg(beans, size=(640, 480)) -> bytes:
    """พื้นขาว + วงรีสีเมล็ด · beans = [(cx, cy, color)]"""
    w, h = size
    img = np.full((h, w, 3), 235, np.uint8)
    for cx, cy, color in beans:
        cv2.ellipse(img, (int(cx), int(cy)), (int(w * 0.045), int(w * 0.03)), 30, 0, 360, color, -1)
    img = np.clip(img + np.random.default_rng(0).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def grid(n, colors, size=(640, 480)):
    w, h = size
    return [((i % 4 + 1) * w / 5, (i // 4 + 1) * h / 4, colors[i % len(colors)]) for i in range(n)]


SYNTH = {
    "single": beans_jpeg([(320, 240, MEDIUM)]),
    "five_same": beans_jpeg(grid(5, [DARK])),
    "mixed_three": beans_jpeg(grid(6, [LIGHT, MEDIUM, DARK])),
    "empty": beans_jpeg([]),
}


@pytest.fixture
def models(tmp_path):
    b1 = load(write_model(tmp_path / "b1", "b1_linear"))
    beans = load(write_model(tmp_path / "beans", "b1_linear_beans"))
    return b1, beans


def core(r):
    return {k: r[k] for k in CORE}


@pytest.mark.parametrize("name", list(SYNTH))
def test_core_fields_identical_to_b1(models, name):
    b1, beans = models
    r1, r2 = b1.predict_bytes(SYNTH[name]), beans.predict_bytes(SYNTH[name])
    assert_valid_result(r2)
    assert core(r1) == core(r2)
    assert "mixed_roast" not in r2["warnings"]


def test_counts_and_contract(models):
    _, beans = models
    single = beans.predict_bytes(SYNTH["single"])
    assert single["n_beans"] == 1 and len(single["beans"]) == 1
    five = beans.predict_bytes(SYNTH["five_same"])
    assert five["n_beans"] == 5 and five["proportions"]["dark"] == 1.0
    mixed = beans.predict_bytes(SYNTH["mixed_three"])
    assert mixed["n_beans"] == 6
    assert sum(v > 0 for v in mixed["proportions"].values()) >= 2  # ผสมจริงต้องเห็นมากกว่า 1 คลาส
    assert "mixed_roast" not in mixed["warnings"]                     # ภาคผนวก: ห้ามส่ง mixed_roast
    for r in (single, five, mixed):
        assert abs(sum(r["proportions"].values()) - 1.0) <= 1e-6
        assert r["n_beans"] >= 0
        for b in r["beans"]:
            x, y, w, h = b["bbox"]
            assert all(isinstance(v, int) for v in b["bbox"]) and w > 0 and h > 0
            assert 0 <= x < 640 and 0 <= y < 480


def test_empty_image_has_no_bean_fields(models):
    _, beans = models
    r = beans.predict_bytes(SYNTH["empty"])
    assert_valid_result(r)
    assert r["status"] == "ok" and r["n_beans"] is None and r["proportions"] is None and r["beans"] == []


def test_bbox_in_decoded_image_pixels(tmp_path):
    p = load(write_model(tmp_path / "beans", "b1_linear_beans"))
    raw = beans_jpeg([(1000, 750, DARK)], size=(2000, 1500))  # decode ย่อเหลือ 1600×1200
    r = p.predict_bytes(raw)
    x, y, w, h = r["beans"][0]["bbox"]
    cx, cy = x + w / 2, y + h / 2
    assert abs(cx - 800) < 12 and abs(cy - 600) < 12


def test_switch_off_env_equals_b1(tmp_path, monkeypatch):
    monkeypatch.setenv("ROAST_BEANS", "0")
    b1 = load(write_model(tmp_path / "b1", "b1_linear"))
    off = load(write_model(tmp_path / "beans", "b1_linear_beans"))
    assert off.info()["backend"]["beans_enabled"] is False
    for raw in SYNTH.values():
        r1, r2 = b1.predict_bytes(raw), off.predict_bytes(raw)
        assert core(r1) == core(r2)
        assert r2["n_beans"] is None and r2["proportions"] is None and r2["beans"] == []


def test_g5_failure_falls_back_to_b1(models, monkeypatch):
    b1, beans = models
    import roastml.bean_backend as bb

    def boom(*a, **k):
        raise RuntimeError("simulated bean failure")
    monkeypatch.setattr(bb, "split_beans", boom)
    for raw in SYNTH.values():
        r1, r2 = b1.predict_bytes(raw), beans.predict_bytes(raw)
        assert core(r1) == core(r2)
        assert r2["n_beans"] is None and r2["proportions"] is None and r2["beans"] == []


def test_g5_budget_exceeded_falls_back(tmp_path):
    b1 = load(write_model(tmp_path / "b1", "b1_linear"))
    slow = load(write_model(tmp_path / "beans", "b1_linear_beans", bean_counting={"budget_ms": 1e-6}))
    r1, r2 = b1.predict_bytes(SYNTH["five_same"]), slow.predict_bytes(SYNTH["five_same"])
    assert core(r1) == core(r2) and r2["n_beans"] is None and r2["beans"] == []


def test_blob_cap_returns_null_bean_fields(tmp_path):
    b1 = load(write_model(tmp_path / "b1", "b1_linear"))
    capped = load(write_model(tmp_path / "beans", "b1_linear_beans", bean_counting={"max_blobs": 3}))
    r1, r2 = b1.predict_bytes(SYNTH["five_same"]), capped.predict_bytes(SYNTH["five_same"])
    assert_valid_result(r2)
    assert core(r1) == core(r2) and r2["n_beans"] is None and r2["proportions"] is None and r2["beans"] == []


def test_mixed_override_config_is_rejected(tmp_path):
    with pytest.raises(ModelLoadError):
        load(write_model(tmp_path / "beans", "b1_linear_beans", bean_counting={"mixed_override": True}))


@pytest.mark.parametrize("raw", [b"", os.urandom(4096), SYNTH["single"][:300], b"%PDF-1.4\n%%EOF", None])
def test_fuzz_never_raises(models, raw):
    _, beans = models
    r = beans.predict_bytes(raw)
    assert_valid_result(r)


def test_one_pixel_image(models):
    b1, beans = models
    buf = io.BytesIO(); Image.new("RGB", (1, 1), DARK).save(buf, "PNG")
    r1, r2 = b1.predict_bytes(buf.getvalue()), beans.predict_bytes(buf.getvalue())
    assert core(r1) == core(r2) and r2["n_beans"] is None


def test_proportions_sum_exactly_one():
    p = api._clean_proportions({"light": 1 / 3, "medium": 1 / 3, "dark": 1 / 3})
    assert abs(sum(p.values()) - 1.0) <= 1e-6
    assert api._clean_proportions(None) is None


def test_no_sklearn_or_torch_imported(tmp_path):
    d = write_model(tmp_path / "beans", "b1_linear_beans")
    code = ("import sys, roastml.api as a, roastml.bean_backend, roastml.beans; a.load(sys.argv[1]);"
            "bad=[m for m in ('sklearn','torch','matplotlib','onnxruntime') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code, str(d)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert r.returncode == 0, r.stdout + r.stderr
