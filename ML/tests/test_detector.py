"""detector runtime (letterbox / decode / NMS), feature ต่อกรอบ และ backend det_b1_beans — ข้อมูลสังเคราะห์ทั้งหมด"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image
from test_contract import assert_valid_result
from test_linear_backend import write_model

from roastml import det_backend
from roastml.api import ModelLoadError, load
from roastml.detector import DetConfig, container_boxes, decode_output, letterbox, nms
from roastml.features import pixel_stats
from roastml.perbean import box_bean_features, fuse_image_prior
from roastml.segment import SegConfig

DARK, LIGHT = (50, 32, 22), (170, 125, 85)


def scene_bytes(boxes_colors, size=(640, 480)):
    """พื้นขาว + สี่เหลี่ยมสีเมล็ด → JPEG"""
    arr = np.full((size[1], size[0], 3), 240, np.uint8)
    for (x, y, w, h), color in boxes_colors:
        arr[y:y + h, x:x + w] = color
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95)
    return buf.getvalue()


class FakeDetector:
    """แทน BeanDetector: คืนกรอบที่กำหนด (ไม่ต้องมี onnxruntime → รันในโหมดจำลอง Pi ได้)"""
    boxes = np.zeros((0, 4), np.int64)

    def __init__(self, path, cfg):
        self.cfg, self.size = cfg, 640

    def detect(self, rgb):
        b = np.asarray(type(self).boxes, np.int64).reshape(-1, 4)
        return b.copy(), np.linspace(0.9, 0.5, len(b))


@pytest.fixture
def fake_det(monkeypatch):
    monkeypatch.setattr(det_backend, "BeanDetector", FakeDetector)
    FakeDetector.boxes = np.zeros((0, 4), np.int64)
    return FakeDetector


def det_model(tmp_path, **extra):
    d = write_model(tmp_path / "m", "det_b1_beans", detector_file="bean.onnx", **extra)
    (d / "bean.onnx").write_bytes(b"fake")
    return d


# ------------------------------------------------------------------ detector math
def test_letterbox_keeps_aspect_and_maps_back():
    rgb = np.zeros((300, 600, 3), np.uint8)
    out, s, px, py = letterbox(rgb, 640)
    assert out.shape == (640, 640, 3) and s == pytest.approx(640 / 600)
    assert (px, py) == (0, (640 - 320) // 2)
    assert out[0, 0, 0] == 114 and out[py + 5, 5, 0] == 0  # ขอบเทา / เนื้อภาพ


def test_nms_suppresses_overlap_and_respects_max_det():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [100, 100, 110, 110]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert nms(boxes, scores, 0.5, 100).tolist() == [0, 2, 3]
    assert nms(boxes, scores, 0.9, 100).tolist() == [0, 1, 2, 3]   # IoU 0.68 < 0.9 → ไม่ตัด
    assert nms(boxes, scores, 0.5, 2).tolist() == [0, 2]
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5, 10).tolist() == []


def test_container_box_is_dropped_but_dense_neighbours_are_kept():
    small = np.array([[10 * i, 10 * j, 10 * i + 12, 10 * j + 12] for i in range(4) for j in range(4)], float)   # 16 เมล็ดซ้อนกันเล็กน้อย
    plate = np.array([[-5, -5, 50, 50]], float)                                                              # กรอบจานครอบทั้งหมด
    flags = container_boxes(np.vstack([small, plate]), 5, 4.0)
    assert flags.tolist() == [False] * 16 + [True]
    assert not container_boxes(np.vstack([small, plate]), 0, 4.0).any()            # ปิดกฎ
    assert not container_boxes(small[:3], 5, 4.0).any()                            # กรอบน้อยกว่าเกณฑ์


@pytest.mark.parametrize("transpose", [False, True])
def test_decode_output_both_layouts(transpose):
    rows = np.array([[100, 100, 20, 10, 0.9], [300, 200, 40, 40, 0.1]], np.float32)  # cx cy w h score
    out = rows[None] if transpose else rows.T[None]
    boxes, scores = decode_output(out, 0.25, channels_first=not transpose)
    assert boxes.tolist() == [[90.0, 95.0, 110.0, 105.0]] and scores.tolist() == pytest.approx([0.9])
    with pytest.raises(ValueError):
        decode_output(np.zeros((2, 5, 3)), 0.25)


@pytest.mark.parametrize("bad", [{"conf": 0}, {"iou": 1.0}, {"max_det": 0}, {"max_det": 20000}, {"threads": True}, {"nope": 1},
                                 {"container_min": -1}, {"container_area_ratio": 0.5}])
def test_det_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        DetConfig.from_dict(bad)
    assert DetConfig.from_dict({"conf": 0.3, "max_det": 500}).max_det == 500


def test_real_onnx_detector_roundtrip(tmp_path):
    """กราฟ ONNX ที่คืนค่าคงที่ → BeanDetector ต้องแปลงกรอบกลับพิกัดภาพเดิมถูก"""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from roastml.detector import BeanDetector
    const = np.zeros((1, 5, 3), np.float32)
    const[0, :, 0] = [320, 320, 64, 32, 0.9]      # กลางภาพ input 640
    const[0, :, 1] = [322, 321, 64, 32, 0.8]      # ซ้อนตัวแรก → ถูก NMS ตัด
    const[0, :, 2] = [100, 320, 20, 20, 0.1]      # ต่ำกว่า conf
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["rgb"], ["s"], keepdims=0),
         helper.make_node("Mul", ["s", "zero"], ["z"]),
         helper.make_node("Add", ["boxes", "z"], ["output0"])],
        "const-det", [helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 3])],
        [numpy_helper.from_array(const, "boxes"), numpy_helper.from_array(np.zeros((), np.float32), "zero")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "d.onnx"))
    det = BeanDetector(tmp_path / "d.onnx", DetConfig(threads=1))
    boxes, scores = det.detect(np.zeros((640, 1280, 3), np.uint8))   # scale 0.5, pad_y 160
    assert boxes.tolist() == [[576, 288, 128, 64]] and scores.tolist() == pytest.approx([0.9])


# ------------------------------------------------------------------ per-box features
def test_box_features_match_pixel_stats_of_trimmed_core():
    rng = np.random.default_rng(3)
    lab = np.dstack([rng.uniform(5, 90, (60, 80)), rng.uniform(-10, 30, (60, 80)), rng.uniform(-10, 40, (60, 80))]).astype(np.float32)
    cfg = SegConfig()
    box = np.array([[10, 8, 40, 30]])
    X = box_bean_features(lab, box, cfg)
    core = lab[8 + 6:8 + 24, 10 + 8:10 + 32].reshape(-1, 3).astype(np.float64)   # หดด้านละ 20%
    lo, hi = np.percentile(core[:, 0], [cfg.trim_lo, cfg.trim_hi])
    np.testing.assert_allclose(X[0], pixel_stats(core[(core[:, 0] >= lo) & (core[:, 0] <= hi)]), rtol=1e-9, atol=1e-9)


def test_box_features_every_box_gets_pixels_even_when_covered():
    lab = np.zeros((40, 40, 3), np.float32)
    lab[..., 0] = 30
    lab[10:20, 10:20, 0] = 70
    boxes = np.array([[0, 0, 40, 40], [5, 5, 20, 20], [5, 5, 20, 20]])   # สองกรอบซ้ำกันสนิท
    X = box_bean_features(lab, boxes, SegConfig())
    assert X.shape[0] == 3 and np.isfinite(X).all()
    assert box_bean_features(lab, np.zeros((0, 4)), SegConfig()).shape[0] == 0


def test_ellipse_box_features_ignore_corners():
    lab = np.zeros((60, 60, 3), np.float32)
    lab[..., 0] = 90                                   # พื้นสว่าง
    yy, xx = np.mgrid[0:60, 0:60]
    lab[((xx - 30) / 22.0) ** 2 + ((yy - 30) / 16.0) ** 2 <= 1, 0] = 30   # เมล็ดเข้มรูปวงรี
    box = np.array([[8, 14, 44, 32]])
    rect = box_bean_features(lab, box, SegConfig(), shrink=0.0, shape="rect")[0, 0]
    ell = box_bean_features(lab, box, SegConfig(), shrink=0.1, shape="ellipse")[0, 0]
    assert ell == pytest.approx(30) and rect >= 30
    with pytest.raises(ValueError):
        box_bean_features(lab, box, SegConfig(), shape="circle")


def test_mask_box_features_use_segment_pixels_and_fall_back_per_box():
    rng = np.random.default_rng(5)
    lab = np.dstack([rng.uniform(20, 80, (50, 90)), rng.uniform(0, 20, (50, 90)), rng.uniform(0, 30, (50, 90))]).astype(np.float32)
    mask = np.zeros((50, 90), bool)
    mask[10:30, 10:40] = True                               # segment เห็นเฉพาะเมล็ดแรก
    boxes = np.array([[5, 5, 40, 30], [50, 5, 30, 30]])
    X = box_bean_features(lab, boxes, SegConfig(), shape="mask", mask=mask)
    inside = lab[10:30, 10:40].reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(X[0], pixel_stats(inside), rtol=1e-9, atol=1e-9)        # ไม่ trim ซ้ำ
    np.testing.assert_allclose(X[1], box_bean_features(lab, boxes[1:], SegConfig())[0], rtol=1e-9, atol=1e-9)  # ถอยเป็น rect
    with pytest.raises(ValueError):
        box_bean_features(lab, boxes, SegConfig(), shape="mask")


def test_fuse_image_prior_shrinks_towards_image_but_keeps_strong_beans():
    P = np.array([[0.45, 0.55, 0.0], [0.02, 0.03, 0.95]])       # เมล็ด 1 ลังเล · เมล็ด 2 เข้มชัด
    image = np.array([0.7, 0.2, 0.1])
    fused = fuse_image_prior(P, image, 1.0)
    assert fused.argmax(1).tolist() == [0, 2] and np.allclose(fused.sum(1), 1)
    np.testing.assert_allclose(fuse_image_prior(P, image, 0.0), P / P.sum(1, keepdims=True))
    with pytest.raises(ValueError):
        fuse_image_prior(P, image, -1)


# ------------------------------------------------------------------ backend contract
def check_counts(r):
    assert r["n_beans"] == len(r["beans"]) == sum(r["counts"].values())
    assert r["counts"] == {c: sum(b["label"] == c for b in r["beans"]) for c in ("light", "medium", "dark")}
    assert r["counts"][r["label"]] == max(r["counts"].values())
    w, h = r["image_size"]
    assert all(x >= 0 and y >= 0 and x + bw <= w and y + bh <= h for x, y, bw, bh in (b["bbox"] for b in r["beans"]))


def test_per_box_labels_counts_and_mixed(tmp_path, fake_det):
    p = load(det_model(tmp_path))
    boxes = [(40, 40, 80, 60), (200, 40, 80, 60), (360, 40, 80, 60)]
    fake_det.boxes = boxes
    r = p.predict_bytes(scene_bytes(list(zip(boxes, [DARK, DARK, LIGHT]))))
    assert_valid_result(r)
    check_counts(r)
    assert r["n_beans"] == 3 and r["counts"] == {"light": 1, "medium": 0, "dark": 2} and r["label"] == "dark"
    assert "mixed_roast" in r["warnings"] and r["count_method"] == "exact"
    assert [b["label"] for b in r["beans"]] == ["dark", "dark", "light"]
    assert p.info()["backend"]["backend"] == "det_b1_beans"


def test_broadcast_mode_uses_image_label_and_image_confidence(tmp_path, fake_det):
    p = load(det_model(tmp_path, bean_label_mode="broadcast"))
    boxes = [(40, 40, 200, 300), (300, 40, 200, 300)]
    fake_det.boxes = boxes
    r = p.predict_bytes(scene_bytes(list(zip(boxes, [DARK, DARK]))))
    assert_valid_result(r)
    check_counts(r)
    assert {b["label"] for b in r["beans"]} == {r["label"]} and "mixed_roast" not in r["warnings"]
    assert r["confidence"] == pytest.approx(max(r["probs"].values()), abs=1e-4)
    assert all(b["conf"] == pytest.approx(r["confidence"], abs=1e-3) for b in r["beans"])


def test_group_mode_single_roast_gets_one_label_and_mixed_is_split(tmp_path, fake_det):
    p = load(det_model(tmp_path, bean_label_mode="group", group_config={"min_L_gap": 12.0, "max_k": 3}))
    boxes = [(40 + 110 * i, 40, 90, 70) for i in range(5)]
    fake_det.boxes = boxes
    same = p.predict_bytes(scene_bytes([(b, (48 + 3 * i, 31 + 2 * i, 22 + i)) for i, b in enumerate(boxes)]))  # เข้มใกล้กัน
    mixed = p.predict_bytes(scene_bytes(list(zip(boxes, [DARK, DARK, DARK, LIGHT, LIGHT]))))
    for r in (same, mixed):
        assert_valid_result(r)
        check_counts(r)
    assert same["counts"] == {"light": 0, "medium": 0, "dark": 5} and "mixed_roast" not in same["warnings"]
    assert mixed["counts"]["dark"] == 3 and sum(mixed["counts"].values()) == 5
    assert "mixed_roast" in mixed["warnings"] and mixed["label"] == "dark"


def test_image_prior_mode_valid_and_reports_config(tmp_path, fake_det):
    p = load(det_model(tmp_path, bean_label_mode="image_prior", prior_weight=1.0, box_shape="ellipse", box_shrink=0.1))
    boxes = [(40, 40, 80, 60), (200, 40, 80, 60), (360, 40, 80, 60)]
    fake_det.boxes = boxes
    r = p.predict_bytes(scene_bytes(list(zip(boxes, [DARK, DARK, LIGHT]))))
    assert_valid_result(r)
    check_counts(r)
    assert r["counts"]["dark"] == 2 and "mixed_roast" in r["warnings"]   # เมล็ดอ่อนที่ชัดยังต่างจากภาพได้
    info = p.info()["backend"]
    assert info["bean_label_mode"] == "image_prior" and info["box_shape"] == "ellipse" and info["prior_weight"] == 1.0


def test_mask_shape_backend_matches_contract(tmp_path, fake_det):
    p = load(det_model(tmp_path, box_shape="mask"))
    boxes = [(40, 40, 120, 90), (240, 40, 120, 90), (440, 40, 120, 90)]
    fake_det.boxes = boxes
    r = p.predict_bytes(scene_bytes(list(zip(boxes, [DARK, DARK, LIGHT]))))
    assert_valid_result(r)
    check_counts(r)
    assert r["counts"] == {"light": 1, "medium": 0, "dark": 2} and p.info()["backend"]["box_shape"] == "mask"


def test_no_boxes_reports_zero_and_keeps_image_label(tmp_path, fake_det):
    p = load(det_model(tmp_path))
    r = p.predict_bytes(scene_bytes([]))
    assert_valid_result(r)
    assert r["n_beans"] == 0 and r["counts"] == {"light": 0, "medium": 0, "dark": 0} and r["beans"] == []
    assert r["warnings"] == ["no_beans_detected"] and r["count_method"] == "exact" and r["proportions"] is None


def test_dense_overlapping_boxes_are_marked_estimated(tmp_path, fake_det):
    p = load(det_model(tmp_path))
    boxes = [(40 + 30 * i, 40, 60, 60) for i in range(8)]   # ซ้อนกันครึ่งกรอบ
    fake_det.boxes = boxes
    r = p.predict_bytes(scene_bytes([(b, DARK) for b in boxes]))
    assert_valid_result(r)
    check_counts(r)
    assert r["count_method"] == "estimated" and {"bean_count_estimated", "count_visible_only"} <= set(r["warnings"])


def test_detector_failure_falls_back_to_image_result(tmp_path, fake_det, monkeypatch):
    p = load(det_model(tmp_path))
    monkeypatch.setattr(FakeDetector, "detect", lambda self, rgb: (_ for _ in ()).throw(RuntimeError("boom")))
    r = p.predict_bytes(scene_bytes([((40, 40, 80, 60), DARK)]))
    assert_valid_result(r)
    assert r["status"] == "ok" and r["n_beans"] is None and r["beans"] == [] and r["counts"] is None


def test_roast_beans_env_disables_counting(tmp_path, fake_det, monkeypatch):
    monkeypatch.setenv("ROAST_BEANS", "0")
    fake_det.boxes = [(40, 40, 80, 60)]
    r = load(det_model(tmp_path)).predict_bytes(scene_bytes([((40, 40, 80, 60), DARK)]))
    assert_valid_result(r)
    assert r["n_beans"] is None and r["beans"] == []


@pytest.mark.parametrize("extra", [{"bean_label_mode": "vote"}, {"det_config": {"conf": 2}}, {"det_config": {"x": 1}},
                                   {"group_config": {"min_L_gap": 12.0}}, {"bean_label_mode": "group", "group_config": {"bad": 1}},
                                   {"box_shape": "circle"}, {"box_shrink": 0.6}, {"prior_weight": -1}])
def test_bad_card_fails_at_load(tmp_path, fake_det, extra):
    with pytest.raises(ModelLoadError):
        load(det_model(tmp_path, **extra))


def test_missing_detector_file_key_fails_at_load(tmp_path, fake_det):
    d = write_model(tmp_path / "m", "det_b1_beans")
    with pytest.raises(ModelLoadError):
        load(d)
