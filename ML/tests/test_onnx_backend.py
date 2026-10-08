"""Synthetic-only ONNX/portable runtime contract and export tests."""
from __future__ import annotations

import io
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from roastml.api import ModelLoadError, load
from roastml.rgb_views import RGBViewConfig, build_rgb_views, pool_view_probabilities
from roastml.decode import decode_image
from roastml.tabular_backend import NumpyTabularModel
from roastml.features import FEATURES_ALL
from test_contract import assert_valid_result
from test_linear_backend import write_model


def image_bytes(color=(100, 65, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (96, 80), color).save(buf, "PNG")
    return buf.getvalue()


def onnx_card(tmp_path, **extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model.onnx").write_bytes(b"fake-model")
    card = {"backend": "onnx_rgb", "name": "synthetic-onnx", "model_file": "model.onnx",
            "class_order": ["dark", "light", "medium"], "rgb_view_config": RGBViewConfig(size=32).to_dict(),
            "output_kind": "logits", "temperature": 1.0, "low_conf_threshold": 0.0} | extra
    (tmp_path / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_ort(monkeypatch):
    class Session:
        def __init__(self, path, **kwargs):
            self.kwargs = kwargs
        def get_inputs(self):
            return [types.SimpleNamespace(name="rgb", shape=["batch", 3, 32, 32], type="tensor(float)")]
        def get_outputs(self):
            return [types.SimpleNamespace(name="logits")]
        def get_providers(self):
            return ["CPUExecutionProvider"]
        def run(self, outputs, feed):
            x = feed["rgb"]
            assert x.dtype == np.float32 and x.flags.c_contiguous
            assert x.shape[1:] == (3, 32, 32)
            return [np.tile([1., 3., 2.], (len(x), 1)).astype(np.float32)]
    mod = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=Session, __version__="synthetic")
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def test_load_class_order_fallback_decode_and_profile(tmp_path, fake_ort):
    p = load(onnx_card(tmp_path / "m"))
    for raw in (image_bytes(), image_bytes((255, 255, 255))):
        r = p.predict_bytes(raw)
        assert_valid_result(r)
        assert r["label"] == "light"  # graph order differs from contract order
        profile, stages = p.predict_profile(raw)
        assert profile["probs"] == r["probs"]
        assert stages["n_views"] == 2 and all(stages[k] >= 0 for k in ("decode", "views", "model", "total"))
    assert p.predict_bytes(b"corrupt")["status"] == "bad_image"
    assert p.predict_bytes("path")["status"] == "error"
    assert p.predict_bytes(image_bytes(), source="agtron")["status"] == "error"
    assert p.predict_bytes(image_bytes(), source="rf_robusta", roi="8 8 88 72")["status"] == "error"
    assert p.predict_bytes(image_bytes(), source="agtron", roi="8 8 88 72")["label"] == "light"


def test_onnx_thread_safety_and_low_conf_label(tmp_path, fake_ort):
    p = load(onnx_card(tmp_path / "m", low_conf_threshold=0.99))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(p.predict_bytes, [image_bytes()] * 20))
    assert all(r["status"] == "low_confidence" and r["label"] == "light" for r in results)
    assert len({tuple(r["probs"].items()) for r in results}) == 1


@pytest.mark.parametrize("extra", [{"class_order": ["light", "light", "dark"]}, {"temperature": 0},
                                  {"model_file": "../model.onnx"}, {"output_kind": "wrong"},
                                  {"input_name": "incorrect"}, {"rgb_view_config": {"unknown": True}}])
def test_onnx_loading_errors(tmp_path, fake_ort, extra):
    with pytest.raises(ModelLoadError):
        load(onnx_card(tmp_path / "m", **extra))


def test_onnx_static_batch_and_output_validation(tmp_path, fake_ort, monkeypatch):
    session = fake_ort.InferenceSession
    monkeypatch.setattr(session, "get_inputs", lambda self: [types.SimpleNamespace(name="rgb", shape=[1, 3, 32, 32], type="tensor(float)")])
    p = load(onnx_card(tmp_path / "m"))
    assert p.predict_bytes(image_bytes())["label"] == "light"
    monkeypatch.setattr(session, "run", lambda *args: [np.full((1, 3), np.nan)])
    assert p.predict_bytes(image_bytes())["status"] == "error"


def test_onnx_probability_graph(tmp_path, fake_ort, monkeypatch):
    monkeypatch.setattr(fake_ort.InferenceSession, "run", lambda self, outputs, feed:
                        [np.tile([.6, .3, .1], (len(feed["rgb"]), 1))])
    p = load(onnx_card(tmp_path / "m", output_kind="probabilities"))
    r = p.predict_bytes(image_bytes())
    assert r["label"] == "dark" and r["probs"] == {"light": .3, "medium": .1, "dark": .6}


def test_onnx_missing_dependency_has_load_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ModelLoadError):
        load(onnx_card(tmp_path / "m"))


def test_numpy_mlp_forest_and_bad_trees():
    common = {"classes": ["light", "medium", "dark"], "feature_names": ["L_med"], "temperature": 2.0}
    X = np.zeros((2, len(FEATURES_ALL)))
    X[:, 0] = [-1, 1]
    mlp = NumpyTabularModel(common | {"type": "mlp", "activation": "relu",
                                   "layers": [{"W": [[1., 0., -1.]], "b": [0., 0., 0.]}]})
    assert mlp.proba(X).argmax(1).tolist() == [2, 0]
    extreme = X.copy(); extreme[:, 0] = [-10000, 10000]
    assert mlp.proba(extreme).min() == pytest.approx(np.sqrt(1e-15), rel=1e-6)
    tree = {"children_left": [1, -1, -1], "children_right": [2, -1, -1], "feature": [0, -2, -2],
            "threshold": [0., -2., -2.], "value": [[1, 1, 1], [0, 0, 3], [3, 0, 0]]}
    rf = NumpyTabularModel(common | {"type": "forest", "trees": [tree]})
    assert rf.proba(X).argmax(1).tolist() == [2, 0]
    with pytest.raises(ValueError):
        NumpyTabularModel(common | {"type": "forest", "trees": [tree | {"children_left": [0, -1, -1]}]})


@pytest.mark.parametrize("size,status", [((1, 1), "bad_image"), ((7, 7), "bad_image"),
                                         ((1, 4000), "bad_image"), ((8, 8), "ok")])
@pytest.mark.parametrize("backend", ["b1_linear", "tabular_json"])
def test_tiny_images_are_bad_image(tmp_path, size, status, backend):
    # ด้านสั้นหลัง decode < MIN_SIDE → bad_image ที่ decode (ไม่ขยายภาพแล้วเดา label)
    model = write_model(tmp_path / "m", "b1_linear")
    if backend == "tabular_json":
        card = json.loads((model / "model_card.json").read_text()); card["backend"] = backend
        (model / "model_card.json").write_text(json.dumps(card))
    predictor = load(model)
    buf = io.BytesIO(); Image.new("RGB", size, (100, 65, 40)).save(buf, "PNG")
    result = predictor.predict_bytes(buf.getvalue())
    assert_valid_result(result)
    assert result["status"] == status
    assert (result["label"] is None) == (status == "bad_image")


def test_real_onnx_export_parity_synthetic(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    # Small mean-RGB linear graph checks actual ORT I/O without pretrained data.
    from onnx import TensorProto, helper, numpy_helper
    d = onnx_card(tmp_path / "actual")
    W = np.asarray([[1, -1, .5], [0, 1, -.5], [-1, 0, 1]], np.float32)
    graph = helper.make_graph([
        helper.make_node("GlobalAveragePool", ["rgb"], ["avg"]),
        helper.make_node("Flatten", ["avg"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "W"], ["logits"]),
    ], "synthetic-parity", [helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [None, 3, 32, 32])],
       [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 3])], [numpy_helper.from_array(W, "W")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, d / "model.onnx")
    predictor = load(d)
    raw = image_bytes()
    x = build_rgb_views(decode_image(raw), predictor.backend.config)
    z = x.mean((2, 3)) @ W
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    expected = pool_view_probabilities(p, [0.5, 0.5])
    result = predictor.predict_bytes(raw)
    actual = np.asarray([result["probs"][c] for c in predictor.backend.classes])
    np.testing.assert_allclose(actual, expected, atol=6e-5)
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = list(executor.map(predictor.predict_bytes, [raw] * 12))
    assert all(r["probs"] == result["probs"] and r["label"] == result["label"] for r in parallel)


def test_real_onnx_corrupt_artifact_is_model_load_error(tmp_path):
    pytest.importorskip("onnxruntime")
    with pytest.raises(ModelLoadError, match="session initialization"):
        load(onnx_card(tmp_path / "corrupt"))
