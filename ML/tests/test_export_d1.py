from __future__ import annotations

import numpy as np
import pytest

from roastml.rgb_views import RGBViewConfig, pool_view_probabilities
from tools.compare_tabular import softmax
from tools.export_d1 import calibration_ids, export_fp32, normalise_pixels, ort_session, pool_logits, quantize_int8


def test_calibration_selects_only_training_rows_reproducibly():
    rows = [{"path": f"p{i:03}", "source": f"s{i % 2}", "label": ["light", "medium", "dark"][i % 3]} for i in range(60)]
    train = np.arange(0, 48)
    actual = calibration_ids(rows, train, 12)
    assert len(actual) == len(set(actual)) == 12
    assert set(actual) <= set(train)
    assert set(rows[int(i)]["label"] for i in actual) == {"light", "medium", "dark"}
    np.testing.assert_array_equal(actual, calibration_ids(rows, train, 12))


def test_normalisation_matches_shared_rgb_values():
    cfg = RGBViewConfig()
    pixels = np.zeros((1, 2, 224, 224, 3), np.uint8)
    actual = normalise_pixels(pixels, cfg.to_dict())
    assert actual.shape == (1, 2, 3, 224, 224)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_allclose(actual[0, 0, :, 0, 0], -np.array(cfg.mean) / np.array(cfg.std), rtol=1e-6)


def test_pooling_is_runtime_pooling_including_saturation():
    logits = np.array([[[10000, 0, -10000], [-10, 0, 10]], [[1, 2, 0], [-1, 0, 1]]], float)
    actual = pool_logits(logits, 1.5, [0.5, 0.5])
    expected = np.array([pool_view_probabilities(softmax(z / 1.5), [0.5, 0.5]) for z in logits])
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_real_cpu_onnx_dynamic_batch_and_static_qdq(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3)).eval()
    cfg = RGBViewConfig(size=32).to_dict()
    images = np.random.default_rng(1).integers(0, 256, (6, 2, 32, 32, 3), dtype=np.uint8)
    fp32, int8 = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    meta = export_fp32(model, fp32, 32)
    assert meta["dynamic_batch_verified"]
    quant = quantize_int8(fp32, int8, images, np.arange(4), cfg)
    assert quant["calibration_images"] == 4 and quant["calibration_views"] == 8
    assert quant["node_counts"]["QuantizeLinear"] > 0
    X = normalise_pixels(images[4:6], cfg).reshape(4, 3, 32, 32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(X)).numpy()
    actual = ort_session(fp32).run(["logits"], {"rgb": X})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    quantized = ort_session(int8).run(["logits"], {"rgb": X})[0]
    assert np.isfinite(quantized).all() and quantized.shape == (4, 3)
