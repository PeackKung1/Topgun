"""Frozen evaluator guards and cards use only images made in tmp_path."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from PIL import Image

from test_linear_backend import write_model
from tools import eval_test as et
from tools import model_card as mc


def synthetic_data(root):
    root.mkdir()
    rows = []
    for i, (label, color) in enumerate([("light", (170, 125, 85)), ("medium", (110, 75, 50)), ("dark", (45, 30, 22))]):
        rgb = np.empty((80, 96, 3), np.uint8); rgb[:] = color
        Image.fromarray(rgb).save(root / f"{i}.png")
        rows.append({"path": f"{i}.png", "label": label, "source": "agtron", "split": "test",
                     "roi": "8 8 88 72", "device": f"device{i}"})
    rows.append({"path": "missing.png", "label": "dark", "source": "agtron", "split": "trainval"})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split", "roi", "device"])
        writer.writeheader(); writer.writerows(rows)


def test_gate_before_test_rows_images_models_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "load", lambda *a, **k: pytest.fail("model read before confirmation"))
    assert et.main(["--data-dir", str(tmp_path / "missing"), "--model-dir", str(tmp_path / "missing"),
                    "--output-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    with pytest.raises(PermissionError):
        et.load_test_rows(tmp_path / "missing")
    with pytest.raises(PermissionError):
        et.predict_row(None, tmp_path, {"path": "missing"})


def test_frozen_evaluate_once_does_not_modify_model(tmp_path):
    root = tmp_path / "data"; synthetic_data(root)
    model = write_model(tmp_path / "model")
    before = (model / "model_card.json").read_bytes()
    args = ["--confirm-freeze", "--data-dir", str(root), "--model-dir", str(model), "--output-dir", str(tmp_path / "run")]
    assert et.main(args) == 0
    assert et.main(args) == 3
    report = json.loads((tmp_path / "run" / "frozen_test.json").read_text())
    assert report["n"] == 3 and report["metrics"]["acc"] == 1.0
    assert (model / "model_card.json").read_bytes() == before
    assert (tmp_path / "run" / "predictions.csv").is_file()
    assert json.loads((tmp_path / "run" / "freeze_receipt.json").read_text())["status"] == "completed"


def test_missing_roi_rejected_before_image_read(tmp_path):
    with pytest.raises(ValueError, match="ROI"):
        et.predict_row(None, tmp_path, {"split": "test", "source": "agtron", "path": "missing"}, confirmed=True)


def test_wilson_and_calibration():
    assert et.wilson(0, 0) == [0., 0.]
    assert et.wilson(5, 10) == pytest.approx([.2366, .7634], abs=1e-3)
    c = et.calibration(np.asarray(["light", "dark"]), np.asarray([[1, 0, 0], [0, 0, 1]]))
    assert c["ece"] == 0 and c["bins"][-1]["n"] == 2


def test_card_output_is_immutable_and_keeps_model(tmp_path):
    model = write_model(tmp_path / "model")
    before = (model / "model_card.json").read_bytes()
    report = tmp_path / "comparison.json"; report.write_text('{"summary": "synthetic"}')
    out = tmp_path / "card.json"
    args = ["--model-dir", str(model), "--results", str(report), "--output", str(out)]
    assert mc.main(args) == 0
    assert json.loads(out.read_text())["verification"]["frozen_test"] == "not evaluated"
    assert (model / "model_card.json").read_bytes() == before
    with pytest.raises(SystemExit):
        mc.main(args)
