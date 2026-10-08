"""Trainval-only sampling and backend-neutral benchmark synthetic checks."""
import csv
import io
import json

import pytest
from PIL import Image
from test_linear_backend import write_model
from tools import bench_candidates as bc


def test_sampling_does_not_select_test(tmp_path):
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "label", "split", "roi"])
        writer.writeheader()
        writer.writerows([{"path": "train.png", "source": "ontoum224", "label": "light", "split": "trainval"},
                         {"path": "forbidden.png", "source": "agtron", "label": "dark", "split": "test"}])
    rows = bc.pick_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["path"] == "train.png"
    with pytest.raises(ValueError, match="trainval"):
        bc.checked_image_path(tmp_path, {"split": "test"})
    with pytest.raises(ValueError, match="ROI"):
        bc.checked_image_path(tmp_path, {"split": "trainval", "path": "missing", "source": "agtron"})


def test_jpeg_and_roi_are_scaled_after_exif():
    im = Image.new("RGB", (4000, 3000), (100, 70, 40))
    exif = im.getexif(); exif[274] = 6
    buf = io.BytesIO(); im.save(buf, "JPEG", exif=exif)
    raw, roi = bc.web1600(buf.getvalue(), "300 400 2700 3600")
    converted = Image.open(io.BytesIO(raw))
    assert converted.size == (1200, 1600) and not converted.getexif()
    assert roi == "120 160 1080 1440"


def test_benchmark_profile_and_artifact_hash_synthetic(tmp_path):
    root = tmp_path / "data"; root.mkdir()
    Image.new("RGB", (96, 80), (100, 70, 50)).save(root / "image.png")
    row = {"path": "image.png", "source": "ontoum224", "label": "medium", "split": "trainval"}
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    model = write_model(tmp_path / "model")
    result = bc.benchmark(model, root, [row])
    assert set(result["cases"]) == {"original", "jpeg1600"}
    for case in result["cases"].values():
        assert set(case["latency_ms"]["ALL"]) >= {"decode", "views", "features", "model", "total", "wall"}
        assert case["statuses"] == {"ok": 1}
    assert result["model"]["artifact_sha256"] == bc.fingerprint(model)["artifact_sha256"]
    assert result["verification"]["pi_latency"] == "unverified"
    json.dumps(result, allow_nan=False)


def test_benchmark_enforces_200_images(tmp_path):
    with (tmp_path / "manifest.csv").open("w") as f:
        f.write("path,source,label,split\n")
    with pytest.raises(SystemExit):
        bc.main(["--model", str(tmp_path / "no-model"), "--data-dir", str(tmp_path), "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
