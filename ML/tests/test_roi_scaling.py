from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from roastml.decode import DecodedImage
from roastml.rgb_views import crop_roi
from tools.compare_roi_scaling import extract_updated, legacy_roi


def test_actual_y_scale_retains_last_roi_row_legacy_drops():
    rgb = np.full((1600, 1200, 3), 80, np.uint8)
    rgb[-1] = 180
    decoded = DecodedImage(Image.fromarray(rgb), (3001, 4000), "JPEG")
    roi = "100 0 1000 3999"
    old = legacy_roi(decoded, roi)
    new = crop_roi(decoded, roi)
    assert old.shape[:2] == (1599, 360)
    assert new.shape[:2] == (1600, 360)
    assert old[-1].mean() == 80 and new[-1].mean() == 180


def test_equal_xy_scale_preserves_pixel_identity():
    rgb = np.arange(80*120*3, dtype=np.uint16).reshape(80, 120, 3).astype(np.uint8)
    decoded = DecodedImage(Image.fromarray(rgb), (240, 160), "PNG")
    assert np.array_equal(legacy_roi(decoded, "20 20 200 140"), crop_roi(decoded, "20 20 200 140"))


def test_legacy_reproduction_validates_roi_before_crop():
    decoded = DecodedImage(Image.new("RGB", (120, 80)), (120, 80), "PNG")
    with pytest.raises(ValueError, match="outside"):
        legacy_roi(decoded, "-5 0 100 80")


def test_extraction_refuses_frozen_rows_before_reading_missing_file(tmp_path):
    row = {"source": "agtron", "split": "test", "path": "must-not-open.jpg", "roi": "0 0 30 30"}
    with pytest.raises(ValueError, match="frozen-test"):
        extract_updated(tmp_path, row, {})
    with pytest.raises(ValueError, match="mandatory Agtron"):
        extract_updated(tmp_path, {**row, "split": "trainval", "roi": ""}, {})
