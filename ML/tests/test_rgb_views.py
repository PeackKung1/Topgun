import io

import numpy as np
import pytest
from PIL import Image

from roastml.decode import DecodedImage, decode_image
from roastml.rgb_views import RGBViewConfig, build_rgb_views, crop_roi, pool_view_probabilities, resize_pad


def decoded(size=(80, 40)):
    return DecodedImage(Image.new("RGB", size, (100, 60, 30)), size, "PNG")


def test_full_view_pad_preserves_aspect_and_channel_order():
    arr = np.zeros((10, 20, 3), np.uint8)
    arr[:] = (200, 10, 30)
    p = resize_pad(arr, 40, (1, 2, 3))
    assert tuple(p[0, 0]) == (1, 2, 3)
    assert tuple(p[20, 20]) == (200, 10, 30)
    x = build_rgb_views(decoded(), {"mean": [0, 0, 0], "std": [1, 1, 1], "seg_config": {"white_balance": False}})
    assert x.shape == (2, 3, 224, 224)
    assert x.dtype == np.float32 and x.flags.c_contiguous
    assert x[0, 0, 112, 112] > x[0, 1, 112, 112] > x[0, 2, 112, 112]


def test_roi_after_exif_and_independent_scale():
    im = Image.new("RGB", (40, 80))
    im.paste((255, 0, 0), (0, 0, 20, 40))
    exif = Image.Exif(); exif[274] = 6
    b = io.BytesIO(); im.save(b, format="JPEG", exif=exif)
    d = decode_image(b.getvalue(), max_side=37)
    assert d.orig_size == (80, 40)
    roi = crop_roi(d, "40 0 80 20")
    assert roi.shape[:2] == (round(20 * d.image.height / 40), d.image.width-round(40*d.image.width/80))
    assert roi[..., 0].mean() > 200


@pytest.mark.parametrize("roi", ["", "0 0 81 40", "0 0 0 20", "1 2 3", "x 0 2 3", "-1 0 20 20"])
def test_bad_agtron_roi_never_falls_back_to_labelled_full_image(roi):
    with pytest.raises(ValueError):
        build_rgb_views(decoded(), source="agtron", roi=roi)


def test_valid_agtron_no_wb_and_tiny_image_fallback():
    x = build_rgb_views(decoded(), source="agtron", roi="10 10 30 30")
    assert np.allclose(x[0], x[1])
    assert build_rgb_views(decoded((1, 1))).shape == (2, 3, 224, 224)


def test_log_probability_pooling():
    p = np.array([[.8, .1, .1], [.1, .8, .1]])
    expected = np.sqrt(p[0]*p[1]); expected /= expected.sum()
    assert np.allclose(pool_view_probabilities(p, [.5, .5]), expected)
    with pytest.raises(ValueError):
        pool_view_probabilities(p, [1])
    with pytest.raises(ValueError):
        RGBViewConfig(view_weights=(1, -1))
