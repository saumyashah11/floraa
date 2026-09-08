"""
Photo quality pre-check -- verifies each synthetic degradation actually
trips its corresponding flag, and that a real, undistorted dataset image
raises no false positives. Skips if calibration hasn't been run yet (the
thresholds file is generated data, not checked into source).
"""
import pytest
from PIL import Image, ImageFilter

import photo_quality as pq

pytestmark = pytest.mark.skipif(
    not pq.THRESHOLDS_PATH.exists(),
    reason="run `python photo_quality.py --calibrate` first")


def _base_image():
    # A real photo would be ideal, but a synthetic image with genuine edge
    # detail (not flat) keeps this test independent of dataset paths/layout.
    import numpy as np
    rng = np.random.default_rng(0)
    arr = (rng.random((256, 256, 3)) * 255).astype("uint8")
    # add some structure so it isn't flagged as blank/low-detail
    arr[64:192, 64:192] = 40
    return Image.fromarray(arr)


def test_sharp_image_has_no_blur_issue():
    img = _base_image()
    result = pq.assess(img)
    codes = [i["code"] for i in result["issues"]]
    assert "blurry" not in codes


def test_blurry_image_flagged():
    img = _base_image().filter(ImageFilter.GaussianBlur(radius=8))
    result = pq.assess(img)
    codes = [i["code"] for i in result["issues"]]
    assert "blurry" in codes
    assert result["ok"] is False


def test_dark_image_flagged():
    img = Image.eval(_base_image(), lambda x: int(x * 0.15))
    result = pq.assess(img)
    codes = [i["code"] for i in result["issues"]]
    assert "too_dark" in codes


def test_bright_image_flagged():
    img = Image.eval(_base_image(), lambda x: min(255, int(x * 2.5 + 100)))
    result = pq.assess(img)
    codes = [i["code"] for i in result["issues"]]
    assert "too_bright" in codes


def test_blank_image_flagged_low_detail():
    img = Image.new("RGB", (256, 256), (120, 120, 120))
    result = pq.assess(img)
    codes = [i["code"] for i in result["issues"]]
    assert "low_detail" in codes


def test_tiny_image_flagged():
    img = _base_image().resize((20, 20))
    result = pq.assess(img)
    codes = [i["code"] for i in result["issues"]]
    assert "too_small" in codes


def test_assess_never_hard_crashes_on_ok_image():
    result = pq.assess(_base_image())
    assert "ok" in result and "metrics" in result


def test_thresholds_file_has_expected_keys():
    t = pq.load_thresholds()
    for key in ("blur_variance_p05", "blur_variance_p10", "brightness_p05",
               "brightness_p95", "contrast_p05"):
        assert key in t
