import numpy as np
import pytest

from visionsim.utils.color import linearrgb_to_srgb, srgb_to_linearrgb, to_linearrgb


def test_linearrgb_to_srgb_and_back():
    img = np.random.random((100, 100))
    round_trip_img = srgb_to_linearrgb(linearrgb_to_srgb(img))
    assert np.allclose(img, round_trip_img)


def test_srgb_to_linearrgb_and_back():
    img = np.random.random((100, 100))
    round_trip_img = linearrgb_to_srgb(srgb_to_linearrgb(img))
    assert np.allclose(img, round_trip_img)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_to_linearrgb_display_referred(dtype):
    max_value = np.iinfo(dtype).max
    img = np.random.randint(0, max_value + 1, size=(32, 32, 3)).astype(dtype)
    assert np.allclose(to_linearrgb(img, "frame.png"), srgb_to_linearrgb(img / max_value))


@pytest.mark.parametrize("suffix", [".exr", ".hdr", ".EXR"])
def test_to_linearrgb_scene_referred(suffix):
    # Linear intensities, including highlights above one, are passed through untouched
    img = (np.random.random((32, 32, 3)) * 10).astype(np.float32)
    assert np.array_equal(to_linearrgb(img, f"frame{suffix}"), img)


def test_to_linearrgb_format_agnostic():
    # The same scene saved as an 8-bit PNG or as an EXR yields the same intensities, up to quantization
    linear = np.random.random((32, 32, 3))
    png = np.round(linearrgb_to_srgb(linear) * 255).astype(np.uint8)
    assert np.allclose(to_linearrgb(png, "frame.png"), to_linearrgb(linear, "frame.exr"), atol=5e-3)


def test_to_linearrgb_clips_negative_intensities():
    # Denoisers can leave small negative values in linear renders, which cannot be sampled as photon counts
    img = np.array([[[-1e-5, 0.5, np.nan]]], dtype=np.float32)
    linear = to_linearrgb(img, "frame.exr")
    assert linear[0, 0, 0] == 0 and linear[0, 0, 1] == 0.5 and np.isnan(linear[0, 0, 2])
