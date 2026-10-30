"""Stage 2a/2b — Active light source: pulse kernels and scene radiance."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import visionsim.emulate.aspc as aspc
from visionsim.emulate.aspc.emitter import DynamicSource, PulsedLaser
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.utils import radiance_photons, tof2depth, ureg, watts2photons

REL = 1e-9

# Taps of the shipped pulse_shape_custom.txt, and an asymmetric measured response with its peak on the center tap
SHIPPED_TAPS = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
MEASURED_TAPS = [0, 0, 0, 207, 255, 113, 52, 34, 25]
BW = 5 * ureg.centimeter


def _laser(**kw):
    defaults = {
        "wavelength": 550 * ureg.nanometer,
        "pulse_repetition": 10e6 * ureg.hertz,
        "pulse_width": 1 * ureg.nanosecond,
        "avg_watts": 1 * ureg.milliwatt,
        "pulse_shape": "gaussian",
    }
    defaults.update(kw)
    return PulsedLaser(**defaults)


# =========================================================================== #
# Stage 2a — pulse kernels                                                     #
# =========================================================================== #
class TestPulseKernelNormalization:
    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_sum_normalized_integrates_to_one(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert float(np.sum(kernel)) == pytest.approx(1.0, rel=1e-6)

    def test_custom_callable_sum_normalized_to_one(self):
        ds = DynamicSource(
            pulse_width=1 * ureg.nanosecond,
            pulse_shape="custom",
            pulse_shape_custom=lambda z: np.exp(-(z**2)),
        )
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert float(np.sum(kernel)) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_max_normalized_peaks_at_one(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="max")
        assert float(np.max(kernel)) == pytest.approx(1.0, rel=1e-6)


class TestPulseKernelShape:
    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_kernel_is_non_negative(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert float(np.min(kernel)) >= 0.0

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_x_and_kernel_lengths_match(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        x, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert len(x) == len(kernel)

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_wider_pulse_gives_wider_kernel(self, shape):
        bw = 1 * ureg.centimeter
        _, narrow = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape).get_kernel(bw, "sum")
        _, wide = DynamicSource(pulse_width=4 * ureg.nanosecond, pulse_shape=shape).get_kernel(bw, "sum")
        assert len(wide) > len(narrow)

    def test_square_kernel_has_flat_plateau(self):
        ds = DynamicSource(pulse_width=2 * ureg.nanosecond, pulse_shape="square")
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        nonzero = kernel[kernel > 0]
        assert nonzero.size > 0
        assert float(np.ptp(nonzero)) == pytest.approx(0.0, abs=1e-12)

    def test_pulse_shape_custom_is_ignored_by_named_shapes(self):
        plain = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape="gaussian")
        with_taps = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape="gaussian", pulse_shape_custom=[1, 2])
        np.testing.assert_array_equal(plain.get_kernel(BW)[1], with_taps.get_kernel(BW)[1])


# =========================================================================== #
# Stage 2a — custom pulse shapes                                               #
# =========================================================================== #
def _custom(taps, pulse_width=1 * ureg.nanosecond):
    return DynamicSource(pulse_width=pulse_width, pulse_shape="custom", pulse_shape_custom=taps)


class TestCustomPulseTaps:
    """Taps sampled at the bin width, with tap len(taps) // 2 on the bin of the return."""

    @pytest.mark.parametrize("as_type", [list, tuple, np.array])
    def test_taps_are_used_as_given(self, as_type):
        x, kernel = _custom(as_type(MEASURED_TAPS)).get_kernel(BW, normalize="sum")
        expected = np.array(MEASURED_TAPS, dtype=np.float64)
        np.testing.assert_allclose(kernel, expected / expected.sum(), rtol=1e-12)
        np.testing.assert_allclose(x, (np.arange(9) - 4) * 0.05, rtol=1e-12)

    @pytest.mark.parametrize("as_str", [False, True])
    @pytest.mark.parametrize("suffix", [".txt", ".npy"])
    def test_taps_from_file(self, tmp_path, suffix, as_str):
        path = tmp_path / f"irf{suffix}"
        if suffix == ".npy":
            np.save(path, np.array(MEASURED_TAPS, dtype=np.float64))
        else:
            path.write_text("# measured response, one tap per bin\n" + "\n".join(map(str, MEASURED_TAPS)) + "\n")
        _, kernel = _custom(str(path) if as_str else path).get_kernel(BW, normalize=None)
        np.testing.assert_array_equal(kernel, MEASURED_TAPS)

    def test_shipped_taps_file(self):
        path = Path(aspc.__file__).parent / "pulse_shape_custom.txt"
        _, kernel = _custom(path).get_kernel(BW, normalize=None)
        np.testing.assert_array_equal(kernel, SHIPPED_TAPS)

    def test_taps_do_not_depend_on_pulse_width(self):
        _, narrow = _custom(MEASURED_TAPS, pulse_width=0.1 * ureg.nanosecond).get_kernel(BW)
        _, wide = _custom(MEASURED_TAPS, pulse_width=20 * ureg.nanosecond).get_kernel(BW)
        np.testing.assert_array_equal(narrow, wide)

    def test_normalization(self):
        source = _custom(MEASURED_TAPS)
        assert float(np.sum(source.get_kernel(BW, normalize="sum")[1])) == pytest.approx(1.0, rel=1e-12)
        assert float(np.max(source.get_kernel(BW, normalize="max")[1])) == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_array_equal(source.get_kernel(BW, normalize=None)[1], MEASURED_TAPS)

    def test_even_number_of_taps_keeps_its_center(self):
        x, kernel = _custom([1, 2, 3, 4]).get_kernel(BW, normalize=None)
        np.testing.assert_array_equal(kernel, [1, 2, 3, 4, 0])
        np.testing.assert_allclose(x, [-0.1, -0.05, 0.0, 0.05, 0.1], atol=1e-15)

    @pytest.mark.parametrize("taps", [MEASURED_TAPS, [0.0, 1.0, 3.0, 2.0, 0.5, 0.25]], ids=["odd", "even"])
    def test_center_tap_lands_on_the_return(self, taps):
        n_bins, return_bin = 40, 15
        _, kernel = _custom(taps).get_kernel(BW)
        transient = torch.zeros(1, n_bins)
        transient[0, return_bin] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rates = Histogrammer(HistConfig()).calculate_arrival_rates(
                torch.tensor(kernel, dtype=torch.float32), transient, 0.0, n_bins
            )

        expected = np.zeros(n_bins)
        start = return_bin - len(taps) // 2
        expected[start : start + len(taps)] = np.array(taps) / np.sum(taps)
        np.testing.assert_allclose(rates[0].numpy(), expected, rtol=1e-6, atol=1e-7)


class TestCustomPulseCallable:
    def test_callable_is_sampled_at_the_documented_offsets(self):
        seen = []

        def triangle(u):
            seen.append(np.array(u, copy=True))
            return np.clip(1 - np.abs(u), 0, None)

        x, _ = _custom(triangle).get_kernel(BW)
        pw_m = tof2depth(1 * ureg.nanosecond).to(ureg.meter).magnitude
        half_num = max(3 * round(pw_m / 0.05), 5)
        np.testing.assert_allclose(x, np.arange(-half_num, half_num + 1) * 0.05, rtol=1e-12)
        np.testing.assert_allclose(seen[0], x / pw_m, rtol=1e-12)


class TestCustomPulseErrors:
    """Unknown shapes and unusable custom pulses raise instead of becoming a 1-tap delta."""

    @pytest.mark.parametrize("shape", ["gausian", "Gaussian", "delta", "", None, Path("irf.txt")])
    def test_unknown_pulse_shape_raises(self, shape):
        with pytest.raises(ValueError, match="Unknown pulse_shape"):
            DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)

    def test_unknown_pulse_shape_set_after_init_raises(self):
        source = DynamicSource(pulse_width=1 * ureg.nanosecond)
        source.pulse_shape = "gausian"
        with pytest.raises(ValueError, match="Unknown pulse_shape"):
            source.get_kernel(BW)

    def test_custom_requires_pulse_shape_custom(self):
        with pytest.raises(ValueError, match="requires pulse_shape_custom"):
            DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape="custom")

    @pytest.mark.parametrize("as_str", [False, True])
    def test_missing_file_raises(self, tmp_path, as_str):
        path = tmp_path / "missing_irf.txt"
        with pytest.raises(FileNotFoundError, match="missing_irf.txt"):
            _custom(str(path) if as_str else path)

    def test_expression_string_is_not_evaluated(self):
        with pytest.raises(FileNotFoundError, match="!expr"):
            _custom("lambda x: np.sinc(x / 3)")

    @pytest.mark.parametrize(
        "taps, error, match",
        [
            ([], ValueError, "non-empty 1-D"),
            (3.0, ValueError, "non-empty 1-D"),
            ([[0, 1, 0], [0, 2, 0]], ValueError, "non-empty 1-D"),
            ([0, 1, np.nan], ValueError, "non-finite"),
            ([0, 1, np.inf], ValueError, "non-finite"),
            ([0.1, 1, -0.05], ValueError, "negative"),
            ([0, 0, 0], ValueError, "zero everywhere"),
            ({"peak": 1}, TypeError, "pulse_shape_custom must be"),
        ],
        ids=["empty", "scalar", "2d", "nan", "inf", "negative", "zero", "dict"],
    )
    def test_unusable_taps_raise(self, taps, error, match):
        with pytest.raises(error, match=match):
            _custom(taps)

    def test_unusable_taps_in_a_file_raise(self, tmp_path):
        path = tmp_path / "irf.txt"
        path.write_text("0 1 -2 1 0\n")
        with pytest.raises(ValueError, match="negative"):
            _custom(path)

    @pytest.mark.parametrize(
        "func, match",
        [
            (np.cos, "negative"),
            (lambda u: np.full_like(u, np.nan), "non-finite"),
            (np.zeros_like, "zero everywhere"),
            (lambda u: 1.0, "one value per offset"),
        ],
        ids=["negative", "nan", "zero", "scalar"],
    )
    def test_unusable_callable_raises(self, func, match):
        source = _custom(func)
        with pytest.raises(ValueError, match=match):
            source.get_kernel(BW)


# =========================================================================== #
# Stage 2b — active-source scene radiance                                     #
# =========================================================================== #
class TestPulsedLaserRadiance:
    def test_returns_photon_radiance_units(self):
        r = _laser().get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian)
        assert r.to(radiance_photons).magnitude > 0
        assert r.dimensionality == radiance_photons.dimensionality

    def test_inverse_square_falloff(self):
        laser = _laser()
        near = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        far = laser.get_scene_radiance(0.5, 10.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        assert near / far == pytest.approx(4.0, rel=1e-4)

    def test_linear_in_albedo(self):
        laser = _laser()
        lo = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        hi = laser.get_scene_radiance(1.0, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        assert hi / lo == pytest.approx(2.0, rel=REL)

    def test_inverse_in_num_pixels(self):
        laser = _laser()
        few = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        many = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 2000, 1e-6 * ureg.steradian).magnitude
        assert few / many == pytest.approx(2.0, rel=REL)


class TestPulsedLaserDerivedQuantities:
    def test_max_resolvable_depth(self):
        f = 10e6 * ureg.hertz
        laser = _laser(pulse_repetition=f)
        expected = tof2depth(1 / f).to(ureg.meter).magnitude
        assert laser.max_resolvable_depth.to(ureg.meter).magnitude == pytest.approx(expected, rel=REL)

    def test_num_photons_per_cycle(self):
        f, lam, watts = 10e6 * ureg.hertz, 550 * ureg.nanometer, 1 * ureg.milliwatt
        laser = _laser(pulse_repetition=f, wavelength=lam, avg_watts=watts)
        expected = watts2photons(watts, 1 / f, lam).to(ureg.count).magnitude
        assert laser.num_photons_per_cycle.to(ureg.count).magnitude == pytest.approx(expected, rel=REL)

    def test_higher_pulse_repetition_shrinks_max_depth(self):
        slow = _laser(pulse_repetition=5e6 * ureg.hertz).max_resolvable_depth.to(ureg.meter).magnitude
        fast = _laser(pulse_repetition=20e6 * ureg.hertz).max_resolvable_depth.to(ureg.meter).magnitude
        assert fast < slow
