"""Each return keeps its position within its bin when the pulse is applied.

A time-to-digital converter (TDC) with bins [j, j+1) records a photon in bin floor(t / bin_width). A return that
arrives at t = (k + f) bins, with a pulse p, therefore gives bin j the expected share
``integral over [j, j+1) of p(t' - t) dt'``, which moves continuously with f. The transients hold the returns without a
pulse (an ideal TDC); the arrival rates spread each return by the pulse from its exact arrival time, using the pulse's
cumulative distribution from ``DynamicSource.get_pulse_cdf``. Before, each return was floored to its bin first and then
convolved with the pulse sampled at whole bins, so every return in a bin gave the same histogram: a centroid error of
0.5 - f bins (RMS 0.29 bin) against a TDC, and a peak narrower by a variance of 1/12 bin^2.
"""

import numpy as np
import pytest
import torch
from scipy.constants import c as C_LIGHT
from scipy.special import ndtr

from visionsim.emulate.aspc.emitter import (
    CDF_MAX_STEPS_PER_BIN,
    CDF_STEPS_PER_SIGMA,
    CDF_STEPS_PER_TAP,
    CDF_STEPS_PER_WIDTH,
    DynamicSource,
)
from visionsim.emulate.aspc.emulator import ASPCEmulator
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.utils import ureg

BW_M, N_BINS = 0.01, 64  # 1 cm bins
PERIOD_M = N_BINS * BW_M  # the window is one laser period
FWHM = 2.35482
TAPS = [0.5, 3.0, 1.0, 0.25, 0.0, 0.1]  # asymmetric, even count
# Copies of 6 returns that make a zone large enough for the faster route of every pulse below (more returns than
# bins times nodes, about 515 nodes for the narrowest Gaussian)
MANY = 6000
# Cumulative distributions with jumps (a repeated x), as their point masses (share, offset) and uniform parts
# (share, start, end), all in bins
JUMPS = {
    "jump-at-the-return": (([0.0, 0.0, 1.0], [0.0, 0.5, 1.0]), [(0.5, 0.0)], [(0.5, 0.0, 1.0)]),
    "jump-inside": (
        ([-1.0, 0.7, 0.7, 2.0], [0.0, 0.3, 0.8, 1.0]),
        [(0.5, 0.7)],
        [(0.3, -1.0, 0.7), (0.2, 0.7, 2.0)],
    ),
    "jump-at-the-end": (([-0.4, 1.3, 1.3], [0.0, 0.25, 1.0]), [(0.75, 1.3)], [(0.25, -0.4, 1.3)]),
}


def _source(shape="gaussian", width_bins=2.0, **kwargs):
    """A source whose pulse_width is width_bins bins of BW_M."""
    pulse_width = 2 * width_bins * BW_M / C_LIGHT * ureg.second
    return DynamicSource(pulse_width=pulse_width, pulse_shape=shape, **kwargs)


def _gaussian_cdf(sigma_bins):
    return _source("gaussian", sigma_bins * FWHM).get_pulse_cdf(BW_M * ureg.meter)


def _gaussian_shares(tau, sigma, n_bins=N_BINS):
    """Exact shares of the bins [j, j+1) of a Gaussian of sigma centered on each tau (all in bins)."""
    edges = np.arange(n_bins + 1)
    return np.diff(ndtr((edges - np.asarray(tau, dtype=np.float64)[..., None]) / sigma), axis=-1)


def _uniform_shares(tau, lo, hi, n_bins=N_BINS):
    """Shares of the bins [j, j+1) of a pulse uniform over [tau + lo, tau + hi)."""
    j = np.arange(n_bins)
    t = np.asarray(tau, dtype=np.float64)[..., None]
    return np.clip(np.minimum(j + 1, t + hi) - np.maximum(j, t + lo), 0, None) / (hi - lo)


def _point_shares(tau, offset, n_bins=N_BINS):
    """Shares of a pulse whose photons all arrive offset bins after tau: all in the bin of tau + offset."""
    j = np.floor(np.asarray(tau, dtype=np.float64) + offset)
    return (np.arange(n_bins) == j[..., None]).astype(np.float64)


def _gaussian_taps(sigma, offset=0.0, half=25):
    """Taps of a Gaussian of sigma centered offset bins after a return at the center of a bin: its bin integrals."""
    i = np.arange(-half, half + 1)
    return ndtr((i + 0.5 - offset) / sigma) - ndtr((i - 0.5 - offset) / sigma)


def _zone(tau, weights=None, pulse_cdf=None, n_bins=N_BINS, dtype=torch.float64, **kwargs):
    """calculate_transients of one zone over frames of returns at tau bins, shape (H, W) or (F, H, W)."""
    tau = np.asarray(tau, dtype=np.float64)
    depth = torch.tensor(tau * BW_M, dtype=dtype)
    irradiance = torch.ones_like(depth) if weights is None else torch.tensor(weights, dtype=dtype)
    masks = torch.ones((1, *depth.shape[-2:]))
    kwargs = {"bin_width": BW_M, **kwargs}
    max_depth = kwargs.pop("max_depth", PERIOD_M)
    transients, _ = Histogrammer(HistConfig()).calculate_transients(
        irradiance, depth, torch.zeros_like(depth), masks, n_bins, max_depth, pulse_cdf=pulse_cdf, **kwargs
    )
    return transients


def _returns(n=6, seed=0, lo=10.0, hi=50.0):
    """Returns at random ranges (in bins) with random weights, as a (2, n // 2) frame."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (2, n // 2)), rng.uniform(0.2, 1.0, (2, n // 2))


def _expected(tau, weights, shares_fn):
    """The zone's row: the mean over its pixels of weight times the shares."""
    shares = shares_fn(tau.ravel())
    return (weights.ravel()[:, None] * shares).sum(0) / tau.size


def _tiled(tau, weights, copies):
    """The same returns repeated, which leaves the zone mean unchanged but makes the zone large."""
    return np.tile(tau, (1, copies)), np.tile(weights, (1, copies))


def _centroid(rows):
    j = np.arange(rows.shape[-1])
    return (rows * j).sum(-1) / rows.sum(-1)


def _variance(rows):
    j = np.arange(rows.shape[-1])
    return (rows * (j - _centroid(rows)[..., None]) ** 2).sum(-1) / rows.sum(-1)


# =========================================================================== #
# DynamicSource.get_pulse_cdf                                                 #
# =========================================================================== #
class TestGetPulseCdf:
    @pytest.mark.parametrize("sigma", [0.05, 0.3, 1.0, 2.5, 17.0, 40.0])
    def test_gaussian_is_exact_at_steps_of_sigma_over_32(self, sigma):
        x, cdf = _gaussian_cdf(sigma)
        steps = min(np.ceil(CDF_STEPS_PER_SIGMA / sigma), CDF_MAX_STEPS_PER_BIN)
        np.testing.assert_allclose(np.diff(x), 1 / steps, rtol=1e-9)
        assert 1 / steps <= sigma / 32 + 1e-12
        assert x[0] <= -8 * sigma and x[-1] >= 8 * sigma
        np.testing.assert_allclose(cdf, ndtr(x / sigma), rtol=0, atol=1e-14)
        assert cdf[0] == 0 and cdf[-1] == 1

    def test_gaussian_steps_are_never_finer_than_the_cap(self):
        x, cdf = _gaussian_cdf(1e-5)
        np.testing.assert_allclose(np.diff(x), 1 / CDF_MAX_STEPS_PER_BIN, rtol=1e-9)
        np.testing.assert_allclose(cdf, [0.0, 0.5, 1.0], atol=1e-15)

    @pytest.mark.parametrize("width", [0.3, 1.0, 2.5])
    def test_square_is_uniform_over_its_width(self, width):
        x, cdf = _source("square", width).get_pulse_cdf(BW_M * ureg.meter)
        np.testing.assert_allclose(x, [-width / 2, width / 2], rtol=1e-12)
        np.testing.assert_array_equal(cdf, [0.0, 1.0])

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_zero_width_is_a_step_at_the_return(self, shape):
        x, cdf = _source(shape, 0.0).get_pulse_cdf(BW_M * ureg.meter)
        np.testing.assert_array_equal(x, [0.0, 0.0])
        np.testing.assert_array_equal(cdf, [0.0, 1.0])

    @pytest.mark.parametrize(
        "taps",
        [TAPS, [1.0, 2.0, 1.0], [1.0], [0.0, 5.0, 0.0, 0.0, 1e-12, 3.0, 3.0, 3.0, 0.0, 1.0]],
        ids=["even", "odd", "one", "steps"],
    )
    def test_taps_fix_the_cdf_at_the_edges_of_their_bins(self, taps):
        """The edges of the taps' bins are among the steps, with the taps' cumulative sums; the cdf never falls."""
        x, cdf = _source("custom", pulse_shape_custom=taps).get_pulse_cdf(BW_M * ureg.meter)
        edges = np.arange(len(taps) + 1) - len(taps) // 2 - 0.5
        np.testing.assert_allclose(np.diff(x), 1 / CDF_STEPS_PER_TAP, rtol=1e-9)
        np.testing.assert_array_equal(x[::CDF_STEPS_PER_TAP], edges)
        np.testing.assert_allclose(
            cdf[::CDF_STEPS_PER_TAP], np.concatenate(([0.0], np.cumsum(taps))) / np.sum(taps), rtol=0, atol=1e-15
        )
        assert cdf[0] == 0 and cdf[-1] == 1 and np.all(np.diff(cdf) >= 0)

    @pytest.mark.parametrize("taps", [[1.0, 2.0, 1.0], TAPS, list(_gaussian_taps(1.0, 0.3, half=3))])
    def test_zero_taps_around_the_pulse_do_not_change_it(self, taps):
        """Guard: as many zeros before the taps as after them (time zero stays) leave the pulse as it is."""
        padded = [0.0, 0.0, *taps, 0.0, 0.0]
        t = np.linspace(-8, 8, 3001)
        cdf = _source("custom", pulse_shape_custom=taps).get_pulse_cdf(BW_M * ureg.meter)
        padded_cdf = _source("custom", pulse_shape_custom=padded).get_pulse_cdf(BW_M * ureg.meter)
        np.testing.assert_allclose(np.interp(t, *padded_cdf), np.interp(t, *cdf), rtol=0, atol=1e-15)

    def test_taps_do_not_jump_at_the_edges_of_their_bins(self):
        """The pulse's density is continuous, and 0 at both ends; with each tap uniform within its bin, it jumped by
        a quarter of the photons per bin at every edge of [1, 2, 1]."""
        _, cdf = _source("custom", pulse_shape_custom=[1.0, 2.0, 1.0]).get_pulse_cdf(BW_M * ureg.meter)
        density = np.diff(cdf) * CDF_STEPS_PER_TAP  # photons per bin, over each step
        inner = np.arange(1, 3) * CDF_STEPS_PER_TAP  # the steps that start at the inner edges
        assert np.abs(density[inner] - density[inner - 1]).max() < 0.01
        assert density[0] < 0.02 and density[-1] < 0.02

    def test_taps_do_not_depend_on_pulse_width(self):
        narrow = _source("custom", 0.1, pulse_shape_custom=TAPS).get_pulse_cdf(BW_M * ureg.meter)
        wide = _source("custom", 30.0, pulse_shape_custom=TAPS).get_pulse_cdf(BW_M * ureg.meter)
        np.testing.assert_array_equal(narrow[0], wide[0])
        np.testing.assert_array_equal(narrow[1], wide[1])

    @pytest.mark.parametrize("width,steps", [(2.0, CDF_STEPS_PER_WIDTH), (0.25, 4 * CDF_STEPS_PER_WIDTH)])
    def test_callable_is_sampled_at_step_centers(self, width, steps):
        seen = []

        def triangle(u):
            seen.append(np.array(u, copy=True))
            return np.clip(1 - np.abs(u), 0, None)

        x, cdf = _source("custom", width, pulse_shape_custom=triangle).get_pulse_cdf(BW_M * ureg.meter)
        half_num = max(3 * max(1, round(width)), 5) * steps
        centers = np.arange(-half_num, half_num + 1) / steps
        np.testing.assert_allclose(seen[0], centers / width, rtol=1e-12)
        np.testing.assert_allclose(x, np.append(centers - 0.5 / steps, centers[-1] + 0.5 / steps), atol=1e-12)
        density = np.clip(1 - np.abs(centers / width), 0, None)
        np.testing.assert_allclose(np.diff(cdf), density / density.sum(), atol=1e-15)

    def test_callable_narrower_than_the_cap_uses_the_cap(self):
        x, _ = _source("custom", 1e-4, pulse_shape_custom=lambda u: np.exp(-(u**2))).get_pulse_cdf(BW_M * ureg.meter)
        np.testing.assert_allclose(np.diff(x), 1 / CDF_MAX_STEPS_PER_BIN, rtol=1e-9)

    @pytest.mark.parametrize("bad", [np.cos, lambda u: np.full_like(u, np.nan), np.zeros_like, lambda u: 1.0])
    def test_unusable_callable_raises(self, bad):
        with pytest.raises(ValueError):
            _source("custom", 2.0, pulse_shape_custom=bad).get_pulse_cdf(BW_M * ureg.meter)

    @pytest.mark.parametrize("method", ["get_pulse_cdf", "get_kernel"])
    def test_callable_needs_a_positive_pulse_width(self, method):
        """Its offsets are in units of pulse_width; a width of 0 raised a ZeroDivisionError or a non-finite pulse."""
        source = _source("custom", 0.0, pulse_shape_custom=lambda u: np.exp(-(u**2)))
        with pytest.raises(ValueError, match="pulse_width"):
            getattr(source, method)(BW_M * ureg.meter)

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_bin_width_as_time_or_distance(self, shape):
        source = _source(shape, 2.5)
        x_m, cdf_m = source.get_pulse_cdf(BW_M * ureg.meter)
        x_s, cdf_s = source.get_pulse_cdf(2 * BW_M / C_LIGHT * ureg.second)
        np.testing.assert_allclose(x_s, x_m, rtol=1e-9)
        np.testing.assert_allclose(cdf_s, cdf_m, rtol=1e-9, atol=1e-15)

    def test_unknown_shape_set_after_init_raises(self):
        source = _source("gaussian", 2.0)
        source.pulse_shape = "gausian"
        with pytest.raises(ValueError, match="Unknown pulse_shape"):
            source.get_pulse_cdf(BW_M * ureg.meter)


# =========================================================================== #
# calculate_transients with a pulse                                           #
# =========================================================================== #
class TestSpreadReturns:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("bin_width", [BW_M, None])
    @pytest.mark.parametrize("time_offset", [0.0, 0.123])
    def test_zero_width_pulse_gives_the_transient(self, dtype, bin_width, time_offset):
        tau, weights = _returns(400)
        tau[0, :4] = [10.0, 20.0, 30.5, 63.999]  # bin edges and the end of the window
        cdf = _source("gaussian", 0.0).get_pulse_cdf(BW_M * ureg.meter)
        kwargs = {"dtype": dtype, "bin_width": bin_width, "time_offset": time_offset}
        spread = _zone(tau, weights, pulse_cdf=cdf, **kwargs)
        assert spread.dtype == dtype
        torch.testing.assert_close(spread, _zone(tau, weights, **kwargs), rtol=1e-6, atol=0)

    @pytest.mark.parametrize("sigma", [0.04, 0.1, 0.35, 0.88, 2.55, 17.0])
    @pytest.mark.parametrize("copies", [1, MANY], ids=["few-returns", "many-returns"])
    def test_gaussian_matches_its_bin_integrals(self, sigma, copies):
        tau, weights = _tiled(*_returns(6), copies)
        spread = _zone(tau, weights, pulse_cdf=_gaussian_cdf(sigma))[0].numpy()
        expected = _expected(tau, weights, lambda t: _gaussian_shares(t, sigma))
        np.testing.assert_allclose(spread, expected, rtol=0, atol=1e-4 * weights.mean())

    @pytest.mark.parametrize("sigma", [1 / 32, 0.05, 0.3, 0.5, 1.0, 3.0, 17.0])
    def test_gaussian_share_of_every_bin_is_within_1e_4(self, sigma):
        tau = 30 + (np.arange(400) + 0.5) / 400
        rows = _zone(tau[:, None, None], pulse_cdf=_gaussian_cdf(sigma)).numpy()
        np.testing.assert_allclose(rows, _gaussian_shares(tau, sigma), rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "pulse",
        [
            ("gaussian", 0.1 * FWHM, None),
            ("gaussian", 2.55 * FWHM, None),
            ("square", 2.3, None),
            ("custom", 1.5, lambda u: np.exp(-(u**2))),
            ("custom", 2.0, TAPS),
        ],
        ids=["narrow-gaussian", "gaussian", "square", "callable", "taps"],
    )
    def test_many_returns_give_the_same_row_as_few(self, pulse):
        """Large zones take a faster route, which must not change the result."""
        shape, width, custom = pulse
        kwargs = {} if custom is None else {"pulse_shape_custom": custom}
        cdf = _source(shape, width, **kwargs).get_pulse_cdf(BW_M * ureg.meter)
        tau, weights = _returns(6)
        few = _zone(tau, weights, pulse_cdf=cdf)
        many = _zone(*_tiled(tau, weights, MANY), pulse_cdf=cdf)
        torch.testing.assert_close(many, few, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("width", [0.3, 1.0, 2.3])
    @pytest.mark.parametrize("copies", [1, MANY], ids=["few-returns", "many-returns"])
    def test_square_is_exact(self, width, copies):
        tau, weights = _tiled(*_returns(6), copies)
        spread = _zone(tau, weights, pulse_cdf=_source("square", width).get_pulse_cdf(BW_M * ureg.meter))
        expected = _expected(tau, weights, lambda t: _uniform_shares(t, -width / 2, width / 2))
        np.testing.assert_allclose(spread[0].numpy(), expected, rtol=0, atol=1e-12)

    @pytest.mark.parametrize(
        "fwhm, offset, peak_error, centroid_error, variance_error",
        [
            (2.45, 0.0, 0.010, 2e-5, 3e-4),
            (2.45, 0.5, 0.014, 2e-5, 3e-4),
            (4.0, 0.5, 0.0014, 2e-5, 3e-4),
            (8.0, 0.0, 1e-4, 2e-5, 3e-4),
            (1.5, 0.0, 0.005, 0.004, 0.008),
        ],
    )
    def test_taps_of_a_gaussian_match_its_bin_integrals(self, fwhm, offset, peak_error, centroid_error, variance_error):
        """The accuracy that DynamicSource states for taps, for returns all over a bin.

        Each tap was uniform within its bin, which widened a return d bins from the center of its bin by a variance of
        |d| * (1 - |d|) bin^2: for 2.45 bins, a bin was off by up to 8% of the peak share and the variance by 1/4 bin^2.
        """
        sigma = fwhm / FWHM
        taps = _gaussian_taps(sigma, offset)
        cdf = _source("custom", pulse_shape_custom=taps).get_pulse_cdf(BW_M * ureg.meter)
        tau = 30 + (np.arange(400) + 0.5) / 400
        rows = _zone(tau[:, None, None], pulse_cdf=cdf).numpy()
        exact = _gaussian_shares(tau + offset, sigma)
        assert np.abs(rows - exact).max() < peak_error * exact.max()
        assert np.abs(_centroid(rows) - _centroid(exact)).max() < centroid_error
        assert np.abs(_variance(rows) - _variance(exact)).max() < variance_error

    @pytest.mark.parametrize("copies", [1, MANY], ids=["few-returns", "many-returns"])
    def test_taps_of_a_gaussian_match_its_bin_integrals_in_large_zones(self, copies):
        sigma = 2.45 / FWHM
        tau, weights = _tiled(*_returns(6), copies)
        cdf = _source("custom", pulse_shape_custom=_gaussian_taps(sigma)).get_pulse_cdf(BW_M * ureg.meter)
        spread = _zone(tau, weights, pulse_cdf=cdf)[0].numpy()
        expected = _expected(tau, weights, lambda t: _gaussian_shares(t, sigma))
        np.testing.assert_allclose(spread, expected, rtol=0, atol=0.01 * expected.max())

    def test_taps_at_the_center_of_a_bin_are_the_taps(self):
        cdf = _source("custom", pulse_shape_custom=TAPS).get_pulse_cdf(BW_M * ureg.meter)
        spread = _zone(np.full((1, 1), 30.5), pulse_cdf=cdf)[0].numpy()
        expected = np.zeros(N_BINS)
        expected[30 - len(TAPS) // 2 : 30 - len(TAPS) // 2 + len(TAPS)] = np.array(TAPS) / np.sum(TAPS)
        np.testing.assert_allclose(spread, expected, atol=1e-15)

    def test_taps_at_the_start_of_a_bin_are_split_with_the_bin_before(self):
        """A single tap is a bump over the bin centered on the return, whose cdf is 3 s^2 - 2 s^3 at s of that bin."""
        cdf = _source("custom", pulse_shape_custom=[1.0]).get_pulse_cdf(BW_M * ureg.meter)
        spread = _zone(np.full((1, 1), 30.25), pulse_cdf=cdf)[0].numpy()
        # The bin centered on 30.25 starts at 29.75, so 0.25 of it falls in bin 29
        assert spread[29] == pytest.approx(3 * 0.25**2 - 2 * 0.25**3) and spread[30] == pytest.approx(0.84375)
        assert spread.sum() == pytest.approx(1.0)

    def test_callable_gaussian_matches_the_gaussian(self):
        sigma = 1.2
        callable_cdf = _source(
            "custom", sigma * FWHM, pulse_shape_custom=lambda u: np.exp(-0.5 * (u * FWHM) ** 2)
        ).get_pulse_cdf(BW_M * ureg.meter)
        tau, weights = _returns(40)
        spread = _zone(tau, weights, pulse_cdf=callable_cdf)[0].numpy()
        expected = _expected(tau, weights, lambda t: _gaussian_shares(t, sigma))
        np.testing.assert_allclose(spread, expected, rtol=0, atol=1e-3 * weights.mean())

    def test_gaussian_narrower_than_the_finest_step_lands_in_the_bin_of_the_return(self):
        """Away from the bin edges, a very narrow Gaussian acts like no pulse at all."""
        rng = np.random.default_rng(1)
        tau = rng.integers(5, 60, (4, 25)) + rng.uniform(2 / CDF_MAX_STEPS_PER_BIN, 1 - 2 / CDF_MAX_STEPS_PER_BIN)
        spread = _zone(tau, pulse_cdf=_gaussian_cdf(1e-6))
        torch.testing.assert_close(spread, _zone(tau), rtol=1e-12, atol=0)

    @pytest.mark.parametrize("name", list(JUMPS))
    @pytest.mark.parametrize("copies", [1, MANY], ids=["few-returns", "many-returns"])
    def test_jump_in_the_cdf_is_a_point_mass(self, name, copies):
        """The photons of a jump arrive at one offset; large zones interpolated straight across it."""
        pulse_cdf, points, uniforms = JUMPS[name]
        tau, weights = _tiled(*_returns(12), copies)  # phases on both sides of every jump
        spread = _zone(tau, weights, pulse_cdf=pulse_cdf)[0].numpy()

        def shares(t):
            point = sum(share * _point_shares(t, offset) for share, offset in points)
            return point + sum(share * _uniform_shares(t, lo, hi) for share, lo, hi in uniforms)

        np.testing.assert_allclose(spread, _expected(tau, weights, shares), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("n_returns", [1, 200])
    def test_jump_does_not_depend_on_the_size_of_the_zone(self, n_returns):
        # Half the photons with the return, half over the next bin: 0.5 + 0.5 * 0.05 in bin 30, 0.5 * 0.95 in bin 31
        row = _zone(np.full((1, n_returns), 30.95), pulse_cdf=([0.0, 0.0, 1.0], [0.0, 0.5, 1.0]))[0].numpy()
        np.testing.assert_allclose(row[29:33], [0.0, 0.525, 0.475, 0.0], rtol=0, atol=1e-9)

    @pytest.mark.parametrize("copies", [1, MANY], ids=["few-returns", "many-returns"])
    def test_zero_width_pulse_away_from_the_return_is_delayed(self, copies):
        """A step at x[0] = 2.5 bins puts every photon 2.5 bins after the return, as a step of 1e-9 bins there does."""
        step = ([2.5, 2.5], [0.0, 1.0])
        assert np.flatnonzero(_zone(np.full((1, 1), 30.2), pulse_cdf=step)[0].numpy()).tolist() == [32]

        tau, weights = _tiled(*_returns(6), copies)
        spread = _zone(tau, weights, pulse_cdf=step)[0].numpy()
        expected = _expected(tau, weights, lambda t: _point_shares(t, 2.5))
        np.testing.assert_allclose(spread, expected, rtol=0, atol=1e-12)
        narrow = _zone(tau, weights, pulse_cdf=([2.5, 2.5 + 1e-9], [0.0, 1.0]))[0].numpy()
        np.testing.assert_allclose(narrow, expected, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("pulse_width_s", [1e-20, 1e-24])
    @pytest.mark.parametrize("copies", [1, MANY], ids=["few-returns", "many-returns"])
    def test_square_much_narrower_than_a_bin_lands_in_the_bin_of_the_return(self, pulse_width_s, copies):
        """Its edges are closer than the rounding of the large-zone route, which spread it over three bins."""
        source = DynamicSource(pulse_width=pulse_width_s * ureg.second, pulse_shape="square")
        cdf = source.get_pulse_cdf(BW_M * ureg.meter)
        tau, weights = _tiled(*_returns(6), copies)
        torch.testing.assert_close(_zone(tau, weights, pulse_cdf=cdf), _zone(tau, weights), rtol=1e-12, atol=0)

    @pytest.mark.parametrize("sigma", [0.88, 1.04, 2.55])
    def test_centroid_follows_the_return_within_its_bin(self, sigma):
        """The TDC's centroid is tau - 0.5 bins; flooring first gave every return in a bin the same centroid."""
        phases = (np.arange(200) + 0.5) / 200
        tau = 30 + phases
        cdf = _gaussian_cdf(sigma)
        rows = np.stack([_zone(np.full((1, 1), t), pulse_cdf=cdf)[0].numpy() for t in tau])
        tdc = _gaussian_shares(tau, sigma)
        error = _centroid(rows) - _centroid(tdc)
        assert np.sqrt(np.mean(error**2)) < 1e-6
        assert np.abs(error).max() < 1e-6
        np.testing.assert_allclose(_centroid(rows), tau - 0.5, atol=1e-6)
        # Integrating over the bins widens the peak by the variance of a bin, 1/12 bin^2, as a TDC does; the steps of
        # the pulse's cumulative distribution add step^2 / 6 on average, and less than step^2 / 2 at any phase
        step = 1 / np.ceil(CDF_STEPS_PER_SIGMA / sigma)
        gap = _variance(rows) - _variance(tdc)
        assert np.all(gap >= -1e-6) and np.all(gap < step**2 / 2)
        assert gap.mean() == pytest.approx(step**2 / 6, rel=0.05)
        np.testing.assert_allclose(_variance(tdc), sigma**2 + 1 / 12, atol=1e-6)

    def test_photons_are_kept_inside_the_window(self):
        tau, weights = _returns(40, lo=15, hi=45)
        transient = _zone(tau, weights)
        spread = _zone(tau, weights, pulse_cdf=_gaussian_cdf(1.5))
        assert float(spread.sum()) == pytest.approx(float(transient.sum()), rel=1e-12)

    def test_pulse_is_cut_at_both_ends_of_the_window(self):
        sigma, tau = 1.5, np.array([[0.3, 63.6]])
        spread = _zone(tau, pulse_cdf=_gaussian_cdf(sigma))[0].numpy()
        expected = _gaussian_shares(tau.ravel(), sigma).sum(0) / 2
        np.testing.assert_allclose(spread, expected, atol=1e-4)
        assert spread.sum() < 1 - 0.2

    def test_returns_after_a_shorter_window_add_nothing(self):
        """A window of 64 bins in a period of 100 bins: a return at 80 bins is not recorded, pulse and all."""
        cdf = _gaussian_cdf(3.0)
        spread = _zone(np.array([[80.0, 70.2]]), pulse_cdf=cdf, max_depth=100 * BW_M)
        assert float(spread.abs().sum()) == 0.0

    @pytest.mark.parametrize("time_offset_bins", [0.0, 7.37, -3.62])
    def test_aliasing_and_time_offset_keep_the_phase(self, time_offset_bins):
        sigma = 1.3
        tau = np.array([[20.37, 20.37 + N_BINS], [33.81, 33.81 + 3 * N_BINS]])
        spread = _zone(tau, pulse_cdf=_gaussian_cdf(sigma), time_offset=time_offset_bins * BW_M)[0].numpy()
        arrival = np.remainder(tau.ravel() + time_offset_bins, N_BINS)
        expected = _gaussian_shares(arrival, sigma).sum(0) / tau.size
        np.testing.assert_allclose(spread, expected, rtol=0, atol=1e-4)

    def test_frame_batch_matches_single_frames(self):
        cdf = _gaussian_cdf(0.9)
        frames = [_returns(30, seed=s) for s in range(3)]
        batch = _zone(np.stack([t for t, _ in frames]), np.stack([w for _, w in frames]), pulse_cdf=cdf)
        single = torch.cat([_zone(t, w, pulse_cdf=cdf) for t, w in frames])
        torch.testing.assert_close(batch, single, rtol=1e-12, atol=0)

    def test_depth_outside_the_valid_range_adds_nothing(self):
        cdf = _gaussian_cdf(0.9)
        tau = np.array([[20.4, 0.0, -3.0, np.inf, np.nan, 1e12]])
        spread = _zone(tau, pulse_cdf=cdf)
        only = _zone(tau[:, :1], pulse_cdf=cdf)
        torch.testing.assert_close(spread * tau.size, only, rtol=1e-12, atol=1e-15)

    def test_gradients_reach_the_irradiance_and_the_range(self):
        sigma = 1.1
        depth = torch.tensor([[20.3, 41.7]], dtype=torch.float64) * BW_M
        depth.requires_grad_(True)
        irradiance = torch.tensor([[1.0, 0.5]], dtype=torch.float64, requires_grad=True)
        rows, _ = Histogrammer(HistConfig()).calculate_transients(
            irradiance,
            depth,
            torch.zeros_like(irradiance),
            torch.ones(1, 1, 2),
            N_BINS,
            PERIOD_M,
            bin_width=BW_M,
            pulse_cdf=_gaussian_cdf(sigma),
        )
        centroid = (rows[0] * torch.arange(N_BINS)).sum() / rows[0].sum()
        centroid.backward()
        assert torch.all(torch.isfinite(irradiance.grad)) and torch.all(irradiance.grad != 0)
        # The centroid moves with each return's range, by its share of the zone's photons
        np.testing.assert_allclose(depth.grad.numpy() * BW_M, [[2 / 3, 1 / 3]], rtol=1e-6)

    @pytest.mark.parametrize(
        "pulse_cdf",
        [
            ([0.0, 1.0], [0.0, 0.5, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([0.0, 1.0, 2.0], [0.0, 0.7, 0.6]),
            ([0.0, 1.0], [0.1, 1.0]),
            ([0.0, 1.0], [0.0, 2.0]),
            ([0.0], [1.0]),
        ],
        ids=["lengths", "x-decreasing", "cdf-decreasing", "not-from-0", "not-to-1", "one-point"],
    )
    def test_invalid_pulse_cdf_raises(self, pulse_cdf):
        with pytest.raises(ValueError, match="pulse_cdf"):
            _zone(np.full((1, 1), 20.5), pulse_cdf=pulse_cdf)


# =========================================================================== #
# Detector.get_arrival_rates                                                  #
# =========================================================================== #
RES, BIN_M, N_BINS_EMU = 4, 0.01412, 128
F_WINDOW = C_LIGHT / (2 * N_BINS_EMU * BIN_M)  # one period is the window


def _config(pulse_shape="gaussian", pulse_width_ps=230.0, f_rep=F_WINDOW, **laser):
    """One zone over a RES x RES sensor without falloff; 128 bins of 14.12 mm (94.2 ps)."""
    return {
        "histogrammer": {
            "n_bins": N_BINS_EMU,
            "bin_width": BIN_M * ureg.meter,
            "n_pulses": 1000,
            "dead_time": (1 / f_rep) * ureg.second,
            "free_running": False,
            "fast_sim": True,
            "vignette": False,
            "pixel_fov_list": [[0.0, 1.0, 0.0, 1.0]],
        },
        "active_source": {
            "pulsed_laser": {
                "pulse_repetition": f_rep * ureg.hertz,
                "pulse_width": pulse_width_ps * ureg.picosecond,
                "avg_watts": 1e-3 * ureg.watt,
                "pulse_shape": pulse_shape,
                **laser,
            }
        },
        "ambient_source": {"sun": {"enabled": False}},
        "sensor": {
            "size": (RES, RES),
            "pixel_pitch": 10 * ureg.micrometer,
            "f_number": 1.4,
            "fov": [20 * ureg.degree, 20 * ureg.degree],
            "falloff": "none",
        },
    }


def _emulate(emulator, rng, ambient=None):
    """Transient and arrival rates of the zone for a flat frame at range rng (meters), or an (RES, RES) range map."""
    depth = torch.as_tensor(np.broadcast_to(rng, (RES, RES)).copy(), dtype=torch.float64)
    emulator.detector.get_transients(
        depth_frame=depth * ureg.meter,
        albedo_frame=torch.full((RES, RES), 0.5, dtype=torch.float64),
        emitter=emulator.emitter,
        scene=None,
        ambient_frame=ambient,
        normal_frame=torch.tensor([0.0, 0.0, 1.0]).expand(RES, RES, 3),
    )
    rates = emulator.detector.get_arrival_rates(emulator.emitter)
    return emulator.detector.transients[0].double().numpy(), rates[0].double().numpy()


class TestArrivalRates:
    @pytest.mark.parametrize("phase", [0.001, 0.25, 0.5, 0.75, 0.999])
    def test_arrival_rates_keep_the_position_within_the_bin(self, phase):
        emulator = ASPCEmulator(_config())
        transient, rates = _emulate(emulator, (40 + phase) * BIN_M)
        assert np.flatnonzero(transient).tolist() == [40]
        sigma = 230e-12 * C_LIGHT / 2 / FWHM / BIN_M
        expected = transient.sum() * _gaussian_shares(40 + phase, sigma, N_BINS_EMU)
        np.testing.assert_allclose(rates, expected, rtol=0, atol=1e-4 * transient.sum())
        assert _centroid(rates) == pytest.approx(40 + phase - 0.5, abs=1e-4)

    def test_returns_in_one_bin_no_longer_give_the_same_histogram(self):
        emulator = ASPCEmulator(_config())
        early = _emulate(emulator, 40.1 * BIN_M)[1]
        late = _emulate(emulator, 40.9 * BIN_M)[1]
        assert _centroid(late) - _centroid(early) == pytest.approx(0.8, abs=1e-4)

    def test_transients_stay_binned(self):
        """Guard: the transients hold the returns without a pulse, in the bin a TDC records them in."""
        emulator = ASPCEmulator(_config())
        for phase in (0.0, 0.5, 0.999):
            transient, _ = _emulate(emulator, (40 + phase) * BIN_M)
            assert np.flatnonzero(transient).tolist() == [40]

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_zero_width_pulse_gives_the_transient(self, shape):
        emulator = ASPCEmulator(_config(shape, pulse_width_ps=0.0))
        rng = np.linspace(0.3, 1.5, RES * RES).reshape(RES, RES)
        transient, rates = _emulate(emulator, rng)
        np.testing.assert_allclose(rates, transient, rtol=1e-6, atol=0)

    def test_square_pulse_is_exact(self):
        width = 2.3  # bins
        emulator = ASPCEmulator(_config("square", pulse_width_ps=width * 2 * BIN_M / C_LIGHT * 1e12))
        transient, rates = _emulate(emulator, 40.3 * BIN_M)
        expected = transient.sum() * _uniform_shares(40.3, -width / 2, width / 2, N_BINS_EMU)
        np.testing.assert_allclose(rates, expected, rtol=1e-5, atol=1e-7 * transient.sum())

    def test_custom_taps_at_the_center_of_a_bin(self):
        """Guard: a return at the center of a bin still gives the taps, centered on its bin."""
        emulator = ASPCEmulator(_config("custom", pulse_shape_custom=TAPS))
        transient, rates = _emulate(emulator, 40.5 * BIN_M)
        expected = np.zeros(N_BINS_EMU)
        start = 40 - len(TAPS) // 2
        expected[start : start + len(TAPS)] = transient.sum() * np.array(TAPS) / np.sum(TAPS)
        np.testing.assert_allclose(rates, expected, rtol=1e-5, atol=1e-7 * transient.sum())

    @pytest.mark.parametrize("phase", [0.001, 0.25, 0.75, 0.999])
    def test_custom_taps_of_a_gaussian_keep_its_peak(self, phase):
        """Taps that are a Gaussian's bin integrals (2.44 bins wide) lost up to 8% of its peak away from a bin center."""
        sigma = 230e-12 * C_LIGHT / 2 / FWHM / BIN_M
        emulator = ASPCEmulator(_config("custom", pulse_shape_custom=_gaussian_taps(sigma)))
        transient, rates = _emulate(emulator, (40 + phase) * BIN_M)
        expected = transient.sum() * _gaussian_shares(40 + phase, sigma, N_BINS_EMU)
        np.testing.assert_allclose(rates, expected, rtol=0, atol=0.01 * expected.max())

    def test_ambient_offset_is_added_to_every_bin(self):
        """Guard: the ambient offset is still added on top of the pulse."""
        emulator = ASPCEmulator(_config())
        _, without = _emulate(emulator, 40.3 * BIN_M)
        ambient = torch.full((RES, RES), 1e9, dtype=torch.float64)  # about a tenth of the peak
        _, with_ambient = _emulate(emulator, 40.3 * BIN_M, ambient=ambient)
        offset = float(emulator.detector.ambient_offsets[0])
        assert offset > 0
        np.testing.assert_allclose(with_ambient - without, offset, rtol=1e-5)

    def test_arrival_rates_follow_the_last_frame(self):
        emulator = ASPCEmulator(_config())
        _emulate(emulator, 20.2 * BIN_M)
        _, rates = _emulate(emulator, 70.7 * BIN_M)
        assert _centroid(rates) == pytest.approx(70.2, abs=1e-4)

    def test_arrival_rates_need_the_transients_first(self):
        """Guard: the arrival rates still come from the last transients."""
        emulator = ASPCEmulator(_config())
        with pytest.raises(RuntimeError, match="get_transients"):
            emulator.detector.get_arrival_rates(emulator.emitter)

    def test_replaced_transients_raise(self):
        """The arrival rates come from the returns of get_transients, so replaced transients would be ignored."""
        emulator = ASPCEmulator(_config())
        _emulate(emulator, 40.3 * BIN_M)
        emulator.detector.transients = torch.zeros_like(emulator.detector.transients)
        with pytest.raises(RuntimeError, match="replaced"):
            emulator.detector.get_arrival_rates(emulator.emitter)
        # The next frame's transients are used again
        _, rates = _emulate(emulator, 70.7 * BIN_M)
        assert _centroid(rates) == pytest.approx(70.2, abs=1e-4)

    def test_photon_counts_follow_the_arrival_rates(self):
        """The photon histogram has the centroid of the arrival rates, not of the bin's start."""
        emulator = ASPCEmulator(_config(avg_watts=1e-5 * ureg.watt))  # little pile-up
        depth = torch.full((RES, RES), 40.95 * BIN_M, dtype=torch.float64) * ureg.meter
        normals = torch.tensor([0.0, 0.0, 1.0]).expand(RES, RES, 3)
        ewh = emulator.process_frame(depth, torch.ones(RES, RES), normal_frame=normals)
        assert _centroid(ewh[0]) == pytest.approx(40.45, abs=1e-3)


class TestCalculateArrivalRates:
    def test_without_irf_only_the_ambient_is_added(self):
        transients = torch.rand(3, N_BINS, dtype=torch.float64)
        offsets = [torch.tensor(0.1), torch.tensor(0.0), torch.tensor(2.5)]
        rates = Histogrammer(HistConfig()).calculate_arrival_rates(None, transients, offsets, N_BINS)
        expected = transients + torch.tensor([[0.1], [0.0], [2.5]], dtype=torch.float64)
        torch.testing.assert_close(rates, expected.float())

    def test_with_irf_the_transient_is_still_convolved(self):
        """Guard: a binned transient and a kernel sampled at whole bins still give their convolution."""
        transients = torch.zeros(1, N_BINS)
        transients[0, 20] = 1.0
        irf = torch.tensor([0.25, 0.5, 0.25])
        rates = Histogrammer(HistConfig()).calculate_arrival_rates(irf, transients, 0.0, N_BINS)
        torch.testing.assert_close(rates[0, 19:22], irf)
