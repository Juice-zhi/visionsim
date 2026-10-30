"""Fast path of the EWH: expected detections per bin and cycle, the optional photon noise, and the emulator wiring."""

import pytest
import torch

from visionsim.emulate.aspc.emulator import ASPCEmulator
from visionsim.emulate.aspc.histogrammers import (
    HistConfig,
    Histogrammer,
    batch_distorted_transient_async,
    batch_distorted_transient_sync,
    batch_expected_detections_async,
    batch_expected_detections_sync,
    batch_sample_detections,
)
from visionsim.emulate.aspc.utils import c, ureg

B = 12


def _phi(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) + 0.1


def _first_photon_reference(phi):
    """Probability that a synchronous cycle records its first photon in each bin, in float64."""
    phi = phi.double()
    return -torch.expm1(-phi) * torch.exp(-(torch.cumsum(phi, dim=-1) - phi))


def _free_running_reference(phi, dead_time_bins):
    """Detections per bin and cycle of a free-running SPAD with a non-paralyzable dead time, in float64.

    Bin i records a photon with probability p_i = 1 - exp(-phi_i) when the SPAD is armed, i.e. when none of the
    dead_time_bins - 1 bins before it recorded one. At most one of them can, so r_i = p_i * (1 - sum of their r), a
    linear system in r.
    """
    phi = phi.double()
    n = phi.shape[-1]
    p = -torch.expm1(-phi)
    dead = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for g in range(i - dead_time_bins + 1, i):
            dead[i, g % n] += 1.0
    return torch.linalg.solve(torch.eye(n, dtype=torch.float64) + p.unsqueeze(-1) * dead, p)


# =========================================================================== #
# Expected detections per bin and laser cycle                                 #
# =========================================================================== #
class TestExpectedDetectionsSync:
    def test_is_the_first_photon_probability(self):
        phi = _phi((3, B), seed=1) * torch.tensor([[1.0], [0.01], [0.2]])
        out = batch_expected_detections_sync(phi, 0, B)
        assert torch.allclose(out.double(), _first_photon_reference(phi), rtol=1e-6, atol=0)

    def test_rows_sum_to_the_probability_of_a_detection(self):
        phi = _phi((3, B), seed=1) * torch.tensor([[1.0], [0.01], [0.2]])
        out = batch_expected_detections_sync(phi, 0, B)
        expected = -torch.expm1(-phi.double().sum(-1))
        assert torch.allclose(out.double().sum(-1), expected, rtol=1e-6)
        assert torch.all(out.sum(-1) < 1.0)

    def test_shape_is_the_normalised_model(self):
        phi = _phi((3, B), seed=1)
        out = batch_expected_detections_sync(phi, 0, B)
        assert torch.allclose(out / out.sum(-1, keepdim=True), batch_distorted_transient_sync(phi, 0, B), atol=1e-6)

    @pytest.mark.parametrize("scale", [1e-4, 1e-6, 1e-9])
    def test_low_flux(self, scale):
        phi = _phi((2, B), seed=1) * scale
        out = batch_expected_detections_sync(phi, 0, B)
        assert torch.allclose(out.double(), _first_photon_reference(phi), rtol=1e-6, atol=0)

    def test_dead_time_is_not_used(self):
        phi = _phi((2, B), seed=1)
        assert torch.equal(batch_expected_detections_sync(phi, 0, B), batch_expected_detections_sync(phi, 5, B))


class TestExpectedDetectionsFreeRunning:
    def test_no_dead_time_counts_every_photon(self):
        phi = _phi((3, B), seed=1) * torch.tensor([[1.0], [0.01], [3.0]])
        assert torch.equal(batch_expected_detections_async(phi, 0, B), phi)

    @pytest.mark.parametrize("dt", [1, 2, 3, 7, B, B + 5, 3 * B + 1])
    def test_matches_the_dead_time_reference(self, dt):
        phi = _phi((3, B), seed=2) * torch.tensor([[1.0], [0.05], [0.3]])
        out = batch_expected_detections_async(phi, dt, B)
        reference = torch.stack([_free_running_reference(row, dt) for row in phi])
        assert torch.allclose(out.double(), reference, rtol=1e-5, atol=0)

    @pytest.mark.parametrize("dt", [1, B, 3 * B + 1])
    def test_every_bin_bright_needs_more_iterations(self, dt):
        # The SPAD nearly always fires in the bin where it re-arms, so the chain mixes slowly
        phi = _phi((B,), seed=2) * 3.0
        out = batch_expected_detections_async(phi, dt, B, n_iterations=1000)
        assert torch.allclose(out[0].double(), _free_running_reference(phi, dt), rtol=1e-5, atol=0)

    @pytest.mark.parametrize("lam", [0.01, 0.2, 2.0])
    @pytest.mark.parametrize("tau", [1, 3, 10])
    def test_flat_flux_matches_the_renewal_rate(self, lam, tau):
        p = 1.0 - torch.exp(torch.tensor(-lam, dtype=torch.float64))
        out = batch_expected_detections_async(torch.full((1, B), lam), tau, B)
        expected = torch.full((1, B), float(1.0 / (tau + (1.0 - p) / p)), dtype=torch.float64)
        assert torch.allclose(out.double(), expected, rtol=1e-5)

    def test_dead_time_reduces_detections(self):
        phi = _phi((1, B), seed=2)
        totals = [float(batch_expected_detections_async(phi, dt, B).sum()) for dt in (0, 1, 3, B, 2 * B)]
        assert totals == sorted(totals, reverse=True)
        assert totals[-1] < 0.5

    def test_shape_is_the_normalised_model(self):
        phi = _phi((3, B), seed=2)
        out = batch_expected_detections_async(phi, 3, B)
        assert torch.allclose(out / out.sum(-1, keepdim=True), batch_distorted_transient_async(phi, 3, B), atol=1e-6)

    @pytest.mark.parametrize("dt", [3, B + 5])
    def test_low_flux(self, dt):
        phi = _phi((2, B), seed=2) * 1e-6
        out = batch_expected_detections_async(phi, dt, B)
        reference = torch.stack([_free_running_reference(row, dt) for row in phi])
        assert torch.allclose(out.double(), reference, rtol=1e-5, atol=0)

    def test_accepts_a_single_pixel(self):
        phi = _phi((B,), seed=2)
        assert torch.equal(batch_expected_detections_async(phi, 3, B), batch_expected_detections_async(phi[None], 3, B))


EXPECTED_MODELS = [
    pytest.param(batch_expected_detections_sync, 0, id="gated"),
    pytest.param(batch_expected_detections_async, 0, id="free-running-dt0"),
    pytest.param(batch_expected_detections_async, 3, id="free-running-dt3"),
    pytest.param(batch_expected_detections_async, B + 5, id="free-running-dt-over-a-period"),
]


class TestExpectedDetectionsContract:
    @pytest.mark.parametrize("model,dt", EXPECTED_MODELS)
    def test_zero_flux_rows(self, model, dt):
        phi = _phi((3, B), seed=3)
        phi[1] = 0.0
        out = model(phi, dt, B)
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[1]) == 0
        assert torch.all(out[0] > 0) and torch.all(out[2] > 0)

    @pytest.mark.parametrize("model,dt", EXPECTED_MODELS)
    def test_negative_rates_are_clamped(self, model, dt):
        phi = _phi((1, B), seed=3)
        phi[0, 2] = -0.05
        out = model(phi, dt, B)
        assert float(out[0, 2]) == 0.0
        assert torch.all(out >= 0)

    @pytest.mark.parametrize("model,dt", EXPECTED_MODELS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_keeps_input_dtype(self, model, dt, dtype):
        assert model(_phi((2, B)).to(dtype), dt, B).dtype == dtype

    @pytest.mark.parametrize("model,dt", EXPECTED_MODELS)
    def test_gradient_matches_finite_differences(self, model, dt):
        # Weighting the bins makes the result depend on both the shape and the total
        base = _phi((1, B), seed=3).double() * 0.3
        weights = torch.arange(1, B + 1, dtype=torch.float64)
        phi = base.clone().requires_grad_(True)
        (model(phi, dt, B) * weights).sum().backward()

        def f(p):
            with torch.no_grad():
                return float((model(p, dt, B) * weights).sum())

        eps = 1e-6
        for i in (0, 3, B - 1):
            up, dn = base.clone(), base.clone()
            up[0, i] += eps
            dn[0, i] -= eps
            fd = (f(up) - f(dn)) / (2 * eps)
            assert float(phi.grad[0, i]) == pytest.approx(fd, rel=1e-5, abs=1e-8), f"bin {i}"


# =========================================================================== #
# Photon noise                                                                #
# =========================================================================== #
NOISE_DRAWS = 4000
NOISE_MODES = [
    pytest.param(False, 0, id="gated"),
    pytest.param(True, 0, id="free-running-dt0"),
    pytest.param(True, 3, id="free-running-dt3"),
]


def _draws(phi, n_pulses, free_running, dt, seed=0):
    """NOISE_DRAWS noisy histograms of the same arrival rates, one per row."""
    rates = phi.expand(NOISE_DRAWS, -1).contiguous()
    generator = torch.Generator().manual_seed(seed)
    return batch_sample_detections(rates, n_pulses, phi.shape[-1], free_running, dt, generator=generator).double()


def _assert_moments(counts, mean, var):
    """The mean and variance over the rows of counts agree with the given ones, bin by bin."""
    k = counts.shape[0]
    assert torch.all((counts.mean(0) - mean).abs() <= 5 * torch.sqrt(var / k)), (counts.mean(0), mean)
    assert torch.allclose(counts.var(0), var, rtol=0.15), (counts.var(0), var)


class TestSampleDetections:
    @pytest.mark.parametrize("free_running,dt", NOISE_MODES)
    def test_counts_are_whole_numbers_and_reproducible(self, free_running, dt):
        phi = _phi((3, B), seed=7)
        a, b, c = (
            batch_sample_detections(phi, 500, B, free_running, dt, generator=torch.Generator().manual_seed(s))
            for s in (0, 0, 1)
        )
        assert a.dtype == phi.dtype
        assert torch.all(a >= 0) and torch.equal(a, a.round())
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    @pytest.mark.parametrize("free_running,dt", NOISE_MODES)
    def test_zero_flux_gives_no_counts(self, free_running, dt):
        assert torch.count_nonzero(batch_sample_detections(torch.zeros(2, B), 500, B, free_running, dt)) == 0

    def test_gated_is_multinomial(self):
        phi = _phi((B,), seed=2) * 0.3
        n_pulses = 200
        counts = _draws(phi, n_pulses, False, 0)
        p = _first_photon_reference(phi)
        assert torch.all(counts.sum(-1) <= n_pulses)
        _assert_moments(counts, n_pulses * p, n_pulses * p * (1 - p))
        p_any = p.sum(-1, keepdim=True)
        _assert_moments(counts.sum(-1, keepdim=True), n_pulses * p_any, n_pulses * p_any * (1 - p_any))

    def test_free_running_without_dead_time_is_poisson(self):
        phi = _phi((B,), seed=2)
        counts = _draws(phi, 50, True, 0)
        _assert_moments(counts, 50 * phi.double(), 50 * phi.double())

    def test_free_running_with_dead_time_is_binomial_per_bin(self):
        phi = _phi((B,), seed=2)
        n_pulses = 200
        counts = _draws(phi, n_pulses, True, 3)
        r = _free_running_reference(phi, 3)
        assert torch.all(counts <= n_pulses)
        _assert_moments(counts, n_pulses * r, n_pulses * r * (1 - r))


class TestSimulateEwhNoise:
    def test_default_is_the_noise_free_expectation(self):
        assert HistConfig().fast_sim_noise is False
        h = Histogrammer(HistConfig())
        rates = _phi((2, B), seed=7) * 0.1
        first = h.simulate_ewh(rates, 500, B, free_running=False, fast_sim=True)
        second = h.simulate_ewh(rates, 500, B, free_running=False, fast_sim=True)
        expected = batch_expected_detections_sync(rates, 0, B) * 500
        for a, b, e in zip(first, second, expected):
            assert torch.equal(a, b) and torch.equal(a, e)
        assert not torch.equal(first[0], first[0].round())

    @pytest.mark.parametrize("free_running,dt", NOISE_MODES)
    def test_noise_option_draws_the_counts(self, free_running, dt):
        h = Histogrammer(HistConfig())
        rates = _phi((3, B), seed=7)
        noisy = h.simulate_ewh(
            rates,
            500,
            B,
            free_running=free_running,
            dead_time_bins=dt,
            fast_sim=True,
            fast_sim_noise=True,
            generator=torch.Generator().manual_seed(0),
        )
        drawn = batch_sample_detections(rates, 500, B, free_running, dt, generator=torch.Generator().manual_seed(0))
        assert torch.equal(torch.stack(noisy), drawn)

    def test_noise_option_does_not_affect_the_monte_carlo(self):
        h = Histogrammer(HistConfig())
        rates = _phi((2, B), seed=7)
        kwargs = dict(free_running=True, dead_time_bins=3, fast_sim=False)
        quiet = h.simulate_ewh(rates, 500, B, generator=torch.Generator().manual_seed(0), **kwargs)
        noisy = h.simulate_ewh(rates, 500, B, fast_sim_noise=True, generator=torch.Generator().manual_seed(0), **kwargs)
        assert all(torch.equal(a, b) for a, b in zip(quiet, noisy))


# =========================================================================== #
# Emulator                                                                    #
# =========================================================================== #
EMU_RES, EMU_BINS, EMU_BIN_M, EMU_PULSES = 8, 64, 0.01, 10000


def _emulator(free_running, dead_time_bins, avg_watts=0.01, fast_sim_noise=False):
    """A single zone over an 8 x 8 sensor, with 64 bins of 1 cm spanning one laser period."""
    rep_hz = c / (2 * EMU_BINS * EMU_BIN_M)
    hist = HistConfig(
        min_depth=0 * ureg.meter,
        max_depth=EMU_BINS * EMU_BIN_M * ureg.meter,
        n_bins=EMU_BINS,
        bin_width=EMU_BIN_M * ureg.meter,
        n_pulses=EMU_PULSES,
        dead_time=dead_time_bins / (EMU_BINS * rep_hz) * ureg.second,
        free_running=free_running,
        fast_sim=True,
        fast_sim_noise=fast_sim_noise,
    )
    config = {
        "histogrammer": hist.__dict__,
        "active_source": {
            "pulsed_laser": {
                "pulse_repetition": rep_hz * ureg.hertz,
                "pulse_width": 100 * ureg.picosecond,
                "avg_watts": avg_watts * ureg.watt,
            }
        },
        "ambient_source": {"sun": {"enabled": False}},
        "sensor": {
            "size": (EMU_RES, EMU_RES),
            "pixel_pitch": 10 * ureg.micrometer,
            "f_number": 1.4,
            "fov": (20 * ureg.degree, 20 * ureg.degree),
        },
    }
    return ASPCEmulator(config)


def _wall(albedo):
    depth = torch.full((EMU_RES, EMU_RES), 0.3) * ureg.meter
    return depth, torch.full((EMU_RES, EMU_RES), albedo)


class TestEmulatorFastSim:
    @pytest.mark.parametrize("albedo", [1.0, 0.01])
    def test_gated_total_follows_the_flux(self, albedo):
        emulator = _emulator(free_running=False, dead_time_bins=EMU_BINS)
        hist = emulator.process_frame(*_wall(albedo))[0]
        lam = emulator.detector.arrival_rates[0].double().sum()
        assert float(lam) < 0.5
        assert float(hist.sum()) == pytest.approx(EMU_PULSES * float(-torch.expm1(-lam)), rel=1e-5)

    @pytest.mark.parametrize("dt", [0, 16])
    def test_free_running_total_follows_the_flux(self, dt):
        emulator = _emulator(free_running=True, dead_time_bins=dt, avg_watts=0.1)
        hist = emulator.process_frame(*_wall(1.0))[0]
        rates = emulator.detector.arrival_rates[0]
        expected = _free_running_reference(rates, dt) if dt else rates.double()
        assert float(hist.sum()) == pytest.approx(EMU_PULSES * float(expected.sum()), rel=1e-5)

    def test_noise_option_reaches_the_histogrammer(self):
        emulator = _emulator(free_running=False, dead_time_bins=EMU_BINS, fast_sim_noise=True)
        assert emulator.detector.histogrammer.fast_sim_noise is True
        hist = emulator.process_frame(*_wall(1.0))[0]
        assert (hist == hist.round()).all()
        assert 0 < hist.sum() < EMU_PULSES
