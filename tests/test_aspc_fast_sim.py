"""Fast path of the EWH: expected detections per bin and cycle, the optional photon noise, and the emulator wiring."""

import itertools
import math
from functools import partial

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


def _multi_hit_reference(phi, dead_time_bins, max_detections):
    """Detections per bin and cycle of a synchronous SPAD that records up to max_detections photons (None: no cap)
    with a dead time of at least one bin, in float64.

    With a dead time a bin records at most one photon, so only which bins hold a photon matters. This walks the SPAD
    over every such pattern and weights its detections by the probability of the pattern.
    """
    p = (-torch.expm1(-phi.double())).tolist()
    out = [0.0] * len(p)
    for pattern in itertools.product((False, True), repeat=len(p)):
        weight = math.prod(p_i if hit else 1.0 - p_i for p_i, hit in zip(p, pattern))
        armed, count = 0, 0
        for i, hit in enumerate(pattern):
            if hit and i >= armed and (max_detections is None or count < max_detections):
                out[i] += weight
                count += 1
                armed = i + dead_time_bins
    return torch.tensor(out, dtype=torch.float64)


def _capped_reference(phi, max_detections, n_terms=80):
    """Detections per bin and cycle of a synchronous SPAD without dead time that records the first max_detections
    photons of each cycle, in float64.

    Bin i records min(N, max(max_detections - C, 0)) photons, where N ~ Poisson(phi_i) and C ~ Poisson(sum of the
    earlier phi) are independent.
    """
    phi = phi.double().tolist()

    def pmf(lam):
        return [math.exp(-lam) * lam**n / math.factorial(n) for n in range(n_terms)]

    out, earlier = [], 0.0
    for lam in phi:
        before, here = pmf(earlier), pmf(lam)
        out.append(
            sum(before[c] * here[n] * min(n, max(max_detections - c, 0)) for c in range(n_terms) for n in range(n_terms))
        )
        earlier += lam
    return torch.tensor(out, dtype=torch.float64)


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


class TestExpectedDetectionsSyncMultiHit:
    """Synchronous with more than one detection per cycle: the SPAD re-arms after each detection and its dead time."""

    @pytest.mark.parametrize("cap", [None, 2, 3])
    @pytest.mark.parametrize("dt", [1, 2, 3, 5, B - 1])
    def test_matches_the_walk_over_every_pattern(self, dt, cap):
        phi = _phi((2, B), seed=4) * torch.tensor([[1.0], [0.1]])
        out = batch_expected_detections_sync(phi, dt, B, cap)
        reference = torch.stack([_multi_hit_reference(row, dt, cap) for row in phi])
        assert torch.allclose(out.double(), reference, rtol=1e-6, atol=0)

    def test_records_more_than_one_photon_per_cycle(self):
        phi = _phi((1, B), seed=4)
        single, multi = (batch_expected_detections_sync(phi, 2, B, cap).sum() for cap in (1, None))
        assert float(single) < 1.0 < float(multi)

    def test_no_dead_time_and_no_cap_counts_every_photon(self):
        phi = _phi((3, B), seed=1) * torch.tensor([[1.0], [0.01], [3.0]])
        assert torch.equal(batch_expected_detections_sync(phi, 0, B, None), phi)

    @pytest.mark.parametrize("cap", [2, 3, 6])
    def test_no_dead_time_records_the_first_photons(self, cap):
        phi = _phi((2, B), seed=4).double() * torch.tensor([[1.0], [0.1]], dtype=torch.float64)
        out = batch_expected_detections_sync(phi, 0, B, cap)
        reference = torch.stack([_capped_reference(row, cap) for row in phi])
        assert torch.allclose(out, reference, rtol=1e-9, atol=1e-15)
        lam = phi.sum(-1)
        assert torch.all(out.sum(-1) < torch.minimum(lam, torch.full_like(lam, cap)))

    @pytest.mark.parametrize("cap", [None, 2])
    @pytest.mark.parametrize("dt", [B, B + 5])
    def test_dead_time_over_the_cycle_records_the_first_photon(self, dt, cap):
        phi = _phi((2, B), seed=4)
        assert torch.equal(batch_expected_detections_sync(phi, dt, B, cap), batch_expected_detections_sync(phi, 0, B))

    @pytest.mark.parametrize("dt", [1, 3, 5])
    def test_a_cap_the_dead_time_cannot_reach_is_no_cap(self, dt):
        phi = _phi((2, B), seed=4).double()
        most = -(-B // dt)
        no_cap = batch_expected_detections_sync(phi, dt, B, None)
        assert torch.equal(batch_expected_detections_sync(phi, dt, B, most), no_cap)
        assert float((no_cap - batch_expected_detections_sync(phi, dt, B, most - 1)).sum()) > 0

    def test_totals_grow_with_the_cap_and_fall_with_the_dead_time(self):
        phi = _phi((1, B), seed=4)
        by_cap = [float(batch_expected_detections_sync(phi, 2, B, cap).sum()) for cap in (1, 2, 3, None)]
        by_dead_time = [float(batch_expected_detections_sync(phi, dt, B, None).sum()) for dt in (0, 1, 2, 5, B)]
        assert by_cap == sorted(by_cap) and len(set(by_cap)) == len(by_cap)
        assert by_dead_time == sorted(by_dead_time, reverse=True) and len(set(by_dead_time)) == len(by_dead_time)

    @pytest.mark.parametrize("scale", [1e-6, 1e-9])
    @pytest.mark.parametrize("dt,cap", [(3, None), (3, 2), (0, 2)])
    def test_low_flux(self, scale, dt, cap):
        phi = _phi((1, B), seed=4) * scale
        out = batch_expected_detections_sync(phi, dt, B, cap)[0].double()
        reference = _capped_reference(phi[0], cap) if dt == 0 else _multi_hit_reference(phi[0], dt, cap)
        assert torch.allclose(out, reference, rtol=1e-6, atol=0)

    @pytest.mark.parametrize("cap", [0, -1, 1.5, "2"])
    def test_rejects_an_invalid_cap(self, cap):
        with pytest.raises(ValueError, match="max_detections_per_cycle"):
            batch_expected_detections_sync(_phi((1, B)), 3, B, cap)


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
    pytest.param(partial(batch_expected_detections_sync, max_detections_per_cycle=None), 0, id="gated-no-cap-dt0"),
    pytest.param(partial(batch_expected_detections_sync, max_detections_per_cycle=None), 3, id="gated-no-cap-dt3"),
    pytest.param(partial(batch_expected_detections_sync, max_detections_per_cycle=2), 0, id="gated-cap-2-dt0"),
    pytest.param(partial(batch_expected_detections_sync, max_detections_per_cycle=2), 3, id="gated-cap-2-dt3"),
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
    pytest.param(False, 0, 1, id="gated"),
    pytest.param(False, 0, None, id="gated-no-cap-dt0"),
    pytest.param(False, 0, 2, id="gated-cap-2-dt0"),
    pytest.param(False, 3, None, id="gated-no-cap-dt3"),
    pytest.param(True, 0, 1, id="free-running-dt0"),
    pytest.param(True, 3, 1, id="free-running-dt3"),
]


def _draws(phi, n_pulses, free_running, dt, seed=0, cap=1):
    """NOISE_DRAWS noisy histograms of the same arrival rates, one per row."""
    rates = phi.expand(NOISE_DRAWS, -1).contiguous()
    generator = torch.Generator().manual_seed(seed)
    return batch_sample_detections(
        rates, n_pulses, phi.shape[-1], free_running, dt, generator=generator, max_detections_per_cycle=cap
    ).double()


def _assert_moments(counts, mean, var):
    """The mean and variance over the rows of counts agree with the given ones, bin by bin."""
    k = counts.shape[0]
    assert torch.all((counts.mean(0) - mean).abs() <= 5 * torch.sqrt(var / k)), (counts.mean(0), mean)
    assert torch.allclose(counts.var(0), var, rtol=0.15), (counts.var(0), var)


class TestSampleDetections:
    @pytest.mark.parametrize("free_running,dt,cap", NOISE_MODES)
    def test_counts_are_whole_numbers_and_reproducible(self, free_running, dt, cap):
        phi = _phi((3, B), seed=7)
        a, b, c = (
            batch_sample_detections(
                phi,
                500,
                B,
                free_running,
                dt,
                generator=torch.Generator().manual_seed(s),
                max_detections_per_cycle=cap,
            )
            for s in (0, 0, 1)
        )
        assert a.dtype == phi.dtype
        assert torch.all(a >= 0) and torch.equal(a, a.round())
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    @pytest.mark.parametrize("free_running,dt,cap", NOISE_MODES)
    def test_zero_flux_gives_no_counts(self, free_running, dt, cap):
        counts = batch_sample_detections(torch.zeros(2, B), 500, B, free_running, dt, max_detections_per_cycle=cap)
        assert torch.count_nonzero(counts) == 0

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

    def test_gated_multi_hit_with_dead_time_is_binomial_per_bin(self):
        phi = _phi((B,), seed=2)
        n_pulses = 200
        counts = _draws(phi, n_pulses, False, 3, cap=None)
        r = _multi_hit_reference(phi, 3, None)
        assert torch.all(counts <= n_pulses)
        assert float(counts.sum(-1).mean()) > n_pulses
        _assert_moments(counts, n_pulses * r, n_pulses * r * (1 - r))

    def test_gated_without_dead_time_or_cap_is_poisson(self):
        phi = _phi((B,), seed=2)
        counts = _draws(phi, 50, False, 0, cap=None)
        _assert_moments(counts, 50 * phi.double(), 50 * phi.double())

    def test_gated_cap_without_dead_time_is_drawn_around_the_capped_mean(self):
        phi = _phi((B,), seed=2)
        counts = _draws(phi, 50, False, 0, cap=2)
        mean = 50 * _capped_reference(phi, 2)
        assert torch.all((counts.mean(0) - mean).abs() <= 5 * torch.sqrt(mean / NOISE_DRAWS)), (counts.mean(0), mean)

    @pytest.mark.parametrize("cap", [None, 3])
    def test_gated_dead_time_over_the_cycle_is_the_single_hit_draw(self, cap):
        phi = _phi((3, B), seed=7)
        a, b = (
            batch_sample_detections(
                phi, 500, B, False, B, generator=torch.Generator().manual_seed(0), max_detections_per_cycle=c
            )
            for c in (cap, 1)
        )
        assert torch.equal(a, b)
        assert torch.all(a.sum(-1) <= 500)

    @pytest.mark.parametrize("cap", [0, -1, 2.5])
    def test_gated_rejects_an_invalid_cap(self, cap):
        with pytest.raises(ValueError, match="max_detections_per_cycle"):
            batch_sample_detections(_phi((1, B)), 500, B, False, 3, max_detections_per_cycle=cap)


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

    @pytest.mark.parametrize("free_running,dt,cap", NOISE_MODES)
    def test_noise_option_draws_the_counts(self, free_running, dt, cap):
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
            max_detections_per_cycle=cap,
        )
        drawn = batch_sample_detections(
            rates, 500, B, free_running, dt, generator=torch.Generator().manual_seed(0), max_detections_per_cycle=cap
        )
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


def _emulator(free_running, dead_time_bins, avg_watts=0.01, **hist_kwargs):
    """A single zone over an 8 x 8 sensor, with 64 bins of 1 cm spanning one laser period. hist_kwargs override the
    fields of HistConfig; fast_sim is on by default."""
    rep_hz = c / (2 * EMU_BINS * EMU_BIN_M)
    hist = HistConfig(
        **{
            "min_depth": 0 * ureg.meter,
            "max_depth": EMU_BINS * EMU_BIN_M * ureg.meter,
            "n_bins": EMU_BINS,
            "bin_width": EMU_BIN_M * ureg.meter,
            "n_pulses": EMU_PULSES,
            "dead_time": dead_time_bins / (EMU_BINS * rep_hz) * ureg.second,
            "free_running": free_running,
            "fast_sim": True,
            **hist_kwargs,
        }
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

    @pytest.mark.parametrize(
        "hist_kwargs",
        [{}, {"max_detections_per_cycle": None}, {"max_detections_per_cycle": 2}],
        ids=["default", "no-cap", "cap-2"],
    )
    def test_gated_monte_carlo_matches_the_fast_path(self, hist_kwargs):
        # A bright return over a few bins and a dead time of one bin, so a multi-hit cycle often records several photons
        fast = _emulator(False, 1, avg_watts=0.1, **hist_kwargs)
        mc = _emulator(False, 1, avg_watts=0.1, fast_sim=False, **hist_kwargs)
        expected = torch.from_numpy(fast.process_frame(*_wall(1.0))[0]).double()
        torch.manual_seed(0)
        counts = torch.from_numpy(mc.process_frame(*_wall(1.0))[0]).double()
        assert float(counts.sum()) == pytest.approx(float(expected.sum()), abs=5 * float(expected.sum()) ** 0.5)
        assert torch.all((counts - expected).abs() <= 6 * torch.sqrt(expected) + 1)

    def test_gated_default_records_at_most_one_photon_per_cycle(self):
        emulator = _emulator(False, 1, avg_watts=0.1, fast_sim=False)
        assert emulator.detector.histogrammer.max_detections_per_cycle == 1
        torch.manual_seed(0)
        hist = emulator.process_frame(*_wall(1.0))[0]
        lam = emulator.detector.arrival_rates[0].double().sum()
        expected = EMU_PULSES * float(-torch.expm1(-lam))
        assert hist.sum() <= EMU_PULSES
        assert float(hist.sum()) == pytest.approx(expected, abs=5 * expected**0.5)

    def test_gated_without_cap_total_follows_the_dead_time(self):
        emulator = _emulator(False, 1, avg_watts=0.1, max_detections_per_cycle=None)
        hist = emulator.process_frame(*_wall(1.0))[0]
        rates = emulator.detector.arrival_rates[0]
        # The return covers bins 27 to 33; the reference walks every pattern of the bins around it
        assert float(rates[:24].sum() + rates[36:].sum()) < 1e-9
        expected = EMU_PULSES * float(_multi_hit_reference(rates[24:36], 1, None).sum())
        assert float(hist.sum()) == pytest.approx(expected, rel=1e-5)
        assert hist.sum() > EMU_PULSES

    def test_noise_option_reaches_the_histogrammer(self):
        emulator = _emulator(free_running=False, dead_time_bins=EMU_BINS, fast_sim_noise=True)
        assert emulator.detector.histogrammer.fast_sim_noise is True
        hist = emulator.process_frame(*_wall(1.0))[0]
        assert (hist == hist.round()).all()
        assert 0 < hist.sum() < EMU_PULSES
