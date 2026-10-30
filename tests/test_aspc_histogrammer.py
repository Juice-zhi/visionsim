"""Stage 3 — Histogrammer dead-time / pile-up forward models."""

import pytest
import torch

from visionsim.emulate.aspc.histogrammers import (
    HistConfig,
    Histogrammer,
    HistogrammerEDH,
    batch_distorted_transient_async,
    batch_distorted_transient_sync,
    calculate_distorted_transient,
    calculate_distorted_transient_sync,
)

B = 12


def _phi(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) + 0.1


def _com(x):
    idx = torch.arange(x.shape[-1], dtype=x.dtype)
    return float((idx * x).sum() / x.sum())


# =========================================================================== #
# 3a-1 — closed-form outputs are proper probability distributions               #
# =========================================================================== #
class TestNormalization:
    def test_batch_sync_rows_sum_to_one(self):
        out = batch_distorted_transient_sync(_phi((5, B)), 0, B)
        assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)

    def test_batch_async_deadtime_rows_sum_to_one(self):
        out = batch_distorted_transient_async(_phi((5, B)), 3, B)
        assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)

    def test_batch_async_no_deadtime_rows_sum_to_one(self):
        out = batch_distorted_transient_async(_phi((5, B)), 0, B)
        assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)

    def test_per_pixel_async_sums_to_one(self):
        out = calculate_distorted_transient(_phi((B,)), 3, B)
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)

    def test_per_pixel_sync_sums_to_one(self):
        out = calculate_distorted_transient_sync(_phi((B,)), B, B)
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)

    def test_distributions_are_non_negative(self):
        assert batch_distorted_transient_sync(_phi((5, B)), 0, B).min() >= -1e-8
        assert batch_distorted_transient_async(_phi((5, B)), 3, B).min() >= -1e-8


# =========================================================================== #
# 3a-2 — zero-flux robustness                                                  #
# =========================================================================== #
class TestZeroFlux:
    def test_batch_sync_zero_flux(self):
        out = batch_distorted_transient_sync(torch.zeros(4, B), 0, B)
        assert torch.count_nonzero(out) == 0
        assert torch.isfinite(out).all()

    def test_batch_async_zero_flux(self):
        out = batch_distorted_transient_async(torch.zeros(4, B), 3, B)
        assert torch.count_nonzero(out) == 0
        assert torch.isfinite(out).all()

    def test_batch_sync_mixed_zero_and_nonzero_rows(self):
        phi = _phi((3, B))
        phi[1] = 0.0
        out = batch_distorted_transient_sync(phi, 0, B)
        assert torch.count_nonzero(out[1]) == 0
        assert torch.isclose(out[0].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(out[2].sum(), torch.tensor(1.0), atol=1e-5)

    def test_batch_async_mixed_zero_and_nonzero_rows(self):
        phi = _phi((3, B))
        phi[1] = 0.0
        out = batch_distorted_transient_async(phi, 3, B)
        assert torch.count_nonzero(out[1]) == 0
        assert torch.isclose(out[0].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(out[2].sum(), torch.tensor(1.0), atol=1e-5)


# =========================================================================== #
# 3a-3 — low-flux precision of the synchronous model                           #
# =========================================================================== #
def _coates_reference(phi):
    """Single-hit synchronous detection distribution in float64, normalised to sum to one."""
    phi = phi.double()
    p = -torch.expm1(-phi) * torch.exp(-(torch.cumsum(phi, dim=-1) - phi))
    return p / p.sum(dim=-1, keepdim=True)


class TestSyncLowFlux:
    @pytest.mark.parametrize("scale", [1e-3, 1e-4, 1e-6])
    def test_matches_float64_reference(self, scale):
        phi = _phi((2, B)) * scale
        out = batch_distorted_transient_sync(phi, 0, B)
        assert torch.allclose(out.double(), _coates_reference(phi), rtol=1e-5, atol=0)

    def test_weak_ambient_is_not_lost_under_a_return(self):
        phi = torch.full((1, B), 1e-8)
        phi[0, 4] = 0.1
        out = batch_distorted_transient_sync(phi, 0, B)
        assert torch.all(out > 0)
        assert torch.allclose(out.double(), _coates_reference(phi), rtol=1e-5, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_keeps_input_dtype(self, dtype):
        assert batch_distorted_transient_sync(_phi((2, B)).to(dtype), 0, B).dtype == dtype


# =========================================================================== #
# 3b — pile-up physics                                                         #
# =========================================================================== #
class TestPileUpPhysics:
    def test_no_deadtime_preserves_shape(self):
        phi = _phi((1, B))
        out = batch_distorted_transient_async(phi, 0, B)
        ratio = out / phi
        assert float(ratio.std() / ratio.mean()) < 1e-5

    def test_deadtime_biases_toward_early_bins(self):
        flat = torch.full((1, B), 0.5)
        s = batch_distorted_transient_sync(flat, 0, B)[0]
        assert torch.all(s[1:] <= s[:-1] + 1e-7)
        assert s[0] > s[-1]

    def test_higher_flux_shifts_detection_earlier(self):
        lo = batch_distorted_transient_sync(torch.full((1, B), 0.2), 0, B)[0]
        hi = batch_distorted_transient_sync(torch.full((1, B), 2.0), 0, B)[0]
        assert _com(hi) < _com(lo)


# =========================================================================== #
# 3c — cross-checks between implementations                                    #
# =========================================================================== #
class TestImplementationCrossChecks:
    def test_batch_async_matches_eigendecomposition(self):
        phi = _phi((B,), seed=2)
        batched = batch_distorted_transient_async(phi.unsqueeze(0), 3, B, n_iterations=500)[0]
        eigen = calculate_distorted_transient(phi, 3, B)
        assert torch.allclose(batched, eigen, atol=1e-5)

    def test_sync_single_hit_matches_coates(self):
        phi = _phi((B,), seed=2)
        coates = batch_distorted_transient_sync(phi.unsqueeze(0), B, B)[0]
        multihit = calculate_distorted_transient_sync(phi, B, B)
        assert torch.allclose(coates, multihit, atol=1e-3)


# =========================================================================== #
# 3d — Monte-Carlo ground truth vs closed-form forward models                  #
# =========================================================================== #
MC_B = 10
MC_PULSES = 20000


def _mc(phi, free_running, dead_time_bins, n_pulses=MC_PULSES, seed=0):
    torch.manual_seed(seed)
    h = Histogrammer(HistConfig())
    hist = h.simulate_pixel_ewh(phi, n_pulses, phi.shape[-1], free_running, dead_time_bins, fast_sim=False)
    return hist / hist.sum()


class TestMonteCarloForwardModel:
    def test_gated_single_hit_matches_coates(self):
        phi = _phi((MC_B,), seed=3)
        mc = _mc(phi, free_running=False, dead_time_bins=MC_B)
        coates = batch_distorted_transient_sync(phi.unsqueeze(0), MC_B, MC_B)[0]
        assert torch.allclose(mc, coates, atol=0.02)

    def test_gated_single_hit_matches_calc_sync(self):
        phi = _phi((MC_B,), seed=3)
        mc = _mc(phi, free_running=False, dead_time_bins=MC_B)
        calc_sync = calculate_distorted_transient_sync(phi, MC_B, MC_B)
        assert torch.allclose(mc, calc_sync, atol=0.02)

    def test_dt0_low_flux_matches_async(self):
        phi = torch.full((MC_B,), 0.02)
        mc = _mc(phi, free_running=True, dead_time_bins=0, n_pulses=40000)
        async_ = batch_distorted_transient_async(phi.unsqueeze(0), 0, MC_B)[0]
        assert torch.allclose(mc, async_, atol=0.02)

    def test_dt0_ground_truth_is_arrival_rate(self):
        phi = torch.rand(MC_B, generator=torch.Generator().manual_seed(5)) * 2.0 + 0.5
        mc = _mc(phi, free_running=True, dead_time_bins=0, n_pulses=40000)
        assert torch.allclose(mc, phi / phi.sum(), atol=0.02)

        bernoulli = 1.0 - torch.exp(-phi)
        bernoulli = bernoulli / bernoulli.sum()
        assert not torch.allclose(mc, bernoulli, atol=0.02)

    def test_dt0_high_flux_matches_async(self):
        phi = torch.rand(MC_B, generator=torch.Generator().manual_seed(5)) * 2.0 + 0.5
        mc = _mc(phi, free_running=True, dead_time_bins=0, n_pulses=40000)
        async_ = batch_distorted_transient_async(phi.unsqueeze(0), 0, MC_B)[0]
        assert torch.allclose(mc, async_, atol=0.02)


# =========================================================================== #
# 3e — simulate_ewh / simulate_ewh_diff dispatcher wiring                      #
# =========================================================================== #
class TestSimulateEwhDispatch:
    def test_fast_path_totals_follow_the_flux(self):
        """The fast path gives the expected number of detections; it must not rescale every histogram to n_pulses."""
        h = Histogrammer(HistConfig())
        n_pulses = 500
        rates = _phi((3, B), seed=7) * torch.tensor([[1.0], [0.01], [0.2]])
        lam = rates.double().sum(-1)
        flat = torch.tensor([[0.5], [0.005], [0.05]]).expand(3, B)
        p = 1.0 - torch.exp(-flat[:, 0].double())
        cases = [
            (rates, False, 0, n_pulses * (1.0 - torch.exp(-lam))),  # first photon of each cycle
            (rates, True, 0, n_pulses * lam),  # every photon
            (flat, True, 3, n_pulses * B / (3 + (1.0 - p) / p)),  # non-paralyzable dead time of 3 bins
        ]
        for phi, free_running, dt, expected in cases:
            hists = h.simulate_ewh(phi, n_pulses, B, free_running=free_running, dead_time_bins=dt, fast_sim=True)
            totals = torch.stack([hist.sum() for hist in hists]).double()
            assert torch.allclose(totals, expected, rtol=1e-5), (free_running, dt, totals, expected)

    @pytest.mark.parametrize(
        "free_running,dt",
        [(False, MC_B), (True, 0), (True, 3), (True, MC_B + 4)],
        ids=["gated-single-hit", "free-running-dt0", "free-running-dt3", "free-running-dt-over-a-period"],
    )
    def test_fast_path_matches_monte_carlo(self, free_running, dt):
        h = Histogrammer(HistConfig())
        rates = torch.stack([_phi((MC_B,), seed=3), _phi((MC_B,), seed=4) * 0.05])
        fast = h.simulate_ewh(rates, MC_PULSES, MC_B, free_running=free_running, dead_time_bins=dt, fast_sim=True)
        torch.manual_seed(0)
        mc = h.simulate_ewh(rates, MC_PULSES, MC_B, free_running=free_running, dead_time_bins=dt, fast_sim=False)
        for f, m in zip(fast, mc):
            f, m = f.double(), m.double()
            assert float(m.sum()) == pytest.approx(float(f.sum()), abs=5 * float(f.sum()) ** 0.5)
            assert torch.all((m - f).abs() <= 6 * torch.sqrt(f) + 1)

    def test_diff_returns_expected_value(self):
        h = Histogrammer(HistConfig())
        rates = _phi((3, B), seed=7)
        out = h.simulate_ewh_diff(rates, 500, B, free_running=True, dead_time_bins=0)
        assert torch.allclose(out, rates * 500)

    def test_diff_rejects_nonzero_dead_time(self):
        h = Histogrammer(HistConfig())
        rates = _phi((3, B), seed=7)
        with pytest.raises(AssertionError):
            h.simulate_ewh_diff(rates, 500, B, free_running=True, dead_time_bins=5)


# =========================================================================== #
# 3e-2 — detections per synchronous cycle: one detector for both paths         #
# =========================================================================== #
def _first_photon_counts(phi, n_pulses):
    """Expected histogram of a synchronous cycle that records only its first photon, in float64."""
    phi = phi.double()
    return n_pulses * -torch.expm1(-phi) * torch.exp(-(torch.cumsum(phi, dim=-1) - phi))


SYNC_CAPS = [
    pytest.param(3, 1, id="single-hit-dt3"),
    pytest.param(0, 1, id="single-hit-dt0"),
    pytest.param(3, None, id="no-cap-dt3"),
    pytest.param(1, None, id="no-cap-dt1"),
    pytest.param(MC_B - 1, None, id="no-cap-dt-just-under-a-period"),
    pytest.param(0, None, id="no-cap-dt0"),
    pytest.param(3, 2, id="cap-2-dt3"),
    pytest.param(0, 2, id="cap-2-dt0"),
]


class TestDetectionsPerCycle:
    """``max_detections_per_cycle``: the Monte Carlo and the fast path simulate the same synchronous detector."""

    def test_default_is_single_hit(self):
        assert HistConfig().max_detections_per_cycle == 1
        assert Histogrammer(HistConfig()).max_detections_per_cycle == 1

    @pytest.mark.parametrize("dt", [0, 3, MC_B - 1])
    def test_monte_carlo_default_records_the_first_photon(self, dt):
        """With the default settings a synchronous cycle records at most one photon, whatever the dead time."""
        h = Histogrammer(HistConfig())
        phi = _phi((MC_B,), seed=3)
        torch.manual_seed(0)
        mc = h.simulate_ewh(phi[None], MC_PULSES, MC_B, free_running=False, dead_time_bins=dt, fast_sim=False)[0]
        expected = _first_photon_counts(phi, MC_PULSES)
        assert float(mc.sum()) <= MC_PULSES
        assert torch.all((mc.double() - expected).abs() <= 6 * torch.sqrt(expected) + 1)

    @pytest.mark.parametrize("dt,cap", SYNC_CAPS)
    def test_fast_path_matches_monte_carlo(self, dt, cap):
        h = Histogrammer(HistConfig())
        rates = torch.stack([_phi((MC_B,), seed=3), _phi((MC_B,), seed=4) * 0.05])
        kwargs = dict(free_running=False, dead_time_bins=dt, max_detections_per_cycle=cap)
        fast = h.simulate_ewh(rates, MC_PULSES, MC_B, fast_sim=True, **kwargs)
        torch.manual_seed(0)
        mc = h.simulate_ewh(rates, MC_PULSES, MC_B, fast_sim=False, **kwargs)
        for f, m in zip(fast, mc):
            f, m = f.double(), m.double()
            assert float(m.sum()) == pytest.approx(float(f.sum()), abs=5 * float(f.sum()) ** 0.5)
            assert torch.all((m - f).abs() <= 6 * torch.sqrt(f) + 1)

    def test_multi_hit_records_more_than_one_photon_per_cycle(self):
        h = Histogrammer(HistConfig())
        rates = _phi((1, MC_B), seed=3)
        kwargs = dict(free_running=False, dead_time_bins=3, fast_sim=False)
        totals = []
        for cap in (1, 2, None):
            torch.manual_seed(0)
            totals.append(float(h.simulate_ewh(rates, MC_PULSES, MC_B, max_detections_per_cycle=cap, **kwargs)[0].sum()))
        assert totals[0] <= MC_PULSES < totals[1] < totals[2]

    @pytest.mark.parametrize("fast_sim", [True, False])
    def test_free_running_does_not_use_the_cap(self, fast_sim):
        h = Histogrammer(HistConfig())
        rates = _phi((2, MC_B), seed=3)
        hists = []
        for cap in (1, 2, None):
            torch.manual_seed(0)
            hists.append(
                h.simulate_ewh(
                    rates,
                    2000,
                    MC_B,
                    free_running=True,
                    dead_time_bins=3,
                    fast_sim=fast_sim,
                    max_detections_per_cycle=cap,
                )
            )
        for other in hists[1:]:
            assert all(torch.equal(a, b) for a, b in zip(hists[0], other))

    @pytest.mark.parametrize("fast_sim", [True, False])
    @pytest.mark.parametrize("cap", [0, -1, 1.5])
    def test_invalid_cap_raises(self, fast_sim, cap):
        h = Histogrammer(HistConfig())
        with pytest.raises(ValueError, match="max_detections_per_cycle"):
            h.simulate_ewh(
                _phi((1, MC_B)),
                100,
                MC_B,
                free_running=False,
                dead_time_bins=3,
                fast_sim=fast_sim,
                max_detections_per_cycle=cap,
            )


# =========================================================================== #
# 3f — EDH is a reduction of the EWH                                           #
# =========================================================================== #
EDH_PULSES = 4000


def _edh(n_edh_bins=4, phi=None, free_running=True, dt=0, seed=0, n_pulses=EDH_PULSES):
    h = HistogrammerEDH(HistConfig())
    phi = _phi((B,), seed=seed) if phi is None else phi
    return h.simulate_pixel_edh(
        phi,
        n_pulses,
        n_edh_bins,
        free_running,
        dt,
        generator=torch.Generator().manual_seed(seed),
    )


class TestEdhFromEwh:
    @pytest.mark.parametrize("free_running,dt", [(True, 0), (True, 3), (False, 0), (False, 5)])
    def test_photon_hist_matches_ewh_at_same_seed(self, free_running, dt):
        phi = _phi((B,), seed=2)
        edh_h = HistogrammerEDH(HistConfig())
        ewh_h = Histogrammer(HistConfig())
        hist_edh, _ = edh_h.simulate_pixel_edh(
            phi,
            EDH_PULSES,
            4,
            free_running,
            dt,
            generator=torch.Generator().manual_seed(9),
        )
        hist_ewh = ewh_h.simulate_pixel_ewh(
            phi,
            EDH_PULSES,
            B,
            free_running,
            dt,
            generator=torch.Generator().manual_seed(9),
        )
        assert torch.equal(hist_edh, hist_ewh)

    @pytest.mark.parametrize("cap", [1, None])
    def test_synchronous_photon_hist_uses_the_cap(self, cap):
        phi = _phi((B,), seed=2)
        hist_edh, _ = HistogrammerEDH(HistConfig()).simulate_pixel_edh(
            phi, EDH_PULSES, 4, False, 3, generator=torch.Generator().manual_seed(9), max_detections_per_cycle=cap
        )
        hist_ewh = Histogrammer(HistConfig()).simulate_pixel_ewh(
            phi, EDH_PULSES, B, False, 3, generator=torch.Generator().manual_seed(9), max_detections_per_cycle=cap
        )
        assert torch.equal(hist_edh, hist_ewh)
        assert (float(hist_edh.sum()) <= EDH_PULSES) == (cap == 1)

    def test_synchronous_default_is_single_hit(self):
        phi = _phi((B,), seed=2)
        hists, _ = HistogrammerEDH(HistConfig()).simulate_edh(
            phi[None], EDH_PULSES, 4, free_running=False, dead_time_bins=3
        )
        assert float(hists[0].sum()) <= EDH_PULSES

    def test_photon_hist_is_at_full_tdc_resolution(self):
        hist, edh = _edh(n_edh_bins=4)
        assert hist.shape == (B,)
        assert edh.shape == (5,)

    def test_boundaries_are_monotonic_and_span_the_cycle(self):
        _, edh = _edh(n_edh_bins=4)
        assert float(edh[0]) == 0.0
        assert float(edh[-1]) == B
        assert torch.all(edh[1:] >= edh[:-1]), edh

    def test_bins_carry_equal_photon_mass(self):
        n_edh = 4
        phi = torch.linspace(0.05, 0.6, B)
        hist, edh = _edh(n_edh_bins=n_edh, phi=phi, seed=4, n_pulses=20000)

        cum = torch.cat([torch.zeros(1), torch.cumsum(hist, 0)])

        def cdf(x):
            k = min(int(x), B - 1)
            return float(cum[k] + (x - k) * hist[k]) if x < B else float(cum[B])

        mass = [cdf(float(edh[i + 1])) - cdf(float(edh[i])) for i in range(n_edh)]
        target = float(hist.sum()) / n_edh
        for m in mass:
            assert m == pytest.approx(target, rel=1e-3), (mass, target)

    def test_flat_transient_gives_evenly_spaced_boundaries(self):
        n_edh = 4
        _hist, edh = _edh(n_edh_bins=n_edh, phi=torch.full((B,), 0.3), seed=6, n_pulses=20000)
        expected = torch.linspace(0, B, n_edh + 1)
        assert torch.allclose(edh, expected, atol=1.0), (edh, expected)

    def test_negative_arrival_rates_do_not_produce_nan(self):
        phi = _phi((B,), seed=8)
        phi[2] = -0.05
        hist, edh = _edh(phi=phi, seed=8)
        assert not torch.isnan(hist).any()
        assert not torch.isnan(edh).any()
        assert float(hist[2]) == 0.0

    def test_dead_time_reduces_counts(self):
        phi = _phi((B,), seed=1) * 2.0
        no_dt, _ = _edh(phi=phi, dt=0, seed=1)
        with_dt, _ = _edh(phi=phi, dt=4, seed=1)
        assert float(with_dt.sum()) < float(no_dt.sum())

    def test_simulate_edh_returns_one_entry_per_pixel(self):
        h = HistogrammerEDH(HistConfig())
        rates = _phi((3, B), seed=7)
        hists, edhs = h.simulate_edh(rates, 1000, 4, free_running=True, dead_time_bins=0)
        assert len(hists) == 3 and len(edhs) == 3
        assert all(x.shape == (B,) for x in hists)
        assert all(e.shape == (5,) for e in edhs)
