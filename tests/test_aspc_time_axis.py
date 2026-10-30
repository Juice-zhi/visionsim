"""The histogram's time axis: bin width, window and time zero.

The histogram has ``n_bins`` bins of ``bin_width`` (a one-way distance, c * t / 2), so its window is
``n_bins * bin_width``, which must fit in one laser period but may be shorter. ``time_offset`` is the time from the
start of the window to the laser pulse. A return at range r arrives ``2 * r / c + time_offset`` after the start of the
window, modulo one laser period, and lands in bin ``floor((2 * r / c + time_offset) / bin_duration)`` if that is one of
the window's bins; later returns are not recorded. Without a bin width, or with one whose window is one whole period,
the period is divided into ``n_bins`` bins exactly as before. The photons per laser cycle follow from the repetition
rate, whatever the bin width.
"""

import numpy as np
import pytest
import torch
from scipy.constants import c as C_LIGHT

from visionsim.emulate.aspc.detector import Detector
from visionsim.emulate.aspc.emitter import PulsedLaser
from visionsim.emulate.aspc.emulator import ASPCEmulator
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.utils import histogram_bin_width, ureg

# 128 bins of 14.1 mm (94 ps), a window of 1.8048 m
RES, N_BINS, BIN_M = 4, 128, 0.0141
WINDOW_M = N_BINS * BIN_M
F_WINDOW = C_LIGHT / (2 * WINDOW_M)  # 83.05 MHz: one period is the window
BIN_S = 2 * BIN_M / C_LIGHT


def _period_m(f_rep):
    return C_LIGHT / (2 * f_rep)


def _config(
    f_rep=10e6,
    bin_width=BIN_M,
    n_bins=N_BINS,
    time_offset_bins=0.0,
    free_running=False,
    dead_time=None,
    pulse_width_ps=50.0,
    **hist_kwargs,
):
    """One zone over a RES x RES sensor without falloff or ambient light. bin_width is in meters or None."""
    return {
        "histogrammer": {
            "n_bins": n_bins,
            "bin_width": None if bin_width is None else bin_width * ureg.meter,
            "time_offset": time_offset_bins * BIN_S * ureg.second,
            "n_pulses": 1000,
            "dead_time": (1 / f_rep if dead_time is None else dead_time) * ureg.second,
            "free_running": free_running,
            "fast_sim": True,
            "vignette": False,
            "pixel_fov_list": [[0.0, 1.0, 0.0, 1.0]],
            **hist_kwargs,
        },
        "active_source": {
            "pulsed_laser": {
                "pulse_repetition": f_rep * ureg.hertz,
                "pulse_width": pulse_width_ps * ureg.picosecond,
                "avg_watts": 1e-3 * ureg.watt,
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


def _transient(emulator, rng, ambient=None):
    """The zone's transient for a flat frame at range rng, or for an (RES, RES) range map."""
    depth = torch.as_tensor(np.broadcast_to(rng, (RES, RES)).copy(), dtype=torch.float64)
    transients, _ = emulator.detector.get_transients(
        depth_frame=depth * ureg.meter,
        albedo_frame=torch.full((RES, RES), 0.5, dtype=torch.float64),
        emitter=emulator.emitter,
        scene=None,
        ambient_frame=ambient,
        normal_frame=torch.tensor([0.0, 0.0, 1.0]).expand(RES, RES, 3),
    )
    return transients[0].double().numpy()


def _bins(emulator, rng):
    """Non-zero bins of the transient of a flat frame at range rng."""
    return np.flatnonzero(_transient(emulator, rng)).tolist()


# =========================================================================== #
# calculate_transients                                                        #
# =========================================================================== #
B, PERIOD, BW, H, W = 20, 10.0, 0.1, 2, 2  # a 2 m window of 20 bins in a 10 m period


def _calculate(depths, offsets=None, **kwargs):
    """Transient and ambient offset of a 2 x 2 frame of unit irradiance, one zone, binned into B bins."""
    depth = torch.as_tensor(np.broadcast_to(depths, (H, W)).copy(), dtype=torch.float64)
    offset = torch.zeros(H, W) if offsets is None else torch.full((H, W), float(offsets))
    t, amb = Histogrammer(HistConfig()).calculate_transients(
        torch.ones(H, W), depth, offset, torch.ones(1, H, W), B, PERIOD, **kwargs
    )
    return t[0].double().numpy(), float(amb[0])


class TestCalculateTransients:
    @pytest.mark.parametrize("depth,expected", [(0.05, 0), (0.55, 5), (1.234, 12), (1.95, 19)])
    def test_return_lands_in_its_bin_of_the_bin_width(self, depth, expected):
        t, _ = _calculate(depth, bin_width=BW)
        assert np.flatnonzero(t).tolist() == [expected]
        assert t.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("depth", [2.0, 2.55, 5.0, 9.99])
    def test_return_after_the_window_is_not_recorded(self, depth):
        t, _ = _calculate(depth, bin_width=BW)
        assert t.sum() == 0.0

    def test_dropped_returns_still_count_in_the_zone_mean(self):
        t, _ = _calculate([[0.55, 0.55], [3.0, 3.0]], bin_width=BW)
        assert np.flatnonzero(t).tolist() == [5]
        assert t[5] == pytest.approx(0.5)

    @pytest.mark.parametrize("depth,expected", [(10.55, 5), (20.55, 5), (11.95, 19), (30.05, 0)])
    def test_return_aliases_by_the_laser_period(self, depth, expected):
        t, _ = _calculate(depth, bin_width=BW)
        assert np.flatnonzero(t).tolist() == [expected]

    @pytest.mark.parametrize("depth", [12.5, 15.0])
    def test_aliased_return_after_the_window_is_not_recorded(self, depth):
        t, _ = _calculate(depth, bin_width=BW)
        assert t.sum() == 0.0

    @pytest.mark.parametrize("depth,expected", [(0.05, 3), (0.55, 8), (1.65, 19), (9.75, 0), (9.85, 1)])
    def test_time_offset_delays_every_return(self, depth, expected):
        t, _ = _calculate(depth, bin_width=BW, time_offset=0.3)
        assert np.flatnonzero(t).tolist() == [expected]

    def test_time_offset_pushes_late_returns_out_of_the_window(self):
        t, _ = _calculate(1.75, bin_width=BW, time_offset=0.3)
        assert t.sum() == 0.0

    @pytest.mark.parametrize("depth,expected", [(0.35, 0), (0.55, 2), (2.25, 19)])
    def test_negative_time_offset_opens_the_window_after_the_pulse(self, depth, expected):
        t, _ = _calculate(depth, bin_width=BW, time_offset=-0.3)
        assert np.flatnonzero(t).tolist() == [expected]

    def test_return_before_a_late_window_is_not_recorded(self):
        # It arrives 0.1 m before the window opens, i.e. at the end of the previous period, outside the window
        t, _ = _calculate(0.2, bin_width=BW, time_offset=-0.3)
        assert t.sum() == 0.0

    @pytest.mark.parametrize("depth,expected", [(0.2, 19), (0.55, 0), (9.9, 19)])
    def test_without_bin_width_the_window_is_the_period(self, depth, expected):
        # B bins of PERIOD / B = 0.5 m, so every return lands in the window
        t, _ = _calculate(depth, time_offset=-0.3)
        assert np.flatnonzero(t).tolist() == [expected]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("time_offset", [0.0, 0.3, -0.3])
    def test_window_of_one_period_bins_like_an_unset_bin_width(self, dtype, time_offset):
        # Bin edges in float32 and float64, returns beyond the period, and a bin width that is off by rounding
        n_bins, period = 672, C_LIGHT / (2 * 10e6)
        bin_m = period / n_bins
        depths = [k * bin_m for k in (1, 20, 400, 671)] + [period + 2.5 * bin_m, 3 * period + 0.1]
        depth = torch.tensor(depths * 2, dtype=dtype).reshape(3, 4)
        hist = Histogrammer(HistConfig())
        args = (torch.ones(3, 4, dtype=dtype), depth, torch.ones(3, 4), torch.ones(1, 3, 4), n_bins, period)
        unset, unset_amb = hist.calculate_transients(*args, time_offset=time_offset)
        for bw in (bin_m, bin_m * (1 + 1e-7), bin_m * (1 - 1e-7)):
            configured, configured_amb = hist.calculate_transients(*args, bin_width=bw, time_offset=time_offset)
            assert torch.equal(configured, unset)
            assert float(configured_amb[0]) == float(unset_amb[0])

    def test_ambient_offset_is_spread_over_the_period(self):
        _, amb = _calculate(0.55, offsets=2.0, bin_width=BW)
        assert amb == pytest.approx(2.0 * BW / PERIOD)

    def test_ambient_offset_without_bin_width_is_spread_over_the_bins(self):
        _, amb = _calculate(0.55, offsets=2.0)
        assert amb == pytest.approx(2.0 / B)

    def test_frame_batch_matches_single_frames(self):
        hist = Histogrammer(HistConfig())
        depths = torch.tensor([[[0.55, 3.0], [10.55, 1.234]], [[0.05, 0.05], [12.5, 1.95]]], dtype=torch.float64)
        offsets = torch.full((2, H, W), 2.0)
        kwargs = dict(bin_width=BW, time_offset=0.1)
        batch, batch_amb = hist.calculate_transients(
            torch.ones(2, H, W), depths, offsets, torch.ones(1, H, W), B, PERIOD, **kwargs
        )
        for i in range(2):
            single, single_amb = hist.calculate_transients(
                torch.ones(H, W), depths[i], offsets[i], torch.ones(1, H, W), B, PERIOD, **kwargs
            )
            assert torch.equal(batch[i], single[0])
            assert float(batch_amb[i]) == pytest.approx(float(single_amb[0]))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_depth_dtype_does_not_change_the_bins(self, dtype):
        depth = torch.tensor([[0.55, 10.55], [1.234, 3.0]], dtype=dtype)
        t, _ = Histogrammer(HistConfig()).calculate_transients(
            torch.ones(H, W, dtype=dtype), depth, torch.zeros(H, W), torch.ones(1, H, W), B, PERIOD, bin_width=BW
        )
        assert np.flatnonzero(t[0].numpy()).tolist() == [5, 12]
        assert t.dtype == dtype


# =========================================================================== #
# histogram_bin_width                                                         #
# =========================================================================== #
class TestHistogramBinWidth:
    def test_unset_divides_the_period(self):
        bw = histogram_bin_width(672, 10 * ureg.megahertz)
        assert bw.to(ureg.meter).magnitude == pytest.approx(_period_m(10e6) / 672)

    def test_configured_is_used_as_given(self):
        bw = histogram_bin_width(N_BINS, 79.48 * ureg.megahertz, 14.1 * ureg.millimeter)
        assert bw.units == ureg.meter
        assert bw.magnitude == pytest.approx(BIN_M)

    @pytest.mark.parametrize("n_bins,bin_m", [(N_BINS, BIN_M), (64, 0.01), (672, 0.1124), (1, 3.0)])
    def test_window_of_one_period_is_accepted(self, n_bins, bin_m):
        f_rep = C_LIGHT / (2 * n_bins * bin_m)
        assert histogram_bin_width(n_bins, f_rep * ureg.hertz, bin_m * ureg.meter).magnitude == pytest.approx(bin_m)

    @pytest.mark.parametrize("f_rep", [10e6, F_WINDOW / 1.06, F_WINDOW / 1.045])
    def test_window_shorter_than_the_period_is_accepted(self, f_rep):
        assert histogram_bin_width(N_BINS, f_rep * ureg.hertz, BIN_M * ureg.meter).magnitude == pytest.approx(BIN_M)

    @pytest.mark.parametrize("excess", [1.0001, 1.0007, 1.045, 1.06, 2.0])
    def test_window_longer_than_the_period_raises(self, excess):
        with pytest.raises(ValueError, match="exceeds one laser period"):
            histogram_bin_width(N_BINS, F_WINDOW * excess * ureg.hertz, BIN_M * ureg.meter)

    @pytest.mark.parametrize("bin_m", [0.0, -0.01])
    def test_bin_width_must_be_positive(self, bin_m):
        with pytest.raises(ValueError, match="positive"):
            histogram_bin_width(N_BINS, 10 * ureg.megahertz, bin_m * ureg.meter)


# =========================================================================== #
# Detector and emulator                                                       #
# =========================================================================== #
class TestBinWidthInTheEmulator:
    @pytest.mark.parametrize("f_rep", [10e6, 79.48e6, F_WINDOW])
    def test_configured_bin_width_is_used(self, f_rep):
        emulator = ASPCEmulator(_config(f_rep))
        assert emulator.detector.get_bin_width(emulator.emitter).to(ureg.meter).magnitude == pytest.approx(BIN_M)
        # Not snapped to period / n_bins: at 79.48 MHz that is 14.73 mm, and 1.2 m would land in bin 81
        assert [_bins(emulator, r) for r in (0.05, 0.5, 1.2)] == [[3], [35], [85]]

    @pytest.mark.parametrize("f_rep", [10e6, 40e6])
    def test_unset_bin_width_divides_the_period(self, f_rep):
        emulator = ASPCEmulator(_config(f_rep, bin_width=None))
        bin_m = _period_m(f_rep) / N_BINS
        assert emulator.detector.get_bin_width(emulator.emitter).to(ureg.meter).magnitude == pytest.approx(bin_m)
        assert _bins(emulator, 0.3) == [int(np.floor(0.3 / bin_m))]
        assert _bins(emulator, _period_m(f_rep) + 0.3) == [int(np.floor(0.3 / bin_m))]

    def test_return_after_the_window_is_not_recorded(self):
        emulator = ASPCEmulator(_config(10e6))
        assert _transient(emulator, 2.0).sum() == 0.0
        emulator.detector.get_arrival_rates(emulator.emitter)
        assert float(emulator.detector.arrival_rates.abs().sum()) == 0.0
        assert float(torch.stack(emulator.detector.get_ewh(emulator.emitter)).sum()) == 0.0

    @pytest.mark.parametrize("f_rep,expected", [(79.48e6, 8), (F_WINDOW, 13)])
    def test_return_beyond_the_period_aliases(self, f_rep, expected):
        emulator = ASPCEmulator(_config(f_rep))
        assert _bins(emulator, 2.0) == [expected]
        assert expected == int(np.floor((2.0 - _period_m(f_rep)) / BIN_M))

    def test_time_offset_sets_time_zero(self):
        emulator = ASPCEmulator(_config(10e6, time_offset_bins=14))
        assert [_bins(emulator, r) for r in (0.0001, 0.05, 0.5, 1.2)] == [[14], [17], [49], [99]]
        # 2.0 m arrives after the 128-bin window
        assert _transient(emulator, 2.0).sum() == 0.0

    def test_time_offset_does_not_change_the_signal(self):
        plain = ASPCEmulator(_config(10e6))
        shifted = ASPCEmulator(_config(10e6, time_offset_bins=14))
        for r in (0.05, 0.5, 1.2):
            assert _transient(shifted, r).sum() == pytest.approx(_transient(plain, r).sum(), rel=1e-12, abs=0)

    def test_negative_time_offset_opens_the_window_after_the_pulse(self):
        emulator = ASPCEmulator(_config(10e6, time_offset_bins=-10))
        assert _bins(emulator, 0.5) == [25]
        assert _transient(emulator, 0.05).sum() == 0.0

    def test_time_offset_with_a_window_of_one_period_wraps(self):
        emulator = ASPCEmulator(_config(10e6, bin_width=None, time_offset_bins=-1000))
        bin_m = _period_m(10e6) / N_BINS
        expected = int(np.floor((0.05 - 1000 * BIN_M) / bin_m)) % N_BINS
        assert _bins(emulator, 0.05) == [expected]

    def test_window_longer_than_the_period_raises_when_built(self):
        with pytest.raises(ValueError, match="exceeds one laser period"):
            ASPCEmulator(_config(F_WINDOW * 1.0007))

    def test_detector_checks_the_window(self):
        sensor = SPADSensor(size=(RES, RES), fov=20 * ureg.degree)
        hist = Histogrammer(HistConfig(n_bins=N_BINS, bin_width=BIN_M * ureg.meter, free_running=False))
        detector = Detector(sensor=sensor, histogrammer=hist)
        emitter = PulsedLaser(pulse_repetition=F_WINDOW * 1.06 * ureg.hertz)
        with pytest.raises(ValueError, match="exceeds one laser period"):
            detector.get_transients(torch.full((RES, RES), 0.5) * ureg.meter, torch.ones(RES, RES), emitter)

    def test_free_running_with_dead_time_needs_a_window_of_one_period(self):
        with pytest.raises(ValueError, match="free-running"):
            ASPCEmulator(_config(10e6, free_running=True, dead_time=10e-9))

    @pytest.mark.parametrize(
        "f_rep,bin_width,dead_time",
        [(10e6, BIN_M, 0.0), (10e6, BIN_M, 0.4 * BIN_S), (F_WINDOW, BIN_M, 10e-9), (10e6, None, 10e-9)],
        ids=["no-dead-time", "dead-time-under-half-a-bin", "window-of-one-period", "unset-bin-width"],
    )
    def test_free_running_is_accepted(self, f_rep, bin_width, dead_time):
        emulator = ASPCEmulator(_config(f_rep, bin_width=bin_width, free_running=True, dead_time=dead_time))
        ewh = emulator.process_frame(torch.full((RES, RES), 0.5, dtype=torch.float64) * ureg.meter, torch.ones(RES, RES))
        assert ewh.shape == (1, N_BINS)

    @pytest.mark.parametrize("bin_width,dead_time,expected", [(BIN_M, 1e-9, 11), (BIN_M, 10e-9, 106), (None, 10e-9, 13)])
    def test_dead_time_is_counted_in_bins_of_the_bin_width(self, monkeypatch, bin_width, dead_time, expected):
        emulator = ASPCEmulator(_config(10e6, bin_width=bin_width, dead_time=dead_time))
        seen = {}

        def spy(arrival_rates, **kwargs):
            seen.update(kwargs)
            return [row for row in arrival_rates]

        monkeypatch.setattr(emulator.detector.histogrammer, "simulate_ewh", spy)
        emulator.process_frame(torch.full((RES, RES), 0.5, dtype=torch.float64) * ureg.meter, torch.ones(RES, RES))
        bin_s = BIN_S if bin_width is not None else 1 / (10e6 * N_BINS)
        assert seen["dead_time_bins"] == expected == round(dead_time / bin_s)

    @pytest.mark.parametrize("bin_width", [BIN_M, BIN_M / 2, None])
    def test_pulse_is_sampled_at_the_bin_width(self, bin_width):
        emulator = ASPCEmulator(_config(10e6, bin_width=bin_width, pulse_width_ps=300.0))
        transient = _transient(emulator, 0.5)
        rates = emulator.detector.get_arrival_rates(emulator.emitter)[0].double().numpy()
        bw = emulator.detector.get_bin_width(emulator.emitter)
        _, kernel = emulator.emitter.get_kernel(bw, normalize="sum")
        assert np.allclose(rates, np.convolve(transient, kernel, mode="same"), rtol=1e-5, atol=1e-12)
        # A 300 ps pulse is 3.2 bins of 94 ps, or 6.4 bins of 47 ps (FWHM)
        peak = int(np.argmax(rates))
        above_half = np.flatnonzero(rates >= rates[peak] / 2)
        fwhm_bins = 300e-12 * C_LIGHT / 2 / bw.to(ureg.meter).magnitude
        assert above_half.size == pytest.approx(fwhm_bins, abs=1.01)

    def test_photons_per_cycle_follow_the_repetition_rate(self):
        """Guard: the bin width does not change the photons per cycle, which are avg_watts over one period."""
        totals = {
            (f, bw): _transient(ASPCEmulator(_config(f, bin_width=bw)), 0.5).sum()
            for f in (10e6, 20e6)
            for bw in (BIN_M, BIN_M / 2, None)
        }
        for bw in (BIN_M / 2, None):
            assert totals[(10e6, bw)] == pytest.approx(totals[(10e6, BIN_M)], rel=1e-9, abs=0)
        assert totals[(10e6, BIN_M)] / totals[(20e6, BIN_M)] == pytest.approx(2.0, rel=1e-6)

    @pytest.mark.parametrize("bin_width", [BIN_M, BIN_M / 2, None])
    def test_ambient_per_bin_follows_the_bin_width(self, bin_width):
        emulator = ASPCEmulator(_config(10e6, bin_width=bin_width))
        ambient = torch.full((RES, RES), 0.3, dtype=torch.float64)
        _transient(emulator, 0.5, ambient=ambient)
        per_bin = float(emulator.detector.ambient_offsets[0])
        per_cycle = float(emulator.detector._get_ambient_offset(ambient, emulator.emitter, None, ambient).mean())
        bin_m = emulator.detector.get_bin_width(emulator.emitter).to(ureg.meter).magnitude
        assert per_bin == pytest.approx(per_cycle * bin_m / _period_m(10e6), rel=1e-6, abs=0)

    @pytest.mark.parametrize("fast_sim", [True, False], ids=["fast-sim", "monte-carlo"])
    def test_histogram_has_n_bins_with_a_short_window(self, fast_sim):
        emulator = ASPCEmulator(_config(10e6, fast_sim=fast_sim))
        ewh = emulator.process_frame(torch.full((RES, RES), 0.5, dtype=torch.float64) * ureg.meter, torch.ones(RES, RES))
        assert ewh.shape == (1, N_BINS)
        assert int(np.argmax(ewh[0])) == 35

    def test_return_after_the_window_leaves_the_histogram_as_without_a_return(self):
        """A synchronous SPAD is re-armed at every window, so a return after the window changes nothing."""
        emulator = ASPCEmulator(_config(10e6))
        ambient = torch.full((RES, RES), 1e10, dtype=torch.float64)
        histograms = [
            emulator.process_frame(
                torch.full((RES, RES), rng, dtype=torch.float64) * ureg.meter, torch.ones(RES, RES), ambient
            )
            for rng in (5.0, 0.0)
        ]
        after_window, without_return = histograms
        assert after_window.sum() > 0
        assert np.array_equal(after_window, without_return)


# =========================================================================== #
# Plotting helper                                                             #
# =========================================================================== #
@pytest.fixture
def plot_utils():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    with matplotlib.rc_context():
        from visionsim.emulate.aspc.examples import ascp_plot_utils

        yield ascp_plot_utils
        ascp_plot_utils.plt.close("all")


class TestPlotBinToRange:
    @pytest.mark.parametrize("time_offset_bins", [0.0, 14.0, -3.5])
    def test_inverts_the_binning(self, plot_utils, time_offset_bins):
        ranges = np.random.default_rng(0).uniform(0.05, 1.5, 200)
        t0 = time_offset_bins * BIN_S
        bins = np.floor((ranges + C_LIGHT * t0 / 2) / BIN_M)
        start = plot_utils.bin_to_range(bins, BIN_M, t0)
        assert np.all(start <= ranges + 1e-12)
        assert np.all(ranges < start + BIN_M)

    def test_accepts_quantities(self, plot_utils):
        value = plot_utils.bin_to_range(10, 14.1 * ureg.millimeter, 2 * BIN_S * 1e9 * ureg.nanosecond)
        assert value == pytest.approx(8 * BIN_M)

    @pytest.mark.parametrize("time_offset_bins", [0, 14])
    def test_reconstruction_uses_the_bin_width_and_time_offset(self, plot_utils, tmp_path, time_offset_bins):
        zones = [[0, 0.5, 0, 0.5], [0, 0.5, 0.5, 1], [0.5, 1, 0, 0.5], [0.5, 1, 0.5, 1]]
        emulator = ASPCEmulator(_config(10e6, time_offset_bins=time_offset_bins, pixel_fov_list=zones))
        zone_ranges = np.array([[0.3, 0.75], [1.1, 1.52]])
        rng = torch.as_tensor(np.kron(zone_ranges, np.ones((2, 2))), dtype=torch.float64)
        albedo = torch.full((RES, RES), 0.5, dtype=torch.float64)
        ewh = emulator.process_frame(rng * ureg.meter, albedo)
        detector = emulator.detector
        plot_utils.plot_spad_sensor_grid(
            detector.histogrammer,
            detector.get_perpixel_fov_masks(rng, vignette=False),
            (2, 2),
            albedo,
            rng,
            detector.transients,
            detector.arrival_rates,
            list(torch.as_tensor(ewh)),
            save_path=tmp_path / "scene_reconstruction.svg",
        )
        expected = np.floor(zone_ranges / BIN_M) * BIN_M
        assert np.allclose(np.load(tmp_path / "estimated_depth_from_sensor.npy"), expected)
        assert np.allclose(np.load(tmp_path / "truedepth_quantized_by_ntbins.npy"), expected)

    def test_reconstruction_needs_a_bin_width(self, plot_utils, tmp_path):
        emulator = ASPCEmulator(_config(10e6, bin_width=None))
        rng = torch.full((RES, RES), 0.5, dtype=torch.float64)
        ewh = emulator.process_frame(rng * ureg.meter, torch.ones(RES, RES))
        detector = emulator.detector
        args = (
            detector.histogrammer,
            detector.get_perpixel_fov_masks(rng, vignette=False),
            (1, 1),
            torch.ones(RES, RES),
            rng,
            detector.transients,
            detector.arrival_rates,
            list(torch.as_tensor(ewh)),
        )
        with pytest.raises(ValueError, match="get_bin_width"):
            plot_utils.plot_spad_sensor_grid(*args, save_path=tmp_path / "scene_reconstruction.svg")
        bin_width = detector.get_bin_width(emulator.emitter)
        plot_utils.plot_spad_sensor_grid(*args, save_path=tmp_path / "scene_reconstruction.svg", bin_width=bin_width)
        estimate = np.load(tmp_path / "estimated_depth_from_sensor.npy")
        assert estimate[0, 0] == pytest.approx(np.floor(0.5 / bin_width.magnitude) * bin_width.magnitude)
