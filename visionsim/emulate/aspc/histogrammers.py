"""Ground-truth photon-detection simulator for SPAD sensor systems."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from pint import Quantity
from torch import Tensor

from visionsim.utils.progress import ElapsedProgress
from visionsim.emulate.aspc.utils import DEPTH_CUTOFF, get_irradiance_with_fov, ureg, validate_units

# =============================================================================
# MONTE-CARLO SAMPLING & DEAD-TIME WALK KERNELS
# =============================================================================


def sample_photon_arrivals(
    phi_bar: Tensor,
    n_pulses: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw per-bin photon arrival counts for n_pulses laser cycles."""
    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    if n_pulses <= 0:
        raise ValueError(f"n_pulses must be > 0, got {n_pulses}")

    rates = torch.clamp(phi_bar.to(torch.float32), min=0.0)
    rates = rates.expand(n_pulses, -1)
    counts = torch.poisson(rates, generator=generator)
    return counts.to(torch.int64)


def _sample_occupied_cells(
    rates: Tensor,
    n_pulses: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sparse equivalent of sample_photon_arrivals."""
    empty = torch.zeros(0, dtype=torch.int64)
    total_rate = float(rates.sum())
    if total_rate <= 0.0:
        return empty, empty

    lam = torch.tensor(total_rate * n_pulses, dtype=torch.float64)
    n_photons = int(torch.poisson(lam, generator=generator).item())
    if n_photons == 0:
        return empty, empty

    n_tbins = int(rates.shape[0])
    bins = torch.multinomial(rates.to(torch.float64), n_photons, replacement=True, generator=generator)
    cycles = torch.randint(0, n_pulses, (n_photons,), generator=generator, dtype=torch.int64)

    g, _ = torch.sort(cycles * n_tbins + bins)
    occupied, counts = torch.unique_consecutive(g, return_counts=True)
    return occupied.to(torch.int64), counts.to(torch.int64)


def _walk_occupied(
    occupied: np.ndarray,
    occ_counts: np.ndarray,
    n_tbins: int,
    dead_time_bins: int,
    free_running: bool,
    paralyzable: bool,
    max_det_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential dead-time scan over occupied bins. Returns (cycles, bins)."""
    out_cycle: np.ndarray = np.empty(int(occ_counts.sum()), dtype=np.int64)
    out_bin: np.ndarray = np.empty(out_cycle.shape[0], dtype=np.int64)
    n_out = 0

    t_armed = -1
    cur_cycle = -1
    n_det_this_cycle = 0

    for i in range(occupied.shape[0]):
        g = int(occupied[i])
        cycle = g // n_tbins

        if cycle != cur_cycle:
            cur_cycle = cycle
            n_det_this_cycle = 0
            if not free_running:
                t_armed = cycle * n_tbins

        if n_det_this_cycle >= max_det_per_cycle:
            continue

        if g >= t_armed:
            n_emit = int(occ_counts[i]) if dead_time_bins == 0 else 1
            headroom = max_det_per_cycle - n_det_this_cycle
            n_emit = min(n_emit, headroom)

            for _ in range(n_emit):
                out_cycle[n_out] = cycle
                out_bin[n_out] = g - cycle * n_tbins
                n_out += 1
            n_det_this_cycle += n_emit
            t_armed = g + dead_time_bins
        elif paralyzable:
            t_armed = g + dead_time_bins

    return out_cycle[:n_out], out_bin[:n_out]


def _deadtime_walk(
    counts: np.ndarray,
    n_tbins: int,
    dead_time_bins: int,
    free_running: bool,
    paralyzable: bool,
    max_det_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense-input wrapper around _walk_occupied."""
    flat = counts.reshape(-1)
    occupied = np.nonzero(flat)[0]
    return _walk_occupied(
        occupied,
        flat[occupied],
        n_tbins,
        dead_time_bins,
        free_running,
        paralyzable,
        max_det_per_cycle,
    )


def _check_max_detections_per_cycle(max_detections_per_cycle: int | None) -> int | None:
    """Returns the per-cycle cap as an int, or None for no cap. It must be None or a positive integer."""
    if max_detections_per_cycle is None:
        return None
    if int(max_detections_per_cycle) != max_detections_per_cycle or max_detections_per_cycle < 1:
        raise ValueError(
            f"max_detections_per_cycle must be None or a positive integer, got {max_detections_per_cycle!r}"
        )
    return int(max_detections_per_cycle)


def simulate_photon_timestamps(
    phi_bar: Tensor,
    n_pulses: int,
    *,
    dead_time_bins: int = 0,
    free_running: bool = True,
    paralyzable: bool = False,
    max_detections_per_cycle: int | None = -1,
    generator: torch.Generator | None = None,
    sparse: bool | None = None,
) -> Tensor:
    """Simulate photon detection timestamps for a single pixel.

    max_detections_per_cycle caps the detections of a synchronous laser cycle: 1 keeps its first photon only, while
    None (no cap) or more than 1 re-arms the SPAD dead_time_bins after each detection. -1 selects 1 when synchronous
    and None when free running, which must be None.
    """
    if int(dead_time_bins) != dead_time_bins or dead_time_bins < 0:
        raise ValueError(f"dead_time_bins must be a non-negative integer, got {dead_time_bins!r}")
    dead_time_bins = int(dead_time_bins)

    if max_detections_per_cycle == -1:
        max_detections_per_cycle = None if free_running else 1
    if free_running and max_detections_per_cycle is not None:
        raise ValueError(
            "free-running mode is inherently multi-hit; max_detections_per_cycle "
            f"must be None, got {max_detections_per_cycle}"
        )
    max_detections_per_cycle = _check_max_detections_per_cycle(max_detections_per_cycle)

    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    n_tbins = int(phi_bar.shape[-1])
    rates = torch.clamp(phi_bar.to(torch.float32), min=0.0)

    if sparse is None:
        sparse = float(rates.sum()) * n_pulses < 0.25 * n_pulses * n_tbins

    if sparse:
        occupied, occ_counts = _sample_occupied_cells(rates, n_pulses, generator=generator)
        occupied_np, occ_counts_np = occupied.numpy(), occ_counts.numpy()
    else:
        flat = sample_photon_arrivals(rates, n_pulses, generator=generator).reshape(-1).numpy()
        occupied_np = np.nonzero(flat)[0]
        occ_counts_np = flat[occupied_np]

    if not free_running and max_detections_per_cycle == 1:
        if occupied_np.shape[0] == 0:
            return torch.zeros((0, 2), dtype=torch.int64)
        cycle_of = occupied_np // n_tbins
        first = np.empty(cycle_of.shape[0], dtype=bool)
        first[0] = True
        np.not_equal(cycle_of[1:], cycle_of[:-1], out=first[1:])
        g = occupied_np[first]
        return torch.stack([torch.from_numpy(cycle_of[first]), torch.from_numpy(g % n_tbins)], dim=1).to(torch.int64)

    cap = (1 << 62) if max_detections_per_cycle is None else int(max_detections_per_cycle)
    cyc, bns = _walk_occupied(
        occupied_np,
        occ_counts_np,
        n_tbins,
        dead_time_bins,
        bool(free_running),
        bool(paralyzable),
        cap,
    )
    return torch.stack([torch.from_numpy(cyc), torch.from_numpy(bns)], dim=1).to(torch.int64)


def timestamps_to_histogram(
    timestamps: Tensor,
    n_tbins: int,
    n_hist_bins: int | None = None,
) -> Tensor:
    """Reduce (cycle, bin) timestamps to an equi-width photon histogram."""
    if n_hist_bins is None:
        n_hist_bins = n_tbins
    if n_hist_bins > n_tbins:
        raise ValueError(f"n_hist_bins ({n_hist_bins}) cannot exceed TDC resolution n_tbins ({n_tbins})")
    if n_tbins % n_hist_bins != 0:
        raise ValueError(f"n_tbins ({n_tbins}) must be an exact multiple of n_hist_bins ({n_hist_bins})")

    if timestamps.numel() == 0:
        return torch.zeros(n_hist_bins, dtype=torch.float32)

    idx = timestamps[:, 1] * n_hist_bins // n_tbins
    return torch.bincount(idx, minlength=n_hist_bins).to(torch.float32)


# =============================================================================
# PILE-UP DISTORTION FORWARD MODELS
# =============================================================================


def _first_photon_probabilities(phi_bar: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Probability that a synchronous laser cycle records its first photon in each bin, in float64.

    Bin i records a photon with probability 1 - exp(-phi_i) if no earlier bin of the cycle did. These probabilities are
    computed with expm1 in float64: in float32 they lose most of their digits at low per-bin rates (and become 0 below
    about 3e-8).
    """
    phi = torch.clamp(phi_bar[:, :n_bins].to(torch.float64), min=0.0)
    p_detect = -torch.expm1(-phi)
    survival = torch.ones_like(p_detect)
    survival[:, 1:] = torch.cumprod(1.0 - p_detect[:, :-1], dim=-1)
    return p_detect * survival


def batch_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
) -> torch.Tensor:
    """Batched single-hit synchronous forward model (Coates), normalised to sum to one per row.

    This is the shape of :func:`batch_expected_detections_sync` with one detection per cycle. dead_time_bins is not
    used. It is computed in float64 and has the dtype of phi_bar.
    """
    prob_detect = _first_photon_probabilities(phi_bar, n_bins)
    total = prob_detect.sum(dim=-1, keepdim=True)
    return (prob_detect / (total + 1e-12)).to(phi_bar.dtype)


def _sync_detection_cap(max_detections_per_cycle: int | None, dead_time_bins: int, n_bins: int) -> int | None:
    """The most detections a synchronous cycle of n_bins can record, or None for no cap.

    It is max_detections_per_cycle, except that a cycle records only its first photon when the dead time outlasts it.
    """
    max_detections = _check_max_detections_per_cycle(max_detections_per_cycle)
    return 1 if int(dead_time_bins) >= n_bins else max_detections


def _multi_hit_detections(phi: torch.Tensor, dead_time_bins: int, max_detections: int | None) -> torch.Tensor:
    """Expected detections per bin of a synchronous laser cycle that records up to max_detections photons.

    Without dead time every photon is recorded until the cap is reached. The bins up to i then hold E[min(X, k)]
    detections, where X ~ Poisson(lam) counts the photons in them and k is the cap, and
    E[min(X, k)] = lam * Q(k - 1, lam) + k * P(k, lam) with the regularized incomplete gamma functions P and Q.

    With a dead time of d >= 1 bins, a bin records at most one photon and detections are at least d bins apart. Let
    S_c(m) be the probability that the c-th detection of the cycle falls in bin m or earlier, with S_0 = 1. The SPAD is
    ready for its c-th detection in bin i if the (c-1)-th fell in bin i - d or earlier and the c-th has not happened
    yet, which has probability S_{c-1}(i - d) - S_c(i - 1), and it then records a photon with probability
    1 - exp(-phi_i).

    Args:
        phi: clamped float64 arrival rates per bin and cycle, shape (N, B).
        dead_time_bins: dead time in bins.
        max_detections: the cap, at least 2, or None for no cap.

    Returns:
        The expected detections per bin and cycle, shape (N, B).
    """
    if dead_time_bins == 0:
        if max_detections is None:
            return phi
        lam = torch.cumsum(phi, dim=-1)
        cap = float(max_detections)
        held = lam * torch.special.gammaincc(torch.full_like(lam, cap - 1.0), lam)
        held = held + cap * torch.special.gammainc(torch.full_like(lam, cap), lam)
        return torch.clamp(torch.diff(held, dim=-1, prepend=torch.zeros_like(held[:, :1])), min=0.0)

    N, B = phi.shape
    # A cycle of B bins holds at most ceil(B / dead_time_bins) detections
    most = -(-B // dead_time_bins)
    n_max = most if max_detections is None else min(max_detections, most)

    p_detect = -torch.expm1(-phi)
    ready_for_first = torch.ones_like(phi[:, :1])
    # done[m + 1][:, c - 1] is S_c(m); done[0] stands for every bin before the cycle
    done = [torch.zeros(N, n_max, dtype=phi.dtype, device=phi.device)]
    detections = []
    for i in range(B):
        previous = torch.cat([ready_for_first, done[max(i - dead_time_bins + 1, 0)][:, :-1]], dim=-1)
        ready = torch.clamp(previous - done[i], min=0.0)
        detected = p_detect[:, i : i + 1] * ready
        done.append(done[i] + detected)
        detections.append(detected.sum(dim=-1))
    return torch.stack(detections, dim=-1)


def batch_expected_detections_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
    max_detections_per_cycle: int | None = 1,
) -> torch.Tensor:
    """Batched expected detections per laser cycle in each bin of a synchronous SPAD, which is re-armed at the start of
    every cycle.

    - With max_detections_per_cycle 1 (single-hit, Coates), a cycle records its first photon only, in bin i with
      probability (1 - exp(-phi_i)) * exp(-sum_{k<i} phi_k). A row sums to 1 - exp(-sum_i phi_i), the probability that a
      cycle records a photon at all, and dead_time_bins is not used.
    - With None (no cap) or more than 1 (multi-hit), the SPAD is dead for dead_time_bins bins after each detection and
      then records the next photon of the same cycle, up to the cap. Without dead time and without a cap it records
      every photon, so this is phi itself.

    n_pulses times a row is the expected histogram. It is computed in float64 and has the dtype of phi_bar.
    """
    max_detections = _sync_detection_cap(max_detections_per_cycle, dead_time_bins, phi_bar[:, :n_bins].shape[-1])
    if max_detections == 1:
        return _first_photon_probabilities(phi_bar, n_bins).to(phi_bar.dtype)

    phi = torch.clamp(phi_bar[:, :n_bins].to(torch.float64), min=0.0)
    return _multi_hit_detections(phi, int(dead_time_bins), max_detections).to(phi_bar.dtype)


def _free_running_chain(
    phi: torch.Tensor,
    dead_time_bins: int,
    n_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detection statistics of a free-running SPAD with a non-paralyzable dead time of at least one bin.

    The bin of each detection is a Markov chain: after a detection in bin j, the SPAD re-arms in bin
    (j + dead_time_bins) mod B and records the next photon from there on. Power iteration gives the stationary
    distribution v of that bin. The mean time between detections, T_j, is the dead time plus the mean wait for a photon
    after re-arming, so a cycle of B bins holds B / sum_j v_j T_j detections on average.

    Args:
        phi: clamped arrival rates per bin and cycle, shape (N, B).
        dead_time_bins: dead time in bins, at least 1.
        n_iterations: number of power iterations.

    Returns:
        The stationary distribution, shape (N, B), and the mean number of detections per cycle, shape (N, 1). Both are
        0 for rows without photons.
    """
    N, B = phi.shape
    device = phi.device
    dtype = phi.dtype
    total = phi.sum(dim=-1, keepdim=True)

    dt = int(dead_time_bins) % B

    phi_ext = torch.cat([phi, phi[:, :-1]], dim=-1)
    cum = torch.zeros(N, 2 * B, device=device, dtype=dtype)
    cum[:, 1:] = torch.cumsum(phi_ext, dim=-1)

    k_idx = torch.arange(B, device=device)
    i_idx = torch.arange(B, device=device)

    s = (i_idx.unsqueeze(0) - k_idx.unsqueeze(1)) % B
    k_exp = k_idx.unsqueeze(1).expand(B, B)
    end_exp = k_exp + s

    cum_k = cum[:, k_exp]
    cum_end = cum[:, end_exp]
    cyclic_sum = cum_end - cum_k

    # M[k, i]: probability that the first photon after re-arming in bin k lands in bin i, whichever cycle that takes
    p_i = (-torch.expm1(-phi)).unsqueeze(1)
    p_cycle = -torch.expm1(-total)
    M = p_i * torch.exp(-cyclic_sum) / (p_cycle.view(N, 1, 1) + 1e-15)
    M = M / (M.sum(dim=-1, keepdim=True) + 1e-15)

    j_idx = torch.arange(B, device=device)
    rearm_k = (j_idx + dt) % B
    T = M[:, rearm_k, :]

    v = torch.ones(N, 1, B, device=device, dtype=dtype) / B
    for _ in range(n_iterations):
        v = torch.bmm(v, T)
        v = v / (v.sum(dim=-1, keepdim=True) + 1e-15)

    has_photons = total > 0
    distorted = torch.where(has_photons, v.squeeze(1), 0.0)

    # Mean wait in bins after re-arming in bin k: the offset of the photon from bin k, plus B bins for each full cycle
    # without a photon (a cycle is empty with probability exp(-total))
    wait = (M * s.to(dtype)).sum(dim=-1) + B * torch.exp(-total) / (p_cycle + 1e-15)
    time_between = (distorted * (int(dead_time_bins) + wait[:, rearm_k])).sum(dim=-1, keepdim=True)
    detections_per_cycle = torch.where(has_photons, B / torch.where(has_photons, time_between, 1.0), 0.0)

    return distorted, detections_per_cycle


def batch_distorted_transient_async(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
    n_iterations: int = 150,
) -> torch.Tensor:
    """Batched asynchronous (free-running) forward model using power iteration, normalised to sum to one per row.

    This is the shape of :func:`batch_expected_detections_async`.
    """
    if phi_bar.ndim == 1:
        phi_bar = phi_bar.unsqueeze(0)

    phi = torch.clamp(phi_bar[:, :n_bins], min=0.0)
    total = phi.sum(dim=-1, keepdim=True)

    if dead_time_bins == 0:
        zero_mask = total.squeeze(-1) == 0
        result = phi / (total + 1e-12)
        result[zero_mask] = 0.0
        return result

    distorted, _ = _free_running_chain(phi, dead_time_bins, n_iterations)
    return distorted


def batch_expected_detections_async(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
    n_iterations: int = 150,
) -> torch.Tensor:
    """Batched expected detections per laser cycle in each bin of a free-running (asynchronous) SPAD.

    Without dead time every photon is detected, so this is phi itself. With a non-paralyzable dead time it is the shape
    of :func:`batch_distorted_transient_async` times the mean number of detections per cycle, which follows from the
    mean time between detections: the dead time plus the mean wait for the next photon after re-arming. n_pulses times
    a row is the expected histogram. When nearly every bin is bright enough to fire as soon as the SPAD re-arms, the
    power iteration converges slowly and needs a larger n_iterations.
    """
    if phi_bar.ndim == 1:
        phi_bar = phi_bar.unsqueeze(0)

    phi = torch.clamp(phi_bar[:, :n_bins], min=0.0)
    if dead_time_bins == 0:
        return phi

    distorted, detections_per_cycle = _free_running_chain(phi, dead_time_bins, n_iterations)
    return distorted * detections_per_cycle


def batch_sample_detections(
    phi_bar: torch.Tensor,
    n_pulses: int,
    n_bins: int,
    free_running: bool,
    dead_time_bins: int,
    generator: torch.Generator | None = None,
    max_detections_per_cycle: int | None = 1,
) -> torch.Tensor:
    """Batched noisy histograms over n_pulses laser cycles, drawn around the expected counts of the closed-form models.

    - Synchronous with one detection per cycle (max_detections_per_cycle 1, or a dead time that outlasts the cycle):
      a multinomial over the bins and "no detection", which is exact. It is drawn bin by bin: of the cycles without a
      detection so far, each records a photon in bin i with probability 1 - exp(-phi_i).
    - Without dead time (free running, or synchronous multi-hit): every photon is counted, up to the cap of a
      synchronous cycle, so the bins are independent Poisson counts around the expected counts. This is exact without
      a cap, and overstates the spread of the bins in which the cap is often reached.
    - With dead time (free running, or synchronous multi-hit): independent binomial counts per bin around
      :func:`batch_expected_detections_async` or :func:`batch_expected_detections_sync`, since a bin records at most
      one detection per cycle. Synchronous cycles are independent, so each bin on its own is exact, but the bins are
      drawn independently although the dead time anti-correlates the detections of a cycle. Free running, it is an
      approximation. Per bin it is close to the Monte Carlo when the dead time is a small fraction of the period, but
      it ignores the correlations that dead time introduces between bins and cycles, so it overstates the spread of the
      total count when the dead time is comparable with the period or longer.

    max_detections_per_cycle is only used when synchronous, as in :func:`batch_expected_detections_sync`. The counts
    are whole numbers with the dtype of phi_bar.
    """
    if phi_bar.ndim == 1:
        phi_bar = phi_bar.unsqueeze(0)
    max_detections = (
        None
        if free_running
        else _sync_detection_cap(max_detections_per_cycle, dead_time_bins, phi_bar[:, :n_bins].shape[-1])
    )

    if max_detections == 1:
        p_detect = -torch.expm1(-torch.clamp(phi_bar[:, :n_bins].to(torch.float64), min=0.0))
        counts = torch.zeros_like(p_detect)
        remaining = torch.full_like(p_detect[:, 0], float(n_pulses))
        for i in range(p_detect.shape[-1]):
            counts[:, i] = torch.binomial(remaining, p_detect[:, i], generator=generator)
            remaining = remaining - counts[:, i]
    elif dead_time_bins == 0:
        expected = torch.clamp(phi_bar[:, :n_bins].to(torch.float64), min=0.0)
        if max_detections is not None:
            expected = _multi_hit_detections(expected, 0, max_detections)
        counts = torch.poisson(expected * n_pulses, generator=generator)
    else:
        if free_running:
            p_bin = batch_expected_detections_async(phi_bar, dead_time_bins, n_bins)
        else:
            p_bin = batch_expected_detections_sync(phi_bar, dead_time_bins, n_bins, max_detections)
        p_bin = p_bin.to(torch.float64)
        counts = torch.binomial(
            torch.full_like(p_bin, float(n_pulses)), torch.clamp(p_bin, 0.0, 1.0), generator=generator
        )
    return counts.to(phi_bar.dtype)


def calculate_distorted_transient(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
) -> torch.Tensor:
    is_1d = phi_bar.ndim == 1
    phi = phi_bar.unsqueeze(0) if is_1d else phi_bar
    out = batch_distorted_transient_async(phi, dead_time_bins, n_bins)
    return out.squeeze(0) if is_1d else out


def calculate_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
) -> torch.Tensor:
    is_1d = phi_bar.ndim == 1
    phi = phi_bar.unsqueeze(0) if is_1d else phi_bar
    out = batch_distorted_transient_sync(phi, dead_time_bins, n_bins)
    return out.squeeze(0) if is_1d else out


# =============================================================================
# HISTOGRAMMER BASE CLASS
# =============================================================================


class HistogrammerBase:
    """Base class for SPAD readout and TDC histogramming operations."""

    def __init__(self):
        pass

    def get_pixel_fov_mask(
        self,
        empty_ref: torch.Tensor,
        r1: float = 0,
        r2: float = 1.0,
        c1: float = 0,
        c2: float = 1.0,
        vignette: bool = True,
    ) -> torch.Tensor:
        if empty_ref.ndim == 3:
            h, w = empty_ref.shape[1], empty_ref.shape[2]
        else:
            h, w = empty_ref.shape[0], empty_ref.shape[1]

        y_min_idx, y_max_idx = round(r1 * h), round(r2 * h)
        x_min_idx, x_max_idx = round(c1 * w), round(c2 * w)

        mask = torch.zeros((h, w), dtype=torch.float32, device=empty_ref.device)
        mask[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = 1.0

        if vignette:
            y_coords = torch.linspace(-1, 1, h, device=empty_ref.device).unsqueeze(1)
            x_coords = torch.linspace(-1, 1, w, device=empty_ref.device).unsqueeze(0)
            r_sq = y_coords**2 + x_coords**2
            vignette_mask = torch.clamp(1.0 - 0.5 * r_sq, min=0.0)
            mask = mask * vignette_mask

        return mask

    def get_perpixel_fov_masks(
        self,
        empty_ref: torch.Tensor,
        pixel_fov_list: list | None = None,
        vignette: bool = True,
    ) -> torch.Tensor:
        if pixel_fov_list is None:
            pixel_fov_list = getattr(self, "pixel_fov_list", [[0, 1.0, 0, 1.0]])

        masks = [
            self.get_pixel_fov_mask(empty_ref, r1=fov[0], r2=fov[1], c1=fov[2], c2=fov[3], vignette=vignette)
            for fov in pixel_fov_list
        ]
        return torch.stack(masks)

    def calculate_transients(
        self,
        irradiance_frames: torch.Tensor,
        depth_frames: torch.Tensor,
        offsets: torch.Tensor,
        fov_masks: torch.Tensor,
        gt_ntime_bins: int,
        max_depth: float,
        sensor_fov: list | None = None,
        pixel_fov_list: list | None = None,
        w: int | None = None,
        h: int | None = None,
        omega: float | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Bins each zone's per-pixel laser returns by depth into a transient and averages its ambient offset.

        ``depth_frames`` hold each pixel's range, the one-way distance along its ray (not planar depth), so a return
        at range r lands in bin ``floor(r * gt_ntime_bins / max_depth)``. Depths beyond max_depth wrap around (range
        aliasing). Depths that are not finite, not positive, or at least DEPTH_CUTOFF (rays that miss every object)
        add no laser return, but their pixels still count in the zone mean and in the ambient offset.

        A zone's transient is the mean of its pixels' irradiance, weighted by its FOV mask. If ``sensor_fov``,
        ``pixel_fov_list``, ``w``, ``h`` and ``omega`` are all given, each zone is also rescaled by
        :func:`~visionsim.emulate.aspc.utils.get_irradiance_with_fov`, which is only meant for irradiance that does
        not already carry each pixel's solid angle.
        """
        if hasattr(irradiance_frames, "magnitude"):
            irradiance_frames = irradiance_frames.magnitude
        if hasattr(depth_frames, "magnitude"):
            depth_frames = depth_frames.magnitude
        if hasattr(offsets, "magnitude"):
            offsets = offsets.magnitude

        device = irradiance_frames.device
        if fov_masks.device != device:
            fov_masks = fov_masks.to(device)
        if depth_frames.device != device:
            depth_frames = depth_frames.to(device)
        if offsets.device != device:
            offsets = offsets.to(device)

        transient_list = []
        ambient_offsets = []

        if len(irradiance_frames.shape) == 3:
            for frame_idx in range(irradiance_frames.shape[0]):
                with ElapsedProgress(disable=True) as progress:
                    task = progress.add_task("Processing FOV masks", total=len(fov_masks))
                    for mask_idx, fov_mask in enumerate(fov_masks):
                        index_mask = fov_mask > 0
                        current_irradiance_vals = irradiance_frames[frame_idx][index_mask]

                        n_fov_pixels = int(index_mask.sum())
                        if n_fov_pixels == 0:
                            transient_list.append(
                                torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device)
                            )
                            ambient_offsets.append(torch.zeros((), device=device))
                            continue

                        if (
                            sensor_fov is not None
                            and pixel_fov_list is not None
                            and w is not None
                            and h is not None
                            and omega is not None
                        ):
                            fov_irradiance_vals = get_irradiance_with_fov(
                                current_irradiance_vals, sensor_fov, pixel_fov_list[mask_idx], omega, w, h
                            )
                        else:
                            fov_irradiance_vals = current_irradiance_vals

                        if hasattr(fov_irradiance_vals, "magnitude"):
                            fov_irradiance_vals = fov_irradiance_vals.magnitude

                        current_depth_vals = depth_frames[frame_idx][index_mask]
                        masked_offsets = offsets[frame_idx][index_mask]

                        fov_irradiance_vals = fov_irradiance_vals * fov_mask[index_mask]

                        valid = (
                            torch.isfinite(current_depth_vals)
                            & (current_depth_vals > 0)
                            & (current_depth_vals < DEPTH_CUTOFF)
                        )
                        current_depth_vals = current_depth_vals[valid]
                        fov_irradiance_vals = fov_irradiance_vals[valid]

                        ambient_offsets.append(masked_offsets.sum() / (n_fov_pixels * gt_ntime_bins))

                        if current_depth_vals.numel() == 0:
                            transient_list.append(
                                torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device)
                            )
                            continue

                        transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
                        transient_idx = torch.remainder(transient_idx, gt_ntime_bins)

                        row = torch.zeros(gt_ntime_bins, dtype=fov_irradiance_vals.dtype, device=device)
                        row = row.scatter_add(0, transient_idx, fov_irradiance_vals / n_fov_pixels)
                        transient_list.append(row.unsqueeze(0))
                        progress.advance(task)
            return torch.concat(transient_list), ambient_offsets
        else:
            with ElapsedProgress(disable=True) as progress:
                task = progress.add_task("Processing FOV masks", total=len(fov_masks))
                for mask_idx, fov_mask in enumerate(fov_masks):
                    index_mask = fov_mask > 0
                    current_irradiance_vals = irradiance_frames[index_mask]

                    n_fov_pixels = int(index_mask.sum())
                    if n_fov_pixels == 0:
                        transient_list.append(torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device))
                        ambient_offsets.append(torch.zeros((), device=device))
                        continue

                    if (
                        sensor_fov is not None
                        and pixel_fov_list is not None
                        and w is not None
                        and h is not None
                        and omega is not None
                    ):
                        fov_irradiance_vals = get_irradiance_with_fov(
                            current_irradiance_vals, sensor_fov, pixel_fov_list[mask_idx], omega, w, h
                        )
                    else:
                        fov_irradiance_vals = current_irradiance_vals

                    if hasattr(fov_irradiance_vals, "magnitude"):
                        fov_irradiance_vals = fov_irradiance_vals.magnitude

                    current_depth_vals = depth_frames[index_mask]
                    masked_offsets = offsets[index_mask]

                    fov_irradiance_vals = fov_irradiance_vals * fov_mask[index_mask]

                    valid = (
                        torch.isfinite(current_depth_vals)
                        & (current_depth_vals > 0)
                        & (current_depth_vals < DEPTH_CUTOFF)
                    )
                    current_depth_vals = current_depth_vals[valid]
                    fov_irradiance_vals = fov_irradiance_vals[valid]

                    ambient_offsets.append(masked_offsets.sum() / (n_fov_pixels * gt_ntime_bins))

                    if current_depth_vals.numel() == 0:
                        transient_list.append(torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device))
                        continue

                    transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
                    transient_idx = torch.remainder(transient_idx, gt_ntime_bins)

                    row = torch.zeros(gt_ntime_bins, dtype=fov_irradiance_vals.dtype, device=device)
                    row = row.scatter_add(0, transient_idx, fov_irradiance_vals / n_fov_pixels)
                    transient_list.append(row.unsqueeze(0))
                    progress.advance(task)
            return torch.concat(transient_list), ambient_offsets

    def calculate_arrival_rates(
        self, irf: torch.Tensor, transients: torch.Tensor, offset, gt_ntime_bins: int
    ) -> torch.Tensor:
        arrival_rates = torch.zeros_like(transients, dtype=torch.float32, device=transients.device)
        for i in range(transients.shape[0]):
            convolved_signal = F.conv1d(transients[i].view(1, 1, -1), irf.flip(-1).view(1, 1, -1), padding="same").view(
                -1
            )

            if isinstance(offset, (list, tuple)):
                background = offset[i] if i < len(offset) else (offset[0] if len(offset) > 0 else 0.0)
            elif isinstance(offset, torch.Tensor):
                background = offset[i] if offset.dim() > 0 and i < offset.shape[0] else offset
            else:
                background = offset

            if not isinstance(background, torch.Tensor):
                background = torch.tensor(background, dtype=torch.float32, device=transients.device)

            background_extended = background.expand_as(convolved_signal) if background.dim() == 0 else background
            arrival_rates[i, :] = convolved_signal + background_extended
        return arrival_rates

    def simulate_pixel_photon_hist(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool,
        dead_time_bins: int,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
        max_detections_per_cycle: int | None = 1,
    ) -> torch.Tensor:
        """Monte Carlo histogram of one pixel over n_pulses laser cycles.

        A synchronous cycle records up to max_detections_per_cycle photons (see :class:`HistConfig`); a free-running
        SPAD records every photon its dead time allows and does not use the cap.
        """
        n_tbins = phi_bar.shape[-1]
        timestamps = simulate_photon_timestamps(
            phi_bar,
            n_pulses,
            dead_time_bins=int(dead_time_bins),
            free_running=bool(free_running),
            paralyzable=bool(paralyzable),
            max_detections_per_cycle=None if free_running else _check_max_detections_per_cycle(max_detections_per_cycle),
            generator=generator,
        )
        photon_hist = timestamps_to_histogram(timestamps, n_tbins, n_bins)
        return photon_hist.to(phi_bar.device)


@dataclass
class HistConfig:
    """Settings of a :class:`Histogrammer` or :class:`HistogrammerEDH`."""

    min_depth: Quantity = 0 * ureg.meter
    max_depth: Quantity = 20 * ureg.meter
    n_bins: int = 1000
    bin_width: Quantity = 0.03 * ureg.meter
    n_pulses: int = 100
    """Number of laser cycles integrated into each histogram"""
    dead_time: Quantity = 10e-9 * ureg.second
    """Time after a detection during which the SPAD cannot detect another photon (non-paralyzable). A synchronous
    single-hit cycle (``max_detections_per_cycle=1``) does not use it"""
    free_running: bool = True
    """If true, the SPAD re-arms after its dead time regardless of the laser cycle (free running). If false, it is also
    re-armed at the start of every laser cycle (synchronous)"""
    max_detections_per_cycle: int | None = 1
    """Only used when synchronous: the most photons a laser cycle can record. 1 is a single-hit time-to-digital
    converter, which keeps the first photon of each cycle, so ``dead_time`` does not matter. None (no cap) or more than
    1 is a multi-hit one: after each detection the SPAD is dead for ``dead_time`` and then records the next photon of
    the same cycle, up to the cap. A free-running SPAD records every photon its dead time allows. The Monte Carlo and
    ``fast_sim`` both use this setting, so they model the same detector"""
    fast_sim: bool = True
    """If true, compute the histograms from closed-form pile-up models instead of simulating every photon (Monte
    Carlo). They give the expected number of detections in each bin over ``n_pulses`` cycles, so the totals follow the
    flux: a synchronous cycle records up to ``max_detections_per_cycle`` photons, a free-running SPAD without dead time
    records every photon, and one with dead time as many as its dead time allows. By default the result is this
    noise-free expectation, in floats; see ``fast_sim_noise``."""
    fast_sim_noise: bool = False
    """Only used with ``fast_sim``. If false, return the noise-free expected counts. If true, draw whole-number counts
    around them instead (see :func:`batch_sample_detections`): multinomial when a synchronous cycle records one photon,
    Poisson per bin without dead time, and binomial per bin with dead time, which ignores the correlations dead time
    introduces between bins (and, free running, between cycles). The Monte Carlo (``fast_sim=False``) is always
    noisy."""
    vignette: bool = False
    pixel_fov_list: list = field(default_factory=lambda: [[0, 1.0, 0, 1.0]])


class Histogrammer(HistogrammerBase):
    """Equi-Width Histogrammer (EWH) class."""

    @validate_units()
    def __init__(self, config: HistConfig | None = None):
        super().__init__()
        if config is None:
            config = HistConfig()
        for k, v in config.__dict__.items():
            setattr(self, k, v)

    def simulate_pixel_ewh(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool,
        dead_time_bins: int,
        fast_sim: bool = True,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
        max_detections_per_cycle: int | None = 1,
    ) -> torch.Tensor:
        return self.simulate_pixel_photon_hist(
            phi_bar,
            n_pulses,
            n_bins,
            free_running,
            dead_time_bins,
            paralyzable=paralyzable,
            generator=generator,
            max_detections_per_cycle=max_detections_per_cycle,
        )

    def simulate_ewh(
        self,
        arrival_rates: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool = True,
        dead_time_bins: int = 0,
        fast_sim: bool = True,
        fast_sim_noise: bool = False,
        generator: torch.Generator | None = None,
        max_detections_per_cycle: int | None = 1,
    ) -> list[torch.Tensor]:
        """Simulates the histogram of each row of arrival_rates over n_pulses laser cycles.

        With fast_sim, the histograms come from the closed-form pile-up models: n_pulses times the expected detections
        per cycle (:func:`batch_expected_detections_sync` or :func:`batch_expected_detections_async`), which is noise
        free, or with fast_sim_noise a noisy draw around it (:func:`batch_sample_detections`). Otherwise every photon
        is simulated (Monte Carlo). A synchronous cycle records up to max_detections_per_cycle photons in either case.
        See :class:`HistConfig`.
        """
        if fast_sim:
            if fast_sim_noise:
                photon_hists = batch_sample_detections(
                    arrival_rates,
                    n_pulses,
                    n_bins,
                    free_running,
                    dead_time_bins,
                    generator=generator,
                    max_detections_per_cycle=max_detections_per_cycle,
                )
            elif free_running:
                photon_hists = batch_expected_detections_async(arrival_rates, dead_time_bins, n_bins) * n_pulses
            else:
                photon_hists = (
                    batch_expected_detections_sync(arrival_rates, dead_time_bins, n_bins, max_detections_per_cycle)
                    * n_pulses
                )
            return [photon_hists[i] for i in range(photon_hists.shape[0])]

        ewh_pixel_list = []
        for p_idx in range(arrival_rates.shape[0]):
            ewh_pixel_list.append(
                self.simulate_pixel_ewh(
                    arrival_rates[p_idx],
                    n_pulses,
                    n_bins,
                    free_running,
                    dead_time_bins,
                    fast_sim=fast_sim,
                    generator=generator,
                    max_detections_per_cycle=max_detections_per_cycle,
                )
            )
        return ewh_pixel_list

    def simulate_ewh_diff(
        self,
        arrival_rates: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool = True,
        dead_time_bins: int = 0,
    ) -> torch.Tensor:
        assert dead_time_bins == 0, "Current differentiable EWH does not support non-zero dead time"
        return arrival_rates * n_pulses


class HistogrammerEDH(HistogrammerBase):
    """Equi-Depth Histogrammer (EDH) class."""

    def __init__(self, config: HistConfig | None = None):
        super().__init__()
        if config is None:
            config = HistConfig()
        for k, v in config.__dict__.items():
            setattr(self, k, v)

    def photon_hist2edh(self, photon_hist, n_edh_bins):
        gt_ntime_bins = photon_hist.shape[-1]
        tr_img = photon_hist + torch.max(photon_hist) * 1e-10 / gt_ntime_bins
        tr_cs = torch.cumsum(tr_img, axis=-1)
        tr_sum = torch.sum(tr_img, axis=-1)
        edh_bins = torch.zeros(n_edh_bins + 1, device=photon_hist.device)
        edh_bins[-1] = gt_ntime_bins

        for idx in range(edh_bins.shape[-1] - 2):
            e1_ori = tr_cs - tr_sum * (idx + 1.0) / n_edh_bins
            e1 = e1_ori * 1.0
            e2 = e1_ori * 1.0
            e1[e1_ori > 0] = -1e12
            e2[e1_ori < 0] = 1e12

            neg_max_val_, neg_max_idx_ = torch.max(e1, axis=-1)
            pos_min_val_, _ = torch.min(e2, axis=-1)
            a1 = torch.abs(neg_max_val_)
            b1 = pos_min_val_
            edh_bins[idx + 1] = neg_max_idx_ + a1 * 1.0 / (a1 + b1 + 1e-14)

        edh_bins[1:-1] += 1
        return edh_bins

    def simulate_pixel_edh(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool,
        dead_time_bins: int,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
        max_detections_per_cycle: int | None = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_tbins = phi_bar.shape[-1]
        photon_hist = self.simulate_pixel_photon_hist(
            phi_bar,
            n_pulses,
            n_tbins,
            free_running,
            dead_time_bins,
            paralyzable=paralyzable,
            generator=generator,
            max_detections_per_cycle=max_detections_per_cycle,
        )
        photon_edh = self.photon_hist2edh(photon_hist, n_bins)
        return photon_hist, photon_edh

    def simulate_edh(
        self,
        arrival_rates: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool = True,
        dead_time_bins: int = 0,
        max_detections_per_cycle: int | None = 1,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        edh_pixel_list = []
        photon_hist_pixel_list = []

        for p_idx in range(arrival_rates.shape[0]):
            pixel_photon_hist, pixel_edh = self.simulate_pixel_edh(
                arrival_rates[p_idx],
                n_pulses,
                n_bins,
                free_running,
                dead_time_bins,
                max_detections_per_cycle=max_detections_per_cycle,
            )
            photon_hist_pixel_list.append(pixel_photon_hist)
            edh_pixel_list.append(pixel_edh)
        return photon_hist_pixel_list, edh_pixel_list
