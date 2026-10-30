from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from pint import Quantity
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.special import ndtr

from visionsim.emulate.aspc.utils import (
    radiance_photons,
    tof2depth,
    ureg,
    validate_units,
    watts2photons,
)

# Accepted values of pulse_shape. A "custom" pulse takes its shape from pulse_shape_custom.
PULSE_SHAPES = ("gaussian", "square", "custom")

# Resolution of DynamicSource.get_pulse_cdf: steps per sigma of a Gaussian pulse, steps per bin (or per pulse_width, if
# shorter) of a custom callable, steps per bin of custom taps, and the most steps per bin of a Gaussian or a callable
CDF_STEPS_PER_SIGMA = 32
CDF_STEPS_PER_WIDTH = 16
CDF_STEPS_PER_TAP = 32
CDF_MAX_STEPS_PER_BIN = 1024


def _one_way_meters(value: Quantity) -> float:
    """A distance, or a time of flight as the one-way distance c * t / 2, in meters."""
    if value.check(ureg.meter.dimensionality):
        return value.to(ureg.meter).magnitude
    return tof2depth(value).to(ureg.meter).magnitude


def load_pulse_taps(taps: npt.ArrayLike | str | Path | None) -> np.ndarray:
    """Returns the taps of a custom pulse shape as a 1-D float64 array, checked but not normalized.

    ``taps`` is a 1-D sequence or array, or the path to a ``.npy`` file (read with ``np.load``) or to a text file of
    whitespace-separated values (read with ``np.loadtxt``, ``#`` starts a comment). See ``DynamicSource`` for how
    the taps are sampled.
    """
    if taps is None:
        raise ValueError(
            "pulse_shape 'custom' requires pulse_shape_custom: a callable, an array of taps, or the path to a .npy "
            "or text file of taps."
        )
    if isinstance(taps, (str, Path)):
        path = Path(taps)
        if not path.is_file():
            raise FileNotFoundError(
                f"pulse_shape_custom file '{taps}' does not exist. A custom pulse shape is a callable, an array of "
                "taps, or the path to a .npy or text file of taps (in a YAML config, use !expr for a lambda)."
            )
        taps = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, ndmin=1)

    try:
        kernel = np.asarray(taps, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise TypeError(
            "pulse_shape_custom must be a callable, an array of taps, or the path to a .npy or text file of taps, "
            f"got {type(taps).__name__}."
        ) from err

    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError(f"Custom pulse taps must be a non-empty 1-D array, got shape {kernel.shape}.")
    _check_custom_kernel(kernel)
    return kernel


def _check_custom_kernel(kernel: np.ndarray) -> None:
    """Raises if a custom pulse shape cannot be used as photon arrival weights."""
    if not np.all(np.isfinite(kernel)):
        raise ValueError("Custom pulse shape has non-finite values.")
    if np.any(kernel < 0):
        raise ValueError(f"Custom pulse shape has negative values (min {kernel.min():g}), photon rates must be >= 0.")
    if not np.any(kernel > 0):
        raise ValueError("Custom pulse shape is zero everywhere, which would remove the laser return.")


def _taps_cdf(taps: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative distribution of custom taps at ``steps`` points per bin, as ``DynamicSource.get_pulse_cdf`` gives it.

    The taps fix the cumulative distribution at the edges of their bins. In between, each run of positive taps is the
    cubic spline through its knots with zero density at both ends (a zero tap, or none, on either side), whose slopes
    (the densities at the knots) are then limited to ``[0, 3 * min(left tap, right tap)]`` (Hyman's filter, from the
    Fritsch-Carlson condition), which keeps the cubic monotone between knots, so that no bin gets a negative share.
    Interpolating each run on its own keeps zero taps around the pulse from changing it.
    """
    density = taps / np.sum(taps)
    knots = np.arange(taps.size + 1) - taps.size // 2 - 0.5
    known = np.concatenate(([0.0], np.cumsum(density)))
    slopes = np.zeros_like(knots)
    # Runs of positive taps start..stop - 1, between knots start and stop
    edges = np.flatnonzero(np.diff(np.concatenate(([0], density > 0, [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        run = slice(start, stop + 1)
        slopes[run] = CubicSpline(knots[run], known[run], bc_type="clamped")(knots[run], 1)
    slopes = np.clip(slopes, 0.0, 3 * np.minimum(np.append(0.0, density), np.append(density, 0.0)))
    # Every knot is one of the steps, so a return at the center of a bin still gives the taps
    x = knots[0] + np.arange(taps.size * steps + 1) / steps
    cdf = CubicHermiteSpline(knots, known, slopes)(x)
    # Rounding can leave dips of a few ulp where the density is 0
    return x, np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))


class DynamicSource:
    """Base class for pulsed light sources.

    The pulse shape, which also stands in for the instrument response function (IRF), is ``"gaussian"`` or
    ``"square"`` with a full width of ``pulse_width``, or ``"custom"``. A custom shape is given by
    ``pulse_shape_custom`` as one of:

    - a callable ``f(u)`` of the offset from the return in units of ``pulse_width``, sampled as described in
      ``get_kernel`` and ``get_pulse_cdf``;
    - a 1-D sequence or array of taps sampled at the histogram bin width;
    - the path (``str`` or ``Path``) to a ``.npy`` file or to a whitespace-separated text file of such taps.

    Taps are relative weights of consecutive bins, as recorded for a return at the center of a bin: the pulse
    integrated over each bin. Tap ``len(taps) // 2`` is time zero and lands on the bin of the return, and tap ``i`` is
    delayed by ``i - len(taps) // 2`` bins (negative means earlier). A return elsewhere in its bin is spread by a
    smooth pulse with these bin integrals (see ``get_pulse_cdf``), so a return at the center of a bin gives the taps.
    To put time zero elsewhere, e.g. on the peak of an asymmetric IRF, pad the taps with zeros, which does not change
    the pulse otherwise. Taps must be finite, non-negative and not all zero. They are used at whatever bin width the
    histogram has, so an IRF measured at another bin width must be resampled first (or given as a callable, e.g. one
    that wraps ``np.interp``), and ``pulse_width`` does not change them. Like the other shapes, the taps are normalized
    by ``get_kernel`` and ``get_pulse_cdf``, so that a return keeps its photon count.

    The taps do not say how the pulse is spread within a bin, so the pulse between bin edges is an interpolation, which
    is close to a time-to-digital converter for a smooth pulse a few bins wide. Given the bin integrals of a Gaussian
    with a full width at half maximum of 2.45 bins, centered on the return, a return anywhere in its bin puts each bin
    within 1% of the Gaussian's peak share of the exact bin integral (1.4% for a Gaussian centered elsewhere in the
    bin), and its centroid and variance within 2e-5 bin and 3e-4 bin^2; at 4 bins, each bin is within 0.14% of the
    peak share, at 8 bins 0.01%. Narrower pulses are not resolved by taps: at 1.5 bins the centroid is off by up to
    0.004 bin and the variance by up to 0.008 bin^2, and a single tap becomes a bump one bin wide, centered on the
    return, whose centroid follows the return in steps, off a straight line by up to 0.1 bin, as a time-to-digital
    converter's would for a pulse that narrow. A callable, sampled within each bin, has none of these errors.
    """

    @validate_units()
    def __init__(
        self,
        pulse_width: Quantity = 1.0 * ureg.nanosecond,
        pulse_shape: str = "gaussian",
        pulse_shape_custom: Callable[[np.ndarray], npt.ArrayLike] | npt.ArrayLike | str | Path | None = None,
    ):
        if not isinstance(pulse_shape, str) or pulse_shape not in PULSE_SHAPES:
            raise ValueError(
                f"Unknown pulse_shape {pulse_shape!r}, expected 'gaussian', 'square' or 'custom' (with "
                "pulse_shape_custom set to a callable, an array of taps, or the path to a file of taps)."
            )
        if pulse_shape == "custom" and not callable(pulse_shape_custom):
            # Read a file once, here, so that missing, unreadable or unusable taps fail when the source is built
            pulse_shape_custom = load_pulse_taps(pulse_shape_custom)

        self.pulse_width = pulse_width
        self.pulse_shape = pulse_shape
        self.pulse_shape_custom = pulse_shape_custom

    def get_kernel(self, bin_width: Quantity, normalize: str | None = "sum") -> tuple[np.ndarray, np.ndarray]:
        """Samples the pulse shape at the histogram bin width.

        Every shape is point-sampled at multiples of the bin width, centered on the return. A ``"gaussian"`` pulse
        and a custom callable span ``H = max(3 * round(pulse_width / bin_width), 5)`` bins on each side of the
        center, and the callable gets ``u = x / pulse_width`` and must return one non-negative value per item of
        ``u``. Custom taps are used as given, with a zero appended to an even number of taps so that the kernel has
        a center tap (still tap ``len(taps) // 2``).

        Convolving a binned transient with this kernel puts each return's pulse at its bin, wherever in the bin the
        return arrived. ``get_pulse_cdf`` describes the pulse within a bin instead, which keeps that position.

        Args:
            bin_width: histogram bin width, as a one-way distance (c * t / 2) or as a time.
            normalize: "sum" scales the kernel to a unit sum, "max" to a unit peak, anything else leaves it as is.

        Returns:
            The offsets ``x`` of the taps from the return as one-way distances in meters (0 at the center tap,
            positive for later bins), and the kernel.
        """
        pw_m = _one_way_meters(self.pulse_width)
        bw_m = _one_way_meters(bin_width)

        num_bins = max(1, round(pw_m / bw_m))
        if self.pulse_shape == "gaussian":
            half_num = max(3 * num_bins, 5)
            x = np.arange(-half_num, half_num + 1) * bw_m
            sigma = pw_m / 2.35482
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
        elif self.pulse_shape == "square":
            half_num = num_bins
            x = np.arange(-half_num, half_num + 1) * bw_m
            kernel = np.where(np.abs(x) <= pw_m / 2, 1.0, 0.0)
        elif self.pulse_shape == "custom" and callable(self.pulse_shape_custom):
            self._check_custom_callable_width()
            half_num = max(3 * num_bins, 5)
            x = np.arange(-half_num, half_num + 1) * bw_m
            kernel = self._sample_custom_callable(x / pw_m)
        elif self.pulse_shape == "custom" and not callable(self.pulse_shape_custom):
            kernel = load_pulse_taps(self.pulse_shape_custom)
            if kernel.size % 2 == 0:
                kernel = np.append(kernel, 0.0)
            half_num = kernel.size // 2
            x = np.arange(-half_num, half_num + 1) * bw_m
        else:
            raise ValueError(f"Unknown pulse_shape {self.pulse_shape!r}, expected one of {PULSE_SHAPES}.")

        if normalize == "sum":
            s: float = float(np.sum(kernel))
            if s > 0:
                kernel = kernel / s
        elif normalize == "max":
            m: float = float(np.max(kernel))
            if m > 0:
                kernel = kernel / m

        return x, kernel

    def get_pulse_cdf(self, bin_width: Quantity) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative distribution of the pulse's photons over time, in units of histogram bins.

        Unlike ``get_kernel``, which samples the pulse at whole bins, this describes the pulse within a bin, so that
        each return can be spread over the bins from its exact arrival time (see
        :meth:`~visionsim.emulate.aspc.histogrammers.HistogrammerBase.calculate_transients`). The pulse is taken to
        be uniform between consecutive offsets ``x``, so its cumulative distribution is the linear interpolation of
        ``(x, cdf)``, 0 before ``x[0]`` and 1 after ``x[-1]``:

        - ``"gaussian"``: the exact cumulative distribution at steps of sigma / 32 or less, out to 8 sigma on each
          side, with sigma = ``pulse_width`` / 2.35482 (``pulse_width`` is the full width at half maximum). Taking
          the pulse as uniform within each step changes a bin's share of a return by less than 1e-4;
        - ``"square"``: exact, uniform over ``pulse_width`` centered on the return;
        - custom taps: tap ``i`` is the share of the bin-wide interval centered ``i - len(taps) // 2`` bins after the
          return, which fixes the cumulative distribution at the edges of these intervals, so a return at the center
          of a bin gives the taps. In between, each run of positive taps is a cubic spline through these knots with
          zero density at its ends, whose slopes are limited to keep it monotone (Hyman's monotone cubic
          interpolation), sampled every 1/32 of a bin. ``DynamicSource`` gives its accuracy;
        - a custom callable: sampled every 1/16 of a bin, or of ``pulse_width`` if that is shorter, over the span of
          ``get_kernel``, each sample standing for the photons of the step centered on it.

        Steps of a Gaussian or a callable are never finer than 1/1024 of a bin, which limits the resolution of a
        Gaussian with sigma under 1/32 of a bin. A ``pulse_width`` of 0 (``"gaussian"`` or ``"square"``) gives
        ``x = [0, 0]``: every photon arrives with the return. A custom callable needs a positive ``pulse_width``, the
        unit of its offsets.

        Args:
            bin_width: histogram bin width, as a one-way distance (c * t / 2) or as a time.

        Returns:
            The offsets ``x`` from the return in bins (non-decreasing, positive for later), and the fraction ``cdf`` of
            the pulse's photons that arrive before each, rising from 0 to 1.
        """
        width = _one_way_meters(self.pulse_width) / _one_way_meters(bin_width)

        if self.pulse_shape in ("gaussian", "square") and width == 0:
            return np.zeros(2), np.array([0.0, 1.0])

        if self.pulse_shape == "gaussian":
            sigma = width / 2.35482
            steps = min(math.ceil(CDF_STEPS_PER_SIGMA / sigma), CDF_MAX_STEPS_PER_BIN)
            half_num = math.ceil(8 * sigma * steps)
            x = np.arange(-half_num, half_num + 1) / steps
            cdf = ndtr(x / sigma)
        elif self.pulse_shape == "square":
            x = np.array([-width / 2, width / 2])
            cdf = np.array([0.0, 1.0])
        elif self.pulse_shape == "custom" and callable(self.pulse_shape_custom):
            self._check_custom_callable_width()
            steps = min(math.ceil(CDF_STEPS_PER_WIDTH / min(width, 1.0)), CDF_MAX_STEPS_PER_BIN)
            half_num = max(3 * max(1, round(width)), 5) * steps
            centers = np.arange(-half_num, half_num + 1) / steps
            density = self._sample_custom_callable(centers / width)
            x = np.append(centers - 0.5 / steps, centers[-1] + 0.5 / steps)
            cdf = np.concatenate(([0.0], np.cumsum(density)))
        elif self.pulse_shape == "custom" and not callable(self.pulse_shape_custom):
            x, cdf = _taps_cdf(load_pulse_taps(self.pulse_shape_custom), CDF_STEPS_PER_TAP)
        else:
            raise ValueError(f"Unknown pulse_shape {self.pulse_shape!r}, expected one of {PULSE_SHAPES}.")

        # The Gaussian is cut at 8 sigma; every shape keeps its photon count
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        return x, cdf

    def _check_custom_callable_width(self) -> None:
        """Raises if a custom callable pulse shape cannot be sampled, as its offsets are in units of ``pulse_width``."""
        if not _one_way_meters(self.pulse_width) > 0:
            raise ValueError(
                "A custom pulse shape callable takes offsets in units of pulse_width, which must then be positive, got "
                f"pulse_width = {self.pulse_width}."
            )

    def _sample_custom_callable(self, u: np.ndarray) -> np.ndarray:
        """Evaluates a custom callable pulse shape at offsets ``u`` from the return, in units of ``pulse_width``."""
        assert callable(self.pulse_shape_custom)
        kernel = np.asarray(self.pulse_shape_custom(u), dtype=np.float64)
        if kernel.shape != u.shape:
            raise ValueError(
                f"Custom pulse callable must return one value per offset, shape {u.shape}, got {kernel.shape}."
            )
        _check_custom_kernel(kernel)
        return kernel


class PulsedLaser(DynamicSource):
    """Active illumination pulsed laser source. See ``DynamicSource`` for ``pulse_shape`` and ``pulse_shape_custom``."""

    @validate_units()
    def __init__(
        self,
        wavelength: Quantity = 940 * ureg.nanometer,
        pulse_repetition: Quantity = 10 * ureg.megahertz,
        pulse_width: Quantity = 6 * ureg.nanosecond,
        avg_watts: Quantity = 1e-7 * ureg.watt,
        pulse_shape: str = "gaussian",
        pulse_shape_custom: Callable[[np.ndarray], npt.ArrayLike] | npt.ArrayLike | str | Path | None = None,
    ):
        super().__init__(
            pulse_width=pulse_width,
            pulse_shape=pulse_shape,
            pulse_shape_custom=pulse_shape_custom,
        )
        self.wavelength = wavelength
        self.pulse_repetition = pulse_repetition
        self.avg_watts = avg_watts

    @property
    def max_resolvable_depth(self) -> Quantity:
        """One laser period as a one-way distance, c / (2 * pulse_repetition): the unambiguous range."""
        return tof2depth(1 / self.pulse_repetition)

    @property
    def num_photons_per_cycle(self) -> Quantity:
        """Photons emitted per laser cycle, avg_watts over one period, whatever the histogram's bin width."""
        return watts2photons(self.avg_watts, 1 / self.pulse_repetition, self.wavelength)

    def get_scene_radiance(
        self,
        albedo: torch.Tensor | float,
        depth: torch.Tensor | Quantity | float,
        num_pixels: int,
        omega: Quantity,
        cos_incidence: torch.Tensor | float = 1.0,
    ) -> Quantity:
        """Radiance of each pixel's laser return, in photons.

        The laser spreads its photons evenly over the sensor's field of view, of solid angle ``num_pixels * omega``,
        and lights a Lambertian surface, whose radiance is ``albedo * cos_incidence / (pi * depth^2)`` times the
        photons per steradian. ``depth`` is the range along the pixel's ray in meters (not planar depth), and
        ``cos_incidence`` the cosine of the angle between the surface normal and the ray (1 for a surface facing the
        sensor), see :func:`~visionsim.emulate.aspc.utils.incidence_cosine`.
        """
        n_photons = self.num_photons_per_cycle.magnitude
        depth_m = depth.magnitude if hasattr(depth, "magnitude") else depth
        albedo_val = albedo.magnitude if hasattr(albedo, "magnitude") else albedo

        # Total emitter solid angle = num_pixels * omega
        omega_val = omega.magnitude if hasattr(omega, "magnitude") else omega
        total_omega = omega_val * num_pixels

        # Radiance is total photons divided by total illuminated solid angle and area
        rad_val = (n_photons * albedo_val * cos_incidence) / (np.pi * (depth_m**2) * total_omega)
        return rad_val * radiance_photons
