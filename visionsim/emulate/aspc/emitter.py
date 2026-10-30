from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from pint import Quantity

from visionsim.emulate.aspc.utils import (
    radiance_photons,
    tof2depth,
    ureg,
    validate_units,
    watts2photons,
)

# Accepted values of pulse_shape. A "custom" pulse takes its shape from pulse_shape_custom.
PULSE_SHAPES = ("gaussian", "square", "custom")


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


class DynamicSource:
    """Base class for pulsed light sources.

    The pulse shape, which also stands in for the instrument response function (IRF), is ``"gaussian"`` or
    ``"square"`` with a full width of ``pulse_width``, or ``"custom"``. A custom shape is given by
    ``pulse_shape_custom`` as one of:

    - a callable ``f(u)`` of the offset from the return in units of ``pulse_width``, sampled as described in
      ``get_kernel``;
    - a 1-D sequence or array of taps sampled at the histogram bin width;
    - the path (``str`` or ``Path``) to a ``.npy`` file or to a whitespace-separated text file of such taps.

    Taps are relative weights of consecutive bins. Tap ``len(taps) // 2`` is time zero and lands on the bin of the
    return, and tap ``i`` is delayed by ``i - len(taps) // 2`` bins (negative means earlier). To put time zero
    elsewhere, e.g. on the peak of an asymmetric IRF, pad the taps with zeros. Taps must be finite, non-negative and
    not all zero. They are used at whatever bin width the histogram has, so an IRF measured at another bin width
    must be resampled first (or given as a callable, e.g. one that wraps ``np.interp``), and ``pulse_width`` does
    not change them. Like the other shapes, the taps are normalized by ``get_kernel``, to a unit sum by default, so
    that a return keeps its photon count.
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

        Args:
            bin_width: histogram bin width, as a one-way distance (c * t / 2) or as a time.
            normalize: "sum" scales the kernel to a unit sum, "max" to a unit peak, anything else leaves it as is.

        Returns:
            The offsets ``x`` of the taps from the return as one-way distances in meters (0 at the center tap,
            positive for later bins), and the kernel.
        """
        if self.pulse_width.check(ureg.meter.dimensionality):
            pw_m = self.pulse_width.to(ureg.meter).magnitude
        else:
            pw_m = tof2depth(self.pulse_width).to(ureg.meter).magnitude

        if bin_width.check(ureg.meter.dimensionality):
            bw_m = bin_width.to(ureg.meter).magnitude
        else:
            bw_m = tof2depth(bin_width).to(ureg.meter).magnitude

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
            half_num = max(3 * num_bins, 5)
            x = np.arange(-half_num, half_num + 1) * bw_m
            kernel = np.asarray(self.pulse_shape_custom(x / pw_m), dtype=np.float64)
            if kernel.shape != x.shape:
                raise ValueError(
                    f"Custom pulse callable must return one value per offset, shape {x.shape}, got {kernel.shape}."
                )
            _check_custom_kernel(kernel)
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
        return tof2depth(1 / self.pulse_repetition)

    @property
    def num_photons_per_cycle(self) -> Quantity:
        return watts2photons(self.avg_watts, 1 / self.pulse_repetition, self.wavelength)

    def get_scene_radiance(
        self,
        albedo: torch.Tensor | float,
        depth: torch.Tensor | Quantity | float,
        num_pixels: int,
        omega: Quantity,
    ) -> Quantity:
        n_photons = self.num_photons_per_cycle.magnitude
        depth_m = depth.magnitude if hasattr(depth, "magnitude") else depth
        albedo_val = albedo.magnitude if hasattr(albedo, "magnitude") else albedo

        # Total emitter solid angle = num_pixels * omega
        omega_val = omega.magnitude if hasattr(omega, "magnitude") else omega
        total_omega = omega_val * num_pixels

        # Radiance is total photons divided by total illuminated solid angle and area
        rad_val = (n_photons * albedo_val) / (np.pi * (depth_m**2) * total_omega)
        return rad_val * radiance_photons
