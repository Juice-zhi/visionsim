"""Time-resolved imaging of participating media by active sensors, such as single-photon lidars and ToF cameras.

A laser flash, co-located with the camera, illuminates the scene, and the light that comes back is binned by its
time of flight. Surfaces return a pulse delayed by their distance and attenuated twice by the medium, while the
medium itself scatters part of the flash back towards the sensor (backscatter), which shows up as a smooth decay
from near the sensor, and can hide far surfaces. Ambient light, which is the passive radiance computed by
:func:`apply_medium <visionsim.medium.render.apply_medium>`, adds a constant background. All of these come from
the same medium and lighting, so active and passive sensors stay consistent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt

from visionsim.medium.model import Medium
from visionsim.medium.optics import density, henyey_greenstein, optical_depth
from visionsim.medium.render import camera_rays

SPEED_OF_LIGHT: float = 299_792_458.0
"""Speed of light, in m/s"""


class Flash(BaseModel):
    """Pulsed laser flash of an active sensor, co-located with the camera, and the sensor's time bins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wavelength: PositiveFloat = 905.0
    """wavelength of the laser, in nm"""
    bins: PositiveInt = 1000
    """number of time bins per laser cycle"""
    bin_width: PositiveFloat = 0.4e-9
    """duration of each time bin, in seconds"""
    pulse_width: PositiveFloat = 1e-9
    """full width at half maximum of the laser's Gaussian pulse, in seconds"""
    signal: NonNegativeFloat = 1.0
    """expected number of photons, per pulse, returned by a white Lambertian surface facing the sensor at
    ``reference_distance``, without any medium"""
    reference_distance: PositiveFloat = 10.0
    """distance, in meters, at which ``signal`` is defined"""
    background: NonNegativeFloat = 0.5
    """expected number of ambient photons, per laser cycle and over all bins, for a pixel of unit passive radiance"""
    min_range: NonNegativeFloat = 0.5
    """distance, in meters, from which the laser illuminates what the pixel sees, as the laser and the sensor are
    not exactly co-located. It avoids the backscatter's singularity at the sensor"""

    @property
    def bin_range(self) -> float:
        """Distance, in meters, covered by a single time bin."""
        return SPEED_OF_LIGHT * self.bin_width / 2

    @property
    def max_range(self) -> float:
        """Distance, in meters, covered by all time bins, returns from beyond it are lost."""
        return self.bins * self.bin_range


class Transient(NamedTuple):
    """Expected number of photons per time bin and per laser cycle, all of shape (h, w, bins)."""

    photons: torch.Tensor
    """all photons, i.e. ``surface + backscatter + background``"""
    surface: torch.Tensor
    """photons returned by surfaces"""
    backscatter: torch.Tensor
    """photons scattered back by the medium"""
    background: torch.Tensor
    """ambient photons"""


def _pulse(flash: Flash, **kwargs) -> torch.Tensor:
    """Laser pulse integrated over each time bin, centered on the middle bin and normalized to sum to one."""
    sigma = flash.pulse_width / (2 * math.sqrt(2 * math.log(2))) / flash.bin_width
    half = max(1, math.ceil(4 * sigma))
    edges = torch.arange(-half, half + 2, **kwargs) - 0.5
    cdf = 0.5 * (1 + torch.erf(edges / (sigma * math.sqrt(2))))
    kernel = cdf[1:] - cdf[:-1]
    return kernel / kernel.sum()


def _convolve(photons: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    shape = photons.shape
    flat = torch.nn.functional.conv1d(photons.reshape(-1, 1, shape[-1]), kernel.flip(0)[None, None], padding="same")
    return flat.reshape(shape)


def flash_transient(
    depth: npt.ArrayLike | torch.Tensor,
    albedo: npt.ArrayLike | torch.Tensor,
    cos_incidence: npt.ArrayLike | torch.Tensor,
    camera: Mapping[str, Any],
    transform_matrix: npt.ArrayLike,
    medium: Medium | None,
    flash: Flash,
    ambient_radiance: npt.ArrayLike | torch.Tensor | None = None,
    steps: int | None = None,
    nodes: int = 4,
    time: float = 0.0,
    background_depth: float = 1e9,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Transient:
    """Expected photon counts of an active sensor, per time bin and per laser cycle, seeing through a medium.

    Surfaces are assumed to be Lambertian. The backscatter of each time bin, ``∫ σ(r) · p(180°) · T(r)² / r² dr``
    over the distances ``r`` covered by the bin, is integrated with the medium's closed-form transmittance and a
    Gauss-Legendre quadrature per bin by default. As bins are much shorter than the scale over which the medium
    varies, this converges very quickly. Alternatively, when ``steps`` is set, rays are marched in uniform steps
    whose backscatter is deposited in the bin they fall in, as time-resolved renderers usually do, which is the
    ray marching counterpart. The pulse's shape is then applied to both surface returns and backscatter.

    Args:
        depth (npt.ArrayLike | torch.Tensor): Depth of the scene, as saved by Blender, of shape (h, w) or (h, w, 1).
        albedo (npt.ArrayLike | torch.Tensor): Surface albedo at the laser's wavelength, of shape (h, w) or (h, w, 1).
        cos_incidence (npt.ArrayLike | torch.Tensor): Cosine of the angle between the surface normal and the
            direction towards the sensor, of shape (h, w) or (h, w, 1).
        camera (Mapping[str, Any]): Camera intrinsics, see :func:`camera_rays <visionsim.medium.render.camera_rays>`.
        transform_matrix (npt.ArrayLike): Camera-to-world transform.
        medium (Medium | None): Participating medium, if any.
        flash (Flash): Laser flash and time bins.
        ambient_radiance (npt.ArrayLike | torch.Tensor | None, optional): Passive radiance at the laser's wavelength,
            of shape (h, w) or (h, w, 1), which sets the background. Defaults to None (no background).
        steps (int | None, optional): If set, march each ray in this many steps instead of integrating each bin.
            Defaults to None.
        nodes (int, optional): Number of Gauss-Legendre nodes per bin. Defaults to 4.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        background_depth (float, optional): Depth from which pixels are considered to see the background.
            Defaults to 1e9.
        device (torch.device | str | None, optional): Device to run on. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float64.

    Returns:
        Transient: Expected photon counts, in total and broken down by where the photons come from.
    """
    kwargs: dict[str, Any] = {"dtype": dtype, "device": device}

    def as_map(x: npt.ArrayLike | torch.Tensor) -> torch.Tensor:
        x = x.to(**kwargs) if torch.is_tensor(x) else torch.tensor(np.asarray(x), **kwargs)
        return x[..., 0] if x.ndim == 3 else x

    depth, albedo, cos_incidence = as_map(depth), as_map(albedo), as_map(cos_incidence)
    origin, directions, scale = camera_rays(camera, transform_matrix, **kwargs)
    background = ~torch.isfinite(depth) | (depth >= background_depth)
    distance = torch.where(background, torch.full_like(depth, torch.inf), depth * scale)
    reach = distance.clamp_max(flash.max_range)
    h, w = int(depth.shape[0]), int(depth.shape[1])
    n_bins, bin_range = flash.bins, flash.bin_range

    beta = medium.extinction_at((flash.wavelength,))[0] if medium is not None else 0.0
    backscatter = torch.zeros(h, w, n_bins, **kwargs)
    scale_factor = flash.signal * math.pi * flash.reference_distance**2

    if medium is not None and beta > 0 and steps is None:
        # Integrate each bin with Gauss-Legendre nodes, between the minimum range and the surface
        t, weights = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
        phase = henyey_greenstein(torch.tensor(-1.0, **kwargs), medium.anisotropy)
        edges = torch.arange(n_bins + 1, **kwargs) * bin_range
        for chunk in torch.arange(n_bins, device=device).split(64):
            lo = edges[chunk].clamp(min=flash.min_range)[None, None, :].expand(h, w, -1).minimum(reach[..., None])
            hi = edges[chunk + 1].clamp(min=flash.min_range)[None, None, :].expand(h, w, -1).minimum(reach[..., None])
            r = lo[..., None] + (hi - lo)[..., None] * (t + 1) / 2
            points = origin + r[..., None] * directions[:, :, None, None, :]
            sigma = beta * density(medium, points, time)
            tau = beta * optical_depth(medium, origin, directions[:, :, None, None, :], r, time=time)
            integrand = medium.albedo * sigma * phase * torch.exp(-2 * tau) / r.clamp_min(1e-9) ** 2
            backscatter[..., chunk] = scale_factor * (integrand * weights).sum(dim=-1) * (hi - lo) / 2
        transmittance = torch.exp(-beta * optical_depth(medium, origin, directions, reach, time=time))
    elif medium is not None and beta > 0 and steps is not None:
        # March in uniform steps, depositing the backscatter of each step into the bin it falls in
        phase = float(henyey_greenstein(torch.tensor(-1.0, **kwargs), medium.anisotropy))
        step = reach / steps
        transmittance = torch.ones_like(reach)
        for i in range(steps):
            s = (i + 0.5) * step
            sigma = beta * density(medium, origin + s[..., None] * directions, time)
            # ∫ σ T² over the step, for a constant extinction σ, is T² (1 - e^(-2 σ Δ)) / 2
            deposit = (
                medium.albedo * phase * transmittance**2 * -torch.expm1(-2 * sigma * step) / 2 / s.clamp_min(1e-9) ** 2
            )
            deposit = torch.where(s >= flash.min_range, scale_factor * deposit, torch.zeros_like(deposit))
            index = (s / bin_range).long().clamp_max(n_bins - 1)
            backscatter.scatter_add_(-1, index[..., None], deposit[..., None])
            transmittance = transmittance * torch.exp(-sigma * step)
    else:
        transmittance = torch.ones_like(reach)

    # Surface returns, twice attenuated by the medium, placed between the two nearest bin centers
    surface = torch.zeros_like(backscatter)
    amplitude = flash.signal * albedo * cos_incidence.clamp_min(0) * (flash.reference_distance / reach) ** 2
    amplitude = torch.where(distance < flash.max_range, amplitude * transmittance**2, torch.zeros_like(amplitude))
    position = (reach / bin_range - 0.5).clamp(0, n_bins - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp_max(n_bins - 1)
    fraction = position - lower
    surface.scatter_add_(-1, lower[..., None], (amplitude * (1 - fraction))[..., None])
    surface.scatter_add_(-1, upper[..., None], (amplitude * fraction)[..., None])

    kernel = _pulse(flash, **kwargs)
    surface, backscatter = _convolve(surface, kernel), _convolve(backscatter, kernel)
    ambient = torch.zeros(h, w, **kwargs) if ambient_radiance is None else as_map(ambient_radiance)
    background_photons = (flash.background * ambient.clamp_min(0) / n_bins)[..., None].expand(h, w, n_bins)
    return Transient(surface + backscatter + background_photons, surface, backscatter, background_photons.clone())


def capture_histogram(photons: torch.Tensor, cycles: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Histograms captured by a single-photon sensor that records the first photon it detects in each laser cycle.

    A photon is detected in bin ``k`` if at least one photon arrives in it, and none did in the previous bins, so
    that strong early returns hide later ones (pile-up). Counts are sampled independently per bin, with a Poisson
    distribution, which approximates the multinomial distribution of first photons over many cycles.

    Args:
        photons (torch.Tensor): Expected number of photons per bin and per cycle, of shape (..., bins).
        cycles (int): Number of laser cycles.
        generator (torch.Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        torch.Tensor: Photon counts, of shape (..., bins).
    """
    before = torch.cumsum(photons, dim=-1) - photons
    probability = -torch.expm1(-photons) * torch.exp(-before)
    return torch.poisson(cycles * probability, generator=generator)


def estimate_distance(histogram: torch.Tensor, cycles: int, flash: Flash) -> torch.Tensor:
    """Distance to the strongest return, after undoing pile-up with Coates' estimator.

    Args:
        histogram (torch.Tensor): Photon counts of a first-photon sensor, of shape (..., bins).
        cycles (int): Number of laser cycles over which the histogram was accumulated.
        flash (Flash): Laser flash and time bins.

    Returns:
        torch.Tensor: Distance, in meters, of the bin with the most photons, of shape (...).
    """
    remaining = cycles - (torch.cumsum(histogram, dim=-1) - histogram)
    flux = -torch.log1p(-(histogram / remaining.clamp_min(1)).clamp(max=1 - 1e-6))
    return (flux.argmax(dim=-1).to(histogram.dtype) + 0.5) * flash.bin_range
