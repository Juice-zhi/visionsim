"""Ray marching of participating media, i.e. the conventional numerical integration of light transport.

This integrates the same single scattering model as :func:`apply_medium <visionsim.medium.render.apply_medium>`,
but numerically, by marching along each camera ray in uniform steps as volume renderers typically do. It converges
to the closed form as the number of steps increases, but is slower and suffers from banding when using few steps,
which makes it a baseline against which to compare the closed form.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from visionsim.medium.model import HeightFog, Lighting, Medium
from visionsim.medium.optics import density, henyey_greenstein, sky_quadrature
from visionsim.medium.render import _SKY_NODES, RGB_WAVELENGTHS, MediumResult, _by_pixels, _per_channel, camera_rays


def _sun_transmittance(
    points: torch.Tensor,
    towards_sun: torch.Tensor,
    medium: Medium,
    beta: torch.Tensor,
    shadow_steps: int,
    time: float,
) -> torch.Tensor:
    """Transmittance between points in an exponential height fog and a sun, of shape (..., c)."""
    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    sun_z = float(towards_sun[2])
    if sun_z <= 0:
        return torch.zeros(*points.shape[:-1], len(beta), dtype=points.dtype, device=points.device)

    if shadow_steps <= 0:
        # The fog above a point has an optical depth of density * falloff, vertically
        return torch.exp(-density(medium, points, time)[..., None] * beta * fog.falloff / sun_z)

    # March towards the sun until the fog's density becomes negligible (e^-14 ≈ 1e-6 of its base density)
    top = fog.base_height + 14 * fog.falloff
    length = ((top - points[..., 2]) / sun_z).clamp_min(0)
    step = length / shadow_steps
    depth = torch.zeros_like(length)
    for j in range(shadow_steps):
        samples = points + ((j + 0.5) * step)[..., None] * towards_sun
        depth = depth + density(medium, samples, time) * step
    return torch.exp(-depth[..., None] * beta)


def ray_march_medium(
    radiance: npt.ArrayLike | torch.Tensor,
    depth: npt.ArrayLike | torch.Tensor,
    camera: Mapping[str, Any],
    transform_matrix: npt.ArrayLike,
    medium: Medium,
    lighting: Lighting,
    steps: int = 64,
    shadow_steps: int = 0,
    wavelengths: Sequence[float] | None = None,
    time: float = 0.0,
    background_depth: float = 1e9,
    max_distance: float = 1000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> MediumResult:
    """Add a participating medium to a rendered frame by ray marching.

    Each camera ray is split into ``steps`` uniform steps, over which the extinction is assumed to be constant, as
    evaluated at the middle of the step, and the light scattered towards the camera is integrated analytically
    (the energy conserving scheme of Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite",
    2015). Light reaching each sample is computed as in :func:`apply_medium <visionsim.medium.render.apply_medium>`,
    except that, when ``shadow_steps`` is positive, sunlight is attenuated by marching towards the sun instead of
    in closed form, as generic volume renderers do. Skylight is always attenuated in closed form, since marching
    towards every direction of the sky would be prohibitively expensive.

    Args:
        radiance (npt.ArrayLike | torch.Tensor): Linear radiance of the scene without the medium, of shape (h, w, c).
        depth (npt.ArrayLike | torch.Tensor): Depth of the scene, as saved by Blender, of shape (h, w) or (h, w, 1).
        camera (Mapping[str, Any]): Camera intrinsics, see :func:`camera_rays <visionsim.medium.render.camera_rays>`.
        transform_matrix (npt.ArrayLike): Camera-to-world transform.
        medium (Medium): Participating medium.
        lighting (Lighting): Lighting of the medium.
        steps (int, optional): Number of steps along each camera ray. Defaults to 64.
        shadow_steps (int, optional): Number of steps towards the sun, from each sample, when sunlight is attenuated
            by the medium. Defaults to 0, i.e. the attenuation is computed in closed form.
        wavelengths (Sequence[float] | None, optional): Effective wavelength, in nm, of each channel of the radiance.
            Defaults to 550nm for grayscale images and to the RGB wavelengths otherwise.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        background_depth (float, optional): Depth from which pixels are considered to see the background.
            Defaults to 1e9.
        max_distance (float, optional): Distance up to which rays that see the background are marched, in meters.
            Defaults to 1000.0.
        device (torch.device | str | None, optional): Device to run on. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float64.

    Raises:
        ValueError: raised if the shapes of the inputs, wavelengths and lighting do not match.

    Returns:
        MediumResult: Radiance seen through the medium, alongside its transmittance, optical depth and in-scattering.
    """
    kwargs: dict[str, Any] = {"dtype": dtype, "device": device}
    radiance = radiance.to(**kwargs) if torch.is_tensor(radiance) else torch.tensor(np.asarray(radiance), **kwargs)
    depth = depth.to(**kwargs) if torch.is_tensor(depth) else torch.tensor(np.asarray(depth), **kwargs)
    radiance = radiance[..., None] if radiance.ndim == 2 else radiance
    depth = depth[..., 0] if depth.ndim == 3 else depth

    wavelengths = tuple(wavelengths or ((550.0,) if radiance.shape[-1] < 3 else RGB_WAVELENGTHS))
    if (n := len(wavelengths)) > radiance.shape[-1]:
        raise ValueError(f"Got {n} wavelengths for a radiance with only {radiance.shape[-1]} channels.")

    origin, directions, scale = camera_rays(camera, transform_matrix, **kwargs)
    if depth.shape != directions.shape[:2] or radiance.shape[:2] != directions.shape[:2]:
        raise ValueError("Radiance and depth do not match the camera's resolution.")

    background = ~torch.isfinite(depth) | (depth >= background_depth)
    distance = torch.where(background, torch.full_like(depth, max_distance), depth * scale)
    beta = torch.as_tensor(medium.extinction_at(wavelengths), **kwargs)
    ambient = medium.albedo * _per_channel(lighting.ambient, n, "ambient", **kwargs)
    sky = medium.albedo * _per_channel(lighting.sky, n, "sky", **kwargs)
    lit_by_sky = bool((sky != 0).any())
    # When extinction doesn't depend on wavelength, as for fog, skylight is attenuated the same in every channel
    gray = bool((beta == beta[0]).all())

    suns = []
    for sun in lighting.suns:
        towards_sun = torch.as_tensor(sun.direction, **kwargs)
        suns.append((towards_sun / towards_sun.norm(), _per_channel(sun.irradiance, n, "sun irradiance", **kwargs)))

    def march(directions: torch.Tensor, distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Light which is the same at every sample of a ray
        constant = ambient.expand(*directions.shape[:-1], n).clone()
        if lit_by_sky:
            sky_elevations, sky_weights = sky_quadrature(directions, medium.anisotropy)
            if not medium.sun_attenuation:
                constant = constant + sky * sky_weights.sum(dim=-1, keepdim=True)
        sunlight = []
        for towards_sun, irradiance in suns:
            phase = henyey_greenstein(directions @ towards_sun, medium.anisotropy)[..., None]
            sunlight.append((towards_sun, medium.albedo * phase * irradiance))

        step = (distance / steps)[..., None]
        transmittance = torch.ones_like(constant)
        optical_depth = torch.zeros_like(constant)
        inscatter = torch.zeros_like(constant)

        for i in range(steps):
            points = origin + (i + 0.5) * step * directions
            sigma = density(medium, points, time)[..., None] * beta
            source = constant.clone()

            for towards_sun, weight in sunlight:
                if medium.sun_attenuation:
                    weight = weight * _sun_transmittance(points, towards_sun, medium, beta, shadow_steps, time)
                source = source + weight

            if lit_by_sky and medium.sun_attenuation:
                # Each direction of the sky is attenuated by the fog above the sample, i.e. by density * falloff / ω_z
                fog = medium.components[0]
                assert isinstance(fog, HeightFog)
                above = (sigma[..., :1] if gray else sigma)[..., None, :] * fog.falloff
                elevations = sky_elevations[..., None]
                attenuation = torch.exp(-above / elevations.clamp_min(1e-12)) * (elevations > 0)
                source = source + sky * (sky_weights[..., None] * attenuation).sum(dim=-2)

            step_transmittance = torch.exp(-sigma * step)
            inscatter = inscatter + transmittance * source * (1 - step_transmittance)
            optical_depth = optical_depth + sigma * step
            transmittance = transmittance * step_transmittance
        return transmittance, optical_depth, inscatter

    # Integrating skylight needs hundreds of values per pixel, so rays are marched in chunks to bound memory
    nodes = (_SKY_NODES if lit_by_sky else 1) * (1 if gray else n)
    transmittance, optical_depth, inscatter = _by_pixels(march, directions, distance, elements_per_pixel=nodes)

    return MediumResult(
        radiance=torch.cat([transmittance * radiance[..., :n] + inscatter, radiance[..., n:]], dim=-1),
        transmittance=transmittance,
        optical_depth=optical_depth,
        inscatter=inscatter,
    )
