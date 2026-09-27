from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from visionsim.medium.model import HeightFog, Lighting, Medium
from visionsim.medium.optics import height_fog_sun_inscatter, henyey_greenstein, optical_depth

RGB_WAVELENGTHS: tuple[float, float, float] = (610.0, 550.0, 465.0)
"""Approximate effective wavelengths, in nm, of the red, green and blue channels of linear sRGB images"""


class MediumResult(NamedTuple):
    """Radiance seen through a participating medium, alongside ground truth quantities, all of shape (h, w, c)."""

    radiance: torch.Tensor
    """radiance reaching the camera, i.e. ``transmittance * surface_radiance + inscatter``, where any channels
    beyond the modeled wavelengths (such as alpha) are passed through unchanged"""
    transmittance: torch.Tensor
    """fraction of the surface radiance that reaches the camera"""
    optical_depth: torch.Tensor
    """optical depth between the camera and the surface, i.e. ``-ln(transmittance)``"""
    inscatter: torch.Tensor
    """radiance scattered towards the camera by the medium"""


def camera_rays(
    camera: Mapping[str, Any],
    transform_matrix: npt.ArrayLike,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rays going through the center of each pixel of a pinhole camera.

    Args:
        camera (Mapping[str, Any]): Camera intrinsics, with keys "fl_x", "fl_y", "cx", "cy", "w" and "h".
        transform_matrix (npt.ArrayLike): Camera-to-world transform, following Blender's convention where
            the camera looks down its -Z axis with +Y pointing up.
        device (torch.device | str | None, optional): Device on which to create the rays. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float64.

    Raises:
        NotImplementedError: raised if the camera has lens distortion.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The origin of the rays, of shape (3,), their unit
        world-space directions, of shape (h, w, 3), and the ratio between the distance along each ray and
        the depth (i.e. the distance along the optical axis), of shape (h, w).
    """
    if any(camera.get(k) for k in ("k1", "k2", "k3", "k4", "p1", "p2")):
        raise NotImplementedError("Cameras with lens distortion are not yet supported.")

    pose = torch.as_tensor(np.asarray(transform_matrix, dtype=float), dtype=dtype, device=device)
    rows = torch.arange(int(camera["h"]), dtype=dtype, device=device) + 0.5
    cols = torch.arange(int(camera["w"]), dtype=dtype, device=device) + 0.5
    v, u = torch.meshgrid(rows, cols, indexing="ij")

    local = torch.stack(
        [(u - camera["cx"]) / camera["fl_x"], -(v - camera["cy"]) / camera["fl_y"], -torch.ones_like(u)], dim=-1
    )
    scale = local.norm(dim=-1)
    directions = local @ pose[:3, :3].T
    return pose[:3, 3], directions / directions.norm(dim=-1, keepdim=True), scale


def _per_channel(values: Sequence[float], n: int, name: str, **kwargs) -> torch.Tensor:
    if len(values) not in (1, n):
        raise ValueError(f"Expected {name} to have 1 or {n} values (one per wavelength), got {len(values)}.")
    return torch.as_tensor([values[0]] * n if len(values) == 1 else list(values), **kwargs)


def apply_medium(
    radiance: npt.ArrayLike | torch.Tensor,
    depth: npt.ArrayLike | torch.Tensor,
    camera: Mapping[str, Any],
    transform_matrix: npt.ArrayLike,
    medium: Medium,
    lighting: Lighting,
    wavelengths: Sequence[float] | None = None,
    time: float = 0.0,
    background_depth: float = 1e9,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> MediumResult:
    """Add a participating medium to a rendered frame, in closed form.

    The radiance of each surface is attenuated by the medium's transmittance, and light from the environment
    and from suns is scattered towards the camera (single scattering). Every quantity is computed in closed form,
    without any ray marching, so the result is exact and noise-free for the given model. This makes it suitable
    as the common input of all sensor emulators, which then only add their own noise.

    Note:
        Shadows cast onto the medium (light shafts), point lights, multiple scattering, and the dimming of
        surfaces lit through the medium are not yet modeled. The environment is assumed to light the
        medium uniformly from all directions.

    Args:
        radiance (npt.ArrayLike | torch.Tensor): Linear radiance of the scene without the medium, of shape (h, w, c).
        depth (npt.ArrayLike | torch.Tensor): Depth of the scene, i.e. the distance along the camera's optical axis
            as saved by Blender, of shape (h, w) or (h, w, 1).
        camera (Mapping[str, Any]): Camera intrinsics, see :func:`camera_rays`.
        transform_matrix (npt.ArrayLike): Camera-to-world transform, see :func:`camera_rays`.
        medium (Medium): Participating medium.
        lighting (Lighting): Lighting of the medium, whose colors have one value per wavelength, or a single one.
        wavelengths (Sequence[float] | None, optional): Effective wavelength, in nm, of each channel of the radiance.
            Further channels, such as alpha, are left untouched. Defaults to 550nm for grayscale images and to
            :data:`RGB_WAVELENGTHS` otherwise.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        background_depth (float, optional): Depth from which pixels are considered to see the background,
            in which case they are integrated to infinity. Defaults to 1e9.
        device (torch.device | str | None, optional): Device to run on. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float64.

    Raises:
        ValueError: raised if the shapes of the inputs, wavelengths and lighting do not match.

    Returns:
        MediumResult: Radiance seen through the medium, alongside its transmittance, optical depth and in-scattering.
    """
    kwargs: dict[str, Any] = {"dtype": dtype, "device": device}
    radiance = torch.as_tensor(np.asarray(radiance) if not torch.is_tensor(radiance) else radiance, **kwargs)
    depth = torch.as_tensor(np.asarray(depth) if not torch.is_tensor(depth) else depth, **kwargs)
    radiance = radiance[..., None] if radiance.ndim == 2 else radiance
    depth = depth[..., 0] if depth.ndim == 3 else depth

    wavelengths = tuple(wavelengths or ((550.0,) if radiance.shape[-1] < 3 else RGB_WAVELENGTHS))
    if (n := len(wavelengths)) > radiance.shape[-1]:
        raise ValueError(f"Got {n} wavelengths for a radiance with only {radiance.shape[-1]} channels.")

    origin, directions, scale = camera_rays(camera, transform_matrix, **kwargs)
    if depth.shape != directions.shape[:2] or radiance.shape[:2] != directions.shape[:2]:
        raise ValueError(
            f"Radiance {tuple(radiance.shape)} and depth {tuple(depth.shape)} do not match the camera's "
            f"resolution ({camera['h']}, {camera['w']})."
        )

    background = ~torch.isfinite(depth) | (depth >= background_depth)
    distance = torch.where(background, torch.full_like(depth, torch.inf), depth * scale)

    # Scale relative optical depth by the extinction at each wavelength, taking care of infinite rays in void
    beta = torch.as_tensor(medium.extinction_at(wavelengths), **kwargs)
    tau_rel = optical_depth(medium, origin, directions, distance, time=time)[..., None]
    tau = torch.where(beta > 0, tau_rel * beta, torch.zeros_like(tau_rel))
    transmittance = torch.exp(-tau)

    # Sources which are constant along rays result in exactly `J * (1 - T)`, whatever the density
    source = medium.albedo * _per_channel(lighting.ambient, n, "ambient", **kwargs)
    source = source.expand(*tau.shape).clone()
    attenuated_suns = []

    for sun in lighting.suns:
        towards_sun = torch.as_tensor(sun.direction, **kwargs)
        towards_sun = towards_sun / towards_sun.norm()
        phase = henyey_greenstein(directions @ towards_sun, medium.anisotropy)[..., None]
        irradiance = _per_channel(sun.irradiance, n, "sun irradiance", **kwargs)

        if medium.sun_attenuation:
            attenuated_suns.append((float(towards_sun[2]), medium.albedo * phase * irradiance))
        else:
            source = source + medium.albedo * phase * irradiance
    inscatter = source * -torch.expm1(-tau)

    if attenuated_suns:
        fog = medium.components[0]
        assert isinstance(fog, HeightFog)
        origin_extinction = beta * fog.density * torch.exp(-(origin[2] - fog.base_height) / fog.falloff)

        for sun_z, weight in attenuated_suns:
            inscatter = inscatter + weight * height_fog_sun_inscatter(
                tau, origin_extinction, fog.falloff, directions[..., 2:3], distance[..., None], sun_z
            )

    return MediumResult(
        radiance=torch.cat([transmittance * radiance[..., :n] + inscatter, radiance[..., n:]], dim=-1),
        transmittance=transmittance,
        optical_depth=tau,
        inscatter=inscatter,
    )
