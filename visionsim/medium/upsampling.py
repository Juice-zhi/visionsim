"""Add a participating medium to a frame at a lower resolution, and upsample it, which saves most of the time.

Most of the cost of :func:`apply_medium <visionsim.medium.render.apply_medium>` grows with the number of pixels: tracing
shadows along each ray, integrating lamps, halos and reflected light along it, and lighting each surface. The light
that the medium scatters is smooth across the image, though, except where the depth of the scene jumps, so that it can
be computed along the rays of a fraction of the pixels and interpolated in between.

The medium is added along the rays of every ``factor``-th pixel along each axis, exactly those of the full frame, see
:func:`low_resolution_camera`. The light scattered towards each ray is ``J · (1 - T)``, where ``T`` is the
transmittance along the ray and ``J`` the mean light scattered per unit of optical depth, weighted by the
transmittance. ``1 - T`` changes abruptly at the edges of objects and is cheap and exact at every pixel, while ``J``,
which only depends on where along the ray light is scattered, is nearly continuous across them: ``J`` is upsampled and
multiplied by the exact ``1 - T`` of each pixel. The ratio by which the medium scales the light of surfaces is
upsampled too. Both are interpolated bilinearly between the four surrounding samples, weighted by how close their
depths are (joint bilateral upsampling, Kopf et al., "Joint Bilateral Upsampling", 2007), and, for surfaces, how
close their normals are, falling back on the sample of the closest depth where none is close. The transmittance and
the optical depth are exact at every pixel, while the light scattered per unit of optical depth and the ratio by which
the medium scales the light of surfaces are interpolated, which is exact where they vary linearly across the image and
the four samples lie at the depth of the pixel (and, for surfaces, have its normal), but blurs the edges of the shadows
that objects cast onto the medium, and the light of lamps right around them, over ``factor`` pixels.

Given the light that the surface of each pixel received, measured by its albedo or its irradiance (see
:func:`apply_medium <visionsim.medium.render.apply_medium>`), the ratio by which the medium scales the light of a
surface grows without bound where the surface received little light, such as deep in the shadow of a lamp, which the
medium's glow lights, while the light of the glow it reflects doesn't. Interpolated, the large ratio of a sample in the
shadow would multiply the light of the lit pixels next to it. The irradiance of surfaces with and without the medium
is upsampled instead, guided by the light received too, so that pixels take the samples on their side of the edges of
shadows, and the ratio is formed at every pixel from them and from the light that its own surface received, with the
mean irradiances of the frame's surfaces, as at the samples, see :func:`surface_attenuation
<visionsim.medium.surfaces.surface_attenuation>`: the light of each surface then stays as bounded as the irradiance of
the samples.

In the scenes of ``examples/medium`` (frames 255 and 345), at 1/4 of the resolution of 320x180 frames, the light
scattered by the fog differs from that of every pixel by 0.6% (relative L1) by day, and by 1.1% at night, where the
cube's shadow cuts the light of a lamp, and whole frames by 0.4% and 0.65%; at 1/2, by 0.45% and 0.25%. At 1/4 of
1920x1080 (frame 255, whose depth was rendered at that resolution), as the same edges cover a smaller share of the
pixels, the light of the fog differs by 0.1% at night, and by 0.4% by day, most of which comes from the light that
objects reflect, as it is gathered at the mean position of the samples in each cell of a grid, which moves with the
resolution (0.18% without it). This is an order of magnitude below the error of the model against Cycles, which it
barely changes: by 0.04 percentage points for whole frames by day, and by 0.09 at night, at 1/4 of 320x180, and by 0.09
points for surfaces, both by day and at night. Given the irradiance of each pixel, from Blender's diffuse light passes,
it changes the error of surfaces by 0.02 points by day and 0.01 at night, and with the night's light of surfaces in the
lamp's shadow cut a hundredfold, as if they were much darker, the light of the lit pixels next to the shadow stays
within 37% of that at every pixel, against up to 1500 times brighter when the ratio itself is interpolated, and that of
all surfaces within 0.24%, against 0.59% without the light measured.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from visionsim.medium.model import Effects, Lighting, Medium
from visionsim.medium.optics import optical_depth
from visionsim.medium.render import RGB_WAVELENGTHS, MediumResult, _measured_irradiance, apply_medium, camera_rays
from visionsim.medium.surfaces import surface_attenuation

_DEPTH_SIGMA = 0.05
"""Standard deviation, in natural log of depth, of the weights of samples by how close their depth is to the pixel's"""
_NORMAL_SIGMA = 0.1
"""Scale, in one minus the cosine between normals, of the weights of samples by how close their normal is"""
_LIGHT_SIGMA = 0.5
"""Standard deviation, in natural log of the light that surfaces received, of the weights of samples by how close it is
to the pixel's: the edges of shadows cut it by several times, while it varies much less between samples on either side"""


def low_resolution_camera(camera: Mapping[str, Any], factor: int) -> dict[str, Any]:
    """Intrinsics of a camera with ``factor`` times fewer pixels along each axis, whose rays are those through the
    pixels ``(factor · i + factor // 2, factor · j + factor // 2)`` of the given camera.

    Args:
        camera (Mapping[str, Any]): Camera intrinsics, with keys "fl_x", "fl_y", "cx", "cy", "w" and "h".
        factor (int): Ratio of the resolutions along each axis.

    Returns:
        dict[str, Any]: Intrinsics of the camera with fewer pixels, with the other keys of ``camera`` unchanged.
    """
    # Pixel i of the small camera, at i + 1/2, sees what pixel factor * i + factor // 2 of the full one does, at
    # factor * i + factor // 2 + 1/2
    shift = factor // 2 + 0.5 - factor / 2
    return dict(camera) | {
        "fl_x": camera["fl_x"] / factor,
        "fl_y": camera["fl_y"] / factor,
        "cx": (camera["cx"] - shift) / factor,
        "cy": (camera["cy"] - shift) / factor,
        "w": int(camera["w"]) // factor,
        "h": int(camera["h"]) // factor,
    }


def subsample(image: torch.Tensor, factor: int) -> torch.Tensor:
    """Pixels ``(factor · i + factor // 2, factor · j + factor // 2)`` of an image of shape (h, w, ...), those that
    :func:`low_resolution_camera` sees."""
    h, w = image.shape[0] // factor, image.shape[1] // factor
    return image[factor // 2 :: factor, factor // 2 :: factor][:h, :w]


def joint_upsample(
    values: torch.Tensor,
    low_depth: torch.Tensor,
    depth: torch.Tensor,
    factor: int,
    low_normals: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    low_light: torch.Tensor | None = None,
    light: torch.Tensor | None = None,
) -> torch.Tensor:
    """Upsample values given at the pixels of :func:`subsample` to every pixel, bilinearly between the four
    surrounding samples, weighted by how close their depths (and normals, and the light their surfaces received, if
    given) are to the pixel's.

    Args:
        values (torch.Tensor): Values at the samples, of shape (h', w', c).
        low_depth (torch.Tensor): Depth at the samples, positive and finite, of shape (h', w').
        depth (torch.Tensor): Depth at every pixel, positive and finite, of shape (h, w).
        factor (int): Ratio of the resolutions along each axis.
        low_normals (torch.Tensor | None, optional): Unit normals at the samples, of shape (h', w', 3). Defaults to
            None, i.e. only depths guide the interpolation.
        normals (torch.Tensor | None, optional): Unit normals at every pixel, of shape (h, w, 3). Defaults to None.
        low_light (torch.Tensor | None, optional): Light that the surfaces of the samples received, such as their
            irradiance, of shape (h', w'), zero where it isn't known, which then doesn't weigh them. Defaults to None,
            i.e. it doesn't guide the interpolation.
        light (torch.Tensor | None, optional): Light that the surfaces of every pixel received, of shape (h, w).
            Defaults to None.

    Returns:
        torch.Tensor: Values at every pixel, of shape (h, w, c).
    """
    return _joint_upsample(
        values, low_depth, depth, factor, low_normals, normals, at_once=depth.is_cuda, low_light=low_light, light=light
    )


def _joint_upsample(
    values: torch.Tensor,
    low_depth: torch.Tensor,
    depth: torch.Tensor,
    factor: int,
    low_normals: torch.Tensor | None,
    normals: torch.Tensor | None,
    at_once: bool,
    low_light: torch.Tensor | None = None,
    light: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same as :func:`joint_upsample`, where the four samples around each pixel are weighted at once, along a new
    dimension, which launches fewer kernels on a GPU, or one after another, which moves less memory on the CPU. Both
    give the same result."""
    kwargs: dict[str, Any] = {"dtype": depth.dtype, "device": depth.device}
    rows, columns = low_depth.shape
    height, width = depth.shape
    # Position of each pixel in units of samples, which are at factor * i + factor // 2
    fy = ((torch.arange(height, **kwargs) - factor // 2) / factor).clamp(0, rows - 1)
    fx = ((torch.arange(width, **kwargs) - factor // 2) / factor).clamp(0, columns - 1)
    y0, x0 = fy.floor().clamp(max=max(rows - 2, 0)).long(), fx.floor().clamp(max=max(columns - 2, 0)).long()
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    log_depth, low_log = torch.log(depth), torch.log(low_depth)
    guided = low_normals is not None and normals is not None
    lit = low_light is not None and light is not None
    if lit:
        assert low_light is not None and light is not None
        # The log of the light received, which isn't a number where it isn't known
        low_lit = torch.where(low_light > 0, low_light, torch.nan).log()
        log_light = torch.where(light > 0, light, torch.nan).log()

    def light_gap(sampled: torch.Tensor) -> torch.Tensor:
        # How far apart the light received at the samples and the pixel is, nothing where it isn't known
        gap = (sampled - log_light).abs()
        return torch.where(torch.isnan(gap), torch.zeros_like(gap), gap)

    if at_once:
        # Index of the four samples around each pixel, along a new first dimension, from the sample at the lowest row
        # and column, then along columns, then rows
        yi = torch.stack([y0, (y0 + 1).clamp(max=rows - 1)])[:, None, :, None]
        xi = torch.stack([x0, (x0 + 1).clamp(max=columns - 1)])[None, :, None, :]
        index = (yi * columns + xi).reshape(4, height, width)
        gap = (low_log.reshape(-1)[index] - log_depth).abs()
        bilinear = torch.stack([1 - wy, wy])[:, None] * torch.stack([1 - wx, wx])[None, :]
        weight = bilinear.reshape(4, height, width) * torch.exp(-((gap / _DEPTH_SIGMA) ** 2) / 2)
        if guided:
            assert low_normals is not None and normals is not None
            cosine = (low_normals.reshape(-1, 3)[index] * normals).sum(dim=-1)
            weight, gap = weight * torch.exp(-(1 - cosine) / _NORMAL_SIGMA), gap + (1 - cosine)
        if lit:
            # As far apart as depths of the same number of standard deviations
            apart = light_gap(low_lit.reshape(-1)[index])
            weight, gap = (
                weight * torch.exp(-((apart / _LIGHT_SIGMA) ** 2) / 2),
                gap + apart * _DEPTH_SIGMA / _LIGHT_SIGMA,
            )
        samples = values.reshape(-1, values.shape[-1])[index]
        weighted = weight[..., None] * samples
        total, weights = (
            weighted[0] + weighted[1] + weighted[2] + weighted[3],
            weight[0] + weight[1] + weight[2] + weight[3],
        )
        # The first of the closest samples, where gaps that aren't numbers, from guides out of their domain, count as
        # the widest
        closest = torch.nan_to_num(gap, nan=torch.inf).argmin(dim=0)
        closest = closest[None, ..., None].expand(1, height, width, values.shape[-1])
        best = samples.gather(0, closest)[0]
    else:
        total, weights, best, best_gap = torch.zeros(height, width, values.shape[-1], **kwargs), 0, None, None
        for dy in (0, 1):
            for dx in (0, 1):
                yi = (y0 + dy).clamp(max=rows - 1)[:, None].expand(height, width)
                xi = (x0 + dx).clamp(max=columns - 1)[None, :].expand(height, width)
                gap = (low_log[yi, xi] - log_depth).abs()
                weight = (wy if dy else 1 - wy) * (wx if dx else 1 - wx) * torch.exp(-((gap / _DEPTH_SIGMA) ** 2) / 2)
                if guided:
                    assert low_normals is not None and normals is not None
                    cosine = (low_normals[yi, xi] * normals).sum(dim=-1)
                    weight, gap = weight * torch.exp(-(1 - cosine) / _NORMAL_SIGMA), gap + (1 - cosine)
                if lit:
                    apart = light_gap(low_lit[yi, xi])
                    weight = weight * torch.exp(-((apart / _LIGHT_SIGMA) ** 2) / 2)
                    gap = gap + apart * _DEPTH_SIGMA / _LIGHT_SIGMA
                sample = values[yi, xi]
                total, weights = total + weight[..., None] * sample, weights + weight
                gap = torch.nan_to_num(gap, nan=torch.inf)  # as when samples are weighted at once
                if best is None or best_gap is None:
                    best, best_gap = sample, gap
                else:
                    closer = gap < best_gap
                    best, best_gap = torch.where(closer[..., None], sample, best), torch.where(closer, gap, best_gap)
    assert best is not None and isinstance(weights, torch.Tensor)
    # Pixels whose samples all lie at other depths, e.g. thin objects seen between samples, take the closest one
    return torch.where((weights > 1e-3)[..., None], total / weights.clamp_min(1e-30)[..., None], best)


def apply_medium_upsampled(
    radiance: npt.ArrayLike | torch.Tensor,
    depth: npt.ArrayLike | torch.Tensor,
    camera: Mapping[str, Any],
    transform_matrix: npt.ArrayLike,
    medium: Medium,
    lighting: Lighting,
    factor: int = 4,
    normals: npt.ArrayLike | torch.Tensor | None = None,
    emission: npt.ArrayLike | torch.Tensor | None = None,
    effects: Effects | None = None,
    albedo: npt.ArrayLike | torch.Tensor | None = None,
    irradiance: npt.ArrayLike | torch.Tensor | None = None,
    **kwargs: Any,
) -> MediumResult:
    """Add a participating medium to a rendered frame along the rays of every ``factor``-th pixel along each axis,
    and upsample it to every pixel, see the module's description.

    Args:
        radiance (npt.ArrayLike | torch.Tensor): Linear radiance of the scene without the medium, of shape (h, w, c).
        depth (npt.ArrayLike | torch.Tensor): Depth of the scene, as given to :func:`apply_medium
            <visionsim.medium.render.apply_medium>`, of shape (h, w) or (h, w, 1).
        camera (Mapping[str, Any]): Camera intrinsics, see :func:`camera_rays <visionsim.medium.render.camera_rays>`.
        transform_matrix (npt.ArrayLike): Camera-to-world transform.
        medium (Medium): Participating medium.
        lighting (Lighting): Lighting of the medium.
        factor (int, optional): Ratio of the resolutions along each axis at which the medium is added. 1 adds it at
            every pixel, as :func:`apply_medium <visionsim.medium.render.apply_medium>`. Defaults to 4.
        normals (npt.ArrayLike | torch.Tensor | None, optional): Normals of the surfaces seen by each pixel, of shape
            (h, w, 3), see :func:`apply_medium <visionsim.medium.render.apply_medium>`. Defaults to None.
        emission (npt.ArrayLike | torch.Tensor | None, optional): Light that surfaces emit, of the same shape as
            ``radiance``. Defaults to None.
        effects (Effects | None, optional): Terms of the model to compute, see :class:`Effects
            <visionsim.medium.model.Effects>`. Defaults to None, i.e. every term that the inputs allow.
        albedo (npt.ArrayLike | torch.Tensor | None, optional): Albedo of the surfaces seen by each pixel, of shape
            (h, w, c) or (h, w, 1), from which the light that they received is measured, see :func:`apply_medium
            <visionsim.medium.render.apply_medium>`, at the samples and at every pixel, where the ratio by which the
            medium scales the light of surfaces is formed from their irradiance with and without the medium, which is
            upsampled, see the module's description. Defaults to None.
        irradiance (npt.ArrayLike | torch.Tensor | None, optional): Irradiance that the surfaces seen by each pixel
            received without the medium, of shape (h, w, c) or (h, w, 1), which takes precedence over ``albedo``, see
            :func:`apply_medium <visionsim.medium.render.apply_medium>`. Defaults to None.
        **kwargs: Further arguments of :func:`apply_medium <visionsim.medium.render.apply_medium>`, except
            ``shadows``, which are traced along the rays of every pixel.

    Raises:
        ValueError: raised if ``shadows`` are given while ``factor`` is greater than 1.

    Returns:
        MediumResult: Radiance seen through the medium at every pixel, alongside its transmittance, optical depth,
        in-scattering, and the ratio by which it scales the light of surfaces.
    """
    args = (camera, transform_matrix, medium, lighting)
    if factor <= 1:
        return apply_medium(
            radiance,
            depth,
            *args,
            normals=normals,
            emission=emission,
            effects=effects,
            albedo=albedo,
            irradiance=irradiance,
            **kwargs,
        )
    if kwargs.get("shadows") is not None:
        raise ValueError("Shadows traced along the rays of every pixel can't be reused at a lower resolution.")
    options: dict[str, Any] = {"dtype": kwargs.get("dtype", torch.float64), "device": kwargs.get("device")}

    def tensor(x: npt.ArrayLike | torch.Tensor) -> torch.Tensor:
        return x.to(**options) if torch.is_tensor(x) else torch.tensor(np.asarray(x), **options)

    radiance, depth = tensor(radiance), tensor(depth)
    radiance = radiance[..., None] if radiance.ndim == 2 else radiance
    depth = depth[..., 0] if depth.ndim == 3 else depth
    small = low_resolution_camera(camera, factor)
    low_normals = subsample(tensor(normals)[..., :3], factor) if normals is not None else None
    low_emission = subsample(tensor(emission), factor) if emission is not None else None

    def per_pixel(x: npt.ArrayLike | torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        values = tensor(x)
        return values[..., None] if values.ndim == 2 else values

    full_albedo, full_irradiance = per_pixel(albedo), per_pixel(irradiance)
    low = apply_medium(
        subsample(radiance, factor),
        subsample(depth, factor),
        small,
        *args[1:],
        normals=low_normals,
        emission=low_emission,
        effects=effects,
        albedo=subsample(full_albedo, factor) if full_albedo is not None else None,
        irradiance=subsample(full_irradiance, factor) if full_irradiance is not None else None,
        **kwargs,
    )

    # The exact transmittance of every pixel, as apply_medium computes it
    n = low.transmittance.shape[-1]
    wavelengths = tuple(kwargs.get("wavelengths") or ((550.0,) if radiance.shape[-1] < 3 else RGB_WAVELENGTHS))
    origin, directions, scale = camera_rays(camera, transform_matrix, **options)
    background_depth = kwargs.get("background_depth", 1e9)
    background = ~torch.isfinite(depth) | (depth >= background_depth)
    distance = torch.where(background, torch.full_like(depth, torch.inf), depth * scale)
    beta = torch.as_tensor(medium.extinction_at(wavelengths), **options)
    tau_rel = optical_depth(medium, origin, directions, distance, time=kwargs.get("time", 0.0))[..., None]
    tau = torch.where(beta > 0, tau_rel * beta, torch.zeros_like(tau_rel))
    transmittance = torch.exp(-tau)

    # Depths guide the interpolation; the background is taken at the depth from which pixels see it
    guide = torch.where(background, torch.full_like(depth, background_depth), depth.clamp_min(1e-6))
    low_guide = subsample(guide, factor)
    scattered_fraction = (1 - low.transmittance).clamp_min(1e-12)
    mean_source = torch.where(
        1 - low.transmittance > 1e-9, low.inscatter / scattered_fraction, torch.zeros_like(low.inscatter)
    )
    inscatter = joint_upsample(mean_source, low_guide, guide, factor) * -torch.expm1(-tau)

    surfaces, illumination = radiance[..., :n], None
    clear_irradiance = surface_irradiance = None
    if low.illumination is not None:
        full_normals = None
        if normals is not None and low_normals is not None:
            full_normals = tensor(normals)[..., :3]
            full_normals = full_normals / full_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            low_normals = low_normals / low_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        illumination = joint_upsample(low.illumination, low_guide, guide, factor, low_normals, full_normals)
        emitted = 0.0 if emission is None else tensor(emission)[..., :n]
        assert low.clear_irradiance is not None and low.surface_irradiance is not None
        given = {"occlusion": kwargs.get("occlusion") is not None, "albedo": albedo is not None}
        resolved = (effects or Effects()).resolve(
            medium, lighting, **given, normals=True, irradiance=irradiance is not None
        )
        low_seen = ~subsample(background, factor)
        measured = None
        if resolved.received_irradiance and bool(low_seen.any()):
            measured = _measured_irradiance(lighting, surfaces - emitted, full_albedo, full_irradiance)
        # Samples whose surfaces received as much light as the pixel's, on the same side of the edges of shadows, guide
        # the irradiances, where it is measured
        light = None
        if measured is not None:
            light = torch.where(measured[1], measured[0], torch.zeros_like(measured[0])).mean(dim=-1)
        lit = joint_upsample(
            torch.cat([low.clear_irradiance, low.surface_irradiance], dim=-1),
            low_guide,
            guide,
            factor,
            low_normals,
            full_normals,
            subsample(light, factor) if light is not None else None,
            light,
        )
        clear_irradiance, surface_irradiance = lit[..., :n], lit[..., n:]
        if measured is not None:
            # The ratio of surfaces whose light received is measured grows without bound where they received little,
            # such as in the shadow of a lamp, unlike the light of the glow they reflect: it is formed at every pixel
            # from the irradiances with and without the medium, which are upsampled, and the light that the pixel's
            # own surface received, with the mean irradiances of the frame's surfaces, as at the samples
            received, known = measured
            seen = ~background
            frame = (low.clear_irradiance[low_seen].mean(dim=0), low.surface_irradiance[low_seen].mean(dim=0))
            illumination = illumination.clone()
            illumination[seen] = surface_attenuation(
                clear_irradiance[seen],
                surface_irradiance[seen],
                received=received[seen],
                measured=known[seen] if resolved.measured_indirect else None,
                frame=frame,
            )
        surfaces = illumination * (surfaces - emitted) + emitted
    return MediumResult(
        radiance=torch.cat([transmittance * surfaces + inscatter, radiance[..., n:]], dim=-1),
        transmittance=transmittance,
        optical_depth=tau,
        inscatter=inscatter,
        illumination=illumination,
        surface_irradiance=surface_irradiance,
        clear_irradiance=clear_irradiance,
    )
