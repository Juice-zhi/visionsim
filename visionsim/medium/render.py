from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from visionsim.medium.halos import cached_lamp_halo, halo_glow, halo_inscatter, halo_threshold
from visionsim.medium.lights import (
    AREA_LEVELS,
    AreaLevel,
    Emitter,
    area_weights,
    closest_distances,
    emitters,
    level_weights,
    patch_emitters,
    patch_rectangle,
)
from visionsim.medium.model import AreaLight, Effects, EmissiveSurface, HeightFog, Lamp, Lighting, Medium, SpotLight
from visionsim.medium.occlusion import (
    Occlusion,
    ShadowMaps,
    Shadows,
    cell_profile,
    lamp_map_index,
    lamp_shadow,
    maps_to,
    object_reflections,
    occlusion_to,
    shade,
    shadows_to,
    trace,
)
from visionsim.medium.optics import (
    _per_channel,
    height_fog_sun_inscatter,
    henyey_greenstein,
    light_angles,
    optical_depth,
    point_light_inscatter,
    sky_quadrature,
)
from visionsim.medium.reflections import (
    GROUND_CLEARANCE,
    cached_lamp_reflection,
    reflection_inscatter,
    scene_surfaces,
    scene_surfels,
)
from visionsim.medium.scattering import (
    TABLE_SIZE,
    ElevationTable,
    cached_camera_tables,
    cached_multiple_scattering,
    environment_lit,
    lookup,
    relative_azimuths,
)
from visionsim.medium.surfaces import Glow, ground_shares, object_radiance, surface_attenuation, surface_irradiance

RGB_WAVELENGTHS: tuple[float, float, float] = (610.0, 550.0, 465.0)
"""Approximate effective wavelengths, in nm, of the red, green and blue channels of linear sRGB images"""

_SKY_NODES = 3 * 16 * 24
"""Number of directions over which skylight is integrated for each pixel, at the default resolution of
:func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`"""
_CHUNK_ELEMENTS = 4e7
"""Rough bound on the number of elements of intermediate tensors when processing pixels in chunks"""
_POINT_LIGHT_NODES = 32
"""Number of quadrature nodes along each ray used to integrate light from point and spot lights"""
_LAMP_SHADOW_SAMPLES = 256
"""Number of samples of the visibility of lamps along each ray, see :func:`lamp_shadow
<visionsim.medium.occlusion.lamp_shadow>`"""


class MediumResult(NamedTuple):
    """Radiance seen through a participating medium, alongside ground truth quantities, all of shape (h, w, c)."""

    radiance: torch.Tensor
    """radiance reaching the camera, i.e. ``transmittance * illumination * surface_radiance + inscatter``, where any
    channels beyond the modeled wavelengths (such as alpha) are passed through unchanged"""
    transmittance: torch.Tensor
    """fraction of the surface radiance that reaches the camera"""
    optical_depth: torch.Tensor
    """optical depth between the camera and the surface, i.e. ``-ln(transmittance)``"""
    inscatter: torch.Tensor
    """radiance scattered towards the camera by the medium"""
    illumination: torch.Tensor | None = None
    """ratio by which the medium scales the light of surfaces lit through it, see :mod:`visionsim.medium.surfaces`,
    or None if surfaces are left as they were rendered without the medium. Given the light that each surface received,
    see ``albedo`` and ``irradiance`` of :func:`apply_medium`, it can be large where surfaces received almost no light,
    such as deep in the shadow of a lamp, which the glow of the medium lights, while the light of these surfaces, which
    it scales, stays as bounded as the glow"""
    surface_irradiance: torch.Tensor | None = None
    """irradiance of the surfaces seen by each pixel through the medium, from suns, the sky and lamps, which it dims, and
    from its glow, as modeled from the position and normal of each surface, see :func:`surface_irradiance
    <visionsim.medium.surfaces.surface_irradiance>`, zero where pixels see the background, or None as
    ``illumination``. It leaves out the light that reaches surfaces from other surfaces, see :func:`surface_attenuation
    <visionsim.medium.surfaces.surface_attenuation>`"""
    clear_irradiance: torch.Tensor | None = None
    """irradiance of the same surfaces from the same lights without the medium, or None as ``illumination``, which is the
    ratio of both once the light that reaches surfaces from other surfaces is added to each"""


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


def _by_pixels(fn: Callable[..., Any], *images: torch.Tensor, elements_per_pixel: int) -> Any:
    # Integrating skylight needs hundreds of values per pixel, which exhausts GPU memory at high resolutions unless
    # pixels are processed in chunks. Images are of shape (h, w, ...), and `fn` returns tensors, or tuples of tensors,
    # with one row per pixel of the chunk, which are stitched back into images
    h, w = images[0].shape[:2]
    flat = [image.reshape(h * w, *image.shape[2:]) for image in images]
    size = max(1, int(_CHUNK_ELEMENTS // elements_per_pixel))
    chunks = [fn(*(image[i : i + size] for image in flat)) for i in range(0, h * w, size)]

    def stitch(parts: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(parts)).reshape(h, w, *parts[0].shape[1:])

    return tuple(map(stitch, zip(*chunks))) if isinstance(chunks[0], tuple) else stitch(chunks)


def lamp_inscatter(
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    beta: torch.Tensor,
    lamp: Lamp,
    time: float = 0.0,
    area_levels: tuple[AreaLevel, ...] = AREA_LEVELS,
    maps: ShadowMaps | None = None,
    map_tolerance: float = 0.01,
    visibility_samples: int = _LAMP_SHADOW_SAMPLES,
) -> torch.Tensor:
    """Light of a lamp scattered once towards the camera along rays, before it is scaled by the medium's albedo.

    Lamps are made of point emitters, see :mod:`visionsim.medium.lights`, whose light is integrated with
    :func:`point_light_inscatter <visionsim.medium.optics.point_light_inscatter>`. Rays that pass far from area lights
    see them as a single emitter, and those that pass closer to them as finer grids of emitters, see
    :func:`area_weights <visionsim.medium.lights.area_weights>`. Given maps of lamps, objects cast the shadows of the
    lamp's center onto the medium, see :func:`lamp_shadow <visionsim.medium.occlusion.lamp_shadow>`.

    Args:
        medium (Medium): Participating medium.
        origin (torch.Tensor): Ray origin, of shape (3,), or one per ray, of shape (r, 3).
        directions (torch.Tensor): Unit ray directions, of shape (r, 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (r,), can be infinite.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        lamp (Lamp): Point, spot or area light, or emissive surface.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        area_levels (tuple[AreaLevel, ...], optional): Grids of emitters through which rays that pass close to area
            lights see them. Defaults to :data:`AREA_LEVELS <visionsim.medium.lights.AREA_LEVELS>`.
        maps (ShadowMaps | None, optional): Maps of lamps, see :attr:`Occlusion.lamp_maps
            <visionsim.medium.occlusion.Occlusion.lamp_maps>`, among which the lamp's map is found by its position.
            Defaults to None, i.e. the lamp's light is not occluded.
        map_tolerance (float, optional): Distance, in meters, within which the lamp casts the shadows of a map rendered
            at another position, see :attr:`Occlusion.lamp_tolerance
            <visionsim.medium.occlusion.Occlusion.lamp_tolerance>`. Defaults to 0.01.
        visibility_samples (int, optional): Number of samples of the visibility of the lamp along each ray, see
            :func:`lamp_shadow <visionsim.medium.occlusion.lamp_shadow>`. Defaults to 256.

    Returns:
        torch.Tensor: Radiance scattered towards the camera, for an albedo of one, of shape (r, c).
    """
    n = len(beta)
    kwargs: dict[str, Any] = {"dtype": directions.dtype, "device": directions.device}
    # Maps of area lights are rendered just in front of them
    center = torch.as_tensor(lamp.position, **kwargs)
    shifted = center + 1e-3 * torch.as_tensor(lamp.direction, **kwargs) if isinstance(lamp, AreaLight) else center
    index = lamp_map_index(maps, shifted, map_tolerance)

    def take(origin: torch.Tensor, rays: torch.Tensor | slice) -> torch.Tensor:
        # Origins are either shared by all rays, or one per ray, which follow the rays
        return origin[rays] if origin.ndim > 1 else origin

    def scatter(
        group: list[Emitter],
        nodes: int,
        origin: torch.Tensor,
        directions: torch.Tensor,
        distance: torch.Tensor,
        weights: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        # Light of a group of emitters along rays, each optionally weighted per ray. Emitters that only differ by their
        # position and intensity, such as those of a grid, are integrated at once along copies of the rays
        batches: dict[tuple, list[int]] = {}
        for e, emitter in enumerate(group):
            weighted = weights is not None and weights[e] is not None
            shared = (emitter.radius, id(emitter.axis), emitter.cone, id(emitter.profile), id(emitter.falloff))
            batches.setdefault((*shared, id(emitter.pattern), weighted), []).append(e)
        size = max(1, int(_CHUNK_ELEMENTS // max(nodes * (8 + 2 * n), 12 * visibility_samples * (index is not None))))
        parts = []
        for i in range(0, len(directions), size):
            chunk = slice(i, i + size)
            starts, rays, lengths = take(origin, chunk), directions[chunk], distance[chunk]
            shadow, seen = None, None
            if maps is not None and index is not None:
                shadow = lamp_shadow(maps, index, starts, rays, lengths, samples=visibility_samples)
                # Rays entirely in the lamp's shadow, such as those of other rooms, receive none of its light
                if not bool(shadow.lit.all()):
                    seen = shadow.lit.nonzero().squeeze(-1)
                    shadow, rays, lengths = shadow.select(seen), rays[seen], lengths[seen]
                    starts = take(starts, seen)
            total = rays.new_zeros(len(rays), n)
            for batch in batches.values() if len(rays) else []:
                ray_weights = None
                if weights is not None and weights[batch[0]] is not None:
                    ray_weights = torch.stack([w for e in batch if (w := weights[e]) is not None])[:, chunk]
                    ray_weights = ray_weights if seen is None else ray_weights[:, seen]
                    used = (ray_weights > 0).any(dim=1)
                    batch, ray_weights = [e for e, u in zip(batch, used.tolist()) if u], ray_weights[used]
                count = max(1, int(_CHUNK_ELEMENTS // (len(rays) * nodes * (8 + 2 * n) * (1 + (shadow is not None)))))
                for j in range(0, len(batch), count):
                    part, emitter = batch[j : j + count], group[batch[j]]
                    copies = len(part)
                    positions = torch.stack([group[e].position for e in part]).repeat_interleave(len(rays), dim=0)
                    light = point_light_inscatter(
                        medium,
                        starts.repeat(copies, 1) if starts.ndim > 1 else starts,
                        rays.repeat(copies, 1),
                        lengths.repeat(copies),
                        positions if copies > 1 else emitter.position,
                        beta,
                        radius=emitter.radius,
                        time=time,
                        nodes=nodes,
                        axis=emitter.axis,
                        cone=emitter.cone,
                        profile=emitter.profile,
                        falloff=emitter.falloff,
                        shadow=shadow.select(torch.arange(len(rays), device=rays.device).repeat(copies)).average
                        if shadow is not None
                        else None,
                        pattern=emitter.pattern,
                    ).reshape(copies, len(rays), n)
                    if ray_weights is not None:
                        light = light * ray_weights[j : j + count, :, None]
                    total = total + (torch.stack([group[e].intensity for e in part])[:, None, :] * light).sum(dim=0)
            if seen is not None:
                total = total.new_zeros(len(directions[i : i + size]), n).index_copy(0, seen, total)
            parts.append(total)
        return torch.cat(parts)

    result = directions.new_zeros(len(directions), n)
    if isinstance(lamp, EmissiveSurface):
        # Patches are seen as a single emitter from afar, and flat ones as finer grids by rays that pass close to them,
        # as area lights are
        singles: list[Emitter] = []
        single_weights: list[torch.Tensor | None] = []
        for k in range(len(lamp.areas)):
            single = patch_emitters(lamp, k, n, **kwargs)
            rectangle = patch_rectangle(lamp, k) if area_levels and single else None
            if rectangle is None:
                singles, single_weights = singles + single, single_weights + [None] * len(single)
                continue
            middle = torch.as_tensor(lamp.positions[k], **kwargs)
            closest = closest_distances(origin, directions, distance, middle)
            weights = level_weights(closest / (math.hypot(*rectangle[2]) / 2), area_levels)
            singles, single_weights = singles + single, single_weights + [weights[-1]] * len(single)
            for level, weight in zip(area_levels, weights):
                if len(rays := (weight > 0).nonzero().squeeze(-1)):
                    group = patch_emitters(lamp, k, n, level.samples, **kwargs)
                    light = scatter(group, level.nodes, take(origin, rays), directions[rays], distance[rays])
                    result = result.index_add(0, rays, weight[rays, None] * light)
        if singles:
            result = result + scatter(singles, _POINT_LIGHT_NODES, origin, directions, distance, single_weights)
        return result

    if not isinstance(lamp, (SpotLight, AreaLight)):
        lit = torch.ones_like(distance, dtype=torch.bool)
    else:
        # Rays that never enter the cone of a spot, or that stay behind the plane of an area light, aren't lit at all
        axis = torch.as_tensor(lamp.direction, **kwargs)
        cone = max(math.cos(lamp.angle / 2), 0.0) if isinstance(lamp, SpotLight) else 0.0
        angles = light_angles(origin, directions, distance, center, axis=axis / axis.norm(), cone=cone)
        lit = angles.end > angles.start

    if not isinstance(lamp, AreaLight):
        rays = lit.nonzero().squeeze(-1)
        if len(rays):
            group = emitters(lamp, n, **kwargs)
            light = scatter(group, _POINT_LIGHT_NODES, take(origin, rays), directions[rays], distance[rays])
            result = result.index_copy(0, rays, light)
        return result

    weights = area_weights(lamp, closest_distances(origin, directions, distance, center), area_levels)
    grids = [(level.samples, level.nodes) for level in area_levels] + [(1, _POINT_LIGHT_NODES)]
    for (samples, nodes), weight in zip(grids, weights):
        if len(rays := ((weight > 0) & lit).nonzero().squeeze(-1)):
            group = emitters(lamp, n, samples, **kwargs)
            light = scatter(group, nodes, take(origin, rays), directions[rays], distance[rays])
            result = result.index_add(0, rays, weight[rays, None] * light)
    return result


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
    table_size: tuple[int, int] = TABLE_SIZE,
    occlusion: Occlusion | None = None,
    shadows: Shadows | None = None,
    shadow_step: float = 8.0,
    shadow_samples: int = 8,
    effects: Effects | None = None,
    normals: npt.ArrayLike | torch.Tensor | None = None,
    normals_space: Literal["world", "camera"] = "world",
    emission: npt.ArrayLike | torch.Tensor | None = None,
    albedo: npt.ArrayLike | torch.Tensor | None = None,
    irradiance: npt.ArrayLike | torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> MediumResult:
    """Add a participating medium to a rendered frame, in closed form.

    The radiance of each surface is attenuated by the medium's transmittance, and light from the environment,
    suns and lamps is scattered towards the camera. Sunlight is integrated in closed form, without any ray
    marching, light that reaches the medium from many directions (from the sky, from the ground, and light scattered
    more than once) is integrated once per frame along rays of every elevation, and for suns' light scattered more than
    once of every azimuth from the sun's, and interpolated, see :mod:`visionsim.medium.scattering`, and light from
    point, spot and area lights and emissive surfaces is integrated with a quadrature that removes its singularity,
    see :func:`lamp_inscatter`. When shadow maps of the scene are given, objects cast shadows onto the medium (light
    shafts) and hide part of the sky from it, see
    :mod:`visionsim.medium.occlusion`. Given the normals of surfaces, surfaces are also lit through the medium, which
    dims their light on its way to them while its glow lights them, see :mod:`visionsim.medium.surfaces`. The result is
    deterministic and noise-free for the given model, which makes it suitable as the common input of all sensor
    emulators, which then only add their own noise. Each optional term of the model can be turned off, to save time or
    to see what it does, see :class:`Effects <visionsim.medium.model.Effects>`.

    Note:
        Light from lamps is only scattered more than once with ``multiple_scattering``, see
        :mod:`visionsim.medium.halos`, whose glow then lights surfaces too. Given the albedo of the ground or of
        objects, the light of lamps that surfaces reflect into the medium is scattered once too, see
        :mod:`visionsim.medium.reflections`, and more with ``multiple_scattering``. With shadow maps, the light of
        suns and the sky that objects reflect is scattered too, see :func:`object_reflections
        <visionsim.medium.occlusion.object_reflections>`, once, or, with ``multiple_scattering``, also forward on its
        way through the similarity relation, while the light of suns and the sky that the ground reflects is scattered
        as many times as the medium's own, and lacks the light of suns in their shadows, and the light of the sky and
        of the medium's glow that objects hide from the ground next to them. The sky is assumed to have a uniform
        radiance above the horizon, and to be occluded by the ground below it.

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
        table_size (tuple[int, int], optional): Number of elevations and optical depths at which light from the sky,
            the ground and multiple scattering is tabulated along rays. Defaults to :data:`TABLE_SIZE
            <visionsim.medium.scattering.TABLE_SIZE>`. The variations of light scattered more than once with the
            azimuth from each sun, which are smooth, are tabulated at :data:`AZIMUTH_TABLE_SIZE
            <visionsim.medium.scattering.AZIMUTH_TABLE_SIZE>` elevations and optical depths.
        occlusion (Occlusion | None, optional): Shadow maps of the scene, see :func:`load_occlusion
            <visionsim.medium.occlusion.load_occlusion>`, through which objects cast shadows onto the medium. This
            requires ``sun_attenuation``. With ``multiple_scattering``, objects also stop the halos of the lamps that
            have maps, through a grid of the voxels they occupy, which is built along with each halo, once per lamp
            and medium, in a few seconds, or for every frame for lamps that move, see :func:`cached_lamp_halo
            <visionsim.medium.halos.cached_lamp_halo>`. Defaults to None.
        shadows (Shadows | None, optional): Shadows already traced along the rays of this frame by
            :func:`trace_shadows`, which avoids tracing them again when adding several media to the same frame. Takes
            precedence over ``occlusion`` for the shadows of suns and the sky, while the terms that need the shadow
            maps themselves, such as the light of suns and the sky that objects reflect into the medium, which depends
            on the medium, the shadows of lamps and those of surfaces, are only computed with ``occlusion``, see the
            condition ``occlusion`` of :data:`EFFECT_INPUTS <visionsim.medium.model.EFFECT_INPUTS>`. Defaults to None.
        shadow_step (float, optional): Length of the intervals of rays over which the visibility of suns is checked
            at once, in texels of their shadow maps, see :func:`trace <visionsim.medium.occlusion.trace>`.
            Defaults to 8.
        shadow_samples (int, optional): Number of samples of the visibility of the sky along each ray.
            Defaults to 8.
        effects (Effects | None, optional): Terms of the model to compute, see :class:`Effects
            <visionsim.medium.model.Effects>`, e.g. ``Effects(reflected_daylight=False)`` leaves out the light of suns
            and the sky that objects reflect into the medium (see :func:`object_reflections
            <visionsim.medium.occlusion.object_reflections>`), which takes a quarter of the time of a frame by day on
            the CPU, and a third at a quarter of the resolution, see :func:`apply_medium_upsampled
            <visionsim.medium.upsampling.apply_medium_upsampled>`. Terms whose prerequisites aren't met, such as shadow
            maps or another term, are left out as if they were off, see :meth:`Effects.effective
            <visionsim.medium.model.Effects.effective>`, given ``shadows``, ``occlusion``, ``normals``, ``albedo`` and
            ``irradiance``. Shadows given as ``shadows`` only serve the terms they were traced with. Defaults to None,
            i.e. every term that the inputs allow.
        normals (npt.ArrayLike | torch.Tensor | None, optional): Normals of the surfaces seen by each pixel, of shape
            (h, w, 3), through which surfaces are lit through the medium, see :mod:`visionsim.medium.surfaces`.
            Defaults to None, i.e. surfaces keep the light they were rendered with.
        normals_space (Literal["world", "camera"], optional): Whether normals are in world space, as Blender's normal
            pass, or in the space of the camera (x towards the right, y up and z towards the back), as saved by
            :meth:`include_normals <visionsim.simulate.blender.BlenderService.exposed_include_normals>`.
            Defaults to "world".
        emission (npt.ArrayLike | torch.Tensor | None, optional): Light that surfaces emit, of the same shape as
            ``radiance``, e.g. Blender's emission pass, which isn't affected by the light reaching them. Defaults to
            None, i.e. surfaces don't emit light.
        albedo (npt.ArrayLike | torch.Tensor | None, optional): Albedo of the diffuse surfaces seen by each pixel, of
            shape (h, w, c), or (h, w, 1) for all channels, e.g. Blender's diffuse color pass (see
            :meth:`include_diffuse_pass <visionsim.simulate.blender.BlenderService.exposed_include_diffuse_pass>`),
            from which the light that surfaces lit through the medium received is measured, as π times their radiance,
            without the light they emit, over their albedo, see :func:`received_irradiance`, rather than estimated from
            the mean albedo of the ground and of objects of the ``lighting``. The radiance of glossy surfaces also holds
            their specular reflection, which their diffuse albedo leaves out, so that they seem to have received more
            light than they did: prefer ``irradiance`` where it is known. Further channels, such as alpha, are left
            out, and those of a color albedo averaged for grayscale frames. Where it is zero, as for metals, glass, or
            surfaces that only emit light, the light received isn't known. It doesn't change the mean albedos of the
            ``lighting`` with which the ground and objects reflect light into the medium, see :func:`mean_albedos`. Only
            used with ``normals``. Defaults to None.
        irradiance (npt.ArrayLike | torch.Tensor | None, optional): Irradiance that the surfaces seen by each pixel
            received without the medium, from every light, of shape (h, w, c), or (h, w, 1) for all channels, e.g. π
            times the sum of Blender's diffuse direct and indirect light passes, saved along with its diffuse color pass,
            which measures the light that glossy surfaces received too, and takes precedence over ``albedo``. Further
            channels are left out, and those of a color irradiance averaged for grayscale frames, as for ``albedo``.
            Where it is zero, the light received isn't known. Only used with ``normals``. Defaults to None.
        device (torch.device | str | None, optional): Device to run on. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float64.

    Raises:
        ValueError: raised if the shapes of the inputs, wavelengths and lighting do not match.

    Returns:
        MediumResult: Radiance seen through the medium, alongside its transmittance, optical depth, in-scattering, and
        the ratio by which it scales the light of surfaces.
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
    sky = medium.albedo * _per_channel(lighting.sky, n, "sky", **kwargs)
    attenuated_suns = []

    shadowed = occlusion is not None or shadows is not None
    if shadowed and not medium.sun_attenuation:
        raise ValueError("Shadows can only be cast onto media that attenuate sunlight, see `Medium.sun_attenuation`.")
    # Terms whose prerequisites aren't met are left out, as if they were off
    given = {
        "shadows": shadowed,
        "occlusion": occlusion is not None,
        "normals": normals is not None,
        "albedo": albedo is not None,
        "irradiance": irradiance is not None,
    }
    effects = (effects or Effects()).resolve(medium, lighting, **given)

    for sun in lighting.suns:
        towards_sun = torch.as_tensor(sun.direction, **kwargs)
        towards_sun = towards_sun / towards_sun.norm()
        sun_irradiance = _per_channel(sun.irradiance, n, "sun irradiance", **kwargs)

        if medium.sun_attenuation:
            attenuated_suns.append((towards_sun, medium.albedo * sun_irradiance))
        else:
            source = (
                source
                + medium.albedo
                * henyey_greenstein(directions @ towards_sun, medium.anisotropy)[..., None]
                * sun_irradiance
            )

    # Skylight comes from the upper hemisphere only, weighted by how much the phase function sends it to the camera
    lit_by_sky = bool((sky != 0).any())
    if lit_by_sky and not medium.sun_attenuation:

        def sky_phase(directions: torch.Tensor) -> torch.Tensor:
            return sky_quadrature(directions, medium.anisotropy)[1].sum(dim=-1, keepdim=True)

        source = source + sky * _by_pixels(sky_phase, directions, elements_per_pixel=_SKY_NODES)
    inscatter = source * -torch.expm1(-tau)

    if medium.sun_attenuation:
        fog = medium.components[0]
        assert isinstance(fog, HeightFog)
        origin_extinction = beta * fog.density * torch.exp(-(origin[2] - fog.base_height) / fog.falloff)
        # When extinction doesn't depend on wavelength, as for fog, integrals are the same for every channel, up to
        # the color of the light
        channels = slice(0, 1) if bool((beta == beta[0]).all()) else slice(None)

        # Light from the sky, the ground and the fog itself reaches the fog from many directions, so that the light
        # scattered towards a ray only depends on its elevation, and for suns' light scattered more than once, on its
        # azimuth from the sun's: it is integrated along rays of every elevation (and azimuth) once, and interpolated
        # for each pixel. Tables only depend on the camera's height, and are kept for the next frames
        def tabulate(combined: bool = False) -> dict[str, ElevationTable]:
            depth_above = origin_extinction[channels] * fog.falloff
            return cached_camera_tables(medium, lighting, beta[channels], n, depth_above, table_size, combined)

        # Objects only hide sources tabulated one by one from the cells of the sky they hide
        occluding = shadowed and (effects.sun_shadows or effects.sky_occlusion)
        tables = tabulate(combined=not effects.sky_occlusion) if occluding else {}
        if not occluding or not (attenuated_suns or tables):
            # Without shadows, or without suns or skylight to occlude, e.g. at night
            args = (origin_extinction, fog.falloff, directions[..., 2:3], distance[..., None])
            for towards_sun, color in attenuated_suns:
                phase = henyey_greenstein(directions @ towards_sun, medium.anisotropy)[..., None]
                inscatter = inscatter + color * phase * height_fog_sun_inscatter(tau, *args, float(towards_sun[2]))
            for table in tabulate(combined=True).values():
                azimuths = None if table.towards is None else relative_azimuths(directions, table.towards)
                inscatter = inscatter + lookup(table, directions[..., 2], tau[..., channels], azimuths)
        else:
            rays, lengths = directions.reshape(-1, 3), distance.reshape(-1)
            moved = occlusion_to(occlusion, device=device, dtype=dtype) if occlusion is not None else None
            scattered = effects.scattering_occlusion and "scattered" in tables
            scattering = cached_multiple_scattering(medium, lighting, beta[channels], n) if scattered else None
            # Light scattered more than once arrives brighter from the cells of the sky towards suns, which hide more
            # of it
            profile = None
            if scattering is not None and effects.azimuthal_occlusion:
                layout = moved if moved is not None else shadows
                assert layout is not None
                profile = cell_profile(scattering, layout.sky_edges, layout.sky_counts)
            if shadows is None:
                assert moved is not None
                towards = [towards_sun for towards_sun, _ in attenuated_suns]
                # These shadows only serve this medium, and don't need the visibility of each cell of the sky
                shadows = trace(
                    moved,
                    medium,
                    origin,
                    rays,
                    lengths,
                    towards,
                    sun_step=shadow_step,
                    sky_samples=shadow_samples,
                    ground_height=lighting.ground_height,
                    keep_cells=False,
                    shafts=effects.sun_shadows,
                    sky=effects.sky_occlusion,
                    ground_shadows=effects.ground_shadows,
                    profile=profile,
                    ground_sky_occlusion=effects.ground_sky_occlusion,
                )
            else:
                shadows = _shadows_of(shadows, effects)
            shadows = shadows_to(shadows, device=device, dtype=dtype)
            daylight_reflected = None
            if moved is not None and effects.reflected_daylight:
                radiance_of = _object_surfaces(medium, lighting, beta, moved)
                if radiance_of is not None:
                    # The light that the medium scatters forward on its way keeps going towards points, through the
                    # reduced extinction of the similarity relation, as the medium scatters it more than once
                    daylight_reflected = object_reflections(
                        moved,
                        medium,
                        origin,
                        rays,
                        shadows,
                        radiance_of,
                        beta[channels],
                        similarity=effects.scattered_daylight_reflections,
                    )
            # Shares of the ground's light that suns bring, which it lacks in their shadows, and that the sky and the
            # glow of the medium bring, which it lacks where objects hide the sky from it
            shares, sky_share = None, None
            if effects.ground_shadows or effects.ground_sky_occlusion:
                shares = ground_shares(medium, lighting, beta[channels], n)
                if effects.ground_sky_occlusion:
                    sky_share = torch.ones(n, **kwargs)
                    for share in shares:
                        sky_share = sky_share - share if share is not None else sky_share
                    sky_share = sky_share.clamp_min(0)
            shaded = shade(
                shadows,
                medium,
                origin,
                rays,
                lengths,
                beta[channels],
                suns=attenuated_suns,
                tables=tables,
                scattering=scattering,
                ground_shares=shares if effects.ground_shadows else None,
                reflected=daylight_reflected,
                profile=profile,
                ground_sky_share=sky_share,
            )
            inscatter = inscatter + shaded.reshape(*distance.shape, -1)

    # Lamps and emissive surfaces shine as in Cycles, see `visionsim.medium.lights`, and objects cast their shadows
    # Single precision is enough to look up the maps of lamps, and faster. Those of emissive surfaces see through them.
    # Halos are cached by the identity of the maps given, which stay the same from one frame to the next
    tolerance = occlusion.lamp_tolerance if occlusion is not None else 0.01
    stored = (
        {False: None, True: None} if occlusion is None else {False: occlusion.lamp_maps, True: occlusion.emissive_maps}
    )
    single = {
        k: maps_to(m, device=device, dtype=torch.float32) if m is not None and effects.lamp_shadows else None
        for k, m in stored.items()
    }
    rays, lengths = directions.reshape(-1, 3), distance.reshape(-1)
    objects = lighting.ground_albedo if lighting.object_albedo is None else lighting.object_albedo
    reflecting = (
        effects.lamp_reflections and medium.albedo > 0 and max((*lighting.ground_albedo, *objects), default=0) > 0
    )
    albedos = (tuple(lighting.ground_albedo), tuple(objects)) if reflecting else None
    # Objects hide the light that surfaces reflect as the scene's shadow maps record them, cached by their identity
    occluders = scene_surfaces(occlusion, lighting.ground_height) if reflecting and occlusion is not None else None
    specular = (lighting.surface_roughness, lighting.surface_ior) if effects.specular_reflections else None
    # Every surface that the scene's shadow maps record receives the light of other surfaces, cached by their identity
    surfels = None
    if reflecting and occlusion is not None and effects.bounce_surfaces:
        surfels = scene_surfels(occlusion, lighting.ground_height)
    glows: list[Glow | None] = []
    for lamp in lighting.lamps:
        emissive = isinstance(lamp, EmissiveSurface)
        maps = single[emissive]
        light = lamp_inscatter(medium, origin, rays, lengths, beta, lamp, time, maps=maps, map_tolerance=tolerance)
        inscatter = inscatter + medium.albedo * light.reshape(*distance.shape, n)
        # Surfaces lit by lamps reflect their light into the medium, which scatters it once, see
        # `visionsim.medium.reflections`
        reflection, glow = None, None
        if effects.halos and halo_threshold(medium, lamp, beta):
            # In dense media, the light of lamps scattered more than once spreads a halo around lamps, which objects
            # stop, and whose glow lights surfaces, see `visionsim.medium.halos`. The light that surfaces reflect, lit
            # directly and by the glow, is scattered more than once along with it, and kept with it
            ground, kept = lighting.ground_height, stored[emissive]
            halo = cached_lamp_halo(
                medium,
                lamp,
                beta,
                ground,
                kept,
                time=time,
                map_tolerance=tolerance,
                occlusion=occlusion if effects.halo_occlusion else None,
                albedos=albedos,
                occluders=occluders,
                shadows=effects.lamp_shadows,
                scattered_reflection=effects.scattered_reflections,
                relit_scattering=effects.relit_scattering,
                relit=effects.relit_reflections,
                bounces=effects.reflection_bounces,
                specular=specular,
                surfels=surfels,
                interreflections=effects.interreflections,
                shadowed_reflection=effects.reflection_shadows,
                occluded_reflection=effects.reflection_occlusion,
                shadow_edges=effects.halo_shadow_edges,
            )
            halo_light = halo_inscatter(halo, medium, origin, rays, lengths, beta, time=time)
            inscatter = inscatter + halo_light.reshape(*distance.shape, n)
            reflection, glow = halo.reflection, (partial(halo_glow, halo) if effects.lamp_glow else None)
        elif albedos is not None:
            reflection = cached_lamp_reflection(
                medium,
                lamp,
                beta,
                lighting.ground_height,
                *albedos,
                stored[emissive],
                occluders,
                time=time,
                map_tolerance=tolerance,
                bounces=effects.reflection_bounces,
                shadowed=effects.reflection_shadows,
                occluded=effects.reflection_occlusion,
                specular=specular,
                surfels=surfels,
                interreflections=effects.interreflections,
            )
        glows.append(glow)
        if reflection is not None:
            reflected = reflection_inscatter(reflection, medium, origin, rays, lengths, beta, time=time)
            inscatter = inscatter + reflected.reshape(*distance.shape, n)

    # Surfaces are lit through the medium, which dims the light they reflect, but not the light they emit
    surfaces, illumination, lit = radiance[..., :n], None, None
    if normals is not None:
        emitted = 0.0 if emission is None else torch.as_tensor(np.asarray(emission), **kwargs)[..., :n]
        illumination, *lit = _illumination(
            normals,
            normals_space,
            transform_matrix,
            origin,
            directions,
            distance,
            medium,
            lighting,
            beta,
            occlusion,
            time,
            glows,
            surfaces - emitted if effects.received_irradiance else None,
            effects,
            albedo if effects.received_irradiance else None,
            irradiance if effects.received_irradiance else None,
        )
        surfaces = illumination * (surfaces - emitted) + emitted

    return MediumResult(
        radiance=torch.cat([transmittance * surfaces + inscatter, radiance[..., n:]], dim=-1),
        transmittance=transmittance,
        optical_depth=tau,
        inscatter=inscatter,
        illumination=illumination,
        surface_irradiance=lit[1] if lit else None,
        clear_irradiance=lit[0] if lit else None,
    )


def _object_surfaces(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, occlusion: Occlusion
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
    """Radiance of the objects' surfaces, which reflect the light of suns and the sky into the medium, as a function of
    points on them and of their normals, see :func:`object_radiance <visionsim.medium.surfaces.object_radiance>`, or
    None if they reflect none, e.g. at night."""
    albedo = lighting.ground_albedo if lighting.object_albedo is None else lighting.object_albedo
    if medium.albedo <= 0 or max(albedo, default=0.0) <= 0 or not environment_lit(lighting):
        return None
    return partial(object_radiance, medium, lighting, beta, occlusion=occlusion)


def _shadows_of(shadows: Shadows, effects: Effects) -> Shadows:
    """Shadows traced beforehand, see :func:`trace_shadows`, without what the terms of the model leave out: the segments
    along which suns are hidden without ``sun_shadows``, the samples of the sky's visibility without ``sky_occlusion``,
    where the light from below the horizon starts on the ground in the shadows of suns without ``ground_shadows``, the
    share of the sky that the ground doesn't see there without ``ground_sky_occlusion``, and the sums weighted by the
    radiance arriving from each cell of the sky without ``azimuthal_occlusion``."""
    if not effects.sun_shadows:
        # Suns are kept for their shadows on the ground, by their index
        shadows = shadows._replace(
            suns=tuple(
                sun._replace(**{name: getattr(sun, name)[:0] for name in sun._fields[1:]}) for sun in shadows.suns
            )
        )
    if not effects.ground_shadows:
        shadows = shadows._replace(shadowed_below=(), shadowed_weights_below=None)
    if not effects.ground_sky_occlusion:
        shadows = shadows._replace(unseen_weights_below=None)
    if not effects.azimuthal_occlusion:
        shadows = shadows._replace(scattered=None)
    if not effects.sky_occlusion:
        # Without samples along rays, as when the scene has no maps of the sky. Ground shadows, which need them, are
        # off, and so are the share of the sky that the ground doesn't see and the sums weighted by the radiance
        # arriving from each cell
        samples = ("bounds", "middles", "visible", "visible_below", "visible_weights", "visible_weights_below")
        shadows = shadows._replace(
            **{name: getattr(shadows, name)[:, :0] for name in samples}, scattered=None, unseen_weights_below=None
        )
    return shadows


def _illumination(
    normals: npt.ArrayLike | torch.Tensor,
    normals_space: str,
    transform_matrix: npt.ArrayLike,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    occlusion: Occlusion | None,
    time: float,
    glows: Sequence[Glow | None] | None = None,
    reflected: torch.Tensor | None = None,
    effects: Effects | None = None,
    albedo: npt.ArrayLike | torch.Tensor | None = None,
    irradiance: npt.ArrayLike | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ratio by which a medium scales the light of the surfaces seen by each pixel, see
    :func:`surface_attenuation <visionsim.medium.surfaces.surface_attenuation>`, of shape (h, w, c), where the glow
    around each lamp, if any, lights them, see :func:`surface_irradiance
    <visionsim.medium.surfaces.surface_irradiance>`, where the light that surfaces reflect without the medium, of
    shape (h, w, c), if given, sets the irradiance they received, see :func:`received_irradiance`, measured by the
    irradiance of each pixel, if given, or else by its albedo, if given, or else estimated from the mean albedos of the
    lighting, given the terms of the model to compute, along with the irradiance of the surfaces through the medium and
    without it, of shape (h, w, c), zero where pixels see the background."""
    kwargs: dict[str, Any] = {"dtype": directions.dtype, "device": directions.device}
    normals = normals.to(**kwargs) if torch.is_tensor(normals) else torch.tensor(np.asarray(normals), **kwargs)
    normals = normals[..., :3]
    if normals.shape[:2] != directions.shape[:2]:
        raise ValueError(f"Normals {tuple(normals.shape)} do not match the camera's resolution.")
    # The light that the surface of each pixel received, where it is measured
    measured_light = None
    if reflected is not None:
        measured_light = _measured_irradiance(lighting, reflected.to(**kwargs), albedo, irradiance)
    if normals_space == "camera":
        rotation = torch.as_tensor(np.asarray(transform_matrix, dtype=float), **kwargs)[:3, :3]
        normals = normals @ (rotation / rotation.norm(dim=0)).T
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Surfaces are lit on the side the camera sees
    normals = torch.where(((normals * directions).sum(dim=-1) > 0)[..., None], -normals, normals)

    seen = torch.isfinite(distance)
    illumination = torch.ones(*distance.shape, len(beta), **kwargs)
    clear_map, through_map = torch.zeros_like(illumination), torch.zeros_like(illumination)
    if bool(seen.any()):
        points = origin + directions[seen] * distance[seen][:, None]
        rays = (origin, directions[seen], distance[seen])
        clear, through = surface_irradiance(
            medium,
            lighting,
            beta,
            points,
            normals[seen],
            rays=rays,
            occlusion=occlusion,
            time=time,
            glows=glows,
            effects=effects,
        )
        received, measured = None, None
        if measured_light is not None:
            received, measured = measured_light[0][seen], measured_light[1][seen]
        elif reflected is not None:
            received = received_irradiance(lighting, points, reflected[seen])
        if effects is not None and not effects.measured_indirect:
            # The light measured still bounds the light that reaches surfaces indirectly from below
            measured = None
        illumination[seen] = surface_attenuation(clear, through, received=received, measured=measured)
        clear_map[seen], through_map[seen] = clear.to(clear_map), through.to(through_map)
    return illumination, clear_map, through_map


def _measured_irradiance(
    lighting: Lighting,
    reflected: torch.Tensor,
    albedo: npt.ArrayLike | torch.Tensor | None = None,
    irradiance: npt.ArrayLike | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Irradiance that the surface seen by each pixel received without the medium, as measured by its ``irradiance`` if
    given, or else by the radiance it reflects and its ``albedo``, see :func:`received_irradiance`, as given to
    :func:`apply_medium`, of shape (h, w, c), and where it is known, i.e. given and positive, as a boolean of the same
    shape, or None if neither is given. ``reflected`` is the radiance that surfaces reflect without the medium, without
    the light they emit, of shape (h, w, c)."""
    pixels, n = tuple(reflected.shape[:2]), reflected.shape[-1]
    kwargs: dict[str, Any] = {"dtype": reflected.dtype, "device": reflected.device}
    if irradiance is not None:
        received = _per_pixel(irradiance, pixels, n, "Irradiance", **kwargs).clamp_min(0)
        return received, received > 0
    if albedo is None:
        return None
    albedos = _per_pixel(albedo, pixels, n, "Albedo", **kwargs)
    flat = received_irradiance(lighting, None, reflected.reshape(-1, n), albedo=albedos.reshape(-1, n))
    received = flat.reshape(*pixels, n) if flat is not None else torch.zeros_like(reflected)
    return received, albedos > 0


def _per_pixel(
    values: npt.ArrayLike | torch.Tensor, pixels: tuple[int, ...], channels: int, name: str = "Values", **kwargs: Any
) -> torch.Tensor:
    """Values given for each pixel, such as an albedo, with as many channels as the radiance whose medium is computed.

    Args:
        values (npt.ArrayLike | torch.Tensor): Values of each pixel, of shape (h, w, c), or (h, w, 1) or (h, w) for
            every channel. Further channels, such as alpha, are left out, and, for a single channel, as of grayscale
            frames, the first three, as of a color image, are averaged.
        pixels (tuple[int, ...]): Number of rows and columns of pixels, (h, w).
        channels (int): Number of channels of the radiance.
        name (str, optional): Name of the values, by which errors refer to them. Defaults to "Values".
        **kwargs: Device and dtype of the result.

    Raises:
        ValueError: If the values don't have the given number of pixels, or have fewer channels but one.

    Returns:
        torch.Tensor: Values of shape (h, w, channels).
    """
    values = values.to(**kwargs) if torch.is_tensor(values) else torch.tensor(np.asarray(values), **kwargs)
    values = values[..., None] if values.ndim == 2 else values
    if values.ndim != 3 or tuple(values.shape[:2]) != tuple(pixels) or 1 < values.shape[-1] < channels:
        raise ValueError(f"{name} {tuple(values.shape)} does not match the camera's resolution and channels.")
    if channels == 1 and values.shape[-1] >= 3:
        values = values[..., :3].mean(dim=-1, keepdim=True)
    return values[..., :channels].expand(*pixels, channels)


def received_irradiance(
    lighting: Lighting, points: torch.Tensor | None, reflected: torch.Tensor, albedo: torch.Tensor | None = None
) -> torch.Tensor | None:
    """Irradiance that diffuse surfaces received from every light, given the radiance they reflect and their albedo.

    Given the albedo of each surface, such as Blender's diffuse color pass, this is the irradiance it received, as
    long as it is diffuse: the specular reflection of glossy surfaces, which their radiance holds but their diffuse
    albedo doesn't, makes it seem higher, e.g. by 20% on the Principled ground of ``examples/medium`` by day, and 12%
    on its objects, against Blender's diffuse light passes. Otherwise, points less than :data:`GROUND_CLEARANCE
    <visionsim.medium.reflections.GROUND_CLEARANCE>` above the lighting's ``ground_height`` have the albedo of the
    ground, as for the light of lamps that surfaces reflect, see :mod:`visionsim.medium.reflections`, and others that of
    objects. As these are the mean albedos of the scene's surfaces, this is the irradiance of surfaces of the mean albedo
    that reflect the same light, which is too low on dark surfaces, and too high on bright ones, such as the dark and
    light squares of a checkerboard.

    Args:
        lighting (Lighting): Lighting of the scene, with the albedo of the ground and of objects, per channel or a
            single one.
        points (torch.Tensor | None): Points on surfaces, of shape (n, 3), only needed without ``albedo``.
        reflected (torch.Tensor): Radiance that the surfaces reflect, without the light they emit, of shape (n, c).
        albedo (torch.Tensor | None, optional): Albedo of each surface, of shape (n, c), which replaces the albedos of
            the lighting, zero where it isn't known. Defaults to None, i.e. those of the lighting.

    Raises:
        ValueError: If neither ``points`` nor ``albedo`` are given.

    Returns:
        torch.Tensor | None: Irradiance, π times the radiance over the albedo, of shape (n, c), which is zero where the
        albedo is zero, i.e. unknown, or None where it is everywhere.
    """
    n = reflected.shape[-1]
    kwargs: dict[str, Any] = {"dtype": reflected.dtype, "device": reflected.device}
    if albedo is None:
        ground = _per_channel(lighting.ground_albedo, n, "ground albedo", **kwargs)
        objects = ground
        if lighting.object_albedo is not None:
            objects = _per_channel(lighting.object_albedo, n, "object albedo", **kwargs)
        if not bool((ground > 0).any() or (objects > 0).any()):
            return None
        if points is None:
            raise ValueError("Points on surfaces are needed to tell the ground from objects without their albedo.")
        on_ground = (points[:, 2] < lighting.ground_height + GROUND_CLEARANCE)[:, None].to(reflected.device)
        albedo = torch.where(on_ground, ground, objects)
    elif not bool((albedo > 0).any()):
        return None
    albedo = albedo.to(**kwargs)
    received = math.pi * reflected.clamp_min(0) / albedo.clamp_min(1e-12)
    return torch.where(albedo > 0, received, torch.zeros_like(received))


def mean_albedos(
    frames: Iterable[tuple[Any, Any, Mapping[str, Any], npt.ArrayLike, Any]],
    ground_height: float = 0.0,
    background_depth: float = 1e9,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Mean albedo of the ground and of objects, with which they reflect light into the medium, from the albedo of the
    surfaces seen in frames, such as Blender's diffuse color pass.

    The ground and objects each reflect light into the medium with a single albedo, that of their surfaces on average,
    see :attr:`Lighting.ground_albedo <visionsim.medium.model.Lighting.ground_albedo>` and :attr:`Lighting.object_albedo
    <visionsim.medium.model.Lighting.object_albedo>`. Surfaces that receive more light reflect more of it, so that the
    mean albedo is weighted by the light that each surface received, ``π · radiance / albedo``, as found from the
    radiance of frames rendered without the medium: it is the albedo of grey surfaces that would reflect as much light
    as the surfaces seen, under the same light, ``Σ radiance / Σ (radiance / albedo)`` per channel over the pixels that
    see them, which only counts the surfaces that the camera sees, and in proportion to their size in the frames. As the
    radiance of glossy surfaces holds their specular reflection too, it should be the radiance that they reflect
    diffusely where it is known, such as the albedo times Blender's diffuse light passes, whose ratio to the albedo is
    the light received. Without the radiance, every pixel counts as much. Points less than :data:`GROUND_CLEARANCE
    <visionsim.medium.reflections.GROUND_CLEARANCE>` above ``ground_height`` are on the ground, and others on objects,
    as for :func:`received_irradiance`, while pixels whose albedo is zero, such as those of metals, glass or surfaces
    that only emit light, aren't counted. A diffuse albedo, such as Blender's diffuse color pass, leaves out the specular
    reflection of surfaces, e.g. that of the specular layer of a Principled BSDF, which reflects light into the medium
    too, mostly at grazing angles: the albedo found is then less than the light that surfaces reflect implies.

    As the light that objects reflect into the medium doesn't depend on what the camera sees, the albedo should be
    estimated once for a sequence, from frames that see most of the scene's objects, rather than for each frame, which
    would make the medium flicker as objects come into view, and compute the halos of lamps and the light that surfaces
    reflect again at every frame, as they are kept for given albedos, e.g.::

        # The light that surfaces reflect diffusely, from Blender's diffuse color and light passes
        diffuse = [albedo[i] * (direct[i] + indirect[i]) for i in range(len(poses))]
        frames = [(albedo[i], depth[i], camera, poses[i], diffuse[i]) for i in range(0, len(poses), 10)]
        ground, objects = mean_albedos(frames, lighting.ground_height)
        lighting = lighting.model_copy(update={"object_albedo": objects})

    Args:
        frames (Iterable[tuple[Any, Any, Mapping[str, Any], npt.ArrayLike, Any]]): For each frame, the albedo of the
            surface seen by each pixel, of shape (h, w, c) or (h, w, 1), its depth, of shape (h, w) or (h, w, 1), as
            given to :func:`apply_medium`, the camera's intrinsics and camera-to-world transform, see
            :func:`camera_rays`, and the radiance of surfaces without the medium, without the light they emit, of shape
            (h, w, c) without alpha, or (h, w, 1), or None. The albedo has the channels of the radiance, as for
            :func:`apply_medium`, and three without it.
        ground_height (float, optional): Height of the ground, in meters, see :attr:`Lighting.ground_height
            <visionsim.medium.model.Lighting.ground_height>`. Defaults to 0.0.
        background_depth (float, optional): Depth from which pixels see the background, which they don't count.
            Defaults to 1e9.

    Returns:
        tuple[tuple[float, ...] | None, tuple[float, ...] | None]: Mean albedo per channel of the ground and of
        objects, each None if no pixel sees them.
    """

    def tensor(x: Any) -> torch.Tensor:
        values = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, dtype=float))
        return values.to(torch.float64)

    # Light reflected and received, summed over the pixels that see the ground, and objects
    sums: list[torch.Tensor | None] = [None, None]
    for albedo, depth, camera, transform_matrix, radiance in frames:
        depths = tensor(depth)
        depths = depths[..., 0] if depths.ndim == 3 else depths
        origin, directions, scale = camera_rays(camera, transform_matrix)
        pixels = tuple(directions.shape[:2])
        if tuple(depths.shape) != pixels:
            raise ValueError(f"Depth {tuple(depths.shape)} doesn't match the camera.")
        light = None if radiance is None else tensor(radiance)
        light = light[..., None] if light is not None and light.ndim == 2 else light
        # The same albedo for every channel of the radiance, as when those of grey surfaces are collapsed, and alpha left
        # out
        values = _per_pixel(tensor(albedo), pixels, 3 if light is None else light.shape[-1], "Albedo")
        seen = torch.isfinite(depths) & (depths < background_depth)
        heights = origin[2] + directions[..., 2] * torch.where(seen, depths, torch.zeros_like(depths)) * scale
        weights = torch.ones_like(values)
        if light is not None:
            # Each pixel counts the light its surface received, of which it reflects its albedo
            weights = light.clamp_min(0) / values.clamp_min(1e-12)
        weights = torch.where(values > 0, weights, torch.zeros_like(weights))
        on_ground = heights < ground_height + GROUND_CLEARANCE
        for k, where in enumerate((seen & on_ground, seen & ~on_ground)):
            total = torch.stack([(weights * values)[where].sum(dim=0), weights[where].sum(dim=0)])
            previous = sums[k]
            sums[k] = total if previous is None else previous + total
    means = [None if s is None or not bool((s[1] > 0).all()) else tuple((s[0] / s[1]).tolist()) for s in sums]
    return means[0], means[1]


def trace_shadows(
    occlusion: Occlusion,
    depth: npt.ArrayLike | torch.Tensor,
    camera: Mapping[str, Any],
    transform_matrix: npt.ArrayLike,
    medium: Medium,
    lighting: Lighting,
    background_depth: float = 1e9,
    shadow_step: float = 8.0,
    shadow_samples: int = 8,
    effects: Effects | None = None,
    wavelengths: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Shadows:
    """Trace where objects hide suns and the sky along the rays of a frame, to add several media to it.

    Tracing shadows takes most of the time of :func:`apply_medium` with shadow maps, but doesn't depend on the medium,
    so it can be done once per frame and reused for media of similar density, whose light is scattered from roughly
    the same places along rays, see :func:`trace <visionsim.medium.occlusion.trace>`. The light that objects reflect
    into each medium, which depends on it, is gathered by :func:`apply_medium` given the shadow maps too. For instance,
    to render a frame with fogs of several visibilities::

        shadows = trace_shadows(occlusion, depth, camera, pose, media[0], lighting)
        frames = [
            apply_medium(radiance, depth, camera, pose, m, lighting, occlusion=occlusion, shadows=shadows)
            for m in media
        ]

    Args:
        occlusion (Occlusion): Shadow maps of the scene, see :func:`load_occlusion
            <visionsim.medium.occlusion.load_occlusion>`.
        depth (npt.ArrayLike | torch.Tensor): Depth of the scene, as given to :func:`apply_medium`.
        camera (Mapping[str, Any]): Camera intrinsics, see :func:`camera_rays`.
        transform_matrix (npt.ArrayLike): Camera-to-world transform, see :func:`camera_rays`.
        medium (Medium): Participating medium, made of a single height fog, which sets where rays are sampled.
        lighting (Lighting): Lighting of the medium, whose suns cast shadows.
        background_depth (float, optional): Depth from which pixels are considered to see the background.
            Defaults to 1e9.
        shadow_step (float, optional): Length of the intervals of rays over which the visibility of suns is checked
            at once, in texels of their shadow maps, see :func:`trace <visionsim.medium.occlusion.trace>`.
            Defaults to 8.
        shadow_samples (int, optional): Number of samples of the visibility of the sky along each ray.
            Defaults to 8.
        effects (Effects | None, optional): Terms of the model whose shadows are traced, see :class:`Effects
            <visionsim.medium.model.Effects>`: suns are only traced with ``sun_shadows``, the sky with
            ``sky_occlusion``, where the light from below the horizon starts on the ground in the shadows of suns with
            ``ground_shadows``, the share of the sky that the ground doesn't see there with ``ground_sky_occlusion``,
            and the sums of the phase function weighted by the radiance of light scattered at least once arriving from
            each cell of the sky with ``azimuthal_occlusion``, for this medium and lighting, see
            :attr:`Shadows.scattered <visionsim.medium.occlusion.Shadows.scattered>`, so that :func:`apply_medium`
            leaves out the others. Defaults to None, i.e. all of them.
        wavelengths (Sequence[float] | None, optional): Effective wavelength, in nm, of each channel of the frames
            the shadows serve, whose light scattered more than once weighs the cells of the sky. Defaults to
            :data:`RGB_WAVELENGTHS` for lightings of three colors, and to 550nm otherwise.
        device (torch.device | str | None, optional): Device to run on. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float64.

    Raises:
        ValueError: raised if the medium doesn't attenuate sunlight, or if the depth doesn't match the camera.

    Returns:
        Shadows: Where suns and cells of the sky are hidden along each ray, to be given to :func:`apply_medium`.
    """
    if not medium.sun_attenuation:
        raise ValueError("Shadows can only be cast onto media that attenuate sunlight, see `Medium.sun_attenuation`.")
    kwargs: dict[str, Any] = {"dtype": dtype, "device": device}
    depth = depth.to(**kwargs) if torch.is_tensor(depth) else torch.tensor(np.asarray(depth), **kwargs)
    depth = depth[..., 0] if depth.ndim == 3 else depth
    origin, directions, scale = camera_rays(camera, transform_matrix, **kwargs)
    if depth.shape != directions.shape[:2]:
        raise ValueError(f"Depth {tuple(depth.shape)} does not match the camera's resolution.")
    background = ~torch.isfinite(depth) | (depth >= background_depth)
    distance = torch.where(background, torch.full_like(depth, torch.inf), depth * scale)
    towards = [torch.as_tensor(sun.direction, **kwargs) for sun in lighting.suns]
    effects = (effects or Effects()).resolve(medium, lighting, occlusion=True)
    profile = None
    if effects.azimuthal_occlusion and environment_lit(lighting):
        colors = [lighting.sky, lighting.ground_albedo, *(sun.irradiance for sun in lighting.suns)]
        n = max(len(color) for color in colors)
        wavelengths = tuple(wavelengths or (RGB_WAVELENGTHS if n == 3 else (550.0,) * n))
        beta = torch.as_tensor(medium.extinction_at(wavelengths), **kwargs)
        channels = slice(0, 1) if bool((beta == beta[0]).all()) else slice(None)
        scattering = cached_multiple_scattering(medium, lighting, beta[channels], len(beta))
        profile = cell_profile(scattering, occlusion.sky_edges, occlusion.sky_counts)
    return trace(
        occlusion_to(occlusion, device=device, dtype=dtype),
        medium,
        origin,
        directions.reshape(-1, 3),
        distance.reshape(-1),
        [t / t.norm() for t in towards],
        sun_step=shadow_step,
        sky_samples=shadow_samples,
        ground_height=lighting.ground_height,
        shafts=effects.sun_shadows,
        sky=effects.sky_occlusion,
        ground_shadows=effects.ground_shadows,
        profile=profile,
        ground_sky_occlusion=effects.ground_sky_occlusion,
    )
