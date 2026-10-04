from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from visionsim.medium.lights import AREA_LEVELS, AreaLevel, Emitter, area_weights, closest_distances, emitters
from visionsim.medium.model import AreaLight, HeightFog, Lamp, Lighting, Medium, PointLight, SpotLight
from visionsim.medium.occlusion import (
    Occlusion,
    ShadowMaps,
    Shadows,
    lamp_map_index,
    lamp_shadow,
    maps_to,
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
from visionsim.medium.scattering import (
    TABLE_SIZE,
    cached_multiple_scattering,
    elevation_source,
    elevation_sources,
    lookup,
    tabulate_along_rays,
)
from visionsim.medium.surfaces import surface_attenuation, surface_irradiance

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
    or None if surfaces are left as they were rendered without the medium"""


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
) -> torch.Tensor:
    """Light of a lamp scattered once towards the camera along rays, before it is scaled by the medium's albedo.

    Lamps are made of point emitters, see :mod:`visionsim.medium.lights`, whose light is integrated with
    :func:`point_light_inscatter <visionsim.medium.optics.point_light_inscatter>`. Rays that pass far from area lights
    see them as a single emitter, and those that pass closer to them as finer grids of emitters, see
    :func:`area_weights <visionsim.medium.lights.area_weights>`. Given maps of lamps, objects cast the shadows of the
    lamp's center onto the medium, see :func:`lamp_shadow <visionsim.medium.occlusion.lamp_shadow>`.

    Args:
        medium (Medium): Participating medium.
        origin (torch.Tensor): Ray origin, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (r, 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (r,), can be infinite.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        lamp (Lamp): Point, spot or area light.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        area_levels (tuple[AreaLevel, ...], optional): Grids of emitters through which rays that pass close to area
            lights see them. Defaults to :data:`AREA_LEVELS <visionsim.medium.lights.AREA_LEVELS>`.
        maps (ShadowMaps | None, optional): Maps of lamps, see :attr:`Occlusion.lamp_maps
            <visionsim.medium.occlusion.Occlusion.lamp_maps>`, among which the lamp's map is found by its position.
            Defaults to None, i.e. the lamp's light is not occluded.

    Returns:
        torch.Tensor: Radiance scattered towards the camera, for an albedo of one, of shape (r, c).
    """
    n = len(beta)
    kwargs: dict[str, Any] = {"dtype": directions.dtype, "device": directions.device}
    # Maps of area lights are rendered just in front of them
    center = torch.as_tensor(lamp.position, **kwargs)
    shifted = center + 1e-3 * torch.as_tensor(lamp.direction, **kwargs) if isinstance(lamp, AreaLight) else center
    index = lamp_map_index(maps, shifted)

    def scatter(group: list[Emitter], nodes: int, directions: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        size = max(1, int(_CHUNK_ELEMENTS // max(nodes * (8 + 2 * n), 12 * _LAMP_SHADOW_SAMPLES * (index is not None))))
        parts = []
        for i in range(0, len(directions), size):
            rays, lengths = directions[i : i + size], distance[i : i + size]
            shadow, seen = None, None
            if maps is not None and index is not None:
                shadow = lamp_shadow(maps, index, origin, rays, lengths, samples=_LAMP_SHADOW_SAMPLES)
                # Rays entirely in the lamp's shadow, such as those of other rooms, receive none of its light
                if not bool(shadow.lit.all()):
                    seen = shadow.lit.nonzero().squeeze(-1)
                    shadow, rays, lengths = shadow.select(seen), rays[seen], lengths[seen]
            total = rays.new_zeros(len(rays), n)
            for emitter in group if len(rays) else []:
                light = point_light_inscatter(
                    medium,
                    origin,
                    rays,
                    lengths,
                    emitter.position,
                    beta,
                    radius=emitter.radius,
                    time=time,
                    nodes=nodes,
                    axis=emitter.axis,
                    cone=emitter.cone,
                    profile=emitter.profile,
                    falloff=emitter.falloff,
                    shadow=shadow.average if shadow is not None else None,
                )
                total = total + emitter.intensity * light
            if seen is not None:
                total = total.new_zeros(len(directions[i : i + size]), n).index_copy(0, seen, total)
            parts.append(total)
        return torch.cat(parts)

    result = directions.new_zeros(len(directions), n)
    if isinstance(lamp, PointLight):
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
            light = scatter(emitters(lamp, n, **kwargs), _POINT_LIGHT_NODES, directions[rays], distance[rays])
            result = result.index_copy(0, rays, light)
        return result

    weights = area_weights(lamp, closest_distances(origin, directions, distance, center), area_levels)
    grids = [(level.samples, level.nodes) for level in area_levels] + [(1, _POINT_LIGHT_NODES)]
    for (samples, nodes), weight in zip(grids, weights):
        if len(rays := ((weight > 0) & lit).nonzero().squeeze(-1)):
            light = scatter(emitters(lamp, n, samples, **kwargs), nodes, directions[rays], distance[rays])
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
    normals: npt.ArrayLike | torch.Tensor | None = None,
    normals_space: Literal["world", "camera"] = "world",
    emission: npt.ArrayLike | torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> MediumResult:
    """Add a participating medium to a rendered frame, in closed form.

    The radiance of each surface is attenuated by the medium's transmittance, and light from the environment,
    suns and lamps is scattered towards the camera. Sunlight is integrated in closed form, without any ray
    marching, light that reaches the medium from many directions (from the sky, from the ground, and light scattered
    more than once) is integrated once per frame along rays of every elevation and interpolated, see
    :mod:`visionsim.medium.scattering`, and light from point, spot and area lights is integrated with a quadrature that
    removes its singularity, see :func:`lamp_inscatter`. When shadow maps of the scene are given, objects cast shadows
    onto the medium (light shafts) and hide part of the sky from it, see :mod:`visionsim.medium.occlusion`. Given the
    normals of surfaces, surfaces are also lit through the medium, which dims their light on its way to them while its
    glow lights them, see :mod:`visionsim.medium.surfaces`. The result is deterministic and noise-free for the given
    model, which makes it suitable as the common input of all sensor emulators, which then only add their own noise.

    Note:
        Light from lamps is only scattered once, and light reflected by the ground into the medium isn't occluded.
        The sky is assumed to have a uniform radiance above the horizon, and to be occluded by the ground below it.

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
            <visionsim.medium.scattering.TABLE_SIZE>`.
        occlusion (Occlusion | None, optional): Shadow maps of the scene, see :func:`load_occlusion
            <visionsim.medium.occlusion.load_occlusion>`, through which objects cast shadows onto the medium. This
            requires ``sun_attenuation``. Defaults to None.
        shadows (Shadows | None, optional): Shadows already traced along the rays of this frame by
            :func:`trace_shadows`, which avoids tracing them again when adding several media to the same frame. Takes
            precedence over ``occlusion``. Defaults to None.
        shadow_step (float, optional): Length of the intervals of rays over which the visibility of suns is checked
            at once, in texels of their shadow maps, see :func:`trace <visionsim.medium.occlusion.trace>`.
            Defaults to 8.
        shadow_samples (int, optional): Number of samples of the visibility of the sky along each ray.
            Defaults to 8.
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

    if (occlusion is not None or shadows is not None) and not medium.sun_attenuation:
        raise ValueError("Shadows can only be cast onto media that attenuate sunlight, see `Medium.sun_attenuation`.")

    for sun in lighting.suns:
        towards_sun = torch.as_tensor(sun.direction, **kwargs)
        towards_sun = towards_sun / towards_sun.norm()
        irradiance = _per_channel(sun.irradiance, n, "sun irradiance", **kwargs)

        if medium.sun_attenuation:
            attenuated_suns.append((towards_sun, medium.albedo * irradiance))
        else:
            source = (
                source
                + medium.albedo * henyey_greenstein(directions @ towards_sun, medium.anisotropy)[..., None] * irradiance
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
        # scattered towards a ray only depends on its elevation: it is integrated along rays of every elevation once,
        # and interpolated for each pixel
        def tabulate(source: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            return tabulate_along_rays(source, origin_extinction[channels] * fog.falloff, table_size)

        sources = elevation_sources(medium, lighting, beta[channels], n)
        if (occlusion is None and shadows is None) or not (attenuated_suns or sources):
            # Without shadows, or without suns or skylight to occlude, e.g. at night
            args = (origin_extinction, fog.falloff, directions[..., 2:3], distance[..., None])
            for towards_sun, color in attenuated_suns:
                phase = henyey_greenstein(directions @ towards_sun, medium.anisotropy)[..., None]
                inscatter = inscatter + color * phase * height_fog_sun_inscatter(tau, *args, float(towards_sun[2]))
            if source := elevation_source(medium, lighting, beta[channels], n):
                inscatter = inscatter + lookup(tabulate(source), directions[..., 2], tau[..., channels])
        else:
            rays, lengths = directions.reshape(-1, 3), distance.reshape(-1)
            if shadows is None:
                assert occlusion is not None
                towards = [towards_sun for towards_sun, _ in attenuated_suns]
                occlusion = occlusion_to(occlusion, device=device, dtype=dtype)
                shadows = trace(
                    occlusion,
                    medium,
                    origin,
                    rays,
                    lengths,
                    towards,
                    sun_step=shadow_step,
                    sky_samples=shadow_samples,
                    ground_height=lighting.ground_height,
                )
            scattered = medium.multiple_scattering and "scattered" in sources
            shaded = shade(
                shadows_to(shadows, device=device, dtype=dtype),
                medium,
                origin,
                rays,
                lengths,
                beta[channels],
                suns=attenuated_suns,
                tables={name: tabulate(source) for name, source in sources.items()},
                scattering=cached_multiple_scattering(medium, lighting, beta[channels], n) if scattered else None,
            )
            inscatter = inscatter + shaded.reshape(*distance.shape, -1)

    # Point, spot and area lights shine as in Cycles, see `visionsim.medium.lights`, and objects cast their shadows
    lamp_maps = None
    if occlusion is not None and occlusion.lamp_maps is not None:
        # Single precision is enough to look up the maps of lamps, and faster
        lamp_maps = maps_to(occlusion.lamp_maps, device=device, dtype=torch.float32)
    for lamp in lighting.lamps:
        light = lamp_inscatter(
            medium, origin, directions.reshape(-1, 3), distance.reshape(-1), beta, lamp, time, maps=lamp_maps
        )
        inscatter = inscatter + medium.albedo * light.reshape(*distance.shape, n)

    # Surfaces are lit through the medium, which dims the light they reflect, but not the light they emit
    surfaces, illumination = radiance[..., :n], None
    if normals is not None:
        illumination = _illumination(
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
        )
        emitted = 0.0 if emission is None else torch.as_tensor(np.asarray(emission), **kwargs)[..., :n]
        surfaces = illumination * (surfaces - emitted) + emitted

    return MediumResult(
        radiance=torch.cat([transmittance * surfaces + inscatter, radiance[..., n:]], dim=-1),
        transmittance=transmittance,
        optical_depth=tau,
        inscatter=inscatter,
        illumination=illumination,
    )


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
) -> torch.Tensor:
    """Ratio by which a medium scales the light of the surfaces seen by each pixel, see
    :func:`surface_attenuation <visionsim.medium.surfaces.surface_attenuation>`, of shape (h, w, c)."""
    kwargs: dict[str, Any] = {"dtype": directions.dtype, "device": directions.device}
    normals = normals.to(**kwargs) if torch.is_tensor(normals) else torch.tensor(np.asarray(normals), **kwargs)
    normals = normals[..., :3]
    if normals.shape[:2] != directions.shape[:2]:
        raise ValueError(f"Normals {tuple(normals.shape)} do not match the camera's resolution.")
    if normals_space == "camera":
        rotation = torch.as_tensor(np.asarray(transform_matrix, dtype=float), **kwargs)[:3, :3]
        normals = normals @ (rotation / rotation.norm(dim=0)).T
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Surfaces are lit on the side the camera sees
    normals = torch.where(((normals * directions).sum(dim=-1) > 0)[..., None], -normals, normals)

    seen = torch.isfinite(distance)
    illumination = torch.ones(*distance.shape, len(beta), **kwargs)
    if bool(seen.any()):
        points = origin + directions[seen] * distance[seen][:, None]
        rays = (origin, directions[seen], distance[seen])
        clear, through = surface_irradiance(
            medium, lighting, beta, points, normals[seen], rays=rays, occlusion=occlusion, time=time
        )
        illumination[seen] = surface_attenuation(clear, through)
    return illumination


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
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Shadows:
    """Trace where objects hide suns and the sky along the rays of a frame, to add several media to it.

    Tracing shadows takes most of the time of :func:`apply_medium` with shadow maps, but doesn't depend on the medium,
    so it can be done once per frame and reused for media of similar density, whose light is scattered from roughly
    the same places along rays, see :func:`trace <visionsim.medium.occlusion.trace>`. For instance, to render a frame
    with fogs of several visibilities::

        shadows = trace_shadows(occlusion, depth, camera, pose, media[0], lighting)
        frames = [apply_medium(radiance, depth, camera, pose, m, lighting, shadows=shadows) for m in media]

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
    )
