"""Light that reaches surfaces through a participating medium, which changes how bright the surfaces of a render are.

A frame rendered without the medium shows surfaces lit by suns, the sky and lamps as if nothing were in the way. Through
the medium, their light is attenuated on its way to the surfaces, while the medium's own glow, i.e. the light it
scatters, lights them too. The radiance of each surface is scaled by the ratio of the irradiance it receives with and
without the medium, estimated from the position and the normal of the surface, assuming that it is diffuse:

- Sunlight is attenuated towards the sun, skylight over the sky above the surface, and light reflected by the ground
  over the ground below the horizon, in closed form for an exponential height fog.
- The glow of the medium reaches surfaces from every direction: the light of suns scattered once, from the closed form
  along each direction and with the medium's phase function, as a surface facing the sun receives much more of the
  light that the fog scatters forward than one facing away from it; the light of the sky and the ground scattered once;
  and, with ``multiple_scattering``, light scattered more times, from the tables of :mod:`visionsim.medium.scattering`.
  The glow only depends on the height of surfaces, on the elevation of their normal and on its azimuth relative to
  suns, so it is tabulated once per frame, see :func:`surface_tables`.
- Light from lamps is attenuated towards each of their emitters with a reduced extinction, as the light that the
  medium scatters mostly keeps going forward, towards surfaces: ``1 - albedo · g`` times the extinction, the similarity
  relation of radiative transfer (e.g. Wyman et al., "Similarity relations for anisotropic scattering in monte carlo
  simulations of deeply penetrating neutral particles", 1989). Against Cycles, it lights surfaces better than the
  delta-Eddington approximation, ``1 - albedo · g²``.
- Light that reaches surfaces indirectly, after bouncing off other surfaces, makes up an ambient share of the frame's
  direct light, and is dimmed as the frame's direct light is on average, see :func:`surface_attenuation`.

With shadow maps, surfaces only receive the light of the suns, lamps and cells of the sky they see, and the glow of
suns and of the sky in proportion to the sky they see, as the medium around surfaces that don't see the sky, e.g.
indoors, isn't lit by it either. Otherwise, the glow assumes that surfaces stand in the open, i.e. that objects don't
shadow the medium around them.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import torch

from visionsim.medium.lights import emitters
from visionsim.medium.model import AreaLight, HeightFog, Lighting, Medium
from visionsim.medium.occlusion import (
    Occlusion,
    lamp_map_index,
    lamp_visibility,
    maps_to,
    select_maps,
    visibility,
)
from visionsim.medium.optics import _per_channel, height_fog_sun_inscatter, henyey_greenstein, optical_depth
from visionsim.medium.scattering import (
    _gather_nodes,
    cached_multiple_scattering,
    depth_above_ground,
    ground_radiance,
    ground_source,
    scattered_source,
    sky_source,
)

AMBIENT = 0.2
"""Share of the frame's mean direct irradiance that reaches surfaces indirectly, after bouncing off other surfaces"""
_HEIGHTS = (40, 12.0)
"""Number of heights at which the irradiance of surfaces is tabulated, spread uniformly from the ground up to this many
falloffs above it, as the light scattered more than once"""
_NORMALS = 33
"""Number of vertical components of normals, spread uniformly over [-1, 1], at which the irradiance is tabulated"""
_AZIMUTHS = 17
"""Number of azimuths of normals relative to suns, spread uniformly over [0, π], at which their glow is tabulated"""
_GLOW_DIRECTIONS = (24, 48)
"""Number of elevations per hemisphere and of azimuths of the directions over which the glow of suns is integrated"""


def _unit(vector: Any) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in vector)
    norm = math.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def _clamped_cosine(normal_z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Cosine between normals and directions, clamped to zero, integrated over the azimuths of the directions.

    Args:
        normal_z (torch.Tensor): Vertical component of unit normals.
        mu (torch.Tensor): Vertical component of unit directions, broadcastable with ``normal_z``.

    Returns:
        torch.Tensor: Integral over azimuths, in [0, 2π].
    """
    a = normal_z * mu
    b = (1 - normal_z**2).clamp_min(0).sqrt() * (1 - mu**2).clamp_min(0).sqrt()
    phi = torch.arccos((-a / b.clamp_min(1e-12)).clamp(-1, 1))
    return torch.where(b > 1e-9, 2 * (a * phi + b * torch.sin(phi)), 2 * math.pi * a.clamp_min(0))


class SurfaceTables(NamedTuple):
    """Irradiance of surfaces in an exponential height fog, which only depends on their height and normal."""

    depths: torch.Tensor
    """optical depth of the fog above each tabulated height, from the ground up, of shape (h, c) or (h, 1)"""
    clear: torch.Tensor
    """irradiance from the sky and the ground without the fog, for each vertical component of normals, of shape
    (m, c)"""
    through: torch.Tensor
    """irradiance from the sky and the ground through the fog, and from the glow of the fog that doesn't come from suns
    directly, of shape (h, m, c)"""
    suns: tuple[torch.Tensor, ...]
    """irradiance from the glow of each sun's light scattered once by the fog, for each azimuth of normals relative to
    the sun, of shape (h, m, a, c)"""


def _sun_glow_integral(
    depths: torch.Tensor, depth_ground: torch.Tensor, falloff: float, mu: torch.Tensor, sun_z: float
) -> torch.Tensor:
    """Integral of ``σ · T · T_sun`` from points at each tabulated height along directions of vertical components
    ``mu``, up to the top of the fog or down to the ground, of shape (h, d, c), see :func:`height_fog_sun_inscatter
    <visionsim.medium.optics.height_fog_sun_inscatter>`."""
    c0 = depths[:, None, :]  # (h, 1, c)
    v = mu[None, :, None].expand(len(depths), -1, depths.shape[-1])  # (h, d, c)
    above_ground = falloff * torch.log((depth_ground / depths.clamp_min(1e-30)).clamp_min(1))[:, None, :]
    down = v < 0
    distance = torch.where(down, above_ground / (-v).clamp_min(1e-9), torch.full_like(v, math.inf))
    tau = torch.where(down, (depth_ground - c0).clamp_min(0) / (-v).clamp_min(1e-9), c0 / v.clamp_min(1e-9))
    return height_fog_sun_inscatter(tau, c0 / falloff, falloff, v, distance, sun_z)


def surface_tables(medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int) -> SurfaceTables:
    """Tabulate the irradiance of surfaces in an exponential height fog, against their height and normal.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the scene.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.

    Returns:
        SurfaceTables: Irradiance tables, whose glow includes light scattered more than once only with
        ``multiple_scattering``.
    """
    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    g, albedo = medium.anisotropy, medium.albedo
    sky = _per_channel(lighting.sky, channels, "sky", **kwargs)
    ground = ground_radiance(medium, lighting, beta, channels)
    depth_ground = depth_above_ground(medium, lighting, beta)
    n_heights, height_scale = _HEIGHTS
    depths = depth_ground[None, :] * torch.exp(-torch.linspace(0, height_scale, n_heights, **kwargs))[:, None]
    table = cached_multiple_scattering(medium, lighting, beta, channels) if medium.multiple_scattering else None

    # Light of the sky and the ground through the fog, and the fog's glow, from directions of each elevation
    mu, weights = _gather_nodes(24, **kwargs)
    up = mu > 0
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(16))
    t, w = (t + 1) / 2, w / 2
    arriving = torch.zeros(n_heights, len(mu), channels, **kwargs)
    for h in range(n_heights):
        c0 = depths[h]
        # Transmittance to the top of the fog, or to the ground
        end = torch.where(
            up[:, None],
            torch.exp(-c0 / torch.where(up, mu, 1.0)[:, None]),
            torch.exp(-(depth_ground - c0).clamp_min(0) / mu.abs()[:, None]),
        )
        y = (1 - end)[:, None, :] * t[None, :, None]
        depth = (c0 + mu[:, None, None] * torch.log1p(-y).clamp_min(-80.0)).clamp_min(0)
        source = sky * sky_source(mu, depth, g, n_angle=8, n_azimuth=12)
        source = source + ground * ground_source(mu, depth, depth_ground, g, n_angle=8, n_azimuth=12)
        glow = albedo * (source * w[None, :, None]).sum(dim=1) * (1 - end)
        if table is not None:
            glow = glow + (scattered_source(table, mu, depth) * w[None, :, None]).sum(dim=1) * (1 - end)
        arriving[h] = glow + torch.where(up[:, None], sky * end, ground * end)

    normals_z = torch.linspace(-1, 1, _NORMALS, **kwargs)
    cosines = _clamped_cosine(normals_z[:, None], mu[None, :]) * weights  # (m, d)
    # Without fog, the ground reflects the light of suns and of the sky
    sun_irradiance = sum(
        (_per_channel(s.irradiance, channels, "sun", **kwargs) * max(_unit(s.direction)[2], 0.0) for s in lighting.suns),
        torch.zeros(channels, **kwargs),
    )
    clear_ground = _per_channel(lighting.ground_albedo, channels, "ground albedo", **kwargs) / math.pi
    clear_ground = clear_ground * (sun_irradiance + math.pi * sky)
    clear = sky * (cosines * up).sum(dim=-1)[:, None] + clear_ground * (cosines * ~up).sum(dim=-1)[:, None]
    through = torch.einsum("md,hdc->hmc", cosines, arriving)

    # Glow of each sun over directions of every azimuth, with the sun at azimuth zero, and normals at each azimuth
    elevations, elevation_weights = _gather_nodes(_GLOW_DIRECTIONS[0], **kwargs)
    phi = (torch.arange(_GLOW_DIRECTIONS[1], **kwargs) + 0.5) / _GLOW_DIRECTIONS[1] * 2 * math.pi
    ez, ep = torch.meshgrid(elevations, phi, indexing="ij")
    sphere = torch.stack([(1 - ez**2).sqrt() * torch.cos(ep), (1 - ez**2).sqrt() * torch.sin(ep), ez], -1).reshape(-1, 3)
    solid = (elevation_weights[:, None] * 2 * math.pi / _GLOW_DIRECTIONS[1]).expand_as(ez).reshape(-1)
    nz, na = torch.meshgrid(normals_z, torch.linspace(0, math.pi, _AZIMUTHS, **kwargs), indexing="ij")
    horizontal = (1 - nz**2).clamp_min(0).sqrt()
    normals = torch.stack([horizontal * torch.cos(na), horizontal * torch.sin(na), nz], -1).reshape(-1, 3)
    facing = (normals @ sphere.T).clamp_min(0) * solid  # (m * a, directions)
    suns = []
    for sun in lighting.suns:
        sun_z = _unit(sun.direction)[2]
        if sun_z <= 0:
            suns.append(torch.zeros(n_heights, _NORMALS, _AZIMUTHS, channels, **kwargs))
            continue
        phase = henyey_greenstein(sphere @ sphere.new_tensor([math.sqrt(1 - sun_z**2), 0.0, sun_z]), g)
        integral = _sun_glow_integral(depths, depth_ground, fog.falloff, sphere[:, 2], sun_z) * phase[:, None]
        irradiance = albedo * _per_channel(sun.irradiance, channels, "sun irradiance", **kwargs)
        glow = torch.einsum("nd,hdc->hnc", facing, integral) * irradiance
        suns.append(glow.reshape(n_heights, _NORMALS, _AZIMUTHS, channels))
    return SurfaceTables(depths, clear, through, tuple(suns))


def _interpolate(table: torch.Tensor, *indices: torch.Tensor) -> torch.Tensor:
    """Multilinear interpolation of a table of shape (i, j, ..., c) at fractional indices along its first dimensions."""
    lows = [index.floor().clamp(0, size - 2).long() for index, size in zip(indices, table.shape)]
    fractions = [(index - low)[:, None] for index, low in zip(indices, lows)]
    result = torch.zeros(len(indices[0]), table.shape[-1], dtype=table.dtype, device=table.device)
    for corner in range(2 ** len(indices)):
        weight = torch.ones_like(fractions[0])
        at = []
        for d, (low, fraction) in enumerate(zip(lows, fractions)):
            upper = (corner >> d) & 1
            weight = weight * (fraction if upper else 1 - fraction)
            at.append(low + upper)
        result = result + weight * table[tuple(at)]
    return result


def _sky_seen(occlusion: Occlusion, normals: torch.Tensor, origin, directions, distance) -> torch.Tensor:
    """Share of the irradiance of a uniform sky that reaches surfaces from the cells of the sky they see, of shape (n,).

    Surfaces are looked up in the shadow maps of the cells along the camera rays that see them."""
    maps = maps_to(occlusion.sky_maps, device=normals.device, dtype=torch.float32)
    towards = maps.axes[:, 2].to(normals)  # (k, 3)
    edges, counts = occlusion.sky_edges, occlusion.sky_counts
    solid = [2 * math.pi * (high - low) / count for low, high, count in zip(edges[:-1], edges[1:], counts)]
    solid_angles = torch.as_tensor([a for a, count in zip(solid, counts) for _ in range(count)]).to(normals)
    weights = (normals @ towards.T).clamp_min(0) * solid_angles  # (n, k)
    seen = []
    for i in range(0, len(normals), 4096):
        rays = (directions[i : i + 4096].float(), distance[i : i + 4096].float()[:, None])
        lit = visibility(maps, origin.float(), *rays)[:, 0].to(normals)
        seen.append((lit * weights[i : i + 4096]).sum(dim=-1) / weights[i : i + 4096].sum(dim=-1).clamp_min(1e-12))
    return torch.cat(seen)


def surface_irradiance(
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    rays: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    occlusion: Occlusion | None = None,
    time: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Irradiance of surfaces from suns, the sky and lamps, without and with a participating medium.

    Args:
        medium (Medium): Participating medium.
        lighting (Lighting): Lighting of the scene, whose colors have one value per channel, or a single one.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        points (torch.Tensor): Points on surfaces, of shape (n, 3).
        normals (torch.Tensor): Unit world-space normals of the surfaces, of shape (n, 3).
        rays (tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, optional): Origin, of shape (3,), unit
            directions, of shape (n, 3), and lengths, of shape (n,), of the camera rays that see the points, through
            which they are looked up in the shadow maps of suns and of the sky. Defaults to None.
        occlusion (Occlusion | None, optional): Shadow maps of the scene, through which surfaces only receive the light
            of what they see. Those of suns and of the sky require ``rays``. Defaults to None.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Irradiance without and with the medium, of shape (n, c).
    """
    n = len(beta)
    kwargs: dict[str, Any] = {"dtype": points.dtype, "device": points.device}
    sky = _per_channel(lighting.sky, n, "sky", **kwargs)
    normal_z = normals[:, 2].clamp(-1, 1)
    normal_index = (normal_z + 1) / 2 * (_NORMALS - 1)

    # Share of the sky that surfaces see, and through which they receive the glow of suns and of the sky
    open_sky = torch.ones(len(points), 1, **kwargs)
    if rays is not None and occlusion is not None and len(occlusion.sky_maps.texels):
        open_sky = _sky_seen(occlusion, normals, *rays)[:, None]

    # Suns and the sky are only attenuated by height fogs, through which their glow is modeled too
    tables, column, height_index = None, None, None
    if medium.sun_attenuation and isinstance(fog := medium.components[0], HeightFog):
        channels = slice(0, 1) if bool((beta == beta[0]).all()) else slice(None)
        tables = surface_tables(medium, lighting, beta[channels], n)
        column = (
            beta[channels] * fog.density * fog.falloff * torch.exp(-(points[:, 2:3] - fog.base_height) / fog.falloff)
        )
        n_heights, height_scale = _HEIGHTS
        height = torch.log(tables.depths[0, 0] / column[:, 0].clamp_min(1e-30)).clamp(0, height_scale)
        height_index = height / height_scale * (n_heights - 1)
        clear = open_sky * _interpolate(tables.clear, normal_index)
        through = open_sky * _interpolate(tables.through, height_index, normal_index)
    else:
        mu, weights = _gather_nodes(24, **kwargs)
        seen = (_clamped_cosine(normal_z[:, None], mu[None, :]) * weights * (mu > 0)).sum(dim=-1)[:, None]
        clear = through = open_sky * sky * seen

    for k, sun in enumerate(lighting.suns):
        towards = _unit(sun.direction)
        irradiance = _per_channel(sun.irradiance, n, "sun irradiance", **kwargs)
        lit = (normals @ normals.new_tensor(towards)).clamp_min(0)[:, None] * irradiance
        if rays is not None and occlusion is not None and k < len(occlusion.sun_maps.texels):
            maps = select_maps(maps_to(occlusion.sun_maps, device=points.device), k)
            origin, directions, distance = (tensor.float() for tensor in rays)
            lit = lit * visibility(maps, origin, directions, distance[:, None])[:, 0].to(lit)
        clear = clear + lit
        if tables is None or column is None or height_index is None:
            through = through + lit
            continue
        if towards[2] > 0:
            through = through + lit * torch.exp(-column / towards[2])
        # Glow of the sun's light scattered once, for normals at their azimuth from the sun
        azimuth = torch.atan2(normals[:, 1], normals[:, 0]) - math.atan2(towards[1], towards[0])
        azimuth_index = torch.remainder(azimuth + math.pi, 2 * math.pi).sub(math.pi).abs() / math.pi * (_AZIMUTHS - 1)
        through = through + open_sky * _interpolate(tables.suns[k], height_index, normal_index, azimuth_index)

    # Lamps, whose light is attenuated with a reduced extinction, as scattered light mostly keeps going forward
    reduced = 1 - medium.albedo * medium.anisotropy
    lamp_maps = None
    if occlusion is not None and occlusion.lamp_maps is not None:
        lamp_maps = maps_to(occlusion.lamp_maps, device=points.device)
    for lamp in lighting.lamps:
        seen_lamp = torch.ones(len(points), 1, **kwargs)
        center = torch.as_tensor(lamp.position, **kwargs)
        if isinstance(lamp, AreaLight):
            # Maps of area lights are rendered just in front of them
            center = center + 1e-3 * center.new_tensor(_unit(lamp.direction))
        if lamp_maps is not None and (index := lamp_map_index(lamp_maps, center)) is not None:
            seen_lamp = lamp_visibility(lamp_maps, index, points.float())[:, None].to(points)
        for emitter in emitters(lamp, n, samples=4, **kwargs):
            to_light = emitter.position - points
            # Distances are clamped at the emitter's radius, unlike directions
            distance = to_light.norm(dim=-1).clamp_min(1e-12)
            r = distance.clamp_min(max(emitter.radius, 1e-3))
            u = to_light / distance[:, None]
            factor = (normals * u).sum(dim=-1).clamp_min(0) / (r * r)
            if emitter.axis is not None:
                cosine = -(u @ emitter.axis)
                factor = factor * (cosine >= emitter.cone)
                if emitter.profile is not None:
                    factor = factor * emitter.profile(cosine)
            if emitter.falloff is not None:
                factor = factor * emitter.falloff(r)
            if emitter.pattern is not None:
                factor = factor * emitter.pattern(-u)
            lit = seen_lamp * factor[:, None] * emitter.intensity
            depth = optical_depth(medium, points, u, distance, time=time)[:, None]
            clear, through = clear + lit, through + lit * torch.exp(-reduced * depth * beta)
    return clear, through


def surface_attenuation(clear: torch.Tensor, through: torch.Tensor, ambient: float = AMBIENT) -> torch.Tensor:
    """Ratio by which a medium scales the light of surfaces, given their irradiance without and with it.

    Light that reaches surfaces indirectly, after bouncing off other surfaces, makes up an ``ambient`` share of the mean
    irradiance of all surfaces, and is dimmed as their irradiance is on average, so that surfaces that receive little
    direct light are dimmed as the frame is on average.

    Args:
        clear (torch.Tensor): Irradiance of surfaces without the medium, of shape (n, c).
        through (torch.Tensor): Irradiance of surfaces with the medium, of shape (n, c).
        ambient (float, optional): Share of the mean irradiance that reaches surfaces indirectly. Defaults to
            :data:`AMBIENT`.

    Returns:
        torch.Tensor: Ratio of the light of surfaces with and without the medium, of shape (n, c).
    """
    if not len(clear):
        return torch.ones_like(clear)
    indirect = ambient * clear.mean(dim=0)
    mean = through.sum(dim=0) / clear.sum(dim=0).clamp_min(1e-30)
    return (through + mean * indirect) / (clear + indirect).clamp_min(1e-30)
