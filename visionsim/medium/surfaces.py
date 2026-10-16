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
  and, with ``multiple_scattering``, light scattered more times, from the tables of :mod:`visionsim.medium.scattering`,
  which is still brighter towards suns, and the ground lit by the glow too. The glow only depends on the height of
  surfaces, on the elevation of their normal and on its azimuth relative to suns, so it is tabulated once per medium
  and lighting, see :func:`surface_tables`.
- Light from lamps is attenuated towards each of their emitters with a reduced extinction, as the light that the
  medium scatters mostly keeps going forward, towards surfaces: ``1 - albedo · g`` times the extinction, the similarity
  relation of radiative transfer (e.g. Wyman et al., "Similarity relations for anisotropic scattering in monte carlo
  simulations of deeply penetrating neutral particles", 1989). Against Cycles, it lights surfaces better than the
  delta-Eddington approximation, ``1 - albedo · g²``. With ``multiple_scattering``, their light is attenuated with the
  full extinction instead, and their glow lights surfaces from every direction, in the lamp's shadows too: the light
  of the lamp scattered once and more, and the light reflected by surfaces and scattered, as the halos of lamps
  tabulate it, see :func:`halo_glow <visionsim.medium.halos.halo_glow>`, which fades out at the extent of halos, beyond
  which the similarity relation takes over. Far from the lamp, the similarity relation misses much of the glow, such
  as the light that the lamp sends up, which the medium scatters back down: in the night scene of ``examples/medium``,
  the ground 8 m from the lamp's foot receives 1.08 times as much light as without fog, against 0.91 times with the
  similarity relation, and 0.61 times directly.
- Light that reaches surfaces indirectly, after bouncing off other surfaces, makes up an ambient share of the frame's
  direct light, and is dimmed as the frame's direct light is on average, see :func:`surface_attenuation`. Given the
  mean albedo of the ground and of objects, it is at least the rest of the light that their radiance implies they
  received: next to brightly lit surfaces, it can be most of their light, by which the glow of the medium is weighed,
  so that surfaces reflect as much of the glow as their albedo does, rather than as much as the ambient share alone
  implies. Given the light that the surface of each pixel received, measured by its irradiance, such as Blender's
  diffuse light passes, or by its albedo, such as Blender's diffuse color pass, along with its radiance, it is exactly
  that rest, which the ambient share only guesses, and each surface reflects the glow with its own albedo, which
  follows textures and colors, rather than with the mean albedo. The radiance of glossy surfaces also holds their
  specular reflection, which their diffuse albedo leaves out, so that the light measured by the albedo is too high, and
  the glow they reflect too low, by as much as their specular reflection would add to it.

With shadow maps, surfaces only receive the light of the suns, lamps and cells of the sky they see, and the glow of
suns and of the sky in proportion to the sky they see, as the medium around surfaces that don't see the sky, e.g.
indoors, isn't lit by it either. Surfaces in the shadow of a sun lack the glow of its light scattered once by the
medium in the shadow: it comes from the medium along directions close to the sun's, as the phase function has it,
which are looked up in the sun's map. Deep in a shadow, it is all gone, while near its edges, and on the sides of
objects turned away from the sun, whose medium the sun mostly lights, much of it is left. In the day scene of
``examples/medium``, the shadowed ground is within 1% of Cycles', against 12% too bright with all of the glow, and 9%
too dark without any of it. Otherwise, the glow assumes that surfaces stand in the open, i.e. that objects don't
shadow the medium around them.

The radiance of objects that reflect light into the medium, see :func:`object_radiance`, is found the same way, but
for its glow, which surfaces in shadow lack entirely, as looking it up along many directions would cost too much for
the many points that reflect light.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import torch

from visionsim.medium.lights import emitters
from visionsim.medium.model import AreaLight, Effects, EmissiveSurface, HeightFog, Lamp, Lighting, Medium
from visionsim.medium.occlusion import (
    Occlusion,
    ShadowMaps,
    lamp_map_index,
    lamp_visibility,
    maps_to,
    select_maps,
    visibility,
)
from visionsim.medium.optics import (
    _per_channel,
    height_fog_optical_depth,
    height_fog_sun_inscatter,
    henyey_greenstein,
    optical_depth,
)
from visionsim.medium.scattering import (
    ScatteringTable,
    SunScattering,
    _gather_nodes,
    cached_multiple_scattering,
    cosine_harmonics,
    depth_above_ground,
    environment_key,
    environment_lit,
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
_SHADOW_NODES = (3, 4)
"""Number of angles in each of the three pieces of the range of angles from a sun, and of azimuths, of the directions
along which the visibility of the sun is averaged over the glow of its light scattered once, for surfaces in shadow,
see :func:`_glow_visibility`"""
_SKY_SEEN_CHUNKS = (4096, 2**22)
"""Number of points looked up at once in the maps of the sky on the CPU, small enough for the values of all maps to fit in
its caches, and of values of all maps on a GPU, which then launches fewer kernels"""

Glow = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
"""Glow of the medium around a lamp, given points of shape (n, 3) and unit normals of shape (n, 3): irradiance of
surfaces from it, of shape (n, c), and share of the lamp's light whose glow it models at the points, of shape (n,), see
:func:`halo_glow <visionsim.medium.halos.halo_glow>`"""


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
    directly, if surfaces receive it, of shape (h, m, c)"""
    suns: tuple[torch.Tensor, ...]
    """irradiance from the glow of each sun's light scattered once by the fog, and of how its light scattered more than
    once varies with azimuth, for each azimuth of normals relative to the sun, of shape (h, m, a, c), or none when
    surfaces receive no glow"""
    once: tuple[torch.Tensor, ...] = ()
    """irradiance from the glow of each sun's light scattered once by the fog alone, which surfaces in the sun's shadow
    lack, of shape (h, m, a, c)"""


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


def surface_tables(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int, glow: bool = True
) -> SurfaceTables:
    """Tabulate the irradiance of surfaces in an exponential height fog, against their height and normal.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the scene.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.
        glow (bool, optional): If true, surfaces receive the glow of the fog too. Otherwise, they only receive the light
            of suns, the sky and the ground attenuated by the fog, and the tables have no glow of suns. Defaults to
            True.

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
    table = cached_multiple_scattering(medium, lighting, beta, channels) if glow and medium.multiple_scattering else None

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
        arriving[h] = torch.where(up[:, None], sky * end, ground * end)
        if not glow:
            continue
        y = (1 - end)[:, None, :] * t[None, :, None]
        depth = (c0 + mu[:, None, None] * torch.log1p(-y).clamp_min(-80.0)).clamp_min(0)
        source = sky * sky_source(mu, depth, g, n_angle=8, n_azimuth=12)
        source = source + ground * ground_source(mu, depth, depth_ground, g, n_angle=8, n_azimuth=12)
        glowing = albedo * (source * w[None, :, None]).sum(dim=1) * (1 - end)
        if table is not None:
            glowing = glowing + (scattered_source(table, mu, depth) * w[None, :, None]).sum(dim=1) * (1 - end)
        arriving[h] = glowing + arriving[h]

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
    if not glow:
        return SurfaceTables(depths, clear, through, ())

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
    suns, once = [], []
    varying = {sun.sun_index: sun for sun in table.suns} if table is not None else {}
    for k, sun in enumerate(lighting.suns):
        sun_z = _unit(sun.direction)[2]
        if sun_z <= 0:
            suns.append(torch.zeros(n_heights, _NORMALS, _AZIMUTHS, channels, **kwargs))
            once.append(suns[-1])
            continue
        phase = henyey_greenstein(sphere @ sphere.new_tensor([math.sqrt(1 - sun_z**2), 0.0, sun_z]), g)
        integral = _sun_glow_integral(depths, depth_ground, fog.falloff, sphere[:, 2], sun_z) * phase[:, None]
        irradiance = albedo * _per_channel(sun.irradiance, channels, "sun irradiance", **kwargs)
        sun_glow = torch.einsum("nd,hdc->hnc", facing, integral) * irradiance
        sun_glow = sun_glow.reshape(n_heights, _NORMALS, _AZIMUTHS, channels)
        once.append(sun_glow)
        if table is not None and k in varying:
            sun_glow = sun_glow + _scattered_glow(table, varying[k], depths, normals_z)
        suns.append(sun_glow)
    return SurfaceTables(depths, clear, through, tuple(suns), tuple(once))


def _scattered_glow(
    table: ScatteringTable, sun: SunScattering, depths: torch.Tensor, normals_z: torch.Tensor
) -> torch.Tensor:
    """Irradiance of surfaces from the variations of a sun's light scattered more than once with the azimuth from the
    sun's, of shape (h, m, a, c), for each tabulated height, vertical component of normals and azimuth of normals
    relative to the sun.

    Radiance that varies as ``cos(kφ)`` with the azimuth of the directions it comes from gives surfaces whose normal is
    at an azimuth ``ψ`` an irradiance that varies as ``cos(kψ)``, in proportion to the ``k``-th harmonic of the clamped
    cosine between normals and directions, see :func:`cosine_harmonics
    <visionsim.medium.scattering.cosine_harmonics>`.
    """
    modes, n_heights, n_directions, channels = sun.arriving.shape
    kwargs: dict[str, Any] = {"dtype": sun.arriving.dtype, "device": sun.arriving.device}
    _, weights = _gather_nodes(n_directions // 2, **kwargs)
    cosine = cosine_harmonics(normals_z, table.directions, lambda cos: cos.clamp_min(0), modes)[1:].to(**kwargs)
    # Light arriving at the heights of the surfaces' tables, interpolated between those of light scattered more than once
    index = table.height_index(depths).expand(-1, channels)  # (h, c)
    low = index.floor().clamp(0, n_heights - 2).long()[:, None, :]
    fraction = (index[:, None, :] - low)[None]
    rows = torch.arange(n_directions, device=index.device)[None, :, None]
    columns = torch.arange(channels, device=index.device)
    below, above = sun.arriving[:, low, rows, columns], sun.arriving[:, low + 1, rows, columns]  # (modes, h, d, c)
    harmonics = torch.einsum("knd,khdc,d->hnkc", cosine, below + fraction * (above - below), weights)
    azimuths = torch.linspace(0, math.pi, _AZIMUTHS, **kwargs)
    return torch.einsum("hnkc,ak->hnac", harmonics, torch.cos(azimuths[:, None] * torch.arange(1, modes + 1, **kwargs)))


@lru_cache(maxsize=8)
def _cached_surface_tables(
    medium: str, lighting: str, beta: tuple, channels: int, dtype: torch.dtype, device: str, glow: bool
) -> SurfaceTables:
    return surface_tables(
        Medium.model_validate_json(medium),
        Lighting.model_validate_json(lighting),
        torch.tensor(beta, dtype=dtype, device=device),
        channels,
        glow,
    )


def cached_surface_tables(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int, glow: bool = True
) -> SurfaceTables:
    """Same as :func:`surface_tables`, cached for the last few media and lightings, as it does not depend on the camera
    and can be reused for every frame."""
    return _cached_surface_tables(
        medium.model_dump_json(),
        environment_key(lighting),
        tuple(beta.tolist()),
        channels,
        beta.dtype,
        str(beta.device),
        glow,
    )


def _interpolate(table: torch.Tensor, *indices: torch.Tensor, at_once: bool | None = None) -> torch.Tensor:
    """Multilinear interpolation of a table of shape (i, j, ..., c) at fractional indices along its first dimensions,
    whose corners around each point are gathered at once, which launches fewer kernels, by default on a GPU, or one
    after another, which moves less memory on the CPU, for the same result."""
    n, dims = len(indices[0]), len(indices)
    lows = [index.floor().clamp(0, size - 2).long() for index, size in zip(indices, table.shape)]
    fractions = [index - low for index, low in zip(indices, lows)]
    if not (table.is_cuda if at_once is None else at_once):
        result = torch.zeros(n, table.shape[-1], dtype=table.dtype, device=table.device)
        for corner in range(2**dims):
            weight = torch.ones_like(fractions[0])
            at = []
            for d, (low, fraction) in enumerate(zip(lows, fractions)):
                upper = (corner >> d) & 1
                weight = weight * (fraction if upper else 1 - fraction)
                at.append(low + upper)
            result = result + weight[:, None] * table[tuple(at)]
        return result
    # Corners along a new dimension, numbered by the bits of their upper indices, the lowest bit for the first index,
    # in the table flattened over its first dimensions
    index, weight = None, None
    for d, (low, fraction) in enumerate(zip(lows, fractions)):
        shape = (n,) + (1,) * (dims - 1 - d) + (2,) + (1,) * d
        rows = (torch.stack([low, low + 1], dim=-1) * math.prod(table.shape[d + 1 : -1])).reshape(shape)
        weights = torch.stack([1 - fraction, fraction], dim=-1).reshape(shape)
        index = rows if index is None else index + rows
        weight = weights if weight is None else weight * weights
    assert index is not None and weight is not None
    products = weight.reshape(n, -1, 1) * table.reshape(-1, table.shape[-1])[index.reshape(n, -1)]
    result = products[:, 0]
    for corner in range(1, products.shape[1]):
        result = result + products[:, corner]
    return result


def _glow_visibility(
    maps: ShadowMaps,
    medium: Medium,
    beta: torch.Tensor,
    towards: tuple[float, float, float],
    ground_height: float,
    points: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    """Visible fraction of the glow of a sun's light scattered once that reaches surfaces, according to the sun's map,
    of shape (n, 1).

    The glow comes from the medium along directions around the sun's, which follow the phase function towards it, see
    :func:`phase_directions <visionsim.medium.reflections.phase_directions>`, weighted by the clamped cosine with the
    surfaces' normals and by the light of the sun that the medium scatters along them, up to the top of the fog or
    down to the ground, in closed form. Along each direction, the sun is looked up where half of that light has been
    scattered, as the medium is lit or in shadow over stretches longer than the mean free path."""
    # This module is imported by that of reflections
    from visionsim.medium.reflections import phase_directions

    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    # The directions only depend on the sun's, and are found on the CPU, which spares many tiny operators on a GPU
    sun = torch.tensor(towards, dtype=torch.float64)
    pieces = [phase_directions(sun, medium.anisotropy, *_SHADOW_NODES, below=below) for below in (False, True)]
    nodes, weights = (torch.cat(parts).to(points) for parts in zip(*pieces))  # (d, 3), (d,)
    sun = points.new_tensor(towards)
    beta = beta[:1]
    v_z = nodes[None, :, 2:3].expand(len(points), -1, 1)
    height = (points[:, 2:3] - ground_height).clamp_min(0)
    length = torch.where(
        v_z[..., 0] < 0, height / (-v_z[..., 0]).clamp_min(1e-9), torch.full_like(v_z[..., 0], math.inf)
    )
    depth = height_fog_optical_depth(fog, points[:, None, :], nodes[None], length)[..., None] * beta  # (n, d, 1)
    extinction = beta * fog.density * torch.exp(-(points[:, 2:3] - fog.base_height) / fog.falloff)  # (n, 1)
    scattered = height_fog_sun_inscatter(depth, extinction[:, None], fog.falloff, v_z, length[..., None], sun[2])
    weight = (normals @ nodes.T).clamp_min(0) * weights * scattered[..., 0]  # (n, d)
    # Distance along each direction at which half of the light it brings has been scattered, inverting the optical
    # depth of the height fog
    half = -torch.log1p(torch.expm1(-depth[..., 0]) / 2) / beta / fog.density
    k = torch.exp(-(points[:, 2:3] - fog.base_height) / fog.falloff)
    x = (half * v_z[..., 0] / (k * fog.falloff)).clamp_max(1 - 1e-7)
    small = x.abs() < 1e-6
    ratio = torch.where(small, 1 + x / 2, -torch.log1p(-x) / torch.where(small, torch.ones_like(x), x))
    middle = points[:, None, :] + torch.minimum(half / k * ratio, length)[..., None] * nodes[None]
    ones = points.new_ones(middle.numel() // 3, 1, dtype=maps.depths.dtype)
    seen = visibility(maps, ones.new_zeros(3), middle.reshape(-1, 3).to(ones), ones)[:, 0, 0].reshape(weight.shape)
    total = weight.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, (weight * seen.to(weight)).sum(dim=-1, keepdim=True) / total.clamp_min(1e-30), 1.0)


@lru_cache(maxsize=8)
def _cell_solid_angles(
    edges: tuple[float, ...], counts: tuple[int, ...], dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Solid angle of each cell of the sky, of shape (k,)."""
    solid = [2 * math.pi * (high - low) / count for low, high, count in zip(edges[:-1], edges[1:], counts)]
    return torch.as_tensor([a for a, count in zip(solid, counts) for _ in range(count)]).to(dtype=dtype, device=device)


def _sky_seen(occlusion: Occlusion, normals: torch.Tensor, origin, directions, distance) -> torch.Tensor:
    """Share of the irradiance of a uniform sky that reaches surfaces from the cells of the sky they see, of shape (n,).

    Surfaces are looked up in the shadow maps of the cells along the camera rays that see them."""
    maps = maps_to(occlusion.sky_maps, device=normals.device, dtype=torch.float32)
    towards = maps.axes[:, 2].to(normals)  # (k, 3)
    edges, counts = tuple(occlusion.sky_edges), tuple(occlusion.sky_counts)
    solid_angles = _cell_solid_angles(edges, counts, normals.dtype, str(normals.device))
    weights = (normals @ towards.T).clamp_min(0) * solid_angles  # (n, k)
    seen = []
    points, values = _SKY_SEEN_CHUNKS
    size = max(1, values // len(towards)) if normals.is_cuda else points
    for i in range(0, len(normals), size):
        rays = (directions[i : i + size].float(), distance[i : i + size].float()[:, None])
        lit = visibility(maps, origin.float(), *rays)[:, 0].to(normals)
        seen.append((lit * weights[i : i + size]).sum(dim=-1) / weights[i : i + size].sum(dim=-1).clamp_min(1e-12))
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
    glows: Sequence[Glow | None] | None = None,
    effects: Effects | None = None,
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
            of what they see. The points are looked up in the maps of suns along ``rays`` if given, and directly
            otherwise, while those of the sky require ``rays``: without them, surfaces see the whole sky. Defaults to
            None.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        glows (Sequence[Glow | None] | None, optional): For each of the lighting's lamps, a function giving the
            irradiance of surfaces from the glow of the medium around the lamp, given points of shape (n, 3) and unit
            normals of shape (n, 3), of shape (n, c), and the share of the lamp's light whose glow it models, of shape
            (n,), such as :func:`halo_glow <visionsim.medium.halos.halo_glow>`, or None. That share of the lamp's
            light is then attenuated with the full extinction, and the glow added, rather than attenuated with the
            reduced extinction of the similarity relation, see :func:`lamp_irradiance`. Defaults to None.
        effects (Effects | None, optional): Terms of the model to compute, see :class:`Effects
            <visionsim.medium.model.Effects>`, of which those of surfaces: without ``surface_shadows``, surfaces receive
            the light of suns and lamps whose maps don't see them, and without ``surface_sky_occlusion``, the light of
            the whole sky; without ``daylight_glow``, they don't receive the glow of suns and the sky, and without
            ``glow_shadows``, surfaces that a sun's map doesn't see still receive the glow of its light scattered once.
            With it, those surfaces receive the part of that glow that comes from the medium that the sun lights around
            them, along directions close to the sun's, with ``glow_shadow_edges``, and lack all of it otherwise, as
            deep in a shadow. Defaults to None, i.e. all of them.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Irradiance without and with the medium, of shape (n, c).
    """
    n = len(beta)
    kwargs: dict[str, Any] = {"dtype": points.dtype, "device": points.device}
    effects = effects or Effects()
    sky = _per_channel(lighting.sky, n, "sky", **kwargs)
    normal_z = normals[:, 2].clamp(-1, 1)
    normal_index = (normal_z + 1) / 2 * (_NORMALS - 1)

    # Share of the sky that surfaces see, and through which they receive the glow of suns and of the sky, if they shine
    open_sky = torch.ones(len(points), 1, **kwargs)
    seeing = rays is not None and occlusion is not None and len(occlusion.sky_maps.texels) and environment_lit(lighting)
    if effects.surface_sky_occlusion and seeing:
        assert rays is not None and occlusion is not None
        open_sky = _sky_seen(occlusion, normals, *rays)[:, None]

    # Suns and the sky are only attenuated by height fogs, through which their glow is modeled too
    tables, column, height_index = None, None, None
    if medium.sun_attenuation and isinstance(fog := medium.components[0], HeightFog):
        channels = slice(0, 1) if bool((beta == beta[0]).all()) else slice(None)
        tables = cached_surface_tables(medium, lighting, beta[channels], n, glow=effects.daylight_glow)
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
        # Whether the sun's map sees the surfaces, which then receive its light, and the glow of its light scattered
        # once by the medium around them
        seen = None
        shading = effects.surface_shadows or (effects.daylight_glow and effects.glow_shadows)
        if shading and occlusion is not None and k < len(occlusion.sun_maps.texels):
            maps = select_maps(maps_to(occlusion.sun_maps, device=points.device), k)
            if rays is not None:
                origin, directions, distance = (tensor.float() for tensor in rays)
                seen = visibility(maps, origin, directions, distance[:, None])[:, 0].to(lit)
            else:
                ones = points.new_ones(len(points), 1, dtype=torch.float32)
                seen = visibility(maps, ones.new_zeros(3), points.float(), ones)[:, 0].to(lit)
            if effects.surface_shadows:
                lit = lit * seen
        clear = clear + lit
        if tables is None or column is None or height_index is None:
            through = through + lit
            continue
        if towards[2] > 0:
            through = through + lit * torch.exp(-column / towards[2])
        if not tables.suns:
            # Without the glow of the medium
            continue
        # Glow of the sun's light scattered once, for normals at their azimuth from the sun, which surfaces in the sun's
        # shadow lack, as the medium around them is in the shadow too
        azimuth = torch.atan2(normals[:, 1], normals[:, 0]) - math.atan2(towards[1], towards[0])
        azimuth_index = torch.remainder(azimuth + math.pi, 2 * math.pi).sub(math.pi).abs() / math.pi * (_AZIMUTHS - 1)
        glow = _interpolate(tables.suns[k], height_index, normal_index, azimuth_index)
        if effects.glow_shadows and seen is not None and tables.once:
            glow_seen = seen
            shadowed = (seen[:, 0] < 1).nonzero()[:, 0]
            if effects.glow_shadow_edges and len(shadowed) and occlusion is not None:
                maps = select_maps(maps_to(occlusion.sun_maps, device=points.device), k)
                part = _glow_visibility(
                    maps, medium, beta, towards, lighting.ground_height, points[shadowed], normals[shadowed]
                )
                glow_seen = seen.index_put((shadowed,), torch.maximum(part, seen[shadowed]))
            glow = glow - (1 - glow_seen) * _interpolate(tables.once[k], height_index, normal_index, azimuth_index)
        through = through + open_sky * glow

    # Lamps, whose light is attenuated with a reduced extinction, see `lamp_irradiance`, in the shadows of their maps.
    # Maps of emissive surfaces see through them
    found = (
        {False: None, True: None}
        if occlusion is None or not effects.surface_shadows
        else {False: occlusion.lamp_maps, True: occlusion.emissive_maps}
    )
    lamp_maps = {k: maps_to(m, device=points.device) if m is not None else None for k, m in found.items()}
    tolerance = occlusion.lamp_tolerance if occlusion is not None else 0.01
    for k, lamp in enumerate(lighting.lamps):
        seen_lamp = torch.ones(len(points), 1, **kwargs)
        center = torch.as_tensor(lamp.position, **kwargs)
        if isinstance(lamp, AreaLight):
            # Maps of area lights are rendered just in front of them
            center = center + 1e-3 * center.new_tensor(_unit(lamp.direction))
        chosen = lamp_maps[isinstance(lamp, EmissiveSurface)]
        if chosen is not None and (index := lamp_map_index(chosen, center, tolerance)) is not None:
            seen_lamp = lamp_visibility(chosen, index, points.float())[:, None].to(points)
        glow = glows[k] if glows is not None else None
        if glow is None:
            lit, dimmed = lamp_irradiance(medium, lamp, beta, points, normals, time)
            clear, through = clear + seen_lamp * lit, through + seen_lamp * dimmed
            continue
        # Where the glow is modeled, the lamp's light is attenuated with the full extinction, and the glow reaches
        # surfaces in the lamp's shadows too
        irradiance, share = glow(points, normals)
        lit, dimmed = lamp_irradiance(medium, lamp, beta, points, normals, time, similarity=1 - share.to(points))
        clear, through = clear + seen_lamp * lit, through + seen_lamp * dimmed + irradiance.to(through)
    return clear, through


def object_radiance(
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    occlusion: Occlusion | None = None,
) -> torch.Tensor:
    """Radiance of objects lit through a medium by suns, the sky and the medium's glow, as Lambertian surfaces of the
    lighting's ``object_albedo``, or of the ground's albedo if it has none.

    Points are looked up in the shadow maps of suns, and see the whole sky. Those in a sun's shadow lack the glow of its
    light scattered once, as deep in a shadow, see :func:`surface_irradiance`. Lamps are left out, as the light of lamps
    that surfaces reflect into the medium is modeled by :mod:`visionsim.medium.reflections`. As objects reflect light
    from many points, the irradiance of :func:`surface_tables` is interpolated between heights only, at the nearest
    tabulated normal, which is much cheaper.

    Args:
        medium (Medium): Participating medium.
        lighting (Lighting): Lighting of the scene, whose colors have one value per channel, or a single one.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        points (torch.Tensor): Points on the surfaces of objects, of shape (n, 3).
        normals (torch.Tensor): Unit normals of the surfaces, of shape (n, 3).
        occlusion (Occlusion | None, optional): Shadow maps of the scene, whose suns' maps cast shadows onto the
            objects. Defaults to None.

    Returns:
        torch.Tensor: Radiance of the surfaces, of shape (n, c).
    """
    kwargs: dict[str, Any] = {"dtype": points.dtype, "device": points.device}
    n = len(beta)
    albedo = _per_channel(
        lighting.ground_albedo if lighting.object_albedo is None else lighting.object_albedo,
        n,
        "object albedo",
        **kwargs,
    )
    daylight = lighting.model_copy(update={"points": [], "spots": [], "areas": [], "emissive": []})
    fog = medium.components[0]
    if not medium.sun_attenuation or not isinstance(fog, HeightFog):
        _, through = surface_irradiance(
            medium, daylight, beta, points, normals, occlusion=occlusion, effects=Effects(glow_shadow_edges=False)
        )
        return albedo / math.pi * through

    channels = slice(0, 1) if bool((beta == beta[0]).all()) else slice(None)
    tables = cached_surface_tables(medium, daylight, beta[channels], n)
    column = beta[channels] * fog.density * fog.falloff * torch.exp(-(points[:, 2:3] - fog.base_height) / fog.falloff)
    n_heights, height_scale = _HEIGHTS
    height = torch.log(tables.depths[0, 0] / column[:, 0].clamp_min(1e-30)).clamp(0, height_scale)
    height = height / height_scale * (n_heights - 1)
    low = height.floor().clamp(0, n_heights - 2).long()
    fraction = (height - low)[:, None]
    normal = ((normals[:, 2].clamp(-1, 1) + 1) / 2 * (_NORMALS - 1)).round().long()

    def at_heights(table: torch.Tensor, *index: torch.Tensor) -> torch.Tensor:
        below = table[(low, normal, *index)]
        return below + fraction * (table[(low + 1, normal, *index)] - below)

    radiance = at_heights(tables.through)
    for k, sun in enumerate(lighting.suns):
        towards = _unit(sun.direction)
        irradiance = _per_channel(sun.irradiance, n, "sun irradiance", **kwargs)
        seen = torch.ones(len(points), 1, **kwargs)
        if occlusion is not None and k < len(occlusion.sun_maps.texels):
            maps = select_maps(maps_to(occlusion.sun_maps, device=points.device), k)
            ones = points.new_ones(len(points), 1, dtype=torch.float32)
            seen = visibility(maps, ones.new_zeros(3), points.float(), ones)[:, 0].to(points)
        if towards[2] > 0:
            lit = (normals @ normals.new_tensor(towards)).clamp_min(0)[:, None] * irradiance * seen
            radiance = radiance + lit * torch.exp(-column / towards[2])
        azimuth = torch.atan2(normals[:, 1], normals[:, 0]) - math.atan2(towards[1], towards[0])
        azimuth = torch.remainder(azimuth + math.pi, 2 * math.pi).sub(math.pi).abs() / math.pi * (_AZIMUTHS - 1)
        index = azimuth.round().long()
        radiance = radiance + at_heights(tables.suns[k], index) - (1 - seen) * at_heights(tables.once[k], index)
    return albedo / math.pi * radiance


def ground_shares(medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int) -> list[torch.Tensor | None]:
    """Share of the radiance of the ground that each sun brings, which the ground lacks in the sun's shadow.

    The sun lights the ground directly and, with ``multiple_scattering``, through the glow of its light scattered once
    by the medium, which the ground lacks in the sun's shadow too, as the medium around it is in the shadow as well,
    see :func:`ground_radiance <visionsim.medium.scattering.ground_radiance>`. This holds deep within shadows, while
    near their edges, some of the glow comes from the medium beside them.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the scene.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.

    Returns:
        list[torch.Tensor | None]: For each sun, the share of the ground's radiance it brings, of shape (channels,),
        or None for suns below the horizon.
    """
    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    ground = ground_radiance(medium, lighting, beta, channels)
    depth = depth_above_ground(medium, lighting, beta)
    reflectance = _per_channel(lighting.ground_albedo, channels, "ground albedo", **kwargs) / math.pi
    tables = cached_surface_tables(medium, lighting, beta, channels) if medium.multiple_scattering else None
    shares: list[torch.Tensor | None] = []
    for k, sun in enumerate(lighting.suns):
        sun_z = _unit(sun.direction)[2]
        if sun_z <= 0:
            shares.append(None)
            continue
        irradiance = (
            _per_channel(sun.irradiance, channels, "sun irradiance", **kwargs) * sun_z * torch.exp(-depth / sun_z)
        )
        if tables is not None and tables.once:
            # Glow on the ground, which faces up, at the lowest height of the tables
            irradiance = irradiance + tables.once[k][0, -1, 0]
        shares.append(torch.where(ground > 0, reflectance * irradiance / ground.clamp_min(1e-30), 0.0).clamp(0, 1))
    return shares


def lamp_irradiance(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    time: float = 0.0,
    similarity: bool | torch.Tensor = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Irradiance of surfaces from a lamp, without and with a participating medium, but for the shadows of objects.

    Through the medium, the light of each of the lamp's emitters is attenuated with an extinction reduced by the
    similarity relation, ``1 - albedo · g`` times it, as the light that the medium scatters mostly keeps going forward,
    towards surfaces, unless the glow of the medium that lights surfaces is modeled on its own, see
    :func:`halo_glow <visionsim.medium.halos.halo_glow>`, in which case it is attenuated with the full extinction.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light, or emissive surface.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        points (torch.Tensor): Points on surfaces, of shape (n, 3).
        normals (torch.Tensor): Unit normals of the surfaces, of shape (n, 3), or (3,) for all of them.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        similarity (bool | torch.Tensor, optional): Whether the extinction is reduced by the similarity relation,
            rather than full, or the share of the light of each point that it is, of shape (n,), the rest of which is
            attenuated with the full extinction. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Irradiance without and with the medium, of shape (n, c).
    """
    reduced = 1 - medium.albedo * medium.anisotropy
    clear = through = points.new_zeros(len(points), len(beta))
    for emitter in emitters(lamp, len(beta), samples=4, dtype=points.dtype, device=points.device):
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
        lit = factor[:, None] * emitter.intensity
        depth = optical_depth(medium, points, u, distance, time=time)[:, None]
        if isinstance(similarity, torch.Tensor):
            share = similarity.to(points)[:, None]
            dimmed = share * torch.exp(-reduced * depth * beta) + (1 - share) * torch.exp(-depth * beta)
        else:
            dimmed = torch.exp(-(reduced if similarity else 1.0) * depth * beta)
        clear, through = clear + lit, through + lit * dimmed
    return clear, through


def surface_attenuation(
    clear: torch.Tensor,
    through: torch.Tensor,
    ambient: float = AMBIENT,
    received: torch.Tensor | None = None,
    measured: torch.Tensor | None = None,
    frame: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Ratio by which a medium scales the light of surfaces, given their irradiance without and with it.

    Light that reaches surfaces indirectly, after bouncing off other surfaces, makes up an ``ambient`` share of the mean
    irradiance of all surfaces, and is dimmed as their irradiance is on average, so that surfaces that receive little
    direct light are dimmed as the frame is on average. Given the irradiance that surfaces received without the medium,
    as their radiance and albedo imply, the rest of it beyond their direct irradiance reaches them indirectly too, where
    it is more: close to brightly lit surfaces, light that bounced off them can be most of the light of others, such as
    in the shadow of a lamp next to the ground it lights, where the glow of the medium, which ``through`` includes, would
    otherwise be divided by the ambient share alone, rather than by all the light that their radiance reflects. Where
    the irradiance received is ``measured``, i.e. given for each surface, or found from its own albedo, rather than
    from the mean albedo of the scene's surfaces, the rest of it is all the light that reaches the surface indirectly,
    which can be less than the ambient share, e.g. on the ground in the shadow of a lamp, far from the surfaces it
    lights: the ambient share is a guess of this light, which the measured irradiance replaces. Each surface then
    reflects ``albedo / π`` times the glow of the medium, with its own albedo, on top of the rest of its light, dimmed
    as the frame's is, which is bounded however little light it received, while the ratio isn't. Where the irradiance
    measured is zero, as for surfaces that reflect no light, the ambient share is kept.

    Args:
        clear (torch.Tensor): Irradiance of surfaces without the medium, of shape (n, c).
        through (torch.Tensor): Irradiance of surfaces with the medium, of shape (n, c).
        ambient (float, optional): Share of the mean irradiance that reaches surfaces indirectly. Defaults to
            :data:`AMBIENT`.
        received (torch.Tensor | None, optional): Irradiance that surfaces received without the medium, from every
            light, of shape (n, c), such as π times the radiance of diffuse surfaces over their albedo, or zero where
            it isn't known. Defaults to None, i.e. it isn't known anywhere.
        measured (torch.Tensor | None, optional): Whether ``received`` was measured for each surface, of shape (n, c),
            boolean, in which case the light that reaches it indirectly is the rest of it, rather than at least the
            ambient share, where it isn't zero. Defaults to None, i.e. nowhere.
        frame (tuple[torch.Tensor, torch.Tensor] | None, optional): Mean irradiance of the surfaces of the frame without
            and with the medium, of shape (c,) each, which set the ambient share and how the light that reaches surfaces
            indirectly is dimmed, when the surfaces given are only some of them, or the irradiances given are
            interpolated from those of other surfaces. Defaults to None, i.e. the mean irradiance of the surfaces given.

    Returns:
        torch.Tensor: Ratio of the light of surfaces with and without the medium, of shape (n, c).
    """
    if not len(clear):
        return torch.ones_like(clear)
    level = clear.mean(dim=0) if frame is None else frame[0].to(clear)
    indirect = (ambient * level).expand_as(clear)
    if received is not None:
        received = received.to(clear)
        rest = received - clear
        indirect = torch.maximum(indirect, rest)
        if measured is not None:
            # A surface that received no light at all tells nothing of the ratio, which is then that of the guess
            exact = measured.to(clear.device) & (received > 0)
            indirect = torch.where(exact, rest.clamp_min(0), indirect)
    if frame is None:
        mean = through.sum(dim=0) / clear.sum(dim=0).clamp_min(1e-30)
    else:
        mean = frame[1].to(clear) / frame[0].to(clear).clamp_min(1e-30)
    return (through + mean * indirect) / (clear + indirect).clamp_min(1e-30)
