"""Shadows that objects cast onto a participating medium: light shafts, and the sky hidden by nearby objects.

The closed form of :func:`apply_medium <visionsim.medium.render.apply_medium>` assumes that light from suns and the
sky reaches every point of the medium, only attenuated by the medium itself. Objects however block sunlight from the
medium behind them, which casts light shafts, and hide part of the sky from the medium around them, such as the fog in
front of a wall. Both are modeled with shadow maps, i.e. orthographic depth maps of the scene rendered along the
direction of each sun, and along the central direction of each cell of a partition of the sky into bands of elevation
split into equal ranges of azimuth, see :meth:`BlenderService.save_occlusion
<visionsim.simulate.blender.BlenderService.exposed_save_occlusion>`. A point sees the light along a map's direction
when it lies in front of the first surface the map recorded at its position.

Visibility is sampled along camera rays where they cross the region in which objects can cast shadows, and the
closed-form integral of each source between consecutive samples is weighted by the visibility at their midpoint, which
keeps the result deterministic and exact wherever nothing is occluded. Sunlight casts sharp shadows, and is sampled
every few texels of its map. Skylight is sampled more coarsely, with samples spread evenly over the light reaching the
camera, as each point is lit by many cells of the sky, which smooths out its variations. Each cell contributes in
proportion to the phase function integrated over the cell, times the mean attenuation of skylight over its
elevations. Light scattered more than once reaches the medium from all directions too, and is occluded in the same
way, with cells weighted by the radiance arriving from their elevations, assuming that light coming from below the
horizon is not occluded.

Lamps cast shadows through equirectangular maps of the distance between each lamp and the first surface in every
direction, see :func:`lamp_visibility`, whose visibility is sampled along rays at angles evenly spread as seen from the
lamp and averaged around each node of the quadrature that integrates the lamp's light, see :func:`lamp_shadow`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache, partial
from typing import NamedTuple

import numpy as np
import torch

from visionsim.medium.model import HeightFog, Medium
from visionsim.medium.optics import height_fog_optical_depth, height_fog_sun_inscatter, henyey_greenstein
from visionsim.medium.scattering import ElevationTable, ScatteringTable, lookup

_TABLE_SIZE = (129, 129)
"""Number of elevations and azimuths of view directions at which the phase function is integrated over each cell"""
_SHADOW_TENSORS = (
    "bounds",
    "middles",
    "visible",
    "visible_below",
    "visible_weights",
    "weights",
    "visible_weights_below",
    "weights_below",
)
"""Fields of :class:`Shadows` which hold one value per ray"""


class ShadowMaps(NamedTuple):
    """Orthographic depth maps of a scene, rendered along the directions from which light reaches it.

    Each map is rendered by a camera looking along the opposite of its third axis, which points towards the light, and
    records for each texel the distance from the camera's plane to the first surface, or a huge value if there is
    none. Maps are stored one after another, each row-major with rows from the bottom to the top of the map.
    """

    depths: torch.Tensor
    """depth of every texel of every map, of shape (texels,)"""
    offsets: torch.Tensor
    """index of the first texel of each map, of shape (k,)"""
    shapes: torch.Tensor
    """height and width of each map in texels, of shape (k, 2)"""
    origins: torch.Tensor
    """position of the camera of each map, of shape (k, 3)"""
    axes: torch.Tensor
    """axes of the camera of each map, of shape (k, 3, 3), whose rows point towards the right of the map, its top,
    and the light"""
    texels: torch.Tensor
    """size of the texels of each map, in meters, of shape (k,)"""


class Occlusion(NamedTuple):
    """Shadow maps along the directions of suns and of the cells of the sky, see :func:`load_occlusion`."""

    sun_maps: ShadowMaps
    """one map per sun"""
    sun_directions: torch.Tensor
    """unit direction towards each sun, of shape (s, 3)"""
    sky_maps: ShadowMaps
    """one map per cell of the sky, ordered by band then azimuth"""
    sky_edges: tuple[float, ...]
    """sine of the elevation at the edges of the sky's bands, from the horizon up"""
    sky_counts: tuple[int, ...]
    """number of cells of each band, which split its azimuths evenly starting from the x axis"""
    bounds: torch.Tensor
    """corners of the box that contains every object casting shadows, of shape (2, 3)"""
    lamp_maps: ShadowMaps | None = None
    """one equirectangular map per lamp, which records the distance between the lamp, at the map's origin, and the first
    surface in every direction, see :func:`lamp_visibility`, and whose texels are in radians"""


def select_maps(maps: ShadowMaps, index: int) -> ShadowMaps:
    """Shadow maps reduced to a single one of them, which still refers to all texels."""
    return maps._replace(**{name: getattr(maps, name)[index : index + 1] for name in maps._fields[1:]})


def maps_to(
    maps: ShadowMaps, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> ShadowMaps:
    """Move shadow maps to a device, and cast their floating point values to a given precision."""
    return ShadowMaps(*(m.to(device=device, dtype=dtype if m.is_floating_point() else m.dtype) for m in maps))


def occlusion_to(
    occlusion: Occlusion, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Occlusion:
    """Move shadow maps to a device, and cast their floating point values to a given precision."""
    move = partial(maps_to, device=device, dtype=dtype)
    return occlusion._replace(
        sun_maps=move(occlusion.sun_maps),
        sun_directions=occlusion.sun_directions.to(device=device, dtype=dtype),
        sky_maps=move(occlusion.sky_maps),
        bounds=occlusion.bounds.to(device=device, dtype=dtype),
        lamp_maps=move(occlusion.lamp_maps) if occlusion.lamp_maps is not None else None,
    )


def shadows_to(
    shadows: Shadows, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Shadows:
    """Move shadows to a device, and cast their floating point values to a given precision."""

    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)

    return shadows._replace(
        suns=tuple(SunShadows(*(move(t) for t in sun)) for sun in shadows.suns),
        **{name: move(getattr(shadows, name)) for name in _SHADOW_TENSORS},
    )


def sky_cell_directions(edges: Sequence[float], counts: Sequence[int]) -> np.ndarray:
    """Central direction of each cell of the sky, whose elevation splits its band's solid angle in two.

    Args:
        edges (Sequence[float]): Sine of the elevation at the edges of the bands, from the horizon up.
        counts (Sequence[int]): Number of cells of each band.

    Returns:
        np.ndarray: Unit directions, ordered by band then azimuth, of shape (k, 3).
    """
    directions = []
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        mu = (low + high) / 2
        for j in range(count):
            phi = 2 * math.pi * (j + 0.5) / count
            directions.append((math.sqrt(1 - mu * mu) * math.cos(phi), math.sqrt(1 - mu * mu) * math.sin(phi), mu))
    return np.asarray(directions, dtype=float).reshape(-1, 3)


def load_occlusion(
    path: str | os.PathLike, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Occlusion:
    """Load shadow maps saved by :meth:`BlenderService.save_occlusion
    <visionsim.simulate.blender.BlenderService.exposed_save_occlusion>`.

    Args:
        path (str | os.PathLike): Path of the ``.npz`` file.
        device (torch.device | str | None, optional): Device on which to load the maps. Defaults to None (CPU).
        dtype (torch.dtype, optional): Floating point precision. Defaults to torch.float32.

    Raises:
        ValueError: raised if the cells of the sky do not match their maps.

    Returns:
        Occlusion: Shadow maps.
    """
    kwargs = {"dtype": dtype, "device": device}
    with np.load(path) as data:

        def maps(prefix: str) -> ShadowMaps:
            return ShadowMaps(
                depths=torch.as_tensor(data[f"{prefix}_depths"], **kwargs),
                offsets=torch.as_tensor(data[f"{prefix}_offsets"], dtype=torch.long, device=device),
                shapes=torch.as_tensor(data[f"{prefix}_shapes"], dtype=torch.long, device=device).reshape(-1, 2),
                origins=torch.as_tensor(data[f"{prefix}_origins"], **kwargs).reshape(-1, 3),
                axes=torch.as_tensor(data[f"{prefix}_axes"], **kwargs).reshape(-1, 3, 3),
                texels=torch.as_tensor(data[f"{prefix}_texels"], **kwargs).reshape(-1),
            )

        edges = tuple(float(e) for e in data["sky_edges"])
        counts = tuple(int(c) for c in data["sky_counts"])
        occlusion = Occlusion(
            sun_maps=maps("sun"),
            sun_directions=torch.as_tensor(data["sun_directions"], **kwargs).reshape(-1, 3),
            sky_maps=maps("sky"),
            sky_edges=edges,
            sky_counts=counts,
            bounds=torch.as_tensor(data["bounds"], **kwargs).reshape(2, 3),
            # Maps saved before lamps cast shadows don't have any
            lamp_maps=maps("lamp") if "lamp_depths" in data else None,
        )

    expected = sky_cell_directions(edges, counts)
    towards = occlusion.sky_maps.axes[:, 2].double().cpu().numpy()
    if towards.shape != expected.shape or not np.allclose(towards, expected, atol=1e-4):
        raise ValueError("The shadow maps of the sky do not match the cells of the sky.")
    return occlusion


def _texel_coordinates(
    maps: ShadowMaps, origin: torch.Tensor, directions: torch.Tensor, distances: torch.Tensor, bias: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinates of points along rays in each map, in texels from the center of its first texel, and their depth
    minus the bias, each of shape (n, m, k)."""
    # Coordinates along each map's axes are affine along rays, and are converted to texels
    scale = torch.stack([1 / maps.texels, 1 / maps.texels, -torch.ones_like(maps.texels)], dim=-1)  # (k, 3)
    half = maps.shapes.to(maps.texels) / 2 - 0.5
    shift = torch.stack([half[:, 1], half[:, 0], -bias * maps.texels], dim=-1)
    start = ((origin - maps.origins)[:, None, :] * maps.axes).sum(dim=-1) * scale + shift  # (k, 3)
    step = torch.einsum("nd,kad->nka", directions, maps.axes) * scale  # (n, k, 3)
    x, y, depth = (torch.addcmul(start[:, a], distances[..., None], step[:, None, :, a]) for a in range(3))
    return x, y, depth


def visibility(
    maps: ShadowMaps,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distances: torch.Tensor,
    bias: float = 0.5,
    filtered: bool = True,
) -> torch.Tensor:
    """Fraction of light that reaches points along rays from the direction of each shadow map.

    Points outside of a map are lit. When filtered, each point is compared with the four texels around it, and the
    results are interpolated bilinearly (percentage-closer filtering), which smooths the edges of shadows.

    Args:
        maps (ShadowMaps): Shadow maps.
        origin (torch.Tensor): Common origin of the rays, of shape (3,).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        distances (torch.Tensor): Distance of points along each ray, of shape (n, m).
        bias (float, optional): Distance, in texels, by which points can lie behind the recorded surface and still be
            lit, which avoids shadowing points next to lit surfaces. Defaults to 0.5.
        filtered (bool, optional): If false, only compare points with the nearest texel. Defaults to True.

    Returns:
        torch.Tensor: Visibility in [0, 1], of shape (n, m, k).
    """
    x, y, depth = _texel_coordinates(maps, origin, directions, distances, bias)
    # Texels are indexed with 32 bits, which halves the memory of indices, once coordinates are clamped around maps
    width, height, offsets = maps.shapes[:, 1].int(), maps.shapes[:, 0].int(), maps.offsets.int()
    right, top = width.to(x.dtype), height.to(x.dtype)

    def lit_at(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        index = torch.where(inside, offsets + yi * width + xi, offsets)
        return ~inside | (depth <= maps.depths[index])

    def texel(coordinate: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        # Far enough from the map that the next texel, which filtering also reads, is still outside of it
        return torch.minimum(coordinate.clamp_min(-2), size).int()

    if not filtered:
        return lit_at(texel(x.round(), right), texel(y.round(), top)).to(x.dtype)

    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    x0, y0 = texel(x0, right), texel(y0, top)
    bottom = torch.lerp(lit_at(x0, y0).to(x.dtype), lit_at(x0 + 1, y0).to(x.dtype), fx)
    upper = torch.lerp(lit_at(x0, y0 + 1).to(x.dtype), lit_at(x0 + 1, y0 + 1).to(x.dtype), fx)
    return torch.lerp(bottom, upper, fy)


def lamp_visibility(maps: ShadowMaps, index: int, points: torch.Tensor, bias: float = 1.0) -> torch.Tensor:
    """Fraction of the light of a lamp that reaches points, according to its equirectangular map.

    Maps of lamps are rendered by a camera at the lamp that looks along +x with +z up, such that a direction
    ``(x, y, z)`` is seen at ``u = (π - atan2(y, x)) / 2π`` along the width of the map and ``v = 1 - acos(z) / π`` along
    its height, from the bottom, as for environment textures, and record the distance to the first surface. Each point
    is compared with the four texels around its direction, and the results are interpolated bilinearly
    (percentage-closer filtering), wrapping around the map horizontally.

    Args:
        maps (ShadowMaps): Maps of lamps.
        index (int): Index of the lamp's map.
        points (torch.Tensor): Points, of shape (..., 3).
        bias (float, optional): Distance, in texels at the distance of each point, by which points can lie behind the
            recorded surface and still be lit. Defaults to 1.0.

    Returns:
        torch.Tensor: Visibility in [0, 1], of shape (...).
    """
    offset = points - maps.origins[index]
    distance = offset.norm(dim=-1)
    height, width = (int(size) for size in maps.shapes[index])
    first = int(maps.offsets[index])
    u = (math.pi - torch.atan2(offset[..., 1], offset[..., 0])) / (2 * math.pi)
    v = 1 - torch.acos((offset[..., 2] / distance.clamp_min(1e-12)).clamp(-1, 1)) / math.pi
    x, y = u * width - 0.5, v * height - 0.5
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    columns = [torch.remainder(x0, width).long(), torch.remainder(x0 + 1, width).long()]
    rows = [y0.clamp(0, height - 1).long(), (y0 + 1).clamp(0, height - 1).long()]
    behind = distance * (1 - bias * float(maps.texels[index]))

    def lit(row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        return (behind <= maps.depths[first + row * width + column]).to(points.dtype)

    bottom = torch.lerp(lit(rows[0], columns[0]), lit(rows[0], columns[1]), fx)
    upper = torch.lerp(lit(rows[1], columns[0]), lit(rows[1], columns[1]), fx)
    return torch.lerp(bottom, upper, fy)


def lamp_map_index(maps: ShadowMaps | None, position: torch.Tensor, tolerance: float = 0.01) -> int | None:
    """Index of the map of the lamp at a position, or None if there is none within ``tolerance`` meters."""
    if maps is None or not len(maps.origins):
        return None
    distances = (maps.origins - position.to(maps.origins)).norm(dim=-1)
    index = int(distances.argmin())
    return index if float(distances[index]) <= tolerance else None


@dataclass(frozen=True)
class LampShadow:
    """Visibility of a lamp sampled along rays, which can be averaged over any interval of the rays.

    Visibility is sampled at angles evenly spread between the ends of each ray, as seen from the lamp, which matches
    the angular resolution of its map whatever the distance, see :func:`lamp_shadow`. It is constant around each
    sample, so its integral along the ray is piecewise linear, and its average over an interval is exact for this
    piecewise-constant visibility. This weights the quadrature of :func:`point_light_inscatter
    <visionsim.medium.optics.point_light_inscatter>` by the visibility around each of its nodes, rather than at the
    nodes, so that the edges of shadows don't spoil the quadrature.
    """

    along: torch.Tensor
    """distance along each ray to its point of closest approach to the lamp, of shape (r,)"""
    closest: torch.Tensor
    """distance between the lamp and each ray, of shape (r,)"""
    start: torch.Tensor
    """angle at which the lamp sees the start of each ray, from its point of closest approach, of shape (r,)"""
    span: torch.Tensor
    """angle between the ends of each ray as seen from the lamp, of shape (r,)"""
    cumulative: torch.Tensor
    """sum of the visibility of the samples before each sample along each ray, and of all of them, of shape (r, m + 1)"""

    @property
    def lit(self) -> torch.Tensor:
        """Whether some part of each ray is lit by the lamp, of shape (r,)."""
        return self.cumulative[:, -1] > 0

    def __call__(self, bounds: torch.Tensor) -> torch.Tensor:
        """Visibility averaged over intervals of the rays.

        Args:
            bounds (torch.Tensor): Increasing distances along each ray at the bounds of consecutive intervals, of shape
                (r, k + 1).

        Returns:
            torch.Tensor: Visibility averaged over each interval, of shape (r, k).
        """
        samples = self.cumulative.shape[-1] - 1
        angles = torch.atan2(bounds.to(self.along) - self.along[:, None], self.closest[:, None])
        x = ((angles - self.start[:, None]) / self.span.clamp_min(1e-12)[:, None] * samples).clamp(0, samples)
        i = x.floor().long().clamp(max=samples - 1)
        low, high = self.cumulative.gather(-1, i), self.cumulative.gather(-1, i + 1)
        width = x.diff(dim=-1)
        # Intervals too short to average over take the visibility where they are
        mean = (low + (x - i) * (high - low)).diff(dim=-1) / width.clamp_min(1e-9)
        return torch.where(width > 1e-4, mean, (high - low)[:, :-1]).clamp(0, 1).to(bounds.dtype)

    def select(self, rays: torch.Tensor) -> LampShadow:
        """Visibility along a subset of the rays, given their indices."""
        return LampShadow(*(getattr(self, field.name)[rays] for field in fields(self)))


def lamp_shadow(
    maps: ShadowMaps,
    index: int,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    samples: int = 256,
    bias: float = 1.0,
) -> LampShadow:
    """Sample the visibility of a lamp along rays, at angles evenly spread between their ends as seen from the lamp.

    Args:
        maps (ShadowMaps): Maps of lamps.
        index (int): Index of the lamp's map, whose origin is the lamp's position.
        origin (torch.Tensor): Ray origin, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (r, 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (r,), can be infinite.
        samples (int, optional): Number of samples of the visibility along each ray. Defaults to 256.
        bias (float, optional): Distance, in texels, by which points can lie behind the surfaces recorded by the map
            and still be lit. Defaults to 1.0.

    Returns:
        LampShadow: Visibility along the rays, which can be averaged over any of their intervals.
    """
    # Visibility is computed with the precision of the maps, typically single precision which is enough for it
    dtype = maps.depths.dtype
    origin, directions, distance = origin.to(dtype), directions.to(dtype), distance.to(dtype)
    offset = maps.origins[index] - origin
    along = directions @ offset
    closest = ((offset * offset).sum() - along * along).clamp_min(0).sqrt().clamp_min(1e-4)
    # Angles stay within ±π/2, which single precision would otherwise round beyond for infinite rays
    limit = math.pi / 2 - 1e-6
    start = torch.atan2(-along, closest).clamp(-limit, limit)
    end = torch.atan2(distance - along, closest).clamp(-limit, limit)
    fractions = (torch.arange(samples, dtype=dtype, device=directions.device) + 0.5) / samples
    theta = start[:, None] + (end - start)[:, None] * fractions
    points = origin + (along[:, None] + closest[:, None] * torch.tan(theta))[..., None] * directions[:, None, :]
    visible = lamp_visibility(maps, index, points, bias)
    cumulative = torch.cat([torch.zeros_like(visible[:, :1]), visible.cumsum(dim=-1)], dim=-1)
    return LampShadow(along, closest, start, end - start, cumulative)


def _blocker_distances(maps: ShadowMaps, points: torch.Tensor, bias: float) -> torch.Tensor:
    """Distance from points towards the light of each shadow map to the surface recorded by its nearest texel, where
    that surface hides the light, or infinity otherwise, of shape (u, k)."""
    ones = torch.ones_like(points[:, :1])
    x, y, depth = (c[:, 0] for c in _texel_coordinates(maps, torch.zeros_like(points[0]), points, ones, bias))
    width, height, offsets = maps.shapes[:, 1].int(), maps.shapes[:, 0].int(), maps.offsets.int()
    xi = torch.minimum(x.round().clamp_min(-2), width.to(x.dtype)).int()
    yi = torch.minimum(y.round().clamp_min(-2), height.to(x.dtype)).int()
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    recorded = maps.depths[torch.where(inside, offsets + yi * width + xi, offsets)]
    return torch.where(inside & (depth > recorded), depth - recorded, torch.full_like(depth, torch.inf))


@lru_cache(maxsize=8)
def _cell_phase_table(
    edges: tuple[float, ...], counts: tuple[int, ...], g: float, dtype: torch.dtype, device: str, nodes: int = 16
) -> torch.Tensor:
    """Phase function integrated over a cell of each band centered at azimuth zero, for view directions on a grid of
    vertical components in [-1, 1] and of azimuths in [0, π] relative to the cell, of shape (b, *_TABLE_SIZE)."""
    kwargs = {"dtype": torch.float64, "device": device}
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
    view_mu = torch.linspace(-1, 1, _TABLE_SIZE[0], **kwargs)[:, None, None, None]
    view_phi = torch.linspace(0, math.pi, _TABLE_SIZE[1], **kwargs)[None, :, None, None]
    view_s = (1 - view_mu**2).clamp_min(0).sqrt()
    tables = []
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        mu = (low + high) / 2 + (high - low) / 2 * t[:, None]
        phi = math.pi / count * t[None, :]
        weights = (high - low) / 2 * w[:, None] * math.pi / count * w[None, :]
        cos = view_mu * mu + view_s * (1 - mu**2).clamp_min(0).sqrt() * torch.cos(phi - view_phi)
        tables.append((weights * henyey_greenstein(cos, g)).sum(dim=(-2, -1)))
    return torch.stack(tables).to(dtype)


def cell_phases(
    edges: Sequence[float], counts: Sequence[int], g: float, directions: torch.Tensor, below: bool = False
) -> torch.Tensor:
    """Henyey-Greenstein phase function integrated over each cell of the sky, for rays of given directions.

    Args:
        edges (Sequence[float]): Sine of the elevation at the edges of the bands, from the horizon up.
        counts (Sequence[int]): Number of cells of each band.
        g (float): Asymmetry parameter of the phase function.
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        below (bool, optional): If true, integrate over the cells mirrored below the horizon instead.
            Defaults to False.

    Returns:
        torch.Tensor: Integral of the phase function over each cell, of shape (n, k).
    """
    table = _cell_phase_table(tuple(edges), tuple(counts), g, directions.dtype, str(directions.device))
    band = torch.repeat_interleave(
        torch.arange(len(counts), device=directions.device), torch.as_tensor(counts, device=directions.device)
    )
    azimuth = torch.cat([2 * math.pi * (torch.arange(c) + 0.5) / c for c in counts]).to(directions)
    view_mu = -directions[:, 2:3] if below else directions[:, 2:3]
    relative = torch.atan2(directions[:, 1:2], directions[:, 0:1]) - azimuth
    relative = torch.remainder(relative + math.pi, 2 * math.pi) - math.pi

    fi = ((view_mu.clamp(-1, 1) + 1) / 2 * (_TABLE_SIZE[0] - 1)).expand_as(relative)
    fj = relative.abs() / math.pi * (_TABLE_SIZE[1] - 1)
    i = fi.floor().clamp(0, _TABLE_SIZE[0] - 2).long()
    j = fj.floor().clamp(0, _TABLE_SIZE[1] - 2).long()
    wi, wj = fi - i, fj - j
    near = table[band, i, j] + wj * (table[band, i, j + 1] - table[band, i, j])
    far = table[band, i + 1, j] + wj * (table[band, i + 1, j + 1] - table[band, i + 1, j])
    return near + wi * (far - near)


@lru_cache(maxsize=8)
def _band_attenuation_table(
    edges: tuple[float, ...], dtype: torch.dtype, device: str, size: int = 256, nodes: int = 64
) -> torch.Tensor:
    """Logarithm of the mean of ``exp(-c / μ)`` over the elevations of each band, for ``log10(c)`` spread over
    [-6, 2], of shape (size, b), which is smoother than the mean itself."""
    kwargs = {"dtype": torch.float64, "device": device}
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
    depth = torch.logspace(-6, 2, size, **kwargs)[:, None, None]
    low, high = torch.as_tensor(edges[:-1], **kwargs), torch.as_tensor(edges[1:], **kwargs)
    mu = (low + high)[:, None] / 2 + (high - low)[:, None] / 2 * t  # (b, nodes)
    return torch.logsumexp(torch.log(w / 2) - depth / mu, dim=-1).to(dtype)


def band_attenuation(edges: Sequence[float], depth_above: torch.Tensor) -> torch.Tensor:
    """Mean attenuation ``exp(-c / μ)`` of skylight over the elevations ``μ`` of each band of the sky.

    Args:
        edges (Sequence[float]): Sine of the elevation at the edges of the bands, from the horizon up.
        depth_above (torch.Tensor): Optical depth ``c`` of the fog above points, of any shape (...).

    Returns:
        torch.Tensor: Mean attenuation of each band, of shape (..., b).
    """
    table = _band_attenuation_table(tuple(edges), depth_above.dtype, str(depth_above.device))
    f = (torch.log10(depth_above.clamp_min(1e-30)) + 6) / 8 * (len(table) - 1)
    f = f.clamp(0, len(table) - 1)
    i = f.floor().clamp(0, len(table) - 2).long()
    return torch.exp(torch.lerp(table[i], table[i + 1], (f - i)[..., None]))


def band_radiance(table: ScatteringTable, edges: Sequence[float], nodes: int = 8) -> torch.Tensor:
    """Mean radiance scattered once that arrives from the elevations of each band, above and below the horizon.

    Args:
        table (ScatteringTable): Light scattered more than once, including the light it was gathered from.
        edges (Sequence[float]): Sine of the elevation at the edges of the bands, from the horizon up.
        nodes (int, optional): Number of elevations at which each band is averaged. Defaults to 8.

    Returns:
        torch.Tensor: Mean radiance arriving from each band, of shape (heights, 2, b, c), where the second dimension
        is above, then below the horizon.
    """
    kwargs = {"dtype": table.gathered.dtype, "device": table.gathered.device}
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
    low, high = torch.as_tensor(edges[:-1], **kwargs), torch.as_tensor(edges[1:], **kwargs)
    mu = (low + high)[:, None] / 2 + (high - low)[:, None] / 2 * t  # (b, nodes)
    result = []
    for sign in (1.0, -1.0):
        # Linear interpolation over the gathering directions, whose vertical components are sorted
        x = (sign * mu).reshape(-1).contiguous()
        j = torch.searchsorted(table.directions, x).clamp(1, len(table.directions) - 1)
        x0, x1 = table.directions[j - 1], table.directions[j]
        f = ((x - x0) / (x1 - x0)).clamp(0, 1)[None, :, None]
        values = table.gathered[:, j - 1] + f * (table.gathered[:, j] - table.gathered[:, j - 1])  # (h, b*nodes, c)
        values = values.reshape(len(table.gathered), *mu.shape, -1)
        result.append((values * w[:, None] / 2).sum(dim=-2))
    return torch.stack(result, dim=1)


def _ray_box(origin: torch.Tensor, directions: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    """Distances at which rays enter and leave an axis-aligned box, which are equal when they miss it."""
    parallel = directions == 0
    safe = torch.where(parallel, torch.ones_like(directions), directions)
    t1, t2 = (low - origin) / safe, (high - origin) / safe
    inside = (origin >= low) & (origin <= high)
    huge = torch.full_like(t1, torch.inf)
    near = torch.where(parallel, torch.where(inside, -huge, huge), torch.minimum(t1, t2)).amax(dim=-1)
    far = torch.where(parallel, torch.where(inside, huge, -huge), torch.maximum(t1, t2)).amin(dim=-1)
    near = near.clamp_min(0)
    return near, torch.maximum(near, far)


def _distance_at(fog: HeightFog, origin_z: float, directions_z: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Distance along rays through an exponential height fog at which a relative optical depth is reached, the
    inverse of :func:`height_fog_optical_depth <visionsim.medium.optics.height_fog_optical_depth>`."""
    k = fog.density * math.exp(-(origin_z - fog.base_height) / fog.falloff)
    x = (depth * directions_z / (k * fog.falloff)).clamp_max(1 - 1e-7)
    small = x.abs() < 1e-6
    ratio = torch.where(small, 1 + x / 2 + x * x / 3, -torch.log1p(-x) / torch.where(small, torch.ones_like(x), x))
    return depth / k * ratio


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30), torch.ones_like(numerator))


def _chunks(count: int, size: int) -> Iterator[slice]:
    """Slices of at most ``size`` consecutive rays."""
    return (slice(start, start + size) for start in range(0, count, max(1, size)))


def _ragged_chunks(counts: torch.Tensor, size: int) -> Iterator[tuple[slice, int]]:
    """Slices of consecutive rays with at most ``size`` samples in all, given the number of samples of each ray, or
    single rays with more samples, along with their number of samples."""
    total = torch.cumsum(counts, dim=0).cpu()
    start = 0
    while start < len(total):
        before = int(total[start - 1]) if start else 0
        stop = max(int(torch.searchsorted(total, before + max(1, size), right=True)), start + 1)
        yield slice(start, stop), int(total[stop - 1]) - before
        start = stop


def _ragged(counts: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the ray of each sample, and of each sample along its ray, of shape (s,), when rays have given numbers
    of samples, laid out one ray after another, given their total."""
    ray = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts, output_size=total)
    first = torch.cumsum(counts, dim=0) - counts
    return ray, torch.arange(total, device=counts.device) - first[ray]


def _pack(bits: torch.Tensor) -> torch.Tensor:
    """Pack booleans along the last dimension into bytes, of shape (..., ceil(k / 8))."""
    padded = torch.nn.functional.pad(bits.to(torch.uint8), (0, -bits.shape[-1] % 8))
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device)
    return (padded.reshape(*bits.shape[:-1], -1, 8) * weights).sum(dim=-1).to(torch.uint8)


def _unpack(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Booleans packed by :func:`_pack`, as zeros and ones of shape (..., count)."""
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    return ((packed[..., None] >> shifts) & 1).reshape(*packed.shape[:-1], -1)[..., :count]


def _hidden_segments(
    hidden: torch.Tensor,
    ray: torch.Tensor,
    first: torch.Tensor,
    last: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Segments of rays along which light is hidden, from its hidden fraction along consecutive intervals of rays,
    laid out one ray after another, given the ray of each interval, whether it is the first or last of its ray, and
    where it starts and ends, all of shape (s,). Runs of intervals that are entirely hidden are merged."""
    full = hidden >= 1
    partial = (hidden > 0) & ~full
    edge = torch.zeros_like(full[:1])
    begin = (full & (first | ~torch.cat([edge, full[:-1]]))).nonzero()[:, 0]
    end = (full & (last | ~torch.cat([full[1:], edge]))).nonzero()[:, 0]
    some = partial.nonzero()[:, 0]
    return (
        torch.cat([ray[begin], ray[some]]),
        torch.cat([starts[begin], starts[some]]),
        torch.cat([ends[end], ends[some]]),
        torch.cat([torch.ones_like(starts[begin]), hidden[some]]),
    )


def _snap(points: torch.Tensor, low: torch.Tensor, high: torch.Tensor, size: float) -> torch.Tensor:
    """Index of the cell of a grid of a given size over a box that contains each point, of shape (...), where points
    outside of the box are moved to its closest cell."""
    dims = torch.ceil((high - low) / size).long().clamp_min(1)
    cell = torch.floor((points - low) / size).nan_to_num(0).clamp(-1, 2**40).long()
    cell = torch.minimum(cell.clamp_min(0), dims - 1)
    return cell[..., 0] + dims[0] * (cell[..., 1] + dims[1] * cell[..., 2])


def _depth_windows(maps: ShadowMaps, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Least and greatest depth recorded by the first of shadow maps over the square window of texels of a given size
    whose first texel is each texel, from ``size`` texels before the map to its end, where texels outside of the map
    let all light through, each of shape (h + size, w + size)."""
    height, width = (int(s) for s in maps.shapes[0])
    offset = int(maps.offsets[0])
    image = maps.depths[offset : offset + height * width].reshape(1, 1, height, width)
    image = torch.nn.functional.pad(image, (size, size - 1, size, size - 1), value=torch.inf)

    def pool(values: torch.Tensor) -> torch.Tensor:
        # Along rows, then columns
        values = torch.nn.functional.max_pool2d(values, (1, size), stride=1)
        return torch.nn.functional.max_pool2d(values, (size, 1), stride=1)[0, 0]

    return -pool(-image), pool(image)


def _surely_lit(
    maps: ShadowMaps,
    windows: tuple[torch.Tensor, torch.Tensor],
    size: int,
    origin: torch.Tensor,
    directions: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    bias: float,
) -> torch.Tensor:
    """Whether intervals along rays are entirely lit (1) or hidden (0) by the first of shadow maps, given the depths
    over windows of its texels from :func:`_depth_windows`, or may be partly hidden (-1), of shape (s,)."""
    distances = torch.stack([starts, ends], dim=-1)
    x, y, depth = (c[..., 0] for c in _texel_coordinates(maps, origin, directions, distances, bias))  # (s, 2)
    least, most = windows
    # The window from the first texel that filtered lookups read along each axis must also hold the last one
    x0, y0 = (c.amin(dim=-1).nan_to_num(0).clamp(-size, 2.0**30).floor() for c in (x, y))
    narrow = (x.amax(dim=-1).floor() + 1 - x0 < size) & (y.amax(dim=-1).floor() + 1 - y0 < size)
    # Windows beyond the map are moved back to its last texels, which only makes them less likely to be sure
    rows, columns = least.shape
    index = (y0.clamp(max=rows - size - 1).long() + size) * columns + x0.clamp(max=columns - size - 1).long() + size
    lit = narrow & (depth.amax(dim=-1) <= least.reshape(-1)[index])
    hidden = narrow & (depth.amin(dim=-1) > most.reshape(-1)[index])
    return torch.where(lit, 1, torch.where(hidden, 0, -1))


_DIRECTION_BIN = 0.01
"""Size of the bins of directions of rays which share the weights of the sky's cells, in units of their components"""


def _direction_bins(directions: torch.Tensor, size: float) -> torch.Tensor:
    """Bin of each unit direction on a grid of a given size over its components, of shape (n,)."""
    side = 2 * math.ceil(1 / size) + 3
    index = (torch.floor(directions / size).long() + math.ceil(1 / size) + 1).clamp(0, side - 1)
    return index[:, 0] + side * (index[:, 1] + side * index[:, 2])


def _cell_centers(index: torch.Tensor, low: torch.Tensor, high: torch.Tensor, size: float) -> torch.Tensor:
    """Center of the cells of the grid of :func:`_snap` with given indices, of shape (u, 3)."""
    dims = torch.ceil((high - low) / size).long().clamp_min(1)
    x, rest = index % dims[0], index // dims[0]
    cell = torch.stack([x, rest % dims[1], rest // dims[1]], dim=-1)
    return low + (cell.to(low.dtype) + 0.5) * size


class SunShadows(NamedTuple):
    """Segments of rays along which a sun is hidden, entirely or partly, see :func:`trace`."""

    direction: torch.Tensor
    """unit direction towards the sun, of shape (3,)"""
    rays: torch.Tensor
    """index of the ray of each segment, of shape (e,)"""
    starts: torch.Tensor
    """distance along its ray at which each segment starts, of shape (e,)"""
    ends: torch.Tensor
    """distance along its ray at which each segment ends, of shape (e,)"""
    hidden: torch.Tensor
    """fraction of the sun hidden along each segment, of shape (e,)"""


class Shadows(NamedTuple):
    """Where objects hide suns and cells of the sky along the rays of a frame, see :func:`trace`. As this doesn't
    depend on the medium, it can be reused to add several media to the same frame, see :func:`shade`.

    Cells of the sky are weighted by the phase function integrated over them, which only depends on the medium's
    anisotropy: their weights are kept for the anisotropy of the medium the shadows were traced with, and recomputed
    from the visibility of each cell for media of other anisotropies.

    Cells mirrored below the horizon, from which the ground and the medium beneath light the medium, are hidden by
    objects that hide the cell above the horizon, as objects stand on the ground, unless they stand further than where
    light from below the horizon starts, on the ground."""

    rays: int
    """number of rays"""
    suns: tuple[SunShadows, ...]
    """segments along which each sun with a shadow map is hidden"""
    bounds: torch.Tensor
    """distance along each ray of the boundaries of the samples of the sky's visibility, of shape (n, m + 1)"""
    middles: torch.Tensor
    """distance along each ray of the middle of each sample, where visibility was looked up, of shape (n, m)"""
    visible: torch.Tensor
    """whether each cell of the sky is visible from the middle of each sample, packed as one bit per cell, of shape
    (n, m, ceil(k / 8))"""
    visible_below: torch.Tensor
    """same as ``visible`` for the cells mirrored below the horizon"""
    sky_edges: tuple[float, ...]
    """sine of the elevation at the edges of the sky's bands, from the horizon up"""
    sky_counts: tuple[int, ...]
    """number of cells of each band"""
    anisotropy: float
    """anisotropy of the phase function the following weights were computed with"""
    visible_weights: torch.Tensor
    """sum over the visible cells of each band of the phase function integrated over them, from the middle of each
    sample, of shape (n, m, b), which is the same as ``weights`` for bands that are entirely visible"""
    weights: torch.Tensor
    """sum over the cells of each band of the phase function integrated over them, of shape (n, b)"""
    visible_weights_below: torch.Tensor
    """same as ``visible_weights`` for the cells mirrored below the horizon"""
    weights_below: torch.Tensor
    """same as ``weights`` for the cells mirrored below the horizon, of shape (n, b)"""


def trace(
    occlusion: Occlusion,
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    suns: Sequence[torch.Tensor],
    sun_step: float = 8.0,
    max_sun_samples: int = 1024,
    sky_samples: int = 8,
    sky_grid: float = 1.0,
    ground_height: float = 0.0,
    bias: float = 0.5,
    max_depth: float = 12.0,
    memory: float = 2**30,
) -> Shadows:
    """Trace where the scene's objects hide suns and the sky along rays through an exponential height fog.

    Rays are only sampled where objects can cast shadows, i.e. anywhere below them as light comes from above, and as
    long as light scattered there can still reach the camera. Sunlight casts sharp shadows, and is sampled every few
    texels of its map, while the visibility of the sky, which is softened by the many cells it comes from, is sampled
    more coarsely, at points spread evenly over the light that the medium scatters towards the camera. Rays close to
    each other then sample places close to each other, so that these points are snapped to a grid finer than the
    texels of the sky's maps, and the sky is only looked up once per cell of the grid. The result doesn't depend on
    the medium otherwise, so that it can be reused for media of similar density, whose light is then scattered from
    roughly the same places.

    Args:
        occlusion (Occlusion): Shadow maps of the scene, on the same device and with the same precision as the rays.
        medium (Medium): Participating medium, made of a single height fog, which sets where rays are sampled.
        origin (torch.Tensor): Common origin of the rays, of shape (3,).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        distance (torch.Tensor): Length of the rays, of shape (n,), which is infinite for rays that hit nothing.
        suns (Sequence[torch.Tensor]): Unit direction towards each sun, of shape (3,). Suns without a shadow map cast
            no shadows.
        sun_step (float, optional): Length of the intervals of rays over which sunlight's visibility is checked at
            once, in texels of its map. Those that a shadow's edge may cross are sampled every texel. Defaults to 8.
        max_sun_samples (int, optional): Maximum number of such intervals along a ray, beyond which they get longer.
            Defaults to 1024.
        sky_samples (int, optional): Number of samples of the sky's visibility along a ray. Defaults to 8.
        sky_grid (float, optional): Size of the cells of the grid to which samples of the sky's visibility are
            snapped, in texels of the finest map of the sky. Defaults to 1.
        ground_height (float, optional): Height of the ground, where light from below the horizon starts.
            Defaults to 0.
        bias (float, optional): Depth bias of shadow maps, in texels, see :func:`visibility`. Defaults to 0.5.
        max_depth (float, optional): Optical depth beyond which rays are no longer sampled, as hardly any light
            scattered further reaches the camera. Defaults to 12.
        memory (float, optional): Rough bound, in bytes, on the memory used by intermediate values, which are computed
            for as many rays at once as fit. Defaults to 1 GiB.

    Returns:
        Shadows: Where suns and cells of the sky are hidden along each ray.
    """
    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    kwargs = {"dtype": directions.dtype, "device": directions.device}
    n = len(directions)
    origin_z = float(origin[2])
    v_z = directions[:, 2:3]
    beta = medium.extinction

    # Shadows of the objects' tops reach points below them, down to the ground or the camera's height, at a
    # horizontal distance that depends on the light's elevation
    low, high = occlusion.bounds
    reach = _distance_at(fog, origin_z, v_z[:, 0], torch.full_like(v_z[:, 0], max_depth / beta))
    lowest = float((origin_z + v_z[:, 0].clamp_max(0) * torch.minimum(distance, reach)).min()) if n else origin_z
    height = float(high[2]) - min(float(low[2]), lowest)
    low = torch.cat([low[:2], low.new_full((1,), -1e9)])

    def segment(margin: float) -> tuple[torch.Tensor, torch.Tensor]:
        pad = torch.as_tensor([margin, margin, 0.0], **kwargs)
        near, far = _ray_box(origin, directions, low - pad, high + pad)
        return near, torch.maximum(near, torch.minimum(torch.minimum(far, distance), reach))

    hidden_suns = []
    texels = occlusion.sun_maps.texels.tolist()
    for towards in suns:
        sun_z = float(towards[2])
        alignment = (occlusion.sun_directions @ towards).tolist()
        if sun_z <= 0 or not alignment or max(alignment) < 0.9999:
            continue
        index = alignment.index(max(alignment))
        maps = select_maps(occlusion.sun_maps, index)
        start, end = segment(height * math.sqrt(1 - sun_z**2) / sun_z)
        # Each ray is split into intervals of a few texels, laid out one ray after another, so that the many rays
        # that only cross a short stretch where objects can cast shadows have few of them. Intervals that are surely
        # lit or hidden, as they lie above or below every surface around them, are only looked up once, and the
        # others, which a shadow's edge may cross, every texel
        length, texel = end - start, texels[index]
        counts = torch.ceil(length / (sun_step * texel)).clamp(0, max_sun_samples).long()
        window = math.ceil(sun_step) + 2
        windows = _depth_windows(maps, window)
        empty = [torch.zeros(0, dtype=torch.long, device=directions.device)] + [torch.zeros(0, **kwargs)] * 3
        segments = [empty]
        for rays, count in _ragged_chunks(counts, int(memory // 256)):
            ray, j = _ragged(counts[rays], count)
            ray = ray + (rays.start or 0)
            total = counts[ray]
            starts = torch.addcmul(start[ray], length[ray], j / total)
            ends = torch.addcmul(start[ray], length[ray], (j + 1) / total)
            sure = _surely_lit(maps, windows, window, origin, directions[ray], starts, ends, bias)
            segments.append(list(_hidden_segments((sure == 0).to(start), ray, j == 0, j == total - 1, starts, ends)))
            unsure = (sure < 0).nonzero()[:, 0]
            steps = torch.ceil((ends - starts)[unsure] / texel).clamp_min(1).long()
            for part, count in _ragged_chunks(steps, int(memory // 256)):
                which, i = _ragged(steps[part], count)
                interval, parts = unsure[part][which], steps[part][which]
                below = torch.addcmul(starts[interval], (ends - starts)[interval], i / parts)
                above = torch.addcmul(starts[interval], (ends - starts)[interval], (i + 1) / parts)
                lit = visibility(maps, origin, directions[ray[interval]], ((below + above) / 2)[:, None], bias)
                refined = _hidden_segments(1 - lit[:, 0, 0], ray[interval], i == 0, i == parts - 1, below, above)
                segments.append(list(refined))
        hidden_suns.append(SunShadows(towards, *(torch.cat(columns) for columns in zip(*segments))))

    # Skylight comes from many cells, which smooths out its variations, so it is sampled more coarsely, evenly in the
    # fraction of light scattered or absorbed along the ray, i.e. in how much light each sample can contribute.
    # Points beside the objects can have low cells of the sky hidden, even far from them
    edges, counts = occlusion.sky_edges, occlusion.sky_counts
    cells, bands = len(occlusion.sky_maps.texels), len(counts)
    if not cells:
        empty, bits = torch.zeros(n, 0, **kwargs), torch.zeros(n, 0, 0, dtype=torch.uint8, device=directions.device)
        return Shadows(
            n,
            tuple(hidden_suns),
            bounds=empty,
            middles=empty,
            visible=bits,
            visible_below=bits,
            sky_edges=edges,
            sky_counts=counts,
            anisotropy=medium.anisotropy,
            visible_weights=torch.zeros(n, 0, bands, **kwargs),
            weights=torch.zeros(n, bands, **kwargs),
            visible_weights_below=torch.zeros(n, 0, bands, **kwargs),
            weights_below=torch.zeros(n, bands, **kwargs),
        )
    low_mu = (edges[0] + edges[1]) / 2
    margin = height * math.sqrt(1 - low_mu**2) / low_mu
    start, end = segment(margin)
    first = -torch.expm1(-height_fog_optical_depth(fog, origin, directions, start)[:, None] * beta)
    last = -torch.expm1(-height_fog_optical_depth(fog, origin, directions, end)[:, None] * beta)
    y = torch.addcmul(first, last - first, torch.linspace(0, 1, 2 * sky_samples + 1, **kwargs)).clamp_max(1 - 1e-7)
    at = _distance_at(fog, origin_z, v_z, -torch.log1p(-y) / beta)
    bounds, middles = at[:, 0::2].contiguous(), at[:, 1::2].contiguous()

    # The weights of the sky's cells only depend on the direction of rays, and are shared by rays of similar directions
    bins, ray_bin = torch.unique(_direction_bins(directions, _DIRECTION_BIN), return_inverse=True)  # (d,), (n,)
    mean = torch.zeros(len(bins), 3, **kwargs).index_add_(0, ray_bin, directions)
    mean = torch.nn.functional.normalize(mean, dim=-1)
    one_hot = _band_one_hot(counts, **kwargs)
    phases = cell_phases(edges, counts, medium.anisotropy, mean)  # (d, k)
    phases_below = cell_phases(edges, counts, medium.anisotropy, mean, below=True)

    # Rays close to each other also sample places close to each other, so that points are snapped to a grid finer than
    # the maps' texels, and the sky's visibility is only looked up once per cell of the grid
    size = sky_grid * float(occlusion.sky_maps.texels.min())
    pad = torch.as_tensor([margin, margin, 0.0], **kwargs)
    box = (torch.cat([low[:2], low.new_full((1,), lowest)]) - pad, high + pad)
    grid = [_snap(origin + directions[rays, None] * middles[rays, ..., None], *box, size) for rays in _chunks(n, 2**20)]
    used, grid_cell = torch.unique(torch.cat(grid), return_inverse=True)  # (v,), (n, m)
    # Light from below the horizon, mirrored from each cell, starts on the ground at most this far horizontally
    towards = occlusion.sky_maps.axes[:, 2]  # (k, 3)
    across = towards[:, :2].norm(dim=-1)
    chunk = int(memory // (cells * 64))
    packed: dict[bool, list[torch.Tensor]] = {False: [], True: []}
    for part in _chunks(len(used), chunk):
        centers = _cell_centers(used[part], *box, size)
        blocker = _blocker_distances(occlusion.sky_maps, centers, bias)  # (u, k)
        reach = (centers[:, 2:3] - ground_height).clamp_min(0) * across / towards[:, 2]
        packed[False].append(_pack(torch.isinf(blocker)))
        packed[True].append(_pack(blocker * across >= reach))
    visible, visible_below = torch.cat(packed[False]), torch.cat(packed[True])

    # Samples in the same cell of the grid, along rays of similar directions, see the same fraction of each band
    groups, group = torch.unique(grid_cell * len(bins) + ray_bin[:, None], return_inverse=True)  # (u,), (n, m)
    fractions: dict[bool, list[torch.Tensor]] = {False: [], True: []}
    for part in _chunks(len(groups), chunk):
        cell, direction = groups[part] // len(bins), groups[part] % len(bins)
        for below, bits, weighted in ((False, visible, phases), (True, visible_below, phases_below)):
            lit = _unpack(bits[cell], cells).to(directions.dtype)
            fractions[below].append(_ratio((lit * weighted[direction]) @ one_hot, weighted[direction] @ one_hot))
    weights, weights_below = (phases @ one_hot)[ray_bin], (phases_below @ one_hot)[ray_bin]
    return Shadows(
        n,
        tuple(hidden_suns),
        bounds=bounds,
        middles=middles,
        visible=visible[grid_cell],
        visible_below=visible_below[grid_cell],
        sky_edges=edges,
        sky_counts=counts,
        anisotropy=medium.anisotropy,
        visible_weights=torch.cat(fractions[False])[group] * weights[:, None],
        weights=weights,
        visible_weights_below=torch.cat(fractions[True])[group] * weights_below[:, None],
        weights_below=weights_below,
    )


def _band_one_hot(counts: Sequence[int], dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    """Band of each cell of the sky, one-hot encoded, of shape (k, b)."""
    bands = torch.repeat_interleave(torch.arange(len(counts), device=device), torch.as_tensor(counts, device=device))
    return torch.nn.functional.one_hot(bands, len(counts)).to(dtype)


def _band_weights(
    edges: Sequence[float],
    counts: Sequence[int],
    g: float,
    directions: torch.Tensor,
    lit: torch.Tensor,
    lit_below: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase function integrated over the visible cells of each band from points along rays, given the visibility of
    each cell, of shape (r, m, k), and over all the cells of each band, above and below the horizon, as the fields of
    :class:`Shadows` of the same names."""
    one_hot = _band_one_hot(counts, directions.dtype, directions.device)  # (k, b)
    phases = cell_phases(edges, counts, g, directions)  # (r, k)
    below = cell_phases(edges, counts, g, directions, below=True)
    return (
        torch.bmm(lit, phases[:, :, None] * one_hot),
        phases @ one_hot,
        torch.bmm(lit_below, below[:, :, None] * one_hot),
        below @ one_hot,
    )


def _arriving(
    radiance: torch.Tensor, heights: torch.Tensor, weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Light scattered more than once arriving at samples along rays from the visible cells of the sky, and from all
    of them, given the radiance arriving from each band at each height, of shape (h, 2, b), the index of the height of
    each sample, of shape (r, m), and the weights of the cells of each band, see :func:`_band_weights`."""
    up, down = radiance[heights, 0], radiance[heights, 1]  # (r, m, b)
    visible, total, visible_below, total_below = weights
    seen = up * visible + down * visible_below
    return seen.sum(dim=-1), (up * total[:, None] + down * total_below[:, None]).sum(dim=-1)


def shade(
    shadows: Shadows,
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    beta: torch.Tensor,
    suns: Sequence[tuple[torch.Tensor, torch.Tensor]],
    tables: Mapping[str, ElevationTable],
    scattering: ScatteringTable | None = None,
    memory: float = 2**30,
) -> torch.Tensor:
    """Light from suns, the sky, the ground and multiple scattering, scattered towards rays through an exponential
    height fog, given where objects hide the suns and the sky along them.

    The closed-form integral of each source is weighted, along each segment where a sun is hidden, by the fraction of
    the sun that is hidden, and between consecutive samples of the sky's visibility, by the visible fraction of the
    light arriving from all cells of the sky. Each cell contributes in proportion to the phase function integrated
    over the cell, times the mean attenuation of skylight over its elevations. Light scattered more than once is
    occluded in the same way, with cells weighted by the radiance arriving from their elevations, above and below the
    horizon, and so is light reflected by the ground into the medium, which arrives from below the horizon, see
    :class:`Shadows`.

    Args:
        shadows (Shadows): Where objects hide suns and the sky along the rays, see :func:`trace`.
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        origin (torch.Tensor): Common origin of the rays, of shape (3,).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        distance (torch.Tensor): Length of the rays, of shape (n,), which is infinite for rays that hit nothing.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) when it is the same for all.
        suns (Sequence[tuple[torch.Tensor, torch.Tensor]]): Unit direction towards each sun, of shape (3,), and its
            irradiance times the medium's albedo, of shape (c',). Suns that weren't traced cast no shadows.
        tables (Mapping[str, ElevationTable]): Light from the sky, the ground and light scattered more than once,
            tabulated along rays and keyed by "sky", "ground" and "scattered", see
            :func:`tabulate_along_rays <visionsim.medium.scattering.tabulate_along_rays>`.
        scattering (ScatteringTable | None, optional): Light scattered more than once, needed to occlude the
            "scattered" table, which is otherwise not occluded. Defaults to None.
        memory (float, optional): Rough bound, in bytes, on the memory used by intermediate values, which are computed
            for as many rays at once as fit. Defaults to 1 GiB.

    Raises:
        ValueError: raised if the shadows were traced along a different number of rays.

    Returns:
        torch.Tensor: Scattered radiance, of shape (n, c').
    """
    if shadows.rays != len(directions):
        raise ValueError(f"Shadows were traced along {shadows.rays} rays, but {len(directions)} were given.")
    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    kwargs = {"dtype": directions.dtype, "device": directions.device}
    n = len(directions)
    origin_z = float(origin[2])
    origin_extinction = beta * fog.density * math.exp(-(origin_z - fog.base_height) / fog.falloff)
    v_z = directions[:, 2:3]
    full_depth = height_fog_optical_depth(fog, origin, directions, distance)[:, None] * beta  # (n, c)

    total = torch.zeros(n, 1, **kwargs)
    for towards, color in suns:
        sun_z = float(towards[2])
        weight = color * henyey_greenstein(directions @ towards, medium.anisotropy)[:, None]
        unoccluded = height_fog_sun_inscatter(full_depth, origin_extinction, fog.falloff, v_z, distance[:, None], sun_z)
        traced = [s for s in shadows.suns if float(s.direction.to(towards) @ towards) >= 0.9999]
        if sun_z <= 0 or not traced or not len(traced[0].rays):
            total = total + weight * unoccluded
            continue

        # The light that the medium scatters along each hidden segment is the difference of the closed-form
        # integrals up to its ends
        segments = traced[0]
        rays = directions[segments.rays]
        ends = torch.stack([segments.starts, segments.ends], dim=-1)  # (e, 2)
        depth = height_fog_optical_depth(fog, origin, rays[:, None, :], ends)[..., None] * beta
        partial = height_fog_sun_inscatter(
            depth, origin_extinction, fog.falloff, rays[:, None, 2:3], ends[..., None], sun_z
        )
        shadowed = torch.zeros_like(unoccluded).index_add_(
            0, segments.rays, segments.hidden[:, None] * (partial[:, 1] - partial[:, 0])
        )
        total = total + weight * (unoccluded - shadowed)

    whole = [lookup(table, v_z[:, 0], full_depth) for table in tables.values()]
    total = total + sum(whole[1:], whole[0]) if whole else total
    edges, counts = shadows.sky_edges, shadows.sky_counts
    cells, samples = sum(counts), shadows.middles.shape[1]
    occluded = {
        name: table
        for name, table in tables.items()
        if name in ("sky", "ground") or (name == "scattered" and scattering)
    }
    if not occluded or not shadows.middles.numel():
        return total

    # The weights of cells only need to be recomputed for media of another anisotropy than the shadows were traced with
    reweight = abs(medium.anisotropy - shadows.anisotropy) > 1e-12
    radiance = band_radiance(scattering, edges).sum(dim=-1) if "scattered" in occluded and scattering else None

    for rays in _chunks(n, int(memory // (samples * (cells * 48 if reweight else len(counts) * 128)))):
        bounds, middle = shadows.bounds[rays], shadows.middles[rays]
        if reweight:
            lit, lit_below = (
                _unpack(bits[rays], cells).to(directions.dtype) for bits in (shadows.visible, shadows.visible_below)
            )
            weights = _band_weights(edges, counts, medium.anisotropy, directions[rays], lit, lit_below)
        else:
            weights = tuple(
                getattr(shadows, name)[rays]
                for name in ("visible_weights", "weights", "visible_weights_below", "weights_below")
            )
        visible, whole_band, visible_below, whole_below = weights
        # Optical depth of the fog above each sample, per channel
        above = fog.density * torch.exp(-(origin_z + middle * v_z[rays] - fog.base_height) / fog.falloff)
        above = (above * fog.falloff)[..., None] * beta  # (r, m, c)

        ratios = {}
        if "sky" in occluded:
            attenuation = band_attenuation(edges, above)  # (r, m, c, b)
            visible_sky = (attenuation * visible[:, :, None]).sum(dim=-1)
            ratios["sky"] = _ratio(visible_sky, (attenuation * whole_band[:, None, None]).sum(dim=-1))
        if "ground" in occluded:
            # Light reflected by the ground arrives from below the horizon, which objects standing nearby hide
            ratios["ground"] = _ratio(visible_below.sum(dim=-1), whole_below.sum(dim=-1)[:, None])[..., None]
        if radiance is not None and scattering is not None:
            # The cells are weighted by the radiance arriving from them summed over channels, as their visibility is
            # the same for all channels, which is much cheaper than a ratio per channel and hardly differs from it
            n_heights = len(radiance)
            heights = torch.log(scattering.depth_ground[:1] / above[..., :1].clamp_min(1e-30))[..., 0]
            fh = heights.clamp(0, scattering.height_scale) / scattering.height_scale * (n_heights - 1)
            h = fh.floor().clamp(0, n_heights - 2).long()
            (low_visible, low_total), (high_visible, high_total) = (
                _arriving(radiance, index, weights) for index in (h, h + 1)
            )
            weight = fh - h
            visible_light = torch.lerp(low_visible, high_visible, weight)
            ratios["scattered"] = _ratio(visible_light, torch.lerp(low_total, high_total, weight))[..., None]

        boundaries = height_fog_optical_depth(fog, origin, directions[rays, None, :], bounds)[..., None] * beta
        for name, ratio in ratios.items():
            partial = lookup(occluded[name], v_z[rays].expand(-1, boundaries.shape[1]), boundaries)
            total[rays] = total[rays] - ((1 - ratio) * partial.diff(dim=1)).sum(dim=1)
    return total


def occluded_inscatter(
    occlusion: Occlusion,
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    beta: torch.Tensor,
    suns: Sequence[tuple[torch.Tensor, torch.Tensor]],
    tables: Mapping[str, ElevationTable],
    scattering: ScatteringTable | None = None,
    sun_step: float = 8.0,
    max_sun_samples: int = 1024,
    sky_samples: int = 8,
    sky_grid: float = 1.0,
    ground_height: float = 0.0,
    bias: float = 0.5,
    max_depth: float = 12.0,
    memory: float = 2**30,
) -> torch.Tensor:
    """Light from suns, the sky, the ground and multiple scattering, scattered towards rays through an exponential
    height fog, accounting for the shadows of the scene's objects, see :func:`trace` and :func:`shade`.

    Args:
        occlusion (Occlusion): Shadow maps of the scene, on the same device and with the same precision as the rays.
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        origin (torch.Tensor): Common origin of the rays, of shape (3,).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        distance (torch.Tensor): Length of the rays, of shape (n,), which is infinite for rays that hit nothing.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) when it is the same for all.
        suns (Sequence[tuple[torch.Tensor, torch.Tensor]]): Unit direction towards each sun, of shape (3,), and its
            irradiance times the medium's albedo, of shape (c',).
        tables (Mapping[str, ElevationTable]): Light from the sky, the ground and light scattered more than once,
            tabulated along rays and keyed by "sky", "ground" and "scattered", see
            :func:`tabulate_along_rays <visionsim.medium.scattering.tabulate_along_rays>`.
        scattering (ScatteringTable | None, optional): Light scattered more than once, needed to occlude the
            "scattered" table, which is otherwise not occluded. Defaults to None.
        sun_step (float, optional): Length of the intervals of rays over which sunlight's visibility is checked at
            once, in texels of its map. Those that a shadow's edge may cross are sampled every texel. Defaults to 8.
        max_sun_samples (int, optional): Maximum number of such intervals along a ray, beyond which they get longer.
            Defaults to 1024.
        sky_samples (int, optional): Number of samples of the sky's visibility along a ray. Defaults to 8.
        sky_grid (float, optional): Size of the cells of the grid to which samples of the sky's visibility are
            snapped, in texels of the finest map of the sky. Defaults to 1.
        ground_height (float, optional): Height of the ground, where light from below the horizon starts.
            Defaults to 0.
        bias (float, optional): Depth bias of shadow maps, in texels, see :func:`visibility`. Defaults to 0.5.
        max_depth (float, optional): Optical depth beyond which rays are no longer sampled, as hardly any light
            scattered further reaches the camera. Defaults to 12.
        memory (float, optional): Rough bound, in bytes, on the memory used by intermediate values, which are computed
            for as many rays at once as fit. Defaults to 1 GiB.

    Returns:
        torch.Tensor: Scattered radiance, of shape (n, c').
    """
    shadows = trace(
        occlusion,
        medium,
        origin,
        directions,
        distance,
        [towards for towards, _ in suns],
        sun_step=sun_step,
        max_sun_samples=max_sun_samples,
        sky_samples=sky_samples,
        sky_grid=sky_grid,
        ground_height=ground_height,
        bias=bias,
        max_depth=max_depth,
        memory=memory,
    )
    return shade(shadows, medium, origin, directions, distance, beta, suns, tables, scattering, memory=memory)
