"""Light of lamps scattered more than once by a medium, which spreads a halo around them.

Light that a lamp sends into the medium is scattered once towards the camera in closed form, see :func:`lamp_inscatter
<visionsim.medium.render.lamp_inscatter>`. In dense fog, a good share of it is scattered again before reaching the
camera, which spreads a halo much wider than the glow of light scattered once: in the fog of ``examples/medium``, light
scattered more than once adds 18% to the light of a lamp scattered once, and as much again as it a few meters away.

It is computed on a grid of points around the lamp, spread logarithmically in distance from the lamp and evenly in
solid angle around it. At each point, the light scattered once is gathered from directions concentrated towards the
lamp, where most of it comes from, and scattered again with the medium's phase function towards outgoing directions,
which are tabulated against their angle from the direction away from the lamp, clustered near zero as this light mostly
keeps going away from the lamp, and against their azimuth around it, as the light is no longer the same all around it
where objects cast shadows: in a shadow, it comes from the medium lit beside it. Tables leave out the density of the
medium, which varies much faster with height than the light it scatters, and which multiplies them where they are
looked up. Higher orders are iterated by integrating each order along the gathering rays, and later ones are
extrapolated as a geometric series at each point. Tables are integrated along camera rays, whose nodes are spread
evenly in angle as seen from the lamp, and interpolated in log space along distances and outgoing directions, which is
exact for powers of the distance to the lamp, and linearly between directions from the lamp, which keeps the mean of the
light where shadows make it change by orders of magnitude from one direction of the grid to the next. As they don't
depend on the camera, they are computed once per lamp and medium, see :func:`cached_lamp_halo`. They hold the logarithm
of the light relative to its peak at each point of the grid, which is small near the peak, where most of the light is,
so that half precision changes the light looked up by 0.02% on average and 0.2% at most in the night scene of
``examples/medium``, and they hold it once for all channels when these only differ by the color of the lamp, as when
extinction doesn't depend on wavelength: about 5 MB per lamp at the default resolution, along with the lamp's map and
2 MB of glow (see below), or up to 9 times as much on a grid refined around objects (see below), 36 MB and 15 MB for
the lamp of the night scene. As the halo spreads far beyond lamps, the light scattered once is gathered from a single
emitter per lamp: area lights aren't split into grids, and the patches of emissive surfaces are merged into one, see
:func:`merge_patches <visionsim.medium.lights.merge_patches>`.

The ground, at the lighting's ``ground_height``, stops light going down (and reflects it, see below). Points of the grid
under the ground gather the light at the ground right above them, and tabulate it against outgoing directions around the
direction away from the lamp there, as for the points above the ground that are interpolated from them. Given the shadow
maps of the scene, objects stop light too: the light scattered once along the gathering rays is that of the lamp, whose
shadows they cast, and the gathering rays stop at the first object they hit, as found in the visual hull of the objects
seen by the maps, see :func:`occupancy <visionsim.medium.occlusion.occupancy>`. Without it, points in the shadow of an
object gather the light scattered once on its other side, through it. Points of the grid inside objects gather no
light, and are left out where tables are interpolated, so that the light on one side of an object doesn't leak to its
other side; their values are filled in from their neighbors, and only used where no point around lies outside. As the
light that keeps going away from the lamp, which makes most of the halo, is only scattered where the lamp shines, points
lit by the lamp and points in its shadow are only interpolated from points of the grid alike, according to the lamp's
map.

Light scattered twice fades across the edges of the lamp's shadows, and into them, over angles, as seen from the lamp,
much smaller than the spacing of the grid near the objects that cast them: in a shadow, it only comes from the medium
lit beside it, at angles from the direction of the lamp that grow with the distance to the shadow's edge, where the
phase function and the lamp's light, brightest near the lamp, weaken. In the night scene of ``examples/medium``, 3 m
behind the edge of the cube, it falls tenfold within 9° of the edge of its shadow, as seen from the lamp, and a
thousandfold within 30°, while the directions of the grid are 15° apart. Around the objects that cast the lamp's shadows
within the halo's extent, as its map records them, the directions of the grid are refined, see :func:`_refined_axes`.
Light scattered more than twice is gathered from the light scattered twice, and from the light of surfaces scattered
once, so that the lamp's shadows shape it alike, but it varies more slowly: it is gathered at every point in the lamp's
shadows and next to them, while elsewhere, its ratio to the light it is gathered from, which varies slowly there, is
interpolated from the middle points of the cells of the even grid, see :func:`_gathering_points`. In the night scene,
this is within 0.3% of gathering it at every point, in a quarter of the time. Most of the light that reaches the medium
right behind an object passes right beside it, across the voxels occupied around it, up to about a voxel beyond it:
where the grid is refined, gathering rays only stop at a run of occupied voxels that they go through over about a voxel,
where they entered it, see :data:`_HIT_DEPTH`. In the night scene, this brings the lamp's light scattered more than
once in the cube's shadow from 1.26 times the light that Cycles' fog scatters more than once to 0.96 times, as
elsewhere (from 0.92 times), which takes 3 to 4 times as long to compute, see :attr:`Effects.halo_shadow_edges
<visionsim.medium.model.Effects.halo_shadow_edges>`. Both are needed: on the refined grid, rays that stop at the first
occupied voxel make the lamp's light scattered more than once 0.90 times Cycles' in the shadow, against 0.94 times
elsewhere, while on the even grid, rays that go through these voxels make it 1.35 times Cycles' in the shadow, as the
two errors no longer partly make up for each other, so that they only go on where the grid is refined.

The light gathered at each point of the grid also lights surfaces there: it is the glow of the medium, i.e. the light of
the lamp scattered once and more, and the light reflected by surfaces and scattered, that reaches a point from every
direction. Its irradiance on surfaces of any orientation is tabulated at each point from the light gathered for each
order, see :func:`halo_glow`, and later orders are extrapolated as the light they scatter is. In the night scene of
``examples/medium``, the glow adds 19% to the light that reaches the ground the camera sees directly from the lamp, 2%
of which is light reflected by surfaces, within 1% of gathering it along rays from each point of the ground. Where the
glow lights surfaces, the light of the lamp reaches them with the full extinction, rather than with the reduced
extinction of the similarity relation, which also counts the light that the medium scatters forward on its way, see
:func:`lamp_irradiance <visionsim.medium.surfaces.lamp_irradiance>`. Over the outermost cell of the grid, the glow fades
out, and the similarity relation takes over, which holds beyond the halo's extent, see :func:`halo_share`.

Surfaces lit by the lamp reflect part of its light into the medium, see :mod:`visionsim.medium.reflections`, which
camera rays gather once scattered, see :func:`reflection_inscatter <visionsim.medium.reflections.reflection_inscatter>`,
and which is scattered more than once here, given as ``reflection``: the light that surfaces reflect, gathered at each
point of the grid from directions, from the ground and from objects, see :func:`arriving_light
<visionsim.medium.reflections.arriving_light>`, and scattered once there, is added to the lamp's light scattered twice
before it is gathered again, up to one order less than the lamp's light. Surfaces are lit by the lamp directly, with the
full extinction, and by the glow, see :func:`relit_reflection <visionsim.medium.reflections.relit_reflection>`: by the
light of the lamp scattered once, as soon as it is gathered, before their light is scattered more than once, and by the
whole glow at the end, which camera rays gather once scattered, see :attr:`LampHalo.reflection`. The glow of each
later order lights surfaces too: the light of surfaces lit by the light that arrived at the points along with an order,
scattered once at the points, is added to the next order before it is gathered, so that it is scattered more than once
too. The light of surfaces lit by the glow of the last order, and of the orders extrapolated beyond it, is only
gathered once scattered by camera rays, while the extrapolation, from the ratio of the last two orders, the last of
which holds some light of surfaces lit by the glow of the previous one, follows part of it. In the night scene of
``examples/medium``, the glow beyond the lamp's light scattered once is about a quarter of the glow that lights
surfaces. Cached halos compute the light that surfaces reflect along with them, given the albedo of the ground and of
objects, and keep it, 28 MB more for the lamp of that scene at the default resolution of
:mod:`visionsim.medium.reflections`, or 44 MB with the patches of every surface and the specular layer of surfaces,
whose light towards the points of the grid is kept, and 98 MB when that light differs between channels, e.g. for
colored objects, see :func:`cached_lamp_halo`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch

from visionsim.medium.lights import merge_patches
from visionsim.medium.model import AreaLight, Blob, EmissiveSurface, Lamp, Medium
from visionsim.medium.occlusion import (
    Occlusion,
    Occupancy,
    ShadowMaps,
    hit_distances,
    lamp_map_index,
    lamp_visibility,
    occlusion_to,
    occupancy,
    occupied,
    select_maps,
)
from visionsim.medium.optics import density, henyey_greenstein, optical_depth
from visionsim.medium.reflections import (
    REFLECTION_GRID,
    LampReflection,
    ReflectionGrid,
    Surfels,
    arriving_light,
    dark,
    lamp_reflection,
    relit_reflection,
)
from visionsim.medium.surfaces import _clamped_cosine


class HaloGrid(NamedTuple):
    """Resolution of the tables of the light of lamps scattered more than once."""

    radii: int = 24
    """number of distances to the lamp, spread logarithmically over ``extent``"""
    extent: tuple[float, float] = (0.05, 40.0)
    """smallest and largest distance to the lamp, in meters, beyond which nothing is tabulated"""
    directions: tuple[int, int] = (12, 24)
    """number of polar angles and azimuths of directions from the lamp, whose vertical components and azimuths are
    spread evenly"""
    angles: tuple[int, int] = (32, 12)
    """number of angles between outgoing directions and the direction away from the lamp, clustered near zero, and of
    azimuths around the latter"""
    gathered: tuple[int, int] = (24, 24)
    """number of angles from the direction towards the lamp, clustered near zero, and of azimuths around it, of the
    directions from which light is gathered at each point"""
    orders: int = 4
    """highest order of scattering that is iterated, beyond which orders are extrapolated"""
    refinement: int = 3
    """number of cells into which each polar and azimuthal cell of the directions from the lamp is split where objects
    lie within ``extent``, as the lamp's map records them, and next to them, so as to resolve the lamp's shadows on
    the light scattered more than once, see :func:`_refined_axes`, odd so that the directions of the even grid, from
    which light scattered more than twice is interpolated, are among them, see :func:`_gathering_points`"""


HALO_GRID = HaloGrid()
"""Default resolution of the tables of the light of lamps scattered more than once"""
_SHADOW_SAMPLES = 64
"""Number of samples of the visibility of the lamp along each gathering ray, fewer than along camera rays as gathering
rays are themselves several degrees apart, see :func:`lamp_shadow <visionsim.medium.occlusion.lamp_shadow>`"""
_INSIDE_WEIGHT = 1e-3
"""Weight of the points of the grid that are left out where tables are interpolated, inside objects or lit by the lamp
unlike the point looked up, relative to the others, which only matters where no point around is left"""
_EDGE = 0.1
"""Relative difference of the distances to the surfaces recorded by neighboring texels of a lamp's map beyond which,
if it is more than twice what the surface does from the texel before, the nearer one is the edge of an object, around
which the grid's directions are refined, see :func:`_refined_axes`"""
_HIT_DEPTH = 0.99
"""Distance, in voxels, over which gathering rays go through consecutive occupied voxels before they stop where they
entered them, as rays that hit objects do, which go through a voxel at least, even straight through a single layer of
voxels, while rays that pass right beside objects, across the voxels occupied up to about a voxel around them, see
:func:`occupancy <visionsim.medium.occlusion.occupancy>`, mostly go through less: a little less than a voxel, so that
rounding doesn't let rays through a single layer, see :func:`hit_distances
<visionsim.medium.occlusion.hit_distances>`"""
_SHARED_CHANNELS = 1e-3
"""Largest difference between the logarithms of the light of channels relative to their peaks, below which channels
only differ by the color of the lamp, and are stored once"""
HALO_CACHE_BYTES = 2**30
"""Memory, in bytes, beyond which the least recently used halos are dropped from the cache of :func:`cached_lamp_halo`,
which holds about 100 halos of the default resolution with maps of 512 by 1024 texels, or about 28 with the light
reflected by surfaces that they keep without the switches that need more of it, which isn't cached on its own then,
unlike that of lamps without halos, see :func:`cached_lamp_reflection
<visionsim.medium.reflections.cached_lamp_reflection>`, but only about 10 with every switch on for the night's lamp of
``examples/medium``, whose grid is refined around the cube and the walls, 94 MB each, or fewer when the light of
surfaces differs between channels: the halos of a scene with more lamps than the cache holds are all computed again at
every frame, as each drops the least recently used one"""
_ZONAL = 17
"""Number of cosines between normals and the direction towards the lamp, spread evenly from one down to minus one, at
which the irradiance of the glow's mean around that direction is tabulated, see :func:`halo_glow`"""
_LOBE = (math.pi, *(2 * math.pi / 3,) * 3, *(math.pi / 4,) * 5)
"""Coefficients of the clamped cosine on the real spherical harmonics of degree 0, 1 and 2, see :func:`_harmonics`,
which turn those of radiance into those of the irradiance of surfaces"""


class LampHalo(NamedTuple):
    """Light of a lamp scattered more than once by a medium, tabulated on a grid around the lamp."""

    center: torch.Tensor
    """position of the lamp, of shape (3,)"""
    extent: tuple[float, float]
    """smallest and largest distance to the lamp, in meters, spread logarithmically over the first axis of ``values``"""
    ground: float
    """height of the ground, below which there is no light"""
    values: torch.Tensor
    """logarithm of the radiance scattered towards outgoing directions per meter and per unit of the medium's relative
    density, relative to its peak at each point of the grid, ``peaks``, against distances to the lamp, polar angles and
    azimuths of directions from it, and angles between outgoing directions and the direction away from the lamp and
    their azimuths around it, measured from the direction in which the polar angle grows, of shape (r, t, p, g, b, c)
    or (r, t, p, g, b, 1) if channels only differ by their peaks, in half precision"""
    peaks: torch.Tensor
    """logarithm of the largest radiance scattered towards any outgoing direction at each point of the grid, of shape
    (r, t, p, c)"""
    outside: torch.Tensor | None = None
    """whether each point of the grid lies outside of objects (1) or inside one (0), whose values are filled in from its
    neighbors and only used where no point around lies outside, of shape (r, t, p), or None if none lies inside"""
    lit: torch.Tensor | None = None
    """visible fraction of the lamp at each point of the grid, of shape (r, t, p), or None if it casts no shadows"""
    maps: ShadowMaps | None = None
    """copy of the map of the lamp alone, in single precision, from which the visible fraction of the lamp is looked up
    where tables are interpolated"""
    glow: torch.Tensor | None = None
    """glow of the medium at each point of the grid, i.e. the light of the lamp scattered once and more, and the light
    reflected by surfaces and scattered, that reaches the point from every direction: irradiance of its mean around the
    direction towards the lamp for normals at each of z cosines with that direction, spread evenly from one down to minus
    one, followed by the coefficients of the rest of its radiance on the real spherical harmonics of degree up to 2,
    times the point's distance to the lamp, of shape (r, t, p, z + 9, c), in single precision, see :func:`halo_glow`"""
    reflection: LampReflection | None = None
    """light of the lamp reflected by surfaces, lit by the lamp directly and by the glow, see :func:`relit_reflection
    <visionsim.medium.reflections.relit_reflection>`, which camera rays gather once scattered, or None"""
    axes: tuple[torch.Tensor, torch.Tensor] | None = None
    """vertical components of the directions of the points of the grid from the lamp, decreasing, of shape (t,), and
    their azimuths, increasing within [0, 2π), of shape (p,), refined around objects, see :func:`_refined_axes`, or
    None if both are spread evenly, as by :func:`_lat_long`"""


def _even_axes(n_theta: int, n_phi: int, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertical components and azimuths spread evenly, decreasing from the top and increasing from zero, of shapes
    (n_theta,) and (n_phi,), so that the directions they make each cover the same solid angle."""
    z = 1 - (torch.arange(n_theta, **kwargs) + 0.5) / n_theta * 2
    phi = (torch.arange(n_phi, **kwargs) + 0.5) / n_phi * 2 * math.pi
    return z, phi


def _directions(polar: torch.Tensor, azimuths: torch.Tensor) -> torch.Tensor:
    """Unit directions of every vertical component of shape (t,) with every azimuth of shape (p,), of shape
    (t * p, 3)."""
    z, f = torch.meshgrid(polar, azimuths, indexing="ij")
    s = (1 - z * z).sqrt()
    return torch.stack([s * torch.cos(f), s * torch.sin(f), z], -1).reshape(-1, 3)


def _lat_long(n_theta: int, n_phi: int, **kwargs) -> torch.Tensor:
    """Unit directions whose vertical components and azimuths are spread evenly, so that each covers the same solid
    angle, of shape (n_theta * n_phi, 3)."""
    return _directions(*_even_axes(n_theta, n_phi, **kwargs))


def _refined_axes(
    maps: ShadowMaps, center: torch.Tensor, ground: float, grid: HaloGrid
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Vertical components and azimuths of the directions of the points of the grid from a lamp, refined around the
    edges of the objects that cast its shadows.

    The light that the lamp scatters once and that is scattered again fades across the edges of the lamp's shadows, and
    into them, over angles much smaller than the spacing of the grid's directions, see :mod:`visionsim.medium.halos`.
    Each polar and azimuthal cell of the even grid of directions that holds the edge of an object within the extent of
    the halo, as the lamp's map records it, or that is next to one, is split into ``grid.refinement`` cells of equal
    solid angle, whose centers are the directions of the points of the grid, so that grids stay products of vertical
    components and azimuths, which refines the directions in between edges as well, such as right behind objects. An
    edge is a texel of an object next to one that records a surface further away by more than :data:`_EDGE`, or none,
    and by more than twice what the surface does from the texel before, which leaves out surfaces seen at grazing
    angles, while the ground, which the map needn't record, is a surface, and isn't an object: the walls of a closed
    room, which cast no shadows within it, have no edges, nor does an object where it stands on the ground.

    Args:
        maps (ShadowMaps): Map of the lamp alone, see :func:`lamp_visibility
            <visionsim.medium.occlusion.lamp_visibility>`.
        center (torch.Tensor): Position of the lamp, of shape (3,).
        ground (float): Height of the ground.
        grid (HaloGrid): Resolution of the tables.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | None: Vertical components, decreasing, of shape (t,), and azimuths,
        increasing within [0, 2π), of shape (p,), in the precision of ``center``, or None if no edge of an object lies
        within the extent of the halo, or if the grid isn't refined.
    """
    if grid.refinement <= 1:
        return None
    n_t, n_p = grid.directions
    height, width = (int(size) for size in maps.shapes[0])
    depths = maps.depths[: height * width].reshape(height, width).to(center.dtype)
    kwargs: dict[str, Any] = {"dtype": center.dtype, "device": center.device}
    # Directions of the texels, whose rows go from the bottom of the map up, see `lamp_visibility`
    rows, columns = torch.arange(height, **kwargs), torch.arange(width, **kwargs)
    z = torch.cos(math.pi * (1 - (rows + 0.5) / height))
    azimuths = torch.remainder(math.pi - 2 * math.pi * (columns + 0.5) / width, 2 * math.pi)
    # Distance to the ground along the middle and the lowest direction of each texel. Surfaces at the ground or beyond
    # it aren't objects
    above = float(center[2]) - ground
    lowest = torch.cos(math.pi * (1 - rows / height))
    to_ground = [torch.where(c < 0, above / (-c).clamp_min(1e-12), torch.full_like(c, math.inf)) for c in (z, lowest)]
    objects = depths < torch.clamp(0.99 * to_ground[1], max=grid.extent[1])[:, None]
    surfaces = torch.minimum(depths, to_ground[0][:, None])
    # Edges of objects, against surfaces further away along rows or columns, which wrap around: a jump in distance from
    # one texel to the next, beyond what the surface does from the texel before, as it does when seen at grazing angles

    def shifted(shift: int, axis: int) -> torch.Tensor:
        """Distances of the next texels along an axis, or of the same texel beyond the poles."""
        beside = torch.roll(surfaces, shift, axis)
        if axis == 0:
            first = 0 if shift > 0 else height - 1
            beside[first] = surfaces[first]
        return beside

    further = torch.zeros_like(objects)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        jump, before = shifted(shift, axis) - surfaces, (surfaces - shifted(-shift, axis)).clamp_min(0)
        further |= (jump > _EDGE * surfaces) & (jump > 2 * before)
    edges = objects & further
    if not bool(edges.any()):
        return None
    # Cells that hold edges, and their neighbors, along each axis, the azimuths wrapping around
    polar_cells = torch.zeros(n_t, dtype=torch.bool, device=center.device)
    polar_cells[((1 - z[edges.any(dim=1)]) / 2 * n_t).long().clamp(0, n_t - 1)] = True
    azimuth_cells = torch.zeros(n_p, dtype=torch.bool, device=center.device)
    azimuth_cells[(azimuths[edges.any(dim=0)] / (2 * math.pi) * n_p).long().clamp(0, n_p - 1)] = True
    polar_cells = polar_cells | _shifted(polar_cells, 1, 0, False) | _shifted(polar_cells, -1, 0, False)
    azimuth_cells = azimuth_cells | torch.roll(azimuth_cells, 1) | torch.roll(azimuth_cells, -1)

    def centers(cells: torch.Tensor) -> torch.Tensor:
        """Fractions of the axis at the centers of the cells, of those split, of shape (n,)."""
        split = cells.long() * (grid.refinement - 1) + 1
        index = torch.repeat_interleave(torch.arange(len(cells), device=center.device), split)
        within = torch.arange(len(index), device=center.device) - torch.repeat_interleave(
            torch.cumsum(split, 0) - split, split
        )
        return (index.to(**kwargs) + (within.to(**kwargs) + 0.5) / split[index].to(**kwargs)) / len(cells)

    return 1 - centers(polar_cells) * 2, centers(azimuth_cells) * 2 * math.pi


def _frames_around(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors completing unit axes of shape (n, 3) into orthonormal frames."""
    up = (axis[:, 2].abs() < 0.9)[:, None]
    helper = torch.where(up, axis.new_tensor([0.0, 0.0, 1.0]), axis.new_tensor([1.0, 0.0, 0.0]))
    first = torch.linalg.cross(helper, axis)
    first = first / first.norm(dim=-1, keepdim=True)
    return first, torch.linalg.cross(axis, first)


def _spherical_frames(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors along which the polar angle and the azimuth of unit axes of shape (n, 3) grow, which complete them
    into orthonormal frames that turn smoothly from one axis to the next, except at the poles, where the frames of
    :func:`_frames_around` take over."""
    across = axis[:, :2].norm(dim=-1, keepdim=True)
    pole = across < 1e-6
    safe = torch.where(pole, torch.ones_like(across), across)
    polar = torch.cat([axis[:, 2:3] * axis[:, :2] / safe, -across], dim=-1)
    azimuthal = torch.cat([-axis[:, 1:2] / safe, axis[:, 0:1] / safe, torch.zeros_like(across)], dim=-1)
    first, second = _frames_around(axis)
    return torch.where(pole, first, polar), torch.where(pole, second, azimuthal)


def _around(
    axis: torch.Tensor,
    polar: torch.Tensor,
    azimuth: torch.Tensor,
    frames: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Directions at polar angles and azimuths around unit axes, of shape (n, k, 3) for axes of shape (n, 3) and angles
    of shape (k,), with azimuths measured from the first vector of given frames, those of :func:`_frames_around` by
    default."""
    first, second = _frames_around(axis) if frames is None else frames
    local = torch.stack(
        [torch.sin(polar) * torch.cos(azimuth), torch.sin(polar) * torch.sin(azimuth), torch.cos(polar)], -1
    )
    return (
        local[None, :, 0:1] * first[:, None]
        + local[None, :, 1:2] * second[:, None]
        + local[None, :, 2:3] * axis[:, None]
    )


def _to_ground(origins: torch.Tensor, directions: torch.Tensor, ground: float) -> torch.Tensor:
    """Distance along rays to the ground, or infinity for rays that don't go down."""
    down = directions[..., 2] < 0
    distance = (origins[..., 2] - ground).clamp_min(0) / (-directions[..., 2]).clamp_min(1e-12)
    return torch.where(down, distance, torch.full_like(distance, math.inf))


def _shifted(values: torch.Tensor, shift: int, axis: int, periodic: bool) -> torch.Tensor:
    """Values moved by one along an axis, which wrap around if periodic, and are zero where they would otherwise."""
    rolled = torch.roll(values, shift, axis)
    if not periodic:
        rolled.narrow(axis, 0 if shift > 0 else values.shape[axis] - 1, 1).zero_()
    return rolled


def _fill(values: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Values at the points of the grid, of shape (r, t, p, ...), where those that aren't known, given which are, of
    shape (r, t, p), are replaced by the mean of their known neighbors, spreading out from the known points."""
    values, known = values.clone(), known.clone()
    trailing = (1,) * (values.ndim - 3)
    while bool(known.any()) and not bool(known.all()):
        weight = known.to(values.dtype)
        sums, counts = torch.zeros_like(values), torch.zeros_like(weight)
        for axis in range(3):
            for shift in (1, -1):
                # Azimuths wrap around
                sums += _shifted(values * weight.reshape(*weight.shape, *trailing), shift, axis, axis == 2)
                counts += _shifted(weight, shift, axis, axis == 2)
        newly = ~known & (counts > 0)
        values[newly] = sums[newly] / counts[newly].reshape(-1, *trailing)
        known |= newly
    return values


def _harmonics(directions: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonics of degree 0, 1 and 2 of unit directions of shape (..., 3), of shape (..., 9), ordered by
    degree and then by order, as in Ramamoorthi and Hanrahan, "An Efficient Representation for Irradiance Environment
    Maps", SIGGRAPH 2001."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    linear, quadratic = 0.488603, 1.092548
    return torch.stack(
        [
            torch.full_like(x, 0.282095),
            linear * y,
            linear * z,
            linear * x,
            quadratic * x * y,
            quadratic * y * z,
            0.315392 * (3 * z * z - 1),
            quadratic * x * z,
            quadratic / 2 * (x * x - y * y),
        ],
        dim=-1,
    )


def _neighbors(x: torch.Tensor, size: int, wrap: bool) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Both neighbors of fractional indices along an axis of the grid, and the weight of each, where azimuths wrap
    around."""
    low = x.floor() if wrap else x.floor().clamp(max=size - 2)
    indices = (low.long(), low.long() + 1)
    indices = tuple(torch.remainder(i, size) if wrap else i.clamp(0, size - 1) for i in indices)
    return tuple(zip(indices, (1 - (x - low), x - low)))


class _View(NamedTuple):
    """Points of a grid of directions among those of a refined grid, at every distance from the lamp."""

    shape: tuple[int, int]
    """number of vertical components and of azimuths of their directions"""
    outside: torch.Tensor | None
    """whether each point lies outside of objects, of shape (r, t, p), or None if none lies inside"""
    lit: torch.Tensor | None
    """visible fraction of the lamp at each point, of shape (r, t, p), or None if it casts no shadows"""
    axes: tuple[torch.Tensor, torch.Tensor]
    """vertical components and azimuths of their directions, see :attr:`LampHalo.axes`"""


def _ratio(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Ratio of values to positive reference values, which is zero where there is no reference light, such as inside
    objects, rather than huge."""
    tiny = torch.finfo(reference.dtype).tiny
    return torch.where(reference > tiny, values / reference.clamp_min(tiny), torch.zeros_like(values))


def _middles(cells: torch.Tensor) -> torch.Tensor:
    """Index of the middle one of the nodes of each cell, or of the first of the two in the middle, given the cell of
    each node, which increase, of shape (n,), of shape (cells,): the node at the center of the cell when cells are
    split into an odd number of cells of equal solid angle."""
    _, counts = torch.unique_consecutive(cells, return_counts=True)
    return torch.cumsum(counts, 0) - counts + (counts - 1) // 2


def _gathering_points(
    axes: tuple[torch.Tensor, torch.Tensor], even: tuple[int, int], lit: torch.Tensor | None, radii: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Points of a grid refined around objects at which light scattered more than twice is gathered, rather than
    interpolated from other points, see :func:`lamp_halo`.

    Light scattered more than twice is gathered from the light scattered twice, and from the light of surfaces
    scattered once, which the lamp's shadows shape, but it varies more slowly. Away from the shadows, its ratio to the
    light it is gathered from varies slowly, and is interpolated linearly between the middle points of the cells of the
    even grid, which are its points for an odd refinement, see :func:`_spread`, where it is gathered. In the shadows,
    and next to them, where the light it is gathered from fades over much shorter distances than it does, so that its
    ratio to it grows deeper into the shadows, it is gathered at every point: in the night scene of
    ``examples/medium``, interpolating it there too makes it 8% too bright in the cube's shadow, and 17% next to its
    edges.

    Args:
        axes (tuple[torch.Tensor, torch.Tensor]): Vertical components and azimuths of the directions of the refined
            grid, of shapes (t,) and (p,), see :attr:`LampHalo.axes`.
        even (tuple[int, int]): Number of vertical components and of azimuths of the even grid.
        lit (torch.Tensor | None): Visible fraction of the lamp at each point of the grid, of shape (r, t, p), or None
            if it casts no shadows.
        radii (int): Number of distances to the lamp.

    Returns:
        tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: Indices of the points at which it is
        gathered, increasing, of shape (n,), indices of the middle points of the cells of the even grid, of shape
        (r * t' * p',), and the indices of their vertical components and azimuths among those of the grid, of shapes
        (t',) and (p',).
    """
    n_t, n_p = len(axes[0]), len(axes[1])
    rows = _middles(((1 - axes[0]) / 2 * even[0]).floor())
    columns = _middles((axes[1] / (2 * math.pi) * even[1]).floor())
    index = torch.arange(radii * n_t * n_p, device=axes[0].device).reshape(radii, n_t, n_p)
    middle = index[:, rows][:, :, columns].reshape(-1)
    gathered = torch.zeros(radii * n_t * n_p, dtype=torch.bool, device=axes[0].device)
    gathered[middle] = True
    if lit is not None:
        # Points that the lamp doesn't fully light, and their neighbors along the directions of the grid, whose
        # azimuths wrap around
        shaded = lit < 1
        near = shaded.clone()
        for axis in (1, 2):
            for shift in (1, -1):
                near |= _shifted(shaded, shift, axis, axis == 2)
        gathered |= near.reshape(-1)
    return gathered.nonzero()[:, 0], middle, (rows, columns)


def _spread(
    values: torch.Tensor,
    axes: tuple[torch.Tensor, torch.Tensor],
    view: _View,
    lit: torch.Tensor | None,
) -> torch.Tensor:
    """Values at the points of a view of a refined grid, of shape (r * t' * p', ...), interpolated to every point of
    the grid, of shape (r * t * p, ...), linearly between directions, from the points alike, as tables are, see
    :func:`_alike`. Values at points inside objects are filled in from their neighbors first, as those of tables are, in
    case no point around lies outside.

    Args:
        values (torch.Tensor): Values at the points of the view, of shape (r * t' * p', ...).
        axes (tuple[torch.Tensor, torch.Tensor]): Vertical components and azimuths of the directions of the refined
            grid, of shapes (t,) and (p,), see :attr:`LampHalo.axes`.
        view (_View): Points of the view.
        lit (torch.Tensor | None): Visible fraction of the lamp at each point of the refined grid, of shape (r, t, p),
            or None.

    Returns:
        torch.Tensor: Values at every point of the refined grid, of shape (r * t * p, ...).
    """
    (n_t, n_p), (m_t, m_p) = (len(axes[0]), len(axes[1])), view.shape
    values = values.reshape(-1, m_t, m_p, *values.shape[1:])
    if view.outside is not None:
        values = _fill(values, view.outside)
    trailing = (1,) * (values.ndim - 3)
    polar, azimuths = view.axes
    around = torch.cat([azimuths[-1:] - 2 * math.pi, azimuths, azimuths[:1] + 2 * math.pi])
    t_neighbors = _neighbors(_fractional(-polar, -axes[0]), m_t, False)
    p_neighbors = _neighbors(_fractional(around, axes[1]) - 1, m_p, True)
    result = torch.zeros(len(values), n_t, n_p, *values.shape[3:], dtype=values.dtype, device=values.device)
    total = torch.zeros(len(values), n_t, n_p, dtype=values.dtype, device=values.device)
    for t, t_weight in t_neighbors:
        for p, p_weight in p_neighbors:
            weight = (t_weight[:, None] * p_weight[None, :]).expand(len(values), -1, -1)
            if view.outside is not None or view.lit is not None:
                alike = view.outside[:, t][:, :, p] if view.outside is not None else torch.ones_like(weight)
                if view.lit is not None and lit is not None:
                    alike = alike * (1 - (view.lit[:, t][:, :, p] - lit).abs())
                weight = weight * (alike + _INSIDE_WEIGHT)
            result += weight.reshape(*weight.shape, *trailing) * values[:, t][:, :, p]
            total += weight
    return (result / total.reshape(*total.shape, *trailing)).reshape(-1, *values.shape[3:])


def _grid_neighbors(halo: LampHalo, points: torch.Tensor, shape: tuple[int, int, int]) -> tuple[torch.Tensor, ...]:
    """Distance of points of shape (n, 3) to the lamp, of shape (n,), unit directions away from it, of shape (n, 3),
    and both neighbors of the points along each axis of the grid of the given shape, by distance, polar angle and
    azimuth of directions from the lamp, with their weights."""
    n_r, n_t, n_p = shape
    r_min, r_max = halo.extent
    offset = points - halo.center
    r = offset.norm(dim=-1).clamp_min(1e-12)
    away = offset / r[:, None]
    azimuth = torch.remainder(torch.atan2(away[:, 1], away[:, 0]), 2 * math.pi)
    coordinates = [((torch.log(r / r_min) / math.log(r_max / r_min)) * (n_r - 1)).clamp(0, n_r - 1)]
    if halo.axes is None:
        coordinates += [((1 - away[:, 2]) / 2 * n_t - 0.5).clamp(0, n_t - 1), azimuth / (2 * math.pi) * n_p - 0.5]
    else:
        # Linearly between the vertical components and the azimuths of the grid, the latter wrapping around
        polar, azimuths = (axis.to(points) for axis in halo.axes)
        around = torch.cat([azimuths[-1:] - 2 * math.pi, azimuths, azimuths[:1] + 2 * math.pi])
        coordinates += [_fractional(-polar, -away[:, 2]), _fractional(around, azimuth) - 1]
    return (r, away, *(_neighbors(x, size, wrap) for x, size, wrap in zip(coordinates, shape, (False, False, True))))


def _fractional(nodes: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Fractional indices of values of shape (n,) among increasing nodes, linearly between them, and clamped to the
    first and the last, of shape (n,)."""
    i = (torch.searchsorted(nodes, x.contiguous()) - 1).clamp(0, len(nodes) - 2)
    low, high = nodes[i], nodes[i + 1]
    return i.to(x.dtype) + ((x - low) / (high - low)).clamp(0, 1)


def _alike(halo: LampHalo, points: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Weight of points of the grid of given indices, of shape (n,), relative to the others where points of shape
    (n, 3) are interpolated, or None if all are alike.

    The light on one side of an object doesn't reach its other side, which points inside it would blend, and the light
    that keeps going away from the lamp is only scattered where the lamp shines, so that points lit by the lamp, or in
    its shadow, are only interpolated from points of the grid alike, according to the lamp's map, unless no point
    around is left.
    """
    outside = halo.outside.reshape(-1).to(points.dtype) if halo.outside is not None else None
    lit, visible = None, None
    if halo.lit is not None and halo.maps is not None:
        lit = halo.lit.reshape(-1).to(points.dtype)
        visible = lamp_visibility(halo.maps, 0, points.to(halo.maps.depths.dtype)).to(points.dtype)
    if outside is None and lit is None:
        return None

    def weight(nodes: torch.Tensor) -> torch.Tensor:
        factor = outside[nodes] if outside is not None else torch.ones_like(points[:, 0])
        if lit is not None and visible is not None:
            factor = factor * (1 - (lit[nodes] - visible).abs())
        return factor + _INSIDE_WEIGHT

    return weight


def _interpolate(halo: LampHalo, points: torch.Tensor, outgoing: torch.Tensor) -> torch.Tensor:
    """Radiance scattered per meter and per unit of the medium's relative density towards unit directions ``outgoing``
    at ``points``, both of shape (n, 3), of shape (n, c).

    The table is interpolated in log space along distances to the lamp and outgoing directions, along which it varies as
    powers of them, and linearly between directions from the lamp, which keeps the mean of the light where the shadows
    of objects make it change by orders of magnitude from one direction of the grid to the next. Points of the grid
    inside objects, and points of the grid that the lamp lights unlike the point, according to its map, are left out,
    unless no point around is left, see :func:`_alike`.
    """
    n_r, n_t, n_p, n_g, n_b, shared = halo.values.shape
    n_c = halo.peaks.shape[-1]
    r_max = halo.extent[1]
    r, away, r_neighbors, t_neighbors, p_neighbors = _grid_neighbors(halo, points, (n_r, n_t, n_p))
    first, second = _spherical_frames(away)
    azimuth = torch.atan2((outgoing * second).sum(dim=-1), (outgoing * first).sum(dim=-1))
    g_neighbors = _neighbors(
        (torch.acos((outgoing * away).sum(dim=-1).clamp(-1, 1)) / math.pi).sqrt() * (n_g - 1), n_g, False
    )
    b_neighbors = _neighbors(torch.remainder(azimuth, 2 * math.pi) / (2 * math.pi) * n_b - 0.5, n_b, True)
    outgoing_neighbors = [
        (g * n_b + b, g_weight * b_weight) for g, g_weight in g_neighbors for b, b_weight in b_neighbors
    ]
    values, peaks = halo.values.reshape(-1, shared), halo.peaks.reshape(-1, n_c).to(points.dtype)
    alike = _alike(halo, points)
    result = torch.zeros(len(points), n_c, dtype=points.dtype, device=points.device)
    total = torch.zeros_like(result[:, 0])
    for t, t_weight in t_neighbors:
        for p, p_weight in p_neighbors:
            nodes = [(i * n_t + t) * n_p + p for i, _ in r_neighbors]
            r_weights = [w for _, w in r_neighbors]
            if alike is not None:
                r_weights = [w * alike(node) for w, node in zip(r_weights, nodes)]
            # Log space along distances and outgoing directions, whose corners are looked up at once, and to which the
            # peaks of both distances add, as the weights of outgoing directions sum to one
            index = torch.stack([node * (n_g * n_b) + o for node in nodes for o, _ in outgoing_neighbors], dim=-1)
            weights = torch.stack([r_w * o_w for r_w in r_weights for _, o_w in outgoing_neighbors], dim=-1)
            mass = r_weights[0] + r_weights[1]
            log = torch.einsum("nk,nkc->nc", weights, values[index].to(points.dtype))
            log = log + r_weights[0][:, None] * peaks[nodes[0]] + r_weights[1][:, None] * peaks[nodes[1]]
            log = log / mass[:, None]
            weight = t_weight * p_weight * mass
            result = result + weight[:, None] * torch.exp(log)
            total = total + weight
    inside = (r <= r_max) & (points[:, 2] > halo.ground)
    return torch.where(inside[:, None], result / total[:, None], torch.zeros_like(result))


def halo_share(halo: LampHalo, points: torch.Tensor) -> torch.Tensor:
    """Share of the light of a lamp whose glow a halo models at points, of shape (n,): one within the halo's grid, which
    fades out linearly in log distance to the lamp over the outermost cell of the grid, down to zero at its extent and
    beyond, where the glow isn't tabulated.

    Args:
        halo (LampHalo): Light of the lamp scattered more than once, see :func:`lamp_halo`.
        points (torch.Tensor): Points, of shape (n, 3).

    Returns:
        torch.Tensor: Share in [0, 1], of shape (n,).
    """
    n_r = halo.glow.shape[0] if halo.glow is not None else halo.peaks.shape[0]
    r_min, r_max = halo.extent
    r = (points - halo.center).norm(dim=-1).clamp_min(r_min)
    return ((torch.log(r_max / r) / math.log(r_max / r_min)) * (n_r - 1)).clamp(0, 1)


def halo_glow(halo: LampHalo, points: torch.Tensor, normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Irradiance of surfaces from the glow of the medium around a lamp, i.e. from the light of the lamp scattered once
    and more, and from the light reflected by surfaces and scattered, if any, which reaches them from every direction.

    Light scattered once mostly arrives from about the direction of the lamp, as the medium scatters it forward, and
    about evenly around that direction where nothing shadows it. The irradiance of its mean around that direction is
    tabulated against the cosine between normals and the direction towards the lamp, which holds it exactly, and the rest
    of the radiance on the real spherical harmonics of degree up to 2, whose irradiance is within a few percent of that
    of any radiance (Ramamoorthi and Hanrahan, "An Efficient Representation for Irradiance Environment Maps", SIGGRAPH
    2001), while it is up to 8% off for light from a single direction, such as the light scattered once near the lamp.
    Both are interpolated linearly between the tabulated cosines, and between the points of the grid, times the distance
    to the lamp, as the glow of the light scattered once decreases about as its inverse near the lamp, from points of the
    grid alike, as tables are, see :func:`halo_inscatter`. Points below the ground receive that of the ground right above
    them. Over the outermost cell of the grid, the glow fades out, down to zero at the halo's extent, see
    :func:`halo_share`, where the light of the lamp should be attenuated with the extinction reduced by the similarity
    relation instead, as without the glow, so that surfaces are lit continuously across it, see :func:`surface_irradiance
    <visionsim.medium.surfaces.surface_irradiance>`.

    Args:
        halo (LampHalo): Light of the lamp scattered more than once, with its glow, see :func:`lamp_halo`.
        points (torch.Tensor): Points on surfaces, of shape (n, 3).
        normals (torch.Tensor): Unit normals of the surfaces, of shape (n, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Irradiance per channel, faded out at the halo's extent, of shape (n, c), and
        the share of the lamp's light whose glow it is, see :func:`halo_share`, of shape (n,), or zeros for both if
        the halo has no glow.
    """
    if halo.glow is None:
        return points.new_zeros(len(points), halo.peaks.shape[-1]), points.new_zeros(len(points))
    size = 1 << 16
    if len(points) > size:
        # The glow is looked up at the 8 corners of the grid around each point, so points are processed in chunks
        parts = [halo_glow(halo, points[i : i + size], normals[i : i + size]) for i in range(0, len(points), size)]
        return torch.cat([irradiance for irradiance, _ in parts]), torch.cat([share for _, share in parts])
    n_r, n_t, n_p, n_h, n_c = halo.glow.shape
    r, away, r_neighbors, t_neighbors, p_neighbors = _grid_neighbors(halo, points, (n_r, n_t, n_p))
    glow = halo.glow.reshape(-1, n_h * n_c)
    alike = _alike(halo, points)
    result = points.new_zeros(len(points), n_h * n_c)
    total = points.new_zeros(len(points))
    for t, t_weight in t_neighbors:
        for p, p_weight in p_neighbors:
            for i, r_weight in r_neighbors:
                node = (i * n_t + t) * n_p + p
                weight = t_weight * p_weight * r_weight * (alike(node) if alike is not None else 1.0)
                result = result + weight[:, None] * glow[node].to(points.dtype)
                total = total + weight
    values = (result / (total * r.clamp_min(halo.extent[0]))[:, None]).reshape(-1, n_h, n_c)
    zonal, coefficients = values[:, : n_h - len(_LOBE)], values[:, n_h - len(_LOBE) :]
    # Irradiance of the mean around the direction towards the lamp, at the cosine of each normal with it, from one down
    x = ((1 + (normals * away).sum(dim=-1).clamp(-1, 1)) / 2 * (zonal.shape[1] - 1))[:, None, None]
    low = x.floor().clamp(max=zonal.shape[1] - 2)
    index = low.long().expand(-1, 1, n_c)
    around = torch.lerp(zonal.gather(1, index), zonal.gather(1, index + 1), x - low)[:, 0]
    lobe = normals.new_tensor(_LOBE) * _harmonics(normals)
    share = halo_share(halo, points)
    irradiance = (around + torch.einsum("nh,nhc->nc", lobe, coefficients)).clamp_min(0)
    return share[:, None] * irradiance, share


def halo_inscatter(
    halo: LampHalo,
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    beta: torch.Tensor,
    nodes: int = 48,
    time: float = 0.0,
) -> torch.Tensor:
    """Light of a lamp scattered more than once that reaches the origin of rays, from a table of it.

    The light scattered per meter towards each ray is integrated with Gauss-Legendre quadrature over the angle at which
    the lamp sees points along the ray, as for light scattered once, see :func:`point_light_inscatter
    <visionsim.medium.optics.point_light_inscatter>`.

    Args:
        halo (LampHalo): Light of the lamp scattered more than once, see :func:`lamp_halo`.
        medium (Medium): Participating medium.
        origin (torch.Tensor): Ray origin, of shape (3,), or one per ray, of shape (r, 3).
        directions (torch.Tensor): Unit ray directions, of shape (r, 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (r,), can be infinite.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        nodes (int, optional): Number of quadrature nodes along each ray. Defaults to 48.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        torch.Tensor: Radiance that reaches the origin of each ray, of shape (r, c).
    """
    kwargs: dict[str, Any] = {"dtype": directions.dtype, "device": directions.device}
    size = max(1, int(4e7 // (nodes * 96)))
    if len(directions) > size:
        # Each node is looked up at 32 corners of the table, so rays are processed in chunks
        return torch.cat(
            [
                halo_inscatter(
                    halo,
                    medium,
                    origin[i : i + size] if origin.ndim > 1 else origin,
                    directions[i : i + size],
                    distance[i : i + size],
                    beta,
                    nodes,
                    time,
                )
                for i in range(0, len(directions), size)
            ]
        )
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
    offset = halo.center - origin
    along = (directions * offset).sum(dim=-1)
    closest = ((offset * offset).sum(dim=-1) - along**2).clamp_min(0).sqrt().clamp_min(halo.extent[0])
    start, end = torch.atan2(-along, closest), torch.atan2(distance - along, closest)
    theta = (start + end)[:, None] / 2 + (end - start)[:, None] / 2 * t
    weights = (end - start)[:, None] / 2 * w * closest[:, None] / torch.cos(theta) ** 2
    s = (along[:, None] + closest[:, None] * torch.tan(theta)).clamp_min(0)
    starts = origin[:, None, :] if origin.ndim > 1 else origin
    points = starts + s[..., None] * directions[:, None, :]
    tau = optical_depth(medium, starts, directions[:, None, :], s, time=time)[..., None] * beta
    source = _interpolate(halo, points.reshape(-1, 3), (-directions[:, None, :]).expand_as(points).reshape(-1, 3))
    # The table leaves out the density of the medium, which varies much faster with height than the light it scatters
    source = source.reshape(*s.shape, -1) * density(medium, points, time)[..., None]
    integrand = torch.exp(-tau) * source * weights[..., None]
    return torch.nan_to_num(integrand, nan=0.0, posinf=0.0).sum(dim=-2)


def lamp_halo(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    ground: float = 0.0,
    maps: ShadowMaps | None = None,
    grid: HaloGrid = HALO_GRID,
    time: float = 0.0,
    map_tolerance: float = 0.01,
    occupancy: Occupancy | None = None,
    reflection: LampReflection | None = None,
    relit_scattering: bool = True,
    scattered_reflection: bool = True,
    shadow_edges: bool = True,
) -> LampHalo:
    """Tabulate the light of a lamp scattered more than once by a medium, around the lamp, and its glow.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light, or emissive surface.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        ground (float, optional): Height of the ground, which stops light going down. Defaults to 0.0.
        maps (ShadowMaps | None, optional): Maps of lamps, through which objects cast the lamp's shadows onto the light
            it scatters once, see :func:`lamp_inscatter <visionsim.medium.render.lamp_inscatter>`. Defaults to None.
        grid (HaloGrid, optional): Resolution of the tables. Defaults to :data:`HALO_GRID`.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        map_tolerance (float, optional): Distance within which the lamp casts the shadows of a map rendered at another
            position. Defaults to 0.01.
        occupancy (Occupancy | None, optional): Voxels occupied by objects, which stop the light gathered at each
            point, see :func:`occupancy <visionsim.medium.occlusion.occupancy>`, which should be carved by the lamp's
            own map so as to leave free the medium it lights. Defaults to None, i.e. only the ground stops it.
        reflection (LampReflection | None, optional): Light of the lamp reflected by surfaces lit directly, see
            :func:`lamp_reflection <visionsim.medium.reflections.lamp_reflection>`, which is scattered more than once
            too, up to one order less than the lamp's light, once the glow lights surfaces too, if they are lit with the
            full extinction (``glow``). Defaults to None, i.e. surfaces reflect no light.
        relit_scattering (bool, optional): If true, with ``scattered_reflection`` and surfaces lit with the full
            extinction, the light that surfaces reflect of the glow of each order but the last, and those extrapolated,
            is scattered more than once too, along with the next order, rather than only that of the glow of the lamp's
            light scattered once. Defaults to True.
        scattered_reflection (bool, optional): If true, the light that surfaces reflect is scattered more than once in
            the halo. Otherwise, it is only relit by the glow, if lit with the full extinction, and kept. Defaults to
            True.
        shadow_edges (bool, optional): If true, the directions of the grid are refined around the objects that cast
            the lamp's shadows, as its map records them, see :func:`_refined_axes`, where light scattered more than
            twice is gathered at the points in and next to the lamp's shadows, see :func:`_gathering_points`, and
            gathering rays that pass right beside objects, across the voxels occupied around them over less than a
            voxel, aren't stopped, see :func:`hit_distances <visionsim.medium.occlusion.hit_distances>`, so as to
            resolve the light scattered more than once in and around the lamp's shadows. Defaults to True.

    Returns:
        LampHalo: Light of the lamp scattered more than once, with its glow, and the light reflected by surfaces lit
        directly and by the glow, if any.
    """
    # The light scattered once is computed as along camera rays, from each point of the grid, from a single emitter
    from visionsim.medium.render import lamp_inscatter

    lamp = merge_patches(lamp) if isinstance(lamp, EmissiveSurface) else lamp
    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    g, albedo = medium.anisotropy, medium.albedo
    center = torch.as_tensor(lamp.position, **kwargs)
    # The lamp's map alone is enough, and is kept with the tables
    maps = _lamp_map(maps, lamp, map_tolerance, beta.device)
    # Directions are refined around the objects that cast the lamp's shadows
    axes = _refined_axes(maps, center, ground, grid) if maps is not None and shadow_edges else None
    (r_min, r_max), (n_t, n_p) = grid.extent, grid.directions if axes is None else (len(axes[0]), len(axes[1]))
    radii = r_min * (r_max / r_min) ** (torch.arange(grid.radii, **kwargs) / (grid.radii - 1))
    directions = _lat_long(n_t, n_p, **kwargs) if axes is None else _directions(*axes)
    points = (center + radii[:, None, None] * directions[None]).reshape(-1, 3)
    # Scattering coefficient per unit of relative density, which multiplies the table at each point it is looked up at
    sigma = albedo * beta
    # Light is clamped to the smallest normal number of its precision, whose logarithm is finite, where there is none
    tiny = torch.finfo(beta.dtype).tiny

    # Points under the ground gather the light at the ground right above them, towards which points just above the ground
    # are interpolated, rather than the light that would get through the ground. Outgoing directions are measured from
    # the direction away from the lamp where the light is gathered, as where tables are looked up
    above = torch.cat([points[:, :2], points[:, 2:].clamp_min(ground)], dim=-1)
    towards = (center - above) / (center - above).norm(dim=-1, keepdim=True)
    away = -towards

    # Directions from which light is gathered, at angles from the direction towards the lamp clustered near zero
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(grid.gathered[0]))
    t, w = (t + 1) / 2, w / 2
    polar, azimuth = torch.meshgrid(
        math.pi * t**2, (torch.arange(grid.gathered[1], **kwargs) + 0.5) / grid.gathered[1] * 2 * math.pi, indexing="ij"
    )
    solid_angles = (2 * math.pi * t * w)[:, None] * torch.sin(polar) * 2 * math.pi / grid.gathered[1]
    gathered = _around(towards, polar.reshape(-1), azimuth.reshape(-1))  # (points, k, 3)
    solid_angles = solid_angles.reshape(-1)

    # Outgoing directions, at angles from the direction away from the lamp clustered near zero, and around it, in frames
    # that turn smoothly from one point to the next
    n_g, n_b = grid.angles
    frames = _spherical_frames(away)
    angles = math.pi * torch.linspace(0, 1, n_g, **kwargs) ** 2
    out_polar, out_azimuth = torch.meshgrid(
        angles, (torch.arange(n_b, **kwargs) + 0.5) / n_b * 2 * math.pi, indexing="ij"
    )

    def scatter_towards(j: slice | torch.Tensor, directions: torch.Tensor, light: torch.Tensor) -> torch.Tensor:
        """Light arriving at a chunk of points from directions of shape (n, k, 3), times the solid angle each stands
        for, of shape (n, k, c), scattered towards the outgoing directions, of shape (n, g, b, c)."""
        frame = (frames[0][j], frames[1][j])
        outgoing = _around(away[j], out_polar.reshape(-1), out_azimuth.reshape(-1), frame)  # (n, g * b, 3)
        # Light arriving from a direction travels along its opposite
        phase = henyey_greenstein(-torch.einsum("nod,nkd->nok", outgoing, directions), g)
        return sigma * torch.einsum("nok,nkc->noc", phase, light).reshape(-1, n_g, n_b, len(beta))

    # Indices of the points of the grid at which light is gathered, or None for all of them, as until light scattered
    # more than twice is gathered on a refined grid, see below
    nodes: torch.Tensor | None = None

    def scatter_again(arriving: torch.Tensor) -> torch.Tensor:
        """Light arriving at the points where light is gathered, `nodes`, from the gathered directions, of shape
        (nodes, k, c), scattered again, at every point, see `widen`."""
        scattered = torch.zeros(len(arriving), n_g, n_b, len(beta), **kwargs)
        for i in range(0, len(arriving), 128):
            j = slice(i, i + 128)
            k = j if nodes is None else nodes[j]
            scattered[j] = scatter_towards(k, gathered[k], arriving[j] * solid_angles[:, None])
        return scattered if nodes is None else widen(scattered, twice)

    rays = gathered.reshape(-1, 3)
    lengths = _to_ground(above[:, None, :], gathered, ground).reshape(-1)

    def origins(j: torch.Tensor) -> torch.Tensor:
        """Origins of the gathering rays of given indices, of shape (n, 3), which aren't kept for every ray, as they
        repeat, and would take up much memory on a refined grid."""
        return above[torch.div(j, gathered.shape[1], rounding_mode="floor")]

    # Objects stop the light gathered at each point, and points inside them gather none
    inside = torch.zeros(len(points), dtype=torch.bool, device=beta.device)
    if occupancy is not None:
        inside = occupied(occupancy, above.to(occupancy.low))
        # Occupied voxels extend up to about a voxel beyond objects, which rays that pass right beside them cross. On
        # the even grid, which spreads the light of the medium lit beside the lamp's shadows into them, stopping these
        # rays partly makes up for it, so they only go on where the grid is refined around the objects
        depth = _HIT_DEPTH * occupancy.size if axes is not None else 0.0
        lengths = hit_distances(
            occupancy, above.repeat_interleave(gathered.shape[1], dim=0), rays, lengths, depth=depth
        ).to(lengths)

    def gather(light: Callable[[torch.Tensor], torch.Tensor], size: int) -> torch.Tensor:
        """Light arriving at the points where light is gathered, `nodes`, along the gathering rays of given indices, of
        those outside of objects, of shape (nodes, k, c)."""
        k = gathered.shape[1]
        if nodes is None:
            rows = into = (~inside).repeat_interleave(k).nonzero()[:, 0]
            count = len(points)
        else:
            outer, offsets = (~inside[nodes]).nonzero()[:, 0], torch.arange(k, device=beta.device)
            rows = (nodes[outer][:, None] * k + offsets).reshape(-1)
            into, count = (outer[:, None] * k + offsets).reshape(-1), len(nodes)
        arriving = torch.zeros(count * k, len(beta), **kwargs)
        for i in range(0, len(rows), size):
            arriving[into[i : i + size]] = light(rows[i : i + size])
        return arriving.reshape(count, k, len(beta))

    first = gather(
        lambda j: (
            albedo
            * lamp_inscatter(
                medium,
                origins(j),
                rays[j],
                lengths[j],
                beta,
                lamp,
                time,
                area_levels=(),
                maps=maps,
                map_tolerance=map_tolerance,
                visibility_samples=_SHADOW_SAMPLES,
            )
        ),
        50000,
    )
    orders = [scatter_again(first)]
    outside = (~inside).reshape(grid.radii, n_t, n_p) if bool(inside.any()) else None
    # Where the lamp shines on the points, from its map
    lit = None
    if maps is not None:
        lit = lamp_visibility(maps, 0, above.to(maps.depths.dtype)).to(beta.dtype).reshape(grid.radii, n_t, n_p)

    # Irradiance of each ring of gathering directions around the direction towards the lamp, of even radiance, on
    # surfaces whose normals make each tabulated cosine with that direction, from one down to minus one
    n_rings, n_around = grid.gathered
    cosines = 1 - 2 * torch.arange(_ZONAL, **kwargs) / (_ZONAL - 1)
    ring_solid = solid_angles.reshape(n_rings, n_around).sum(dim=1)
    ring_irradiance = _clamped_cosine(cosines[:, None], torch.cos(math.pi * t**2)[None, :]) * ring_solid / (2 * math.pi)

    def project(arriving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Glow of the light arriving at the points where light is gathered, `nodes`, from the gathered directions, of
        shape (nodes, k, c), at every point, see `widen`: irradiance of its mean over each ring around the direction
        towards the lamp at the tabulated cosines, followed by the coefficients of the rest on the spherical harmonics,
        of shape (points, z + 9, c), and the light arriving from every direction, of shape (points, c)."""
        mean = arriving.reshape(len(arriving), n_rings, n_around, -1).mean(dim=2)
        rest = (arriving.reshape(len(arriving), n_rings, n_around, -1) - mean[:, :, None]).reshape(arriving.shape)
        coefficients = torch.zeros(len(arriving), len(_LOBE), len(beta), **kwargs)
        for i in range(0, len(arriving), 128):
            j = slice(i, i + 128)
            basis = _harmonics(gathered[j if nodes is None else nodes[j]]) * solid_angles[:, None]
            coefficients[j] = torch.einsum("nkh,nkc->nhc", basis, rest[j])
        around = torch.einsum("zr,prc->pzc", ring_irradiance, mean)
        values, brought = torch.cat([around, coefficients], dim=1), torch.einsum("k,pkc->pc", solid_angles, arriving)
        if nodes is None:
            return values, brought
        return widen(values, flux[:, None]), widen(brought, flux)

    def glowing(values: torch.Tensor) -> LampHalo:
        """Halo of the glow alone, whose values at the points, of shape (points, z + 9, c), are tabulated times the
        distance to the lamp, and filled in from their neighbors inside objects, as tables are, in single precision,
        which is ample for light that is interpolated anyway."""
        glow = values.reshape(grid.radii, n_t, n_p, -1, len(beta)) * radii[:, None, None, None, None]
        empty = torch.zeros(0, **kwargs)
        lookup = outside.to(beta.dtype) if outside is not None else None
        return LampHalo(
            center,
            grid.extent,
            ground,
            empty,
            empty,
            lookup,
            lit,
            maps,
            (_fill(glow, outside) if outside is not None else glow).float(),
            axes=axes,
        )

    def reflected_once(reflection: LampReflection) -> torch.Tensor:
        """Light of the lamp reflected by surfaces that the medium scatters once at the points outside of objects,
        towards the outgoing directions, of shape (points, g, b, c), see :func:`arriving_light
        <visionsim.medium.reflections.arriving_light>`."""
        # The gathering directions split the sphere into rings around the direction towards the lamp, whose polar
        # angles π t² are split where the cumulative weights of the quadrature over t are, and each direction stands
        # for the solid angle of its ring's
        rings = solid_angles.reshape(grid.gathered)[:, 0]
        edges = torch.cumsum(w, dim=0)[:-1].contiguous()
        scattered = torch.zeros(len(points), n_g, n_b, len(beta), **kwargs)
        for i in range(0, len(points), 128):
            j = slice(i, i + 128)
            axis = towards[j]

            def directions_density(directions: torch.Tensor, axis: torch.Tensor = axis) -> torch.Tensor:
                cosine = torch.einsum("nqd,nd->nq", directions, axis).clamp(-1, 1)
                return 1 / rings[torch.searchsorted(edges, (torch.acos(cosine) / math.pi).sqrt().contiguous())]

            radiance, sources, light = arriving_light(
                reflection, medium, above[j], gathered[j], directions_density, beta, time
            )
            channels = (-1, -1, len(beta))
            scattered[j] = scatter_towards(j, gathered[j], radiance.expand(*channels) * solid_angles[:, None])
            scattered[j] += scatter_towards(j, sources, light.expand(*channels))
        scattered[inside] = 0
        return scattered

    def relit_once(values: torch.Tensor) -> torch.Tensor | None:
        """Light of surfaces lit by a glow alone, whose values at the points are of shape (points, z + 9, c), see
        `project`, that the medium scatters once at the points, of shape (points, g, b, c), or None if surfaces reflect
        none of it, or aren't lit with the full extinction, to which no glow is added."""
        if reflection is None or reflection.albedos is None:
            return None
        darkened = relit_reflection(dark(reflection), medium, beta, partial(halo_glow, glowing(values)), time)
        return reflected_once(darkened) if len(darkened.nodes.areas) or len(darkened.patches.areas) else None

    # Surfaces are lit by the light of the lamp scattered once on its way to them, whose light is then scattered more
    # than once along with the lamp's, from the third order on, while the light that camera rays gather once scattered,
    # see `reflection_inscatter`, is relit by the whole glow at the end
    glow, flux = project(first)
    # The light arriving along each gathering ray takes up much memory on a refined grid, and isn't needed anymore
    del first
    reflected = None
    if reflection is not None and scattered_reflection and grid.orders >= 3:
        once = relit_reflection(reflection, medium, beta, partial(halo_glow, glowing(glow)), time)
        if len(once.nodes.areas) or len(once.patches.areas):
            reflected = reflected_once(once)

    def table(values: torch.Tensor) -> LampHalo:
        log = torch.log(values.clamp_min(tiny)).reshape(grid.radii, n_t, n_p, n_g * n_b, len(beta))
        if outside is not None:
            log = _fill(log, outside)
        # Relative to their peaks, logarithms near the peaks, which make most of the light, are small, and so precise
        # in half precision
        peaks = log.amax(dim=3)
        relative = log - peaks[:, :, :, None]
        if bool(((relative - relative[..., :1]).abs() <= _SHARED_CHANNELS).all()):
            relative = relative[..., :1]
        relative = relative.reshape(grid.radii, n_t, n_p, n_g, n_b, -1).to(torch.float16)
        return LampHalo(
            center,
            grid.extent,
            ground,
            relative,
            peaks,
            outside.to(log.dtype) if outside is not None else None,
            lit,
            maps,
            axes=axes,
        )

    def next_order(previous: LampHalo, j: torch.Tensor) -> torch.Tensor:
        # Light scattered more than once varies smoothly along the gathering rays, which few nodes integrate
        return halo_inscatter(previous, medium, origins(j), rays[j], lengths[j], beta, nodes=16, time=time)

    # Light scattered more than twice is gathered from the light scattered twice, and from the light of surfaces
    # scattered once, so that the lamp's shadows shape both alike. On a refined grid, it is only gathered at some of
    # its points, see `_gathering_points`: elsewhere, the ratio of each order to the light it is first gathered from,
    # and that of the light it brings to the light scattered once that arrives, which vary slowly there, are
    # interpolated from the middle points of the cells of the even grid
    if axes is not None:
        nodes, middle, (rows, columns) = _gathering_points(axes, grid.directions, lit, grid.radii)
        view = _View(
            (len(rows), len(columns)),
            outside[:, rows][:, :, columns] if outside is not None else None,
            lit[:, rows][:, :, columns] if lit is not None else None,
            (axes[0][rows], axes[1][columns]),
        )
        spread = partial(_spread, axes=axes, view=view, lit=lit)
        among = torch.zeros(len(points), dtype=torch.long, device=beta.device)
        among[nodes] = torch.arange(len(nodes), device=beta.device)
        among = among[middle]
        twice = orders[-1] + reflected if reflected is not None else orders[-1]

    def widen(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Values at the points where light is gathered, `nodes`, of shape (nodes, ...), at every point, of shape
        (points, ...): as gathered where they were, and elsewhere through their ratio to reference values at every
        point, of shape (points, ...) or broadcastable to it, interpolated from the middle points of the cells of the
        even grid."""
        whole = reference * spread(_ratio(values[among], reference[middle]))
        whole[nodes] = values
        return whole

    arrivals = [(glow, flux)]
    for order in range(3, grid.orders + 1):
        # The reflected light scattered once is gathered along with the lamp's light scattered twice, so that it is
        # scattered again along with it, up to one order less than the lamp's light
        sources = orders[-1] + reflected if reflected is not None and order == 3 else orders[-1]
        # So is the light of surfaces lit by the glow of each later order, i.e. by the light that arrived at the points
        # along with the previous order, beyond the lamp's light scattered once, whose glow lit `once`, along with the
        # next order, which the geometric extrapolation of later orders from the last two then follows in part
        relit = relit_once(arrivals[-1][0]) if relit_scattering and reflected is not None and order > 3 else None
        if relit is not None:
            sources = sources + relit
        arriving = gather(partial(next_order, table(sources)), 20000)
        arrivals.append(project(arriving))
        orders.append(scatter_again(arriving))

    # Later orders decrease about geometrically at each point, as the last two did, and so does the light arriving
    total = sum(orders[1:], orders[0])
    glow = sum((values for values, _ in arrivals[1:]), arrivals[0][0])
    if len(orders) > 1:
        ratio = (orders[-1] / orders[-2].clamp_min(tiny)).clamp(0, 0.8)
        total = total + orders[-1] * ratio / (1 - ratio)
        (last, flux), (_, before) = arrivals[-1], arrivals[-2]
        ratio = (flux / before.clamp_min(tiny)).clamp(0, 0.8)[:, None]
        glow = glow + last * ratio / (1 - ratio)
    halo = table(total)._replace(glow=glowing(glow).glow)
    if reflection is not None:
        halo = halo._replace(reflection=relit_reflection(reflection, medium, beta, partial(halo_glow, halo), time))
    return halo


def _own_map_index(maps: ShadowMaps | None, lamp: Lamp, tolerance: float) -> int | None:
    """Index of the map of a lamp among maps of lamps, or None if it has none, as :func:`lamp_inscatter
    <visionsim.medium.render.lamp_inscatter>` finds it."""
    if maps is None:
        return None
    kwargs: dict[str, Any] = {"dtype": maps.origins.dtype, "device": maps.origins.device}
    # Maps of area lights are rendered just in front of them
    center = torch.as_tensor(lamp.position, **kwargs)
    shifted = center + 1e-3 * torch.as_tensor(lamp.direction, **kwargs) if isinstance(lamp, AreaLight) else center
    return lamp_map_index(maps, shifted, tolerance)


def _lamp_map(
    maps: ShadowMaps | None, lamp: Lamp, tolerance: float, device: torch.device | str | None
) -> ShadowMaps | None:
    """Copy of the map of a lamp alone, among maps of lamps, in single precision, which is enough to look it up, and
    which doesn't keep the others in memory, or None if it has none."""
    index = _own_map_index(maps, lamp, tolerance)
    if maps is None or index is None:
        return None
    height, width = (int(size) for size in maps.shapes[index])
    first = int(maps.offsets[index])
    single = select_maps(maps, index)._replace(
        depths=maps.depths[first : first + height * width], offsets=torch.zeros_like(maps.offsets[:1])
    )
    return ShadowMaps(
        *(m.to(device=device, dtype=torch.float32 if m.is_floating_point() else m.dtype, copy=True) for m in single)
    )


def _halo_bytes(halo: LampHalo) -> int:
    """Memory taken up by the tensors of a halo, including its glow and the light reflected by surfaces, in bytes."""
    tensors = [halo.values, halo.peaks, halo.outside, halo.lit, *(halo.maps if halo.maps is not None else ()), halo.glow]
    tensors += list(halo.axes) if halo.axes is not None else []
    own = sum(t.numel() * t.element_size() for t in tensors if t is not None)
    return own + (_tensor_bytes(halo.reflection) if halo.reflection is not None else 0)


def _tensor_bytes(value: Any) -> int:
    """Memory taken up by the tensors of nested tuples, such as a :class:`LampReflection
    <visionsim.medium.reflections.LampReflection>`, in bytes."""
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    return sum(_tensor_bytes(v) for v in value) if isinstance(value, tuple) else 0


_HALOS: OrderedDict[tuple, LampHalo] = OrderedDict()
"""Halos cached by :func:`cached_lamp_halo`, from the least recently used"""


def _computed_lamp_halo(
    medium: str,
    lamp: Lamp,
    beta: tuple,
    ground: float,
    maps: ShadowMaps | None,
    grid: HaloGrid,
    time: float,
    dtype: torch.dtype,
    device: str,
    map_tolerance: float,
    occlusion: Occlusion | None,
    albedos: tuple[tuple[float, ...], tuple[float, ...]] | None,
    occluders: torch.Tensor | None,
    reflection_grid: ReflectionGrid,
    shadows: bool,
    scattered_reflection: bool,
    relit: bool,
    bounces: bool,
    specular: tuple[float, float] | None,
    surfels: Surfels | None,
    interreflections: bool,
    relit_scattering: bool,
    shadowed_reflection: bool,
    occluded_reflection: bool,
    shadow_edges: bool,
) -> LampHalo:
    tensor = torch.tensor(beta, dtype=dtype, device=device)
    medium_model = Medium.model_validate_json(medium)
    voxels = None
    # Without a map of its own, nothing carves the medium that the lamp lights out of the visual hull of the objects,
    # which may take it for the inside of an object, e.g. in a room, so that objects don't stop its halo, as they cast
    # none of its shadows either
    if occlusion is not None and _own_map_index(maps, lamp, map_tolerance) is not None:
        # Objects only matter within the extent of the halo, and above the ground
        center = torch.as_tensor(lamp.position, dtype=torch.float32, device=device)
        reach = torch.full_like(center, grid.extent[1])
        low = torch.cat([(center - reach)[:2], (center - reach)[2:].clamp_min(ground)])
        voxels = occupancy(occlusion_to(occlusion, device=device, dtype=torch.float32), low, center + reach)
    # The light of surfaces lit directly is only needed to scatter it more than once and to relight it, which the halo
    # keeps, so it isn't cached on its own
    reflection = None
    if albedos is not None and medium_model.albedo > 0 and max((*albedos[0], *albedos[1]), default=0.0) > 0:
        reflection = lamp_reflection(
            medium_model,
            lamp,
            tensor,
            ground,
            *albedos,
            maps,
            occluders,
            reflection_grid,
            time,
            map_tolerance,
            glow=relit,
            bounces=bounces,
            shadowed=shadowed_reflection,
            occluded=occluded_reflection,
            specular=specular,
            surfels=surfels,
            interreflections=interreflections,
        )
    # Surfaces keep the lamp's shadows, which the lamp's map tells, whether the medium has them or not
    shadowing = maps if shadows else None
    return lamp_halo(
        medium_model,
        lamp,
        tensor,
        ground,
        shadowing,
        grid,
        time,
        map_tolerance,
        voxels,
        reflection,
        relit_scattering,
        scattered_reflection,
        shadow_edges,
    )


def cached_lamp_halo(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    ground: float = 0.0,
    maps: ShadowMaps | None = None,
    grid: HaloGrid = HALO_GRID,
    time: float = 0.0,
    *,
    map_tolerance: float = 0.01,
    occlusion: Occlusion | None = None,
    albedos: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    occluders: torch.Tensor | None = None,
    reflection_grid: ReflectionGrid = REFLECTION_GRID,
    shadows: bool = True,
    scattered_reflection: bool = True,
    relit: bool = True,
    bounces: bool = True,
    specular: tuple[float, float] | None = None,
    surfels: Surfels | None = None,
    interreflections: bool = True,
    relit_scattering: bool = True,
    shadowed_reflection: bool = True,
    occluded_reflection: bool = True,
    shadow_edges: bool = True,
) -> LampHalo:
    """Same as :func:`lamp_halo`, cached for the last few lamps and media, up to :data:`HALO_CACHE_BYTES`, as it doesn't
    depend on the camera and can be reused for every frame, unless the lamp or the medium moves. Objects stop the light
    gathered at each point when the shadow maps of the scene are given as ``occlusion`` and the lamp has a map of its
    own among ``maps``, through the voxels they occupy within the extent of the halo, see :func:`occupancy
    <visionsim.medium.occlusion.occupancy>`, which takes a few seconds more. Given the albedo of the ground and of
    objects, ``albedos``, surfaces reflect the lamp's light, lit directly with the full extinction, see
    :func:`lamp_reflection <visionsim.medium.reflections.lamp_reflection>`, which is scattered more than once in the
    halo, and relit by its glow, which the halo keeps, see :attr:`LampHalo.reflection`, so that it is computed along
    with the halo, and cached with it, about 2 s more by default on 4 CPU threads, or 15 s with the patches of every
    surface (``surfels``) and the specular layer of surfaces (``specular``), for the lamp of the night scene of
    ``examples/medium``, mostly to scatter their light in the halo. Maps, ``occluders``, the points of the surfaces that
    hide the light of the ground (see :func:`scene_surfaces <visionsim.medium.reflections.scene_surfaces>`), and
    ``surfels``, the pieces of every surface (see :func:`scene_surfels <visionsim.medium.reflections.scene_surfels>`),
    are keyed by identity, so the same ones should be given for every frame, rather than copies of them. Terms of the
    model can be left out (see :class:`Effects <visionsim.medium.model.Effects>`): the specular layer of surfaces,
    without ``specular``, its roughness and index of refraction; every surface receiving the light of the ground and of
    objects, and the glow, without ``surfels``, as only those that the lamp lights do then; light bouncing between
    surfaces more than once, without ``interreflections``, see :func:`lamp_reflection
    <visionsim.medium.reflections.lamp_reflection>`; the light of surfaces lit by the glow of later orders scattered
    more than once, without ``relit_scattering``, see :func:`lamp_halo`; the lamp's shadows on the medium, without
    ``shadows``, as if it had no map, but for the light that surfaces reflect, which keep them; the light that surfaces
    reflect scattered more than once, without ``scattered_reflection``; the glow lighting these surfaces, without
    ``relit``, which are then lit with the reduced extinction of the similarity relation; the light that objects and
    the ground reflect onto each other, without ``bounces``; the lamp's shadows on the ground that reflects its light,
    without ``shadowed_reflection``; objects hiding the light of the ground from the medium behind them, without
    ``occluded_reflection``, see :func:`lamp_reflection <visionsim.medium.reflections.lamp_reflection>`; and the grid
    refined around the objects that cast the lamp's shadows, and rays that pass right beside them, without
    ``shadow_edges``, see :func:`lamp_halo`, which takes 3 to 4 times as long, and 3 times as much memory, with them,
    for the lamp of the night scene of ``examples/medium``."""
    moving = any(isinstance(c, Blob) and any(c.velocity) for c in medium.components)
    key = (
        medium.model_dump_json(),
        lamp,
        tuple(beta.tolist()),
        ground,
        maps,
        grid,
        time if moving else 0.0,
        beta.dtype,
        str(beta.device),
        map_tolerance,
        occlusion,
        (tuple(albedos[0]), tuple(albedos[1])) if albedos is not None else None,
        occluders,
        reflection_grid,
        shadows,
        scattered_reflection,
        relit,
        bounces,
        (float(specular[0]), float(specular[1])) if specular is not None else None,
        surfels,
        interreflections,
        relit_scattering,
        shadowed_reflection,
        occluded_reflection,
        shadow_edges,
    )
    halo = _HALOS.pop(key, None)
    if halo is None:
        halo = _computed_lamp_halo(*key)
    # The least recently used halos are dropped first, but never the one just used
    _HALOS[key] = halo
    while len(_HALOS) > 1 and sum(_halo_bytes(h) for h in _HALOS.values()) > HALO_CACHE_BYTES:
        _HALOS.popitem(last=False)
    return halo


def halo_threshold(medium: Medium, lamp: Lamp, beta: torch.Tensor, extent: float = HALO_GRID.extent[1]) -> bool:
    """Whether the light of a lamp scattered more than once is worth computing, i.e. whether the medium around it has
    an optical depth of at least 1% over the extent of the halo, as the halo grows with it.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light, or emissive surface.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        extent (float, optional): Extent of the halo, in meters. Defaults to that of :class:`HaloGrid`.

    Returns:
        bool: Whether the halo is worth computing.
    """
    at_lamp = density(medium, torch.as_tensor(lamp.position, dtype=beta.dtype, device=beta.device)[None])[0]
    return bool(medium.albedo * at_lamp * beta.max() * extent >= 0.01)
