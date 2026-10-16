"""Shadows that objects cast onto a participating medium, such as light shafts and the sky hidden by nearby objects,
and the light they reflect into it.

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
elevations.

Light scattered more than once reaches the medium from all directions too, and is occluded in the same way, with cells
weighted by the radiance arriving from their elevations, above and below the horizon, which is brighter from the cells
towards suns, as the medium scatters sunlight forward: each cell is weighted by the mean radiance of its band times its
profile, see :func:`cell_profile`. A cell hidden by an object brings none of it, although the medium in front of the
object glows too, which assumes that this medium, lying in the shadows of the object, of suns and of the sky, glows less
than in the open by about as much. This is an empirical finding in the day scene of ``examples/medium``, whose sun is
30° high: light scattered twice, integrated by brute force along every direction from points along camera rays up to
the first object, and lit at every point by suns, the sky and the ground in the shadows of objects, is dimmed by objects
as by hidden cells to within 2% where rays see the sky or the lit ground, while counting the glow in front of objects
as in the open makes it 10 to 18% brighter. It doesn't hold for every object: the medium in front of an object that the
sun lights from behind the point is lit too, and its glow is missing. Light scattered more than twice comes from further
away, through the medium in the shadows of objects too, and objects dim it more than hidden cells do: against Cycles with
a Lambertian ground and objects of the same albedo, it is 0.518 times the light without objects over whole frames,
against 0.499 times in Cycles, 0.782 times where rays see the sky, against 0.748, and 0.664 where they see the ground in
the shadows of objects, against 0.585. On a grid of voxels around the objects of that scene, dimming it along every
direction from each point by the hidden cells of the points it passes doesn't explain it either: it then dims it as
hidden cells do with the glow of the medium in front of objects, and 5% too much without it.

Light from below the horizon, i.e. from the ground and the medium beneath, is looked up in the maps of the cells half a
turn around, which look down along the directions of cells mirrored below the horizon, see :class:`Shadows`. The light
that the ground reflects is weighted by its mean attenuation over the elevations of each cell, and where it starts on
the ground in the shadow of a sun, the ground lacks the sun's light, directly and through the glow of its light
scattered once, as the medium around it is in the shadow too, see :func:`ground_shares
<visionsim.medium.surfaces.ground_shares>`, and so does the light of the ground that the medium scatters on its way up.
In the day scene, Cycles' ground in shadows is 0.35 times as bright as in the sun, against 0.41 times here. Where it
starts on the ground next to objects, which hide part of the sky from the ground, the ground lacks that part of the rest
of its light, which the sky and the glow of the medium bring, given the share of the sky that the ground sees on a grid
over the ground around objects, see :func:`ground_sky`, and so does the light of the ground that the medium scatters on
its way up. Near objects, most of the medium's light other than single scattering involves the ground: in the day
scene, with black objects and a Lambertian ground, 70% of the sky's where rays pass next to objects, which is then 2%
brighter than Cycles', against 12% with the whole sky, and 0.7% over whole frames, against 5.4%.

Objects also reflect the light of suns and of the sky into the medium, which scatters it towards the camera: from each
cell an object hides, a point receives the light of the object's surface where the cell's direction enters it, as lit by
:func:`object_radiance <visionsim.medium.surfaces.object_radiance>`, attenuated on its way, see
:func:`object_reflections`, with multiple scattering by the extinction reduced by the similarity relation, as the light
that the medium scatters forward on its way keeps going towards the point. Scattered once, this light is 0.90 times the
light that the same surfaces reflect into the fog through the exact geometry of the day scene, integrated by brute force
over every direction, as cells are hidden or seen as a whole. Against Cycles with a Lambertian ground and objects of the
same grey albedo, in which the light that objects reflect is 7% of the fog's light, of which the light scattered more
than twice is 0.6 times the light scattered at most twice, about evenly over the frame, it is 0.70 times all of Cycles'
when scattered once only (1.12 times Cycles' scattered at most twice), and 0.87 times with the similarity relation, with
per-pixel differences of 18% of this light once blurred, against 31%. The rest is light scattered sideways on its way,
which then reaches points from other directions, and light that lights other surfaces first, such as the ground next to
sunlit walls, 8% brighter within 1 m of them, neither of which is modeled. The objects of that scene are colored, while
they have the ground's grey albedo here, as the lighting gives them none.

Lamps cast shadows through equirectangular maps of the distance between each lamp and the first surface in every
direction, see :func:`lamp_visibility`, whose visibility is sampled along rays at angles evenly spread as seen from the
lamp and averaged around each node of the quadrature that integrates the lamp's light, see :func:`lamp_shadow`. Lamps
that move have maps at several of their positions, and cast the shadows of the nearest one.

Light that travels through the medium in any direction, such as the light of lamps scattered more than once, is stopped
by objects wherever they are, which the maps also tell: a point lies inside an object if every map hides it behind the
first surface it recorded, i.e. if it lies within the visual hull of the objects as seen by the maps. This is baked
into a grid of voxels, along with the distance from each voxel to the nearest occupied one, which rays skip through to
find the first object they hit, see :func:`occupancy` and :func:`hit_distances`.
"""

from __future__ import annotations

import math
import os
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from typing import Any, NamedTuple, TypeVar

import numpy as np
import torch

from visionsim.medium.model import HeightFog, Medium
from visionsim.medium.optics import (
    height_fog_optical_depth,
    height_fog_sun_inscatter,
    henyey_greenstein,
    similarity_extinction,
)
from visionsim.medium.scattering import ElevationTable, ScatteringTable, lookup, relative_azimuths

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
    surface in every direction, see :func:`lamp_visibility`, and whose texels are in radians. Lamps that move have maps
    at several of their positions"""
    emissive_maps: ShadowMaps | None = None
    """maps of emissive surfaces, as those of lamps, which see through emissive surfaces as they don't hide their own
    light"""
    lamp_tolerance: float = 0.01
    """distance, in meters, within which a lamp casts the shadows of a map rendered at another position, as moving lamps
    get a new map once they move further than a given spacing from their previous ones"""


def select_maps(maps: ShadowMaps, index: int) -> ShadowMaps:
    """Shadow maps reduced to a single one of them, which still refers to all texels."""
    return maps._replace(**{name: getattr(maps, name)[index : index + 1] for name in maps._fields[1:]})


COPY_CACHE_SIZE = 32
"""Number of results derived from the shadow maps of scenes that are kept, such as their copies on another device or in
another precision by :func:`maps_to` and :func:`occlusion_to`, so that they are computed once rather than for every
frame"""
_KEPT: OrderedDict[tuple, tuple[tuple[weakref.ref, ...], tuple[int, ...], Any]] = OrderedDict()
"""Results kept by :func:`_kept`, from the least recently used, with weak references to the tensors they were derived
from and their versions"""
_DERIVED: OrderedDict[tuple, tuple[tuple[weakref.ref, ...], tuple[int, ...], Any]] = OrderedDict()
"""Small results kept by :func:`_kept` apart from copies of maps, so that they don't push them out"""
_DERIVED_SIZE = 16
"""Number of small results kept in :data:`_DERIVED`"""
_Kept = TypeVar("_Kept")


def _kept(
    tensors: Sequence[torch.Tensor],
    key: tuple,
    compute: Callable[[], _Kept],
    store: OrderedDict | None = None,
    size: int | None = None,
) -> _Kept:
    """Result of ``compute``, which only depends on ``tensors`` and on ``key``, kept for as long as the tensors live and
    aren't modified in place, so that the same result isn't computed again, e.g. for every frame.

    Results are looked up by the identity of the tensors, and dropped once any of them is garbage collected, or beyond
    the ``size`` most recently used of a ``store``, by default the :data:`COPY_CACHE_SIZE` results derived from maps.
    Changes in place are told by the version that PyTorch counts for each tensor, which misses those made around its
    operations, through ``.data`` or the memory of a NumPy array shared with the tensor. Inference tensors, made under
    :func:`torch.inference_mode`, don't count their versions, so that nothing derived from them is kept."""
    if any(t.is_inference() for t in tensors):
        return compute()
    kept = _KEPT if store is None else store
    full = (tuple(id(t) for t in tensors), *key)
    versions = tuple(t._version for t in tensors)
    entry = kept.get(full)
    if entry is not None and versions == entry[1] and all(r() is t for r, t in zip(entry[0], tensors)):
        kept.move_to_end(full)
        return entry[2]

    def forget(_: weakref.ref, full: tuple = full) -> None:
        kept.pop(full, None)

    if torch.is_inference_mode_enabled():
        # Results made under inference mode would be inference tensors, from which nothing could be kept in turn, e.g.
        # the depth windows of maps copied to a GPU, while those made outside of it can be used under it as well
        with torch.inference_mode(False), torch.no_grad():
            result = compute()
    else:
        result = compute()
    kept[full] = (tuple(weakref.ref(t, forget) for t in tensors), versions, result)
    kept.move_to_end(full)
    while len(kept) > (COPY_CACHE_SIZE if size is None else size):
        kept.popitem(last=False)
    return result


def _moves(tensor: torch.Tensor, device: torch.device | str | None, dtype: torch.dtype) -> bool:
    """Whether moving a tensor to a device (None for its own) and casting it, if it is a floating point tensor, to a
    precision makes a copy of it."""
    if tensor.is_floating_point() and tensor.dtype != dtype:
        return True
    if device is None:
        return False
    target = torch.device(device)
    return tensor.device.type != target.type or (target.index is not None and tensor.device.index != target.index)


def _cached_copy(
    tensors: Sequence[torch.Tensor],
    device: torch.device | str | None,
    dtype: torch.dtype,
    copy: Callable[[], _Kept],
    key: tuple = (),
) -> _Kept:
    """Result of ``copy``, which moves ``tensors`` to a device and precision, and only depends on them and on ``key``,
    kept by :func:`_kept`, unless the tensors are already there, which isn't a copy."""
    if not any(_moves(t, device, dtype) for t in tensors):
        return copy()
    return _kept(tensors, ("copy", str(device), dtype, *key), copy)


def maps_to(
    maps: ShadowMaps, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> ShadowMaps:
    """Move shadow maps to a device, and cast their floating point values to a given precision.

    The copy is kept, and returned again for the same maps, as long as they live and aren't modified in place, see
    :data:`COPY_CACHE_SIZE`, so that maps used for every frame are only copied once: it should be treated as read-only.
    Only changes made through PyTorch's operations are told, so that maps changed otherwise, e.g. through a NumPy array
    that shares their memory, should be given as new tensors.

    Args:
        maps (ShadowMaps): Shadow maps.
        device (torch.device | str | None, optional): Device to move them to. Defaults to None, i.e. their own.
        dtype (torch.dtype, optional): Precision of their floating point values. Defaults to torch.float32.

    Returns:
        ShadowMaps: Shadow maps on the device, in that precision, which are the given ones if they already were.
    """

    def copy() -> ShadowMaps:
        return ShadowMaps(*(m.to(device=device, dtype=dtype if m.is_floating_point() else m.dtype) for m in maps))

    return _cached_copy(maps, device, dtype, copy)


def occlusion_to(
    occlusion: Occlusion, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Occlusion:
    """Move the shadow maps of a scene to a device, and cast their floating point values to a given precision.

    The copy is kept, and returned again for the same maps, as long as they live and aren't modified in place, see
    :data:`COPY_CACHE_SIZE`, so that maps used for every frame are only copied once: it should be treated as read-only.
    Only changes made through PyTorch's operations are told, so that maps changed otherwise, e.g. through a NumPy array
    that shares their memory, should be given as new tensors.

    Args:
        occlusion (Occlusion): Shadow maps of a scene.
        device (torch.device | str | None, optional): Device to move them to. Defaults to None, i.e. their own.
        dtype (torch.dtype, optional): Precision of their floating point values. Defaults to torch.float32.

    Returns:
        Occlusion: Shadow maps on the device, in that precision.
    """
    move = partial(maps_to, device=device, dtype=dtype)

    def copy() -> Occlusion:
        return occlusion._replace(
            sun_maps=move(occlusion.sun_maps),
            sun_directions=occlusion.sun_directions.to(device=device, dtype=dtype),
            sky_maps=move(occlusion.sky_maps),
            bounds=occlusion.bounds.to(device=device, dtype=dtype),
            lamp_maps=move(occlusion.lamp_maps) if occlusion.lamp_maps is not None else None,
            emissive_maps=move(occlusion.emissive_maps) if occlusion.emissive_maps is not None else None,
        )

    # Copies are kept for the tensors moved and the fields that hold them, as the same maps may be those of lamps or of
    # emissive surfaces, while the other fields, in which occlusions that share maps may differ, are the given ones
    fields = occlusion._asdict()
    moved = tuple(name for name, value in fields.items() if torch.is_tensor(value) or isinstance(value, ShadowMaps))
    tensors = [t for name in moved for t in (fields[name] if isinstance(fields[name], ShadowMaps) else [fields[name]])]
    others = {name: value for name, value in fields.items() if name not in moved}
    return _cached_copy(tensors, device, dtype, copy, key=moved)._replace(**others)


def shadows_to(
    shadows: Shadows, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Shadows:
    """Move shadows to a device, and cast their floating point values to a given precision."""

    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)

    weights, scattered, unseen = shadows.shadowed_weights_below, shadows.scattered, shadows.unseen_weights_below
    return shadows._replace(
        suns=tuple(SunShadows(*(move(t) for t in sun)) for sun in shadows.suns),
        shadowed_below=tuple(move(bits) for bits in shadows.shadowed_below),
        shadowed_weights_below=move(weights) if weights is not None else None,
        scattered=SkyWeights(*(move(t) for t in scattered)) if scattered is not None else None,
        unseen_weights_below=move(unseen) if unseen is not None else None,
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
            emissive_maps=maps("emissive") if "emissive_depths" in data else None,
            lamp_tolerance=max(float(data["lamp_spacing"]), 0.01) if "lamp_spacing" in data else 0.01,
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


class LampShadow(NamedTuple):
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

    def average(self, bounds: torch.Tensor) -> torch.Tensor:
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
        return LampShadow(*(tensor[rays] for tensor in self))


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
        origin (torch.Tensor): Ray origin, of shape (3,), or one per ray, of shape (r, 3).
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
    along = (directions * offset).sum(dim=-1)
    closest = ((offset * offset).sum(dim=-1) - along * along).clamp_min(0).sqrt().clamp_min(1e-4)
    origin = origin[:, None, :] if origin.ndim > 1 else origin
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


class Occupancy(NamedTuple):
    """Voxels occupied by the scene's objects, along with how far each voxel is from them, see :func:`occupancy`."""

    low: torch.Tensor
    """corner of the grid with the lowest coordinates, of shape (3,)"""
    size: float
    """size of the voxels, in meters"""
    clearance: torch.Tensor
    """number of voxels between each voxel and the nearest occupied one along the axes of the grid, i.e. their Chebyshev
    distance in voxels, which is 0 for occupied voxels and capped at a maximum, of shape (x, y, z) and of type uint8"""


def _min_filtered(maps: ShadowMaps, index: int, window: int, wrap: bool = False) -> torch.Tensor:
    """Least depth recorded by a map over the square window of texels of a given odd size centered on each texel, of
    shape (h, w), where windows wrap around horizontally if ``wrap``, as for equirectangular maps of lamps, and
    texels outside of the map let all light through otherwise."""
    height, width = (int(s) for s in maps.shapes[index])
    first = int(maps.offsets[index])
    image = maps.depths[first : first + height * width].reshape(1, 1, height, width)
    if window <= 1:
        return image[0, 0]
    half = window // 2
    image = torch.nn.functional.pad(image, (0, 0, half, half), value=torch.inf)
    # Along columns, then rows
    image = -torch.nn.functional.max_pool2d(-image, (window, 1), stride=1)
    if wrap:
        image = torch.cat([image[..., -half:], image, image[..., :half]], dim=-1)
    else:
        image = torch.nn.functional.pad(image, (half, half), value=torch.inf)
    return -torch.nn.functional.max_pool2d(-image, (1, window), stride=1)[0, 0]


def _hidden_in_orthographic(
    maps: ShadowMaps, index: int, image: torch.Tensor, points: torch.Tensor, margin: float
) -> torch.Tensor:
    """Whether points of shape (n, 3) lie behind the surface that a filtered orthographic map recorded around them, or
    within ``margin`` meters in front of it, of shape (n,). Points outside of the map are lit."""
    selected = select_maps(maps, index)
    ones = torch.ones_like(points[:, :1])
    x, y, depth = (c[:, 0, 0] for c in _texel_coordinates(selected, torch.zeros_like(points[0]), points, ones, 0.0))
    height, width = image.shape
    xi = torch.minimum(x.round().clamp_min(-1), torch.full_like(x, width)).long()
    yi = torch.minimum(y.round().clamp_min(-1), torch.full_like(y, height)).long()
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    return inside & (depth + margin > image[yi.clamp(0, height - 1), xi.clamp(0, width - 1)])


def _hidden_from_lamp(
    maps: ShadowMaps, index: int, image: torch.Tensor, points: torch.Tensor, margin: float
) -> torch.Tensor:
    """Whether points of shape (n, 3) lie behind the surface that a filtered map of a lamp recorded in their direction,
    or within ``margin`` meters in front of it, of shape (n,), with the mapping of :func:`lamp_visibility`."""
    offset = points - maps.origins[index]
    distance = offset.norm(dim=-1)
    height, width = image.shape
    u = (math.pi - torch.atan2(offset[:, 1], offset[:, 0])) / (2 * math.pi)
    v = 1 - torch.acos((offset[:, 2] / distance.clamp_min(1e-12)).clamp(-1, 1)) / math.pi
    column = torch.remainder((u * width).floor(), width).long()
    row = (v * height).floor().clamp(0, height - 1).long()
    return distance + margin > image[row, column]


def occupancy(
    occlusion: Occlusion,
    low: torch.Tensor,
    high: torch.Tensor,
    size: float | None = None,
    max_voxels: int = 2**24,
    max_clearance: int = 32,
    chunk: int = 2**21,
) -> Occupancy | None:
    """Voxels occupied by the scene's objects within a box, i.e. the visual hull of the objects seen by its shadow maps.

    A point lies inside an object if every map hides it behind the first surface it recorded: the maps of the sky's
    cells and of suns look at the objects from above along many directions, and the maps of lamps and of emissive
    surfaces from where they are, which also carves the places under objects that they light, such as a room lit by a
    lamp or a glowing panel. The visual hull contains the objects, and also the places hidden from the whole sky and
    from every lamp and emissive surface, which are dark, such as a closed room without lights, and are taken for the
    inside of objects. As the maps of emissive surfaces see through them, emissive meshes are left out of the hull,
    unless other maps hide them. Voxels are occupied when any part of them may be hidden in every map: each map is
    compared with the least depth it recorded around the voxel, over the texels it covers in orthographic maps and the
    nearest 3 by 3 texels in maps of lamps, up to the voxel's half diagonal in front of it, so that thin objects such as
    walls fill the voxels they cross.

    Args:
        occlusion (Occlusion): Shadow maps of the scene.
        low (torch.Tensor): Corner of the box with the lowest coordinates, of shape (3,), which is clipped to the box
            that contains every object.
        high (torch.Tensor): Corner of the box with the highest coordinates, of shape (3,).
        size (float | None, optional): Size of the voxels, in meters. Defaults to the size of the finest texels of the
            maps of the sky and of suns.
        max_voxels (int, optional): Maximum number of voxels, beyond which they get larger. Defaults to 2**24.
        max_clearance (int, optional): Number of voxels up to which the distance to the nearest occupied voxel is
            counted, which bounds the steps of :func:`hit_distances`. Defaults to 32.
        chunk (int, optional): Number of voxels whose occupancy is looked up at once. Defaults to 2**21.

    Returns:
        Occupancy | None: Occupied voxels, or None if the box holds no objects or there are no maps of the sky, without
        which objects are only seen from a few directions.
    """
    sky, suns = occlusion.sky_maps, occlusion.sun_maps
    lamps = [maps for maps in (occlusion.lamp_maps, occlusion.emissive_maps) if maps is not None]
    if not len(sky.texels):
        return None
    kwargs = {"dtype": sky.depths.dtype, "device": sky.depths.device}
    low = torch.maximum(low.to(**kwargs), occlusion.bounds[0].to(**kwargs))
    high = torch.minimum(high.to(**kwargs), occlusion.bounds[1].to(**kwargs))
    extent = (high - low).clamp_min(0)
    if not bool((extent > 0).all()):
        return None
    texel = float(torch.cat([sky.texels, suns.texels]).min())
    size = max(size or texel, float(extent.prod() / max_voxels) ** (1 / 3))
    dims = torch.ceil(extent / size).long().clamp_min(1)
    # Any part of a voxel lies within its half diagonal of its center
    margin = size * math.sqrt(3) / 2

    # Lamps and emissive surfaces first, as they carve the most from the medium they light, then the sky's cells from
    # the zenith down
    tests: list[Callable[[torch.Tensor], torch.Tensor]] = []
    for maps in lamps:
        for k in range(len(maps.texels)):
            image = _min_filtered(maps, k, 3, wrap=True)
            tests.append(partial(_hidden_from_lamp, maps, k, image, margin=margin))
    orthographic = [(suns, k) for k in range(len(suns.texels))] + [(sky, k) for k in range(len(sky.texels))]
    orthographic.sort(key=lambda pair: -float(pair[0].axes[pair[1], 2, 2]))
    for maps, k in orthographic:
        image = _min_filtered(maps, k, 2 * math.ceil(margin / float(maps.texels[k])) + 1)
        tests.append(partial(_hidden_in_orthographic, maps, k, image, margin=margin))

    occupied = torch.zeros(int(dims.prod()), dtype=torch.bool, device=low.device)
    for part in _chunks(len(occupied), chunk):
        candidates = torch.arange(part.start, min(part.stop, len(occupied)), device=low.device)
        for test in tests:
            if not len(candidates):
                break
            cell = torch.stack(
                [candidates // (dims[1] * dims[2]), candidates // dims[2] % dims[1], candidates % dims[2]]
            )
            centers = low + (cell.T.to(low.dtype) + 0.5) * size
            candidates = candidates[test(centers)]
        occupied[candidates] = True
    occupied = occupied.reshape(*dims.tolist())

    # Chebyshev distance to the nearest occupied voxel, by dilating them one voxel at a time
    clearance = torch.full_like(occupied, max_clearance, dtype=torch.uint8)
    reached = occupied
    for distance in range(max_clearance):
        clearance[reached & (clearance == max_clearance)] = distance
        for axis in range(3):
            grown = reached.clone()
            grown.narrow(axis, 1, reached.shape[axis] - 1).logical_or_(reached.narrow(axis, 0, reached.shape[axis] - 1))
            grown.narrow(axis, 0, reached.shape[axis] - 1).logical_or_(reached.narrow(axis, 1, reached.shape[axis] - 1))
            reached = grown
    return Occupancy(low, size, clearance)


def _cells(occupancy: Occupancy, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the voxel of points of shape (n, 3), clamped to the grid, of shape (n, 3), and whether they lie within
    the grid, of shape (n,)."""
    dims = torch.as_tensor(occupancy.clearance.shape, device=points.device)
    cell = torch.floor((points - occupancy.low.to(points)) / occupancy.size).nan_to_num(-1).clamp(-1, 2**40).long()
    within = ((cell >= 0) & (cell < dims)).all(dim=-1)
    return torch.minimum(cell.clamp_min(0), dims - 1), within


def occupied(occupancy: Occupancy, points: torch.Tensor) -> torch.Tensor:
    """Whether points lie inside objects, i.e. in an occupied voxel, see :func:`occupancy`.

    Args:
        occupancy (Occupancy): Occupied voxels.
        points (torch.Tensor): Points, of shape (n, 3).

    Returns:
        torch.Tensor: Whether each point lies in an occupied voxel, of shape (n,).
    """
    cell, within = _cells(occupancy, points)
    return within & (occupancy.clearance[cell[:, 0], cell[:, 1], cell[:, 2]] == 0)


def hit_distances(
    occupancy: Occupancy,
    origins: torch.Tensor,
    directions: torch.Tensor,
    lengths: torch.Tensor,
    max_steps: int = 100000,
    depth: float = 0.0,
    chunk: int = 2**22,
) -> torch.Tensor:
    """Distance along rays to the first occupied voxel, see :func:`occupancy`, or to the first run of occupied voxels
    that they go through over a given depth.

    Rays skip through the grid by the distance to the nearest occupied voxel, along which they can't meet any, and by at
    least one voxel at a time next to occupied ones, so that they don't miss any voxel they cross. As voxels are
    occupied when any part of them may lie inside an object, occupied voxels extend beyond objects by up to about a
    voxel, which rays that pass right beside an object, such as the light that reaches the medium right behind it from
    its sides, may cross over a short distance. Given a ``depth``, rays only stop at a run of consecutive occupied
    voxels that they go through over at least that distance, as rays that hit an object do, and they stop where they
    entered it, in front of the object, rather than inside of it. Rays that go straight through a single layer of
    voxels, the thinnest that objects fill, go through exactly one voxel, so that a ``depth`` of a whole number of
    voxels would only stop them depending on rounding: it should be a little less.

    Args:
        occupancy (Occupancy): Occupied voxels.
        origins (torch.Tensor): Origin of the rays, of shape (n, 3).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        lengths (torch.Tensor): Length of the rays, of shape (n,), can be infinite.
        max_steps (int, optional): Maximum number of steps along each ray. Defaults to 100000.
        depth (float, optional): Distance, in meters, over which rays go through consecutive occupied voxels before
            they stop where they entered them. Defaults to 0.0, i.e. rays stop at the first occupied voxel.
        chunk (int, optional): Number of rays traced at once, which bounds the memory taken up, about 200 bytes per
            ray, as rays are independent. Defaults to 2**22.

    Returns:
        torch.Tensor: Distance to where rays enter the first occupied voxel, or the first run of occupied voxels that
        they go through over ``depth``, which is 0 for rays that start in it, or the length of rays that don't stop, of
        shape (n,).
    """
    if len(origins) > chunk:
        return torch.cat(
            [
                hit_distances(occupancy, origins[part], directions[part], lengths[part], max_steps, depth, chunk)
                for part in _chunks(len(origins), chunk)
            ]
        )
    dtype = occupancy.low.dtype
    low, size = occupancy.low, occupancy.size
    origins, directions, lengths = origins.to(dtype), directions.to(dtype), lengths.to(dtype)
    high = low + size * torch.as_tensor(occupancy.clearance.shape, dtype=dtype, device=low.device)
    near, far = _ray_box(origins, directions, low, high)
    far = torch.minimum(far, lengths)
    result = lengths.clone()
    active = (near < far).nonzero()[:, 0]
    t = near[active]
    # Where each ray entered the run of occupied voxels it is in, and how far it has gone through it
    entry, run = t.clone(), torch.zeros_like(t)
    for _ in range(max_steps):
        if not len(active):
            break
        points = torch.addcmul(origins[active], t[:, None], directions[active])
        cell, _ = _cells(occupancy, points)
        clearance = occupancy.clearance[cell[:, 0], cell[:, 1], cell[:, 2]].to(dtype)
        occupied = clearance == 0
        # Within the clearance of a voxel, minus one for where the point lies in it, there is no occupied voxel, and
        # stepping out of the voxel only reaches its neighbors
        ray = directions[active]
        bounds = low + (cell + (ray > 0).long()).to(dtype) * size
        exits = torch.where(ray != 0, (bounds - points) / torch.where(ray != 0, ray, 1.0), torch.inf).amin(dim=-1)
        step = torch.maximum((clearance - 1) * size, exits.clamp_min(0) + 1e-3 * size)
        entry = torch.where(occupied & (run == 0), t, entry)
        # Rays stop at a run once they have gone through it over the depth, before their end
        hit = occupied & (run + step >= depth) & (t + (depth - run).clamp_min(0) < far[active])
        run = torch.where(occupied, run + step, torch.zeros_like(run))
        result[active[hit]] = entry[hit]
        t = t + step
        keep = ~hit & (t < far[active])
        active, t, entry, run = active[keep], t[keep], entry[keep], run[keep]
    return result


_NOTHING = 1e9
"""Depth beyond which shadow maps recorded no surface"""


def _signed_depths(maps: ShadowMaps, points: torch.Tensor, bias: float) -> torch.Tensor:
    """Distance from points along the direction of each shadow map, away from its light, to the surface recorded by its
    nearest texel, which is negative where that surface lies between the points and the light, and infinite where the
    map recorded no surface, of shape (u, k)."""
    ones = torch.ones_like(points[:, :1])
    x, y, depth = (c[:, 0] for c in _texel_coordinates(maps, torch.zeros_like(points[0]), points, ones, bias))
    width, height, offsets = maps.shapes[:, 1].int(), maps.shapes[:, 0].int(), maps.offsets.int()
    xi = torch.minimum(x.round().clamp_min(-2), width.to(x.dtype)).int()
    yi = torch.minimum(y.round().clamp_min(-2), height.to(x.dtype)).int()
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    recorded = maps.depths[torch.where(inside, offsets + yi * width + xi, offsets)]
    return torch.where(inside & (recorded < _NOTHING), recorded - depth, torch.full_like(depth, torch.inf))


def _blocker_distances(maps: ShadowMaps, points: torch.Tensor, bias: float) -> torch.Tensor:
    """Distance from points towards the light of each shadow map to the surface recorded by its nearest texel, where
    that surface hides the light, or infinity otherwise, of shape (u, k)."""
    signed = _signed_depths(maps, points, bias)
    return torch.where(signed < 0, -signed, torch.full_like(signed, torch.inf))


def _surface_normals(maps: ShadowMaps, index: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Unit normal, on the side of the light, of the surface that shadow maps recorded at the nearest texel to points,
    from the slopes of the recorded depths, of shape (e, 3), given the index of the map of each point, of shape (e,).
    Across the edges of objects, the slope on the side whose depth changes least is taken."""
    texel, axes = maps.texels[index], maps.axes[index]  # (e,), (e, 3, 3)
    width, height = maps.shapes[index, 1], maps.shapes[index, 0]
    local = ((points - maps.origins[index])[:, None, :] * axes).sum(dim=-1)  # (e, 3)
    xi = (local[:, 0] / texel + width / 2 - 0.5).round().long()
    yi = (local[:, 1] / texel + height / 2 - 0.5).round().long()

    def depth(dx: int, dy: int) -> torch.Tensor:
        i, j = xi + dx, yi + dy
        inside = (i >= 0) & (i < width) & (j >= 0) & (j < height)
        recorded = maps.depths[torch.where(inside, maps.offsets[index] + j * width + i, maps.offsets[index])]
        return torch.where(inside, recorded, torch.full_like(recorded, torch.inf))

    center = depth(0, 0)

    def slope(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        back = torch.where(before < _NOTHING, center - before, torch.full_like(center, torch.inf))
        ahead = torch.where(after < _NOTHING, after - center, torch.full_like(center, torch.inf))
        least = torch.where(back.abs() < ahead.abs(), back, ahead)
        return torch.where(torch.isfinite(least), least, torch.zeros_like(least)) / texel

    # The surface is at depth D(X, Y) below the map's plane, whose normal is the light's direction plus the slopes of D
    # along the map's axes
    normals = axes[:, 2] + slope(depth(-1, 0), depth(1, 0))[:, None] * axes[:, 0]
    normals = normals + slope(depth(0, -1), depth(0, 1))[:, None] * axes[:, 1]
    return torch.nn.functional.normalize(normals, dim=-1)


def _opposite_cells(counts: Sequence[int], device: torch.device | str | None = None) -> torch.Tensor:
    """Index of the cell of the sky half a turn around each cell in the same band, whose map looks along the cell's
    direction mirrored below the horizon, or -1 in bands of an odd number of cells, of shape (k,)."""
    sizes = torch.as_tensor(counts, device=device)
    band = torch.repeat_interleave(torch.arange(len(counts), device=device), sizes)
    count, first = sizes[band], (torch.cumsum(sizes, dim=0) - sizes)[band]
    index = torch.arange(len(band), device=device)
    return torch.where(count % 2 == 0, first + (index - first + count // 2) % count, -1)


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


@lru_cache(maxsize=8)
def _cell_layout(counts: tuple[int, ...], dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Band of each cell of the sky, of shape (k,), and the azimuth of its center, of shape (k,), on a device."""
    band = torch.repeat_interleave(torch.arange(len(counts), device=device), torch.as_tensor(counts, device=device))
    azimuth = torch.cat([2 * math.pi * (torch.arange(c) + 0.5) / c for c in counts]).to(dtype=dtype, device=device)
    return band, azimuth


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
    band, azimuth = _cell_layout(tuple(counts), directions.dtype, str(directions.device))
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


def band_radiance(
    table: ScatteringTable, edges: Sequence[float], nodes: int = 8, gathered: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean radiance scattered at least once that arrives from the elevations of each band, above and below the
    horizon.

    Args:
        table (ScatteringTable): Light scattered more than once, including the light it was gathered from.
        edges (Sequence[float]): Sine of the elevation at the edges of the bands, from the horizon up.
        nodes (int, optional): Number of elevations at which each band is averaged. Defaults to 8.
        gathered (torch.Tensor | None, optional): Part of the light gathered at each height from each direction to
            average instead, of shape (heights, directions, c), e.g. ``table.from_ground``. Defaults to None, i.e.
            ``table.gathered``.

    Returns:
        torch.Tensor: Mean radiance arriving from each band, of shape (heights, 2, b, c), where the second dimension
        is above, then below the horizon. It only depends on the light gathered and its directions, and is kept for
        the next frames.
    """
    gathered = table.gathered if gathered is None else gathered

    def compute() -> torch.Tensor:
        kwargs = {"dtype": gathered.dtype, "device": gathered.device}
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
            values = gathered[:, j - 1] + f * (gathered[:, j] - gathered[:, j - 1])  # (h, b*nodes, c)
            values = values.reshape(len(gathered), *mu.shape, -1)
            result.append((values * w[:, None] / 2).sum(dim=-2))
        return torch.stack(result, dim=1)

    # Kept by the very light that is averaged, along with the directions it was gathered from
    key = ("band radiance", tuple(edges), nodes)
    return _kept((gathered, table.directions), key, compute, store=_DERIVED, size=_DERIVED_SIZE)


def cell_profile(
    table: ScatteringTable, edges: Sequence[float], counts: Sequence[int], nodes: int = 8
) -> torch.Tensor | None:
    """Radiance scattered at least once that arrives from each cell of the sky, relative to its mean over the cell's
    band, above and below the horizon.

    The light of suns scattered once is brightest from directions close to each sun's, as the phase function sends it
    forward, and so is, less so, their light scattered more than once, see :class:`SunScattering
    <visionsim.medium.scattering.SunScattering>`: a cell towards a sun brings more of this light than the mean of its
    band, which weighs cells when objects hide them, and a cell away from it less. The harmonics of its variation with
    the azimuth from each sun are averaged over the elevations of each band and over the azimuths of each cell, and
    summed over heights and channels, as is the mean of the band, i.e. weighted by the light at each height, as the
    profile hardly changes with height.

    Args:
        table (ScatteringTable): Light scattered more than once, including the light it was gathered from.
        edges (Sequence[float]): Sine of the elevation at the edges of the bands, from the horizon up.
        counts (Sequence[int]): Number of cells of each band, which split its azimuths evenly starting from the x axis.
        nodes (int, optional): Number of elevations at which each band is averaged. Defaults to 8.

    Returns:
        torch.Tensor | None: Relative radiance of each cell, which is 1 on average over the cells of each band, of
        shape (2, k), for cells above, then mirrored below the horizon, or None if no sun makes it vary, e.g. without
        suns above the horizon, or for tables computed without their harmonics.
    """
    suns = [sun for sun in table.suns if sun.once is not None]
    if not suns:
        return None

    def compute() -> torch.Tensor:
        kwargs = {"dtype": table.gathered.dtype, "device": table.gathered.device}
        t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
        low, high = torch.as_tensor(edges[:-1], **kwargs), torch.as_tensor(edges[1:], **kwargs)
        mu = (low + high)[:, None] / 2 + (high - low)[:, None] / 2 * t  # (b, nodes)
        band, azimuth = _cell_layout(tuple(counts), table.gathered.dtype, str(table.gathered.device))
        width = 2 * math.pi / torch.as_tensor(counts, **kwargs)  # (b,)
        mean = table.gathered.sum(dim=(0, -1))  # (d,)

        def over_band(values: torch.Tensor, sign: float) -> torch.Tensor:
            # Mean over the elevations of each band, above or below the horizon, of values over the gathering
            # directions, interpolated linearly as their vertical components are sorted, (..., d) -> (..., b)
            x = (sign * mu).reshape(-1).contiguous()
            j = torch.searchsorted(table.directions, x).clamp(1, len(table.directions) - 1)
            x0, x1 = table.directions[j - 1], table.directions[j]
            f = ((x - x0) / (x1 - x0)).clamp(0, 1)
            at = values[..., j - 1] + f * (values[..., j] - values[..., j - 1])
            return (at.reshape(*values.shape[:-1], *mu.shape) * w / 2).sum(dim=-1)

        profiles = []
        for sign in (1.0, -1.0):
            band_mean = over_band(mean, sign)  # (b,)
            variation = torch.zeros_like(azimuth)
            for sun in suns:
                assert sun.once is not None
                harmonics = over_band((sun.once + sun.arriving).sum(dim=(1, -1)), sign)  # (modes, b)
                m = torch.arange(1, len(harmonics) + 1, **kwargs)[:, None]
                # Mean of cos(mφ) over the azimuths of each cell, around its center
                spread = torch.sinc(m * width / (2 * math.pi))  # (modes, b)
                relative = azimuth - torch.atan2(sun.towards[1], sun.towards[0]).to(azimuth)
                variation = variation + ((harmonics * spread)[:, band] * torch.cos(m * relative)).sum(dim=0)
            mean_of = band_mean[band]
            relative_radiance = 1 + variation / mean_of.clamp_min(1e-30)
            profiles.append(torch.where(mean_of > 0, relative_radiance, torch.ones_like(mean_of)))
        return torch.stack(profiles).clamp_min(0)

    # Kept by the very light it is derived from, along with the directions it was gathered from
    tensors = (table.gathered, table.directions, *(t for sun in suns for t in (sun.once, sun.arriving, sun.towards)))
    return _kept(
        tensors, ("cell profile", tuple(edges), tuple(counts), nodes), compute, store=_DERIVED, size=_DERIVED_SIZE
    )


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
    let all light through, each of shape (h + size, w + size). They only depend on the map, and are kept for the next
    frames, see :func:`_kept`."""
    height, width = (int(s) for s in maps.shapes[0])
    offset = int(maps.offsets[0])

    def windows() -> tuple[torch.Tensor, torch.Tensor]:
        image = maps.depths[offset : offset + height * width].reshape(1, 1, height, width)
        image = torch.nn.functional.pad(image, (size, size - 1, size, size - 1), value=torch.inf)

        def pool(values: torch.Tensor) -> torch.Tensor:
            # Along rows, then columns
            values = torch.nn.functional.max_pool2d(values, (1, size), stride=1)
            return torch.nn.functional.max_pool2d(values, (size, 1), stride=1)[0, 0]

        return -pool(-image), pool(image)

    return _kept((maps.depths,), ("windows", offset, height, width, size), windows)


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


class SkyWeights(NamedTuple):
    """Sums over the cells of the sky of each band of the phase function integrated over them, each weighted by the
    radiance of light scattered at least once arriving from it relative to its band's, see :func:`cell_profile`, which
    weigh the cells that hide light scattered more than once, see :func:`trace`."""

    profile: torch.Tensor
    """relative radiance of each cell, above and below the horizon, of shape (2, k)"""
    visible: torch.Tensor
    """sum over the visible cells of each band, from the middle of each sample, of shape (n, m, b)"""
    whole: torch.Tensor
    """sum over the cells of each band, of shape (n, b)"""
    visible_below: torch.Tensor
    """same as ``visible`` for the cells mirrored below the horizon"""
    whole_below: torch.Tensor
    """same as ``whole`` for the cells mirrored below the horizon"""


class Shadows(NamedTuple):
    """Where objects hide suns and cells of the sky along the rays of a frame, see :func:`trace`. As this hardly
    depends on the medium, it can be reused to add several media to the same frame, see :func:`shade`.

    Cells of the sky are weighted by the phase function integrated over them, which only depends on the medium's
    anisotropy: their weights are kept for the anisotropy of the medium the shadows were traced with, and recomputed
    from the visibility of each cell for media of other anisotropies.

    Cells mirrored below the horizon, from which the ground and the medium beneath light the medium, are looked up in
    the map of the cell half a turn around, which looks down along the mirrored direction: they are hidden by the
    surface it records below a point, such as the top of an object under it. At points below the surfaces it records,
    e.g. beside a wall lower than its top, they are hidden by objects that hide the cell above the horizon, as objects
    stand on the ground, unless they stand further than where light from below the horizon starts, on the ground. Where
    the light of a mirrored cell starts on the ground in the shadow of a sun, the ground lacks the sun's light, and where
    it starts on the ground next to objects, which hide part of the sky from it, see :func:`ground_sky`, the ground lacks
    that part of the light of the sky and of the glow of the medium.

    Light scattered more than once arrives brighter from cells towards suns, which hide more of it, see
    :func:`cell_profile`: shadows traced with the relative radiance of each cell also hold the sums of the phase
    function over the cells weighted by it, see :class:`SkyWeights`, which only serve media of a similar density and
    the same lighting.

    Each sample of the sky's visibility holds ``(3 + s) b + 2`` floating point values for ``s`` suns and ``b`` bands,
    e.g. 30 values with 7 bands and a sun, i.e. about 2 kB per ray of 8 samples in double precision, 4 GB for a frame
    of 1920x1080, and ``2 b`` more with the sums weighted by the radiance of each cell. The visibility of each cell,
    packed as bits, adds ``(2 + s) ceil(k / 8)`` bytes, e.g. 42 bytes for 107 cells. Shadows traced without it can't be
    reused for media of another anisotropy, for which the share of each band that the ground doesn't see is kept."""

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
    (n, m, ceil(k / 8)), or (n, m, 0) when traced without it"""
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
    shadowed_below: tuple[torch.Tensor, ...] = ()
    """for each sun of ``suns``, whether the light of each visible cell mirrored below the horizon starts on the ground
    in the sun's shadow, from the middle of each sample, packed as ``visible``, or nothing when traced without the
    visibility of each cell"""
    shadowed_weights_below: torch.Tensor | None = None
    """for each sun of ``suns``, sum over the visible cells mirrored below the horizon of each band whose light starts
    on the ground in the sun's shadow of the phase function integrated over them, from the middle of each sample, of
    shape (n, m, s, b)"""
    ground_height: float = 0.0
    """height of the ground, where light from below the horizon starts"""
    scattered: SkyWeights | None = None
    """sums of the phase function over the cells of each band weighted by the radiance of light scattered at least once
    arriving from each, which weigh the cells that hide light scattered more than once, or None when traced without
    the relative radiance of each cell, which then weighs them by the mean radiance of their band"""
    unseen_weights_below: torch.Tensor | None = None
    """sum over the visible cells mirrored below the horizon of each band of the phase function integrated over them,
    each times the share of the sky that the ground doesn't see where its light starts, see :func:`ground_sky`, from
    the middle of each sample, of shape (n, m, b), or None when traced without it"""


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
    keep_cells: bool = True,
    shafts: bool = True,
    sky: bool = True,
    ground_shadows: bool = True,
    profile: torch.Tensor | None = None,
    ground_sky_occlusion: bool = True,
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
        keep_cells (bool, optional): If true, keep which cells of the sky each sample sees, as one bit per cell, which
            :func:`shade` needs for media of another anisotropy. Otherwise, the shadows take less memory, but only
            serve media of the same anisotropy. Defaults to True.
        shafts (bool, optional): If true, trace the segments of rays along which suns are hidden, i.e. the light
            shafts. Otherwise, suns are only kept for their shadows on the ground, with no segments. Defaults to True.
        sky (bool, optional): If true, sample the visibility of the cells of the sky along rays. Otherwise, the shadows
            have no samples of it, as when the scene has no maps of the sky. Defaults to True.
        ground_shadows (bool, optional): If true, find where the light of the cells mirrored below the horizon starts
            on the ground in the shadows of suns, along with the sky's visibility. Defaults to True.
        profile (torch.Tensor | None, optional): Radiance of light scattered at least once arriving from each cell of
            the sky relative to its band's, above and below the horizon, of shape (2, k), see :func:`cell_profile`, by
            which the phase function is weighted in the sums of :attr:`Shadows.scattered`. Defaults to None, i.e.
            none of these sums.
        ground_sky_occlusion (bool, optional): If true, find the share of the sky that the ground doesn't see where
            the light of the cells mirrored below the horizon starts on it, see :func:`ground_sky`, along with the
            sky's visibility. Defaults to True.

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

    hidden_suns, sun_maps, sun_images = [], [], []
    texels = occlusion.sun_maps.texels.tolist()
    cells = len(occlusion.sky_maps.texels) if sky else 0
    empty = [torch.zeros(0, dtype=torch.long, device=directions.device)] + [torch.zeros(0, **kwargs)] * 3
    for towards in suns:
        sun_z = float(towards[2])
        alignment = (occlusion.sun_directions @ towards).tolist()
        if sun_z <= 0 or not alignment or max(alignment) < 0.9999 or not (shafts or (ground_shadows and cells)):
            continue
        index = alignment.index(max(alignment))
        maps = select_maps(occlusion.sun_maps, index)
        sun_maps.append(maps)
        map_height, map_width = (int(size) for size in maps.shapes[0])
        offset = int(maps.offsets[0])
        sun_images.append(maps.depths[offset : offset + map_height * map_width].reshape(map_height, map_width))
        if not shafts:
            # Only the sun's shadows on the ground are needed
            hidden_suns.append(SunShadows(towards, *empty))
            continue
        start, end = segment(height * math.sqrt(1 - sun_z**2) / sun_z)
        # Each ray is split into intervals of a few texels, laid out one ray after another, so that the many rays
        # that only cross a short stretch where objects can cast shadows have few of them. Intervals that are surely
        # lit or hidden, as they lie above or below every surface around them, are only looked up once, and the
        # others, which a shadow's edge may cross, every texel
        length, texel = end - start, texels[index]
        counts = torch.ceil(length / (sun_step * texel)).clamp(0, max_sun_samples).long()
        window = math.ceil(sun_step) + 2
        windows = _depth_windows(maps, window)
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
    bands = len(counts)
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
            ground_height=ground_height,
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
    band_phases, band_phases_below = phases @ one_hot, phases_below @ one_hot  # (d, b)
    # The same, weighted by the radiance of light scattered at least once arriving from each cell
    if profile is not None:
        profile = profile.to(**kwargs)
        band_profiled = ((phases * profile[0]) @ one_hot, (phases_below * profile[1]) @ one_hot)

    # Rays close to each other also sample places close to each other, so that points are snapped to a grid finer than
    # the maps' texels, and the sky's visibility is only looked up once per cell of the grid. Samples in the same cell,
    # along rays of similar directions, see the same fraction of each band
    size = sky_grid * float(occlusion.sky_maps.texels.min())
    pad = torch.as_tensor([margin, margin, 0.0], **kwargs)
    box = (torch.cat([low[:2], low.new_full((1,), lowest)]) - pad, high + pad)
    grid = [_snap(origin + directions[rays, None] * middles[rays, ..., None], *box, size) for rays in _chunks(n, 2**20)]
    groups = _group(torch.cat(grid), ray_bin, len(bins))
    opposite = _opposite_cells(counts, directions.device)
    band_of = one_hot.argmax(dim=-1)
    # Light from below the horizon starts on the ground this far horizontally from a point, per meter above it
    towards = occlusion.sky_maps.axes[:, 2]
    slopes = towards[:, :2] / towards[:, 2:]

    packed: dict[str, list[torch.Tensor]] = {"above": [], "below": []}
    fractions: dict[str, list[torch.Tensor]] = {"above": [], "below": [], "profiled above": [], "profiled below": []}
    grounds = list(zip(sun_maps, sun_images)) if ground_shadows else []
    shadowed_bits: list[list[torch.Tensor]] = [[] for _ in grounds]
    shadowed_fractions: list[list[torch.Tensor]] = [[] for _ in grounds]
    for part, span, local in groups.split(max(1, int(memory // (cells * 128)))):
        centers = _cell_centers(groups.cells[part], *box, size)
        cell = _sky_cells(occlusion, centers, opposite, ground_height, bias)
        # Where the light from below the horizon starts on the ground in the shadows of suns
        shaded = [
            _ground_shadowed(maps, image, centers, slopes, cell.visible_below, ground_height, bias)
            for maps, image in grounds
        ]
        if keep_cells:
            packed["above"].append(_pack(cell.visible))
            packed["below"].append(_pack(cell.visible_below))
            for s, mask in enumerate(shaded):
                shadowed_bits[s].append(_pack(mask))

        # Fractions of each band that the groups of these cells of the grid see, from the weights of the hidden cells,
        # which are few
        direction = groups.direction[span]
        whole, whole_below = band_phases[direction], band_phases_below[direction]
        size_part = part.stop - part.start
        for index, (name, mask, weighted, total) in enumerate(
            (
                ("above", ~cell.visible, phases, whole),
                ("below", ~cell.visible_below, phases_below, whole_below),
            )
        ):
            rows, columns = mask.nonzero(as_tuple=True)
            if profile is None:
                hidden = _phase_sums(rows, columns, local, size_part, direction, weighted, band_of, len(counts))
                fractions[name].append(_ratio((total - hidden).clamp_min(0), total))
                continue
            # The sums weighted by the radiance arriving from each cell too, at once
            values = torch.stack([torch.ones_like(columns, dtype=weighted.dtype), profile[index][columns]], dim=-1)
            hidden = _phase_sums(rows, columns, local, size_part, direction, weighted, band_of, len(counts), values)
            fractions[name].append(_ratio((total - hidden[..., 0]).clamp_min(0), total))
            total = band_profiled[index][direction]
            fractions[f"profiled {name}"].append(_ratio((total - hidden[..., 1]).clamp_min(0), total))
        for s, mask in enumerate(shaded):
            rows, columns = mask.nonzero(as_tuple=True)
            hidden = _phase_sums(rows, columns, local, size_part, direction, phases_below, band_of, len(counts))
            shadowed_fractions[s].append(_ratio(hidden, whole_below))

    weights, weights_below = band_phases[ray_bin], band_phases_below[ray_bin]
    nothing = torch.zeros(n, middles.shape[1], 0, dtype=torch.uint8, device=directions.device)
    visible = torch.cat(packed["above"])[groups.sample_cell] if keep_cells else nothing
    visible_below = torch.cat(packed["below"])[groups.sample_cell] if keep_cells else nothing
    shadowed = [torch.cat(fraction)[groups.group] * weights_below[:, None] for fraction in shadowed_fractions]
    visible_weights_below = torch.cat(fractions["below"])[groups.group] * weights_below[:, None]
    unseen = None
    if ground_sky_occlusion:
        # The share of the sky that the ground sees changes more slowly than the visibility of cells, and is looked up
        # on a coarser grid
        ground = ground_sky(occlusion, ground_height, bias)
        args = (occlusion, ground, origin, directions, bounds, middles, ray_bin, phases_below, ground_height, bias)
        unseen = _ground_unseen_fractions(*args, memory=memory) * visible_weights_below
    scattered = None
    if profile is not None:
        whole_profiled = band_profiled[0][ray_bin], band_profiled[1][ray_bin]
        scattered = SkyWeights(
            profile,
            visible=torch.cat(fractions["profiled above"])[groups.group] * whole_profiled[0][:, None],
            whole=whole_profiled[0],
            visible_below=torch.cat(fractions["profiled below"])[groups.group] * whole_profiled[1][:, None],
            whole_below=whole_profiled[1],
        )
    return Shadows(
        n,
        tuple(hidden_suns),
        bounds=bounds,
        middles=middles,
        visible=visible,
        visible_below=visible_below,
        sky_edges=edges,
        sky_counts=counts,
        anisotropy=medium.anisotropy,
        visible_weights=torch.cat(fractions["above"])[groups.group] * weights[:, None],
        weights=weights,
        visible_weights_below=visible_weights_below,
        weights_below=weights_below,
        shadowed_below=tuple(torch.cat(bits)[groups.sample_cell] for bits in shadowed_bits) if keep_cells else (),
        shadowed_weights_below=torch.stack(shadowed, dim=2) if shadowed else None,
        ground_height=ground_height,
        scattered=scattered,
        unseen_weights_below=unseen,
    )


class _Groups(NamedTuple):
    """Samples along rays grouped by the cell of a grid they lie in and the bin of their ray's direction, see
    :func:`_group`."""

    cells: torch.Tensor
    """index of each cell of the grid that holds samples, in increasing order, of shape (u,)"""
    sample_cell: torch.Tensor
    """index in ``cells`` of the cell of each sample, of shape (n, m)"""
    cell: torch.Tensor
    """index in ``cells`` of the cell of each group, in increasing order, of shape (g,)"""
    direction: torch.Tensor
    """direction bin of each group, of shape (g,)"""
    group: torch.Tensor
    """group of each sample, of shape (n, m)"""

    def split(self, size: int) -> Iterator[tuple[slice, slice, torch.Tensor]]:
        """Slices of at most ``size`` consecutive cells, along with the slice of their groups, and the index of the cell
        of each of these groups within the slice of cells."""
        parts = [slice(p.start, min(p.stop, len(self.cells))) for p in _chunks(len(self.cells), size)]
        # The groups of all slices are found at once, which only waits once for a GPU
        bounds = self.cell.new_tensor([p.start for p in parts] + [len(self.cells)])
        firsts = torch.searchsorted(self.cell, bounds).tolist()
        for part, first, last in zip(parts, firsts, firsts[1:]):
            yield part, slice(first, last), self.cell[first:last] - part.start


def _group(indices: torch.Tensor, ray_bin: torch.Tensor, bins: int) -> _Groups:
    """Group samples by the cell of a grid they lie in, given its index, of shape (n, m), and by the direction bin of
    their ray, given for each ray, of shape (n,), among a given number of bins."""
    # Groups are found among all samples, and their cells among the far fewer groups
    groups, group = torch.unique(indices * bins + ray_bin[:, None], return_inverse=True)
    cells, cell = torch.unique(groups // bins, return_inverse=True)
    return _Groups(cells, cell[group], cell, groups % bins, group)


def _phase_sums(
    rows: torch.Tensor,
    columns: torch.Tensor,
    local: torch.Tensor,
    grid_cells: int,
    direction: torch.Tensor,
    phases: torch.Tensor,
    slots: torch.Tensor,
    size: int,
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sums over cells of the sky seen from cells of a grid of the phase function integrated over them from each group
    of samples of the cell of the grid, times values if given, in slots of the cells of the sky, e.g. their bands.

    Args:
        rows (torch.Tensor): Cell of the grid of each entry, of shape (e,).
        columns (torch.Tensor): Cell of the sky of each entry, of shape (e,).
        local (torch.Tensor): Cell of the grid of each group, in increasing order, of shape (g,), such that every cell
            of the grid that entries refer to has a group.
        grid_cells (int): Number of cells of the grid, which bounds their indices.
        direction (torch.Tensor): Direction bin of each group, of shape (g,).
        phases (torch.Tensor): Phase function integrated over each cell of the sky from each direction bin, of shape
            (d, k).
        slots (torch.Tensor): Slot of each cell of the sky, of shape (k,).
        size (int): Number of slots.
        values (torch.Tensor | None, optional): Value of each entry, of shape (e, c). Defaults to None.

    Returns:
        torch.Tensor: Sums for each group and slot, of shape (g, size), or (g, size, c) with values.
    """
    counts = torch.zeros(grid_cells, dtype=torch.long, device=local.device).index_add_(0, local, torch.ones_like(local))
    starts = torch.cumsum(counts, dim=0) - counts
    # Each entry is repeated for every group of its cell of the grid
    entry, offset = _ragged(counts[rows], int(counts[rows].sum()))
    group = starts[rows[entry]] + offset
    weight = phases[direction[group], columns[entry]]
    contribution = weight if values is None else weight[:, None] * values[entry]
    sums = weight.new_zeros(len(local) * size, *contribution.shape[1:])
    sums.index_add_(0, group * size + slots[columns[entry]], contribution)
    return sums.reshape(len(local), size, *contribution.shape[1:])


def _ground_shadowed(
    maps: ShadowMaps,
    image: torch.Tensor,
    points: torch.Tensor,
    slopes: torch.Tensor,
    visible: torch.Tensor,
    ground_height: float,
    bias: float,
) -> torch.Tensor:
    """Whether the light of each visible cell mirrored below the horizon starts on the ground in the shadow of a sun,
    from points.

    The ground is looked up in the sun's map at the nearest texel, as each cell samples the ground coarsely anyway. Its
    coordinates in the map are affine in the height of points, along each cell's direction, which spares computing
    where the light of every cell starts on the ground. Every pair of a point and a cell is looked up at once, which
    takes no more time than finding those within the map first, and the same time on every call.

    Args:
        maps (ShadowMaps): Map of the sun.
        image (torch.Tensor): Depths of the map, of shape (h, w).
        points (torch.Tensor): Points, of shape (u, 3).
        slopes (torch.Tensor): Horizontal offset, per meter of height, at which the light of each mirrored cell starts
            on the ground, of shape (k, 2).
        visible (torch.Tensor): Whether each mirrored cell is visible from each point, down to the ground, of shape
            (u, k).
        ground_height (float): Height of the ground.
        bias (float): Depth bias of the map, in texels.

    Returns:
        torch.Tensor: Whether each cell is visible from each point and its light starts on the ground in the sun's
        shadow, of shape (u, k).
    """
    height = (points[:, 2:] - ground_height).clamp_min(0)  # (u, 1)
    below = torch.cat([points[:, :2], torch.full_like(height, ground_height)], dim=-1)
    offsets = torch.cat([slopes, torch.zeros_like(slopes[:, :1])], dim=-1)  # (k, 3)
    # Texel coordinates, and depth minus the bias, of the ground right below points, and their change per meter of
    # height along each cell's direction, as in `_texel_coordinates`
    axes, texel = maps.axes[0], maps.texels[0]
    rows, columns = image.shape
    scale = torch.stack([1 / texel, 1 / texel, -torch.ones_like(texel)])
    shift = torch.stack([texel.new_tensor((columns - 1) / 2), texel.new_tensor((rows - 1) / 2), -bias * texel])
    start = ((below - maps.origins[0]) @ axes.T) * scale + shift  # (u, 3)
    step = (offsets @ axes.T) * scale  # (k, 3)
    x = torch.addcmul(start[:, :1], height, step[:, 0]).round_()  # (u, k)
    y = torch.addcmul(start[:, 1:2], height, step[:, 1]).round_()
    inside = visible & (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
    index = torch.where(inside, y, 0).long() * columns + torch.where(inside, x, 0).long()
    depth = torch.addcmul(start[:, 2:3], height, step[:, 2])
    return inside & (depth > image.reshape(-1)[index])


class GroundSky(NamedTuple):
    """Share of the sky that the ground sees around the objects of a scene, on a grid over the ground, see
    :func:`ground_sky`."""

    low: torch.Tensor
    """horizontal coordinates of the corner of the grid, of shape (2,)"""
    size: float
    """size of the cells of the grid, in meters"""
    share: torch.Tensor
    """share of the irradiance of a uniform sky that the ground receives from the cells of the sky it sees, at the
    center of each cell of the grid, of shape (rows, columns), whose rows go along the y axis and columns along x"""


_GROUND_SKY_TEXELS = 8.0
"""Size of the cells of the grid of :func:`ground_sky`, in texels of the finest map of the sky, finer than the smallest
objects that hide much of the sky from the ground, as the share changes over about their size"""
_GROUND_SKY_CELLS = 2**18
"""Largest number of cells of the grid of :func:`ground_sky`, whose cells are larger over larger scenes"""
_GROUND_SKY_REACH = 3.0
"""Distance from the objects of a scene, in units of their height, over which :func:`ground_sky` finds the share of the
sky that the ground sees, beyond which objects hide at most about 3% of the irradiance of a uniform sky, as an
infinitely long wall would"""


def ground_sky(occlusion: Occlusion, ground_height: float = 0.0, bias: float = 0.5) -> GroundSky:
    """Share of the irradiance of a uniform sky that the ground receives from the cells of the sky it sees, around the
    objects of a scene.

    Objects hide part of the sky from the ground next to them, which then receives less of the light of the sky, and of
    the glow of the medium, than in the open, and reflects less of it into the medium. As surfaces facing up, see
    :func:`surface_irradiance <visionsim.medium.surfaces.surface_irradiance>`, each cell of the sky that the ground sees
    brings its solid angle times the mean cosine of its elevations, the ground being looked up in the maps of the sky
    at the center of each cell of a grid over the ground under the objects, and up to three times their height around
    them, beyond which the ground sees the whole sky. Its cells are 8 texels of the finest map of the sky wide, or
    larger over scenes so large that the grid would have more than 2^18 cells. It only depends on the maps, and is kept
    for the next frames.

    Args:
        occlusion (Occlusion): Shadow maps of the scene, whose maps of the sky don't record the ground.
        ground_height (float, optional): Height of the ground. Defaults to 0.
        bias (float, optional): Depth bias of the maps, in texels, see :func:`visibility`. Defaults to 0.5.

    Returns:
        GroundSky: Share of the sky that the ground sees, on a grid over the ground.
    """
    maps = occlusion.sky_maps
    kwargs = {"dtype": maps.texels.dtype, "device": maps.texels.device}

    def compute() -> GroundSky:
        margin = _GROUND_SKY_REACH * max(float(occlusion.bounds[1, 2]) - ground_height, 0.0)
        corner, extent = occlusion.bounds[0, :2] - margin, occlusion.bounds[1, :2] - occlusion.bounds[0, :2] + 2 * margin
        area = float(extent[0].clamp_min(0)) * float(extent[1].clamp_min(0))
        size = max(_GROUND_SKY_TEXELS * float(maps.texels.min()), math.sqrt(area / _GROUND_SKY_CELLS))
        columns, rows = (int(d) for d in torch.ceil(extent / size).clamp_min(1))
        x = corner[0] + (torch.arange(columns, **kwargs) + 0.5) * size
        y = corner[1] + (torch.arange(rows, **kwargs) + 0.5) * size
        heights = torch.full((rows, columns), ground_height, **kwargs)
        points = torch.stack([x.expand(rows, -1), y[:, None].expand(-1, columns), heights], dim=-1).reshape(-1, 3)
        # Mean cosine of each band's elevations, times the solid angle of each of its cells
        edges, counts = occlusion.sky_edges, occlusion.sky_counts
        weights = torch.as_tensor(
            [math.pi * (top**2 - bottom**2) / count for bottom, top, count in zip(edges[:-1], edges[1:], counts)],
            **kwargs,
        )
        weights = torch.repeat_interleave(weights, torch.as_tensor(counts, device=weights.device))
        share = [
            (_signed_depths(maps, points[part], bias) >= 0).to(weights) @ weights / weights.sum()
            for part in _chunks(len(points), max(1, 2**22 // len(weights)))
        ]
        return GroundSky(corner, size, torch.cat(share).reshape(rows, columns))

    return _kept((maps.depths, maps.axes, occlusion.bounds), ("ground sky", ground_height, bias), compute)


def _ground_unseen(
    ground: GroundSky, points: torch.Tensor, slopes: torch.Tensor, visible: torch.Tensor, ground_height: float
) -> torch.Tensor:
    """Share of the irradiance of a uniform sky that the ground doesn't see where the light of each visible cell mirrored
    below the horizon starts on it, from points of shape (u, 3), given the horizontal offset per meter of height at
    which it starts, of shape (k, 2), see :func:`_ground_shadowed`, and whether each cell is visible, of shape (u, k),
    looked up at the nearest cell of the grid, and zero for cells that aren't visible or start beyond it, of shape
    (u, k)."""
    height = (points[:, 2:] - ground_height).clamp_min(0)  # (u, 1)
    rows, columns = ground.share.shape
    x = ((torch.addcmul(points[:, :1], height, slopes[:, 0]) - ground.low[0]) / ground.size).floor_()  # (u, k)
    y = ((torch.addcmul(points[:, 1:2], height, slopes[:, 1]) - ground.low[1]) / ground.size).floor_()
    inside = visible & (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
    index = torch.where(inside, y, 0).long() * columns + torch.where(inside, x, 0).long()
    return torch.where(inside, 1 - ground.share.reshape(-1)[index], 0)


def _ground_unseen_fractions(
    occlusion: Occlusion,
    ground: GroundSky,
    origin: torch.Tensor,
    directions: torch.Tensor,
    bounds: torch.Tensor,
    middles: torch.Tensor,
    ray_bin: torch.Tensor,
    phases_below: torch.Tensor,
    ground_height: float,
    bias: float,
    memory: float = 2**30,
) -> torch.Tensor:
    """Share of the sky that the ground doesn't see where the light of the visible cells mirrored below the horizon of
    each band starts on it, weighted by the phase function integrated over them, from samples along rays.

    It is looked up on a grid as coarse as the grid of the ground's share of the sky, see :func:`ground_sky`, at the
    mean position of the samples in each of its cells, which lies on the side of objects that the camera sees, for each
    bin of directions of rays, as cells are weighted by the phase function integrated over them.

    Args:
        occlusion (Occlusion): Shadow maps of the scene.
        ground (GroundSky): Share of the sky that the ground sees, see :func:`ground_sky`.
        origin (torch.Tensor): Common origin of the rays, of shape (3,).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        bounds (torch.Tensor): Distance along each ray of the boundaries of the samples, of shape (n, m + 1).
        middles (torch.Tensor): Distance along each ray of the middle of each sample, of shape (n, m).
        ray_bin (torch.Tensor): Bin of the direction of each ray, of shape (n,).
        phases_below (torch.Tensor): Phase function integrated over each cell mirrored below the horizon from each bin
            of directions, of shape (d, k).
        ground_height (float): Height of the ground.
        bias (float): Depth bias of the maps, in texels, see :func:`visibility`.
        memory (float, optional): Rough bound, in bytes, on the memory used by intermediate values. Defaults to 1 GiB.

    Returns:
        torch.Tensor: Share of the sky that the ground doesn't see, of shape (n, m, b), zero for samples of empty
        intervals or that see no cell mirrored below the horizon.
    """
    kwargs = {"dtype": directions.dtype, "device": directions.device}
    counts = occlusion.sky_counts
    n, m = middles.shape
    result = torch.zeros(n, m, len(counts), **kwargs)
    # Samples of empty intervals, e.g. along rays that don't pass by objects, scatter nothing
    rays, samples = (bounds.diff(dim=1) > 0).nonzero(as_tuple=True)
    if not len(rays):
        return result
    points = origin + directions[rays] * middles[rays, samples, None]
    groups = _group(
        _snap(points, points.amin(dim=0), points.amax(dim=0), ground.size)[:, None], ray_bin[rays], len(phases_below)
    )
    flat = groups.sample_cell[:, 0]
    means = torch.zeros(len(groups.cells), 3, **kwargs).index_add_(0, flat, points)
    means = means / torch.zeros(len(groups.cells), 1, **kwargs).index_add_(0, flat, torch.ones_like(points[:, :1]))
    opposite = _opposite_cells(counts, directions.device)
    towards = occlusion.sky_maps.axes[:, 2]
    slopes = towards[:, :2] / towards[:, 2:]
    one_hot = _band_one_hot(counts, **kwargs)
    fractions = []
    for part, span, local in groups.split(max(1, int(memory // (len(towards) * 64)))):
        cell = _sky_cells(occlusion, means[part], opposite, ground_height, bias)
        unseen = _ground_unseen(ground, means[part], slopes, cell.visible_below, ground_height)  # (u, k)
        weighted = phases_below[groups.direction[span]]  # (g, k)
        seen = (weighted * cell.visible_below[local].to(weighted)) @ one_hot
        missed = (weighted * unseen[local].to(weighted)) @ one_hot
        fractions.append(torch.where(seen > 0, missed / seen.clamp_min(1e-30), 0))
    return result.index_put_((rays, samples), torch.cat(fractions)[groups.group[:, 0]])


def object_reflections(
    occlusion: Occlusion,
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    shadows: Shadows,
    surfaces: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    extinction: torch.Tensor | None = None,
    similarity: bool = False,
    grid: float = 4.0,
    bias: float = 0.5,
    memory: float = 2**30,
) -> torch.Tensor:
    """Light of suns and the sky that objects reflect towards the samples of the sky's visibility along rays, which the
    medium scatters towards the camera.

    From each cell of the sky that an object hides from a point, above or mirrored below the horizon, the point receives
    the light of the object's surface where the cell's direction enters it, attenuated by the medium on its way, see
    :func:`_object_light`. In fog, which scatters light forward, much of the light that the medium takes out of it on
    its way is scattered in about the same direction, and still reaches the point, as light scattered more than once:
    with ``similarity``, the light is attenuated with the extinction reduced by the similarity relation of radiative
    transfer, see :func:`similarity_extinction <visionsim.medium.optics.similarity_extinction>`, as the light of lamps
    that lights surfaces, see :func:`lamp_irradiance <visionsim.medium.surfaces.lamp_irradiance>`. In the day scene of
    ``examples/medium``, this brings the light that objects reflect from 0.70 to 0.87 times Cycles' with every volume
    bounce (objects and ground of the same albedo, Lambertian), which is 1.6 times the light that its medium scatters
    once, about uniformly over the frame, while the light scattered sideways on its way, which then reaches points from
    other directions, is still missing. This light changes more slowly than shadows, and is gathered once per cell of a
    grid coarser than that of the sky's visibility, at the mean position of the samples in the cell, which lies on the
    side of objects that the camera sees, for each bin of directions of rays, as cells are weighted by the phase
    function integrated over them. It depends on the medium, unlike shadows, and is gathered for each medium.

    Args:
        occlusion (Occlusion): Shadow maps of the scene, on the same device and with the same precision as the rays.
        medium (Medium): Participating medium, made of a single height fog.
        origin (torch.Tensor): Common origin of the rays, of shape (3,).
        directions (torch.Tensor): Unit directions of the rays, of shape (n, 3).
        shadows (Shadows): Where objects hide suns and the sky along the rays, see :func:`trace`, whose samples of the
            sky's visibility receive the light.
        surfaces (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Radiance of the objects' surfaces, of shape
            (e, c), as a function of points on them, of shape (e, 3), and of their unit normals, of shape (e, 3), e.g.
            :func:`object_radiance <visionsim.medium.surfaces.object_radiance>`.
        extinction (torch.Tensor | None, optional): Extinction coefficient per channel, of shape (c,), or (1,) when it
            is the same for all, which attenuates the light on its way. Defaults to the medium's extinction.
        similarity (bool, optional): If true, the light is attenuated with the extinction reduced by the similarity
            relation, which keeps the light that the medium scatters forward on its way. Defaults to False, i.e. the
            light is scattered once only.
        grid (float, optional): Size of the cells of the grid on which the light is gathered, in texels of the finest
            map of the sky. Defaults to 4.
        bias (float, optional): Depth bias of shadow maps, in texels, see :func:`visibility`. Defaults to 0.5.
        memory (float, optional): Rough bound, in bytes, on the memory used by intermediate values, which are computed
            for as many cells of the grid at once as fit. Defaults to 1 GiB.

    Returns:
        torch.Tensor: Light that the medium scatters towards the camera per unit of optical depth at the middle of each
        sample, given its albedo, of shape (n, m, c), or (n, m, 1) when there is none.
    """
    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    kwargs = {"dtype": directions.dtype, "device": directions.device}
    extinction = torch.as_tensor([medium.extinction], **kwargs) if extinction is None else extinction.to(**kwargs)
    if similarity:
        extinction = similarity_extinction(medium, extinction)
    n, m = shadows.middles.shape
    edges, counts = shadows.sky_edges, shadows.sky_counts
    cells = len(occlusion.sky_maps.texels)
    # Samples of empty intervals, e.g. along rays that don't pass by objects, scatter nothing
    rays, samples = (shadows.bounds.diff(dim=1) > 0).nonzero(as_tuple=True)
    if not cells or not len(rays):
        return torch.zeros(n, m, 1, **kwargs)
    points = origin + directions[rays] * shadows.middles[rays, samples, None]
    bins, ray_bin = torch.unique(_direction_bins(directions, _DIRECTION_BIN), return_inverse=True)
    mean = torch.zeros(len(bins), 3, **kwargs).index_add_(0, ray_bin, directions)
    mean = torch.nn.functional.normalize(mean, dim=-1)
    # Cells above the horizon, then mirrored below it
    g = medium.anisotropy
    phases = torch.cat([cell_phases(edges, counts, g, mean), cell_phases(edges, counts, g, mean, True)], dim=1)
    size = grid * float(occlusion.sky_maps.texels.min())
    groups = _group(_snap(points, points.amin(dim=0), points.amax(dim=0), size)[:, None], ray_bin[rays], len(bins))
    flat = groups.sample_cell[:, 0]
    means = torch.zeros(len(groups.cells), 3, **kwargs).index_add_(0, flat, points)
    means = means / torch.zeros(len(groups.cells), 1, **kwargs).index_add_(0, flat, torch.ones_like(points[:, :1]))
    opposite = _opposite_cells(counts, directions.device)
    nothing = torch.zeros(2 * cells, dtype=torch.long, device=directions.device)
    light = []
    # Points are gathered in as few chunks as fit, which keeps the number of operators low on a GPU, taking about 256
    # bytes per point and cell of the sky in double precision, where a fifth of the cells are hidden
    for part, span, local in groups.split(max(1, int(memory // (cells * 256)))):
        cell = _sky_cells(occlusion, means[part], opposite, shadows.ground_height, bias)
        rows, columns, values = _object_light(occlusion, fog, means[part], cell, surfaces, extinction, bias)
        direction = groups.direction[span]
        light.append(_phase_sums(rows, columns, local, part.stop - part.start, direction, phases, nothing, 1, values))
    gathered = torch.cat(light)[groups.group[:, 0], 0]
    result = gathered.new_zeros(n, m, gathered.shape[-1])
    return result.index_put_((rays, samples), medium.albedo * gathered)


class _SkyCells(NamedTuple):
    """Visibility of the cells of the sky from points, see :func:`_sky_cells`, each of shape (u, k)."""

    signed: torch.Tensor
    """distance from each point along the direction of each cell's map, away from the sky, to the surface it recorded,
    negative where the surface hides the cell, see :func:`_signed_depths`"""
    visible: torch.Tensor
    """whether each cell is visible"""
    visible_below: torch.Tensor
    """whether each cell mirrored below the horizon is visible, down to the ground"""
    below: torch.Tensor
    """distance along each cell mirrored below the horizon to the object under the point that the map of the opposite
    cell records, or infinity where there is none"""
    reach: torch.Tensor
    """distance along each cell mirrored below the horizon down to the ground"""
    mirrored: torch.Tensor
    """direction of each cell mirrored below the horizon, of shape (k, 3)"""


def _sky_cells(
    occlusion: Occlusion, points: torch.Tensor, opposite: torch.Tensor, ground_height: float, bias: float
) -> _SkyCells:
    """Visibility of the cells of the sky from points of shape (u, 3), above and mirrored below the horizon.

    The map of the cell half a turn around a cell looks down along the cell's direction mirrored below the horizon: a
    point in front of the surface it records sees that surface below it, which hides the ground beyond it. Points behind
    that surface, which lies above them, can't tell from it what lies below them: objects that hide the cell above the
    horizon are taken to stand on the ground, and to hide the mirrored cell too if they stand closer than where its
    light starts on the ground, as for points in bands of an odd number of cells, which have no opposite cell."""
    towards = occlusion.sky_maps.axes[:, 2]  # (k, 3)
    mirrored = towards * towards.new_tensor([1.0, 1.0, -1.0])
    signed = _signed_depths(occlusion.sky_maps, points, bias)  # (u, k)
    reach = (points[:, 2:3] - ground_height).clamp_min(0) / towards[:, 2]
    # Hidden by an object closer than where light from below the horizon starts
    standing = (signed < 0) & (-signed < reach)
    seen_below = signed[:, opposite.clamp_min(0)]
    behind = (seen_below < 0) | (opposite < 0)
    below = torch.where(~behind & (seen_below < reach), seen_below, torch.full_like(signed, torch.inf))
    visible_below = torch.where(behind, ~standing, torch.isinf(below))
    return _SkyCells(signed, signed >= 0, visible_below, below, reach, mirrored)


def _lines_in(
    maps: ShadowMaps, index: torch.Tensor, points: torch.Tensor, directions: torch.Tensor, bias: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Texel coordinates, from the center of the first texel, and depth below the map's plane minus the bias, of points
    of shape (e, 3) in one map each, given its index, of shape (e,), and their change per meter along directions of
    shape (e, 3), each of shape (e, 3), see :func:`_texel_coordinates`."""
    axes, texel = maps.axes[index], maps.texels[index]  # (e, 3, 3), (e,)
    scale = torch.stack([1 / texel, 1 / texel, -torch.ones_like(texel)], dim=-1)
    half = maps.shapes[index].to(texel) / 2 - 0.5
    shift = torch.stack([half[:, 1], half[:, 0], -bias * texel], dim=-1)
    start = torch.einsum("ed,ead->ea", points - maps.origins[index], axes) * scale + shift
    return start, torch.einsum("ed,ead->ea", directions, axes) * scale


def _facing_cells(counts: Sequence[int], towards: torch.Tensor) -> torch.Tensor:
    """Index of the cell of the highest band whose azimuth is the most opposite to that of each cell, of shape (k,)."""
    first = len(towards) - counts[-1]
    return first + (towards[:, :2] @ towards[first:, :2].T).argmin(dim=1)


_BISECTIONS = 8
"""Number of bisections that find where the direction of a hidden cell enters the object that hides it, see
:func:`_object_light`"""


def _object_light(
    occlusion: Occlusion,
    fog: HeightFog,
    points: torch.Tensor,
    cell: _SkyCells,
    surfaces: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    extinction: torch.Tensor,
    bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Light that objects reflect towards points of shape (u, 3) from the cells of the sky they hide, and from the cells
    mirrored below the horizon they hide, attenuated on its way, given the radiance of their surfaces, see
    :func:`object_reflections`, as the index of the point, of shape (e,), the index of the cell, of shape (e,), which is
    that of a cell above the horizon, or the number of cells plus that of a cell mirrored below it, and the light, of
    shape (e, c), of each hidden cell. Cells above and below the horizon are gathered at once, as they only differ by
    their direction and by where their surface is first looked for.

    The surface that hides a cell mirrored below the horizon is often the one that the map of the cell half a turn
    around records under the point, as it looks down along the mirrored direction. Otherwise, the cell's own map only
    records the far side of the object that hides it, and where the cell's direction enters the object is found by
    bisection between the point and the far side, in the map of a cell that sees the point from the other side: the
    medium in front of the object facing the point is seen by it, while the inside of objects is hidden from every
    map. That of the cell half a turn around is tried first, then that of the cell of the highest band whose azimuth
    is the most opposite, as few objects hide it. Where no such map sees the point, the far side is taken. The normals
    of surfaces come from the slopes of the depths that the map records around where the direction enters them, where
    they face the point, or are taken horizontal and facing the point otherwise, as for walls standing on the ground.
    Their radiance is looked up a texel in front of them, so that those facing a sun aren't in their own shadow."""
    maps = occlusion.sky_maps
    towards = maps.axes[:, 2]
    opposite = _opposite_cells(occlusion.sky_counts, points.device)
    facing = _facing_cells(occlusion.sky_counts, towards)
    walls = torch.nn.functional.normalize(-towards * towards.new_tensor([1.0, 1.0, 0.0]), dim=-1)
    rows, columns = torch.cat([~cell.visible, ~cell.visible_below], dim=1).nonzero(as_tuple=True)
    mirrored = columns >= len(towards)
    cells = torch.where(mirrored, columns - len(towards), columns)
    start, ray = points[rows], torch.cat([towards, cell.mirrored])[columns]
    below = torch.where(mirrored, cell.below[rows, cells], torch.inf)
    recorded = torch.isfinite(below)
    opposite_cell = opposite[cells]
    seen = (opposite_cell >= 0) & (cell.signed[rows, opposite_cell.clamp_min(0)] >= 0)
    index = torch.where(seen | recorded, opposite_cell.clamp_min(0), facing[cells])
    seen = seen | (cell.signed[rows, facing[cells]] >= 0)
    distance = torch.where(recorded, below, (-cell.signed[rows, cells]).clamp_min(0))
    # Bisection, along the directions that the map of their cell sees, of whether points lie in front of the surface
    # that the other map records
    which = (seen & ~recorded).nonzero()[:, 0]
    lo, hi = torch.zeros_like(distance[which]), distance[which]
    line, step = _lines_in(maps, index[which], start[which], ray[which], bias)
    width, height, first = (x[index[which]] for x in (maps.shapes[:, 1], maps.shapes[:, 0], maps.offsets))
    for _ in range(_BISECTIONS):
        middle = (lo + hi) / 2
        x, y, depth = torch.addcmul(line, middle[:, None], step).unbind(dim=-1)
        x, y = x.round(), y.round()
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        texels = first + torch.where(inside, y, 0).long() * width + torch.where(inside, x, 0).long()
        front = ~inside | (depth <= maps.depths[texels])
        lo, hi = torch.where(front, middle, lo), torch.where(front, hi, middle)
    # The surface lies between the last point found in front of it and the first one found behind it
    distance, before = distance.index_put((which,), hi), distance.index_put((which,), lo)
    found = (seen | recorded).nonzero()[:, 0]
    normals = walls[cells]
    slopes = _surface_normals(maps, index[found], start[found] + distance[found, None] * ray[found])
    toward = (slopes * ray[found]).sum(dim=-1, keepdim=True) < 0
    normals = normals.index_put((found,), torch.where(toward, slopes, normals[found]))
    texel = maps.texels[cells]
    radiance = surfaces(start + (before - texel).clamp_min(0)[:, None] * ray, normals)
    depth = height_fog_optical_depth(fog, start, ray, distance)[:, None] * extinction
    return rows, columns, radiance * torch.exp(-depth)


def _band_one_hot(counts: Sequence[int], dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    """Band of each cell of the sky, one-hot encoded, of shape (k, b), which is kept for the next frames."""
    return _cached_band_one_hot(tuple(counts), dtype, str(torch.device(device or "cpu")))


@lru_cache(maxsize=8)
def _cached_band_one_hot(counts: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    bands = torch.repeat_interleave(torch.arange(len(counts), device=device), torch.as_tensor(counts, device=device))
    return torch.nn.functional.one_hot(bands, len(counts)).to(dtype)


def _band_weights(
    edges: Sequence[float],
    counts: Sequence[int],
    g: float,
    directions: torch.Tensor,
    lit: torch.Tensor,
    lit_below: torch.Tensor,
    shaded: Sequence[torch.Tensor] = (),
    profile: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase function integrated over the visible cells of each band from points along rays, given the visibility of
    each cell, of shape (r, m, k), and over all the cells of each band, above and below the horizon, and over the
    visible cells below the horizon whose light starts on the ground in the shadow of each sun, given whether it does,
    of shape (r, m, k) for each sun, as the fields of :class:`Shadows` of the same names, each cell weighted by the
    relative radiance arriving from it if given, of shape (2, k), see :func:`cell_profile`."""
    one_hot = _band_one_hot(counts, directions.dtype, directions.device)  # (k, b)
    phases = cell_phases(edges, counts, g, directions)  # (r, k)
    below = cell_phases(edges, counts, g, directions, below=True)
    if profile is not None:
        phases, below = phases * profile[0].to(phases), below * profile[1].to(below)
    weighted_below = below[:, :, None] * one_hot
    shadowed = [torch.bmm(bits, weighted_below) for bits in shaded]
    return (
        torch.bmm(lit, phases[:, :, None] * one_hot),
        phases @ one_hot,
        torch.bmm(lit_below, weighted_below),
        below @ one_hot,
        torch.stack(shadowed, dim=2) if shadowed else lit.new_zeros(*lit.shape[:2], 0, len(counts)),
    )


def _arriving(
    radiance: torch.Tensor, heights: torch.Tensor, weights: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Light scattered more than once arriving at samples along rays from the visible cells of the sky, and from all
    of them, given the radiance arriving from each band at each height, of shape (h, 2, b), the index of the height of
    each sample, of shape (r, m), and the weights of the cells of each band, see :func:`_band_weights`."""
    up, down = radiance[heights, 0], radiance[heights, 1]  # (r, m, b)
    visible, total, visible_below, total_below = weights[:4]
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
    ground_shares: Sequence[torch.Tensor | None] | None = None,
    reflected: torch.Tensor | None = None,
    profile: torch.Tensor | None = None,
    ground_sky_share: torch.Tensor | None = None,
) -> torch.Tensor:
    """Light from suns, the sky, the ground and multiple scattering, scattered towards rays through an exponential
    height fog, given where objects hide the suns and the sky along them.

    The closed-form integral of each source is weighted, along each segment where a sun is hidden, by the fraction of
    the sun that is hidden, and between consecutive samples of the sky's visibility, by the visible fraction of the
    light arriving from all cells of the sky. Each cell contributes in proportion to the phase function integrated over
    the cell, times the mean attenuation of skylight over its elevations. Light scattered more than once is occluded in
    the same way, with cells weighted by the radiance arriving from their elevations, above and below the horizon,
    averaged over azimuths, or times the relative radiance arriving from each cell, brighter towards suns, given the
    shadows' :attr:`Shadows.scattered` or a profile, and so are its variations with the azimuth from suns. Light
    reflected by the ground into the medium arrives from below the horizon, see :class:`Shadows`, with cells weighted by
    its mean attenuation over their elevations, and lacks the share of the ground's light that a sun brings where it
    starts on the ground in the sun's shadow, and the share that the sky and the glow of the medium bring times the
    share of the sky that the ground doesn't see there, next to objects, see :func:`ground_sky`. The light that objects
    reflect towards the samples, see :func:`object_reflections`, is scattered between their bounds.

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
            tabulated along rays and keyed by "sky", "ground" and "scattered", and the variations with azimuth of the
            light of each sun scattered more than once, keyed by "scattered <k>", see :func:`tabulate_along_rays
            <visionsim.medium.scattering.tabulate_along_rays>` and :func:`azimuthal_sources
            <visionsim.medium.scattering.azimuthal_sources>`.
        scattering (ScatteringTable | None, optional): Light scattered more than once, needed to occlude the
            "scattered" tables, which are otherwise not occluded. Defaults to None.
        memory (float, optional): Rough bound, in bytes, on the memory used by intermediate values, which are computed
            for as many rays at once as fit. Defaults to 1 GiB.
        ground_shares (Sequence[torch.Tensor | None] | None, optional): For each of ``suns``, the share of the
            radiance of the ground that the sun brings, which the ground lacks in its shadow, of shape (c',), see
            :func:`ground_shares <visionsim.medium.surfaces.ground_shares>`. Defaults to None, i.e. the shadows of suns
            don't darken the ground.
        reflected (torch.Tensor | None, optional): Light that objects reflect towards the samples of the shadows, which
            the medium scatters towards the camera per unit of optical depth, of shape (n, m, c') or (n, m, 1), see
            :func:`object_reflections`. Defaults to None.
        profile (torch.Tensor | None, optional): Radiance of light scattered at least once arriving from each cell of
            the sky relative to its band's, of shape (2, k), see :func:`cell_profile`, by which cells that hide light
            scattered more than once are weighted when they are weighted again for a medium of another anisotropy than
            the shadows were traced with. Defaults to None, i.e. that of the shadows' :attr:`Shadows.scattered`, if
            any. For media of the same anisotropy, the sums of the shadows are used.
        ground_sky_share (torch.Tensor | None, optional): Share of the radiance of the ground that the sky and the glow
            of the medium bring, i.e. all of it but the shares of suns, of shape (c',), which the ground lacks where it
            doesn't see the sky, given the shadows' :attr:`Shadows.unseen_weights_below`. For media of another
            anisotropy than the shadows were traced with, the share of each band of the cells mirrored below the
            horizon whose light starts on ground that doesn't see the sky is taken to be the same. Defaults to None,
            i.e. the ground sees the whole sky.

    Raises:
        ValueError: raised if the shadows were traced along a different number of rays, or without the visibility of
            each cell for a medium of another anisotropy.

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

    def azimuths(table: ElevationTable, rays: torch.Tensor | slice) -> torch.Tensor | None:
        # Tables of light that varies with the azimuth of rays from a sun's need it
        return None if table.towards is None else relative_azimuths(directions[rays], table.towards)

    whole = [lookup(table, v_z[:, 0], full_depth, azimuths(table, slice(None))) for table in tables.values()]
    total = total + sum(whole[1:], whole[0]) if whole else total
    edges, counts = shadows.sky_edges, shadows.sky_counts
    cells, samples = sum(counts), shadows.middles.shape[1]
    # Light scattered more than once, including its variations with the azimuth from suns ("scattered <k>"), is
    # occluded alike
    occluded = {
        name: table
        for name, table in tables.items()
        if name in ("sky", "ground") or (name.startswith("scattered") and scattering)
    }
    if not (occluded or reflected is not None) or not shadows.middles.numel():
        return total

    # The weights of cells only need to be recomputed for media of another anisotropy than the shadows were traced with
    reweight = abs(medium.anisotropy - shadows.anisotropy) > 1e-12
    if reweight and shadows.visible.shape[-1] * 8 < cells:
        raise ValueError("Shadows traced without the visibility of each cell only serve media of the same anisotropy.")
    if profile is None and shadows.scattered is not None:
        profile = shadows.scattered.profile
    radiance = band_radiance(scattering, edges).sum(dim=-1) if "scattered" in tables and scattering else None
    # Suns whose shadows darken the ground, by their index among the traced ones
    darkening = []
    for (towards, _), share in zip(suns, ground_shares or [None] * len(suns)):
        traced_index = [j for j, s in enumerate(shadows.suns) if float(s.direction.to(towards) @ towards) >= 0.9999]
        if share is not None and traced_index and shadows.shadowed_weights_below is not None:
            darkening.append((traced_index[0], share.to(**kwargs)))
    depth_ground = beta * fog.density * math.exp(-(shadows.ground_height - fog.base_height) / fog.falloff) * fog.falloff
    # Light of the ground scattered once on its way up, which lacks the light of suns where it starts on the ground in
    # their shadows, from below the horizon, weighted alike over channels, of shape (h, b) for each darkening sun
    # The share of the ground's light that the sky and the glow of the medium bring, which the ground lacks where it
    # doesn't see the sky, next to objects
    sky_share = None
    if ground_sky_share is not None and shadows.unseen_weights_below is not None:
        sky_share = ground_sky_share.to(**kwargs)
    ground_glow, sky_glow = [], None
    if radiance is not None and scattering is not None and scattering.from_ground is not None:
        below_glow = band_radiance(scattering, edges, gathered=scattering.from_ground)[:, 1]  # (h, b, c)
        ground_glow = [(index, (below_glow * share).sum(dim=-1)) for index, share in darkening]
        sky_glow = (below_glow * sky_share).sum(dim=-1) if sky_share is not None else None
    if reflected is not None:
        total = total.expand(n, max(total.shape[-1], reflected.shape[-1])).clone()

    for rays in _chunks(n, int(memory // (samples * (cells * 48 if reweight else len(counts) * 128)))):
        bounds, middle = shadows.bounds[rays], shadows.middles[rays]
        weights: tuple[torch.Tensor, ...]
        # Weights of the cells that hide light scattered more than once, by the radiance arriving from each if known
        scattered_weights: tuple[torch.Tensor, ...] | None = None
        if reweight:
            lit, lit_below = (
                _unpack(bits[rays], cells).to(directions.dtype) for bits in (shadows.visible, shadows.visible_below)
            )
            shaded = [_unpack(bits[rays], cells).to(directions.dtype) for bits in shadows.shadowed_below]
            weights = _band_weights(edges, counts, medium.anisotropy, directions[rays], lit, lit_below, shaded)
            if profile is not None and radiance is not None:
                args = (edges, counts, medium.anisotropy, directions[rays], lit, lit_below)
                scattered_weights = _band_weights(*args, profile=profile)[:4]
        else:
            weights = tuple(
                getattr(shadows, name)[rays]
                for name in ("visible_weights", "weights", "visible_weights_below", "weights_below")
            )
            if shadows.shadowed_weights_below is not None:
                weights = (*weights, shadows.shadowed_weights_below[rays])
            if shadows.scattered is not None and radiance is not None:
                scattered_weights = tuple(field[rays] for field in shadows.scattered[1:])
        visible, whole_band, visible_below, whole_below = weights[:4]
        unseen = None
        if sky_share is not None:
            assert shadows.unseen_weights_below is not None
            unseen = shadows.unseen_weights_below[rays]
            if reweight:
                # The share of each band's visible cells whose light starts on ground that doesn't see the sky hardly
                # depends on how the phase function weighs them
                unseen = unseen * _ratio(visible_below, shadows.visible_weights_below[rays])
        # Optical depth of the fog above each sample, per channel
        above = fog.density * torch.exp(-(origin_z + middle * v_z[rays] - fog.base_height) / fog.falloff)
        above = (above * fog.falloff)[..., None] * beta  # (r, m, c)
        boundaries = height_fog_optical_depth(fog, origin, directions[rays, None, :], bounds)[..., None] * beta
        if reflected is not None:
            # Light that objects reflect, scattered between the bounds of each sample
            scattered = -torch.exp(-boundaries).diff(dim=1)  # (r, m, c)
            total[rays] = total[rays] + (reflected[rays] * scattered).sum(dim=1)

        ratios = {}
        if "sky" in occluded:
            attenuation = band_attenuation(edges, above)  # (r, m, c, b)
            visible_sky = (attenuation * visible[:, :, None]).sum(dim=-1)
            ratios["sky"] = _ratio(visible_sky, (attenuation * whole_band[:, None, None]).sum(dim=-1))
        if "ground" in occluded:
            # Light reflected by the ground arrives from below the horizon, attenuated by the medium beneath, where
            # objects don't hide it, and the ground lacks the light of suns in their shadows, and of the sky that
            # objects hide from it
            beneath = band_attenuation(edges, (depth_ground - above).clamp_min(0))  # (r, m, c, b)
            visible_ground = (beneath * visible_below[:, :, None]).sum(dim=-1)
            for index, share in darkening:
                visible_ground = visible_ground - share * (beneath * weights[4][:, :, index, None]).sum(dim=-1)
            if unseen is not None:
                visible_ground = visible_ground - sky_share * (beneath * unseen[:, :, None]).sum(dim=-1)
            ratios["ground"] = _ratio(visible_ground, (beneath * whole_below[:, None, None]).sum(dim=-1))
        if radiance is not None and scattering is not None:
            # The cells are weighted by the radiance arriving from them summed over channels, as their visibility is
            # the same for all channels, which is much cheaper than a ratio per channel and hardly differs from it
            fh = scattering.height_index(above[..., :1])[..., 0]
            h = fh.floor().clamp(0, len(radiance) - 2).long()
            (low_visible, low_total), (high_visible, high_total) = (
                _arriving(radiance, index, scattered_weights or weights) for index in (h, h + 1)
            )
            weight = fh - h
            visible_light = torch.lerp(low_visible, high_visible, weight)
            darkened = [(glow, weights[4][:, :, index]) for index, glow in ground_glow]
            if sky_glow is not None and unseen is not None:
                darkened.append((sky_glow, unseen))
            for glow, missed in darkened:
                # Light of the ground scattered on its way up that the ground lacks, from the cells whose light starts
                # where it lacks some, given their weights, of shape (r, m, b)
                darker = torch.lerp((glow[h] * missed).sum(dim=-1), (glow[h + 1] * missed).sum(dim=-1), weight)
                visible_light = visible_light - darker
            ratios["scattered"] = _ratio(visible_light, torch.lerp(low_total, high_total, weight))[..., None]

        for name, table in occluded.items():
            ratio = ratios[name if name in ratios else "scattered"]
            angles = azimuths(table, rays)
            angles = None if angles is None else angles[:, None].expand(-1, boundaries.shape[1])
            partial = lookup(table, v_z[rays].expand(-1, boundaries.shape[1]), boundaries, angles)
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
    surfaces: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ground_shares: Sequence[torch.Tensor | None] | None = None,
    ground_sky_share: torch.Tensor | None = None,
) -> torch.Tensor:
    """Light from suns, the sky, the ground and multiple scattering, scattered towards rays through an exponential
    height fog, accounting for the shadows of the scene's objects, see :func:`trace` and :func:`shade`, with cells of
    the sky that hide light scattered more than once weighted by the radiance arriving from them, see
    :func:`cell_profile`, and, with ``multiple_scattering``, the light that objects reflect attenuated with the
    extinction reduced by the similarity relation, see :func:`object_reflections`.

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
            tabulated along rays and keyed as for :func:`shade`.
        scattering (ScatteringTable | None, optional): Light scattered more than once, needed to occlude the
            "scattered" tables, which are otherwise not occluded. Defaults to None.
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
        surfaces (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional): Radiance of the objects'
            surfaces, which reflect light towards the medium, see :func:`object_reflections`. Defaults to None.
        ground_shares (Sequence[torch.Tensor | None] | None, optional): For each of ``suns``, the share of the
            radiance of the ground that the sun brings, see :func:`shade`. Defaults to None.
        ground_sky_share (torch.Tensor | None, optional): Share of the radiance of the ground that the sky and the glow
            of the medium bring, which the ground lacks where objects hide the sky from it, see :func:`shade`.
            Defaults to None, i.e. the ground sees the whole sky.

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
        keep_cells=False,
        profile=cell_profile(scattering, occlusion.sky_edges, occlusion.sky_counts) if scattering is not None else None,
        ground_sky_occlusion=ground_sky_share is not None,
    )
    reflected = None
    if surfaces is not None:
        similarity = medium.multiple_scattering
        reflected = object_reflections(
            occlusion, medium, origin, directions, shadows, surfaces, beta, similarity, bias=bias, memory=memory
        )
    return shade(
        shadows,
        medium,
        origin,
        directions,
        distance,
        beta,
        suns,
        tables,
        scattering,
        memory,
        ground_shares,
        reflected,
        ground_sky_share=ground_sky_share,
    )
