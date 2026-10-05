from __future__ import annotations

import argparse
import collections
import functools
import inspect
import itertools
import json
import logging
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager, nullcontext
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from visionsim.types import FILE

# Import only when type checking as to not introduce
# dependency for blender. Block module typechecking.
if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Sequence
    from types import TracebackType

    import multiprocess  # type: ignore
    import multiprocess.pool  # type: ignore
    import numpy.typing as npt
    from typing_extensions import Self

    from visionsim.types import COLOR_MODES, EXR_CODECS, FILE_FORMATS, UpdateFn


# These are blender specific modules which aren't easily installed but
# are loaded in when this script is ran from blender.
try:
    import addon_utils  # type: ignore
    import bpy  # type: ignore
    import mathutils  # type: ignore
    from bpy_extras import anim_utils  # type: ignore
except ImportError:
    addon_utils = None
    bpy = None
    mathutils = None

# These are also only needed when ran from blender, but they are separated
# out as we use `bpy is None` as a quick check to see if within blender and
# these might fail independently.
try:
    from visionsim.simulate.compat import file_output_node
    from visionsim.simulate.nodes import (  # type: ignore
        colorize_indices_node_group,
        flow_preview_node_group,
        normal_preview_node_group,
        point_preview_node_group,
        vec2rgba_node_group,
    )
except ImportError:
    if bpy is not None:
        raise RuntimeError("Blender dependencies are missing, please run the post install script.")

# Nice to haves, but not always required
try:
    from rich.logging import RichHandler

    handlers: Iterable[logging.Handler] | None = [RichHandler(level="NOTSET")]
except ImportError:
    handlers = None

import numpy as np
import rpyc  # type: ignore
import rpyc.utils.registry  # type: ignore
import rpyc.utils.server  # type: ignore
from peewee import SqliteDatabase

from visionsim.simulate.schema import _DEFAULT_PRAGMAS, _MODELS, _Camera, _Data, _Frame

# Enable server-side logging
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=handlers)
server_log: logging.Logger = logging.getLogger(__name__)
server_log.setLevel(logging.INFO)

EXPOSED_PREFIX: str = "exposed_"
REGISTRY: tuple[Process, rpyc.utils.registry.UDPRegistryClient] | None = None
ITEMS_PER_SUBFOLDER: int = 1000
INDEX_PADDING: int = np.ceil(np.log10(ITEMS_PER_SUBFOLDER)).astype(int)

# Taken from blender/scripts/addons_core/node_wrangler/operators/save_viewer_image.py
FORMATS: dict[str, str] = {
    "BMP": ".bmp",
    "IRIS": ".rgb",
    "PNG": ".png",
    "JPEG": ".jpeg",
    "JPEG2000": ".jp2",
    "TARGA": ".tga",
    "CINEON": ".cin",
    "DPX": ".dpx",
    "OPEN_EXR": ".exr",
    "HDR": ".hdr",
    "TIFF": ".tif",
    "WEBP": ".webp",
}
COLOR_MODE_CHANNELS = {"BW": 1, "RGB": 3, "RGBA": 4}

SKY_ELEVATIONS: tuple[float, ...] = (0.0, 4.0, 9.0, 16.0, 26.0, 40.0, 60.0, 90.0)
"""Elevations, in degrees, at the edges of the bands into which the sky is split for shadow maps, which are narrower
near the horizon, where fog scatters most skylight towards the camera"""
SKY_CELLS: tuple[int, ...] = (24, 24, 20, 16, 12, 8, 3)
"""Number of cells into which each band of the sky is split, evenly in azimuth"""


def _sky_cell_directions(edges: Sequence[float], counts: Sequence[int]) -> npt.NDArray[np.floating]:
    """Central direction of each cell of the sky, as in :func:`visionsim.medium.occlusion.sky_cell_directions`, which
    cannot be imported from within Blender."""
    directions = []
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        mu = (low + high) / 2
        for j in range(count):
            phi = 2 * np.pi * (j + 0.5) / count
            directions.append((np.sqrt(1 - mu * mu) * np.cos(phi), np.sqrt(1 - mu * mu) * np.sin(phi), mu))
    return np.asarray(directions, dtype=float).reshape(-1, 3)


# Piecewise fit of the linear Rec.709 color of a black body, of unit luminance, used by Cycles' Blackbody node
# (`svm_math_blackbody_color_rec709` and the tables of `kernel/tables.h` in Cycles' sources)
_BLACKBODY_BOUNDS = (965.0, 1167.0, 1449.0, 1902.0, 3315.0, 6365.0)
_BLACKBODY_R = (
    (1.61919106e03, -2.05010916e-03, 5.02995757e00),
    (2.48845471e03, -1.11330907e-03, 3.22621544e00),
    (3.34143193e03, -4.86551192e-04, 1.76486769e00),
    (4.09461742e03, -1.27446582e-04, 7.25731635e-01),
    (4.67028036e03, 2.91258199e-05, 1.26703442e-01),
    (4.59509185e03, 2.87495649e-05, 1.50345020e-01),
    (3.78717450e03, 9.35907826e-06, 3.99075871e-01),
)
_BLACKBODY_G = (
    (-4.88999748e02, 6.04330754e-04, -7.55807526e-02),
    (-7.55994277e02, 3.16730098e-04, 4.78306139e-01),
    (-1.02363977e03, 1.20223470e-04, 9.36662319e-01),
    (-1.26571316e03, 4.87340896e-06, 1.27054498e00),
    (-1.42529332e03, -4.01150431e-05, 1.43972784e00),
    (-1.17554822e03, -2.16378048e-05, 1.30408023e00),
    (-5.00799571e02, -4.59832026e-06, 1.09098763e00),
)
_BLACKBODY_B = (
    (5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03),
    (2.40430366e-11, 5.55021075e-08, -1.98503712e-04, 2.89312858e-02),
    (-1.40949732e-11, 1.89878968e-07, -3.56632824e-04, 9.10767778e-02),
    (-3.61460868e-11, 2.84822009e-07, -4.93211319e-04, 1.56723440e-01),
    (-1.97075738e-11, 1.75359352e-07, -2.50542825e-04, -2.22783266e-02),
    (-1.61997957e-13, -1.64216008e-08, 3.86216271e-04, -7.38077418e-01),
    (6.72650283e-13, -2.73078809e-08, 4.24098264e-04, -7.52335691e-01),
)


def _blackbody(temperature: float) -> list[float]:
    """Linear Rec.709 color of a black body at a temperature in Kelvin, as computed by Cycles' Blackbody node."""
    if temperature >= 12000.0:
        rgb = [0.8262954810464208, 0.9945080501520986, 1.566307710274283]
    elif temperature < 800.0:
        rgb = [5.413294490189271, -0.20319390035873933, -0.0822535242887164]
    else:
        i = sum(temperature >= bound for bound in _BLACKBODY_BOUNDS)
        (r0, r1, r2), (g0, g1, g2), (b0, b1, b2, b3) = _BLACKBODY_R[i], _BLACKBODY_G[i], _BLACKBODY_B[i]
        t = temperature
        rgb = [r0 / t + r1 * t + r2, g0 / t + g1 * t + g2, ((b0 * t + b1) * t + b2) * t + b3]
    return [max(c, 0.0) for c in rgb]


EMISSIVE_LAMPS: int = 8
"""Default number of groups of emissive surfaces, which each cast shadows from a single position"""
EMISSIVE_PATCHES: int = 32
"""Default number of patches into which emissive surfaces are split, over all of their groups"""
EMISSIVE_GAP: float = 0.5
"""Distance, in meters, beyond which emissive surfaces are apart, such as those of different lamps, and only grouped
together if there are more than ``EMISSIVE_LAMPS`` of them"""
_EMISSIVE_SAMPLES = 50_000
"""Number of emissive triangles, drawn by their power, from each side of which rays are traced to find how much of their
light leaves the surfaces"""
_ESCAPE_RAYS = 8
"""Number of rays traced from each side of these triangles, spread as the light they emit"""
_ORIENTATION_TOLERANCE = 0.05
"""Relative error of the area of a patch seen from each direction, as approximated from the moments of the normals of
its faces (see `visionsim.medium.lights.projected_area`), above which it is split by orientation rather than position"""
_PATCH_SIZE = 0.5
"""Size, in meters, below which patches aren't split by position, as they only differ from a point emitter whose
distances are clamped at their radius close to them"""
_NEGLIGIBLE = 0.01
"""Share of the light of all lamps, emissive surfaces included, that the faintest emissive surfaces can make up together
and still be left out"""


def _lamp_power(lights: dict[str, Any]) -> float:
    """Total power of the point, spot and area lights of exported lighting, averaged over color channels, in W."""
    return sum(float(np.mean(lamp["power"])) for kind in ("points", "spots", "areas") for lamp in lights[kind])


def _triangles(
    corners: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Centers, unit normals and areas of triangles, given their corners of shape (t, 3, 3)."""
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = np.linalg.norm(cross, axis=-1) / 2
    return corners.mean(axis=1), cross / np.maximum(2 * areas, 1e-300)[:, None], areas


def _subdivide(
    corners: npt.NDArray[np.floating], max_area: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    """Split triangles in four, between the middles of their sides, until none is larger than ``max_area``, which keeps
    their orientation, and return them along with the index of the triangle each comes from."""
    sources = np.arange(len(corners))
    done, kept = [corners[:0]], [sources[:0]]
    while len(corners):
        large = _triangles(corners)[2] > max_area
        done.append(corners[~large])
        kept.append(sources[~large])
        a, b, c = corners[large, 0], corners[large, 1], corners[large, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        corners = np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        sources = np.tile(sources[large], 4)
    return np.concatenate(done), np.concatenate(kept)


def _separate(
    lower: npt.NDArray[np.floating], upper: npt.NDArray[np.floating], gap: float
) -> list[npt.NDArray[np.integer]]:
    """Split items into groups that are further than ``gap`` apart along some axis, given the corners of their boxes,
    each of shape (n, 3)."""
    pending, groups = [np.arange(len(lower))], []
    while pending:
        group = pending.pop()
        for axis in range(3):
            order = group[np.argsort(lower[group, axis], kind="stable")]
            reach = np.maximum.accumulate(upper[order, axis])
            apart = np.flatnonzero(lower[order[1:], axis] > reach[:-1] + gap) + 1
            if len(apart):
                pending.extend(np.split(order, apart))
                break
        else:
            groups.append(group)
    return groups


def _gather(
    centers: npt.NDArray[np.floating], weights: npt.NDArray[np.floating], count: int
) -> list[npt.NDArray[np.integer]]:
    """Gather items into at most ``count`` groups, by splitting the box around their centers in two halves along its
    longest side, the group of the largest weight times size first."""

    def score(group: npt.NDArray[np.integer]) -> float:
        return float(weights[group].sum() * np.ptp(centers[group], axis=0).max()) if len(group) > 1 else 0.0

    groups = [np.arange(len(weights))]
    scores = [score(groups[0])]
    while len(groups) < count and max(scores) > 0:
        best = int(np.argmax(scores))
        group = groups[best]
        axis = int(np.argmax(np.ptp(centers[group], axis=0)))
        values = centers[group, axis]
        low = values < (values.min() + values.max()) / 2
        groups[best : best + 1] = parts = [group[low], group[~low]]
        scores[best : best + 1] = [score(part) for part in parts]
    return groups


def _orientation(
    normals: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    front: npt.NDArray[np.floating],
    back: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Orientation and facing of faces, see :class:`EmissiveSurface <visionsim.medium.model.EmissiveSurface>`, given
    their unit normals, weights, and the fraction of the light of their front and back that leaves the surfaces."""
    share = weights / weights.sum()
    orientation = np.einsum("k,ki,kj->ij", share * (front + back) / 2, normals, normals)
    return orientation, (share * (front - back) / 2) @ normals


def _fibonacci(count: int) -> npt.NDArray[np.floating]:
    """Unit directions spread evenly over the sphere, of shape (count, 3)."""
    i = np.arange(count) + 0.5
    z = 1 - 2 * i / count
    phi = np.pi * (3 - np.sqrt(5)) * i
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=-1)


_ERROR_DIRECTIONS = _fibonacci(64)
"""Directions in which the area of patches seen from them is checked"""


def _orientation_error(
    normals: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    front: npt.NDArray[np.floating],
    back: npt.NDArray[np.floating],
) -> float:
    """Relative error of the area of faces seen from each direction, as approximated from their orientation, see
    :func:`visionsim.medium.lights.projected_area`."""
    directions = _ERROR_DIRECTIONS
    cosines = directions @ normals.T
    seen = np.maximum(cosines, 0) * front + np.maximum(-cosines, 0) * back
    exact = seen @ weights / weights.sum()
    orientation, facing = _orientation(normals, weights, front, back)
    even = np.sqrt(np.maximum(np.einsum("di,ij,dj->d", directions, orientation, directions), 0))
    scale = np.trace(orientation) / (2 * even.mean()) if even.mean() > 0 else 0.0
    approximation = np.maximum(scale * even + directions @ facing, 0)
    return float(np.sqrt(np.mean((approximation - exact) ** 2)) / max(float(exact.mean()), 1e-300))


def _split_by_orientation(
    normals: npt.NDArray[np.floating], weights: npt.NDArray[np.floating]
) -> npt.NDArray[np.bool_] | None:
    """Split faces in two groups of alike orientations, by clustering ``n nᵀ`` in two, which ignores their sides, or
    None if they are all alike."""
    x, y, z = normals.T
    features = np.stack([x * x, y * y, z * z, np.sqrt(2) * x * y, np.sqrt(2) * x * z, np.sqrt(2) * y * z], axis=-1)
    first = features[np.argmax(weights)]
    means = np.stack([first, features[np.argmax(((features - first) ** 2).sum(axis=-1))]])
    second = np.zeros(len(normals), dtype=bool)
    for _ in range(16):
        second = ((features - means[1]) ** 2).sum(axis=-1) < ((features - means[0]) ** 2).sum(axis=-1)
        if second.all() or not second.any():
            return None
        means = np.stack(
            [np.average(features[side], axis=0, weights=weights[side] + 1e-300) for side in (~second, second)]
        )
    return second


def _escape(
    centers: npt.NDArray[np.floating],
    normals: npt.NDArray[np.floating],
    trace: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.bool_]],
    rng: np.random.Generator,
    rays: int = _ESCAPE_RAYS,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fraction of the light that leaves the front and the back of triangles without hitting any other, by tracing rays
    spread as the light they emit, i.e. as the cosine of the angle to their normal, in directions stratified in
    angle."""
    k = np.arange(rays)
    sine, turn = np.sqrt((k + 0.5) / rays), 2 * np.pi * ((k * (np.sqrt(5) - 1) / 2) % 1)
    angle = turn[None] + rng.uniform(0, 2 * np.pi, (len(centers), 1))
    helper = np.where((np.abs(normals[:, 0]) < 0.9)[:, None], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    tangent = np.cross(normals, helper)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(normals, tangent)
    # Rays start just off the triangles, far enough for the single precision of the traced triangles
    offset = 1e-5 * (10 + np.abs(centers).max(axis=-1, keepdims=True))
    fractions = []
    for side in (1.0, -1.0):
        directions = (
            (sine * np.cos(angle))[..., None] * tangent[:, None]
            + (sine * np.sin(angle))[..., None] * bitangent[:, None]
            + np.sqrt(1 - sine * sine)[None, :, None] * side * normals[:, None]
        )
        origins = np.broadcast_to((centers + side * offset * normals)[:, None], directions.shape)
        hits = trace(origins.reshape(-1, 3), directions.reshape(-1, 3)).reshape(len(centers), rays)
        fractions.append(1 - hits.mean(axis=-1))
    return fractions[0], fractions[1]


def _emissive_lamps(
    corners: npt.NDArray[np.floating],
    radiance: npt.NDArray[np.floating],
    trace: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.bool_]],
    lamps: int = EMISSIVE_LAMPS,
    patches: int = EMISSIVE_PATCHES,
    gap: float = EMISSIVE_GAP,
    samples: int = _EMISSIVE_SAMPLES,
    seed: int = 0,
    other_power: float = 0.0,
) -> list[dict[str, Any]]:
    """Group emissive triangles into lamps, which each cast shadows from a single position, split into patches.

    Triangles further than ``gap`` apart belong to different lamps, as long as there are at most ``lamps`` of them,
    otherwise lamps are gathered by position, after leaving out the faintest surfaces, as long as they make up at most
    1% of the light of all lamps, including other lamps of power ``other_power``, assuming that both sides of the
    surfaces emit. Lamps are then split into ``patches`` patches in total, by repeatedly
    splitting the patch of the largest power times size in two: by the orientation of its faces if the area it shows in
    each direction isn't well approximated from their moments, and otherwise in two halves of equal power along its
    longest side, after splitting triangles larger than a fraction of the patches. The light that leaves each side of
    a sample of the triangles is found by tracing rays from them, as triangles block each other's light.

    Args:
        corners (npt.NDArray[np.floating]): World-space corners of the triangles, of shape (t, 3, 3).
        radiance (npt.NDArray[np.floating]): Radiance of the triangles, of shape (t, c).
        trace (Callable): Function that tells whether rays hit any of the triangles, given their origins and unit
            directions, each of shape (r, 3).
        lamps (int, optional): Maximum number of lamps. Defaults to :data:`EMISSIVE_LAMPS`.
        patches (int, optional): Number of patches, over all lamps, which each have at least one. Defaults to
            :data:`EMISSIVE_PATCHES`.
        gap (float, optional): Distance beyond which triangles are apart. Defaults to :data:`EMISSIVE_GAP`.
        samples (int, optional): Number of triangles from which rays are traced. Defaults to 50,000.
        seed (int, optional): Seed of the random sampling of triangles and rays. Defaults to 0.
        other_power (float, optional): Power of the other lamps of the scene, in W. Defaults to 0.0.

    Returns:
        list[dict[str, Any]]: Lamps, following the schema of :class:`EmissiveSurface
        <visionsim.medium.model.EmissiveSurface>`.
    """
    areas = _triangles(corners)[2]
    useful = (areas > 0) & (radiance.mean(axis=-1) > 0)
    if not useful.any():
        return []
    fine, source = _subdivide(corners[useful], float(areas[useful].sum()) / (4 * patches))
    centers, normals, areas = _triangles(fine)
    radiance = np.maximum(radiance[useful][source], 0)
    power = areas * radiance.mean(axis=-1)
    # Covariance of the points of each triangle around its center
    corners_offset = fine - centers[:, None]
    spread = np.einsum("tki,tkj->tij", corners_offset, corners_offset) / 12

    # Surfaces apart from each other are different lamps, the faintest of which are left out, and which are gathered by
    # position if there are too many
    components = _separate(fine.min(axis=1), fine.max(axis=1), gap)
    totals = np.asarray([power[c].sum() for c in components])
    order = np.argsort(totals, kind="stable")
    left_out = np.cumsum(2 * np.pi * totals[order]) <= _NEGLIGIBLE * (2 * np.pi * totals.sum() + other_power)
    components = [components[i] for i in sorted(order[~left_out])]
    if not components:
        return []
    totals = np.asarray([power[c].sum() for c in components])
    middles = np.stack([np.average(centers[c], axis=0, weights=power[c]) for c in components])
    lamp_groups = [np.concatenate([components[i] for i in g]) for g in _gather(middles, totals, lamps)]

    # Light that leaves each side of a sample of the triangles, drawn by their power
    rng = np.random.default_rng(seed)
    if len(power) <= samples:
        sampled, weights = np.arange(len(power)), power
    else:
        sampled, counts = np.unique(rng.choice(len(power), size=samples, p=power / power.sum()), return_counts=True)
        weights = counts.astype(float)
    front, back = _escape(centers[sampled], normals[sampled], trace, rng)
    slots = np.full(len(power), -1)
    slots[sampled] = np.arange(len(sampled))

    def sample(group: npt.NDArray[np.integer]) -> tuple[npt.NDArray[np.floating], ...]:
        found = slots[group]
        found = found[found >= 0]
        return normals[sampled[found]], weights[found], front[found], back[found]

    # Patches are split, from lamps, to reduce both their size and how poorly the moments of the normals of their faces
    # approximate the area they show in each direction
    reach = max(float(np.ptp(centers, axis=0).max()), 1e-9)

    def cost(group: npt.NDArray[np.integer]) -> tuple[float, float]:
        found = sample(group)
        error = _orientation_error(*found) if len(found[0]) > 1 else 0.0
        size = float(np.ptp(centers[group], axis=0).max()) if len(group) > 1 else 0.0
        size = size if size > _PATCH_SIZE else 0.0
        return float(power[group].sum()) * (size / reach + max(error - _ORIENTATION_TOLERANCE, 0.0)), error

    def halves(group: npt.NDArray[np.integer]) -> list[npt.NDArray[np.integer]]:
        axis = int(np.argmax(np.ptp(centers[group], axis=0)))
        order = group[np.argsort(centers[group, axis], kind="stable")]
        cumulative = np.cumsum(power[order])
        # Split after the triangle whose cumulative power is closest to half of it
        split = 1 + int(np.argmin(np.abs(cumulative[:-1] - cumulative[-1] / 2)))
        return [order[:split], order[split:]]

    groups = list(lamp_groups)
    costs = [cost(group) for group in groups]
    while len(groups) < patches and max(c for c, _ in costs) > 0:
        best = int(np.argmax([c for c, _ in costs]))
        group, (_, error) = groups[best], costs[best]
        if len(group) < 2:
            costs[best] = (0.0, error)
            continue
        candidates = [halves(group)]
        if (
            error > _ORIENTATION_TOLERANCE
            and (second := _split_by_orientation(normals[group], power[group])) is not None
        ):
            candidates.append([group[~second], group[second]])
        options = [(parts, [cost(part) for part in parts]) for parts in candidates]
        parts, part_costs = min(options, key=lambda option: sum(c for c, _ in option[1]))
        groups[best : best + 1] = parts
        costs[best : best + 1] = part_costs

    lamp_of = np.empty(len(power), dtype=int)
    for k, group in enumerate(lamp_groups):
        lamp_of[group] = k
    surfaces: list[list[dict[str, Any]]] = [[] for _ in lamp_groups]
    for group in groups:
        found_normals, found_weights, found_front, found_back = sample(group)
        if not len(found_normals):
            # Faint patches may not have any sampled triangle, of which the brightest are traced instead
            picked = group[np.argsort(power[group])[::-1][:64]]
            found_front, found_back = _escape(centers[picked], normals[picked], trace, rng)
            found_normals, found_weights = normals[picked], power[picked]
        orientation, facing = _orientation(found_normals, found_weights, found_front, found_back)
        if np.trace(orientation) <= 1e-12 and np.abs(facing).max() <= 1e-12:
            # No light leaves these faces, e.g. those enclosed by other emissive surfaces
            continue
        area = float(areas[group].sum())
        position = np.average(centers[group], axis=0, weights=power[group])
        # Covariance of the points of the patch around its center, from those of its triangles
        offsets = centers[group] - position
        covariance = np.einsum("t,tij->ij", power[group], spread[group] + offsets[:, :, None] * offsets[:, None, :])
        covariance /= power[group].sum()
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        surfaces[lamp_of[group[0]]].append(
            {
                "position": position.tolist(),
                "area": area,
                "radiance": ((radiance[group] * areas[group, None]).sum(axis=0) / area).tolist(),
                "orientation": [float(orientation[i, j]) for i, j in pairs],
                "facing": facing.tolist(),
                "spread": [float(covariance[i, j]) for i, j in pairs],
            }
        )

    result = []
    around = _fibonacci(16)
    for group, lamp_patches in zip(lamp_groups, surfaces):
        if not lamp_patches:
            continue
        # Shadows are cast from the lamp's center if the lamp surrounds it, as for a ball, which maps see through.
        # Otherwise the center could lie within some other object, e.g. at the center of a ring of light, and shadows
        # are cast from the center of the triangle nearest to it
        position = np.average(centers[group], axis=0, weights=power[group])
        if not trace(np.broadcast_to(position, around.shape), around).all():
            position = centers[group[np.argmin(np.linalg.norm(centers[group] - position, axis=-1))]]
        result.append(
            {
                "position": position.tolist(),
                **{
                    name: [patch[key] for patch in lamp_patches]
                    for name, key in (
                        ("positions", "position"),
                        ("areas", "area"),
                        ("radiance", "radiance"),
                        ("orientation", "orientation"),
                        ("facing", "facing"),
                        ("spread", "spread"),
                    )
                },
            }
        )
    return result


def require_connected_client(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator which ensures a client is connected.

    Args:
        func (Callable[..., Any]): Function to decorate

    Raises:
        RuntimeError: raised if client is not connected.

    Returns:
        Callable[..., Any]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderClient, *args: Any, **kwargs: Any) -> Any:
        if self.conn is None:
            raise RuntimeError(
                f"'BlenderClient' must be connected to a server instance before calling '{func.__name__}'"
            )
        return func(self, *args, **kwargs)

    return _decorator


def require_connected_clients(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator which ensures all clients are connected.

    Args:
        func (Callable[..., Any]): Function to decorate

    Raises:
        RuntimeError: if at least one client is not connected.

    Returns:
        Callable[..., Any]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderClients, *args: Any, **kwargs: Any) -> Any:
        if any(c.conn is None for c in self):
            raise RuntimeError(
                f"All client instances in 'BlenderClients' must be connected before calling '{func.__name__}'"
            )
        return func(self, *args, **kwargs)

    return _decorator


def require_initialized_service(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator which ensures the render service was initialized.

    Args:
        func (Callable[..., Any]): Function to decorate

    Raises:
        RuntimeError: raised if :meth:`client.initialize <BlenderService.exposed_initialize>` has not been previously called.

    Returns:
        Callable[..., Any]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderService, *args: Any, **kwargs: Any) -> Any:
        if not self._initialized:
            raise RuntimeError(f"'BlenderService' must be initialized before calling '{func.__name__}'")
        return func(self, *args, **kwargs)

    return _decorator


def validate_camera_moved(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator which emits a warning if the camera was not moved.

    Args:
        func (Callable[..., Any]): Function to decorate

    Returns:
        Callable[..., Any]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderService, *args: Any, **kwargs: Any) -> Any:
        prev_matrix = np.array(self.camera.matrix_world.copy())
        retval = func(self, *args, **kwargs)
        post_matrix = np.array(self.camera.matrix_world.copy())

        if np.allclose(prev_matrix, post_matrix):
            self.log.warning("Camera has not moved as intended, perhaps it is still bound by parent or animation?")
        return retval

    return _decorator


class BlenderServer(rpyc.utils.server.Server):
    """Expose a :class:`BlenderService` to the outside world via RPCs.

    Example:
        Once created, it can be started, which will block and await for an external connection from a :class:`BlenderClient`:

        .. code-block:: python

            server = BlenderServer()
            server.start()

        However, this needs to be called within blender's runtime. Instead one can use :meth:`BlenderServer.spawn`
        to spawn one or more blender instances, each with their own server.

    """

    def __init__(
        self,
        hostname: bytes | str | None = None,
        port: bytes | str | int | None = 0,
        service: type[BlenderService] | None = None,
        extra_config: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize a :class:`BlenderServer` instance

        Args:
            hostname (bytes | str | None, optional): the host to bind to. By default, the 'wildcard address' is used
                to listen on all interfaces. If not properly secured, the server can receive traffic from
                unintended or even malicious sources. Defaults to None (wildcard).
            port (bytes | str | int | None, optional): the TCP port to bind to. Defaults to 0 (bind to a random open port).
            service (type[BlenderService], optional): the service to expose, must be a :class:`BlenderService` subclass. Defaults to :class:`BlenderService`.
            extra_config (dict, optional): the configuration dictionary that is passed to the RPyC connection.
                Defaults to ``{"allow_all_attrs": True, "allow_setattr": True}``.
            **kwargs: Additional keyword arguments which are passed to the
                `rpyc.utils.server.Server <https://rpyc.readthedocs.io/en/latest/api/utils_server.html#rpyc.utils.server.Server>`_ constructor.

        Raises:
            RuntimeError: a :class:`BlenderServer` needs to be instantiated from within a blender instance.
            ValueError: the exposed service must be :class:`BlenderService` or subclass.
        """
        if bpy is None:
            raise RuntimeError(f"{type(self).__name__} needs to be instantiated from within blender's python runtime.")
        if service and not issubclass(service, BlenderService):
            raise ValueError("Parameter 'service' must be 'BlenderService' or subclass.")

        super().__init__(
            service or BlenderService,
            hostname=hostname,
            port=port,
            protocol_config=rpyc.core.protocol.DEFAULT_CONFIG | (extra_config or {"allow_all_attrs": True}),
            auto_register=True,
            **kwargs,
        )
        server_log.info(f"Started listening on {self.host}:{self.port}")

    @contextmanager
    @staticmethod
    def spawn(
        jobs: int = 1,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = subprocess.DEVNULL,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Generator[tuple[list[subprocess.Popen], list[tuple[str, int]]]]:
        """Spawn one or more blender instances and start a :class:`BlenderServer` in each.

        This is roughly equivalent to calling ``blender -b --python blender.py`` in many subprocesses,
        where ``blender.py`` initializes and ``start``\\s a server instance. Proper logging and termination of
        these processes is also taken care of.

        Note:
            The returned processes and connection settings are not guaranteed to be in the same order.

        Warning:
            If ``log`` is a file handle or descriptor, such as redirecting Blender logs to subprocess.STDOUT,
            the writing process might get overwhelmed which can cause silent errors, dropped logs and locked
            processes. It is thus not recommended for long render jobs to set ``log`` to anything but DEVNULL
            or a directory.

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for ``timeout``
                (in seconds) before giving up. If negative, a port will be randomly selected and assigned to the
                spawned server, bypassing the need for discovery and timeouts. Note that when a port is assigned
                this context manager will immediately yield, even if the server is not yet ready to accept
                incoming connections. Defaults to assigning a port to spawned server (-1 seconds).
            log (str | os.PathLike | FILE | tuple[FILE, FILE], optional): path to log directory, file handle,
                descriptor or tuple thereof. Stdout and stderr will be captured and saved if supplied.
                Defaults to subprocess.DEVNULL for both stdout/stderr.
            autoexec (bool, optional): if true, allow execution of any embedded python scripts within blender.
                For more, see blender's CLI documentation. Defaults to False.
            executable (str | os.PathLike | None, optional): path to Blender's executable. Defaults to looking
                for blender on $PATH, but is useful when targeting a specific blender install, or when it's installed
                via a package manager such as flatpak. Setting it to "flatpak run --die-with-parent org.blender.Blender"
                might be required when using flatpaks. Defaults to None (system PATH).

        Raises:
            TimeoutError: raise if unable to discover spawned servers in ``timeout`` seconds and kill any spawned processes.

        Yields:
            Generator[tuple[list[subprocess.Popen], list[tuple[str, int]]]]:  A tuple containing:
                - list[subprocess.Popen]: List of ``subprocess.Popen`` corresponding to all spawned servers.
                - list[tuple[str, int]]: List of connection setting for each server, where each element is a (hostname, port) tuple.
        """
        # Note import here as this is a dependency only on the client-side
        from ephemeral_port_reserve import reserve as port_reserve  # type: ignore

        @contextmanager
        def terminate_jobs(procs):
            try:
                yield
            except KeyboardInterrupt:
                server_log.warning("Shutting down gracefully...")
            finally:
                for p in procs:
                    # We need to send two CTRL+C events to blender to kill it, but not on windows
                    if platform.system() != "Windows":
                        p.send_signal(signal.SIGINT)
                        p.send_signal(signal.SIGINT)

                for p in procs:
                    # Ensure process is killed if CTRL+C failed
                    p.terminate()

        BlenderServer.spawn_registry()
        existing = BlenderServer.discover() if timeout > 0 else []
        procs, ports = [], []

        if isinstance(log, (str, os.PathLike)):
            log_dir_path = Path(log).expanduser().resolve()
            log_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            log_dir_path = None

        with ExitStack() as stack:
            for i in range(jobs):
                port = port_reserve("localhost") if timeout < 0 else 0
                autoexec_cmd = "--enable-autoexec" if autoexec else "--disable-autoexec"
                cmd = shlex.split(
                    f"{executable or 'blender'} -b {autoexec_cmd} --python-use-system-env "
                    f"--python {Path(__file__).as_posix()} -- --port {port}"
                )

                if log_dir_path:
                    (log_dir_path / f"job{i:03}").mkdir(parents=True, exist_ok=True)
                    stdout = stack.enter_context(open(log_dir_path / f"job{i:03}" / "stdout.log", "w"))
                    stderr = stack.enter_context(open(log_dir_path / f"job{i:03}" / "stderr.log", "w"))
                else:
                    stdout, stderr = log if isinstance(log, tuple) else (log, subprocess.STDOUT)
                proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, universal_newlines=True)
                procs.append(proc)
                ports.append(port)
            stack.enter_context(terminate_jobs(procs))

            if timeout > 0:
                start = time.time()

                while True:
                    if (time.time() - start) > timeout:
                        # Terminate all procs and close fds
                        stack.close()
                        raise TimeoutError("Unable to spawn and discover server(s) in alloted time.")
                    if len(conns := set(BlenderServer.discover()) - set(existing)) == jobs:
                        break
                    time.sleep(0.1)
            else:
                conns = {("localhost", p) for p in ports}

            yield (procs, list(conns))

    @staticmethod
    def spawn_registry() -> tuple[Process, rpyc.utils.registry.UDPRegistryClient]:
        """Spawn a registry server and client to aid in server discovery, or return cached result.
        While this method can be called directly, it will be invoked automatically by :meth:`discover` and :meth:`spawn`.

        Returns:
            tuple[Process, rpyc.utils.registry.UDPRegistryClient]: A tuple containing:
                - Process: process running the global registry server,
                - rpyc.utils.registry.UDPRegistryClient: global registry client
        """
        global REGISTRY

        if not REGISTRY or not REGISTRY[0].is_alive():
            registry = Process(target=BlenderServer._launch_registry, daemon=True)
            client = rpyc.utils.registry.UDPRegistryClient()
            registry.start()
            REGISTRY = (registry, client)
        return REGISTRY

    @staticmethod
    def _launch_registry():
        try:
            registry = rpyc.utils.registry.UDPRegistryServer()
            registry.start()
        except OSError:
            # Note: Address is likely already in use, meaning there's
            #   already a spawned registry in another thread/process which
            #   we should be able to use. No need to re-spawn one then.
            pass

    @staticmethod
    def discover() -> list[tuple[str, int]]:
        """Discover any :class:`BlenderServer`\\s that are already running and return their connection parameters.

        Note:
            A discoverable server might already be in use and can refuse connection attempts.

        Returns:
            list[tuple[str, int]]: List of connection setting for each server, where each element is a (hostname, port) tuple.
        """
        _, client = BlenderServer.spawn_registry()
        try:
            return list(cast(tuple, client.discover("BLENDER")))
        except (OSError, ValueError, EOFError) as e:
            server_log.warning(f"Failed to discover Blender servers: {e}")
            return []

    def _accept_method(self, sock: socket.socket) -> None:
        # Accept a single connection, and block here until it closes. Any other incoming
        # connections will stall, and run out the `sync_request_timeout` while attempting to connect.
        return self._authenticate_and_serve_client(sock)


class BlenderService(rpyc.Service):
    """Server-side API to interact with blender and render novel views.

    Most of the methods of a :class:`BlenderClient` instance are remote procedure calls to
    a connected blender service. These methods are prefixed by ``exposed_``.
    """

    # Note: This alias is used when discovering servers using the registry.
    #   By default the service name is extracted from the class name, so here
    #   it would be `blender` anyways, but we define an alias here to support
    #   subclasses which might be named differently and not discovered.
    ALIASES: tuple[str] = ("BLENDER",)

    def __init__(self) -> None:
        """Initialize render service.

        Raises:
            RuntimeError: raised if not within blender's runtime.
        """
        if bpy is None:
            raise RuntimeError(f"{type(self).__name__} needs to be instantiated from within blender's python runtime.")
        self._conn: rpyc.Connection | None = None
        self.log: logging.Logger = server_log
        self._initialized: bool = False
        self._keyframe_scale: float = 1.0
        self._warned_no_outputs: bool = False
        self._outputs: dict[str, Any] = {}
        self._camera: bpy.types.Camera | None = None

    def _clear_cached_properties(self) -> None:
        # Based on: https://stackoverflow.com/a/71579485
        for name in dir(type(self)):
            if isinstance(getattr(type(self), name), functools.cached_property):
                vars(self).pop(name, None)  # type: ignore

    def on_connect(self, conn: rpyc.Connection) -> None:
        """Called when the connection is established

        Args:
            conn (rpyc.Connection): Connection object
        """
        # Log to server as logging is likely not setup yet.
        server_log.info("Successfully connected to BlenderClient instance.")
        self._conn = conn

    def on_disconnect(self, conn: rpyc.Connection) -> None:
        """Called when the connection has already terminated. Resets blender runtime.
        (must not perform any IO on the connection)

        Args:
            conn (rpyc.Connection): Connection object
        """
        server_log.info("Successfully disconnected from BlenderClient instance.")
        self._conn = None
        self.reset()

    def reset(self) -> None:
        """Cleans up and resets blender runtime.

        De-initialize service by restoring blender to it's startup state,
        ensuring any cached attributes are cleaned (otherwise objects will be stale),
        and resetting any instance variables that were previously initialized.
        """
        bpy.ops.wm.read_factory_settings()
        self._clear_cached_properties()
        self._initialized = False
        self._keyframe_scale = 1.0
        self._warned_no_outputs = False
        self._outputs = {}
        self._camera = None

    def register_output_type(
        self,
        subpath: str,
        node: bpy.types.CompositorNodeOutputFile,
        slot: bpy.types.NodeOutputFileSlotFile | bpy.types.NodeCompositorFileOutputItem,
        **camera_defaults,
    ) -> None:
        """Register a new output datatype. If this is not called by an ``include_`` method, the
        metadata for that datatype will not be saved to the database and the path to which the data
        is saved will not be updated at every render.

        Warning:
            You must pass in the slot instance that was returned when a new ``file_output_item``
            was created and not simply one of the `node.file_output_items <https://docs.blender.org/api/
            current/bpy.types.CompositorNodeOutputFile.html#bpy.types.CompositorNodeOutputFile.file_output_items>`_
            as these are readonly!

        Args:
            subpath (str): Path suffix, from root data directory, of the new datatype (eg: "previews/depths")
            node (bpy.types.CompositorNodeOutputFile): Output file node responsible for saving
            slot (bpy.types.NodeOutputFileSlotFile | bpy.types.NodeCompositorFileOutputItem): Slot of node which
                will save the output data.
            **camera_defaults: Any addition camera information that will be added by default. Commonly,
                the number of output channels is passed in (eg: c=4 for RGBA).

        Raises:
            RuntimeError: raised if output type has already been registered.
        """
        if subpath in self._outputs:
            raise RuntimeError(f"Cannot register '{subpath}', as it was already registered!")

        if (db_path := self.root_path / subpath / "transforms.db").exists():
            self.log.info(f"Database at {db_path} already exists, overwriting...")
            db_path.unlink()

        db = SqliteDatabase(db_path, pragmas=_DEFAULT_PRAGMAS)
        self._outputs[subpath] = (node, slot, db, camera_defaults)

    def _include_output(
        self,
        subpath: str,
        source_socket: bpy.types.NodeSocket,
        label: str | None = None,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: int = 32,
        preview: bool = False,
        c: int | None = None,
        denoise: bool = False,
    ) -> None:
        """Helper function to create a file output node, link it, and register the output type.

        Args:
            subpath (str): Subpath to save output in.
            source_socket (bpy.types.NodeSocket): Socket to link to output node.
            label (str, optional): Label for output node. Defaults to None.
            file_format (str, optional): Format to save output as. Defaults to "OPEN_EXR".
            color_mode (str, optional): Color mode to save output as. Defaults to "RGB".
            exr_codec (str, optional): EXR codec to use. Defaults to "DWAA".
            bit_depth (int, optional): Bit depth to use. Defaults to 32.
            preview (bool, optional): If true, output node will be configured for preview. Defaults to False.
            c (int, optional): Number of channels for registration. Defaults to None (inferred from color_mode).
            denoise (bool, optional): If true, insert a denoise compositor node before the file output.
                This enables the Cycles denoising data view layer passes (albedo and normal) and connects
                them to the denoise node for higher-quality denoising. Only available for Cycles render engine.
                Defaults to False.
        """
        node, (socket,), (slot,) = file_output_node(
            self.tree,
            self.root_path / subpath / "0000",
            label=label or f"{subpath.replace('/', ' ').title()} Output",
            preview=preview,
            color_mode=color_mode,
        )

        if not preview:
            node.format.color_management = "OVERRIDE"
            node.format.linear_colorspace_settings.name = "Non-Color"
            node.format.file_format = file_format
            node.format.color_mode = color_mode
            node.format.exr_codec = exr_codec
            node.format.color_depth = str(bit_depth)
            slot.name = str(Path(slot.name).with_suffix(FORMATS[file_format.upper()]))

        if denoise and self.scene.render.engine.upper() == "CYCLES":
            # Enable denoising data passes on the view layer so Cycles emits
            # "Denoising Albedo" and "Denoising Normal" outputs from the
            # render layers node. These guide the denoiser and produce
            # significantly cleaner results than denoising the image alone.
            self.view_layer.cycles.denoising_store_passes = True
            denoise_node = self.tree.nodes.new("CompositorNodeDenoise")
            self.tree.links.new(source_socket, denoise_node.inputs["Image"])
            self.tree.links.new(
                self.render_layers.outputs["Denoising Albedo"],
                denoise_node.inputs["Albedo"],
            )
            self.tree.links.new(
                self.render_layers.outputs["Denoising Normal"],
                denoise_node.inputs["Normal"],
            )
            self.tree.links.new(denoise_node.outputs["Image"], socket)
        elif denoise and self.scene.render.engine.upper() != "CYCLES":
            self.log.warning("Cannot produce denoised output when not using CYCLES.")
            self.tree.links.new(source_socket, socket)
        else:
            self.tree.links.new(source_socket, socket)

        if c is None:
            c = COLOR_MODE_CHANNELS.get(color_mode.upper())

        self.register_output_type(subpath, node, slot, c=c)

    def _save_metadata(
        self,
        paths: dict[str, Path],
        camera_info: dict[str, str | float | int],
        transform_matrix: list[float],
        index: int,
    ) -> None:
        """Post-render callback responsible for saving frame metadata to each database.

        Args:
            paths (dict[str, Path]): A dictionary mapping the data type's subpath to the recently rendered file.
                For example, ``{"frames": "0001/321.png", "depths": "0001/321.exr"}``.
            camera_info (dict[str, str  |  float  |  int]): Camera info at current index, as retrieved by ``BlenderService.camera_info``.
            transform_matrix (list[float]): Current camera extrinsic matrix.
            index (int): Current frame index.
        """
        if missing := set(self._outputs.keys()) - set(paths.keys()):
            self.log.warning(f"Metadata for outputs {missing} is missing!")

        for subpath, (_, _, db, defaults) in self._outputs.items():
            with db.connection_context(), db.atomic("IMMEDIATE"), db.bind_ctx(_MODELS):
                db.create_tables(_MODELS, safe=True)
                camera, _ = _Camera.get_or_create(**(defaults | camera_info))
                data = _Data.create(path=paths[subpath])
                _Frame.create(id=index, camera=camera, transform_matrix=transform_matrix, data=data)

    @property
    @require_initialized_service
    def scene(self) -> bpy.types.Scene:
        """Get current blender scene"""
        return bpy.context.scene

    @property
    @require_initialized_service
    def tree(self) -> bpy.types.CompositorNodeTree:
        """Get current scene's node tree"""
        if bpy.app.version >= (5, 0, 0):
            return self.scene.compositing_node_group
        else:
            return self.scene.node_tree

    @functools.cached_property
    @require_initialized_service
    def render_layers(self) -> bpy.types.CompositorNodeRLayers:
        """Get and cache render layers node, create one if needed."""
        for node in self.tree.nodes:
            if node.type == "R_LAYERS":
                return node
        return self.tree.nodes.new("CompositorNodeRLayers")

    @property
    @require_initialized_service
    def view_layer(self) -> bpy.types.ViewLayer:
        """Get current view layer"""
        if not bpy.context.view_layer:
            raise ValueError("Expected at least one view layer, cannot render without it. Please add one manually.")
        return bpy.context.view_layer

    @property
    @require_initialized_service
    def camera(self) -> bpy.types.Camera:
        """Get active camera, detect when it changes."""
        if self._camera is None:
            cameras = [ob for ob in self.scene.objects if ob.type == "CAMERA"]
            if not cameras:
                raise RuntimeError("No camera found, please add one manually.")

            if len(cameras) > 1 and self.scene.camera:
                current_active = self.scene.camera
            else:
                current_active = cameras[0]

            if len(cameras) > 1:
                if self.scene.camera:
                    self.log.warning(f"Multiple cameras found. Using active camera named: '{self.scene.camera.name}'.")
                else:
                    self.log.warning(f"No active camera was found. Using camera named: '{cameras[0].name}'.")
            self._camera = current_active
        elif self._camera != self.scene.camera:
            self.log.warning(
                f"Active camera changed from '{self._camera.name}' to '{self.scene.camera.name}' "
                f"at frame {self.scene.frame_current}."
            )
            self._camera = self.scene.camera

        return self._camera

    @require_initialized_service
    def get_parents(self, obj: bpy.types.Object) -> list[bpy.types.Object]:
        """Recursively retrieves parent objects of a given object in Blender

        Args:
            obj: Object to find parent of.

        Returns:
            list[bpy.types.Object]: Parent objects of obj.
        """
        if getattr(obj, "parent", None):
            return [obj.parent] + self.get_parents(obj.parent)
        return []

    def exposed_with_logger(self, log: logging.Logger) -> None:
        """Use supplied logger, if logger is initialized in client, messages will log to the client.

        Args:
            log (logging.Logger): Logger to use for messages
        """
        self.log = log

    def exposed_initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        """Initialize BlenderService and load blendfile.

        Args:
            blend_file (str | os.PathLike): path of scene file to load.
            root_path (str | os.PathLike): path at which to save rendered results.
            **kwargs: Additional keyword arguments to be passed to
                `bpy.ops.wm.open_mainfile <https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.open_mainfile>`_.
        """
        # TODO: This should perhaps be `exposed_load_file`, and the root_path logic should be moved
        #   to another method which would facilitate writing to local disk/sending renders over the wire.
        if self._initialized:
            self.reset()

        # Load blendfile
        self.root_path: Path = Path(str(root_path)).resolve()
        self.blend_file: Path = Path(str(blend_file)).resolve()
        bpy.ops.wm.open_mainfile(filepath=str(blend_file), **kwargs)
        self.log.info(f"Successfully loaded {blend_file}")

        # Init various variables to track state
        self._use_animation: bool = True
        self._initialized = True

        # Ensure we are using the compositor, and node tree.
        if bpy.app.version >= (5, 0, 0):
            bpy.ops.node.new_compositing_node_group(name="Compositor Nodes")
            self.scene.compositing_node_group = bpy.data.node_groups["Compositor Nodes"]
        else:
            self.scene.use_nodes = True
        self.scene.render.use_compositing = True

        # Set default render settings
        self.scene.render.use_persistent_data = True
        self.scene.render.film_transparent = False

        # Warn if extra file output pipelines are found
        for n in getattr(self.tree, "nodes", []):
            if isinstance(n, bpy.types.CompositorNodeOutputFile) and not n.mute:
                self.log.warning(f"Found unexpected output node {n}")

        # Catalogue any animations that are already disabled, otherwise
        self._disabled_fcurves: set[bpy.types.Action] = {fcurve for fcurve in self.exposed_iter_fcurves() if fcurve.mute}

    @require_initialized_service
    def exposed_iter_fcurves(self, actions: list[bpy.types.Action] | None = None) -> Iterator[bpy.types.FCurve]:
        """Yield fcurves of all actions.

        This abstracts away the API for accessing fcurves which changed to using channelbags in v4.4, see
        `release notes here <https://developer.blender.org/docs/release_notes/4.4/upgrading/slotted_actions/>`_.

        Args:
            actions (list[bpy.types.Action] | None, optional): Only yield fcurves from these actions if specified,
                otherwise use all actions. Defaults to None.

        Yields:
            Iterator[bpy.types.FCurve]: an fcurve object from the scene or action
        """
        if bpy.app.version >= (4, 4, 0):
            for action in actions or bpy.data.actions:
                for slot in action.slots:
                    channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)

                    for fcurve in channelbag.fcurves or []:
                        yield fcurve
        else:
            for action in actions or bpy.data.actions:
                for fcurve in action.fcurves or []:
                    yield fcurve

    @require_initialized_service
    def exposed_get_original_fps(self) -> float:
        """Get effective framerate (fps/fps_base).

        Returns:
            float: Frame rate of scene.
        """
        # Note: Exposed properties are not supported by rpyc.
        return self.scene.render.fps / self.scene.render.fps_base

    @require_initialized_service
    def exposed_animation_range(self) -> range:
        """Get animation range of current scene as range(start, end+1, step).

        Returns:
            range: Range of frames in animation.
        """
        return range(self.scene.frame_start, self.scene.frame_end + 1, self.scene.frame_step)

    @require_initialized_service
    def exposed_animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range of current scene as a tuple of (start, end, step).

        Returns:
            tuple[int, int, int]: Frame start, end, and step of animation.
        """
        return (self.scene.frame_start, self.scene.frame_end, self.scene.frame_step)  # type: ignore

    @require_initialized_service
    def exposed_include_composites(
        self,
        file_format: FILE_FORMATS | None = None,
        color_mode: COLOR_MODES | None = None,
        exr_codec: EXR_CODECS | None = None,
        bit_depth: Literal[8, 16, 32] | None = None,
    ) -> None:
        """Sets up Blender to include the outputs of any existing compositor nodes groups.

        Note: A default arguments of ``None`` means do not change setting inherited from the blendfile's ``Output`` settings.

        Args:
            file_format (str | None, optional): Format to save composited render as. Options vary depending on the version of Blender,
                with the following being broadly available: ('BMP', 'IRIS', 'PNG', 'JPEG', 'JPEG2000', 'TARGA', 'TARGA_RAW',
                'CINEON', 'DPX', 'OPEN_EXR', 'HDR', 'TIFF', 'WEBP'). Defaults to None.
            color_mode (str | None, optional): Typically one of ('BW', 'RGB', 'RGBA'). Defaults to None.
            exr_codec (str | None, optional): Codec used to compress exr file. Only used when ``file_format="OPEN_EXR"``,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to None.
            bit_depth (int | None, optional): Bit depth per channel, also referred to as color-depth. Options depend on the
                chosen file format, with 8, 16 and 32bits being common. Defaults to None.
        """

        def _compositor_not_used() -> bool:
            # Check if any group output nodes are directly connected to the "Image" render layer.
            nodes = collections.deque(n for n in self.tree.nodes if not isinstance(n, bpy.types.NodeGroupOutput))

            # Either there's no group output or it's not connected to anything.
            if not nodes:
                raise RuntimeError("Cannot save compositor outputs, a `Group Output` node was not found.")
            elif all(not i.links for node in nodes for i in node.inputs):
                raise RuntimeError("Cannot save compositor outputs, nothing is connected to the `Group Output` node.")

            while nodes:
                node = nodes.pop()

                for sock in node.inputs:
                    for link in sock.links:
                        if isinstance(link.from_node, bpy.types.NodeReroute):
                            nodes.appendleft(link.from_node)
                        elif (
                            isinstance(link.from_node, bpy.types.CompositorNodeRLayers)
                            and link.from_socket.name == "Image"
                        ):
                            return True
            return False

        if _compositor_not_used():
            self.log.warning(
                "No custom compositing workflow found, the 'composites' and 'frames' outputs might be identical!"
            )

        if file_format is not None:
            self.scene.render.image_settings.file_format = file_format.upper()
        if bit_depth is not None:
            self.scene.render.image_settings.color_depth = str(bit_depth)
        if color_mode is not None:
            self.scene.render.image_settings.color_mode = color_mode.upper()
        if exr_codec is not None:
            self.scene.render.image_settings.exr_codec = exr_codec

        # Note: The compositor output is saved by blender directly through
        #   it's output settings so no file output node is needed.
        self.register_output_type(
            "composites",
            object,
            object,
            c=COLOR_MODE_CHANNELS.get(self.scene.render.image_settings.color_mode.upper()),
        )

    @require_initialized_service
    def exposed_include_frames(
        self,
        file_format: FILE_FORMATS = "PNG",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 8,
    ) -> None:
        """Sets up Blender compositor to include ground truth rendered images, bypassing any existing compositor nodes.

        Note:
            For linear intensity renders, use the "OPEN_EXR" format with and 32 or 16 bits.

        Args:
            file_format (str, optional): Format to save ground truth render as. Options vary depending on the version of Blender,
                with the following being broadly available: ('BMP', 'IRIS', 'PNG', 'JPEG', 'JPEG2000', 'TARGA', 'TARGA_RAW',
                'CINEON', 'DPX', 'OPEN_EXR', 'HDR', 'TIFF', 'WEBP'). Defaults to "PNG".
            color_mode (str, optional): Typically one of ('BW', 'RGB', 'RGBA'). Defaults to "RGB".
            exr_codec (str, optional): Codec used to compress exr file. Only used when ``file_format="OPEN_EXR"``,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, also referred to as color-depth. Options depend on the
                chosen file format, with 8, 16 and 32 bits being common. Defaults to 8 bits.

        Raises:
            ValueError: raised when file-format not understood.
        """
        if file_format.upper() not in FORMATS:
            raise RuntimeError(
                f"File format not understood, got `{file_format}` and expected one of `{', '.join(FORMATS)}`"
            )

        self._include_output(
            "frames",
            self.render_layers.outputs["Image"],
            label="Frame Output",
            file_format=file_format,
            color_mode=color_mode,
            exr_codec=exr_codec,
            bit_depth=bit_depth,
        )

    @require_initialized_service
    def exposed_include_depths(
        self,
        preview: bool = True,
        file_format: FILE_FORMATS = "OPEN_EXR",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        """Sets up Blender compositor to include depth map for rendered images.

        Note:
            The preview colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Args:
            preview (bool, optional): If true, colorized depth maps, helpful for quick visualizations,
                will be generated alongside ground-truth depth maps. Defaults to True.
            file_format (str, optional): Format of depth maps, one of "OPEN_EXR" or "HDR". Defaults to "OPEN_EXR".
            exr_codec (str, optional): Codec used to compress exr file. Only used when ``file_format="OPEN_EXR"``,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, also referred to as color-depth. Options depend on the
                chosen file format, with 8, 16 and 32 bits being common. Defaults to 32 bits.

        Raises:
            ValueError: raised when file-format not understood.
        """
        # TODO: Add colormap option?
        self.view_layer.use_pass_z = True

        if file_format.upper() not in ("OPEN_EXR", "HDR"):
            raise ValueError(f"Expected one of OPEN_EXR/HDR for file_format, got {file_format}.")

        if preview:
            normalize = self.tree.nodes.new("CompositorNodeNormalize")
            self.tree.links.new(self.render_layers.outputs["Depth"], normalize.inputs[0])
            self._include_output(
                "previews/depths",
                normalize.outputs[0],
                label="Preview Depth Output",
                preview=True,
                color_mode="BW",
                c=1,
            )

        color_mode = "BW" if (file_format.upper() == "OPEN_EXR" and bpy.app.version >= (4, 3, 0)) else "RGB"
        self._include_output(
            "depths",
            self.render_layers.outputs["Depth"],
            label="Depth Output",
            file_format=file_format,
            color_mode=color_mode,
            exr_codec=exr_codec,
            bit_depth=bit_depth,
            c=1,
        )

    @require_initialized_service
    def exposed_include_normals(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        """Sets up Blender compositor to include normal map for rendered images.

        Args:
            preview (bool, optional): If true, colorized normal maps will also be generated with each vector
                component being remapped from [-1, 1] to [0-255] where XYZ coordinates are mapped channel-wise to RGB.
                Defaults to True.
            exr_codec (str, optional): Codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, also referred to as color-depth. Either 16 or 32 bits. Defaults to 32 bits.
        """
        self.view_layer.use_pass_normal = True

        # This node group transforms normals from the global coordinate frame
        # to the camera's, and also colors normals as RGB
        normal_group = self.tree.nodes.new("CompositorNodeGroup")
        normal_group.label = "Normal Preview"
        normal_group.node_tree = normal_preview_node_group()
        self.tree.links.new(self.render_layers.outputs["Normal"], normal_group.inputs[0])

        if preview:
            self._include_output(
                "previews/normals",
                normal_group.outputs["RGBA"],
                label="Normals Preview Output",
                preview=True,
                color_mode="RGB",
                c=3,
            )

        vec2rgba = self.tree.nodes.new("CompositorNodeGroup")
        vec2rgba.label = "Vector2RGBA"
        vec2rgba.node_tree = vec2rgba_node_group()

        self.tree.links.new(normal_group.outputs["Vector"], vec2rgba.inputs["Image"])
        self._include_output(
            "normals",
            vec2rgba.outputs["Image"],
            label="Normals Output",
            file_format="OPEN_EXR",
            color_mode="RGB",
            exr_codec=exr_codec,
            bit_depth=bit_depth,
            c=3,
        )

    @require_initialized_service
    def exposed_include_flows(
        self,
        preview: bool = True,
        direction: Literal["forward", "backward", "both"] = "forward",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        """Sets up Blender compositor to include optical flow for rendered images.

        Args:
            preview (bool, optional): If true, also save preview visualizations of flow. Defaults to True.
            direction (str, optional): One of 'forward', 'backward' or 'both'. Direction of flow to colorize
                for preview visualization. Only used when ``preview`` is true, otherwise both forward and backward
                flows are saved. Defaults to "forward".
            exr_codec (str, optional): Codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, also referred to as color-depth. Options depend on the
                chosen file format, with 8, 16 and 32 bits being common. Defaults to 32 bits.

        Note:
            The preview colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raised when ``direction`` is not understood.
            RuntimeError: raised when motion blur is enabled as flow cannot be computed.
        """
        if direction.lower() not in ("forward", "backward", "both"):
            raise ValueError(f"Direction argument should be one of forward, backward or both, got {direction}.")
        if self.scene.render.use_motion_blur:
            raise RuntimeError("Cannot compute optical flow if motion blur is enabled.")

        self.view_layer.use_pass_vector = True

        if preview:
            # Separate forward and backward flows (with a separate color not vector node)
            split_flow = self.tree.nodes.new(type="CompositorNodeSeparateColor")
            split_flow.mode = "RGB"
            self.tree.links.new(self.render_layers.outputs["Vector"], split_flow.inputs["Image"])

            # Instantiate flow preview node group(s) and connect them
            if direction.lower() in ("forward", "both"):
                flow_group = self.tree.nodes.new("CompositorNodeGroup")
                flow_group.node_tree = flow_preview_node_group()
                flow_group.label = "Forward Flow Preview"

                self.tree.links.new(split_flow.outputs[0], flow_group.inputs["x"])
                self.tree.links.new(split_flow.outputs[1], flow_group.inputs["y"])
                self._include_output(
                    "previews/flows/forward",
                    flow_group.outputs["Image"],
                    label="Forward Flow Preview Output",
                    preview=True,
                    color_mode="RGB",
                    c=3,
                )
            if direction.lower() in ("backward", "both"):
                flow_group = self.tree.nodes.new("CompositorNodeGroup")
                flow_group.node_tree = flow_preview_node_group()
                flow_group.label = "Backward Flow Preview"

                self.tree.links.new(split_flow.outputs[2], flow_group.inputs["x"])
                self.tree.links.new(split_flow.outputs[3], flow_group.inputs["y"])
                self._include_output(
                    "previews/flows/backward",
                    flow_group.outputs["Image"],
                    label="Backward Flow Preview Output",
                    preview=True,
                    color_mode="RGB",
                    c=3,
                )

        # Save flows as EXRs, flows are a 4-vec of forward flows x/y then backwards flows x/y
        # before blender 4.3, saving a vector as an image saved only 3 channels even if `color_mode`
        # is set to RGBA. So we add a dummy vec2rgba node to trick blender into treating the
        # vector as an image with 4 channels. This dummy node just splits and recombines channels.
        vec2rgba = self.tree.nodes.new("CompositorNodeGroup")
        vec2rgba.label = "Vector2RGBA"
        vec2rgba.node_tree = vec2rgba_node_group()

        self.tree.links.new(self.render_layers.outputs["Vector"], vec2rgba.inputs["Image"])
        self._include_output(
            "flows",
            vec2rgba.outputs["Image"],
            label="Flow Output",
            file_format="OPEN_EXR",
            color_mode="RGBA",
            exr_codec=exr_codec,
            bit_depth=bit_depth,
            c=4,
        )

    @require_initialized_service
    def _include_ids(
        self,
        id_type: Literal["segmentations", "materials"],
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        shade: bool = False,
    ) -> None:
        """Shared logic for including segmentation or material ID maps."""
        # TODO: Enable assignment of custom IDs for certain objects via a dictionary.

        if self.scene.render.engine.upper() != "CYCLES":
            self.log.warning(f"Cannot produce {id_type} map when not using CYCLES.")
            return

        if id_type == "segmentations":
            self.view_layer.use_pass_object_index = True
            pass_idx_name = "Object Index" if bpy.app.version >= (5, 0, 0) else "IndexOB"
            data = bpy.data.objects
            label = "Segmentation"
        elif id_type == "materials":
            self.view_layer.use_pass_material_index = True
            pass_idx_name = "Material Index" if bpy.app.version >= (5, 0, 0) else "IndexMA"
            data = bpy.data.materials
            label = "Material"
        else:
            raise ValueError(f"Unknown id_type: {id_type}")

        # Assign IDs to every object/material (background will be 0)
        indices = np.arange(len(data))

        if shuffle:
            np.random.seed(seed=seed)
            np.random.shuffle(indices)

        for i, item in zip(indices, data):
            item.pass_index = i + 1

        if preview:
            group = self.tree.nodes.new("CompositorNodeGroup")
            group.label = f"{label} Preview"
            group.node_tree = colorize_indices_node_group(shade=shade)
            group.node_tree.nodes["NormalizeIdx"].inputs["From Max"].default_value = len(data)

            self.tree.links.new(self.render_layers.outputs[pass_idx_name], group.inputs["Index"])

            if shade:
                self.view_layer.use_pass_normal = True
                self.tree.links.new(self.render_layers.outputs["Normal"], group.inputs["Normal"])

            self._include_output(
                f"previews/{id_type}",
                group.outputs["Image"],
                label=f"{label}s Preview Output",
                preview=True,
                color_mode="RGB",
                c=3,
            )

        color_mode = "RGB" if bpy.app.version < (4, 3, 0) else "BW"
        self._include_output(
            id_type,
            self.render_layers.outputs[pass_idx_name],
            label=f"{label}s Output",
            file_format="OPEN_EXR",
            color_mode=color_mode,
            exr_codec=exr_codec,
            bit_depth=bit_depth,
        )

    @require_initialized_service
    def exposed_include_segmentations(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        """Sets up Blender compositor to include segmentation maps for rendered images.

        The preview visualization simply assigns a color to each object ID by mapping the
        objects ID value to a hue using a HSV node with saturation=1 and value=1 (except
        for the background which will have a value of 0 to ensure it is black).

        Args:
            preview (bool, optional): If true, also save preview visualizations of segmentation. Defaults to True.
            shuffle (bool, optional): Shuffle preview colors, helps differentiate object instances. Defaults to True.
            seed (int, optional): Random seed used when shuffling colors. Defaults to 1234.
            exr_codec (str, optional): Codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, also referred to as color-depth.
                Either 16 or 32 bits. Defaults to 32 bits.

        Raises:
            RuntimeError: raised when not using CYCLES, as other renderers do not support a segmentation pass.
        """
        self._include_ids(
            "segmentations",
            preview=preview,
            shuffle=shuffle,
            seed=seed,
            exr_codec=exr_codec,
            bit_depth=bit_depth,
        )

    @require_initialized_service
    def exposed_include_materials(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        """Sets up Blender compositor to include material ID maps for rendered images.

        The preview visualization simply assigns a color to each material ID by mapping the
        materials ID value to a hue using a HSV node with saturation=1 and value=1 (except
        for the background which will have a value of 0 to ensure it is black).

        Args:
            preview (bool, optional): If true, also save preview visualizations of material IDs. Defaults to True.
            shuffle (bool, optional): Shuffle preview colors, helps differentiate material instances. Defaults to True.
            seed (int, optional): Random seed used when shuffling colors. Defaults to 1234.
            exr_codec (str, optional): Codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, also referred to as color-depth.
                Either 16 or 32 bits. Defaults to 32 bits.

        Raises:
            RuntimeError: raised when not using CYCLES, as other renderers do not support a material ID pass.
        """
        self._include_ids(
            "materials",
            preview=preview,
            shuffle=shuffle,
            seed=seed,
            exr_codec=exr_codec,
            bit_depth=bit_depth,
            shade=True,
        )

    @require_initialized_service
    def exposed_include_diffuse_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        """Sets up Blender compositor to include diffuse light passes for rendered images.

        For CYCLES, this includes: Diffuse Direct, Diffuse Indirect and Diffuse Color.
        For EEVEE, this includes: Diffuse Light and Diffuse Color.

        Note:
            When using CYCLES, these extra light passes might be very noisy, especially the indirect ones,
            as they rely on raytracing. To mitigate this, you can either increase the number of samples, or the threshold in the :meth:`cycles_settings <exposed_cycles_settings>`, or/and use the denoise option.

        Args:
            file_format (str, optional): Format to save diffuse passes as. Defaults to "OPEN_EXR".
            color_mode (str, optional): Typically one of ('BW', 'RGB', 'RGBA'). Defaults to "RGB".
            exr_codec (str, optional): Codec used to compress exr file. Only used when ``file_format="OPEN_EXR"``. Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel. Defaults to 32 bits.
            denoise (bool, optional): If true, apply Cycles denoising to the direct and indirect passes
                before saving. The colour pass is left undenoised as it is noise-free by nature.
                Has no effect when not using Cycles. Defaults to True.
        """
        engine = self.scene.render.engine.upper()
        if engine == "CYCLES":
            # (pass_name, subpath, apply_denoise)
            passes = [
                ("Diffuse Direct" if bpy.app.version >= (5, 0, 0) else "DiffDir", "diffuse/direct", denoise),
                ("Diffuse Indirect" if bpy.app.version >= (5, 0, 0) else "DiffInd", "diffuse/indirect", denoise),
                ("Diffuse Color" if bpy.app.version >= (5, 0, 0) else "DiffCol", "diffuse/color", False),
            ]
            self.view_layer.use_pass_diffuse_direct = True
            self.view_layer.use_pass_diffuse_indirect = True
            self.view_layer.use_pass_diffuse_color = True
        elif engine == "BLENDER_EEVEE" or engine == "BLENDER_EEVEE_NEXT":
            passes = [
                ("DiffDir", "diffuse/light", False),
                ("DiffCol", "diffuse/color", False),
            ]
            self.view_layer.use_pass_diffuse_light = True
            self.view_layer.use_pass_diffuse_color = True
        else:
            self.log.warning(f"Diffuse passes are not supported for engine {engine}")
            return

        for pass_name, subpath, apply_denoise in passes:
            self._include_output(
                subpath,
                self.render_layers.outputs[pass_name],
                label=f"{pass_name} Output",
                file_format=file_format,
                color_mode=color_mode,
                exr_codec=exr_codec,
                bit_depth=bit_depth,
                denoise=apply_denoise,
            )

    @require_initialized_service
    def exposed_include_emission(self, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32) -> None:
        """Sets up Blender compositor to include the light that surfaces seen by the camera emit.

        Emitted light doesn't depend on the light that reaches surfaces, so it is left as rendered when a participating
        medium dims the light reaching surfaces, e.g. the light of screens or of lamps seen directly, see
        :mod:`visionsim.medium.surfaces`.

        Args:
            exr_codec (str, optional): Codec used to compress exr file. Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel, either 16 or 32 bits. Defaults to 32 bits.
        """
        self.view_layer.use_pass_emit = True
        name = "Emission" if "Emission" in self.render_layers.outputs else "Emit"
        self._include_output(
            "emission",
            self.render_layers.outputs[name],
            label="Emission Output",
            file_format="OPEN_EXR",
            color_mode="RGB",
            exr_codec=exr_codec,
            bit_depth=bit_depth,
        )

    @require_initialized_service
    def exposed_include_specular_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        """Sets up Blender compositor to include specular light passes for rendered images.

        For CYCLES, this includes: Glossy Direct, Glossy Indirect and Glossy Color.
        For EEVEE, this includes: Specular Light and Specular Color.

        Note:
            When using CYCLES, these extra light passes might be very noisy, especially the indirect ones,
            as they rely on raytracing. To mitigate this, you can either increase the number of samples, or the threshold in the :meth:`cycles_settings <exposed_cycles_settings>`, or/and use the denoise option.

        Args:
            file_format (str, optional): Format to save specular passes as. Defaults to "OPEN_EXR".
            color_mode (str, optional): Typically one of ('BW', 'RGB', 'RGBA'). Defaults to "RGB".
            exr_codec (str, optional): Codec used to compress exr file. Only used when ``file_format="OPEN_EXR"``. Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel. Defaults to 32 bits.
            denoise (bool, optional): If true, apply Cycles denoising to the direct and indirect passes
                before saving. The colour pass is left undenoised as it is noise-free by nature.
                Has no effect when not using Cycles. Defaults to True.
        """
        engine = self.scene.render.engine.upper()
        if engine == "CYCLES":
            # (pass_name, subpath, apply_denoise)
            passes = [
                ("Glossy Direct" if bpy.app.version >= (5, 0, 0) else "GlossDir", "specular/direct", denoise),
                ("Glossy Indirect" if bpy.app.version >= (5, 0, 0) else "GlossInd", "specular/indirect", denoise),
                ("Glossy Color" if bpy.app.version >= (5, 0, 0) else "GlossCol", "specular/color", False),
            ]
            self.view_layer.use_pass_glossy_direct = True
            self.view_layer.use_pass_glossy_indirect = True
            self.view_layer.use_pass_glossy_color = True
        elif engine == "BLENDER_EEVEE" or engine == "BLENDER_EEVEE_NEXT":
            passes = [
                ("GlossDir", "specular/light", False),
                ("GlossCol", "specular/color", False),
            ]
            self.view_layer.use_pass_specular_light = True
            self.view_layer.use_pass_specular_color = True
        else:
            self.log.warning(f"Specular passes are not supported for engine {engine}")
            return

        for pass_name, subpath, apply_denoise in passes:
            self._include_output(
                subpath,
                self.render_layers.outputs[pass_name],
                label=f"{pass_name} Output",
                file_format=file_format,
                color_mode=color_mode,
                exr_codec=exr_codec,
                bit_depth=bit_depth,
                denoise=apply_denoise,
            )

    @require_initialized_service
    def exposed_include_points(
        self,
        preview: bool = True,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        """Sets up Blender compositor to include a world-space point map for each frame.

        Note:
            The point map corresponds to world-space positions, like those used in VGGT [1]_,
            and not the camera-centric positions used in DUSt3R [2]_.

        Args:
            preview (bool, optional): If true, colorized point maps will also be generated, where colors are
                assigned based on the absolute fractional world coordinates. Defaults to True.
            exr_codec (str, optional): Codec used to compress exr file. Defaults to "DWAA".
            bit_depth (int, optional): Bit depth per channel. Either 16 or 32 bits. Defaults to 32 bits.

        .. [1] `VGGT: Visual Geometry Grounded Transformer <https://arxiv.org/abs/2503.11651>`_
        .. [2] `DUSt3R: Geometric 3D Vision Made Easy with Unconstrained Image Collections <https://arxiv.org/abs/2312.14132>`_
        """
        engine = self.scene.render.engine.upper()
        if (engine == "BLENDER_EEVEE" or engine == "BLENDER_EEVEE_NEXT") and bpy.app.version < (4, 2, 0):
            self.log.warning(f"Position pass is not supported for engine {engine} in Blender version {bpy.app.version}")
            return

        self.view_layer.use_pass_position = True

        if preview:
            group = self.tree.nodes.new("CompositorNodeGroup")
            group.node_tree = point_preview_node_group()

            self.tree.links.new(self.render_layers.outputs["Position"], group.inputs["Vector"])

            self._include_output(
                "previews/points",
                group.outputs[0],
                label="Preview Points Output",
                preview=True,
                color_mode="RGB",
                c=3,
            )

        vec2rgba = self.tree.nodes.new("CompositorNodeGroup")
        vec2rgba.label = "Vector2RGBA"
        vec2rgba.node_tree = vec2rgba_node_group()

        self.tree.links.new(self.render_layers.outputs["Position"], vec2rgba.inputs["Image"])
        self._include_output(
            "points",
            vec2rgba.outputs["Image"],
            label="Points Output",
            file_format="OPEN_EXR",
            color_mode="RGB",
            exr_codec=exr_codec,
            bit_depth=bit_depth,
            c=3,
        )

    @require_initialized_service
    def exposed_load_addons(self, *addons: str) -> None:
        """Load blender addons by name (case-insensitive).

        Args:
            *addons (str): name of addons to load.
        """
        for addon in addons:
            addon = addon.strip().lower()
            addon_module = addon_utils.enable(addon, default_set=True)
            self.log.info(f"Loaded addon {addon}: {addon_module}")

    @require_initialized_service
    def exposed_set_resolution(
        self,
        height: tuple[int] | list[int] | int | None = None,
        width: int | None = None,
        resolution_percentage: int = 100,
    ) -> None:
        """Set frame resolution (height, width) in pixels.
        If a single tuple is passed, instead of using keyword arguments, it will be parsed as (height, width).

        Args:
            height (tuple[int] | list[int] | int | None, optional): Height of render in pixels. Defaults to value from file.
            width (int | None, optional): Width of render in pixels. Defaults to value from file.
            resolution_percentage (float, optional): Percentage of the original resolution to render at. Defaults to 100%.

        Raises:
            ValueError: raised if resolution is not understood.
        """
        if isinstance(height, (tuple, list)):
            if width is not None:
                raise ValueError(
                    "Cannot understand desired resolution, either pass a (h, w) tuple, or use keyword arguments."
                )
            height, width = height  # type: ignore

        if height:
            self.scene.render.resolution_y = int(height)
        if width:
            self.scene.render.resolution_x = int(width)
        self.scene.render.resolution_percentage = int(resolution_percentage)

    @require_initialized_service
    def exposed_use_motion_blur(self, enable: bool) -> None:
        """Enable/disable motion blur.

        Args:
            enable (bool): If true, enable motion blur.

        Raises:
            RuntimeError: raised when motion blur is enabled as flow cannot be computed.
        """
        if enable and "flows" in self._outputs:
            raise RuntimeError("Cannot enable motion blur if computing optical flow.")
        self.scene.render.use_motion_blur = enable

    @require_initialized_service
    def exposed_use_animations(self, enable: bool) -> None:
        """Enable/disable all animations.

        Args:
            enable (bool): If true, enable animations.
        """
        for fcurve in self.exposed_iter_fcurves():
            if fcurve not in self._disabled_fcurves:
                fcurve.mute = not enable
        self._use_animation = enable

    @require_initialized_service
    def exposed_cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> list[str]:
        """Enables/activates cycles render devices and settings.

        Note: A default arguments of ``None`` means do not change setting inherited from blendfile.

        Args:
            device_type (str, optional): Name of device to use, one of "cpu", "cuda", "optix", "metal", etc.
                See `blender docs <https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html>`_
                for full list. Defaults to None.
            use_cpu (bool, optional): Boolean flag to enable CPUs alongside GPU devices. Defaults to None.
            adaptive_threshold (float, optional): Set noise threshold upon which to stop taking samples. Defaults to None.
            max_samples (int, optional): Maximum number of samples per pixel to take. Defaults to None.
            use_denoising (bool, optional): If enabled, a denoising pass will be used. Defaults to None.

        Raises:
            RuntimeError: raised when no devices are found.
            ValueError: raised when setting ``use_cpu`` is required.

        Returns:
            list[str]: Name of activated devices.
        """
        if self.scene.render.engine.upper() != "CYCLES":
            self.log.warning(
                f"Using {self.scene.render.engine.upper()} rendering engine, with default OpenGL rendering device(s)."
            )
            return []

        if use_denoising is not None:
            self.scene.cycles.use_denoising = use_denoising

        if max_samples is not None:
            self.scene.cycles.samples = max_samples

        if adaptive_threshold is not None:
            self.scene.cycles.adaptive_threshold = adaptive_threshold

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()

        if not cycles_preferences.devices:
            raise RuntimeError("No devices found!")

        if device_type:
            if use_cpu is None:
                raise ValueError("Parameter `use_cpu` needs to be set if setting device type.")

            for device in cycles_preferences.devices:
                device.use = False

            activated_devices = []
            devices = filter(lambda d: d.type.upper() == device_type.upper(), cycles_preferences.devices)

            for device in itertools.chain(devices, filter(lambda d: d.type == "CPU" and use_cpu, devices)):
                self.log.info(f"Activated device {device.name}, {device.type}")
                activated_devices.append(device.name)
                device.use = True
            cycles_preferences.compute_device_type = "NONE" if device_type.upper() == "CPU" else device_type.upper()
            self.scene.cycles.device = "CPU" if device_type.upper() == "CPU" else "GPU"
            return activated_devices
        return []

    @require_initialized_service
    def exposed_unbind_camera(self, clear_animations: bool = True) -> None:
        """Remove constraints, animations and parents from main camera.

        Note: In order to undo this, you'll need to re-initialize.

        Args:
            clear_animations (bool, optional): If true clear animation data for camera.
        """
        for c in self.camera.constraints:
            self.camera.constraints.remove(c)

        with bpy.context.temp_override(selected_editable_objects=[self.camera]):
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

        if clear_animations:
            self.camera.animation_data_clear()

    @require_initialized_service
    def exposed_move_keyframes(self, scale=1.0, shift=0.0) -> None:
        """Adjusts keyframes in Blender animations, keypoints are first scaled then shifted.

        Args:
            scale (float, optional): Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift (float, optional): Factor used to shift keyframe positions along x-axis. Defaults to 0.0.

        Raises:
            RuntimeError: raised if trying to move keyframes beyond blender's limits.
        """
        # TODO: Refactor this into `exposed_remap_keyframes`, which would allow arbitrary transformation
        #   using a user-supplied remapping function, and redefine move_keyframes in terms of it.
        # TODO: This method can be slow if there's a lot of keyframes
        #   See: https://blender.stackexchange.com/questions/111644
        if scale == 1.0 and shift == 0.0:
            return

        if self._keyframe_scale != 1.0:
            raise NotImplementedError("Rescaling keyframes multiple times is currently unsupported.")

        # No idea why, but if we don't break this out into separate
        # variables the value we store is incorrect, often off by one.
        # We add, then remove one because frame_start and frame_end are inclusive,
        # consider [0, 99], which has length of 100, if scaled by 5, we'd get
        # [0, 495] which has length of 496 instead of 500. So we make end exclusive,
        # shift and scale it, then make it inclusive again.
        start = round(self.scene.frame_start * scale + shift)
        end = round((self.scene.frame_end + 1) * scale + shift) - 1

        if start < 0 or start >= 1_048_574 or end < 0 or end >= 1_048_574:
            raise RuntimeError(
                "Cannot scale and shift keyframes past render limits. For more please see: "
                "https://docs.blender.org/manual/en/latest/advanced/limits.html"
            )
        self.scene.frame_start = start
        self.scene.frame_end = end
        self._keyframe_scale = scale

        for fcurve in self.exposed_iter_fcurves():
            for kfp in fcurve.keyframe_points or []:
                # Note: Updating `co_ui` does not move the handles properly!!!
                kfp.co.x = kfp.co.x * scale + shift
                kfp.handle_left.x = kfp.handle_left.x * scale + shift
                kfp.handle_right.x = kfp.handle_right.x * scale + shift
                kfp.period *= scale
        self.scene.render.motion_blur_shutter /= scale

    @require_initialized_service
    def exposed_set_current_frame(self, frame_number: int) -> None:
        """Set current frame number. This might advance any animations.

        Args:
            frame_number (int): index of frame to skip to.
        """
        self.scene.frame_set(frame_number)

    @require_initialized_service
    def exposed_camera_info(self) -> dict[str, Any]:
        """Return a dictionary with camera intrinsics.

        Returns:
            dict[str, Any]: dictionary containing camera parameters.
        """
        if self.camera.data.type != "PERSP":
            raise RuntimeError(
                f"Only perspective cameras are currently supported, got '{self.camera.data.type}' instead."
            )

        info = {
            k: getattr(self.camera.data, k)
            for k in [
                "angle",
                "angle_x",
                "angle_y",
                "clip_start",
                "clip_end",
                "lens",
                "lens_unit",
                "sensor_height",
                "sensor_width",
                "sensor_fit",
                "shift_x",
                "shift_y",
                "type",
            ]
        }

        # Mirror Blender's camera model (see `BKE_camera_params_compute_viewplane`): the sensor spans the
        #   image along the fit axis (its longest side when fit is AUTO), lens shifts are expressed in units
        #   of that same side, and non-square pixels are accounted for by rescaling the vertical axis.
        render = self.scene.render
        scale = render.resolution_percentage / 100.0
        w, h = int(render.resolution_x * scale), int(render.resolution_y * scale)
        ycor = render.pixel_aspect_y / render.pixel_aspect_x
        sensor_fit = self.camera.data.sensor_fit
        sensor_size = self.camera.data.sensor_height if sensor_fit == "VERTICAL" else self.camera.data.sensor_width

        if sensor_fit == "AUTO":
            sensor_fit = "HORIZONTAL" if render.pixel_aspect_x * w >= render.pixel_aspect_y * h else "VERTICAL"
        viewfac = w if sensor_fit == "HORIZONTAL" else ycor * h

        info["w"], info["h"] = w, h
        info["fl_x"] = float(self.camera.data.lens * viewfac / sensor_size)
        info["fl_y"] = info["fl_x"] / ycor
        info["shift_x"] *= viewfac
        info["shift_y"] *= viewfac / ycor
        info["cx"] = w / 2 - info["shift_x"]
        info["cy"] = h / 2 + info["shift_y"]
        info["fps"] = self.exposed_get_original_fps()
        info["keyframe_scale"] = self._keyframe_scale
        return info

    @require_initialized_service
    def exposed_camera_extrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 4x4 transform matrix encoding the current camera pose.

        Returns:
            npt.NDArray[np.floating]: Current camera pose in matrix form.
        """
        pose = np.array(self.camera.matrix_world)
        pose[:3, :3] /= np.linalg.norm(pose[:3, :3], axis=0)
        return pose

    @staticmethod
    def _mean_equirectangular(image: bpy.types.Image) -> list[float]:
        """Average linear radiance over the upper hemisphere of an equirectangular environment map.

        Args:
            image (bpy.types.Image): Environment map, in equirectangular projection.

        Returns:
            list[float]: Mean RGB radiance above the horizon, weighted by solid angle.
        """
        w, h = image.size
        pixels = np.empty(w * h * image.channels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(h, w, image.channels)[..., :3].astype(float)

        if not image.is_float and image.colorspace_settings.name.lower().startswith("srgb"):
            rgb = np.where(rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

        # Rows go from the bottom to the top of the image, each row spans a band of latitude
        latitudes = (np.arange(h) + 0.5) / h * np.pi - np.pi / 2
        weights = np.cos(latitudes) * (latitudes > 0)
        return ((rgb * weights[:, None, None]).sum(axis=(0, 1)) / (weights.sum() * w)).tolist()

    @require_initialized_service
    def _world_radiance(self) -> list[float]:
        """Average radiance of the world background above the horizon."""
        world = self.scene.world
        if world is None:
            return [0.0, 0.0, 0.0]

        nodes = world.node_tree.nodes if world.node_tree else []
        output = next((n for n in nodes if n.bl_idname == "ShaderNodeOutputWorld" and n.is_active_output), None)
        if output is None or not output.inputs["Surface"].is_linked:
            return list(world.color)

        background = output.inputs["Surface"].links[0].from_node
        if background.bl_idname != "ShaderNodeBackground":
            self.log.warning(f"Unsupported world shader {background.bl_idname}, falling back to the world's color.")
            return list(world.color)

        color, strength = background.inputs["Color"], background.inputs["Strength"]
        if strength.is_linked:
            self.log.warning("World strength is driven by a node, falling back to its default value.")

        if color.is_linked:
            source = color.links[0].from_node
            if source.bl_idname == "ShaderNodeTexEnvironment" and source.image is not None and all(source.image.size):
                return [c * strength.default_value for c in self._mean_equirectangular(source.image)]
            self.log.warning(f"Unsupported world color node {source.bl_idname}, falling back to its default value.")
        return [c * strength.default_value for c in color.default_value[:3]]

    def _light_emission(self, obj: bpy.types.Object) -> tuple[list[float], dict[str, Any]]:
        """Color by which the node tree of a light scales its emission, and how its emission falls off with distance.

        Only the node trees that commonly set a light's emission are evaluated, as Cycles does: an Emission node, whose
        color can come from a Blackbody node, and whose strength can come from a Light Falloff node, or from an IES
        Texture node, whose angular profile is not supported. Other nodes are ignored, with a warning.

        Args:
            obj (bpy.types.Object): Light object.

        Returns:
            tuple[list[float], dict[str, Any]]: Scale of the light's RGB color, and the ``falloff`` and ``smooth`` of
            the light, if its strength is set by a Light Falloff node, see :class:`PointLight
            <visionsim.medium.model.PointLight>`.
        """
        light, white = obj.data, [1.0, 1.0, 1.0]
        tree = light.node_tree if getattr(light, "use_nodes", False) else None
        outputs = [n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputLight"] if tree else []
        output = next((n for n in outputs if n.is_active_output), outputs[0] if outputs else None)
        if output is None or not output.inputs["Surface"].is_linked:
            return white, {}

        emission = output.inputs["Surface"].links[0].from_node
        if emission.bl_idname != "ShaderNodeEmission":
            self.log.warning(f"Unsupported shader {emission.bl_idname} of light '{obj.name}', its nodes are ignored.")
            return white, {}

        color_input, strength_input, falloff = emission.inputs["Color"], emission.inputs["Strength"], {}
        color = list(color_input.default_value[:3])
        if color_input.is_linked:
            source = color_input.links[0].from_node
            if source.bl_idname == "ShaderNodeBlackbody" and not source.inputs["Temperature"].is_linked:
                color = _blackbody(source.inputs["Temperature"].default_value)
            else:
                self.log.warning(f"Unsupported color node {source.bl_idname} of light '{obj.name}', using white.")
                color = white

        strength = strength_input.default_value
        if strength_input.is_linked:
            link = strength_input.links[0]
            source = link.from_node
            if source.bl_idname == "ShaderNodeLightFalloff" and not any(i.is_linked for i in source.inputs):
                # Suns ignore the falloff, as in Cycles
                strength = source.inputs["Strength"].default_value
                if light.type != "SUN":
                    falloff = {
                        "falloff": link.from_socket.identifier.lower(),
                        "smooth": source.inputs["Smooth"].default_value,
                    }
            elif source.bl_idname == "ShaderNodeTexIES" and not source.inputs["Strength"].is_linked:
                # Cycles' IES lookup returns 100 for files it can't read, i.e. uniformly
                strength = source.inputs["Strength"].default_value * 100
                self.log.warning(f"The IES profile of light '{obj.name}' is not supported, its light is uniform.")
            else:
                self.log.warning(f"Unsupported strength node {source.bl_idname} of light '{obj.name}', using 1.")
                strength = 1.0
        return [c * strength for c in color], falloff

    def _shader_emission(self, node: bpy.types.Node, owner: str) -> npt.NDArray[np.floating]:
        """Constant radiance that a shader node emits, as Cycles evaluates it, which is zero if it doesn't emit."""
        kind = node.bl_idname
        if kind in ("ShaderNodeEmission", "ShaderNodeBsdfPrincipled"):
            principled = kind == "ShaderNodeBsdfPrincipled"
            color, strength = (node.inputs[f"Emission {n}" if principled else n] for n in ("Color", "Strength"))
            if strength.is_linked:
                self.log.warning(f"The emission strength of '{owner}' is set by nodes, using its default value.")
            if not strength.default_value:
                return np.zeros(3)
            # The alpha of a Principled BSDF fades its emission too
            alpha = node.inputs["Alpha"] if principled and "Alpha" in node.inputs else None
            fade = 1.0 if alpha is None or alpha.is_linked else float(alpha.default_value)
            return self._color_of(color, owner) * strength.default_value * fade
        if kind == "ShaderNodeAddShader":
            emitted = (self._shader_emission(i.links[0].from_node, owner) for i in node.inputs if i.is_linked)
            return sum(emitted, np.zeros(3))
        if kind == "ShaderNodeMixShader":
            factor = node.inputs[0]
            source = factor.links[0] if factor.is_linked else None
            if source is not None and source.from_node.bl_idname == "ShaderNodeLightPath":
                # Cycles evaluates the emission of surfaces sampled as lights without any type of ray, so that e.g.
                # surfaces only seen by the camera ("shadeless") don't light anything
                fac = 0.0 if source.from_socket.identifier.startswith("Is ") else 0.5
            else:
                fac = 0.5 if source is not None else float(factor.default_value)
            if source is not None and fac == 0.5:
                self.log.warning(f"The mix of the emission of '{owner}' is set by nodes, mixing it evenly.")
            emission = np.zeros(3)
            for socket, weight in ((node.inputs[1], 1 - fac), (node.inputs[2], fac)):
                if socket.is_linked:
                    emission = emission + weight * self._shader_emission(socket.links[0].from_node, owner)
            return emission
        return np.zeros(3)

    def _color_of(self, socket: bpy.types.NodeSocket, owner: str) -> npt.NDArray[np.floating]:
        """Constant linear color of a socket, the mean of an image or of a black body that sets it, or its default."""
        if not socket.is_linked:
            return np.asarray(socket.default_value[:3], dtype=float)
        node = socket.links[0].from_node
        if node.bl_idname == "ShaderNodeRGB":
            return np.asarray(node.outputs[0].default_value[:3], dtype=float)
        if node.bl_idname == "ShaderNodeBlackbody" and not node.inputs["Temperature"].is_linked:
            return np.asarray(_blackbody(node.inputs["Temperature"].default_value))
        if node.bl_idname == "ShaderNodeTexImage" and node.image is not None and all(node.image.size):
            image = node.image
            pixels = np.empty(image.size[0] * image.size[1] * image.channels, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            rgb = pixels.reshape(-1, image.channels)[:, :3].astype(float)
            if not image.is_float and image.colorspace_settings.name.lower().startswith("srgb"):
                rgb = np.where(rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            return rgb.mean(axis=0)
        self.log.warning(f"Unsupported color node {node.bl_idname} of the emission of '{owner}', using its default.")
        return np.asarray(socket.default_value[:3], dtype=float)

    def _material_emission(self, material: bpy.types.Material | None) -> npt.NDArray[np.floating] | None:
        """Constant radiance that a material emits, or None if it doesn't emit light."""
        if material is None or not material.use_nodes or not material.node_tree:
            return None
        outputs = [n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        output = next((n for n in outputs if n.is_active_output), outputs[0] if outputs else None)
        if output is None or not output.inputs["Surface"].is_linked:
            return None
        emission = self._shader_emission(output.inputs["Surface"].links[0].from_node, material.name)
        return emission if (emission > 0).any() else None

    @require_initialized_service
    def _emissive_triangles(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], list[str], list[str]]:
        """World-space corners of the triangles of meshes that emit light, of shape (t, 3, 3), their radiance, of shape
        (t, 3), the names of their materials, and of the objects they belong to, or that instance them."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        emissions: dict[str, npt.NDArray[np.floating] | None] = {}
        corners: list[npt.NDArray[np.floating]] = []
        radiance: list[npt.NDArray[np.floating]] = []
        owners: set[str] = set()
        for instance in depsgraph.object_instances:
            obj = instance.object
            owner = instance.parent.original if instance.is_instance and instance.parent else obj.original
            if (
                obj.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT")
                or owner.hide_render
                or not getattr(owner, "visible_volume_scatter", True)
            ):
                continue
            slots = [slot.material for slot in obj.material_slots]
            for material in slots:
                if material is not None and material.name not in emissions:
                    emissions[material.name] = self._material_emission(material)
            slot_emission = [emissions[m.name] if m is not None else None for m in slots]
            if all(emission is None for emission in slot_emission):
                continue

            mesh = obj.to_mesh()
            mesh.calc_loop_triangles()
            vertices = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get("co", vertices)
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", triangles)
            materials = np.empty(len(mesh.loop_triangles), dtype=np.int64)
            mesh.loop_triangles.foreach_get("material_index", materials)
            obj.to_mesh_clear()

            matrix = np.asarray(instance.matrix_world, dtype=float)
            vertices = vertices.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
            triangles = triangles.reshape(-1, 3)
            for index, emission in enumerate(slot_emission):
                selected = materials == index
                if emission is not None and selected.any():
                    corners.append(vertices[triangles[selected]])
                    radiance.append(np.broadcast_to(emission, (int(selected.sum()), 3)))
                    owners.update((owner.name, obj.original.name))
        names = sorted(name for name, emission in emissions.items() if emission is not None)
        if not corners:
            return np.zeros((0, 3, 3)), np.zeros((0, 3)), names, sorted(owners)
        return np.concatenate(corners), np.concatenate(radiance), names, sorted(owners)

    @require_initialized_service
    def _emissive_surfaces(
        self, lamps: int, patches: int, other_power: float = 0.0
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Emissive surfaces of the scene, grouped into lamps, the names of the materials that emit light, and of the
        objects whose surfaces emit it, or that instance them, see :meth:`lighting_info <exposed_lighting_info>` and
        :func:`_emissive_lamps`."""
        from mathutils.bvhtree import BVHTree  # type: ignore

        corners, radiance, names, owners = self._emissive_triangles()
        if not len(corners):
            return [], names, owners
        # Light that a triangle emits towards another one doesn't leave the surfaces
        count = len(corners)
        tree = BVHTree.FromPolygons(
            corners.reshape(-1, 3).tolist(), np.arange(3 * count).reshape(-1, 3).tolist(), all_triangles=True
        )

        def trace(origins: npt.NDArray[np.floating], directions: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
            return np.fromiter(
                (tree.ray_cast(o, d)[0] is not None for o, d in zip(origins.tolist(), directions.tolist())),
                dtype=bool,
                count=len(origins),
            )

        return _emissive_lamps(corners, radiance, trace, lamps, patches, other_power=other_power), names, owners

    @require_initialized_service
    def exposed_lighting_info(
        self, emissive_lamps: int = EMISSIVE_LAMPS, emissive_patches: int = EMISSIVE_PATCHES
    ) -> dict[str, Any]:
        """Get the lighting of the scene, as needed to light a participating medium consistently with the scene.

        This includes sun, point, spot and area lights, whose intensities account for their exposure, temperature,
        volume factor and common node trees (see :meth:`_light_emission`), as well as the average radiance of the world
        background above the horizon, i.e. the sky (either a constant color or an environment texture). Other world
        shaders, square spots, and the elliptical cones of spots scaled unevenly are not supported, and approximated
        with a warning. Lighting is captured at the current frame, see :meth:`save_lighting <exposed_save_lighting>`
        for lighting that changes over frames, and :mod:`visionsim.medium` for its usage.

        Meshes that emit light, through an Emission shader or the emission of a Principled BSDF, possibly mixed or added
        with other shaders, are included as emissive surfaces. Their radiance is the emission's strength times its
        color, which can come from a Blackbody node, or from an image, whose mean color is used. Surfaces further than
        :data:`EMISSIVE_GAP` apart are grouped into different lamps, which each cast shadows from a single position, as
        long as there are at most ``emissive_lamps`` of them, and lamps are split into patches, whose light is
        proportional to the area of their faces seen from each direction, see :func:`_emissive_lamps`.

        Args:
            emissive_lamps (int, optional): Maximum number of lamps into which emissive surfaces are grouped.
                Defaults to :data:`EMISSIVE_LAMPS`.
            emissive_patches (int, optional): Number of patches into which emissive surfaces are split, in total.
                Defaults to :data:`EMISSIVE_PATCHES`.

        Returns:
            dict[str, Any]: Lighting information, following the schema of :class:`Lighting <visionsim.medium.model.Lighting>`.
        """
        return self._lighting(emissive_lamps, emissive_patches)[0]

    @require_initialized_service
    def _lighting(
        self, emissive_lamps: int, emissive_patches: int, warn: bool = True
    ) -> tuple[dict[str, Any], list[str]]:
        """Lighting of the scene at the current frame, see :meth:`lighting_info <exposed_lighting_info>`, and the names
        of the materials that emit light."""
        lights = self._lights(warn)
        # Emissive surfaces too faint next to the other lamps are left out
        emissive, materials, _ = self._emissive_surfaces(emissive_lamps, emissive_patches, _lamp_power(lights))
        return lights | {"emissive": emissive}, materials

    @require_initialized_service
    def _animated_lighting(
        self, frames: Iterable[int], emissive_lamps: int, emissive_patches: int
    ) -> tuple[dict[int, dict[str, Any]], list[str]]:
        """Lighting of the scene at the frames, among the given ones, at which it changes, and the names of the materials
        that emit light at any of them, see :meth:`save_lighting <exposed_save_lighting>`.

        Lights are exported at every frame, but emissive surfaces only again when the power of the other lamps changes,
        unless any of the objects that emit light, their parents or their materials are animated, in which case they are
        exported at every frame too. The current frame is restored afterwards.
        """
        current, frames = self.scene.frame_current, list(frames)
        changes: dict[int, dict[str, Any]] = {}
        names: set[str] = set()
        emissive: list[dict[str, Any]] | None = None
        previous, animated, power = None, False, None
        try:
            for frame in frames:
                self.scene.frame_set(frame)
                lights = self._lights(warn=False)
                if emissive is None or animated or _lamp_power(lights) != power:
                    power = _lamp_power(lights)
                    surfaces, materials, owners = self._emissive_surfaces(emissive_lamps, emissive_patches, power)
                    if emissive is None and owners and self._animated(owners, materials):
                        self.log.info(f"Emissive surfaces are animated, and exported again at all {len(frames)} frames.")
                        animated = True
                    emissive = surfaces
                    names.update(materials)
                lighting = lights | {"emissive": emissive}
                if lighting != previous:
                    changes[int(frame)] = previous = lighting
        finally:
            self.scene.frame_set(current)
        return changes, sorted(names)

    @staticmethod
    def _animated(objects: Collection[str], materials: Collection[str]) -> bool:
        """Whether any of these objects, or their parents, or these materials may change over time, as they are
        animated, driven, constrained or deformed."""
        deforming = {"ARMATURE", "CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT", "OCEAN", "WAVE", "NODES", "SIMULATION"}

        def moves(obj: bpy.types.Object | None) -> bool:
            while obj is not None:
                keys = getattr(obj.data, "shape_keys", None)
                if obj.animation_data or obj.constraints or (keys is not None and keys.animation_data):
                    return True
                if any(modifier.type in deforming for modifier in getattr(obj, "modifiers", [])):
                    return True
                obj = obj.parent
            return False

        if any(moves(bpy.data.objects.get(name)) for name in objects):
            return True
        found = (bpy.data.materials.get(name) for name in materials)
        if any(m.animation_data or (m.node_tree and m.node_tree.animation_data) for m in found if m is not None):
            return True
        # Materials that don't emit light yet could start to
        emitters = ("ShaderNodeEmission", "ShaderNodeBsdfPrincipled")
        return any(
            m.users
            and m.node_tree
            and m.node_tree.animation_data
            and any(n.bl_idname in emitters for n in m.node_tree.nodes)
            for m in bpy.data.materials
        )

    @require_initialized_service
    def _lights(self, warn: bool = True) -> dict[str, Any]:
        """Sun, point, spot and area lights of the scene, and its sky, at the current frame, see :meth:`lighting_info
        <exposed_lighting_info>`, warning about lights that are animated if ``warn``."""
        suns: list[dict[str, Any]] = []
        points: list[dict[str, Any]] = []
        spots: list[dict[str, Any]] = []
        areas: list[dict[str, Any]] = []

        for obj in self.scene.objects:
            if obj.type != "LIGHT" or obj.hide_render:
                continue
            light = obj.data
            if warn and (obj.animation_data or light.animation_data):
                self.log.warning(f"Light '{obj.name}' is animated, only its state at the current frame is saved.")

            scale = light.energy * 2 ** getattr(light, "exposure", 0.0) * getattr(light, "volume_factor", 1.0)
            emission, falloff = self._light_emission(obj)
            temperature = light.temperature_color if getattr(light, "use_temperature", False) else (1.0, 1.0, 1.0)
            color = [c * e * t * scale for c, e, t in zip(light.color, emission, temperature)]
            matrix = obj.matrix_world.to_3x3()
            # Lamps shine along their local -Z axis
            towards = (matrix @ mathutils.Vector((0.0, 0.0, -1.0))).normalized()
            position = list(obj.matrix_world.translation)
            # Without normalization, Cycles doesn't divide the power of lamps by their area
            normalize = getattr(light, "normalize", True)

            if light.type == "SUN":
                suns.append({"direction": list(-towards), "irradiance": color})
            elif light.type in ("POINT", "SPOT"):
                radius = light.shadow_soft_size
                power = color if normalize or radius == 0 else [c * np.pi * radius**2 for c in color]
                lamp = {"position": position, "power": power, "radius": radius, **falloff}
                if light.type == "POINT":
                    points.append(lamp)
                    continue
                # Cycles stretches the cone of spots along their local axes, along with their scale
                sx, sy, sz = obj.matrix_world.to_scale()
                if abs(sx - sy) > 1e-3 * max(abs(sx), abs(sy)) or light.use_square:
                    self.log.warning(f"The cone of spot light '{obj.name}' isn't circular, which isn't supported.")
                stretch = (abs(sx) + abs(sy)) / 2 / max(abs(sz), 1e-9)
                angle = 2 * np.arctan(np.tan(min(light.spot_size, np.pi - 1e-6) / 2) * stretch)
                spots.append(lamp | {"direction": list(towards), "angle": float(angle), "blend": light.spot_blend})
            elif light.type == "AREA":
                # The object's scale stretches area lights
                axis_u, axis_v = matrix @ mathutils.Vector((1.0, 0.0, 0.0)), matrix @ mathutils.Vector((0.0, 1.0, 0.0))
                ellipse = light.shape in ("DISK", "ELLIPSE")
                size_y = light.size_y if light.shape in ("RECTANGLE", "ELLIPSE") else light.size
                size = [light.size * axis_u.length, size_y * axis_v.length]
                area = size[0] * size[1] * (np.pi / 4 if ellipse else 1.0)
                areas.append(
                    {
                        "position": position,
                        "direction": list(towards),
                        "axis_u": list(axis_u.normalized()),
                        "size": size,
                        "shape": "ellipse" if ellipse else "rectangle",
                        "power": color if normalize else [c * area for c in color],
                        "spread": min(light.spread, np.pi),
                        **falloff,
                    }
                )
            else:
                self.log.warning(f"{light.type.title()} light '{obj.name}' is not supported and will be ignored.")

        return {"sky": self._world_radiance(), "suns": suns, "points": points, "spots": spots, "areas": areas}

    @require_initialized_service
    def exposed_save_lighting(
        self,
        path: str | os.PathLike | None = None,
        emissive_lamps: int = EMISSIVE_LAMPS,
        emissive_patches: int = EMISSIVE_PATCHES,
        frames: Iterable[int] | None = None,
    ) -> None:
        """Save the lighting of the scene, as returned by :meth:`lighting_info <exposed_lighting_info>`, to a JSON file.

        Given frames, such as those that are rendered, the lighting of the first of them is saved, and if the lighting
        changes over them, e.g. as lights move or their power changes, so is the lighting of each frame at which it
        changes, to ``animated-<name>`` next to the file, see :class:`AnimatedLighting
        <visionsim.medium.model.AnimatedLighting>`. Emissive surfaces are exported again at every frame only if any of
        the objects that emit light, their parents or their materials are animated, and their meshes are assumed not
        to deform otherwise.

        Args:
            path (str | os.PathLike | None, optional): Path of the JSON file. Defaults to ``lighting.json`` in the
                root directory of the renders.
            emissive_lamps (int, optional): Maximum number of lamps into which emissive surfaces are grouped, which
                should match that of :meth:`save_occlusion <exposed_save_occlusion>`. Defaults to
                :data:`EMISSIVE_LAMPS`.
            emissive_patches (int, optional): Number of patches into which emissive surfaces are split, in total.
                Defaults to :data:`EMISSIVE_PATCHES`.
            frames (Iterable[int] | None, optional): Frames whose lighting is saved. Defaults to None, i.e. only the
                current frame's.
        """
        path = Path(str(path)) if path else self.root_path / "lighting.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        animated = path.with_name(f"animated-{path.name}")

        frames = [int(frame) for frame in frames] if frames is not None else []
        changes = self._animated_lighting(frames, emissive_lamps, emissive_patches)[0] if frames else {}
        with open(path, "w") as f:
            first = changes[min(changes)] if changes else self.exposed_lighting_info(emissive_lamps, emissive_patches)
            json.dump(first, f, indent=2)
        if len(changes) > 1:
            with open(animated, "w") as f:
                json.dump({"frames": changes}, f, indent=2)
        elif animated.exists():
            # Lighting saved before, which changed over other frames
            animated.unlink()

    @contextmanager
    def _transparent(self, names: Collection[str]) -> Iterator[None]:
        """Make materials transparent while in this context, such that maps see through them."""
        changes = []
        try:
            for name in names:
                material = bpy.data.materials.get(name)
                if material is None or not material.use_nodes or not material.node_tree:
                    continue
                if not getattr(material, "is_editable", material.library is None):
                    self.log.warning(f"Material '{name}' is linked from a library, and hides its own light.")
                    continue
                tree = material.node_tree
                outputs = [n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
                output = next((n for n in outputs if n.is_active_output), outputs[0] if outputs else None)
                if output is None:
                    continue
                surface = output.inputs["Surface"]
                previous = surface.links[0].from_socket if surface.is_linked else None
                node = tree.nodes.new("ShaderNodeBsdfTransparent")
                changes.append((tree, node, surface, previous))
                tree.links.new(node.outputs[0], surface)
            yield
        finally:
            for tree, node, surface, previous in reversed(changes):
                tree.nodes.remove(node)
                if previous is not None:
                    tree.links.new(previous, surface)

    @staticmethod
    def _volume_only(obj: bpy.types.Object) -> bool:
        """Whether an object is only a volume, such as the medium itself when rendered by Cycles."""
        materials = [slot.material for slot in getattr(obj, "material_slots", []) if slot.material]
        outputs = [
            next((n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
            for m in materials
            if m.use_nodes and m.node_tree
        ]
        return bool(outputs) and all(
            o is not None and not o.inputs["Surface"].is_linked and o.inputs["Volume"].is_linked for o in outputs
        )

    @require_initialized_service
    def _shadow_casters(self, exclude: Collection[str]) -> tuple[list[bpy.types.Object], npt.NDArray[np.floating]]:
        """Objects that cast shadows onto a participating medium, and the corners of the box that contains them."""
        volume_only = self._volume_only
        depsgraph = bpy.context.evaluated_depsgraph_get()
        casters, corners = set(), []
        for instance in depsgraph.object_instances:
            obj = instance.object.original
            owner = instance.parent.original if instance.is_instance and instance.parent else obj
            if (
                instance.object.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD"}
                or obj.name in exclude
                or owner.name in exclude
                or owner.hide_render
                or not getattr(owner, "visible_shadow", True)
                or volume_only(obj)
            ):
                continue
            casters.add(owner)
            corners.extend(instance.matrix_world @ mathutils.Vector(corner) for corner in instance.object.bound_box)

        if not corners:
            return [], np.zeros((2, 3))
        points = np.asarray([tuple(corner) for corner in corners], dtype=float)
        return list(casters), np.stack([points.min(axis=0), points.max(axis=0)])

    @require_initialized_service
    def exposed_save_occlusion(
        self,
        path: str | os.PathLike | None = None,
        exclude: Collection[str] = (),
        sky_elevations: Sequence[float] = SKY_ELEVATIONS,
        sky_cells: Sequence[int] = SKY_CELLS,
        sun_resolution: int = 2048,
        sky_resolution: int = 512,
        lamp_resolution: int = 1024,
        emissive_lamps: int = EMISSIVE_LAMPS,
        emissive_patches: int = EMISSIVE_PATCHES,
        frames: Iterable[int] | None = None,
        lamp_spacing: float = 0.1,
    ) -> None:
        """Render and save shadow maps of the scene, through which its objects cast shadows onto a participating medium.

        Orthographic depth maps of the objects that cast shadows are rendered with Cycles along the direction of each
        sun, and along the central direction of each cell of the sky, which is split into bands of elevation, each
        split into equal ranges of azimuth. Maps span the box that contains these objects. The distance to the first
        object in every direction around each lamp (point, spot and area lights, and groups of emissive surfaces, see
        :meth:`lighting_info <exposed_lighting_info>`) is also rendered, in equirectangular maps, those of emissive
        surfaces seeing through them as if they were transparent, as they don't hide their own light. All maps are saved
        to a ``.npz`` file, see :mod:`visionsim.medium.occlusion` for how they are used. Maps are captured at the
        current frame, or given frames, at the first of them, but for lamps that move over them, whose maps are rendered
        at the frames from which they move further than ``lamp_spacing`` from where their maps were rendered, as
        lighting is saved by :meth:`save_lighting <exposed_save_lighting>`. Objects should be static otherwise. Render
        settings, the camera, the current frame and the compositor are restored afterwards.

        Args:
            path (str | os.PathLike | None, optional): Path of the ``.npz`` file. Defaults to ``occlusion.npz`` in the
                root directory of the renders.
            exclude (Collection[str], optional): Names of objects that should not cast shadows, typically a large ground
                plane, which would otherwise make the maps span all of it. Light from below the horizon is ignored
                anyway. Defaults to none.
            sky_elevations (Sequence[float], optional): Elevations, in degrees, at the edges of the bands of the sky,
                from the horizon up. Defaults to :data:`SKY_ELEVATIONS`.
            sky_cells (Sequence[int], optional): Number of cells of each band. Defaults to :data:`SKY_CELLS`.
            sun_resolution (int, optional): Number of texels along the longest side of the maps of suns.
                Defaults to 2048.
            sky_resolution (int, optional): Number of texels along the longest side of the maps of the sky.
                Defaults to 512.
            lamp_resolution (int, optional): Number of texels along the width of the maps of lamps, which span all
                azimuths, and half as many along their height, which spans all elevations. Defaults to 1024.
            emissive_lamps (int, optional): Maximum number of lamps into which emissive surfaces are grouped, which
                should match that of :meth:`save_lighting <exposed_save_lighting>`. Defaults to :data:`EMISSIVE_LAMPS`.
            emissive_patches (int, optional): Number of patches into which emissive surfaces are split, in total.
                Defaults to :data:`EMISSIVE_PATCHES`.
            frames (Iterable[int] | None, optional): Frames over which lamps can move, see :meth:`save_lighting
                <exposed_save_lighting>`. Defaults to None, i.e. only the current frame.
            lamp_spacing (float, optional): Distance, in meters, from which a moving lamp gets a new map, rather than
                casting the shadows of the nearest one. Defaults to 0.1.

        Raises:
            ValueError: raised if the number of bands of the sky and of their edges do not match.
        """
        if len(sky_elevations) != len(sky_cells) + 1:
            raise ValueError(f"Expected {len(sky_cells) + 1} elevations for {len(sky_cells)} bands of the sky.")

        path = Path(str(path)) if path else self.root_path / "occlusion.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        edges = np.sin(np.radians(np.asarray(sky_elevations, dtype=float)))
        frames = [int(frame) for frame in frames] if frames is not None else []
        if frames:
            changes, emissive_materials = self._animated_lighting(frames, emissive_lamps, emissive_patches)
        else:
            lighting, emissive_materials = self._lighting(emissive_lamps, emissive_patches)
            changes = {self.scene.frame_current: lighting}
        first = min(changes)
        suns = [np.asarray(sun["direction"], dtype=float) for sun in changes[first]["suns"]]
        suns = [sun / np.linalg.norm(sun) for sun in suns]
        if any(len(lit["suns"]) != len(suns) for lit in changes.values()) or any(
            np.degrees(
                np.arccos(np.clip(np.dot(sun, np.asarray(s["direction"]) / np.linalg.norm(s["direction"])), -1, 1))
            )
            > 0.1
            for lit in changes.values()
            for sun, s in zip(suns, lit["suns"])
        ):
            self.log.warning("Suns move over the frames, but cast the shadows of the first frame.")
        # Jobs are rendered in order, at their frame, the last ones seeing through emissive surfaces
        jobs = [("sun", sun, sun_resolution, first) for sun in suns]
        jobs += [("sky", d, sky_resolution, first) for d in _sky_cell_directions(edges, sky_cells)]
        for kind in ("lamp", "emissive"):
            # Each lamp gets a map wherever it is further than the spacing from where its maps were rendered
            placed: dict[int, list[npt.NDArray[np.floating]]] = {}
            for frame, lit in sorted(changes.items()):
                # Area lights shine from their surface, from which the map is rendered just in front
                lamps = [np.asarray(lamp["position"], dtype=float) for lamp in lit["points"] + lit["spots"]]
                lamps += [np.asarray(a["position"]) + 1e-3 * np.asarray(a["direction"]) for a in lit["areas"]]
                positions = (
                    lamps if kind == "lamp" else [np.asarray(e["position"], dtype=float) for e in lit["emissive"]]
                )
                for k, position in enumerate(positions):
                    if all(np.linalg.norm(position - other) > lamp_spacing for other in placed.setdefault(k, [])):
                        placed[k].append(position)
                        jobs.append((kind, position, lamp_resolution, frame))

        casters, bounds = self._shadow_casters(exclude)
        if not casters:
            self.log.warning("No object casts shadows, saving empty shadow maps.")
            jobs = []
        center = bounds.mean(axis=0)
        corners = np.asarray(list(itertools.product(*bounds.T)), dtype=float)
        radius = float(np.linalg.norm(corners - center, axis=1).max())
        maps: dict[str, list] = {"sun": [], "sky": [], "lamp": [], "emissive": []}
        current = self.scene.frame_current

        # Render depth only, with a single sample at the center of each texel, and save it instead of the frame
        scene, render, cycles = self.scene, self.scene.render, self.scene.cycles
        settings = [
            (render, "engine", "CYCLES"),
            (render, "resolution_percentage", 100),
            (render, "use_motion_blur", False),
            (render, "use_border", False),
            (render.image_settings, "file_format", "OPEN_EXR"),
            (render.image_settings, "color_depth", "32"),
            (render.image_settings, "color_mode", "RGB"),
            (render.image_settings, "exr_codec", "ZIP"),
            (cycles, "samples", 1),
            (cycles, "use_adaptive_sampling", False),
            (cycles, "use_denoising", False),
            (cycles, "pixel_filter_type", "BOX"),
            (cycles, "filter_width", 0.01),
            (cycles, "max_bounces", 0),
            (self.view_layer, "use_pass_z", True),
            # The depth of transparent surfaces isn't recorded, but that of the surfaces behind them
            (self.view_layer, "pass_alpha_threshold", 0.5),
            # Without a world, which may hold a volume too, rays that miss every object record a huge depth
            (scene, "world", None),
        ]
        restore = [(owner, name, getattr(owner, name)) for owner, name, _ in settings]
        restore += [(render, "resolution_x", render.resolution_x), (render, "resolution_y", render.resolution_y)]
        restore += [(render, "filepath", render.filepath), (scene, "camera", scene.camera)]
        restore += [
            (node, "mute", node.mute) for node in self.tree.nodes if node.bl_idname == "CompositorNodeOutputFile"
        ]
        # Volumes, such as the medium itself when rendered by Cycles, would scatter some of the rays of the maps before
        # they reach any surface, leaving holes in the maps
        hidden = [
            obj
            for obj in scene.objects
            if obj.name in exclude or not getattr(obj, "visible_shadow", True) or self._volume_only(obj)
        ]
        restore += [(obj, "hide_render", obj.hide_render) for obj in hidden]
        restore += [(obj, "visible_camera", obj.visible_camera) for obj in casters]
        output = next(n for n in self.tree.nodes if n.bl_idname in ("NodeGroupOutput", "CompositorNodeComposite"))
        linked = [link.from_socket for link in output.inputs[0].links]
        camera = bpy.data.objects.new("Occlusion", bpy.data.cameras.new("Occlusion"))

        try:
            for owner, name, value in settings:
                setattr(owner, name, value)
            for node in self.tree.nodes:
                if node.bl_idname == "CompositorNodeOutputFile":
                    node.mute = True
            for obj in hidden:
                obj.hide_render = True
            for obj in casters:
                obj.visible_camera = True
            self.tree.links.new(self.render_layers.outputs["Depth"], output.inputs[0])
            scene.collection.objects.link(camera)
            scene.camera = camera
            camera.data.sensor_fit, camera.data.clip_start = "AUTO", 0.01
            if hasattr(camera.data, "panorama_type"):
                camera.data.panorama_type = "EQUIRECTANGULAR"
            else:
                camera.data.cycles.panorama_type = "EQUIRECTANGULAR"

            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                see_through = False
                for index, (kind, target, resolution, frame) in enumerate(jobs):
                    if frame != scene.frame_current:
                        scene.frame_set(frame)
                    if kind == "emissive" and not see_through:
                        stack.enter_context(self._transparent(emissive_materials))
                        see_through = True
                    if kind in ("lamp", "emissive"):
                        # Equirectangular camera looking along +X with +Z up, whose texels map to world directions as
                        # those of environment textures do, at the lamp's position, and whose depth is the distance
                        # to the camera, see `visionsim.medium.occlusion.lamp_visibility`
                        rotation = mathutils.Matrix(((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
                        axes, origin = np.asarray(rotation, dtype=float).T, target
                        width, height, texel = resolution, max(1, resolution // 2), 2 * np.pi / resolution
                        camera.data.type = "PANO"
                        camera.data.clip_end = float(np.linalg.norm(corners - origin, axis=1).max()) + 1
                    else:
                        rotation = mathutils.Vector(target.tolist()).to_track_quat("Z", "Y").to_matrix()
                        axes = np.asarray(rotation, dtype=float).T  # rows: right, up, and towards the light
                        extent = np.abs((corners - center) @ axes[:2].T).max(axis=0)
                        texel = 2 * float(extent.max()) / resolution
                        width, height = (max(1, int(np.ceil(2 * e / texel - 1e-6))) for e in extent)
                        origin = center + axes[2] * (radius + 1)
                        camera.data.type, camera.data.clip_end = "ORTHO", 2 * radius + 2
                        camera.data.ortho_scale = max(width, height) * texel
                    camera.matrix_world = mathutils.Matrix.Translation(origin.tolist()) @ rotation.to_4x4()
                    render.resolution_x, render.resolution_y = width, height
                    render.filepath = str(Path(root) / f"{index}.exr")
                    bpy.ops.render.render(write_still=True)

                    image = bpy.data.images.load(render.filepath)
                    pixels = np.empty(image.size[0] * image.size[1] * image.channels, dtype=np.float32)
                    image.pixels.foreach_get(pixels)
                    depth = pixels.reshape(image.size[1], image.size[0], image.channels)[..., 0].copy()
                    bpy.data.images.remove(image)
                    maps[kind].append((depth, origin, axes, texel))
        finally:
            for link in list(output.inputs[0].links):
                self.tree.links.remove(link)
            for socket in linked:
                self.tree.links.new(socket, output.inputs[0])
            for owner, name, value in restore:
                setattr(owner, name, value)
            camera_data = camera.data
            bpy.data.objects.remove(camera, do_unlink=True)
            bpy.data.cameras.remove(camera_data)
            if scene.frame_current != current:
                scene.frame_set(current)

        def arrays(kind: str) -> dict[str, npt.NDArray]:
            depths = [depth for depth, *_ in maps[kind]]
            return {
                f"{kind}_depths": np.concatenate([d.reshape(-1) for d in depths] or [np.zeros(0)]).astype(np.float32),
                f"{kind}_offsets": np.cumsum([0] + [d.size for d in depths[:-1]]) if depths else np.zeros(0, int),
                f"{kind}_shapes": np.asarray([d.shape for d in depths], dtype=int).reshape(-1, 2),
                f"{kind}_origins": np.asarray([o for _, o, _, _ in maps[kind]], dtype=float).reshape(-1, 3),
                f"{kind}_axes": np.asarray([a for _, _, a, _ in maps[kind]], dtype=float).reshape(-1, 3, 3),
                f"{kind}_texels": np.asarray([t for *_, t in maps[kind]], dtype=float),
            }

        np.savez_compressed(
            path,
            **arrays("sun"),
            **arrays("sky"),
            **arrays("lamp"),
            **arrays("emissive"),
            sun_directions=np.asarray(suns if jobs else [], dtype=float).reshape(-1, 3),
            sky_edges=edges,
            sky_counts=np.asarray(sky_cells if jobs else [0] * len(sky_cells), dtype=int),
            bounds=bounds,
            frame=first,
            lamp_spacing=lamp_spacing,
        )

    @require_initialized_service
    @validate_camera_moved
    def exposed_position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        """Positions and orients camera according to specified parameters. All transformations are local,
        use :meth:`unbind_camera <exposed_unbind_camera>` to ensure position is set in world coordinates.

        Note: Only one of ``look_at`` or ``rotation`` can be set at once.

        Args:
            location (npt.ArrayLike, optional): Location to place camera in 3D space. Defaults to none.
            rotation (npt.ArrayLike, optional): Rotation matrix for camera. Defaults to none.
            look_at (npt.ArrayLike, optional): Location to point camera. Defaults to none.
            in_order (bool, optional): If set, assume current camera pose is from previous/next
                frame and ensure new rotation set by ``look_at`` is compatible with current position.
                Without this, a rotations will stay in the [-pi, pi] range and this wrapping will
                mess up interpolations. Only used when ``look_at`` is set. Defaults to True.

        Raises:
            ValueError: raised if camera orientation is over-defined.
        """
        if look_at is not None and rotation is not None:
            raise ValueError("Only one of `look_at` or `rotation` can be set.")

        if location is not None:
            self.camera.location = mathutils.Vector(location)

        if look_at is not None:
            # point the camera's '-Z' towards `look_at` and use its 'Y' as up
            direction = mathutils.Vector(look_at) - self.camera.location
            rot_euler = direction.to_track_quat("-Z", "Y").to_euler()

            if in_order:
                rot_euler.make_compatible(self.camera.rotation_euler)
            self.camera.rotation_euler = rot_euler

        if rotation is not None:
            location = self.camera.location.copy()
            self.camera.matrix_world = mathutils.Matrix(rotation).to_4x4()
            self.camera.location = location
        self.view_layer.update()

    @require_initialized_service
    @validate_camera_moved
    def exposed_rotate_camera(self, angle: float) -> None:
        """Rotate camera around it's optical axis, relative to current orientation. All transformations are local,
        use :meth:`unbind_camera <exposed_unbind_camera>` to ensure position is set in world coordinates.

        Args:
            angle: Relative amount to rotate by (clockwise, in radians).
        """
        # Camera's '-Z' point outwards, so we negate angle such that
        # camera turns clockwise for a positive angle
        rot_euler = self.camera.rotation_euler.copy()
        rot_euler.rotate_axis("Z", -angle)
        rot_euler.make_compatible(self.camera.rotation_euler)
        self.camera.rotation_euler = rot_euler
        self.view_layer.update()

    @require_initialized_service
    @validate_camera_moved
    def exposed_offset_camera(self, offset: npt.ArrayLike) -> None:
        """Move camera by a given vector in its local coordinate frame.

        Args:
            offset (npt.ArrayLike): Amount to offset by (x, y, z) in local coordinates.
        """
        self.camera.location += self.camera.matrix_world.to_3x3() @ mathutils.Vector(offset)
        self.view_layer.update()

    @require_initialized_service
    def exposed_set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        """Set camera keyframe at given frame number.
        If camera matrix is not supplied, currently set camera position/rotation/scale will be used,
        this allows users to set camera position using :meth:`position_camera <exposed_position_camera>`
        and :meth:`rotate_camera <exposed_rotate_camera>`.

        Args:
            frame_num (int): index of frame to set keyframe for.
            matrix (npt.ArrayLike | None, optional): 4x4 camera transform, if not supplied,
                use current camera matrix. Defaults to None.
        """
        if matrix is not None:
            self.camera.matrix_world = mathutils.Matrix(matrix)
        self.camera.keyframe_insert(data_path="location", frame=frame_num)
        self.camera.keyframe_insert(data_path="rotation_euler", frame=frame_num)
        self.camera.keyframe_insert(data_path="scale", frame=frame_num)

    @require_initialized_service
    def exposed_set_animation_range(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> None:
        """Set animation range for scene.

        Args:
            start (int | None, optional): frame start, inclusive. Defaults to None.
            stop (int | None, optional): frame stop, exclusive. Defaults to None.
            step (int | None, optional): frame interval. Defaults to None.
        """
        if start is not None:
            self.scene.frame_start = start
        if stop:
            self.scene.frame_end = stop - 1
        if step is not None:
            self.scene.frame_step = step

    @require_initialized_service
    def exposed_render_current_frame(self, allow_skips=True, dry_run=False) -> None:
        """Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth, normals, etc.

        Note:
            This method renders the current frame as-is, it assumes the camera position,
            frame number and all other parameters have been set.

        Args:
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
        """
        folder_index = f"{self.scene.frame_current // ITEMS_PER_SUBFOLDER:04}"
        frame_index = f"{self.scene.frame_current % ITEMS_PER_SUBFOLDER:0{INDEX_PADDING}}"
        paths = {}

        if not self._outputs and not self._warned_no_outputs:
            self.log.warning(
                "No outputs are selected, so nothing will be rendered. "
                "Consider including different types of ground truth annotations using "
                "the `include_X` methods (eg: `include_frames`) or equivalently using "
                "`--config.include-frames` if using the CLI."
            )
            self._warned_no_outputs = True

        for subpath, (node, slot, *_) in self._outputs.items():
            if subpath == "composites":
                self.scene.render.filepath = str(self.root_path / subpath / folder_index / frame_index)
                paths[subpath] = Path(self.scene.render.filepath).with_suffix(self.scene.render.file_extension)
            else:
                if bpy.app.version >= (5, 0, 0):
                    node.directory = str(self.root_path / subpath / folder_index)
                else:
                    node.base_path = str(self.root_path / subpath / folder_index)
                slot.name = str(Path(slot.name).with_stem(frame_index).name)
                paths[subpath] = self.root_path / subpath / folder_index / slot.name

        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not dry_run:  # noqa: SIM102
            # Render frame(s), skip the render iff all files exist and `allow_skips`
            if not allow_skips or any(not Path(self.root_path / p).exists() for p in paths.values()):
                # If `write_still` is false, depth/normals/etc can be written but composites will be skipped
                bpy.ops.render.render(animation=False, write_still="composites" in self._outputs)

        # Before Blender 5.0 file output nodes ALWAYS appended the frame number
        # to the filename making our path incorrect, here we rename the file.
        # This does not apply to composites as they are not saved via a output file node.
        # See: https://projects.blender.org/blender/blender/issues/134920
        if bpy.app.version < (5, 0, 0):
            for subpath, path in paths.items():
                if subpath != "composites":
                    path.with_stem(f"{self.scene.frame_current:04}").rename(path)

        # Update databases with frame paths and camera info
        self._save_metadata(
            paths={k: p.relative_to(self.root_path / k) for k, p in paths.items()},
            transform_matrix=self.exposed_camera_extrinsics().tolist(),
            camera_info=self.exposed_camera_info(),
            index=self.scene.frame_current,
        )

    @require_initialized_service
    def exposed_render_frame(self, frame_number: int, allow_skips=True, dry_run=False) -> None:
        """Same as first setting current frame then rendering it.

        Warning:
            Calling this has the side-effect of changing the current frame.

        Args:
            frame_number (int): frame to render
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
        """
        self.exposed_set_current_frame(frame_number)
        self.exposed_render_current_frame(allow_skips=allow_skips, dry_run=dry_run)

    @require_initialized_service
    def exposed_render_frames(
        self, frame_numbers: Iterable[int], allow_skips=True, dry_run=False, update_fn: UpdateFn | None = None
    ) -> None:
        """Render all requested frames and return associated transforms dictionary.

        Args:
            frame_numbers (Iterable[int]): frames to render.
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
            update_fn (UpdateFn, optional): callback function to track render progress. Will first be called with ``total`` kwarg,
                indicating number of steps to be taken, then will be called with ``advance=1`` at every step. Closely mirrors the
                `rich.Progress API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
                Defaults to None.

        Raises:
            RuntimeError: raised if trying to render frames beyond blender's limits.
        """
        # Ensure frame_numbers is a list to find extrema
        frame_numbers = list(frame_numbers)

        # Set total number of steps
        if update_fn is not None:
            update_fn(total=len(frame_numbers))

        # Max number of frames is 1,048,574 as of Blender 3.4
        # See: https://docs.blender.org/manual/en/latest/advanced/limits.html
        if (frame_end := max(frame_numbers)) >= 1_048_574:
            raise RuntimeError(
                f"Blender cannot currently render more than 1,048,574 frames, yet requested you "
                f"requested {max(frame_numbers)} frames to be rendered. For more please see: "
                f"https://docs.blender.org/manual/en/latest/advanced/limits.html"
            )
        if (frame_start := min(frame_numbers)) < 0:
            raise RuntimeError("Cannot render frames at negative indices. You can try shifting keyframes.")

        # Warn if requested frames lie outside animation range
        if self._use_animation and (frame_start < self.scene.frame_start or self.scene.frame_end < frame_end):
            self.log.warning(
                f"Current animation starts at frame #{self.scene.frame_start} and ends at "
                f"#{self.scene.frame_end} (with step={self.scene.frame_step}), but you requested "
                f"some frames between #{frame_start} and to #{frame_end} to be rendered.\n"
            )

        # If we request to render frames outside the nominal animation range,
        # blender will just wrap around and go back to that range. As a workaround,
        # extend the animation range to it's maximum, even if we do not exceed it,
        # it will be restored at the end.
        scene_original_range = self.scene.frame_start, self.scene.frame_end
        self.scene.frame_start, self.scene.frame_end = 0, 1_048_574

        # Capture frames!
        for frame_number in frame_numbers:
            # Tell blender to update camera position and all animations and render frame
            self.exposed_render_frame(frame_number, allow_skips=allow_skips, dry_run=dry_run)

            # Call any progress callbacks
            if update_fn is not None:
                update_fn(advance=1)

        # Restore animation range to original values
        self.scene.frame_start, self.scene.frame_end = scene_original_range

    @require_initialized_service
    def exposed_render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips=True,
        dry_run=False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        """Determines frame range to render, sets camera positions and orientations, and renders all frames in animation range.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling :meth:`move_keyframes(scale=2.0) <exposed_move_keyframes>`
              then calling :meth:`render_animation(frame_start=1, frame_end=100) <exposed_render_animation>` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start (int, optional): Starting index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_end (int, optional): Ending index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_step (int, optional): Skip every nth frame. Defaults to None, meaning value from ``.blend`` file.
            allow_skips (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            dry_run (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            update_fn (UpdateFn, optional): Same as :meth:`render_frames <exposed_render_frames>`.

        Raises:
            ValueError: raised if scene and camera are entirely static.
        """
        frame_start = self.scene.frame_start if frame_start is None else frame_start
        frame_end = self.scene.frame_end if frame_end is None else frame_end
        frame_step = self.scene.frame_step if frame_step is None else frame_step
        frame_range = range(frame_start, frame_end + 1, frame_step)

        if not self._use_animation:
            raise ValueError(
                "Animations are disabled, scene will be entirely static. "
                "To instead render a single frame, use `render_frame`."
            )
        elif all(p.animation_data is None for p in self.get_parents(self.camera)) and self.camera.animation_data is None:
            self.log.warning("Active camera nor it's parents are animated, camera will be static.")

        self.exposed_render_frames(frame_range, allow_skips=allow_skips, dry_run=dry_run, update_fn=update_fn)

    @require_initialized_service
    def exposed_save_file(self, path: str | os.PathLike) -> None:
        """Save the opened blender file. This is useful for introspecting the state of the compositor/scene/etc.

        Args:
            path (str | os.PathLike): path where to save blendfile.

        Raises:
            ValueError: raised if file already exists.
        """
        if (path := Path(str(path)).resolve()) == Path(str(self.blend_file)).resolve():
            raise ValueError("Cannot overwrite currently loaded blend-file!")

        self.log.info(f"Saving scene to {path}...")
        path.parent.mkdir(exist_ok=True, parents=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))


class BlenderClient:
    """Client-side API to interact with blender and render novel views.

    The :class:`BlenderClient` is responsible for communicating with (and potentially spawning)
    separate :class:`BlenderServer`s that will actually perform the rendering via a :class:`BlenderService`.

    The client acts as a context manager, it will connect to it's server when the context is
    entered and cleanly disconnect and close the connection in case of errors or when exiting
    the with-block.

    Many useful methods to interact with blender are provided, such as
    :meth:`set_resolution <BlenderService.exposed_set_resolution>` or
    :meth:`render_animation <BlenderService.exposed_render_animation>`.
    These methods are dynamically generated when the client connects to
    the server. Available methods are directly inherited from :class:`BlenderService`
    (or whichever service the server is exposing), specifically any service method
    starting with ``exposed_`` will be accessible to the client at runtime.
    For example, ``BlenderClient.include_depths`` is a remote procedure call
    to :meth:`BlenderService.exposed_include_depths`.
    """

    def __init__(self, addr: tuple[str, int], timeout: float = 10.0) -> None:
        """Initialize a client with known address of server.
        Note: Using :meth:`auto_connect` or :meth:`spawn` is often more convenient.

        Args:
            addr (tuple[str, int]): Connection tuple containing the hostname and port
            timeout (float, optional): Maximum time in seconds the client will attempt
                to connect to the server for before an error is thrown. Only used when
                entering context manager. Defaults to 10 seconds.
        """
        self.addr: tuple[str, int] = addr
        self.conn: rpyc.Connection | None = None
        self.awaitable: rpyc.AsyncResult | None = None
        self.process: subprocess.Popen | None = None
        self.timeout: float = timeout

    @classmethod
    def auto_connect(cls, timeout: float = 10.0) -> Self:
        """Automatically connect to available server.

        Use :meth:`BlenderServer.discover` to find available server within ``timeout``.

        Note: This doesn't actually connect to the server instance, the connection happens
            when the context manager is entered. This simply creates a client instance with
            the connection settings (i.e: hostname, port) of an existing server. The connection
            might still fail when entering the with-block.

        Args:
            timeout (float, optional): try to discover server instance for ``timeout``
                (in seconds) before giving up. Defaults to 10.0 seconds.

        Raises:
            TimeoutError: raise if unable to discover server in ``timeout`` seconds.

        Returns:
            Self: client instance initialized with connection settings of existing server.
        """
        start = time.time()

        while True:
            if (time.time() - start) > timeout:
                raise TimeoutError("Unable to discover server in alloted time.")
            if conns := set(BlenderServer.discover()):
                break
            time.sleep(0.1)
        return cls(conns.pop())

    @classmethod
    @contextmanager
    def spawn(
        cls,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = subprocess.DEVNULL,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Generator[Self]:
        """Spawn and connect to a blender server.
        The spawned process is accessible through the client's ``process`` attribute.

        Args:
            timeout (float, optional): try to discover spawned instances for ``timeout``
                (in seconds) before giving up. If negative, a port will be randomly selected and assigned to the
                spawned server, bypassing the need for discovery and timeouts. Note that when a port is assigned
                this context manager will immediately yield, even if the server is not yet ready to accept
                incoming connections. Defaults to assigning a port to spawned server (-1 seconds).
            log (str | os.PathLike | FILE | tuple[FILE, FILE], optional): path to log directory, file handle,
                descriptor or tuple thereof. Stdout and stderr will be captured and saved if supplied.
                Defaults to subprocess.DEVNULL for both stdout/stderr.
            autoexec (bool, optional): if true, allow execution of any embedded python scripts within blender.
                For more, see blender's CLI documentation. Defaults to False.
            executable (str | os.PathLike | None, optional): path to Blender's executable. Defaults to looking
                for blender on $PATH, but is useful when targeting a specific blender install, or when it's installed
                via a package manager such as flatpak. Setting it to "flatpak run --die-with-parent org.blender.Blender"
                might be required when using flatpaks. Defaults to None (system PATH).

        Yields:
            Generator[Self]: the connected client
        """
        with (
            BlenderServer.spawn(jobs=1, timeout=timeout, log=log, autoexec=autoexec, executable=executable) as (
                procs,
                conns,
            ),
            cls(conns.pop()) as client,
        ):
            client.process = procs[0]
            yield client
            client.process = None

    @require_connected_client
    def render_animation_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        """Asynchronously call :meth:`render_animation <BlenderService.exposed_render_animation>`
        and return an rpyc.AsyncResult.

        Args:
            *args: Same as :meth:`BlendService.exposed_render_animation`
            *kwargs: Same as :meth:`BlendService.exposed_render_animation`

        Returns:
            rpyc.AsyncResult: Result encapsulating the return value of ``render_animation``.
                After ``wait``ing for the render to finish, it can be accessed using
                the ``.value`` attribute.
        """
        self.conn = cast(rpyc.Connection, self.conn)
        render_animation_async = rpyc.async_(self.conn.root.render_animation)
        async_result = render_animation_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    @require_connected_client
    def render_frames_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        """Asynchronously call :meth:`render_frames <BlenderService.exposed_render_frames>`
        and return an rpyc.AsyncResult.

        Args:
            *args: Same as :meth:`BlendService.exposed_render_frames`
            *kwargs: Same as :meth:`BlendService.exposed_render_frames`

        Returns:
            rpyc.AsyncResult: Result encapsulating the return value of ``render_frames``.
                After ``wait``ing for the render to finish, it can be accessed using
                the ``.value`` attribute.
        """
        self.conn = cast(rpyc.Connection, self.conn)
        render_frames_async = rpyc.async_(self.conn.root.render_frames)
        async_result = render_frames_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    def wait(self) -> None:
        """Block and await any async results."""
        if self.awaitable:
            self.awaitable.wait()

    def __enter__(self) -> Self:
        """Connect to the render server via a context manager.

        Raises:
            TimeoutError: raised if unable to connect to server in time.
        """
        # Loop until we connect or timeout
        start = time.time()

        while True:
            try:
                self.conn = rpyc.connect(*self.addr, config={"sync_request_timeout": -1, "allow_all_attrs": True})
                break
            except ConnectionRefusedError:
                pass

            if (time.time() - start) > self.timeout:
                raise TimeoutError("Unable to connect to server in alloted time.")
            time.sleep(0.1)

        # Setup default logger for client, otherwise warnings
        # in the render service won't propagate to the client
        logger = logging.getLogger(__name__ + str(self.addr))
        logger.setLevel(logging.INFO)
        self.with_logger(logger)
        return self

    def __getattr__(self, name: str) -> rpyc.BaseNetref:
        """Retrieve remote attribute if client is connected.
        This will be called when local attribute is not found.

        Args:
            name (str): Name of attribute.

        Raises:
            AttributeError: raised if attribute is not found.

        Returns:
            rpyc.BaseNetref: remote proxy object.
        """
        # Spoof all `exposed_` methods from the service
        if self.conn is not None and EXPOSED_PREFIX + name in dir(self.conn.root):
            return getattr(self.conn.root, EXPOSED_PREFIX + name)
        raise AttributeError

    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Disconnect to the render server via a context manager.

        Args:
            type (type[BaseException] | None): Type of exception that was caught, if any.
            value (BaseException | None): Value of exception if any.
            traceback (TracebackType | None): Traceback of exception if any.
        """
        if self.conn is not None:
            self.conn.close()


class BlenderClients(tuple):
    """Collection of :class:`BlenderClient` instances.

    Most methods in this class simply call the equivalent method of each client, that is,
    calling ``clients.set_resolution`` is equivalent to calling :meth:`set_resolution <BlenderService.exposed_set_resolution>`
    for each client in clients. Some special methods, namely the :meth:`render_frames` and :meth:`render_animation`
    methods will instead distribute the rendering load to all clients.

    Finally, entering each client's context-manager, and closing each client connection
    is ensured by using this class' context-manager.
    """

    def __new__(cls, *objs: Iterator[BlenderClient | tuple[str, int]]) -> Self:
        """Create a new instance from iterable of clients, or their connection settings.

        Args:
            *objs (Iterator[BlenderClient | tuple[str, int]]): :class:`BlenderClient` instances or their hostnames and ports.

        Raises:
            TypeError: raised when input objects are of incorrect type.
        """
        clients = [BlenderClient(o) if isinstance(o, tuple) else o for o in objs]
        if not all(isinstance(o, BlenderClient) for o in clients):
            raise TypeError("'BlenderClients' can only contain 'BlenderClient' instances or their hostnames and ports.")
        return super().__new__(cls, clients)

    def __init__(self, *objs) -> None:
        """Initialize collection of :class:`BlenderClient` from iterable of clients, or their connection settings.

        Args:
            *objs (Iterator[BlenderClient | tuple[str, int]]): :class:`BlenderClient` instances or their hostnames and ports.
        """
        # Note: At this point the tuple is already initialized because of __new__, i.e: objs == list(self)
        # Not sure why mypy can't resolve the following...
        self.stack: ExitStack = ExitStack()

    def _method_dispatch_factory(self, name: str, method: Callable) -> Callable:
        @functools.wraps(method)
        def inner(*args, **kwargs):
            # Call method for each client, collect results into tuple
            retval = tuple(getattr(client, name)(*args, **kwargs) for client in self)
            return None if all(i is None for i in retval) else retval

        return inner

    def __enter__(self) -> Self:
        """Connect all clients to their render servers via a context manager.

        Raises:
            TimeoutError: raised if unable to connect to servers in time.
        """
        self.stack.__enter__()
        for client in self:
            # Enter each client's context, connecting them all to servers
            self.stack.enter_context(client)

            # Dynamically generate methods that dispatch to all clients
            # TODO: We currently assume all clients use `BlenderService`.
            # TODO: Move this to a __getattr__ method like in BlenderClient!
            for method_name in dir(BlenderService):
                if method_name.startswith(EXPOSED_PREFIX):
                    name = method_name.removeprefix(EXPOSED_PREFIX)

                    if name not in dir(self):
                        method = getattr(BlenderService, method_name)
                        multicall = self._method_dispatch_factory(name, method)
                        setattr(self, name, multicall)
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Disconnect from each render server via a context manager.

        Args:
            type (type[BaseException] | None): Type of exception that was caught, if any.
            value (BaseException | None): Value of exception if any.
            traceback (TracebackType | None): Traceback of exception if any.
        """
        self.stack.__exit__(type, value, traceback)

    @classmethod
    @contextmanager
    def spawn(
        cls,
        jobs: int = 1,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = subprocess.DEVNULL,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Generator[Self]:
        """Spawn and connect to one or more blender servers.
        The spawned processes are accessible through the client's ``process`` attribute.

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for ``timeout``
                (in seconds) before giving up. If negative, a port will be randomly selected and assigned to the
                spawned server, bypassing the need for discovery and timeouts. Note that when a port is assigned
                this context manager will immediately yield, even if the server is not yet ready to accept
                incoming connections. Defaults to assigning a port to spawned server (-1 seconds).
            log (str | os.PathLike | FILE | tuple[FILE, FILE], optional): path to log directory, file handle,
                descriptor or tuple thereof. Stdout and stderr will be captured and saved if supplied.
                Defaults to subprocess.DEVNULL for both stdout/stderr.
            autoexec (bool, optional): if true, allow execution of any embedded python scripts within blender.
                For more, see blender's CLI documentation. Defaults to False.
            executable (str | os.PathLike | None, optional): path to Blender's executable. Defaults to looking
                for blender on $PATH, but is useful when targeting a specific blender install, or when it's installed
                via a package manager such as flatpak. Setting it to "flatpak run --die-with-parent org.blender.Blender"
                might be required when using flatpaks. Defaults to None (system PATH).

        Yields:
            Generator[Self]: the connected clients
        """
        with BlenderServer.spawn(jobs=jobs, timeout=timeout, log=log, autoexec=autoexec, executable=executable) as (  # noqa: SIM117
            procs,
            conns,
        ):
            with cls(*conns) as clients:
                for client, p in zip(clients, procs):
                    client.process = p

                yield clients

                for client in clients:
                    client.process = None

    @contextmanager
    @staticmethod
    def pool(
        jobs: int = 1,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = subprocess.DEVNULL,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
        conns: list[tuple[str, int]] | None = None,
    ) -> Generator[multiprocess.Pool]:
        """Spawns a multiprocessing-like worker pool, each with their own :class:`BlenderClient` instance.
        The function supplied to pool.map/imap/starmap and their async variants will be automagically
        passed a client instance as their first argument that they can use for rendering.

        Example:
            .. code-block:: python

                def render(client, blend_file):
                    root = Path("renders") / Path(blend_file).stem
                    client.initialize(blend_file, root)
                    client.render_animation()

                if __name__ == "__main__":
                    with BlenderClients.pool(2) as pool:
                        pool.map(render, ["monkey.blend", "cube.blend", "metaballs.blend"])

        Note:
            Here we use ``multiprocess`` instead of the builtin multiprocessing library to take
            advantage of the more advanced dill serialization (as opposed to the standard pickling).

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for ``timeout``
                (in seconds) before giving up. If negative, a port will be randomly selected and assigned to the
                spawned server, bypassing the need for discovery and timeouts. Note that when a port is assigned
                this context manager will immediately yield, even if the server is not yet ready to accept
                incoming connections. Defaults to assigning a port to spawned server (-1 seconds).
            log (str | os.PathLike | FILE | tuple[FILE, FILE], optional): path to log directory, file handle,
                descriptor or tuple thereof. Stdout and stderr will be captured and saved if supplied.
                Defaults to subprocess.DEVNULL for both stdout/stderr.
            autoexec (bool, optional): if true, allow execution of any embedded python scripts within blender.
                For more, see blender's CLI documentation. Defaults to False.
            executable (str | os.PathLike | None, optional): path to Blender's executable. Defaults to looking
                for blender on $PATH, but is useful when targeting a specific blender install, or when it's installed
                via a package manager such as flatpak. Setting it to "flatpak run --die-with-parent org.blender.Blender"
                might be required when using flatpaks. Defaults to None (system PATH).
            conns: List of connection tuples containing the hostnames and ports of existing servers.
                If specified, the pool will use these servers (and ``jobs`` and other spawn arguments will
                be ignored) instead of spawning new ones.

        Yields:
            Generator[multiprocess.Pool]: A ``multiprocess.Pool`` instance which has had it's applicator methods
                (map/imap/starmap/etc) monkey-patched to inject a client instance as first argument.
        """
        # Note import here as this is a dependency only on the client-side
        import multiprocess
        import multiprocess.pool

        def inject_client(func, conns):
            # Note: Usually it's good practice to add `@functools.wraps(func)`
            # here, but it makes dill freak out with a rather cryptic
            # "disallowed for security reasons" error... Works fine otherwise.
            def inner(*args, **kwargs):
                conn = conns.get()

                try:
                    with BlenderClient(conn) as client:
                        retval = func(client, *args, **kwargs)
                finally:
                    conns.put(conn)
                return retval

            return inner

        def modify_applicator(applicator, conns):
            @functools.wraps(applicator)
            def inner(func, *args, **kwargs):
                func = inject_client(func, conns)
                return applicator(func, *args, **kwargs)

            return inner

        context_manager = (
            BlenderServer.spawn(jobs=jobs, timeout=timeout, log=log, autoexec=autoexec, executable=executable)
            if conns is None
            else nullcontext(enter_result=(None, conns))
        )

        with multiprocess.Manager() as manager, context_manager as (_, connections):
            q = manager.Queue()

            for conn in connections:
                q.put(conn)

            with multiprocess.Pool(len(connections)) as pool:
                for name, method in inspect.getmembers(pool, predicate=inspect.ismethod):
                    params = list(inspect.signature(method).parameters.keys())

                    # Get all map/starmap/apply/etc variants
                    if not name.startswith("_") and next(iter(params), None) == "func":
                        setattr(pool, name, modify_applicator(method, q))
                yield pool

    @require_connected_clients
    def common_animation_range(self) -> range:
        """Get animation range shared by all clients as range(start, end+1, step).

        Raises:
            RuntimeError: animation ranges for all clients are expected to be the same.

        Returns:
            range: Range of frames in animation.
        """
        start, end, step = self.common_animation_range_tuple()
        return range(start, end + 1, step)

    @require_connected_clients
    def common_animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range shared by all clients as a tuple of (start, end, step).

        Raises:
            RuntimeError: animation ranges for all clients are expected to be the same.

        Returns:
            tuple[int, int, int]: Frame start, end, and step of animation.
        """
        if len(ranges := set(self.animation_range_tuple())) != 1:  # type: ignore
            raise RuntimeError("Found different animation ranges. All connected servers should be in the same state.")
        return ranges.pop()

    @require_connected_clients
    def render_frames(
        self,
        frame_numbers: Collection[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        """Render all requested frames by distributing workload across connected clients and return associated transforms dictionary.

        Warning:
            Assumes all clients are initialized in the same manner, that is, to the same blendfile, with the same animation range,
            render settings, etc.

        Args:
            frame_numbers (Collection[int]): frames to render.
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
            update_fn (UpdateFn, optional): callback function to track render progress. Will first be called with ``total`` kwarg,
                indicating number of steps to be taken, then will be called with ``advance=1`` at every step. Closely mirrors the
                `rich.Progress API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
                Defaults to None.

        Raises:
            RuntimeError: raised if trying to render frames beyond blender's limits.
        """

        # Set total number of steps, disable updates of total from child processes
        def ignore_total(*args, total=None, **kwargs):
            if update_fn is not None:
                return update_fn(*args, **kwargs)

        if update_fn is not None:
            update_fn(total=len(frame_numbers))

        # Equivalent to more-itertools' distribute (round-robin)
        children = itertools.tee(frame_numbers, len(self))
        frame_chunks = [itertools.islice(it, index, None, len(self)) for index, it in enumerate(children)]

        for client, frames in zip(self, frame_chunks):
            client.render_frames_async(frames, allow_skips=allow_skips, dry_run=dry_run, update_fn=ignore_total)

        # Important: wait for all async renders to finish
        self.wait()

    @require_connected_clients
    def render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips=True,
        dry_run=False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        """Determines frame range to render, sets camera positions and orientations, and renders all frames in animation range by distributing
        workload onto all connected clients.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling :meth:`move_keyframes(scale=2.0) <exposed_move_keyframes>`
              then calling :meth:`render_animation(frame_start=1, frame_end=100) <exposed_render_animation>` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start (int, optional): Starting index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_end (int, optional): Ending index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_step (int, optional): Skip every nth frame. Defaults to None, meaning value from ``.blend`` file.
            allow_skips (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            dry_run (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            update_fn (UpdateFn, optional): Same as :meth:`render_frames <exposed_render_frames>`.

        Raises:
            ValueError: raised if scene and camera are entirely static.
        """
        start, end, step = self.common_animation_range_tuple()
        frame_start = start if frame_start is None else frame_start
        frame_end = end if frame_end is None else frame_end
        frame_step = step if frame_step is None else frame_step
        frame_range = range(frame_start, frame_end + 1, frame_step)

        self.render_frames(frame_range, allow_skips=allow_skips, dry_run=dry_run, update_fn=update_fn)

    @require_connected_clients
    def save_file(self, path: str | os.PathLike) -> None:
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc.

        Note: Only saves file once (from a single connected client), assumes all clients have
            been initialized in the same manner.

        Args:
            path (str | os.PathLike): path where to save blendfile.

        Raises:
            ValueError: raised if file already exists.
        """
        client, *_ = self
        client.save_file(path)

    def wait(self) -> None:
        """Wait for all clients at once."""
        awaitables = [client.awaitable for client in self]

        while awaitables:
            awaitables = [a for a in awaitables if a._waiting()]

            for awaitable in awaitables:
                # Here we query the property `awaitable.ready` which enables
                # the underlying connection to poll and serve any incoming events.
                # Roughly equivalent to the following (but does not rely on private API):
                #     awaitable._conn.serve(awaitable._ttl, waiting=awaitable._waiting)
                _ = awaitable.ready


if __name__ == "__main__":
    if bpy is None:
        sys.exit()

    if bpy.app.version < (3, 6, 0):
        raise RuntimeError("Please use newer blender version, at least 3.6.")

    # Get script specific arguments
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    parser = argparse.ArgumentParser("Startup a BlenderServer on a given port.")
    parser.add_argument("-p", "--port", type=int, default=0)
    args, unknown = parser.parse_known_args(sys.argv[index:])

    server = BlenderServer(port=args.port)
    server.start()
