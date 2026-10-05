"""Lamps, i.e. point, spot and area lights and emissive surfaces, as sets of point emitters whose light is scattered by
a medium.

Emission follows Cycles, so that the medium is lit consistently with the scene:

- Point lights shine with a radiant intensity of their power divided by 4π, in every direction.
- Spot lights shine like point lights of the same power, within a cone towards the edge of which their light fades.
- Area lights are one-sided Lambertian emitters, whose radiance is their power divided by π times their area. They are
  split into a grid of patches, each of which shines like a point emitter of radiant intensity ``radiance · area of the
  patch · cos φ``, at an angle ``φ`` from the light's normal, further restricted to the cone of the light's spread.
- Emissive surfaces emit their radiance on both sides, and each of their patches shines like a point emitter of
  radiant intensity ``radiance · area seen``, where the area seen from a direction only counts the faces of the patch
  whose light leaves the surfaces towards it, e.g. not the inside of a closed surface, see :func:`projected_area`.

Rays that pass far from an area light see it as a single emitter, so only rays that pass close to it need a grid, which
is finer for rays that pass closer, see :func:`area_weights`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch

from visionsim.medium.model import AreaLight, EmissiveSurface, Lamp, PointLight, SpotLight
from visionsim.medium.optics import _per_channel


class AreaLevel(NamedTuple):
    """Grid of emitters through which rays that pass within some distance of an area light see it."""

    samples: int
    """number of patches along the longest side of the light"""
    nodes: int
    """number of quadrature nodes along each ray, for each emitter"""
    near: float
    """distance between rays and the light's center, as a multiple of its radius, up to which rays see this grid"""
    far: float
    """distance, as a multiple of the light's radius, from which rays see the next coarser grid instead, with a smooth
    transition in between"""


AREA_LEVELS: tuple[AreaLevel, ...] = (
    AreaLevel(16, 8, 1.0, 1.5),
    AreaLevel(4, 16, 3.0, 4.0),
    AreaLevel(2, 24, 8.0, 10.0),
)
"""Grids of emitters of area lights, from the finest to the coarsest, beyond which rays see a single emitter. Seen
from a distance ``D``, a grid's relative error is of the order of ``(R / (K D))²`` for a light of radius ``R`` split
into ``K`` patches along its sides, with a large factor in fog that mostly scatters light forward, as the phase function
then varies quickly across the light"""


class Emitter(NamedTuple):
    """Point emitter of light, which may only shine within a cone."""

    position: torch.Tensor
    """world-space position, of shape (3,)"""
    intensity: torch.Tensor
    """radiant intensity per channel, in W/sr, which is scaled by ``profile`` and ``pattern`` if any, of shape (c,)"""
    radius: float
    """distance to the emitter below which distances are clamped, in meters"""
    axis: torch.Tensor | None = None
    """unit axis of its cone of light, of shape (3,), or None if it shines in every direction"""
    cone: float = -1.0
    """cosine of the half-angle of its cone of light, outside of which it doesn't shine"""
    profile: Callable[[torch.Tensor], torch.Tensor] | None = None
    """relative radiant intensity, given the cosine of the angle between the axis and directions from the emitter"""
    falloff: Callable[[torch.Tensor], torch.Tensor] | None = None
    """relative radiant intensity, given the distance to the emitter, or None if it physically falls off as 1 / r²"""
    pattern: Callable[[torch.Tensor], torch.Tensor] | None = None
    """relative radiant intensity, given unit directions from the emitter, of shape (..., 3), for emitters that don't
    shine symmetrically around an axis"""


def distance_falloff(distance: torch.Tensor, exponent: int, smooth: float) -> torch.Tensor:
    """Relative intensity of a lamp at given distances, as set by Blender's Light Falloff node.

    Args:
        distance (torch.Tensor): Distances to the lamp, in meters.
        exponent (int): Power of the distance by which the intensity is scaled, i.e. 0 for a quadratic falloff,
            1 for a linear one and 2 for a constant one.
        smooth (float): Smoothing of the light near the lamp, which scales its intensity by ``r² / (smooth + r²)``.

    Returns:
        torch.Tensor: Relative intensity.
    """
    squared = distance * distance
    factor = distance**exponent if exponent else torch.ones_like(distance)
    return factor * squared / (smooth + squared) if smooth > 0 else factor


def spot_falloff(cos_angle: torch.Tensor, cos_half_angle: float, smooth: float) -> torch.Tensor:
    """Relative intensity of a spot light, which fades smoothly towards the edge of its cone, as in Cycles.

    Args:
        cos_angle (torch.Tensor): Cosine of the angle between the spot's direction and directions from the light.
        cos_half_angle (float): Cosine of the half-angle of the spot's cone.
        smooth (float): Inverse of the range of cosines over which light fades, i.e. ``1 / ((1 - cos_half_angle) *
            blend)``, which is infinite for hard edges.

    Returns:
        torch.Tensor: Relative intensity, between zero and one.
    """
    if math.isinf(smooth):
        return (cos_angle >= cos_half_angle).to(cos_angle.dtype)
    x = ((cos_angle - cos_half_angle) * smooth).clamp(0, 1)
    return x * x * (3 - 2 * x)


def area_falloff(cos_angle: torch.Tensor, tan_half_spread: float, normalization: float) -> torch.Tensor:
    """Relative intensity of a patch of an area light, i.e. Lambertian, attenuated by the spread of the light.

    In Cycles, the spread models the grid of a softbox, such that light leaving the light at an angle ``φ`` from its
    normal is attenuated by ``(tan(spread / 2) - tan φ) / (tan(spread / 2) - spread / 2)``, which keeps its power.

    Args:
        cos_angle (torch.Tensor): Cosine of the angle between the light's normal and directions from the light.
        tan_half_spread (float): Tangent of half the light's spread.
        normalization (float): ``1 / (tan(spread / 2) - spread / 2)``, or zero for lights that spread their light over
            the whole half-space, which aren't attenuated.

    Returns:
        torch.Tensor: Relative intensity, which is the cosine for lights without attenuation.
    """
    cos_angle = cos_angle.clamp_min(0)
    if normalization <= 0:
        return cos_angle
    tan_angle = (1 - cos_angle * cos_angle).clamp_min(0).sqrt() / cos_angle.clamp_min(1e-12)
    return cos_angle * ((tan_half_spread - tan_angle) * normalization).clamp_min(0)


def two_sided_falloff(cos_angle: torch.Tensor, front: float, back: float) -> torch.Tensor:
    """Relative intensity of a flat patch of an emissive surface, which is Lambertian on both sides.

    Args:
        cos_angle (torch.Tensor): Cosine of the angle between the patch's normal and directions from the patch.
        front (float): Relative intensity towards the normal.
        back (float): Relative intensity away from the normal.

    Returns:
        torch.Tensor: Relative intensity.
    """
    return torch.where(cos_angle >= 0, front * cos_angle, -back * cos_angle)


def projected_area(
    directions: torch.Tensor, orientation: torch.Tensor, facing: torch.Tensor, scale: float
) -> torch.Tensor:
    """Area of the faces of a patch of an emissive surface seen from given directions, whose light leaves towards them,
    relative to the area of the patch.

    For faces of areas ``a_i`` and unit normals ``n_i``, a fraction ``f_i`` of the light of whose front leaves the
    surface, and ``b_i`` of the light of their back, it is ``Σ a_i (f_i max(n_i·ω, 0) + b_i max(-n_i·ω, 0)) / Σ a_i``
    in a direction ``ω``. This is the sum of an even part, ``Σ a_i s_i |n_i·ω| / Σ a_i`` with ``s_i = (f_i + b_i) / 2``,
    and of an odd one, ``Σ a_i d_i n_i·ω / Σ a_i`` with ``d_i = (f_i - b_i) / 2``, which is exactly ``facing·ω``. The
    even part is approximated by ``scale · sqrt(ωᵀ M ω)``, where ``M = Σ a_i s_i n_i n_iᵀ / Σ a_i`` is the orientation of
    the faces and the scale keeps the light that they emit, see :func:`orientation_scale`. This is exact for flat
    patches, whose normals are all alike, and for faces whose normals are spread evenly around one or every direction,
    such as cylinders and spheres, and hemispheres with their odd part.

    Args:
        directions (torch.Tensor): Unit directions from the patch, of shape (..., 3).
        orientation (torch.Tensor): Orientation ``M`` of the faces, of shape (3, 3).
        facing (torch.Tensor): Normals of the faces weighted by ``d_i``, i.e. ``Σ a_i d_i n_i / Σ a_i``, of shape (3,).
        scale (float): Scale of the even part, see :func:`orientation_scale`.

    Returns:
        torch.Tensor: Relative area seen from each direction, of shape (...).
    """
    quadratic = torch.einsum("...i,ij,...j->...", directions, orientation, directions)
    return (scale * quadratic.clamp_min(0).sqrt() + directions @ facing).clamp_min(0)


def orientation_scale(orientation: torch.Tensor, samples: int = 4096) -> float:
    """Scale of the even part of the area of faces seen from each direction, see :func:`projected_area`, such that they
    emit as much light as the faces do, i.e. such that ``scale · sqrt(ωᵀ M ω)`` integrates to ``2π · trace(M)`` over all
    directions, as ``|n·ω|`` integrates to ``2π``.

    Args:
        orientation (torch.Tensor): Orientation ``M`` of the faces, of shape (3, 3).
        samples (int, optional): Number of directions of the Fibonacci lattice that integrates over the sphere.
            Defaults to 4096.

    Returns:
        float: Scale, which is one for flat patches whose normals are all alike, or zero if the faces don't emit.
    """
    i = torch.arange(samples, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / samples
    phi = math.pi * (3 - math.sqrt(5)) * i
    s = (1 - z * z).sqrt()
    lattice = torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], dim=-1)
    matrix = orientation.detach().to(device="cpu", dtype=torch.float64)
    mean = torch.einsum("ni,ij,nj->n", lattice, matrix, lattice).clamp_min(0).sqrt().mean()
    trace = float(torch.diagonal(matrix).sum())
    return 0.0 if trace <= 0 or float(mean) <= 0 else 2 * math.pi * trace / (4 * math.pi * float(mean))


def area_radius(light: AreaLight) -> float:
    """Distance between the center of an area light and its farthest point, in meters."""
    return math.hypot(*light.size) / 2


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


def _grid(size: tuple[float, float], shape: str, samples: int, **kwargs) -> torch.Tensor:
    """Centers of patches of equal area that cover a rectangle or an ellipse, in units of its sizes, of shape (k, 2)."""
    if shape == "rectangle":
        # Patches are about square, with `samples` of them along the longest side
        counts = [max(1, round(samples * s / max(size))) for s in size]
        u, v = ((torch.arange(c, **kwargs) + 0.5) / c - 0.5 for c in counts)
        return torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1).reshape(-1, 2)

    # Shirley and Chiu's concentric mapping of a square grid onto the disk, which preserves areas
    cells = (torch.arange(samples, **kwargs) + 0.5) / samples * 2 - 1
    a, b = (x.reshape(-1) for x in torch.meshgrid(cells, cells, indexing="ij"))
    outer = a.abs() > b.abs()
    r = torch.where(outer, a, b)
    safe_a, safe_b = torch.where(a == 0, torch.ones_like(a), a), torch.where(b == 0, torch.ones_like(b), b)
    angle = torch.where(outer, math.pi / 4 * b / safe_a, math.pi / 2 - math.pi / 4 * a / safe_b)
    angle = torch.where(r == 0, torch.zeros_like(angle), angle)
    return torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=-1) / 2


def _symmetric(values: tuple[float, ...], **kwargs) -> torch.Tensor:
    """Symmetric matrix given as ``(xx, yy, zz, xy, xz, yz)``, of shape (3, 3)."""
    xx, yy, zz, xy, xz, yz = values
    return torch.tensor([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], **kwargs)


def _flat_normal(orientation: torch.Tensor) -> tuple[torch.Tensor, float] | None:
    """Normal of a flat patch given its orientation, which is then of rank one, and its eigenvalue, or None."""
    values, vectors = torch.linalg.eigh(orientation)
    if values[2] > 0 and values[:2].abs().sum() <= 1e-6 * values[2]:
        return vectors[:, 2], float(values[2])
    return None


def patch_radius(lamp: EmissiveSurface, index: int) -> float:
    """Radius of a patch of an emissive surface, in meters, below which distances to it are clamped: the root mean
    square distance between its points and its center, or half the square root of its area without its spread."""
    if lamp.spread:
        return math.sqrt(max(sum(lamp.spread[index][:3]), 0.0))
    return math.sqrt(lamp.areas[index]) / 2


def patch_rectangle(lamp: EmissiveSurface, index: int) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]] | None:
    """Rectangle of the same spread as a flat patch of an emissive surface, i.e. its axes and its sizes along them, see
    :attr:`EmissiveSurface.spread <visionsim.medium.model.EmissiveSurface.spread>`, or None for other patches, which
    are seen as a point at any distance.

    Args:
        lamp (EmissiveSurface): Emissive surface.
        index (int): Index of the patch.

    Returns:
        tuple[torch.Tensor, torch.Tensor, tuple[float, float]] | None: Unit axes along the longest side of the rectangle
        and along its other side, each of shape (3,), and its sizes, in meters.
    """
    flat = _flat_normal(_symmetric(lamp.orientation[index], dtype=torch.float64))
    if not lamp.spread or flat is None:
        return None
    normal = flat[0]
    # Spread within the patch's plane, of which a uniform rectangle of sides a and b has eigenvalues a² / 12 and b² / 12
    plane = torch.eye(3, dtype=torch.float64) - torch.outer(normal, normal)
    values, vectors = torch.linalg.eigh(plane @ _symmetric(lamp.spread[index], dtype=torch.float64) @ plane)
    sizes = (math.sqrt(12 * max(float(values[2]), 0.0)), math.sqrt(12 * max(float(values[1]), 0.0)))
    return (vectors[:, 2], vectors[:, 1], sizes) if sizes[0] > 0 else None


def merge_patches(lamp: EmissiveSurface) -> EmissiveSurface:
    """The same emissive surface as a single patch, which emits as much light in total and as the moments of the
    normals of all its faces show, e.g. to integrate light that doesn't depend much on the shape of the surface.

    Args:
        lamp (EmissiveSurface): Emissive surface.

    Returns:
        EmissiveSurface: Surface of a single patch, at the mean position of the patches weighted by their light, which
        casts shadows from the same position.
    """
    areas = torch.tensor(lamp.areas, dtype=torch.float64)
    radiance = torch.stack([_per_channel(r, 1, "radiance", dtype=torch.float64).mean() for r in lamp.radiance])
    # Patches are weighted by their light, but their colors are averaged by area, which keeps the light of each channel
    light = areas * radiance
    weights = light / light.sum() if bool(light.sum() > 0) else areas / areas.sum()
    positions = torch.tensor(lamp.positions, dtype=torch.float64)
    position = weights @ positions

    def mean(values: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        return tuple((weights @ torch.tensor(values, dtype=torch.float64)).tolist())

    channels = max(len(r) for r in lamp.radiance)
    colors = torch.stack([_per_channel(r, channels, "radiance", dtype=torch.float64) for r in lamp.radiance])
    spread: tuple[tuple[float, ...], ...] = ()
    if lamp.spread:
        offsets = positions - position
        outer = offsets[:, :, None] * offsets[:, None, :]
        within = torch.stack([_symmetric(s, dtype=torch.float64) for s in lamp.spread])
        covariance = torch.einsum("k,kij->ij", weights, within + outer)
        spread = (tuple(float(covariance[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))),)
    return EmissiveSurface(
        position=lamp.position,
        positions=(tuple(position.tolist()),),
        areas=(float(areas.sum()),),
        radiance=(tuple(((areas @ colors) / areas.sum()).tolist()),),
        orientation=(mean(lamp.orientation),),
        facing=(mean(lamp.facing),) if lamp.facing else (),
        spread=spread,
    )


def patch_emitters(lamp: EmissiveSurface, index: int, channels: int, samples: int = 1, **kwargs) -> list[Emitter]:
    """Point emitters that together shine like a patch of an emissive surface.

    The patch shines in proportion to the area of its faces seen from each direction, see :func:`projected_area`.
    Flat patches are Lambertian emitters on each side, which, as area lights, can be split into a grid of patches over
    the rectangle of the same spread (see :func:`patch_rectangle`), with ``samples`` of them along its longest side.

    Args:
        lamp (EmissiveSurface): Emissive surface.
        index (int): Index of the patch.
        channels (int): Number of channels, i.e. wavelengths, of the intensities.
        samples (int, optional): Number of patches along the longest side of flat patches. Defaults to 1, i.e. a single
            emitter at the patch's center.
        **kwargs: Floating point type and device of the tensors.

    Returns:
        list[Emitter]: Emitters of the patch, if it emits any light.
    """
    area = lamp.areas[index]
    center = torch.as_tensor(lamp.positions[index], **kwargs)
    towards = torch.as_tensor(lamp.facing[index] if lamp.facing else (0.0, 0.0, 0.0), **kwargs)
    orientation = _symmetric(lamp.orientation[index], **kwargs)
    intensity = _per_channel(lamp.radiance[index], channels, "radiance", **kwargs) * area
    flat = _flat_normal(orientation)
    if flat is None:
        scale = orientation_scale(orientation)
        if scale <= 0 and not bool(towards.any()):
            return []
        pattern = partial(projected_area, orientation=orientation, facing=towards, scale=scale)
        return [Emitter(center, intensity, patch_radius(lamp, index), pattern=pattern)]

    # Flat patches shine as a Lambertian emitter on each side. When a single side shines, only its half-space is
    # integrated, otherwise their light is kinked at their plane, which barely affects the quadrature
    normal, value = flat
    along = float(normal @ towards)
    front, back = value + along, value - along
    if min(front, back) <= 1e-9 * max(front, back):
        axis, cone = normal if front > back else -normal, 0.0
        intensity = intensity * max(front, back)
        profile = partial(area_falloff, tan_half_spread=math.inf, normalization=0.0)
    else:
        axis, cone, profile = normal, -1.0, partial(two_sided_falloff, front=front, back=back)
    rectangle = patch_rectangle(lamp, index) if samples > 1 else None
    if rectangle is None:
        return [Emitter(center, intensity, patch_radius(lamp, index), axis, cone, profile)]
    axis_u, axis_v, size = rectangle
    grid = _grid(size, "rectangle", samples, **kwargs)
    offsets = grid[:, :1] * size[0] * axis_u.to(center) + grid[:, 1:] * size[1] * axis_v.to(center)
    radius = math.sqrt(area / len(grid)) / 2
    return [Emitter(center + offset, intensity / len(grid), radius, axis, cone, profile) for offset in offsets]


def emitters(lamp: Lamp, channels: int, samples: int = 1, **kwargs) -> list[Emitter]:
    """Point emitters that together shine like a lamp.

    Args:
        lamp (Lamp): Point, spot or area light, or emissive surface, see :func:`patch_emitters` for its patches.
        channels (int): Number of channels, i.e. wavelengths, of the intensities.
        samples (int, optional): Number of patches along the longest side of area lights, and of flat patches of
            emissive surfaces, which are split into about ``samples²`` patches. Defaults to 1, i.e. area lights are a
            single emitter at their center.
        **kwargs: Floating point type and device of the tensors.

    Returns:
        list[Emitter]: Emitters of the lamp.
    """
    if isinstance(lamp, EmissiveSurface):
        return [e for index in range(len(lamp.areas)) for e in patch_emitters(lamp, index, channels, samples, **kwargs)]

    power = _per_channel(lamp.power, channels, "light power", **kwargs)
    position = torch.as_tensor(lamp.position, **kwargs)
    exponent = {"quadratic": 0, "linear": 1, "constant": 2}[lamp.falloff]
    falloff = partial(distance_falloff, exponent=exponent, smooth=lamp.smooth) if exponent or lamp.smooth else None

    if isinstance(lamp, PointLight):
        return [Emitter(position, power / (4 * math.pi), lamp.radius, falloff=falloff)]

    if isinstance(lamp, SpotLight):
        cos_half = math.cos(lamp.angle / 2)
        smooth = 1 / ((1 - cos_half) * lamp.blend) if lamp.blend > 0 and cos_half < 1 else math.inf
        profile = partial(spot_falloff, cos_half_angle=cos_half, smooth=smooth)
        axis = _unit(torch.as_tensor(lamp.direction, **kwargs))
        return [Emitter(position, power / (4 * math.pi), lamp.radius, axis, max(cos_half, 0.0), profile, falloff)]

    normal = _unit(torch.as_tensor(lamp.direction, **kwargs))
    axis_u = torch.as_tensor(lamp.axis_u, **kwargs)
    axis_u = _unit(axis_u - (axis_u @ normal) * normal)
    axis_v = torch.linalg.cross(normal, axis_u)
    half_spread = min(lamp.spread / 2, math.pi / 2)
    if half_spread < math.pi / 2:
        tan_half = math.tan(half_spread)
        profile = partial(area_falloff, tan_half_spread=tan_half, normalization=1 / (tan_half - half_spread))
    else:
        profile = partial(area_falloff, tan_half_spread=math.inf, normalization=0.0)

    grid = _grid(lamp.size, lamp.shape, samples, **kwargs)
    offsets = grid[:, :1] * lamp.size[0] * axis_u + grid[:, 1:] * lamp.size[1] * axis_v
    # Each patch has a radiance of power / (π area) and an equal share of the area
    intensity = power / (math.pi * len(grid))
    radius = math.sqrt(lamp.area / len(grid)) / 2
    cone = max(math.cos(half_spread), 0.0)
    return [Emitter(position + offset, intensity, radius, normal, cone, profile, falloff) for offset in offsets]


def closest_distances(
    origin: torch.Tensor, directions: torch.Tensor, distance: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
    """Distance between a point and the closest point of each ray, which end after ``distance`` meters.

    Args:
        origin (torch.Tensor): Ray origin, of shape (3,), or one per ray, of shape (..., 3).
        directions (torch.Tensor): Unit ray directions, of shape (..., 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (...), can be infinite.
        position (torch.Tensor): Point, of shape (3,).

    Returns:
        torch.Tensor: Distances, of shape (...).
    """
    offset = position - origin
    along = torch.minimum((directions * offset).sum(dim=-1).clamp_min(0), distance)
    return (along[..., None] * directions - offset).norm(dim=-1)


def area_weights(
    light: AreaLight, closest: torch.Tensor, levels: tuple[AreaLevel, ...] = AREA_LEVELS
) -> list[torch.Tensor]:
    """Weights of the grids of emitters of an area light, for rays that pass at given distances from it.

    Rays see the light through the finest grid up to its ``near`` distance, and through the next coarser one from its
    ``far`` distance, with a smooth transition in between so that no seam shows, and beyond the coarsest grid through a
    single emitter at the light's center.

    Args:
        light (AreaLight): Area light.
        closest (torch.Tensor): Distances between the light's center and the closest point of rays, see
            :func:`closest_distances`.
        levels (tuple[AreaLevel, ...], optional): Grids of emitters, from the finest to the coarsest, whose distances
            must increase. Defaults to :data:`AREA_LEVELS`.

    Returns:
        list[torch.Tensor]: Weight of each grid, followed by the weight of the single emitter, which add up to one,
        each of the same shape as ``closest``.
    """
    return level_weights(closest / area_radius(light), levels)


def level_weights(relative: torch.Tensor, levels: tuple[AreaLevel, ...] = AREA_LEVELS) -> list[torch.Tensor]:
    """Weights of the grids of emitters of a light, for rays that pass at given distances from its center, relative to
    the distance between its center and its farthest point, see :func:`area_weights`."""
    weights, remaining = [], torch.ones_like(relative)
    for level in levels:
        x = ((relative - level.near) / (level.far - level.near)).clamp(0, 1)
        coarser = x * x * (3 - 2 * x)
        weights.append(remaining * (1 - coarser))
        remaining = remaining * coarser
    return [*weights, remaining]
