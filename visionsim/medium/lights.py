"""Lamps, i.e. point, spot and area lights, as sets of point emitters whose light is scattered by a medium.

Emission follows Cycles, so that the medium is lit consistently with the scene:

- Point lights shine with a radiant intensity of their power divided by 4π, in every direction.
- Spot lights shine like point lights of the same power, within a cone towards the edge of which their light fades.
- Area lights are one-sided Lambertian emitters, whose radiance is their power divided by π times their area. They are
  split into a grid of patches, each of which shines like a point emitter of radiant intensity ``radiance · area of the
  patch · cos φ``, at an angle ``φ`` from the light's normal, further restricted to the cone of the light's spread.

Rays that pass far from an area light see it as a single emitter, so only rays that pass close to it need a grid, which
is finer for rays that pass closer, see :func:`area_weights`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch

from visionsim.medium.model import AreaLight, Lamp, PointLight, SpotLight
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
    """radiant intensity per channel, in W/sr, which is scaled by ``profile`` if any, of shape (c,)"""
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


def area_radius(light: AreaLight) -> float:
    """Distance between the center of an area light and its farthest point, in meters."""
    return math.hypot(*light.size) / 2


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


def _grid(light: AreaLight, samples: int, **kwargs) -> torch.Tensor:
    """Centers of patches of equal area that cover an area light, in units of its sizes, of shape (k, 2)."""
    if light.shape == "rectangle":
        # Patches are about square, with `samples` of them along the longest side
        counts = [max(1, round(samples * s / max(light.size))) for s in light.size]
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


def emitters(lamp: Lamp, channels: int, samples: int = 1, **kwargs) -> list[Emitter]:
    """Point emitters that together shine like a lamp.

    Args:
        lamp (Lamp): Point, spot or area light.
        channels (int): Number of channels, i.e. wavelengths, of the intensities.
        samples (int, optional): Number of patches along the longest side of area lights, which are split into about
            ``samples²`` patches. Defaults to 1, i.e. area lights are a single emitter at their center.
        **kwargs: Floating point type and device of the tensors.

    Returns:
        list[Emitter]: Emitters of the lamp.
    """
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

    grid = _grid(lamp, samples, **kwargs)
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
    weights, remaining = [], torch.ones_like(closest)
    for level in levels:
        x = ((closest / area_radius(light) - level.near) / (level.far - level.near)).clamp(0, 1)
        coarser = x * x * (3 - 2 * x)
        weights.append(remaining * (1 - coarser))
        remaining = remaining * coarser
    return [*weights, remaining]
