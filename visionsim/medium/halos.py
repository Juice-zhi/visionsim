"""Light of lamps scattered more than once by a medium, which spreads a halo around them.

Light that a lamp sends into the medium is scattered once towards the camera in closed form, see :func:`lamp_inscatter
<visionsim.medium.render.lamp_inscatter>`. In dense fog, a good share of it is scattered again before reaching the
camera, which spreads a halo much wider than the glow of light scattered once: in the fog of ``examples/medium``, light
scattered more than once adds 18% to the light of a lamp scattered once, and as much again as it a few meters away.

It is computed on a grid of points around the lamp, spread logarithmically in distance from the lamp and evenly in
direction from it. At each point, the light scattered once is gathered from directions concentrated towards the lamp,
where most of it comes from, and scattered again with the medium's phase function. As this light mostly keeps going away
from the lamp, it is tabulated against the angle between the direction it is scattered towards and the direction away
from the lamp, and averaged around the latter, which is exact for a lamp that shines alike in every direction in a
homogeneous medium. Higher orders are iterated by integrating each order along the gathering rays, and later ones are
extrapolated as a geometric series at each point. Tables are interpolated in log space, which is exact for powers of
the distance to the lamp, along camera rays, whose nodes are spread evenly in angle as seen from the lamp. As they don't
depend on the camera, they are computed once per lamp and medium, see :func:`cached_lamp_halo`.

The ground, at the lighting's ``ground_height``, stops light going down, but doesn't reflect any. Light reflected by
other surfaces into the medium isn't modeled either.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import torch

from visionsim.medium.model import Blob, Lamp, Medium
from visionsim.medium.occlusion import ShadowMaps, maps_to
from visionsim.medium.optics import density, henyey_greenstein, optical_depth


class HaloGrid(NamedTuple):
    """Resolution of the tables of the light of lamps scattered more than once."""

    radii: int = 24
    """number of distances to the lamp, spread logarithmically over ``extent``"""
    extent: tuple[float, float] = (0.05, 40.0)
    """smallest and largest distance to the lamp, in meters, beyond which nothing is tabulated"""
    directions: tuple[int, int] = (12, 16)
    """number of polar angles and azimuths of directions from the lamp"""
    angles: tuple[int, int] = (32, 12)
    """number of angles between outgoing directions and the direction away from the lamp, clustered near zero, and of
    azimuths around the latter over which the light is averaged"""
    gathered: tuple[int, int] = (24, 24)
    """number of angles from the direction towards the lamp, clustered near zero, and of azimuths around it, of the
    directions from which light is gathered at each point"""
    orders: int = 4
    """highest order of scattering that is iterated, beyond which orders are extrapolated"""


HALO_GRID = HaloGrid()
"""Default resolution of the tables of the light of lamps scattered more than once"""


class LampHalo(NamedTuple):
    """Light of a lamp scattered more than once by a medium, tabulated on a grid around the lamp."""

    center: torch.Tensor
    """position of the lamp, of shape (3,)"""
    extent: tuple[float, float]
    """smallest and largest distance to the lamp, in meters, spread logarithmically over the first axis of ``values``"""
    ground: float
    """height of the ground, below which there is no light"""
    values: torch.Tensor
    """logarithm of the radiance scattered towards outgoing directions per meter, against distances to the lamp,
    polar angles and azimuths of directions from it, and angles between outgoing directions and the direction away
    from the lamp, of shape (r, t, p, g, c)"""


def _lat_long(n_theta: int, n_phi: int, **kwargs) -> torch.Tensor:
    theta = (torch.arange(n_theta, **kwargs) + 0.5) / n_theta * math.pi
    phi = (torch.arange(n_phi, **kwargs) + 0.5) / n_phi * 2 * math.pi
    t, f = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack([torch.sin(t) * torch.cos(f), torch.sin(t) * torch.sin(f), torch.cos(t)], -1).reshape(-1, 3)


def _frames_around(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors completing unit axes of shape (n, 3) into orthonormal frames."""
    up = (axis[:, 2].abs() < 0.9)[:, None]
    helper = torch.where(up, axis.new_tensor([0.0, 0.0, 1.0]), axis.new_tensor([1.0, 0.0, 0.0]))
    first = torch.linalg.cross(helper, axis)
    first = first / first.norm(dim=-1, keepdim=True)
    return first, torch.linalg.cross(axis, first)


def _around(axis: torch.Tensor, polar: torch.Tensor, azimuth: torch.Tensor) -> torch.Tensor:
    """Directions at polar angles and azimuths around unit axes, of shape (n, k, 3) for axes of shape (n, 3) and angles
    of shape (k,)."""
    first, second = _frames_around(axis)
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


def _interpolate(halo: LampHalo, points: torch.Tensor, outgoing: torch.Tensor) -> torch.Tensor:
    """Radiance scattered per meter towards unit directions ``outgoing`` at ``points``, both of shape (n, 3), of shape
    (n, c)."""
    n_r, n_t, n_p, n_g = halo.values.shape[:4]
    r_min, r_max = halo.extent
    offset = points - halo.center
    r = offset.norm(dim=-1).clamp_min(1e-12)
    away = offset / r[:, None]
    coordinates = [
        ((torch.log(r / r_min) / math.log(r_max / r_min)) * (n_r - 1)).clamp(0, n_r - 1),
        (torch.acos(away[:, 2].clamp(-1, 1)) / math.pi * n_t - 0.5).clamp(0, n_t - 1),
        torch.remainder(torch.atan2(away[:, 1], away[:, 0]), 2 * math.pi) / (2 * math.pi) * n_p - 0.5,
        (torch.acos((outgoing * away).sum(dim=-1).clamp(-1, 1)) / math.pi).sqrt() * (n_g - 1),
    ]
    sizes, periodic = (n_r, n_t, n_p, n_g), (False, False, True, False)
    lows = [x.floor() if wrap else x.floor().clamp(max=size - 2) for x, size, wrap in zip(coordinates, sizes, periodic)]
    result = torch.zeros(len(points), halo.values.shape[-1], dtype=points.dtype, device=points.device)
    for corner in range(16):
        weight = torch.ones(len(points), dtype=points.dtype, device=points.device)
        index = []
        for d, (x, low, size, wrap) in enumerate(zip(coordinates, lows, sizes, periodic)):
            upper = (corner >> d) & 1
            weight = weight * ((x - low) if upper else (1 - (x - low)))
            i = low.long() + upper
            index.append(torch.remainder(i, size) if wrap else i.clamp(0, size - 1))
        result = result + weight[:, None] * halo.values[tuple(index)]
    inside = (r <= r_max) & (points[:, 2] > halo.ground)
    return torch.where(inside[:, None], torch.exp(result), torch.zeros_like(result))


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
    size = max(1, int(4e7 // (nodes * 48)))
    if len(directions) > size:
        # Each node is looked up at 16 corners of the table, so rays are processed in chunks
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
    integrand = torch.exp(-tau) * source.reshape(*s.shape, -1) * weights[..., None]
    return torch.nan_to_num(integrand, nan=0.0, posinf=0.0).sum(dim=-2)


def lamp_halo(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    ground: float = 0.0,
    maps: ShadowMaps | None = None,
    grid: HaloGrid = HALO_GRID,
    time: float = 0.0,
) -> LampHalo:
    """Tabulate the light of a lamp scattered more than once by a medium, around the lamp.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        ground (float, optional): Height of the ground, which stops light going down. Defaults to 0.0.
        maps (ShadowMaps | None, optional): Maps of lamps, through which objects cast the lamp's shadows onto the light
            it scatters once, see :func:`lamp_inscatter <visionsim.medium.render.lamp_inscatter>`. Defaults to None.
        grid (HaloGrid, optional): Resolution of the tables. Defaults to :data:`HALO_GRID`.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        LampHalo: Light of the lamp scattered more than once.
    """
    # The light scattered once is computed as along camera rays, from each point of the grid
    from visionsim.medium.render import lamp_inscatter

    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    g, albedo = medium.anisotropy, medium.albedo
    center = torch.as_tensor(lamp.position, **kwargs)
    (r_min, r_max), n_t, n_p = grid.extent, *grid.directions
    radii = r_min * (r_max / r_min) ** (torch.arange(grid.radii, **kwargs) / (grid.radii - 1))
    points = (center + radii[:, None, None] * _lat_long(n_t, n_p, **kwargs)[None]).reshape(-1, 3)
    away = (points - center) / (points - center).norm(dim=-1, keepdim=True)
    sigma = albedo * density(medium, points, time)[:, None] * beta  # scattering coefficient, (points, c)

    # Directions from which light is gathered, at angles from the direction towards the lamp clustered near zero
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(grid.gathered[0]))
    t, w = (t + 1) / 2, w / 2
    polar, azimuth = torch.meshgrid(
        math.pi * t**2, (torch.arange(grid.gathered[1], **kwargs) + 0.5) / grid.gathered[1] * 2 * math.pi, indexing="ij"
    )
    solid_angles = (2 * math.pi * t * w)[:, None] * torch.sin(polar) * 2 * math.pi / grid.gathered[1]
    gathered = _around(-away, polar.reshape(-1), azimuth.reshape(-1))  # (points, k, 3)
    solid_angles = solid_angles.reshape(-1)

    # Outgoing directions, at angles from the direction away from the lamp clustered near zero, and around it
    n_g, n_b = grid.angles
    angles = math.pi * torch.linspace(0, 1, n_g, **kwargs) ** 2
    out_polar, out_azimuth = torch.meshgrid(
        angles, (torch.arange(n_b, **kwargs) + 0.5) / n_b * 2 * math.pi, indexing="ij"
    )

    def scatter_again(arriving: torch.Tensor) -> torch.Tensor:
        """Light arriving at the points from the gathered directions, of shape (points, k, c), scattered again."""
        scattered = torch.zeros(len(points), n_g, len(beta), **kwargs)
        for i in range(0, len(points), 128):
            j = slice(i, i + 128)
            outgoing = _around(away[j], out_polar.reshape(-1), out_azimuth.reshape(-1))  # (n, g * b, 3)
            # Light arriving from a direction travels along its opposite
            phase = henyey_greenstein(-torch.einsum("nod,nkd->nok", outgoing, gathered[j]), g)
            light = torch.einsum("nok,nkc->noc", phase, arriving[j] * solid_angles[:, None])
            scattered[j] = sigma[j, None, :] * light.reshape(-1, n_g, n_b, len(beta)).mean(dim=2)
        return scattered

    origins = points[:, None, :].expand_as(gathered).reshape(-1, 3)
    rays = gathered.reshape(-1, 3)
    lengths = _to_ground(origins, rays, ground)
    first = torch.cat(
        [
            albedo
            * lamp_inscatter(
                medium, origins[i : i + 50000], rays[i : i + 50000], lengths[i : i + 50000], beta, lamp, time, maps=maps
            )
            for i in range(0, len(rays), 50000)
        ]
    )
    orders = [scatter_again(first.reshape(len(points), -1, len(beta)))]

    def table(values: torch.Tensor) -> LampHalo:
        log = torch.log(values.clamp_min(1e-300)).reshape(grid.radii, n_t, n_p, n_g, len(beta))
        return LampHalo(center, grid.extent, ground, log)

    for _ in range(3, grid.orders + 1):
        previous = table(orders[-1])
        arriving = torch.cat(
            [
                halo_inscatter(
                    previous,
                    medium,
                    origins[i : i + 20000],
                    rays[i : i + 20000],
                    lengths[i : i + 20000],
                    beta,
                    nodes=32,
                    time=time,
                )
                for i in range(0, len(rays), 20000)
            ]
        )
        orders.append(scatter_again(arriving.reshape(len(points), -1, len(beta))))

    # Later orders decrease about geometrically at each point, as the last two did
    total = sum(orders[1:], orders[0])
    if len(orders) > 1:
        ratio = (orders[-1] / orders[-2].clamp_min(1e-300)).clamp(0, 0.8)
        total = total + orders[-1] * ratio / (1 - ratio)
    return table(total)


@lru_cache(maxsize=64)
def _cached_lamp_halo(
    medium: str,
    lamp: Lamp,
    beta: tuple,
    ground: float,
    maps: ShadowMaps | None,
    grid: HaloGrid,
    time: float,
    dtype: torch.dtype,
    device: str,
) -> LampHalo:
    tensor = torch.tensor(beta, dtype=dtype, device=device)
    # Single precision is enough to look up the maps of lamps
    maps = maps_to(maps, device=device, dtype=torch.float32) if maps is not None else None
    return lamp_halo(Medium.model_validate_json(medium), lamp, tensor, ground, maps, grid, time)


def cached_lamp_halo(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    ground: float = 0.0,
    maps: ShadowMaps | None = None,
    grid: HaloGrid = HALO_GRID,
    time: float = 0.0,
) -> LampHalo:
    """Same as :func:`lamp_halo`, cached for the last few lamps and media, as it doesn't depend on the camera and can be
    reused for every frame, unless the medium moves. Maps are keyed by identity, so the same maps should be given for
    every frame, rather than copies of them."""
    moving = any(isinstance(c, Blob) and any(c.velocity) for c in medium.components)
    return _cached_lamp_halo(
        medium.model_dump_json(),
        lamp,
        tuple(beta.tolist()),
        ground,
        maps,
        grid,
        time if moving else 0.0,
        beta.dtype,
        str(beta.device),
    )


def halo_threshold(medium: Medium, lamp: Lamp, beta: torch.Tensor, extent: float = HALO_GRID.extent[1]) -> bool:
    """Whether the light of a lamp scattered more than once is worth computing, i.e. whether the medium around it has
    an optical depth of at least 1% over the extent of the halo, as the halo grows with it.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        extent (float, optional): Extent of the halo, in meters. Defaults to that of :class:`HaloGrid`.

    Returns:
        bool: Whether the halo is worth computing.
    """
    at_lamp = density(medium, torch.as_tensor(lamp.position, dtype=beta.dtype, device=beta.device)[None])[0]
    return bool(medium.albedo * at_lamp * beta.max() * extent >= 0.01)
