"""Light that reaches an exponential height fog from many directions: from the sky, the ground and the fog itself.

Unlike sunlight, which comes from a single direction, skylight, light reflected by the ground and light that was
already scattered by the fog reach every point from many directions. The light they scatter towards a ray then depends
on the ray's elevation and on the height of the point, and, for the light of suns scattered more than once, on the
ray's azimuth from the sun's. Such sources are integrated along camera rays with a table of their integral against the
rays' elevation (and azimuth) and optical depth, computed once per frame since all camera rays start from the same
point, and interpolated for each pixel. As it only depends on the height of the camera, it is kept for the next frames,
see :func:`cached_camera_tables`.

Heights are expressed as ``c = σ(z) · H``, the optical depth of the fog above a point, where ``σ`` is the extinction
coefficient and ``H`` the fog's falloff, such that light coming from an elevation ``μ`` above the horizon is attenuated
by ``exp(-c / μ)`` before reaching the point. Along a ray of elevation ``v_z`` starting where the optical depth above
is ``c0``, the optical depth above a point beyond which the ray has an optical depth ``τ`` is ``c0 - v_z · τ``. As the
fog is horizontally uniform, ``c`` is the only coordinate that light scattered in it depends on.

Light scattered more than once is computed by successive orders of scattering (e.g. Hansen and Travis, "Light
scattering in planetary atmospheres", Space Science Reviews, 1974): at a set of heights, the light of each order is
gathered from all directions, along rays up to the top of the fog or down to the ground, and scattered once more with
the medium's phase function, while the ground reflects the light of each order that reaches it into the next one.
Orders are added until they become negligible, and the remaining ones, which decrease about geometrically, are
extrapolated. Hillaire's approximation ("A Scalable and Production Ready Sky and Atmosphere Rendering Technique",
EGSR 2020), where each order is a fixed fraction of the previous one, that of light scattered isotropically,
underestimates higher orders in fog, which scatters light strongly forward: light keeps going in about the same
direction, and near the horizon, where the fog is optically thick, each order is nearly as bright as the previous one.
In the ground fog of ``examples/medium``, orders beyond the second add about twice as much light as it estimates. The
ground is also lit by the light that the fog scatters, which makes up for much of the light that the fog takes out of
sunlight and skylight on their way to the ground, as it scatters most of it forward, towards the ground: there, it
makes the ground about 40% brighter than sunlight and skylight alone.

The fog is invariant by rotation about the vertical, so that light varying as ``cos(mφ)`` with the azimuth ``φ`` of the
directions it comes from is scattered into light varying the same way, and the harmonics of the azimuth are scattered
independently, see :func:`azimuthal_harmonics`. The light of the sky and of the ground doesn't depend on the azimuth,
while the light of suns, which the fog scatters forward, is still brighter towards them after several scatterings:
it is expanded in harmonics of the azimuth from each sun's, and so is the light that arrives at each height, of which
objects hide more when they hide directions towards suns, see :func:`cell_profile
<visionsim.medium.occlusion.cell_profile>`.

In the scene of ``examples/medium`` without objects, against Cycles with 32 volume bounces and a black ground, the fog's
own light is within 0.6% of Cycles' overall, the light it scatters twice within 0.7%, three times within 1.5% and more
within 4%, also with the camera turned towards the sun, and within 0.7% with a Lambertian ground of the albedo modeled
here, while Cycles' fog is 3.3% brighter over its Principled ground of the same color, whose specular layer reflects
more at grazing angles.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import lru_cache, partial
from typing import NamedTuple

import numpy as np
import torch

from visionsim.medium.model import HeightFog, Lighting, Medium
from visionsim.medium.optics import _per_channel, henyey_greenstein, sky_quadrature

TABLE_SIZE: tuple[int, int] = (256, 257)
"""Default number of elevations and optical depths at which light scattered along camera rays is tabulated"""

AZIMUTHS: int = 33
"""Default number of azimuths of rays from a sun's, spread uniformly over [0, π], at which the variations of the sun's
light scattered more than once are tabulated"""

AZIMUTH_TABLE_SIZE: tuple[int, int] = (64, 65)
"""Default number of elevations and optical depths at which the variations of suns' light scattered more than once with
azimuth are tabulated along camera rays, coarser than :data:`TABLE_SIZE` as they are smoother and fainter than the
light they vary around"""

_STRETCH = 3.0
"""Elevations are tabulated at ``sinh(a t) / sinh(a)`` for ``t`` spread uniformly over [-1, 1], which makes them
about three times denser near the horizon, where light from the sky and the ground changes the most"""


def elevation_grid(n: int, **kwargs) -> torch.Tensor:
    """Vertical components of ray directions at which light is tabulated, denser near the horizon."""
    return torch.sinh(_STRETCH * torch.linspace(-1, 1, n, **kwargs)) / math.sinh(_STRETCH)


def _elevation_index(elevations: torch.Tensor, n: int) -> torch.Tensor:
    """Fractional index of vertical components of ray directions in :func:`elevation_grid`."""
    return (torch.asinh(elevations.clamp(-1, 1) * math.sinh(_STRETCH)) / _STRETCH + 1) / 2 * (n - 1)


class ElevationTable(NamedTuple):
    """Light scattered along rays starting from a common point, tabulated against their elevation and optical depth.

    Elevations are those of :func:`elevation_grid`, and optical depths are spread through the fraction of light
    scattered or absorbed along the ray, ``1 - exp(-τ)``, normalized by its maximum along rays going up and leaving the
    fog. Light that also depends on the azimuth of rays from a sun's, see :func:`azimuthal_sources`, is tabulated
    against that azimuth too, spread uniformly over [0, π] as it is symmetric about the sun's vertical plane.
    """

    values: torch.Tensor
    """integral of the source along rays, of shape (elevations, depths, c), or (elevations, azimuths, depths, c)"""
    depth_above: torch.Tensor
    """optical depth of the fog above the rays' origin, of shape (c,) or (1,) when it is the same for all channels"""
    towards: torch.Tensor | None = None
    """horizontal unit direction towards the sun from which azimuths are measured, of shape (2,), for tables with an
    azimuth axis"""


def relative_azimuths(directions: torch.Tensor, towards: torch.Tensor) -> torch.Tensor:
    """Azimuth of directions from a horizontal direction, in [0, π].

    Args:
        directions (torch.Tensor): Directions, of shape (..., 3).
        towards (torch.Tensor): Horizontal unit direction, of shape (2,).

    Returns:
        torch.Tensor: Absolute azimuth of each direction from ``towards``, of shape (...).
    """
    towards = towards.to(directions)
    along = directions[..., 0] * towards[0] + directions[..., 1] * towards[1]
    across = directions[..., 1] * towards[0] - directions[..., 0] * towards[1]
    return torch.atan2(across.abs(), along)


def complete_elliptic_e(m: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    """Complete elliptic integral of the second kind ``E(m)``, for parameters ``0 <= m < 1``.

    This uses the arithmetic-geometric mean, which converges quadratically, as
    ``E(m) = π / (2 a) · (1 - Σ 2^(n-1) c_n²)`` where ``c_0² = m``.

    Args:
        m (torch.Tensor): Parameter, i.e. the squared elliptic modulus.
        iterations (int, optional): Number of iterations of the arithmetic-geometric mean. Defaults to 12.

    Returns:
        torch.Tensor: ``E(m)``.
    """
    a, b = torch.ones_like(m), (1 - m).clamp_min(0).sqrt()
    total, power = m / 2, 1.0
    for _ in range(iterations):
        a, b, c = (a + b) / 2, (a * b).sqrt(), (a - b) / 2
        total = total + power * c * c
        power *= 2
    return math.pi / (2 * a) * (1 - total)


def azimuthal_phase(a: torch.Tensor, b: torch.Tensor, g: float) -> torch.Tensor:
    """Henyey-Greenstein phase function integrated over the azimuth between two directions of given elevations.

    For directions whose vertical components are ``a`` and ``b`` and whose azimuths differ by ``φ``, the cosine of the
    angle between them is ``ab + sqrt(1 - a²) sqrt(1 - b²) cos(φ)``. With ``A = 1 + g² - 2gab`` and
    ``B = 2g sqrt(1 - a²) sqrt(1 - b²)``, the integral of the phase function over ``φ`` is
    ``(1 - g²) / (4π) · 4 E(2B / (A + B)) / ((A - B) sqrt(A + B))``, where ``E`` is the complete elliptic integral of
    the second kind.

    Args:
        a (torch.Tensor): Vertical component of the first directions.
        b (torch.Tensor): Vertical component of the second directions, broadcastable with ``a``.
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.

    Returns:
        torch.Tensor: Integral of the phase function over the azimuth, in 1/sr (per radian of azimuth). It equals 1/2
        for isotropic scattering.
    """
    s = (1 - a * a).clamp_min(0).sqrt() * (1 - b * b).clamp_min(0).sqrt()
    big_a, big_b = 1 + g * g - 2 * g * a * b, 2 * g * s
    if g < 0:
        # The integral only depends on |B|, the sign of g is carried by A
        big_b = -big_b
    m = (2 * big_b / (big_a + big_b)).clamp(0, 1 - 1e-12)
    return (1 - g * g) / math.pi * complete_elliptic_e(m) / ((big_a - big_b) * (big_a + big_b).sqrt())


def _elevation_directions(elevations: torch.Tensor) -> torch.Tensor:
    """Unit directions with given vertical components, in the x-z plane."""
    return torch.stack([(1 - elevations**2).clamp_min(0).sqrt(), torch.zeros_like(elevations), elevations], dim=-1)


class Hemisphere(NamedTuple):
    """Quadrature of the phase function over a hemisphere of directions, for rays of given elevations."""

    inverse: torch.Tensor
    """inverse of the absolute elevation of each node, of shape (n, k), which is huge for nodes at the horizon as
    their light goes through an infinite amount of fog"""
    weights: torch.Tensor
    """weight of each node, of shape (n, k)"""


def hemisphere(elevations: torch.Tensor, g: float, below: bool = False, **resolution) -> Hemisphere:
    """Quadrature of the phase function over directions above the horizon, or below it, for rays of given elevations.

    Directions below the horizon are those of :func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>` for rays
    mirrored across the horizon, which preserves the angles between directions. Nodes at opposite azimuths around the
    rays, which have the same elevation, are merged, which halves the cost of the sums over them.

    Args:
        elevations (torch.Tensor): Vertical component of the rays' directions, of shape (n,).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        below (bool, optional): If true, integrate over directions below the horizon. Defaults to False.
        **resolution: Resolution of :func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`.

    Returns:
        Hemisphere: Inverse elevations and weights of the quadrature's nodes.
    """
    directions = _elevation_directions(-elevations if below else elevations)
    mu, weights = sky_quadrature(directions, g, folded=True, **resolution)
    return Hemisphere(torch.where(mu > 0, 1 / mu.clamp_min(1e-30), torch.full_like(mu, 1e30)), weights)


def _hemisphere_sum(quadrature: Hemisphere, depth: torch.Tensor, budget: float = 4e7) -> torch.Tensor:
    """Sum of ``weights * exp(-depth / |μ|)`` over quadrature nodes, for each ray and point along it, where ``depth``
    is the vertical optical depth crossed by light, of shape (n, m, c)."""
    rows = max(1, int(budget // max(1, depth[0].numel() * quadrature.weights.shape[-1])))
    parts = [
        (
            quadrature.weights[start : start + rows, None, None, :]
            * torch.exp(
                -depth[start : start + rows, ..., None] * quadrature.inverse[start : start + rows, None, None, :]
            )
        ).sum(dim=-1)
        for start in range(0, len(depth), rows)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def sky_source(
    elevations: torch.Tensor,
    depth_above: torch.Tensor,
    g: float,
    quadrature: Hemisphere | None = None,
    **resolution,
) -> torch.Tensor:
    """Skylight scattered towards rays of given elevations, at points below a given optical depth of fog.

    The sky is assumed to have a unit radiance above the horizon, and to be hidden by the ground below it.

    Args:
        elevations (torch.Tensor): Vertical component of the rays' directions, of shape (n,).
        depth_above (torch.Tensor): Optical depth of the fog above each point along each ray, of shape (n, m, c).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        quadrature (Hemisphere | None, optional): Quadrature above the horizon for these elevations, from
            :func:`hemisphere`, if already computed. Defaults to None.
        **resolution: Resolution of :func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`.

    Returns:
        torch.Tensor: Scattered radiance per unit of scattering, i.e. to be multiplied by the albedo and the sky's
        radiance, of shape (n, m, c).
    """
    return _hemisphere_sum(quadrature or hemisphere(elevations, g, **resolution), depth_above)


def ground_source(
    elevations: torch.Tensor,
    depth_above: torch.Tensor,
    depth_ground: torch.Tensor,
    g: float,
    quadrature: Hemisphere | None = None,
    **resolution,
) -> torch.Tensor:
    """Light from a Lambertian ground, of unit radiance, scattered towards rays of given elevations.

    Light leaving the ground towards a point is attenuated by the fog between them, whose optical depth is
    ``(c_ground - c) / |μ|`` for light coming from an elevation ``μ`` below the horizon.

    Args:
        elevations (torch.Tensor): Vertical component of the rays' directions, of shape (n,).
        depth_above (torch.Tensor): Optical depth of the fog above each point along each ray, of shape (n, m, c).
        depth_ground (torch.Tensor): Optical depth of the fog above the ground, of shape (c,).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        quadrature (Hemisphere | None, optional): Quadrature below the horizon for these elevations, from
            :func:`hemisphere`, if already computed. Defaults to None.
        **resolution: Resolution of :func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`.

    Returns:
        torch.Tensor: Scattered radiance per unit of scattering, i.e. to be multiplied by the albedo and the ground's
        radiance, of shape (n, m, c).
    """
    quadrature = quadrature or hemisphere(elevations, g, below=True, **resolution)
    return _hemisphere_sum(quadrature, (depth_ground - depth_above).clamp_min(0))


def ground_irradiance(
    sky: torch.Tensor, suns: Sequence[tuple[float, torch.Tensor]], depth_ground: torch.Tensor, n: int = 32
) -> torch.Tensor:
    """Irradiance of a horizontal ground, from suns and the sky, attenuated by the fog above it.

    Args:
        sky (torch.Tensor): Radiance of the sky above the horizon, per channel.
        suns (Sequence[tuple[float, torch.Tensor]]): Vertical component of each sun's unit direction, and its
            irradiance per channel.
        depth_ground (torch.Tensor): Optical depth of the fog above the ground, of shape (c,).
        n (int, optional): Number of quadrature nodes over the sky's elevations. Defaults to 32.

    Returns:
        torch.Tensor: Irradiance per channel.
    """
    kwargs = {"dtype": depth_ground.dtype, "device": depth_ground.device}
    # Sky: 2π L ∫ μ exp(-c / μ) dμ over (0, 1], where the integrand smoothly vanishes near the horizon
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n))
    mu, w = (t + 1) / 2, w / 2
    irradiance = 2 * math.pi * sky * (w * mu * torch.exp(-depth_ground[..., None] / mu)).sum(dim=-1)
    for sun_z, sun_irradiance in suns:
        if sun_z > 0:
            irradiance = irradiance + sun_irradiance * sun_z * torch.exp(-depth_ground / sun_z)
    return irradiance


def tabulate_along_rays(
    source: Callable[..., torch.Tensor],
    depth_above: torch.Tensor,
    size: tuple[int, int] = TABLE_SIZE,
    azimuths: int | None = None,
    towards: torch.Tensor | None = None,
) -> ElevationTable:
    """Integrate a source along rays of all elevations starting from a common point in an exponential height fog.

    The integral of ``σ(s) · T(s) · J(s)`` along a ray is computed as the integral of ``J`` against ``y = 1 - T``, the
    fraction of light scattered or absorbed along the ray so far, with the trapezoidal rule. Along rays going up,
    ``y`` reaches ``1 - exp(-c0 / v_z)`` when leaving the fog, and is normalized by it.

    Args:
        source (Callable[..., torch.Tensor]): Source ``J`` evaluated at rays of given elevations, of shape (n,), and,
            for tables with an azimuth axis, of given azimuths, of shape (n,), and at points along them of given
            optical depth above, of shape (n, m, c), returning a tensor of shape (n, m, c'), where c' can differ from c
            if the optical depth is the same for all channels.
        depth_above (torch.Tensor): Optical depth of the fog above the rays' origin, of shape (c,).
        size (tuple[int, int], optional): Number of elevations and optical depths. Defaults to :data:`TABLE_SIZE`.
        azimuths (int | None, optional): Number of azimuths, spread uniformly over [0, π], when the source depends on
            the rays' azimuth from a sun's. Defaults to None, i.e. it only depends on their elevation.
        towards (torch.Tensor | None, optional): Horizontal unit direction towards the sun from which azimuths are
            measured, of shape (2,), kept with the table. Defaults to None.

    Returns:
        ElevationTable: Integral of the source along rays.
    """
    kwargs = {"dtype": depth_above.dtype, "device": depth_above.device}
    elevations = elevation_grid(size[0], **kwargs)
    fraction = torch.linspace(0, 1, size[1], **kwargs)
    if azimuths:
        # Rays of every elevation and azimuth, one after another
        grid = torch.linspace(0, math.pi, azimuths, **kwargs).repeat(size[0])
        elevations = elevations.repeat_interleave(azimuths)
    v = elevations[:, None, None]
    y_max = torch.where(v > 0, -torch.expm1(-depth_above / torch.where(v > 0, v, torch.ones_like(v))), 1.0)
    y = fraction[None, :, None] * y_max
    # Beyond an optical depth of 80, light is gone anyway, and this avoids infinities where y = 1
    depth = torch.log1p(-y.clamp_max(1 - 1e-35)).clamp_min(-80.0)
    depth = (depth_above + v * depth).clamp_min(0)
    values = source(elevations, grid, depth) if azimuths else source(elevations, depth)
    steps = (values[:, 1:] + values[:, :-1]) / 2 * (y_max / (size[1] - 1))
    values = torch.cat([torch.zeros_like(values[:, :1]), steps.cumsum(dim=1)], dim=1)
    if azimuths:
        values = values.reshape(size[0], azimuths, *values.shape[1:])
    return ElevationTable(values, depth_above, towards)


def lookup(
    table: ElevationTable,
    directions_z: torch.Tensor,
    optical_depth: torch.Tensor,
    azimuths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interpolate light scattered along rays from a table computed by :func:`tabulate_along_rays`.

    Args:
        table (ElevationTable): Tabulated light scattered along rays.
        directions_z (torch.Tensor): Vertical component of the rays' unit directions, of shape (...).
        optical_depth (torch.Tensor): Optical depth of the rays, of shape (..., c), where ``c`` matches the table's
            optical depth above.
        azimuths (torch.Tensor | None, optional): Azimuth of the rays from the table's sun, in [0, π], of shape (...),
            see :func:`relative_azimuths`, for tables with an azimuth axis. Defaults to None.

    Returns:
        torch.Tensor: Light scattered along each ray, of shape (..., c') where c' is the table's number of channels.
    """
    n_elevation, *n_azimuth, n_depth, channels = table.values.shape
    v = directions_z.clamp(-1, 1)[..., None]
    y_max = torch.where(v > 0, -torch.expm1(-table.depth_above / torch.where(v > 0, v, 1.0)), 1.0)
    x = (-torch.expm1(-optical_depth) / y_max).clamp(0, 1)
    x = x.expand(*x.shape[:-1], channels) if x.shape[-1] != channels else x

    fi = _elevation_index(v[..., 0], n_elevation)
    i = fi.floor().clamp(0, n_elevation - 2).long()
    wi = fi - i
    fj = x * (n_depth - 1)
    j = fj.floor().clamp(0, n_depth - 2).long()
    # Elevations, and azimuths if any, index the rows of the table flattened over both
    rows, weights = [i, i + 1], [1 - wi, wi]
    if n_azimuth:
        if azimuths is None:
            raise ValueError("Tables with an azimuth axis need the azimuth of rays.")
        fa = azimuths.abs().clamp(0, math.pi) / math.pi * (n_azimuth[0] - 1)
        a = fa.floor().clamp(0, n_azimuth[0] - 2).long()
        wa = fa - a
        rows = [row * n_azimuth[0] + a + da for row in rows for da in (0, 1)]
        weights = [weight * w for weight in weights for w in (1 - wa, wa)]
    return _interpolate_rows(table.values.reshape(-1, n_depth, channels), rows, weights, j, fj - j)


def _interpolate_rows(
    values: torch.Tensor,
    rows: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    index: torch.Tensor,
    fraction: torch.Tensor,
    at_once: bool | None = None,
) -> torch.Tensor:
    """Weighted sum of rows of a table, each interpolated linearly along the table's second dimension.

    On a GPU, the rows are gathered at once, along a new dimension, which launches fewer kernels, as launching them
    takes most of the time when there are few points, while on the CPU, they are gathered one after another, which
    moves less memory. Both give the same result.

    Args:
        values (torch.Tensor): Table, of shape (r, d, c).
        rows (Sequence[torch.Tensor]): Index of a row of the table for each point, of shape (...) each.
        weights (Sequence[torch.Tensor]): Weight of each of these rows, of shape (...) each.
        index (torch.Tensor): Index along the second dimension of the table below which each channel of each point is
            interpolated, in [0, d - 2], of shape (..., c).
        fraction (torch.Tensor): Fraction of the way to the next index, of shape (..., c).
        at_once (bool | None, optional): Whether to gather the rows at once. Defaults to None, i.e. on a GPU.

    Returns:
        torch.Tensor: Weighted sum of the interpolated rows, of shape (..., c).
    """
    _, depths, channels = values.shape
    if values.is_cuda if at_once is None else at_once:
        # Index of the values below each point in the table flattened over all of its dimensions, of shape (..., k, c)
        flat = values.reshape(-1)
        at = torch.stack(list(rows), dim=-1)[..., None] * (depths * channels)
        at = at + (index * channels + torch.arange(channels, device=values.device))[..., None, :]
        near = flat[at]
        weighted = torch.stack(list(weights), dim=-1)[..., None] * (
            near + fraction[..., None, :] * (flat[at + channels] - near)
        )
        total = weighted[..., 0, :]
        for k in range(1, weighted.shape[-2]):
            total = total + weighted[..., k, :]
        return total
    columns = torch.arange(channels, device=values.device)
    total = torch.zeros_like(fraction)
    for row, weight in zip(rows, weights):
        near = values[row[..., None], index, columns]
        total = total + weight[..., None] * (near + fraction * (values[row[..., None], index + 1, columns] - near))
    return total


class SunScattering(NamedTuple):
    """How a sun's light scattered more than once varies with the azimuth of rays from the sun's."""

    sun_index: int
    """index of the sun among the suns of the lighting"""
    towards: torch.Tensor
    """horizontal unit direction towards the sun, of shape (2,)"""
    values: torch.Tensor
    """radiance scattered towards rays, including the albedo, minus its mean over azimuths, for azimuths from the sun's
    spread uniformly over [0, π], of shape (elevations, azimuths, heights, c)"""
    arriving: torch.Tensor
    """harmonics ``m = 1, 2, ...`` of the azimuth from the sun's of the radiance of the sun's light scattered more than
    once that reaches each height from each gathering direction, which varies as their sum of ``arriving[m - 1]
    cos(mφ)``, of shape (modes, heights, directions, c)"""
    once: torch.Tensor | None = None
    """same harmonics for the sun's light scattered once, which the phase function sends mostly from directions close
    to the sun's, of shape (modes, heights, directions, c), so that ``once + arriving`` are those of all the sun's light
    in ``ScatteringTable.gathered``"""


class ScatteringTable(NamedTuple):
    """Light scattered more than once, tabulated against the elevation of rays and the height of points.

    Heights are those at which the optical depth of the fog above them is spread uniformly from that above the ground
    (the first height) to zero (the last one), see :meth:`height_index`.
    """

    values: torch.Tensor
    """radiance scattered towards rays, including the albedo, averaged over their azimuth, of shape
    (elevations, heights, c)"""
    depth_ground: torch.Tensor
    """optical depth of the fog above the ground, of shape (c,) or (1,)"""
    gathered: torch.Tensor
    """radiance of the light scattered at least once that reaches each height from each gathering direction, and that
    the fog scatters again, averaged over azimuths, of shape (heights, directions, c)"""
    directions: torch.Tensor
    """vertical component of the gathering directions, in increasing order, of shape (directions,)"""
    ground: torch.Tensor
    """radiance of the ground, lit by suns and the sky through the fog and by the light that the fog scatters, of
    shape (c,)"""
    suns: tuple[SunScattering, ...] = ()
    """variations with azimuth of the light of each sun above the horizon"""
    from_ground: torch.Tensor | None = None
    """radiance of the light of the ground scattered once that reaches each height from each gathering direction, which
    is part of ``gathered``, of shape (heights, directions, c)"""

    def height_index(self, depth_above: torch.Tensor) -> torch.Tensor:
        """Fractional index of the tabulated heights at points of given optical depth of the fog above them.

        Args:
            depth_above (torch.Tensor): Optical depth of the fog above points, of shape (..., c) or (..., 1).

        Returns:
            torch.Tensor: Fractional index in [0, n_heights - 1], of the same shape.
        """
        return _height_index(depth_above, self.depth_ground, self.values.shape[1])


def _height_index(depth_above: torch.Tensor, depth_ground: torch.Tensor, n: int) -> torch.Tensor:
    """Fractional index of points of given optical depth above among ``n`` heights whose optical depth above is spread
    uniformly from that above the ground to zero."""
    return ((1 - depth_above / depth_ground.clamp_min(1e-30)) * (n - 1)).clamp(0, n - 1)


def _gather_nodes(n: int, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadrature over elevations in (-1, 1), clustered near the horizon and the poles, as (nodes, weights)."""
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n))
    t, w = (t * (3 - t * t) / 2 + 1) / 2, w * 3 * (1 - t * t) / 4
    return torch.cat([-t.flip(0), t]), torch.cat([w.flip(0), w])


def depth_above_ground(medium: Medium, lighting: Lighting, beta: torch.Tensor) -> torch.Tensor:
    """Optical depth of an exponential height fog above the ground, per channel."""
    fog = medium.components[0]
    assert isinstance(fog, HeightFog)
    return beta * fog.density * math.exp(-(lighting.ground_height - fog.base_height) / fog.falloff) * fog.falloff


def direct_ground_radiance(medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int) -> torch.Tensor:
    """Radiance of the ground, a Lambertian plane lit by suns and the sky through the fog, per channel, without the
    light that the fog scatters towards it."""
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    suns = []
    for sun in lighting.suns:
        towards = np.asarray(sun.direction, dtype=float)
        suns.append((towards[2] / np.linalg.norm(towards), _per_channel(sun.irradiance, channels, "sun", **kwargs)))
    sky = _per_channel(lighting.sky, channels, "sky", **kwargs)
    albedo = _per_channel(lighting.ground_albedo, channels, "ground albedo", **kwargs)
    return albedo / math.pi * ground_irradiance(sky, suns, depth_above_ground(medium, lighting, beta))


def ground_radiance(medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int) -> torch.Tensor:
    """Radiance of the ground, a Lambertian plane lit by suns and the sky through the fog, per channel.

    With ``multiple_scattering``, the ground is also lit by the light that the fog scatters, mostly forward, i.e.
    towards the ground, see :func:`multiple_scattering`, which makes up for much of the light that the fog takes out of
    sunlight and skylight on their way to the ground.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.

    Returns:
        torch.Tensor: Radiance of the ground, of shape (channels,).
    """
    if medium.multiple_scattering:
        return cached_multiple_scattering(medium, lighting, beta, channels).ground
    return direct_ground_radiance(medium, lighting, beta, channels)


def azimuthal_harmonics(
    a: torch.Tensor, b: torch.Tensor, g: float, modes: int, samples: int | None = None
) -> torch.Tensor:
    """Cosine harmonics of the Henyey-Greenstein phase function over the azimuth between directions of given elevations.

    For directions whose vertical components are ``a`` and ``b`` and whose azimuths differ by ``φ``, the ``m``-th
    harmonic is ``K_m = ∫ p(ab + sqrt(1 - a²) sqrt(1 - b²) cos(φ)) cos(mφ) dφ`` over [0, 2π], such that light that
    varies as ``cos(mφ)`` with the azimuth of the directions it comes from is scattered towards a direction of azimuth
    ``φ'`` as ``K_m cos(mφ')``. ``K_0`` is :func:`azimuthal_phase`. Harmonics are computed with the trapezoidal rule,
    i.e. with a fast Fourier transform. The phase function is analytic in ``φ`` within a strip of half-width
    ``ln(1 / |g|)`` around the real axis, at worst, so that the error of the ``m``-th harmonic decreases as ``|g|`` to
    the power of the number of samples minus ``m``.

    Args:
        a (torch.Tensor): Vertical component of the first directions, of shape (n,).
        b (torch.Tensor): Vertical component of the second directions, of shape (k,).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        modes (int): Highest harmonic.
        samples (int | None, optional): Number of azimuths at which the phase function is sampled. Defaults to None,
            i.e. enough for harmonics to be accurate to double precision.

    Returns:
        torch.Tensor: Harmonics ``K_0`` to ``K_modes``, of shape (modes + 1, n, k), in double precision.
    """
    if samples is None:
        strip = -math.log(min(max(abs(g), 1e-6), 1 - 1e-6))
        samples = 2 ** math.ceil(math.log2(max(64, modes + 1 + 37 / strip)))
    return cosine_harmonics(a, b, partial(henyey_greenstein, g=g), modes, samples)


def cosine_harmonics(
    a: torch.Tensor,
    b: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    modes: int,
    samples: int = 512,
) -> torch.Tensor:
    """Cosine harmonics of a function of the cosine of the angle between directions over the azimuth between them.

    This is ``∫ f(ab + sqrt(1 - a²) sqrt(1 - b²) cos(φ)) cos(mφ) dφ`` over [0, 2π] for directions whose vertical
    components are ``a`` and ``b``, computed with the trapezoidal rule, i.e. with a fast Fourier transform, see
    :func:`azimuthal_harmonics`.

    Args:
        a (torch.Tensor): Vertical component of the first directions, of shape (n,).
        b (torch.Tensor): Vertical component of the second directions, of shape (k,).
        function (Callable[[torch.Tensor], torch.Tensor]): Function ``f`` of the cosine of the angle between directions.
        modes (int): Highest harmonic.
        samples (int, optional): Number of azimuths at which the function is sampled. Defaults to 512.

    Returns:
        torch.Tensor: Harmonics 0 to ``modes``, of shape (modes + 1, n, k), in double precision.
    """
    a, b = a.to(torch.float64), b.to(torch.float64)
    phi = torch.arange(samples, dtype=torch.float64, device=a.device) * (2 * math.pi / samples)
    sines = (1 - b * b).clamp_min(0).sqrt()[:, None] * torch.cos(phi)  # (k, samples)
    rows = max(1, int(2**22 // (len(b) * samples)))
    parts = []
    for start in range(0, len(a), rows):
        x = a[start : start + rows, None, None]
        cos = (x * b[:, None] + (1 - x * x).clamp_min(0).sqrt() * sines).clamp(-1, 1)
        spectrum = torch.fft.rfft(function(cos), dim=-1).real[..., : modes + 1]
        parts.append(spectrum * (2 * math.pi / samples))
    return torch.cat(parts).permute(2, 0, 1)


def multiple_scattering(
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    channels: int,
    n_heights: int | None = None,
    n_directions: int = 48,
    n_steps: int = 16,
    n_elevations: int = TABLE_SIZE[0],
    higher_orders: bool = True,
    modes: int = 16,
    n_azimuths: int = AZIMUTHS,
    max_orders: int = 100,
    tolerance: float = 1e-4,
) -> ScatteringTable:
    """Compute the light scattered more than once by an exponential height fog, at a set of heights.

    Light scattered once, by the fog (from suns and the sky) or by the ground, is gathered at each height from all
    directions, along rays up to the top of the fog or down to the ground, and scattered a second time towards rays
    of each elevation with the medium's phase function. Each further order is gathered and scattered again in the
    same way, from the light of the previous order interpolated between heights, until it falls below ``tolerance``
    times the first one, and the remaining orders, which then decrease about geometrically, are extrapolated from the
    last two. The ground reflects the light of each order that reaches it, which lights the next one, and its
    radiance includes all of it.

    The fog is invariant by rotation about the vertical, so that each harmonic of the light's azimuth is scattered
    independently, see :func:`azimuthal_harmonics`. The light of the sky and of the ground doesn't depend on the
    azimuth, while that of each sun is expanded in cosines of the azimuth from the sun's, up to ``modes``, which are
    summed at ``n_azimuths`` azimuths over [0, π].

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.
        n_heights (int | None, optional): Number of heights at which light is gathered, from the ground to the top of
            the fog, see :class:`ScatteringTable`. Defaults to None, i.e. 48, or 20 per unit of the optical depth above
            the ground in denser fogs, which keeps the light lost by the interpolation between heights under 0.3%,
            whereas 48 heights lose 3% of it for an optical depth of 10.
        n_directions (int, optional): Number of directions per hemisphere over which light is gathered. Defaults to 48.
        n_steps (int, optional): Number of steps along each gathering ray. Defaults to 16.
        n_elevations (int, optional): Number of ray elevations, see :func:`elevation_grid`. Defaults to 256.
        higher_orders (bool, optional): If false, only return light scattered twice, with the ground lit by the light
            scattered once, e.g. for validation. Defaults to True.
        modes (int, optional): Highest harmonic of the azimuth from suns. Defaults to 16.
        n_azimuths (int, optional): Number of azimuths from suns, over [0, π]. Defaults to 33.
        max_orders (int, optional): Maximum number of orders computed explicitly. Defaults to 100.
        tolerance (float, optional): Size of the last order computed explicitly, relative to the first one.
            Defaults to 1e-4.

    Returns:
        ScatteringTable: Radiance scattered more than once towards rays, including the medium's albedo.
    """
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    g, albedo = medium.anisotropy, medium.albedo
    sky = _per_channel(lighting.sky, channels, "sky", **kwargs)
    reflectance = _per_channel(lighting.ground_albedo, channels, "ground albedo", **kwargs) / math.pi
    direct = direct_ground_radiance(medium, lighting, beta, channels)
    depth_ground = depth_above_ground(medium, lighting, beta)
    n_heights = n_heights or max(48, math.ceil(20 * float(depth_ground.max())))
    # Light only depends on the optical depth of the fog above points, the vertical coordinate of radiative transfer in
    # a horizontally uniform fog, which is spread uniformly: its variations near the ground, where the fog scatters
    # most light, are as well resolved in dense fog as in light fog
    depth_at = depth_ground[None, :] * torch.linspace(1, 0, n_heights, **kwargs)[:, None]  # (heights, c)
    omega, omega_weights = _gather_nodes(n_directions, **kwargs)  # (d,)
    suns, indices = [], []
    for i, sun in enumerate(lighting.suns):
        towards = np.asarray(sun.direction, dtype=float) / np.linalg.norm(sun.direction)
        if towards[2] > 0:
            suns.append((towards, _per_channel(sun.irradiance, channels, "sun", **kwargs)))
            indices.append(i)

    # Along each gathering ray, points are spread evenly in the fraction of light scattered or absorbed between them
    # and the ray's origin, up to the top of the fog or the ground, which gives the light scattered once in closed form
    # for suns, the sky and the ground, and the height of points at which further orders are interpolated
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_steps))
    t, w = (t + 1) / 2, w / 2
    v = omega[:, None]  # (d, 1)
    up = v > 0
    index = torch.zeros(n_heights, len(omega), n_steps, len(beta), **kwargs)
    weights = torch.zeros_like(index)
    sky_once = torch.zeros(n_heights, len(omega), channels, **kwargs)
    ground_once = torch.zeros(n_heights, len(omega), len(beta), **kwargs)
    sun_once = torch.zeros(len(suns), n_heights, len(omega), len(beta), **kwargs)
    # Light scattered once only needs a coarser quadrature, as it is integrated over directions again
    above, beneath = (hemisphere(omega, g, below=below, n_angle=8, n_azimuth=12) for below in (False, True))
    for h in range(n_heights):
        c0 = depth_at[h]  # (c,)
        # Transmittance to the top of the fog, or to the ground
        end = torch.where(
            up, torch.exp(-c0 / torch.where(up, v, torch.ones_like(v))), torch.exp(-(depth_ground - c0) / v.abs())
        )
        y = (1 - end)[:, None, :] * t[None, :, None]  # (d, steps, c)
        depth = (c0 + v[..., None] * torch.log1p(-y).clamp_min(-80.0)).clamp_min(0)
        index[h] = _height_index(depth, depth_ground, n_heights)
        weights[h] = (1 - end)[:, None, :] * w[None, :, None]
        sky_once[h] = albedo * sky * (sky_source(omega, depth, g, quadrature=above) * weights[h]).sum(dim=1)
        below = ground_source(omega, depth, depth_ground, g, quadrature=beneath)
        ground_once[h] = albedo * (below * weights[h]).sum(dim=1)
        for k, (towards, _) in enumerate(suns):
            sun_once[k, h] = (torch.exp(-depth / float(towards[2])) * weights[h]).sum(dim=1)
    # Light arriving at a height from a direction is then a weighted sum of its source at every height, interpolated
    # between heights at the points of the gathering ray, i.e. a linear map for each direction and channel
    lower = index.floor().clamp(0, n_heights - 2).long()
    fraction = index - lower
    shape = (len(beta), len(omega), n_heights, n_heights)
    rows = torch.arange(shape[0] * shape[1] * shape[2], device=beta.device).reshape(shape[0], shape[1], shape[2])
    rows = rows.permute(2, 1, 0)[:, :, None, :] * n_heights  # (heights, d, 1, c), first entry of each map's row
    gathering = torch.zeros(math.prod(shape), **kwargs)
    gathering.index_add_(0, (rows + lower).reshape(-1), (weights * (1 - fraction)).reshape(-1))
    gathering.index_add_(0, (rows + lower + 1).reshape(-1), (weights * fraction).reshape(-1))
    gathering = gathering.reshape(shape).expand(channels, -1, -1, -1)

    def gather(source: torch.Tensor) -> torch.Tensor:
        # Light arriving at each height from each gathering direction, of shape (harmonics, heights, d, c), given its
        # source towards each gathering direction at each height, of shape (harmonics, d, heights, c)
        return torch.einsum("cdhg,kdgc->khdc", gathering, source)

    elevations = elevation_grid(n_elevations, **kwargs)
    towards_z = torch.as_tensor([float(towards[2]) for towards, _ in suns], **kwargs)
    harmonics = azimuthal_harmonics(torch.cat([omega, elevations, towards_z]), omega, g, modes if suns else 0)
    # The mean over azimuths is exact, see `azimuthal_phase`
    mean = azimuthal_phase(torch.cat([omega, elevations, towards_z]).double()[:, None], omega.double()[None, :], g)
    harmonics = torch.cat([mean[None], harmonics[1:]]).to(beta.dtype)
    to_nodes, to_elevations, from_suns = harmonics.split([len(omega), n_elevations, len(suns)], dim=1)

    def scatter(kernel: torch.Tensor, light: torch.Tensor) -> torch.Tensor:
        # Light scattered towards the directions of a kernel of shape (harmonics, x, d), of shape (harmonics, x,
        # heights, c), given the light arriving from each gathering direction, of shape (harmonics, heights, d, c)
        k, h, d, c = light.shape
        weighted = (light * omega_weights[:, None]).transpose(1, 2).reshape(k, d, h * c)
        return albedo * torch.bmm(kernel, weighted).reshape(k, -1, h, c)

    def irradiance(light: torch.Tensor) -> torch.Tensor:
        # Irradiance of the ground from the light arriving at the lowest height from above, given light of shape
        # (heights, d, c)
        return 2 * math.pi * ((omega_weights * omega * up[:, 0])[:, None] * light[0]).sum(dim=0)

    def successive(first: torch.Tensor, kernel: torch.Tensor, lit_ground: bool) -> torch.Tensor:
        # Sum of all orders of light arriving at each height from each direction, given the first one
        total, light = first.clone(), first
        if not higher_orders:
            return total
        first_size = first.abs().sum(dim=(1, 2, 3)).clamp_min(1e-30)
        size, ratio = first_size, torch.zeros_like(first_size)
        for _ in range(max_orders):
            following = gather(scatter(kernel, light))
            if lit_ground:
                # The ground reflects the light that reaches it, which the fog scatters
                following[0] = following[0] + ground_once * (reflectance * irradiance(light[0]))
            ratio, size = following.abs().sum(dim=(1, 2, 3)) / size.clamp_min(1e-30), following.abs().sum(dim=(1, 2, 3))
            total, light = total + following, following
            if bool((size <= tolerance * first_size).all()):
                break
        # Further orders decrease about geometrically
        ratio = ratio.clamp(0, 0.99)[:, None, None, None]
        return total + light * ratio / (1 - ratio)

    # Light scattered once, averaged over azimuths, from suns, the sky and the ground lit directly
    first = sky_once + ground_once * direct
    for k, (_, irradiance_k) in enumerate(suns):
        first = first + albedo * irradiance_k * from_suns[0, k][None, :, None] / (2 * math.pi) * sun_once[k]
    gathered = successive(first[None], to_nodes[:1], lit_ground=True)[0]
    ground = direct + reflectance * irradiance(gathered)
    values = scatter(to_elevations[:1], gathered[None])[0]

    # Harmonics of the azimuth from each sun: its light scattered once arriving from directions of azimuth φ from the
    # sun's varies as the phase function, whose m-th harmonic, K_m cos(mφ) / π, is scattered independently of others
    tables = []
    azimuths = torch.linspace(0, math.pi, n_azimuths, **kwargs)
    cosines = torch.cos(azimuths[:, None] * torch.arange(1, modes + 1, **kwargs))  # (a, modes)
    for k, (towards, irradiance_k) in enumerate(suns):
        horizontal = float(np.hypot(towards[0], towards[1]))
        if modes < 1 or horizontal < 1e-9:
            # A sun at the zenith lights the fog the same at every azimuth
            continue
        first = albedo * irradiance_k * from_suns[1:, k][:, None, :, None] / math.pi * sun_once[k][None]
        light = successive(first, to_nodes[1:], lit_ground=False)
        variation = torch.einsum("mehc,am->eahc", scatter(to_elevations[1:], light), cosines)
        direction = torch.as_tensor([towards[0] / horizontal, towards[1] / horizontal], **kwargs)
        tables.append(SunScattering(indices[k], direction, variation, light - first, first))
    return ScatteringTable(values, depth_ground, gathered, omega, ground, tuple(tables), ground_once * ground)


def scattered_source(table: ScatteringTable, elevations: torch.Tensor, depth_above: torch.Tensor) -> torch.Tensor:
    """Interpolate a :class:`ScatteringTable`, averaged over azimuths, at points along rays of given elevations.

    Args:
        table (ScatteringTable): Light scattered more than once.
        elevations (torch.Tensor): Vertical component of the rays' directions, of shape (n,).
        depth_above (torch.Tensor): Optical depth of the fog above each point along each ray, of shape (n, m, c).

    Returns:
        torch.Tensor: Radiance scattered towards each ray, of shape (n, m, c').
    """
    n_elevations, n_heights, channels = table.values.shape
    fe = _elevation_index(elevations, n_elevations)
    e = fe.floor().clamp(0, n_elevations - 2).long()[:, None, None]
    we = (fe - e[:, 0, 0])[:, None, None]
    fh = table.height_index(depth_above)
    fh = fh.expand(*fh.shape[:-1], channels) if fh.shape[-1] != channels else fh
    h = fh.floor().clamp(0, n_heights - 2).long()
    wh = fh - h
    ch = torch.arange(channels, device=depth_above.device)

    def at(de: int) -> torch.Tensor:
        rows = table.values[e + de, h, ch]
        return rows + wh * (table.values[e + de, h + 1, ch] - rows)

    below = at(0)
    return below + we * (at(1) - below)


def sun_scattered_source(
    table: ScatteringTable,
    sun: int,
    elevations: torch.Tensor,
    azimuths: torch.Tensor,
    depth_above: torch.Tensor,
) -> torch.Tensor:
    """Interpolate how a sun's light scattered more than once varies with azimuth, at points along rays.

    Args:
        table (ScatteringTable): Light scattered more than once.
        sun (int): Index of the sun in ``table.suns``.
        elevations (torch.Tensor): Vertical component of the rays' directions, of shape (n,).
        azimuths (torch.Tensor): Azimuth of the rays from the sun's, in [0, π], of shape (n,), see
            :func:`relative_azimuths`.
        depth_above (torch.Tensor): Optical depth of the fog above each point along each ray, of shape (n, m, c).

    Returns:
        torch.Tensor: Radiance scattered towards each ray minus its mean over azimuths, of shape (n, m, c').
    """
    values = table.suns[sun].values
    n_elevations, n_azimuths, n_heights, channels = values.shape
    fe = _elevation_index(elevations, n_elevations)
    e = fe.floor().clamp(0, n_elevations - 2).long()[:, None, None]
    we = (fe - e[:, 0, 0])[:, None, None]
    fa = azimuths.abs().clamp(0, math.pi) / math.pi * (n_azimuths - 1)
    a = fa.floor().clamp(0, n_azimuths - 2).long()[:, None, None]
    wa = (fa - a[:, 0, 0])[:, None, None]
    fh = table.height_index(depth_above)
    fh = fh.expand(*fh.shape[:-1], channels) if fh.shape[-1] != channels else fh
    h = fh.floor().clamp(0, n_heights - 2).long()
    wh = fh - h
    ch = torch.arange(channels, device=depth_above.device)

    def at(de: int, da: int) -> torch.Tensor:
        rows = values[e + de, a + da, h, ch]
        return rows + wh * (values[e + de, a + da, h + 1, ch] - rows)

    near = at(0, 0) + wa * (at(0, 1) - at(0, 0))
    far = at(1, 0) + wa * (at(1, 1) - at(1, 0))
    return near + we * (far - near)


def elevation_sources(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Light scattered towards rays by sources whose light only depends on the rays' elevation: the sky, the ground,
    and, if enabled, light scattered more than once, averaged over azimuths, see :func:`azimuthal_sources` for the
    rest.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.

    Returns:
        dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]: Radiance scattered from the "sky", the
        "ground" and light already "scattered", for those that shine, as functions of the rays' elevations, of shape
        (n,), and of the optical depth above points along them, of shape (n, m, c), including the medium's albedo,
        of shape (n, m, channels).
    """
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    g = medium.anisotropy
    sky = medium.albedo * _per_channel(lighting.sky, channels, "sky", **kwargs)
    ground = medium.albedo * ground_radiance(medium, lighting, beta, channels)
    depth_ground = depth_above_ground(medium, lighting, beta)
    sources: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}

    def quadrature(below: bool) -> Callable[[torch.Tensor], Hemisphere]:
        # Quadratures only depend on the rays' elevations, which are the same at every step when ray marching
        last: dict = {"elevations": None}

        def cached(elevations: torch.Tensor) -> Hemisphere:
            if elevations is not last["elevations"]:
                last["elevations"], last["quadrature"] = elevations, hemisphere(elevations, g, below=below)
            return last["quadrature"]

        return cached

    if bool((sky != 0).any()):
        above = quadrature(below=False)
        sources["sky"] = lambda elevations, depth: sky * sky_source(elevations, depth, g, quadrature=above(elevations))
    if bool((ground != 0).any()):
        beneath = quadrature(below=True)
        sources["ground"] = lambda elevations, depth: (
            ground * ground_source(elevations, depth, depth_ground, g, quadrature=beneath(elevations))
        )
    # Without suns above the horizon or skylight, e.g. at night, no light is scattered more than once
    if medium.multiple_scattering and environment_lit(lighting):
        table = cached_multiple_scattering(medium, lighting, beta, channels)
        sources["scattered"] = partial(scattered_source, table)
    return sources


def azimuthal_sources(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int
) -> list[tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]:
    """How the light of each sun scattered more than once varies with the azimuth of rays from the sun's, if enabled.

    Light scattered more than once still goes mostly the way sunlight went, as the fog scatters it forward: it is
    brighter towards the sun than away from it. This is the difference with its mean over azimuths, which is part of
    :func:`elevation_sources`.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.

    Returns:
        list[tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]: For each sun
        above the horizon but not at the zenith, its horizontal unit direction, of shape (2,), and the variation of its
        light scattered more than once as a function of the rays' elevations, of shape (n,), of their azimuths from the
        sun's, of shape (n,), and of the optical depth above points along them, of shape (n, m, c), including the
        medium's albedo, of shape (n, m, channels).
    """
    if not medium.multiple_scattering:
        return []
    table = cached_multiple_scattering(medium, lighting, beta, channels)
    return [(sun.towards, partial(sun_scattered_source, table, k)) for k, sun in enumerate(table.suns)]


def elevation_source(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
    """Sum of the sources of :func:`elevation_sources`.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None: Scattered radiance as a function of the rays'
        elevations, of shape (n,), and of the optical depth above points along them, of shape (n, m, c), including
        the medium's albedo, of shape (n, m, channels). None when there is no such light.
    """
    sources = list(elevation_sources(medium, lighting, beta, channels).values())
    if not sources:
        return None

    def source(elevations: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        total = sources[0](elevations, depth)
        for other in sources[1:]:
            total = total + other(elevations, depth)
        return total

    return source


def environment_key(lighting: Lighting) -> str:
    """Key of the parts of a lighting that light reaching the medium from many directions depends on: the sky, suns and
    the ground, leaving out lamps, which don't change it, so that tables cached on it survive lamps that move.

    Args:
        lighting (Lighting): Lighting of the medium.

    Returns:
        str: JSON of the lighting without its lamps.
    """
    return lighting.model_copy(update={"points": [], "spots": [], "areas": [], "emissive": []}).model_dump_json()


def environment_lit(lighting: Lighting) -> bool:
    """Whether suns above the horizon or the sky shine, without which, e.g. at night, neither they, nor the ground, which
    only reflects their light, nor light scattered more than once light the medium.

    Args:
        lighting (Lighting): Lighting of the medium.

    Returns:
        bool: Whether the sky or a sun above the horizon has some light.
    """
    suns = (sun for sun in lighting.suns if sun.direction[2] > 0)
    return any(v != 0 for v in lighting.sky) or any(any(v != 0 for v in sun.irradiance) for sun in suns)


def camera_tables(
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    channels: int,
    depth_above: torch.Tensor,
    size: tuple[int, int] = TABLE_SIZE,
    combined: bool = False,
) -> dict[str, ElevationTable]:
    """Integrate the sources of :func:`elevation_sources` and :func:`azimuthal_sources` along rays of every elevation
    (and azimuth from suns) starting from a camera, see :func:`tabulate_along_rays`.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.
        depth_above (torch.Tensor): Optical depth of the fog above the camera, of the same shape as ``beta``.
        size (tuple[int, int], optional): Number of elevations and optical depths of the tables of the sources of
            :func:`elevation_sources`. Those of :func:`azimuthal_sources` are tabulated at :data:`AZIMUTH_TABLE_SIZE`
            and :data:`AZIMUTHS`. Defaults to :data:`TABLE_SIZE`.
        combined (bool, optional): If true, the sources of :func:`elevation_sources` are summed into a single table, as
            when nothing occludes them. Defaults to False.

    Returns:
        dict[str, ElevationTable]: Tables of the sources of :func:`elevation_sources`, keyed by their names, or by "all"
        for their sum when ``combined``, followed by those of the variations of each sun's light scattered more than
        once with azimuth, keyed by "scattered <k>". Empty when no such light reaches the medium, e.g. at night.
    """
    tables: dict[str, ElevationTable] = {}
    if not combined:
        for name, source in elevation_sources(medium, lighting, beta, channels).items():
            tables[name] = tabulate_along_rays(source, depth_above, size)
    elif summed := elevation_source(medium, lighting, beta, channels):
        tables["all"] = tabulate_along_rays(summed, depth_above, size)
    for k, (towards, varying) in enumerate(azimuthal_sources(medium, lighting, beta, channels)):
        tables[f"scattered {k}"] = tabulate_along_rays(
            varying, depth_above, AZIMUTH_TABLE_SIZE, azimuths=AZIMUTHS, towards=towards
        )
    return tables


CAMERA_TABLES_CACHE_SIZE = 8
"""Number of sets of tables kept by :func:`cached_camera_tables`, about 4 MB each in single precision"""
_CAMERA_TABLES: OrderedDict[tuple, dict[str, ElevationTable]] = OrderedDict()
"""Tables kept by :func:`cached_camera_tables`, from the least recently used"""


def cached_camera_tables(
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    channels: int,
    depth_above: torch.Tensor,
    size: tuple[int, int] = TABLE_SIZE,
    combined: bool = False,
) -> dict[str, ElevationTable]:
    """Same as :func:`camera_tables`, kept for the last :data:`CAMERA_TABLES_CACHE_SIZE` media, lightings and optical
    depths above the camera, i.e. heights of the camera, as tables don't depend on anything else: they are computed once
    for cameras that move horizontally, which saves most of the time spent per frame on light from the sky, the ground
    and light scattered more than once when there are few pixels. Tables are looked up by the exact values of
    ``beta`` and ``depth_above``, so that results are the same as computing them again, and should be treated as
    read-only."""
    key = (
        medium.model_dump_json(),
        environment_key(lighting),
        tuple(torch.cat([beta, depth_above]).tolist()),
        channels,
        tuple(size),
        combined,
        beta.dtype,
        str(beta.device),
    )
    tables = _CAMERA_TABLES.pop(key, None)
    if tables is None:
        tables = camera_tables(medium, lighting, beta, channels, depth_above, size, combined)
    _CAMERA_TABLES[key] = tables
    while len(_CAMERA_TABLES) > CAMERA_TABLES_CACHE_SIZE:
        _CAMERA_TABLES.popitem(last=False)
    return dict(tables)


@lru_cache(maxsize=8)
def _cached_multiple_scattering(
    medium: str, lighting: str, beta: tuple, channels: int, dtype: torch.dtype, device: str
) -> ScatteringTable:
    return multiple_scattering(
        Medium.model_validate_json(medium),
        Lighting.model_validate_json(lighting),
        torch.tensor(beta, dtype=dtype, device=device),
        channels,
    )


def cached_multiple_scattering(medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int) -> ScatteringTable:
    """Same as :func:`multiple_scattering`, cached for the last few media and lightings, as it does not depend on the
    camera and can be reused for every frame."""
    return _cached_multiple_scattering(
        medium.model_dump_json(),
        environment_key(lighting),
        tuple(beta.tolist()),
        channels,
        beta.dtype,
        str(beta.device),
    )
