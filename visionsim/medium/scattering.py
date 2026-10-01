"""Light that reaches an exponential height fog from many directions: from the sky, the ground and the fog itself.

Unlike sunlight, which comes from a single direction, skylight, light reflected by the ground and light that was
already scattered by the fog reach every point from many directions. The light they scatter towards a ray then only
depends on the ray's elevation and on the height of the point, and not on the ray's azimuth. Such sources are
integrated along camera rays with a table of their integral against the rays' elevation and optical depth, computed
once per frame since all camera rays start from the same point, and interpolated for each pixel.

Heights are expressed as ``c = σ(z) · H``, the optical depth of the fog above a point, where ``σ`` is the extinction
coefficient and ``H`` the fog's falloff, such that light coming from an elevation ``μ`` above the horizon is attenuated
by ``exp(-c / μ)`` before reaching the point. Along a ray of elevation ``v_z`` starting where the optical depth above
is ``c0``, the optical depth above a point beyond which the ray has an optical depth ``τ`` is ``c0 - v_z · τ``.

Light scattered more than once is approximated following Hillaire, "A Scalable and Production Ready Sky and
Atmosphere Rendering Technique" (EGSR 2020): light scattered a second time is gathered at a set of heights from light
scattered once (by the fog, or reflected by the ground), and higher orders are assumed to follow the same
distribution, each order being a fraction ``f`` of the previous one, which sums up to a factor ``1 / (1 - f)``.
Unlike Hillaire, the second order is gathered with the medium's phase function rather than an isotropic one, as fog
scatters light strongly forward.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch

from visionsim.medium.model import HeightFog, Lighting, Medium
from visionsim.medium.optics import _per_channel, sky_quadrature

TABLE_SIZE: tuple[int, int] = (256, 257)
"""Default number of elevations and optical depths at which light scattered along camera rays is tabulated"""

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
    fog.
    """

    values: torch.Tensor
    """integral of the source along rays, of shape (elevations, depths, c)"""
    depth_above: torch.Tensor
    """optical depth of the fog above the rays' origin, of shape (c,) or (1,) when it is the same for all channels"""


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
    mirrored across the horizon, which preserves the angles between directions.

    Args:
        elevations (torch.Tensor): Vertical component of the rays' directions, of shape (n,).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        below (bool, optional): If true, integrate over directions below the horizon. Defaults to False.
        **resolution: Resolution of :func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`.

    Returns:
        Hemisphere: Inverse elevations and weights of the quadrature's nodes.
    """
    mu, weights = sky_quadrature(_elevation_directions(-elevations if below else elevations), g, **resolution)
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
    source: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    depth_above: torch.Tensor,
    size: tuple[int, int] = TABLE_SIZE,
) -> ElevationTable:
    """Integrate a source along rays of all elevations starting from a common point in an exponential height fog.

    The integral of ``σ(s) · T(s) · J(s)`` along a ray is computed as the integral of ``J`` against ``y = 1 - T``, the
    fraction of light scattered or absorbed along the ray so far, with the trapezoidal rule. Along rays going up,
    ``y`` reaches ``1 - exp(-c0 / v_z)`` when leaving the fog, and is normalized by it.

    Args:
        source (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Source ``J`` evaluated at rays of given
            elevations, of shape (n,), and points along them of given optical depth above, of shape (n, m, c),
            returning a tensor of shape (n, m, c'), where c' can differ from c if the optical depth is the same for
            all channels.
        depth_above (torch.Tensor): Optical depth of the fog above the rays' origin, of shape (c,).
        size (tuple[int, int], optional): Number of elevations and optical depths. Defaults to :data:`TABLE_SIZE`.

    Returns:
        ElevationTable: Integral of the source along rays.
    """
    kwargs = {"dtype": depth_above.dtype, "device": depth_above.device}
    elevations = elevation_grid(size[0], **kwargs)
    fraction = torch.linspace(0, 1, size[1], **kwargs)
    v = elevations[:, None, None]
    y_max = torch.where(v > 0, -torch.expm1(-depth_above / torch.where(v > 0, v, torch.ones_like(v))), 1.0)
    y = fraction[None, :, None] * y_max
    # Beyond an optical depth of 80, light is gone anyway, and this avoids infinities where y = 1
    depth = torch.log1p(-y.clamp_max(1 - 1e-35)).clamp_min(-80.0)
    values = source(elevations, (depth_above + v * depth).clamp_min(0))
    steps = (values[:, 1:] + values[:, :-1]) / 2 * (y_max / (size[1] - 1))
    return ElevationTable(torch.cat([torch.zeros_like(values[:, :1]), steps.cumsum(dim=1)], dim=1), depth_above)


def lookup(table: ElevationTable, directions_z: torch.Tensor, optical_depth: torch.Tensor) -> torch.Tensor:
    """Interpolate light scattered along rays from a table computed by :func:`tabulate_along_rays`.

    Args:
        table (ElevationTable): Tabulated light scattered along rays.
        directions_z (torch.Tensor): Vertical component of the rays' unit directions, of shape (...).
        optical_depth (torch.Tensor): Optical depth of the rays, of shape (..., c), where ``c`` matches the table's
            optical depth above.

    Returns:
        torch.Tensor: Light scattered along each ray, of shape (..., c') where c' is the table's number of channels.
    """
    n_elevation, n_depth, channels = table.values.shape
    v = directions_z.clamp(-1, 1)[..., None]
    y_max = torch.where(v > 0, -torch.expm1(-table.depth_above / torch.where(v > 0, v, torch.ones_like(v))), 1.0)
    x = (-torch.expm1(-optical_depth) / y_max).clamp(0, 1)
    x = x.expand(*x.shape[:-1], channels) if x.shape[-1] != channels else x

    fi = _elevation_index(v[..., 0], n_elevation)
    i = fi.floor().clamp(0, n_elevation - 2).long()
    wi = (fi - i)[..., None]
    fj = x * (n_depth - 1)
    j = fj.floor().clamp(0, n_depth - 2).long()
    wj = fj - j
    ch = torch.arange(channels, device=x.device)

    def at(di: int, dj: int) -> torch.Tensor:
        return table.values[(i + di)[..., None], j + dj, ch]

    near = at(0, 0) + wj * (at(0, 1) - at(0, 0))
    far = at(1, 0) + wj * (at(1, 1) - at(1, 0))
    return near + wi * (far - near)


class ScatteringTable(NamedTuple):
    """Light scattered more than once, tabulated against the elevation of rays and the height of points."""

    values: torch.Tensor
    """radiance scattered towards rays, including the albedo, of shape (elevations, heights, c)"""
    depth_ground: torch.Tensor
    """optical depth of the fog above the ground, of shape (c,) or (1,)"""
    height_scale: float
    """heights are spread uniformly between the ground and ``height_scale`` falloffs above it"""


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


def ground_radiance(medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int) -> torch.Tensor:
    """Radiance of the ground, a Lambertian plane lit by suns and the sky through the fog, per channel."""
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    suns = []
    for sun in lighting.suns:
        towards = np.asarray(sun.direction, dtype=float)
        suns.append((towards[2] / np.linalg.norm(towards), _per_channel(sun.irradiance, channels, "sun", **kwargs)))
    sky = _per_channel(lighting.sky, channels, "sky", **kwargs)
    albedo = _per_channel(lighting.ground_albedo, channels, "ground albedo", **kwargs)
    return albedo / math.pi * ground_irradiance(sky, suns, depth_above_ground(medium, lighting, beta))


def multiple_scattering(
    medium: Medium,
    lighting: Lighting,
    beta: torch.Tensor,
    channels: int,
    n_heights: int = 40,
    height_scale: float = 12.0,
    n_directions: int = 48,
    n_steps: int = 16,
    n_elevations: int = TABLE_SIZE[0],
    higher_orders: bool = True,
) -> ScatteringTable:
    """Approximate the light scattered more than once by an exponential height fog, at a set of heights.

    Light scattered once, by the fog (from suns and the sky) or by the ground, is gathered at each height from all
    directions, and scattered a second time towards rays of each elevation with the medium's phase function. Light
    scattered once is averaged over azimuths, as its only azimuthal dependence comes from suns. Higher orders are
    assumed to be distributed like the second one, each a fraction ``f`` of the previous, where ``f`` is the fraction
    of light scattered isotropically at that height that is scattered again before leaving the fog or reaching the
    ground, which adds up to ``1 / (1 - f)`` times the second order.

    Args:
        medium (Medium): Participating medium, made of a single height fog, with ``sun_attenuation`` enabled.
        lighting (Lighting): Lighting of the medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) for all channels.
        channels (int): Number of channels of the lighting.
        n_heights (int, optional): Number of heights at which light is gathered. Defaults to 40.
        height_scale (float, optional): Heights span from the ground to this many falloffs above it. Defaults to 12.
        n_directions (int, optional): Number of directions per hemisphere over which light is gathered. Defaults to 48.
        n_steps (int, optional): Number of steps along each gathering ray. Defaults to 16.
        n_elevations (int, optional): Number of ray elevations, spread uniformly over [-1, 1]. Defaults to 256.
        higher_orders (bool, optional): If false, only return light scattered twice, e.g. for validation.
            Defaults to True.

    Returns:
        ScatteringTable: Radiance scattered more than once towards rays, including the medium's albedo.
    """
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    g, albedo = medium.anisotropy, medium.albedo
    sky = _per_channel(lighting.sky, channels, "sky", **kwargs)
    ground = ground_radiance(medium, lighting, beta, channels)
    depth_ground = depth_above_ground(medium, lighting, beta)
    heights = torch.linspace(0, height_scale, n_heights, **kwargs)
    depth_at = depth_ground[None, :] * torch.exp(-heights)[:, None]  # (heights, c)

    # Light scattered once towards points at each height, from gathering directions, averaged over azimuths
    omega, omega_weights = _gather_nodes(n_directions, **kwargs)  # (d,)
    suns = []
    for sun in lighting.suns:
        towards = np.asarray(sun.direction, dtype=float)
        sun_z = float(towards[2] / np.linalg.norm(towards))
        if sun_z > 0:
            phase = azimuthal_phase(omega.double(), omega.new_tensor(sun_z).double(), g).to(beta.dtype) / (2 * math.pi)
            suns.append((sun_z, phase, _per_channel(sun.irradiance, channels, "sun", **kwargs)))
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_steps))
    t, w = (t + 1) / 2, w / 2
    once = torch.zeros(n_heights, len(omega), channels, **kwargs)
    escape = torch.zeros(n_heights, len(omega), len(beta), **kwargs)
    v = omega[:, None]  # (d, 1)
    up = v > 0
    for h in range(n_heights):
        c0 = depth_at[h]  # (c,)
        # Transmittance to the top of the fog, or to the ground
        end = torch.where(
            up, torch.exp(-c0 / torch.where(up, v, torch.ones_like(v))), torch.exp(-(depth_ground - c0) / v.abs())
        )
        escape[h] = end
        y = (1 - end)[:, None, :] * t[None, :, None]  # (d, steps, c)
        depth = (c0 + v[..., None] * torch.log1p(-y).clamp_min(-80.0)).clamp_min(0)
        # Light scattered once only needs a coarser quadrature, as it is integrated over directions again
        source = sky * sky_source(omega, depth, g, n_angle=8, n_azimuth=12)
        source = source + ground * ground_source(omega, depth, depth_ground, g, n_angle=8, n_azimuth=12)
        for sun_z, phase, irradiance in suns:
            source = source + phase[:, None, None] * irradiance * torch.exp(-depth / sun_z)
        once[h] = albedo * (source * w[None, :, None]).sum(dim=1) * (1 - end) + (~up) * end * ground

    # Second order towards rays of each elevation, and fraction of light scattered isotropically that scatters again
    elevations = elevation_grid(n_elevations, **kwargs)
    kernel = azimuthal_phase(omega[None, :].double(), elevations[:, None].double(), g).to(beta.dtype)  # (e, d)
    second = albedo * torch.einsum("ed,hdc,d->ehc", kernel, once, omega_weights)
    again = albedo * ((1 - escape) * omega_weights[None, :, None]).sum(dim=1) / 2  # (heights, c)
    if not higher_orders:
        return ScatteringTable(second, depth_ground, height_scale)
    return ScatteringTable(second / (1 - again[None]).clamp_min(1e-3), depth_ground, height_scale)


def scattered_source(table: ScatteringTable, elevations: torch.Tensor, depth_above: torch.Tensor) -> torch.Tensor:
    """Interpolate a :class:`ScatteringTable` at points along rays of given elevations.

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
    heights = torch.log(table.depth_ground / depth_above.clamp_min(1e-30)).clamp(0, table.height_scale)
    fh = heights / table.height_scale * (n_heights - 1)
    fh = fh.expand(*fh.shape[:-1], channels) if fh.shape[-1] != channels else fh
    h = fh.floor().clamp(0, n_heights - 2).long()
    wh = fh - h
    ch = torch.arange(channels, device=depth_above.device)

    def at(de: int) -> torch.Tensor:
        rows = table.values[e + de, h, ch]
        return rows + wh * (table.values[e + de, h + 1, ch] - rows)

    below = at(0)
    return below + we * (at(1) - below)


def elevation_source(
    medium: Medium, lighting: Lighting, beta: torch.Tensor, channels: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
    """Light scattered towards rays by sources whose light only depends on the rays' elevation: the sky, the ground,
    and, if enabled, light scattered more than once.

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
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    g = medium.anisotropy
    sky = medium.albedo * _per_channel(lighting.sky, channels, "sky", **kwargs)
    ground = medium.albedo * ground_radiance(medium, lighting, beta, channels)
    depth_ground = depth_above_ground(medium, lighting, beta)
    lit_by_sky, lit_by_ground = bool((sky != 0).any()), bool((ground != 0).any())
    table = cached_multiple_scattering(medium, lighting, beta, channels) if medium.multiple_scattering else None
    if not (lit_by_sky or lit_by_ground or table):
        return None

    # Quadratures only depend on the rays' elevations, which are the same at every step when ray marching
    last: dict = {"elevations": None}

    def source(elevations: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if elevations is not last["elevations"]:
            last["elevations"] = elevations
            if lit_by_sky:
                last["sky"] = hemisphere(elevations, g)
            if lit_by_ground:
                last["ground"] = hemisphere(elevations, g, below=True)
        total = torch.zeros(1, **kwargs)
        if lit_by_sky:
            total = total + sky * sky_source(elevations, depth, g, quadrature=last["sky"])
        if lit_by_ground:
            total = total + ground * ground_source(elevations, depth, depth_ground, g, quadrature=last["ground"])
        if table is not None:
            total = total + scattered_source(table, elevations, depth)
        return total

    return source


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
        lighting.model_dump_json(),
        tuple(beta.tolist()),
        channels,
        beta.dtype,
        str(beta.device),
    )
