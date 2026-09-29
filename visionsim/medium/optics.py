"""Closed-form expressions for single scattering in participating media.

All functions operate on torch tensors and broadcast over rays. Rays start at ``origin``, travel along unit
``directions`` in world space (where z is up), and end after ``distance`` meters, typically on a surface. Rays that
do not hit anything can use an infinite distance, in which case the medium is integrated all the way to infinity.

Optical depths are relative, i.e. computed for an extinction coefficient of one, and need to be scaled by the
extinction coefficient at the wavelength of interest (see :meth:`Medium.extinction_at
<visionsim.medium.model.Medium.extinction_at>`).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from visionsim.medium.model import Blob, Component, HeightFog, Homogeneous, Medium


def _exprel(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``(1 - exp(-x)) / x``, which tends to one as x tends to zero."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-safe) / safe)


def henyey_greenstein(cos_theta: torch.Tensor, g: float) -> torch.Tensor:
    """Henyey-Greenstein phase function.

    Args:
        cos_theta (torch.Tensor): Cosine of the scattering angle, i.e. between the direction light travels in
            before and after scattering. For light scattered towards a camera, this is the dot product between
            the camera ray's direction and the direction pointing towards the light.
        g (float): Asymmetry parameter in (-1, 1), positive values scatter light forward.

    Returns:
        torch.Tensor: Phase function value, in 1/sr, which integrates to one over the sphere.
    """
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cos_theta).clamp_min(1e-12) ** 1.5)


def homogeneous_optical_depth(component: Homogeneous, distance: torch.Tensor) -> torch.Tensor:
    """Relative optical depth of a homogeneous component, along rays of a given length.

    Args:
        component (Homogeneous): Homogeneous density component.
        distance (torch.Tensor): Ray lengths in meters, can be infinite.

    Returns:
        torch.Tensor: Relative optical depth of each ray.
    """
    return component.density * distance


def height_fog_optical_depth(
    component: HeightFog, origin: torch.Tensor, directions: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """Relative optical depth of an exponential height fog, along rays of a given length.

    With ``k`` the relative density at the origin, ``H`` the falloff and ``v_z`` the vertical component of the
    ray direction, this is ``k * H / v_z * (1 - exp(-distance * v_z / H))``, which tends to ``k * distance``
    for horizontal rays.

    Args:
        component (HeightFog): Height fog component.
        origin (torch.Tensor): Ray origin, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (..., 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (...), can be infinite.

    Returns:
        torch.Tensor: Relative optical depth of each ray.
    """
    k = component.density * torch.exp(-(origin[2] - component.base_height) / component.falloff)
    dir_z = directions[..., 2]
    x = torch.where(dir_z == 0, torch.zeros_like(distance), distance * dir_z / component.falloff)
    small = x.abs() < 1e-4
    safe_z = torch.where(small, torch.ones_like(dir_z), dir_z)
    safe_x = torch.where(small, torch.ones_like(x), x)
    return k * torch.where(small, distance * (1 - x / 2 + x * x / 6), component.falloff / safe_z * -torch.expm1(-safe_x))


def blob_optical_depth(
    component: Blob,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    time: float = 0.0,
) -> torch.Tensor:
    """Relative optical depth of a Gaussian blob, along rays of a given length.

    Along a ray, the blob's density is a 1D Gaussian centered at the point of closest approach ``t*``, so its
    integral is a difference of error functions: ``k * s * sqrt(pi / 2) * (erf((d - t*) / (s√2)) + erf(t* / (s√2)))``
    where ``k`` is the relative density at the point of closest approach and ``s`` the blob's radius.

    Args:
        component (Blob): Blob component.
        origin (torch.Tensor): Ray origin, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (..., 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (...), can be infinite.
        time (float, optional): Time in seconds, used to move the blob along its velocity. Defaults to 0.0.

    Returns:
        torch.Tensor: Relative optical depth of each ray.
    """
    center = origin.new_tensor(component.center) + time * origin.new_tensor(component.velocity)
    offset = center - origin
    t_star = (directions * offset).sum(dim=-1)
    closest_sq = ((offset * offset).sum() - t_star * t_star).clamp_min(0)
    s = component.radius
    k = component.density * torch.exp(-closest_sq / (2 * s * s))
    return (
        k
        * s
        * math.sqrt(math.pi / 2)
        * (torch.erf((distance - t_star) / (s * math.sqrt(2))) + torch.erf(t_star / (s * math.sqrt(2))))
    )


def component_optical_depth(
    component: Component,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    time: float = 0.0,
) -> torch.Tensor:
    """Relative optical depth of any density component, along rays of a given length.

    Args:
        component (Component): Density component.
        origin (torch.Tensor): Ray origin, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (..., 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (...), can be infinite.
        time (float, optional): Time in seconds, used by moving components. Defaults to 0.0.

    Raises:
        TypeError: raised if the component type is not supported.

    Returns:
        torch.Tensor: Relative optical depth of each ray.
    """
    if isinstance(component, Homogeneous):
        return homogeneous_optical_depth(component, distance)
    if isinstance(component, HeightFog):
        return height_fog_optical_depth(component, origin, directions, distance)
    if isinstance(component, Blob):
        return blob_optical_depth(component, origin, directions, distance, time=time)
    raise TypeError(f"Unsupported medium component {type(component).__name__}.")


def optical_depth(
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    time: float = 0.0,
) -> torch.Tensor:
    """Relative optical depth of a medium, i.e. the sum of the optical depths of its components.

    Args:
        medium (Medium): Participating medium.
        origin (torch.Tensor): Ray origin, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (..., 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (...), can be infinite.
        time (float, optional): Time in seconds, used by moving components. Defaults to 0.0.

    Returns:
        torch.Tensor: Relative optical depth of each ray.
    """
    total = torch.zeros_like(distance)
    for component in medium.components:
        if component.density > 0:
            total = total + component_optical_depth(component, origin, directions, distance, time=time)
    return total


def _hg_cdf(cos_theta: torch.Tensor, g: float) -> torch.Tensor:
    """Fraction of the Henyey-Greenstein phase function's energy below a given cosine of the scattering angle."""
    if abs(g) < 1e-6:
        return (cos_theta + 1) / 2
    return (1 - g * g) / (2 * g) * ((1 + g * g - 2 * g * cos_theta).rsqrt() - 1 / (1 + g))


def _hg_inverse_cdf(xi: torch.Tensor, g: float) -> torch.Tensor:
    """Cosine of the scattering angle at which the Henyey-Greenstein phase function's CDF reaches ``xi``."""
    if abs(g) < 1e-6:
        return 2 * xi - 1
    return ((1 + g * g - ((1 - g * g) / (1 - g + 2 * g * xi)) ** 2) / (2 * g)).clamp(-1, 1)


def sky_quadrature(
    directions: torch.Tensor, g: float, n_angle: int = 16, n_azimuth: int = 24
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadrature of the phase function over the sky, i.e. the directions above the horizon.

    Returns nodes and weights such that ``sum(weights * f(nodes))`` approximates ``∫ p(ω·v) f(ω_z) dω`` over
    directions ``ω`` towards the sky (``ω_z > 0``), for a view direction ``v`` and any smooth function ``f`` of the
    elevation, such as the attenuation of skylight through an exponential height fog. Light from below the horizon
    is assumed to be blocked by the ground. Directions are parametrized by their angle to ``v``, which is sampled
    through the phase function's inverse CDF, and their azimuth around ``v``, whose range above the horizon is known
    in closed form. The angle's range is split where circles of directions around ``v`` start crossing the horizon,
    so the integrand is smooth on each piece and Gauss-Legendre quadrature converges quickly.

    Args:
        directions (torch.Tensor): Unit view directions ``v``, of shape (..., 3).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        n_angle (int, optional): Number of nodes for each of the three pieces of the angle's range. Defaults to 16.
        n_azimuth (int, optional): Number of azimuthal nodes. Defaults to 24. With these defaults, the relative
            error is below 0.1% even for strongly forward scattering media (g = 0.95) seen near the horizon, or for
            skylight attenuated by an optical depth of up to 5 (vertically) before being scattered.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The elevation ``ω_z`` of each node and its weight, both of shape
        (..., 3 * n_angle * n_azimuth). The weights sum to the fraction of the phase function above the horizon.
    """
    kwargs = {"dtype": directions.dtype, "device": directions.device}
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_angle))
    s, b = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_azimuth))

    # Cluster nodes near the ends of each interval, where the integrand has square-root singularities (when
    # circles of directions start crossing the horizon) or boundary layers (due to the attenuation near the horizon)
    t, w = (t * (3 - t * t) / 2 + 1) / 2, w * 3 * (1 - t * t) / 4
    s, b = s * (3 - s * s) / 2, b * 3 * (1 - s * s) / 2

    # Circles of directions at an angle γ from v are entirely above or below the horizon when |cos(γ)| > ρ
    v_z = directions[..., 2:3].clamp(-1, 1)
    rho = (1 - v_z * v_z).clamp_min(0).sqrt()
    bounds = [torch.zeros_like(v_z), _hg_cdf(-rho, g), _hg_cdf(rho, g), torch.ones_like(v_z)]
    xi = torch.cat([lo + (hi - lo) * t for lo, hi in zip(bounds[:-1], bounds[1:])], dim=-1)
    xi_weights = torch.cat([(hi - lo) * w for lo, hi in zip(bounds[:-1], bounds[1:])], dim=-1)

    # Directions at an angle γ from v, whose azimuth ψ is measured from the vertical plane containing v, have
    # an elevation of cos(γ) v_z + sin(γ) cos(ψ) ρ, which is positive for |ψ| < arccos(-cos(γ) v_z / (sin(γ) ρ))
    cos_gamma = _hg_inverse_cdf(xi, g)
    sin_gamma = (1 - cos_gamma * cos_gamma).clamp_min(0).sqrt()
    bound = -cos_gamma * v_z / (sin_gamma * rho).clamp_min(1e-12)
    psi_max = torch.arccos(bound.clamp(-1, 1))

    psi = psi_max[..., None] * s
    mu = cos_gamma[..., None] * v_z[..., None] + sin_gamma[..., None] * torch.cos(psi) * rho[..., None]
    weights = xi_weights[..., None] * psi_max[..., None] * b / (2 * math.pi)
    return mu.flatten(-2).clamp_min(0), weights.flatten(-2)


def height_fog_sun_inscatter(
    optical_depth: torch.Tensor,
    origin_extinction: torch.Tensor,
    falloff: float,
    directions_z: torch.Tensor,
    distance: torch.Tensor,
    sun_z: float | torch.Tensor,
) -> torch.Tensor:
    """Integral of ``σ(s) · T(s) · T_sun(s)`` along rays through an exponential height fog.

    This is the single scattering integral for a unit source, where ``T(s)`` is the transmittance between the ray
    origin and the point at distance ``s``, and ``T_sun(s)`` the transmittance between that point and a sun
    at infinity, through the fog. With ``c0`` the optical depth between the origin and the sun, and ``u`` the ratio
    of the fog's density at the end and start of the ray, it evaluates to
    ``sun_z / (sun_z - v_z) * (exp(-c0) - exp(-τ - c0 * u))``, where the removable singularity at
    ``v_z = sun_z`` is avoided by rewriting it as ``τ * (exp(-c0) - exp(-τ - c0 * u)) / (τ - c0 * (1 - u))``.

    Args:
        optical_depth (torch.Tensor): Optical depth ``τ`` of each ray, of shape (..., c).
        origin_extinction (torch.Tensor): Extinction coefficient at the ray origin, per channel, of shape (c,).
        falloff (float): Height falloff of the fog, in meters.
        directions_z (torch.Tensor): Vertical component of the unit ray directions, of shape (..., 1).
        distance (torch.Tensor): Ray lengths in meters, of shape (..., 1), can be infinite.
        sun_z (float | torch.Tensor): Vertical component of the unit direction pointing towards the sun, either a
            single value or one per ray, broadcastable to (..., 1). This can be used to integrate over directions,
            e.g. of the sky with :func:`sky_quadrature`.

    Returns:
        torch.Tensor: Single scattering integral, of shape (..., c), to be multiplied by the albedo, phase function
        and sun irradiance.
    """
    sun_z = torch.as_tensor(sun_z, dtype=optical_depth.dtype, device=optical_depth.device)
    above = sun_z > 0
    safe_sun_z = torch.where(above, sun_z, torch.ones_like(sun_z))

    c0 = origin_extinction * falloff / safe_sun_z
    x = torch.where(directions_z == 0, torch.zeros_like(distance), distance * directions_z / falloff)
    u = torch.exp(-x)
    delta = optical_depth + c0 * (u - 1)

    # Rays going down, or horizontal ones, never hit the singularity and delta >= 0
    down = safe_sun_z / (safe_sun_z - directions_z.clamp_max(0)) * torch.exp(-c0) * -torch.expm1(-delta)

    # Rays going up, delta can be negative but delta + c0 >= 0, so both exponentials stay bounded
    up = optical_depth * torch.where(
        delta >= 0, torch.exp(-c0) * _exprel(delta), torch.exp(-(c0 + delta)) * _exprel(-delta)
    )
    result = torch.where(directions_z > 0, up, down)

    # Sunlight from below the horizon would have to travel through an infinite amount of fog
    return torch.where((origin_extinction > 0) & above, result, torch.zeros_like(result))
