import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ellipe

from visionsim.medium import Lighting, Medium, Sun, apply_medium, camera_rays
from visionsim.medium.model import Blob, HeightFog, Homogeneous, PointLight
from visionsim.medium.optics import (
    density,
    height_fog_sun_inscatter,
    henyey_greenstein,
    optical_depth,
    point_light_inscatter,
    sky_quadrature,
)
from visionsim.medium.scattering import (
    azimuthal_phase,
    complete_elliptic_e,
    ground_irradiance,
    ground_source,
    multiple_scattering,
    sky_source,
)


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


FOG = Medium(extinction=0.08, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True)
LIGHTING = Lighting(sky=(0.25, 0.35, 0.55), suns=[Sun(direction=(0.4, 0.7, 0.58), irradiance=(4.0,))])


def wide_camera(w=21, h=15, height=1.6):
    """Camera with a wide field of view looking horizontally along +Y, which sees both the sky and the ground."""
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    pose[:3, 3] = [0.0, 0.0, height]
    return {"w": w, "h": h, "fl_x": 4.0, "fl_y": 4.0, "cx": w / 2, "cy": h / 2}, pose


def test_complete_elliptic_e():
    m = np.array([0.0, 1e-6, 0.1, 0.5, 0.9, 0.999, 1 - 1e-8])
    assert np.allclose(complete_elliptic_e(T(m)).numpy(), ellipe(m), rtol=1e-10, atol=0)


@pytest.mark.parametrize("g", [-0.3, 0.0, 0.6, 0.9])
def test_azimuthal_phase_matches_quadrature(g):
    phi = (np.arange(200_000) + 0.5) / 200_000 * 2 * np.pi
    for a, b in [(0.3, 0.3), (0.0, 0.5), (-0.7, 0.2), (0.99, -0.1), (1.0, 0.4), (0.5, 0.500001), (-0.2, -0.2)]:
        cos = a * b + math.sqrt(1 - a * a) * math.sqrt(1 - b * b) * np.cos(phi)
        expected = henyey_greenstein(T(cos), g).mean().item() * 2 * np.pi
        assert azimuthal_phase(T(a), T(b), g).item() == pytest.approx(expected, rel=1e-8)


def brute_hemisphere(v, g, attenuation, below, n_theta=400, n_phi=800):
    """Brute-force integral of the phase function times an attenuation that depends on |ω_z|, over the upper or lower
    hemisphere of directions ω."""
    theta = (np.arange(n_theta) + 0.5) / n_theta * np.pi / 2
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    z = -np.cos(th) if below else np.cos(th)
    omega = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), z], -1).reshape(-1, 3)
    weights = (np.sin(th) * (np.pi / 2 / n_theta) * (2 * np.pi / n_phi)).reshape(-1)
    return (weights * henyey_greenstein(T(omega @ v), g).numpy() * attenuation(np.abs(omega[:, 2]))).sum()


@pytest.mark.parametrize("g", [0.0, 0.6, 0.85])
def test_sky_and_ground_sources_match_brute_force(g):
    c_ground = 0.3
    for v_z in (-0.9, -0.2, 0.0, 0.05, 0.4, 1.0):
        v = np.array([math.sqrt(1 - v_z * v_z), 0.0, v_z])
        for c in (0.0, 0.02, 0.2, 0.29):
            sky = sky_source(T([v_z]), T([[[c]]]), g).item()
            expected = brute_hemisphere(v, g, lambda mu, c=c: np.exp(-c / mu), below=False)
            assert sky == pytest.approx(expected, rel=2e-3, abs=1e-6)
            ground = ground_source(T([v_z]), T([[[c]]]), T([c_ground]), g).item()
            expected = brute_hemisphere(v, g, lambda mu, c=c: np.exp(-(c_ground - c) / mu), below=True)
            assert ground == pytest.approx(expected, rel=2e-3, abs=1e-6)


def test_ground_irradiance():
    sky, depth = T([0.3]), T([0.25])
    mu = np.linspace(0, 1, 200_001)[1:]
    expected = 2 * np.pi * 0.3 * np.trapezoid(mu * np.exp(-0.25 / mu), mu) + 4.0 * 0.5 * np.exp(-0.25 / 0.5)
    assert ground_irradiance(sky, [(0.5, T([4.0])), (-0.2, T([9.0]))], depth).item() == pytest.approx(expected, 1e-6)


def exact_sky_inscatter(medium, camera, pose, depth):
    """Skylight scattered along each pixel's ray, integrating every direction of the sky's quadrature in closed form."""
    origin, directions, scale = camera_rays(camera, pose)
    distance = torch.where(torch.isinf(T(depth)), T(depth), T(depth) * scale)
    fog = medium.components[0]
    tau = optical_depth(medium, origin, directions, distance)[..., None] * medium.extinction
    extinction = medium.extinction * fog.density * math.exp(-(origin[2].item() - fog.base_height) / fog.falloff)
    mu, weights = sky_quadrature(directions, medium.anisotropy)
    integral = height_fog_sun_inscatter(
        tau[..., None, :],
        T([extinction]),
        fog.falloff,
        directions[..., None, 2:3],
        distance[..., None, None],
        mu[..., None],
    )
    return (weights[..., None] * integral).sum(dim=-2)[..., 0]


@pytest.mark.parametrize("g", [0.0, 0.8, 0.95])
def test_sky_table_matches_exact_integration(g):
    rng = np.random.default_rng(7)
    camera, pose = wide_camera()
    medium = FOG.model_copy(update={"anisotropy": g})
    depth = rng.uniform(0.5, 60, size=(15, 21))
    depth[:4] = np.inf  # the top rows see the sky
    result = apply_medium(np.zeros((15, 21, 1)), depth, camera, pose, medium, Lighting(sky=(1.0,)))
    exact = exact_sky_inscatter(medium, camera, pose, depth).numpy()
    relative = np.abs(result.inscatter[..., 0].numpy() - exact) / exact
    assert relative.max() < 1e-3 and relative.mean() < 2e-4


def test_multiple_scattering_vanishes_without_scattering():
    table = multiple_scattering(FOG.model_copy(update={"albedo": 0.0}), LIGHTING, T([0.08]), channels=3)
    assert torch.all(table.values == 0)


def test_multiple_scattering_adds_light():
    camera, pose = wide_camera()
    depth = np.random.default_rng(3).uniform(0.5, 60, size=(15, 21))
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.4,)})
    single = apply_medium(np.zeros((15, 21, 3)), depth, camera, pose, FOG, lighting).inscatter
    multiple = FOG.model_copy(update={"multiple_scattering": True})
    both = apply_medium(np.zeros((15, 21, 3)), depth, camera, pose, multiple, lighting).inscatter
    assert torch.all(both > single)

    # Light scattered more than once is a larger share of all scattered light in denser fog
    dense = multiple.model_copy(update={"extinction": 0.32})
    ratio = (both / single).mean()
    dense_ratio = (
        apply_medium(np.zeros((15, 21, 3)), depth, camera, pose, dense, lighting).inscatter
        / apply_medium(
            np.zeros((15, 21, 3)), depth, camera, pose, dense.model_copy(update={"multiple_scattering": False}), lighting
        ).inscatter
    ).mean()
    assert dense_ratio > ratio > 1


@pytest.mark.parametrize(
    "medium",
    [
        Medium(extinction=0.1, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True),
        Medium(extinction=0.05, anisotropy=0.3, components=[Homogeneous(density=0.5), Blob(center=(1, 6, 1), radius=2)]),
    ],
    ids=["height", "mixture"],
)
def test_point_light_matches_quadrature(medium):
    origin, position = T([0.0, 0.0, 1.5]), T([0.8, 7.0, 2.5])
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(12, 3)) + [0, 3, 0]
    directions[0] = (position - origin).numpy() + [0, 0, 0.02]  # passes 2cm away from the light
    directions = T(directions / np.linalg.norm(directions, axis=-1, keepdims=True))
    distance = T(rng.uniform(3, 30, size=12))
    beta = T([medium.extinction])
    result = point_light_inscatter(medium, origin, directions, distance, position, beta, radius=0.001, nodes=64)

    def integrand(s, v):
        point = origin + s * v
        to_light = position - point
        r = to_light.norm()
        tau = optical_depth(medium, origin, v, T(s)) + optical_depth(medium, point, to_light / r, r)
        sigma = density(medium, point[None])[0] * beta[0]
        phase = henyey_greenstein((v @ to_light) / r, medium.anisotropy)
        return (sigma * torch.exp(-tau * beta[0]) * phase / r**2).item()

    for v, d, value in zip(directions, distance, result[:, 0]):
        along = float(v @ (position - origin))
        expected = quad(
            integrand, 0, float(d), args=(v,), points=[min(max(along, 0), float(d))], limit=500, epsrel=1e-10
        )
        assert value.item() == pytest.approx(expected[0], rel=1e-4)


@pytest.mark.parametrize("attenuation", [True, False])
def test_ray_marching_converges_with_point_lights(attenuation):
    from visionsim.medium.raymarch import ray_march_medium

    camera, pose = wide_camera(w=9, h=7)
    rng = np.random.default_rng(5)
    radiance, depth = rng.uniform(0, 1, size=(7, 9, 3)), rng.uniform(2, 30, size=(7, 9))
    medium = FOG if attenuation else Medium(extinction=0.06, anisotropy=0.5, components=[Homogeneous()])
    lighting = Lighting(points=[PointLight(position=(3.0, 9.0, 6.0), power=(2000.0, 1500.0, 1000.0), radius=0.1)])
    exact = apply_medium(radiance, depth, camera, pose, medium, lighting).radiance
    errors = [
        (
            (ray_march_medium(radiance, depth, camera, pose, medium, lighting, steps=s).radiance - exact).abs().sum()
            / exact.abs().sum()
        ).item()
        for s in (16, 64, 256)
    ]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-4


def test_multiple_scattering_requires_sun_attenuation():
    with pytest.raises(ValueError, match="sun attenuation"):
        Medium(extinction=0.1, multiple_scattering=True)
