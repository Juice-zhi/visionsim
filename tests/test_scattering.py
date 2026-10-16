import itertools
import math
from functools import partial

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ellipe

from visionsim.medium import Lighting, Medium, Sun, apply_medium, camera_rays, scattering
from visionsim.medium.lights import AREA_LEVELS, area_radius, area_weights, closest_distances, emitters, spot_falloff
from visionsim.medium.model import AnimatedLighting, AreaLight, Blob, HeightFog, Homogeneous, PointLight, SpotLight
from visionsim.medium.optics import (
    density,
    height_fog_sun_inscatter,
    henyey_greenstein,
    optical_depth,
    point_light_inscatter,
    sky_quadrature,
)
from visionsim.medium.render import lamp_inscatter
from visionsim.medium.scattering import (
    ElevationTable,
    _gather_nodes,
    azimuthal_harmonics,
    azimuthal_phase,
    cached_camera_tables,
    cached_multiple_scattering,
    camera_tables,
    complete_elliptic_e,
    direct_ground_radiance,
    ground_irradiance,
    ground_source,
    lookup,
    multiple_scattering,
    relative_azimuths,
    sky_source,
    tabulate_along_rays,
)
from visionsim.medium.surfaces import cached_surface_tables


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


@pytest.mark.parametrize("g", [-0.3, 0.6, 0.9])
def test_azimuthal_harmonics_match_quadrature(g):
    a, b = T([0.3, 0.0, -0.7, 0.99, 0.05]), T([0.3, 0.5, 0.2, -0.1, 0.0])
    harmonics = azimuthal_harmonics(a, b, g, modes=6)
    assert torch.allclose(harmonics[0], azimuthal_phase(a[:, None], b[None, :], g), rtol=1e-8)
    phi = (np.arange(200_000) + 0.5) / 200_000 * 2 * np.pi
    for i, j in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 3)]:
        x, y = a[i].item(), b[j].item()
        phase = henyey_greenstein(T(x * y + math.sqrt(1 - x * x) * math.sqrt(1 - y * y) * np.cos(phi)), g).numpy()
        for m in range(7):
            expected = (phase * np.cos(m * phi)).mean() * 2 * np.pi
            assert harmonics[m, i, j].item() == pytest.approx(expected, rel=1e-6, abs=1e-10)


def test_azimuthal_tables_match_exact_integration():
    # A source that varies with the azimuth of rays and not along them scatters (1 - T) times itself along rays
    camera, pose = wide_camera()
    origin, directions, scale = camera_rays(camera, pose)
    depth = T(np.random.default_rng(8).uniform(0.5, 60, size=(15, 21)))
    depth[:4] = math.inf
    distance = torch.where(torch.isinf(depth), depth, depth * scale)
    fog = FOG.components[0]
    tau = optical_depth(FOG, origin, directions, distance)[..., None] * FOG.extinction
    towards = T([0.6, 0.8])

    def source(elevations, azimuths, depth_above):
        return (2 + torch.cos(azimuths) + elevations**2)[:, None, None].expand(*depth_above.shape[:2], 2)

    depth_above = T([FOG.extinction * fog.density * fog.falloff * math.exp(-origin[2].item() / fog.falloff)])
    table = tabulate_along_rays(source, depth_above, azimuths=65, towards=towards)
    assert table.values.shape == (256, 65, 257, 2)
    azimuths = relative_azimuths(directions, table.towards)
    result = lookup(table, directions[..., 2], tau, azimuths)
    expected = (2 + torch.cos(azimuths) + directions[..., 2] ** 2)[..., None] * -torch.expm1(-tau)
    assert torch.allclose(result, expected.expand_as(result), rtol=2e-3, atol=1e-9)
    with pytest.raises(ValueError, match="azimuth"):
        lookup(table, directions[..., 2], tau)


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


@pytest.mark.parametrize("extinction,g,rtol", [(0.08, 0.8, 1e-3), (0.4, 0.8, 1e-3), (0.08, 0.0, 1e-3), (4.0, 0.8, 3e-3)])
def test_multiple_scattering_conserves_energy(extinction, g, rtol):
    """Without absorption, the light entering the fog from above either leaves it at the top or is absorbed by the
    ground, which is lit by suns and the sky through the fog, and by the light that the fog scatters."""
    medium = FOG.model_copy(update={"extinction": extinction, "anisotropy": g})
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.4,)})
    table = multiple_scattering(medium, lighting, T([extinction]), channels=3)
    omega, weights = _gather_nodes(48, dtype=torch.float64)
    # Light leaving the top of the fog, the last height: scattered at least once, and reflected by the ground then
    # attenuated by the whole fog
    down = omega < 0
    unscattered = table.ground * torch.exp(-table.depth_ground / omega[down].abs())[:, None]
    leaving = 2 * math.pi * ((weights * omega.abs())[down][:, None] * (table.gathered[-1, down] + unscattered)).sum(0)
    absorbed = (1 - 0.4) * math.pi * table.ground / 0.4
    sun = np.asarray(LIGHTING.suns[0].direction) / np.linalg.norm(LIGHTING.suns[0].direction)
    entering = 4.0 * sun[2] + math.pi * T(LIGHTING.sky)
    assert torch.allclose(leaving + absorbed, entering, rtol=rtol)
    # The fog's glow makes up for much of the light it takes out of sunlight and skylight on their way to the ground
    direct = direct_ground_radiance(medium, lighting, T([extinction]), 3)
    assert torch.all(table.ground > 1.2 * direct)


def test_light_of_the_ground_scattered_once():
    """The light of the ground scattered once is part of the light gathered from below, and none without a ground that
    reflects light."""
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.4,)})
    table = multiple_scattering(FOG, lighting, T([0.08]), channels=3)
    down = table.directions < 0
    assert table.from_ground is not None and table.from_ground.shape == table.gathered.shape
    assert torch.all(table.from_ground[1:, down] > 0) and torch.all(table.from_ground <= table.gathered)
    dark = multiple_scattering(FOG, LIGHTING, T([0.08]), channels=3)
    assert dark.from_ground is not None and torch.all(dark.from_ground == 0)


def test_multiple_scattering_follows_suns():
    """Light scattered more than once is brighter towards suns than away from them, and its variations with the
    azimuth of rays from each sun average out over azimuths."""
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.4,)})
    table = multiple_scattering(FOG, lighting, T([0.08]), channels=3, n_azimuths=65)
    (sun,) = table.suns
    expected = np.asarray(LIGHTING.suns[0].direction[:2]) / np.linalg.norm(LIGHTING.suns[0].direction[:2])
    assert np.allclose(sun.towards.numpy(), expected)
    # Trapezoidal mean over [0, π]
    weights = torch.full((65,), 1 / 64, dtype=torch.float64)
    weights[[0, -1]] /= 2
    mean = torch.einsum("eahc,a->ehc", sun.values, weights)
    assert mean.abs().max() < 1e-3 * table.values.abs().max()
    towards, away = table.values + sun.values[:, 0], table.values + sun.values[:, -1]
    horizon = slice(100, 156)  # elevations within about 10° of the horizon
    assert torch.all(towards[horizon, :20] > 1.5 * away[horizon, :20]) and torch.all(away > -1e-9)


def test_light_of_suns_scattered_once_arrives_mostly_from_them():
    """The harmonics of the sun's light scattered once that reaches each height from each gathering direction, over its
    mean over azimuths, are those of the phase function between the sun and the direction: summed back at any azimuth,
    they give the phase function, so that this light arrives mostly from directions close to the sun's."""
    lighting = Lighting(suns=LIGHTING.suns)  # neither the sky nor the ground lights the fog
    # Enough harmonics that the phase function's, which decrease about as g^m, are exact
    table = multiple_scattering(FOG, lighting, T([0.08]), channels=1, higher_orders=False, modes=48)
    (sun,) = table.suns
    assert sun.once is not None and sun.once.shape == sun.arriving.shape and sun.arriving.abs().max() == 0
    towards = np.asarray(LIGHTING.suns[0].direction) / np.linalg.norm(LIGHTING.suns[0].direction)
    phi = torch.linspace(0, math.pi, 7, dtype=torch.float64)
    modes = torch.arange(1, len(sun.once) + 1, dtype=torch.float64)
    for h, d in itertools.product((3, 20), range(0, len(table.directions), 7)):
        mean = table.gathered[h, d, 0]
        if mean <= 0:
            continue
        light = mean + (sun.once[:, h, d, 0, None] * torch.cos(modes[:, None] * phi)).sum(dim=0)
        mu = float(table.directions[d])
        cos = mu * towards[2] + math.sqrt(1 - mu**2) * math.hypot(*towards[:2]) * torch.cos(phi)
        phase = henyey_greenstein(cos, FOG.anisotropy)
        # Over the mean over azimuths of the phase function
        expected = phase / (azimuthal_phase(T(mu), T(towards[2]), FOG.anisotropy) / (2 * math.pi))
        assert torch.allclose(light / mean, expected, rtol=1e-4), (h, d)


def test_multiple_scattering_of_several_suns_adds_up():
    camera, pose = wide_camera()
    depth = np.random.default_rng(9).uniform(0.5, 60, size=(15, 21))
    depth[:3] = np.inf
    medium = FOG.model_copy(update={"multiple_scattering": True})
    suns = [Sun(direction=(0.4, 0.7, 0.58), irradiance=(4.0,)), Sun(direction=(-0.8, 0.1, 0.3), irradiance=(2.0,))]
    lightings = [Lighting(suns=s, ground_albedo=(0.4,)) for s in (suns, suns[:1], suns[1:])]
    both, *each = (apply_medium(np.zeros((15, 21, 3)), depth, camera, pose, medium, x).inscatter for x in lightings)
    assert torch.allclose(both, each[0] + each[1], rtol=1e-6)


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


def test_spot_light_matches_quadrature():
    medium = Medium(extinction=0.1, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)])
    origin, position = T([0.0, 0.0, 1.5]), T([0.8, 7.0, 2.5])
    spot = SpotLight(position=(0.8, 7.0, 2.5), direction=(1.0, -0.2, -0.3), power=(1.0,), angle=0.9, blend=0.4)
    (emitter,) = emitters(spot, 1, dtype=torch.float64)
    rng = np.random.default_rng(13)
    directions = rng.normal(size=(16, 3)) + [0, 3, 0]
    directions = T(directions / np.linalg.norm(directions, axis=-1, keepdims=True))
    distance = T(rng.uniform(3, 30, size=16))
    beta = T([medium.extinction])
    result = point_light_inscatter(
        medium, origin, directions, distance, position, beta, nodes=64, axis=emitter.axis, cone=emitter.cone,
        profile=emitter.profile,
    )  # fmt: skip

    def integrand(s, v):
        point = origin + s * v
        to_light = position - point
        r = to_light.norm()
        tau = optical_depth(medium, origin, v, T(s)) + optical_depth(medium, point, to_light / r, r)
        sigma = density(medium, point[None])[0] * beta[0]
        phase = henyey_greenstein((v @ to_light) / r, medium.anisotropy) * emitter.profile(
            -(to_light @ emitter.axis) / r
        )
        return (sigma * torch.exp(-tau * beta[0]) * phase / r**2).item()

    lit = 0
    for v, d, value in zip(directions, distance, result[:, 0]):
        along = float(v @ (position - origin))
        expected = quad(
            integrand, 0, float(d), args=(v,), points=[min(max(along, 0), float(d))], limit=500, epsrel=1e-10
        )
        assert value.item() == pytest.approx(expected[0], rel=1e-4, abs=1e-12)
        lit += expected[0] > 0
    assert 0 < lit < len(directions)  # some rays cross the cone, others don't


def test_spot_light_falls_off_as_in_cycles():
    cos_half, blend = math.cos(0.4), 0.3
    edge = cos_half + (1 - cos_half) * blend
    falloff = spot_falloff(
        T([cos_half - 0.01, cos_half, (cos_half + edge) / 2, edge, 1.0]), cos_half, 1 / (edge - cos_half)
    )
    assert falloff.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    # Hard edges
    assert spot_falloff(T([cos_half - 1e-9, cos_half + 1e-9]), cos_half, math.inf).tolist() == [0.0, 1.0]


def test_point_light_is_two_hemispherical_spots():
    rng = np.random.default_rng(2)
    origin = T([0.0, 0.0, 1.5])
    directions = rng.normal(size=(200, 3)) + [0, 3, 0]
    directions = T(directions / np.linalg.norm(directions, axis=-1, keepdims=True))
    distance = T(np.where(rng.uniform(size=200) < 0.2, np.inf, rng.uniform(2, 40, size=200)))
    beta = T([FOG.extinction])
    point = PointLight(position=(0.8, 7.0, 2.5), power=(1.0,), radius=0.05)
    whole = lamp_inscatter(FOG, origin, directions, distance, beta, point)
    halves = sum(
        lamp_inscatter(
            FOG,
            origin,
            directions,
            distance,
            beta,
            SpotLight(position=point.position, direction=d, power=(1.0,), angle=math.pi, blend=0.0, radius=0.05),
        )
        for d in [(0.3, -0.5, 0.8), (-0.3, 0.5, -0.8)]
    )
    assert torch.allclose(halves, whole, rtol=1e-3)


@pytest.mark.parametrize(
    "lamp",
    [
        AreaLight(position=(0, 0, 0), direction=(0, 0.3, -1), axis_u=(1, 0, 0), size=(1.0, 0.5), power=(5.0,)),
        AreaLight(position=(0, 0, 0), direction=(0, 0, 1), axis_u=(1, 1, 0), size=(0.5, 0.5), power=(5.0,), spread=1),
        AreaLight(position=(0, 0, 0), direction=(1, 0, 0), axis_u=(0, 1, 0), size=(1, 2), power=(5.0,), shape="ellipse"),
        SpotLight(position=(0, 0, 0), direction=(0, 0, -1), power=(5.0,), angle=math.pi, blend=0.0),
    ],
)
def test_lamps_emit_their_power(lamp):
    n = 600
    theta = (torch.arange(n, dtype=torch.float64) + 0.5) / n * math.pi
    phi = (torch.arange(2 * n, dtype=torch.float64) + 0.5) / n * math.pi
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    omega = torch.stack([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()], -1)
    solid_angle = theta.sin() * (math.pi / n) ** 2
    total = 0.0
    for emitter in emitters(lamp, 1, samples=3, dtype=torch.float64):
        cosine = omega @ emitter.axis
        relative = emitter.profile(cosine) * (cosine >= emitter.cone)
        total += emitter.intensity.item() * (relative * solid_angle).sum().item()
    # Spot lights shine with the intensity of a point light of the same power, i.e. half of it over a half-space
    expected = lamp.power[0] / 2 if isinstance(lamp, SpotLight) else lamp.power[0]
    assert total == pytest.approx(expected, rel=1e-3)


def test_area_weights():
    lamp = AreaLight(position=(0, 0, 0), direction=(0, 0, -1), axis_u=(1, 0, 0), size=(1.2, 0.5), power=(1.0,))
    closest = T(np.linspace(0, 20, 2001)) * area_radius(lamp)
    weights = torch.stack(area_weights(lamp, closest))
    assert torch.allclose(weights.sum(dim=0), torch.ones_like(closest)) and torch.all(weights >= 0)
    # The finest grid up to its near distance, and a single emitter beyond the far distance of the coarsest grid
    assert torch.all(weights[0][closest <= AREA_LEVELS[0].near * area_radius(lamp)] == 1)
    assert torch.all(weights[-1][closest >= AREA_LEVELS[-1].far * area_radius(lamp)] == 1)
    # Weights change smoothly, and only between successive grids
    # Smoothsteps change by at most 1.5 times the step over the width of a transition
    steepest = 1.5 * 0.01 / min(level.far - level.near for level in AREA_LEVELS)
    assert (weights.diff(dim=1).abs().max() <= steepest + 1e-9) and torch.all((weights > 0).sum(dim=0) <= 2)


def test_closest_distances():
    origin, position = T([0.0, 0.0, 0.0]), T([1.0, 5.0, 0.0])
    directions = T([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    # Rays pass by the point, away from it, or end before reaching it
    distances = closest_distances(origin, directions, T([10.0, 10.0, 2.0]), position)
    assert distances.tolist() == pytest.approx([1.0, math.hypot(1, 5), math.hypot(1, 3)])


@pytest.mark.parametrize("shape,spread", [("rectangle", math.pi), ("ellipse", math.pi), ("rectangle", 2.3)])
def test_area_light_grids_match_a_fine_grid(shape, spread):
    lamp = AreaLight(
        position=(1.0, 6.0, 2.8), direction=(0.2, -0.3, -1), axis_u=(1, 0, 0), size=(1.2, 0.6), power=(1.0,),
        shape=shape, spread=spread,
    )  # fmt: skip
    rng = np.random.default_rng(4)
    origin, center = T([0.0, 0.0, 1.5]), T(lamp.position)
    # Rays towards points around the light, which mostly pass within a few radii of it
    targets = lamp.position + rng.normal(size=(300, 3)) * rng.uniform(size=(300, 1)) ** 2 * 10 * area_radius(lamp)
    directions = T(targets) - origin
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T(np.where(rng.uniform(size=300) < 0.3, np.inf, 2 * np.linalg.norm(targets - origin.numpy(), axis=-1)))
    beta = T([FOG.extinction])
    ours = lamp_inscatter(FOG, origin, directions, distance, beta, lamp)[:, 0]
    reference = sum(
        e.intensity
        * point_light_inscatter(
            FOG, origin, directions, distance, e.position, beta, e.radius, axis=e.axis, cone=e.cone, profile=e.profile
        )
        for e in emitters(lamp, 1, 32, dtype=torch.float64)
    )[:, 0]
    # Rays that pass right next to the light are the hardest, though they are a small part of images
    error = (ours - reference).abs()
    beyond = closest_distances(origin, directions, distance, center) > area_radius(lamp)
    assert error[beyond].sum() / reference[beyond].sum() < 0.01
    assert error.sum() / reference.sum() < 0.05


@pytest.mark.parametrize("attenuation", [True, False])
def test_ray_marching_converges_with_point_lights(attenuation):
    from visionsim.medium.raymarch import ray_march_medium

    camera, pose = wide_camera(w=9, h=7)
    rng = np.random.default_rng(5)
    radiance, depth = rng.uniform(0, 1, size=(7, 9, 3)), rng.uniform(2, 30, size=(7, 9))
    medium = FOG if attenuation else Medium(extinction=0.06, anisotropy=0.5, components=[Homogeneous()])
    lighting = Lighting(
        points=[PointLight(position=(3.0, 9.0, 6.0), power=(2000.0, 1500.0, 1000.0), radius=0.1)],
        spots=[SpotLight(position=(-2.0, 12.0, 4.0), direction=(0.3, -1.0, -0.6), power=(3000.0,), angle=1.0)],
    )
    exact = apply_medium(radiance, depth, camera, pose, medium, lighting).radiance
    errors = [
        (
            (ray_march_medium(radiance, depth, camera, pose, medium, lighting, steps=s).radiance - exact).abs().sum()
            / exact.abs().sum()
        ).item()
        for s in (16, 64, 256)
    ]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-4


def test_ray_marching_agrees_with_area_lights():
    from visionsim.medium.raymarch import ray_march_medium

    camera, pose = wide_camera(w=9, h=7)
    rng = np.random.default_rng(6)
    radiance, depth = np.zeros((7, 9, 3)), rng.uniform(2, 30, size=(7, 9))
    lighting = Lighting(
        areas=[AreaLight(position=(1.0, 8.0, 4.0), direction=(0, -0.5, -1), axis_u=(1, 0, 0), size=(2, 1), power=(5e3,))]
    )
    exact = apply_medium(radiance, depth, camera, pose, FOG, lighting).inscatter
    marched = ray_march_medium(radiance, depth, camera, pose, FOG, lighting, steps=512, area_samples=16).inscatter
    assert ((marched - exact).abs().sum() / exact.sum()).item() < 0.01


def test_lamps_round_trip():
    lighting = Lighting(
        points=[PointLight(position=(0, 0, 3), power=(10.0,))],
        spots=[SpotLight(position=(1, 0, 3), direction=(0, 0, -1), power=(20.0,), angle=0.5, blend=0.1)],
        areas=[AreaLight(position=(0, 1, 3), direction=(0, 0, -1), axis_u=(1, 0, 0), size=(1, 2), power=(30.0,))],
    )
    assert Lighting.model_validate_json(lighting.model_dump_json()) == lighting
    assert [type(lamp) for lamp in lighting.lamps] == [PointLight, SpotLight, AreaLight]
    assert lighting.areas[0].area == 2 and lighting.areas[0].model_copy(update={"shape": "ellipse"}).area == math.pi / 2
    with pytest.raises(ValueError):
        SpotLight(position=(0, 0, 0), direction=(0, 0, -1), power=(1.0,), angle=4.0)


def test_animated_lighting():
    # The lighting of a frame is that of the last frame at or before it at which it changed, or of the first one
    first, later = Lighting(sky=(1.0,)), Lighting(sky=(2.0,))
    animated = AnimatedLighting(frames={10: first, 20: later})
    assert [animated.at(frame).sky for frame in (5, 10, 15, 20, 25)] == [(1.0,)] * 3 + [(2.0,)] * 2
    assert AnimatedLighting.model_validate_json(animated.model_dump_json()) == animated
    with pytest.raises(ValueError, match="at least one frame"):
        AnimatedLighting(frames={})


def test_multiple_scattering_requires_sun_attenuation():
    with pytest.raises(ValueError, match="sun attenuation"):
        Medium(extinction=0.1, multiple_scattering=True)


def test_cached_tables_ignore_lamps():
    """Light reaching the medium from many directions doesn't depend on lamps, so that lamps that move or change
    don't recompute it."""
    lamp = PointLight(position=(0.0, 3.0, 1.0), power=(100.0,))
    lit = LIGHTING.model_copy(update={"ground_albedo": (0.4,), "points": [lamp]})
    plain = LIGHTING.model_copy(update={"ground_albedo": (0.4,)})
    assert cached_multiple_scattering(FOG, lit, T([0.08]), 3) is cached_multiple_scattering(FOG, plain, T([0.08]), 3)
    assert cached_surface_tables(FOG, lit, T([0.08]), 3) is cached_surface_tables(FOG, plain, T([0.08]), 3)


def test_camera_tables_are_kept_for_each_height():
    """Tables along camera rays only depend on the height of the camera, and are computed once for each."""
    medium = FOG.model_copy(update={"multiple_scattering": True})
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.4,)})
    fog = medium.components[0]
    beta = T([medium.extinction])

    def above(height):
        return beta * fog.density * fog.falloff * math.exp(-(height - fog.base_height) / fog.falloff)

    tables = cached_camera_tables(medium, lighting, beta, 3, above(1.6), (32, 33))
    assert list(tables) == ["sky", "ground", "scattered", "scattered 0"]
    again = cached_camera_tables(medium, lighting, beta, 3, above(1.6), (32, 33))
    assert all(again[name] is table for name, table in tables.items())
    fresh = camera_tables(medium, lighting, beta, 3, above(1.6), (32, 33))
    assert all(torch.equal(fresh[name].values, table.values) for name, table in tables.items())
    higher = cached_camera_tables(medium, lighting, beta, 3, above(2.0), (32, 33))
    assert not torch.equal(higher["sky"].values, tables["sky"].values)
    # Summed, as when nothing occludes them
    combined = cached_camera_tables(medium, lighting, beta, 3, above(1.6), (32, 33), combined=True)
    assert list(combined) == ["all", "scattered 0"]
    total = tables["sky"].values + tables["ground"].values + tables["scattered"].values
    assert torch.allclose(combined["all"].values, total, rtol=1e-12, atol=1e-15)
    # Without suns or skylight, nothing reaches the medium from many directions
    night = Lighting(points=[PointLight(position=(0.0, 3.0, 1.0), power=(100.0,))], ground_albedo=(0.4,))
    assert cached_camera_tables(medium, night, beta, 3, above(1.6), (32, 33)) == {}


@pytest.mark.parametrize("at_once", [False, True])
@pytest.mark.parametrize("azimuth_axis", [False, True])
def test_lookup_of_each_corner(azimuth_axis, at_once, monkeypatch):
    """The corners of the table around each ray are interpolated at once, as on a GPU, as each of them in turn."""
    monkeypatch.setattr(scattering, "_interpolate_rows", partial(scattering._interpolate_rows, at_once=at_once))
    rng = np.random.default_rng(4)
    shape = (16, 9, 17, 2) if azimuth_axis else (16, 17, 2)
    table = ElevationTable(T(rng.uniform(size=shape)), T([0.3]), T([0.6, 0.8]) if azimuth_axis else None)
    directions_z = T(rng.uniform(-1, 1, size=(50, 3)))
    depth = T(rng.uniform(0, 2, size=(50, 3, 1)))
    azimuths = T(rng.uniform(0, math.pi, size=(50, 3))) if azimuth_axis else None

    # Interpolation between the two (or four) rows of the table around each ray, one after another
    n_elevation, *n_azimuth, n_depth, channels = shape
    v = directions_z[..., None]
    y_max = torch.where(v > 0, -torch.expm1(-table.depth_above / torch.where(v > 0, v, 1.0)), 1.0)
    x = (-torch.expm1(-depth) / y_max).clamp(0, 1).expand(50, 3, channels)
    fi = (torch.asinh(directions_z * math.sinh(3.0)) / 3.0 + 1) / 2 * (n_elevation - 1)
    i = fi.floor().clamp(0, n_elevation - 2).long()
    wi = (fi - i)[..., None]
    j = (x * (n_depth - 1)).floor().clamp(0, n_depth - 2).long()
    wj = x * (n_depth - 1) - j
    values = table.values.reshape(-1, n_depth, channels)
    rows, weights = [i, i + 1], [1 - wi, wi]
    if azimuths is not None:
        fa = azimuths / math.pi * (n_azimuth[0] - 1)
        a = fa.floor().clamp(0, n_azimuth[0] - 2).long()
        wa = (fa - a)[..., None]
        rows = [row * n_azimuth[0] + a + da for row in rows for da in (0, 1)]
        weights = [weight * w for weight in weights for w in (1 - wa, wa)]
    ch = torch.arange(channels)
    expected = torch.zeros_like(wj)
    for row, weight in zip(rows, weights):
        near = values[row[..., None], j, ch]
        expected = expected + weight * (near + wj * (values[row[..., None], j + 1, ch] - near))
    assert torch.equal(lookup(table, directions_z, depth, azimuths), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rows_gathered_at_once_on_a_gpu(dtype):
    """On a GPU, where the rows of a table around each ray are gathered at once, they add up as each of them in turn."""
    rng = np.random.default_rng(7)
    kwargs = {"dtype": dtype, "device": "cuda"}
    values = torch.as_tensor(rng.uniform(size=(300, 65, 3)), **kwargs)
    n = 100_000
    for k in (2, 4):
        rows = [torch.as_tensor(rng.integers(0, 300, size=n), device="cuda") for _ in range(k)]
        weights = [torch.as_tensor(rng.uniform(size=n), **kwargs) for _ in range(k)]
        index = torch.as_tensor(rng.integers(0, 64, size=(n, 3)), device="cuda")
        fraction = torch.as_tensor(rng.uniform(size=(n, 3)), **kwargs)
        args = (values, rows, weights, index, fraction)
        assert torch.equal(scattering._interpolate_rows(*args), scattering._interpolate_rows(*args, at_once=False))
