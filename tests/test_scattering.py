import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ellipe

from visionsim.medium import Lighting, Medium, Sun, apply_medium, camera_rays
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
