import math

import numpy as np
import pytest
import torch

from visionsim.medium import Lighting, Medium, Sun, apply_medium
from visionsim.medium.model import HeightFog, Homogeneous, PointLight
from visionsim.medium.surfaces import surface_attenuation, surface_irradiance, surface_tables

FOG = Medium(extinction=0.08, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True)
LIGHTING = Lighting(sky=(0.25, 0.35, 0.55), suns=[Sun(direction=(0.75, 0.43, 0.5), irradiance=(4.0,))])


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def test_surfaces_without_fog_keep_their_light():
    clear = FOG.model_copy(update={"extinction": 0.0})
    lighting = LIGHTING.model_copy(update={"points": [PointLight(position=(2.0, 3.0, 4.0), power=(500.0,))]})
    rng = np.random.default_rng(0)
    points = T(rng.uniform(-5, 5, size=(50, 3)) * [1, 1, 0.2] + [0, 0, 1])
    normals = T(rng.normal(size=(50, 3)))
    normals = normals / normals.norm(dim=-1, keepdim=True)
    without, through = surface_irradiance(clear, lighting, T([0.0, 0.0, 0.0]), points, normals)
    assert torch.allclose(through, without) and torch.all(without > 0)


def test_glow_of_suns_matches_brute_force():
    # Irradiance of the ground from the sun's light scattered once by the fog, from the tables and by brute force
    medium = FOG.model_copy(update={"anisotropy": 0.5})
    tables = surface_tables(medium, LIGHTING, T([medium.extinction]), 3)
    towards = np.asarray(LIGHTING.suns[0].direction) / np.linalg.norm(LIGHTING.suns[0].direction)
    n_mu, n_phi, n_steps = 200, 200, 400
    mu = (np.arange(n_mu) + 0.5) / n_mu  # upward directions, as the ground faces up
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    m, p = np.meshgrid(mu, phi, indexing="ij")
    omega = np.stack([np.sqrt(1 - m**2) * np.cos(p), np.sqrt(1 - m**2) * np.sin(p), m], -1)
    phase = (1 - 0.25) / (4 * np.pi * (1 + 0.25 - 2 * 0.5 * (omega @ towards)) ** 1.5)
    # Along each direction from the ground, s = H · x / μ, where x is the height in falloffs
    x = (np.arange(n_steps) + 0.5) / n_steps * 40
    c0 = medium.extinction * 2.5  # optical depth above the ground
    sigma = medium.extinction * np.exp(-x)
    depth_camera = c0 * (1 - np.exp(-x))  # optical depth between the ground and the point, divided by μ below
    integral = (
        (
            sigma[None, None] * np.exp(-depth_camera[None, None] / m[..., None]) * np.exp(-c0 * np.exp(-x) / towards[2])
        ).sum(-1)
        * (40 / n_steps)
        * 2.5
        / m
    )
    expected = 4.0 * (phase * integral * m).sum() * (1 / n_mu) * (2 * np.pi / n_phi)
    # Normals pointing up, at the ground
    glow = tables.suns[0][0, -1, 0, 0].item()
    assert glow == pytest.approx(expected, rel=0.01)


def test_lamps_are_attenuated_with_the_reduced_extinction():
    medium = Medium(extinction=0.1, anisotropy=0.7, albedo=0.9, components=[Homogeneous()])
    lighting = Lighting(points=[PointLight(position=(0.0, 0.0, 5.0), power=(1000.0,))])
    points, normals = T([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]]), T([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    clear, through = surface_irradiance(medium, lighting, T([medium.extinction]), points, normals)
    distance = (lighting.points[0].position - points.numpy()) ** 2
    distance = np.sqrt(distance.sum(-1))
    reduced = 1 - 0.9 * 0.7
    assert through[:, 0].numpy() == pytest.approx(clear[:, 0].numpy() * np.exp(-reduced * 0.1 * distance))
    assert clear[0, 0].item() == pytest.approx(1000 / (4 * math.pi) / 25)


def test_surface_attenuation_falls_back_to_the_mean():
    clear = T([[1.0], [1.0], [0.0]])
    through = T([[0.5], [0.7], [0.0]])
    ratio = surface_attenuation(clear, through)
    # Lit surfaces are mostly scaled by their own ratio, unlit ones by the mean ratio
    assert ratio[2, 0].item() == pytest.approx(0.6)
    assert ratio[0, 0].item() == pytest.approx((0.5 + 0.6 * 0.2 * 2 / 3) / (1 + 0.2 * 2 / 3))
    assert torch.equal(surface_attenuation(T(np.zeros((0, 1))), T(np.zeros((0, 1)))), T(np.zeros((0, 1))))


def camera(w=8, h=6):
    """Camera looking horizontally along +y, whose -z axis points forward and y axis up, as in Blender."""
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    pose[:3, 3] = [0.0, 0.0, 1.6]
    return {"w": w, "h": h, "fl_x": 6.0, "fl_y": 6.0, "cx": w / 2, "cy": h / 2}, pose


def test_normals_in_camera_space_and_emission():
    intrinsics, pose = camera()
    rng = np.random.default_rng(1)
    radiance, depth = rng.uniform(0.1, 1, size=(6, 8, 3)), rng.uniform(2, 20, size=(6, 8))
    depth[0, 0] = np.inf  # sees the sky
    world = rng.normal(size=(6, 8, 3))
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    in_camera = world @ pose[:3, :3]  # rows of the inverse rotation
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.3,)})
    result = apply_medium(radiance, depth, intrinsics, pose, FOG, lighting, normals=world)
    same = apply_medium(radiance, depth, intrinsics, pose, FOG, lighting, normals=in_camera, normals_space="camera")
    assert torch.allclose(result.illumination, same.illumination)
    assert torch.all(result.illumination[0, 0] == 1) and not torch.allclose(
        result.illumination, torch.ones_like(result.illumination)
    )

    # Surfaces that only emit light keep it
    emitting = apply_medium(radiance, depth, intrinsics, pose, FOG, lighting, normals=world, emission=radiance)
    plain = apply_medium(radiance, depth, intrinsics, pose, FOG, lighting)
    assert torch.allclose(emitting.radiance, plain.radiance) and plain.illumination is None
