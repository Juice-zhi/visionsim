import math

import numpy as np
import pytest
import torch

from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium.halos import HaloGrid, _around, halo_inscatter, halo_threshold, lamp_halo
from visionsim.medium.model import HeightFog, Homogeneous, PointLight
from visionsim.medium.optics import density, henyey_greenstein, optical_depth, point_light_inscatter

SMALL = HaloGrid(radii=14, extent=(0.05, 30.0), directions=(8, 10), angles=(20, 8), gathered=(16, 16), orders=2)
LAMP = PointLight(position=(0.0, 6.0, 2.0), power=(1000.0,), radius=0.05)


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def test_halo_matches_gathering_along_rays():
    # Light scattered twice towards a few rays, from the table, and gathered directly at many points along the rays
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    halo = lamp_halo(medium, LAMP, beta, ground=-100.0, grid=SMALL)
    origin = T([0.0, 0.0, 1.5])
    directions = T([[0.0, 1.0, 0.05], [0.3, 1.0, 0.1], [-0.4, 1.0, -0.05]])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T([20.0, 20.0, 20.0])
    tabulated = halo_inscatter(halo, medium, origin, directions, distance, beta, nodes=64)[:, 0]

    # Gathering directions concentrated towards the lamp, as for the table, but finer
    t, w = (T(a) for a in np.polynomial.legendre.leggauss(32))
    t, w = (t + 1) / 2, w / 2
    polar, azimuth = torch.meshgrid(
        math.pi * t**2, (torch.arange(32, dtype=torch.float64) + 0.5) / 32 * 2 * math.pi, indexing="ij"
    )
    solid = ((2 * math.pi * t * w)[:, None] * torch.sin(polar) * 2 * math.pi / 32).reshape(-1)
    position = T(LAMP.position)
    s = torch.linspace(0.01, 20, 400, dtype=torch.float64)
    for direction, value in zip(directions, tabulated):
        total = 0.0
        for step in s:
            y = origin + step * direction
            towards = (position - y) / (position - y).norm()
            gathered = _around(towards[None], polar.reshape(-1), azimuth.reshape(-1))[0]
            first = point_light_inscatter(
                medium,
                y,
                gathered,
                torch.full((len(gathered),), math.inf, dtype=torch.float64),
                position,
                beta,
                radius=LAMP.radius,
            )[:, 0]
            first = medium.albedo * LAMP.power[0] / (4 * math.pi) * first
            phase = henyey_greenstein(-(gathered @ -direction), medium.anisotropy)
            source = medium.albedo * density(medium, y[None])[0] * beta[0] * (phase * first * solid).sum()
            total += float(torch.exp(-optical_depth(medium, origin, direction, step) * beta[0]) * source) * float(
                s[1] - s[0]
            )
        assert value.item() == pytest.approx(total, rel=0.06)


def test_halo_vanishes_without_scattering():
    medium = Medium(extinction=0.08, albedo=0.0, components=[Homogeneous()])
    halo = lamp_halo(medium, LAMP, T([0.08]), grid=SMALL._replace(orders=3))
    light = halo_inscatter(halo, medium, T([0.0, 0.0, 1.5]), T([[0.0, 1.0, 0.0]]), T([20.0]), T([0.08]))
    # Tables hold logarithms, of at least those of tiny values
    assert light.abs().max().item() < 1e-250


def test_halo_threshold():
    dense = Medium(extinction=0.08, components=[HeightFog(falloff=3.0)], sun_attenuation=True)
    thin = dense.model_copy(update={"extinction": 1e-5})
    assert halo_threshold(dense, LAMP, T([0.08])) and not halo_threshold(thin, LAMP, T([1e-5]))


def test_apply_medium_adds_halos_with_multiple_scattering(monkeypatch):
    monkeypatch.setattr("visionsim.medium.halos.HALO_GRID", SMALL)
    monkeypatch.setattr("visionsim.medium.halos.cached_lamp_halo.__defaults__", (0.0, None, SMALL, 0.0))
    camera = {"w": 6, "h": 4, "fl_x": 4.0, "fl_y": 4.0, "cx": 3.0, "cy": 2.0}
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    pose[:3, 3] = [0.0, 0.0, 1.5]
    depth = np.full((4, 6), 15.0)
    single = Medium(extinction=0.08, anisotropy=0.6, components=[HeightFog(falloff=4.0)], sun_attenuation=True)
    multiple = single.model_copy(update={"multiple_scattering": True})
    lighting = Lighting(points=[LAMP])
    once = apply_medium(np.zeros((4, 6, 3)), depth, camera, pose, single, lighting).inscatter
    more = apply_medium(np.zeros((4, 6, 3)), depth, camera, pose, multiple, lighting).inscatter
    assert torch.all(more > once) and 1.02 < float(more.sum() / once.sum()) < 1.5
