import math
from functools import partial

import numpy as np
import pytest
import torch

from tests.test_occlusion import box, lamp_map, maps_of, room
from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium import halos as halos_module
from visionsim.medium.halos import (
    _HALOS,
    _HIT_DEPTH,
    HaloGrid,
    LampHalo,
    _around,
    _directions,
    _even_axes,
    _gathering_points,
    _halo_bytes,
    _interpolate,
    _lat_long,
    _middles,
    _refined_axes,
    _spread,
    _to_ground,
    _View,
    cached_lamp_halo,
    halo_glow,
    halo_inscatter,
    halo_threshold,
    lamp_halo,
)
from visionsim.medium.model import EmissiveSurface, HeightFog, Homogeneous, PointLight
from visionsim.medium.occlusion import ShadowMaps, hit_distances, occlusion_to, occupancy
from visionsim.medium.optics import density, henyey_greenstein, optical_depth, point_light_inscatter
from visionsim.medium.render import lamp_inscatter

SMALL = HaloGrid(radii=14, extent=(0.05, 30.0), directions=(8, 10), angles=(20, 8), gathered=(16, 16), orders=2)
LAMP = PointLight(position=(0.0, 6.0, 2.0), power=(1000.0,), radius=0.05)


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def logs(halo):
    """Logarithm of the light that a halo tabulates, of shape (r, t, p, g, b, c)."""
    return halo.values.to(halo.peaks.dtype) + halo.peaks[:, :, :, None, None, :]


def nodes_of(halo, grid):
    """Points of the grid of a halo, of shape (r * t * p, 3)."""
    (r_min, r_max), radii = grid.extent, torch.arange(grid.radii, dtype=torch.float64)
    radii = r_min * (r_max / r_min) ** (radii / (grid.radii - 1))
    if halo.axes is None:
        directions = _lat_long(*grid.directions, dtype=torch.float64)
    else:
        directions = _directions(*(axis.double() for axis in halo.axes))
    return (halo.center.double() + radii[:, None, None] * directions[None]).reshape(-1, 3)


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


def test_halo_of_a_ball_is_that_of_a_point_light():
    # A glowing ball, split into patches whose light only leaves outwards, spreads the halo of a point light of the
    # same power, from which its patches are merged
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    radiance, radius = 2000.0, 0.05
    normals = np.eye(3)[[0, 0, 1, 1, 2, 2]] * [[1], [-1], [1], [-1], [1], [-1]]
    ball = EmissiveSurface(
        position=LAMP.position,
        positions=tuple(tuple(np.add(LAMP.position, radius * n)) for n in normals),
        areas=(4 * math.pi * radius**2 / 6,) * 6,
        radiance=((radiance,),) * 6,
        # Each patch, a sixth of a sphere, is approximated by a sphere whose light leaves outwards, and the patches
        # spread over the sphere as its points do, so that the merged patch has its radius
        orientation=((1 / 6, 1 / 6, 1 / 6, 0, 0, 0),) * 6,
        spread=((0.0,) * 6,) * 6,
    )
    point = LAMP.model_copy(update={"power": (math.pi * radiance * 4 * math.pi * radius**2,), "radius": radius})
    ours = lamp_halo(medium, ball, beta, ground=-100.0, grid=SMALL)
    theirs = lamp_halo(medium, point, beta, ground=-100.0, grid=SMALL)
    assert torch.allclose(ours.peaks.exp(), theirs.peaks.exp(), rtol=1e-6)
    # Up to the rounding of half precision
    assert torch.allclose(ours.values.double(), theirs.values.double(), rtol=2**-10, atol=1e-4)


def test_objects_stop_the_halo():
    """Behind a large wall, which hides the lamp, gathering rays stop at the wall instead of gathering the light the lamp
    scatters on its other side, so that there is hardly any halo, while there is as much as without the wall in front
    of it, where nothing stops the light that reaches the points."""
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = torch.tensor([medium.extinction], dtype=torch.float32)
    low, high = [-30.0, 2.9, -30.0], [30.0, 3.1, 30.0]
    occlusion = occlusion_to(
        maps_of(box(low, high), [low, high], lamp=LAMP.position, resolution=256), dtype=torch.float32
    )
    voxels = occupancy(occlusion, T([-30.0, -24.0, -28.0]).float(), T([30.0, 36.0, 32.0]).float())
    lamp_maps = occlusion.lamp_maps
    through = lamp_halo(medium, LAMP, beta, ground=-100.0, maps=lamp_maps, grid=SMALL)
    stopped = lamp_halo(medium, LAMP, beta, ground=-100.0, maps=lamp_maps, grid=SMALL, occupancy=voxels)
    # Single precision tables stay finite where no light arrives
    assert torch.isfinite(logs(stopped)).all() and logs(stopped).min() < logs(through).min() - 5

    # Rays behind the wall, along it and away from it, and in front of it, away from it and along it
    origins = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 9.0, 2.0], [-3.0, 9.0, 2.0]])
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    distance = torch.full((4,), 15.0)
    ours = halo_inscatter(stopped, medium, origins, directions, distance, beta)[:, 0]
    theirs = halo_inscatter(through, medium, origins, directions, distance, beta)[:, 0]
    assert torch.all(ours[:2] < 0.01 * theirs[:2])
    assert torch.allclose(ours[2:], theirs[2:], rtol=0.1)


def test_halo_in_a_shadow_comes_from_the_side():
    """In the shadow beside the edge of a wall, light comes from the medium lit beside the shadow, and is scattered
    more towards the side it goes to than back, as light gathered at a point of the grid finds."""
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    low, high = [0.0, 2.9, -30.0], [30.0, 3.1, 30.0]
    occlusion = maps_of(box(low, high), [low, high], lamp=LAMP.position, resolution=256)
    voxels = occupancy(occlusion, T([-30.0, -24.0, -28.0]), T([30.0, 36.0, 32.0]))
    halo = lamp_halo(medium, LAMP, beta, ground=-100.0, maps=occlusion.lamp_maps, grid=SMALL, occupancy=voxels)

    # A point of the grid in the shadow, a little beyond the edge of the wall
    nodes = nodes_of(halo, SMALL)
    assert halo.lit is not None
    shadowed = nodes[(halo.lit.reshape(-1) == 0) & (nodes[:, 1] < 2.5) & ((nodes[:, 2] - 2).abs() < 1.5)]
    point = shadowed[shadowed[:, 0].clamp_min(0).argmin()]
    assert 0 < point[0] < 5

    # Light gathered from directions spread evenly over the sphere, along rays that stop at the wall
    z, wz = (T(a) for a in np.polynomial.legendre.leggauss(48))
    phi = (torch.arange(96, dtype=torch.float64) + 0.5) / 96 * 2 * math.pi
    z, phi = torch.meshgrid(z, phi, indexing="ij")
    rays = torch.stack([(1 - z**2).sqrt() * torch.cos(phi), (1 - z**2).sqrt() * torch.sin(phi), z], -1).reshape(-1, 3)
    solid = (wz[:, None] * torch.full((96,), 2 * math.pi / 96, dtype=torch.float64)).reshape(-1)
    origins = point.expand_as(rays)
    # Rays stop where those of the refined halo do, at a run of the voxels around the wall about a voxel long
    assert halo.axes is not None
    lengths = hit_distances(voxels, origins, rays, _to_ground(origins, rays, -100.0), depth=_HIT_DEPTH * voxels.size)
    light = lamp_inscatter(medium, origins, rays, lengths, beta, LAMP, area_levels=(), maps=occlusion.lamp_maps)[:, 0]
    # Towards the shadow, away from the lit medium, and back
    outgoing = T([[1.0, -0.3, 0.0], [-1.0, -0.3, 0.0]])
    outgoing = outgoing / outgoing.norm(dim=-1, keepdim=True)
    phase = henyey_greenstein(-(outgoing @ rays.T), medium.anisotropy)
    expected = medium.albedo * beta[0] * (phase * light * solid).sum(dim=-1)
    tabulated = _interpolate(halo, point.expand(2, 3), outgoing)[:, 0]
    assert expected[0] > 1.5 * expected[1]
    assert torch.allclose(tabulated, expected, rtol=0.2)


def test_directions_are_refined_around_objects():
    """The polar and azimuthal cells of the directions of the grid that hold the edges of objects within the halo's
    extent, as the lamp's map records them, and their neighbors, are split into cells of equal solid angle, whose
    centers are the directions of the grid, while the other cells keep the directions of the even grid, and the ground,
    the walls of a closed room and objects beyond the extent have no edges that need them."""
    lamp = T([0.0, 0.0, 2.0])
    grid = HaloGrid(radii=14, extent=(0.05, 30.0), directions=(8, 12))
    # A box on the ground, from 14° up to 27° down as seen from the lamp, and within 11° of +x in azimuth
    maps = lamp_map(lamp.tolist(), box([4.0, -0.8, 0.0], [5.0, 0.8, 3.0]))
    axes = _refined_axes(maps, lamp, 0.0, grid)
    assert axes is not None and grid.refinement == 3
    polar, azimuths = axes
    assert torch.all(polar.diff() < 0) and torch.all(azimuths.diff() > 0)
    assert float(azimuths[0]) > 0 and float(azimuths[-1]) < 2 * math.pi
    # Polar cells 3 to 5 hold the box, and 2 and 6 are next to them; azimuthal cells 11 and 0, and 10 and 1
    even_polar, even_azimuths = _even_axes(8, 12, dtype=torch.float64)
    split = (torch.arange(3, dtype=torch.float64) + 0.5) / 3
    expected = torch.cat([even_polar[:2], 1 - (torch.arange(2, 7)[:, None] + split).reshape(-1) / 4, even_polar[7:]])
    assert torch.allclose(polar, expected)
    expected = torch.cat([split, 1 + split, even_azimuths[2:10] * 6 / math.pi, 10 + split, 11 + split]) * (math.pi / 6)
    assert torch.allclose(azimuths, expected)
    # Nothing to refine without objects within the extent, beyond the ground, in a closed room, which the walls seen at
    # grazing angles don't change, or with a refinement of one
    assert _refined_axes(lamp_map(lamp.tolist(), box([40.0, -0.8, 0.0], [41.0, 0.8, 3.0])), lamp, 0.0, grid) is None
    ground = lamp_map(lamp.tolist(), box([-100.0, -100.0, -1.0], [100.0, 100.0, 0.0]))
    assert _refined_axes(ground, lamp, 0.0, grid) is None
    assert _refined_axes(lamp_map(lamp.tolist(), room()[0]), lamp, 0.0, grid) is None
    assert _refined_axes(maps, lamp, 0.0, grid._replace(refinement=1)) is None


def test_tables_are_interpolated_between_refined_directions():
    """Tables are interpolated linearly between the refined directions of the grid, which keeps light that varies
    linearly with the vertical component of directions, and light at the points of the even grid spreads to the others
    from those alike only."""
    lamp = T([0.0, 0.0, 2.0])
    grid = HaloGrid(radii=14, extent=(0.05, 30.0), directions=(8, 12))
    axes = _refined_axes(lamp_map(lamp.tolist(), box([4.0, -0.8, 0.0], [5.0, 0.8, 3.0])), lamp, 0.0, grid)
    assert axes is not None
    polar, azimuths = axes
    n_t, n_p = len(polar), len(azimuths)
    peaks = torch.log(2 + polar)[None, :, None, None].expand(14, n_t, n_p, 1)
    values = torch.zeros(14, n_t, n_p, 6, 4, 1, dtype=torch.float16)
    halo = LampHalo(lamp, grid.extent, -100.0, values, peaks, axes=axes)
    rng = np.random.default_rng(0)
    directions = T(rng.normal(size=(500, 3)))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    directions = directions[(directions[:, 2] < polar[0]) & (directions[:, 2] > polar[-1])]
    points = lamp + T(rng.uniform(0.5, 20.0, (len(directions), 1))) * directions
    light = _interpolate(halo, points, directions)[:, 0]
    assert torch.allclose(light, 2 + directions[:, 2], rtol=1e-9)

    # Light at the points of the even grid among them, which varies linearly with the vertical component of directions,
    # spreads to the others linearly between them
    middles = (_middles(((1 - polar) / 2 * 8).floor()), _middles((azimuths / (2 * math.pi) * 12).floor()))
    assert torch.allclose(polar[middles[0]], _even_axes(8, 12, dtype=torch.float64)[0])
    view = _View((8, 12), None, None, (polar[middles[0]], azimuths[middles[1]]))
    even = (3 + view.axes[0])[None, :, None].expand(14, 8, 12)
    spread = _spread(even.reshape(-1, 1), axes, view, None).reshape(14, n_t, n_p)
    inner = (polar <= view.axes[0][0]) & (polar >= view.axes[0][-1])
    assert torch.allclose(spread[:, inner], (3 + polar[inner])[None, :, None].expand(14, -1, n_p), rtol=1e-12)
    assert torch.equal(spread[:, middles[0]][:, :, middles[1]], even)
    # Lit by the lamp above the horizon and in its shadow below it, from the points alike only
    lit = (polar > 0).double()[None, :, None].expand(14, n_t, n_p)
    view = view._replace(lit=lit[:, middles[0]][:, :, middles[1]])
    even = torch.where(view.lit > 0.5, 1.0, 2.0)
    spread = _spread(even.reshape(-1, 1), axes, view, lit).reshape(14, n_t, n_p)
    assert torch.allclose(spread, torch.where(lit > 0.5, 1.0, 2.0), rtol=3e-3)


def test_refined_halos_resolve_the_light_in_shadows():
    """In the shadow beyond the edge of a wall, the light scattered twice fades over distances shorter than the spacing
    of the even grid, which a refined grid resolves, and rays that pass right beside the wall gather the light beside
    it, against gathering it at points between the points of the grid."""
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    low, high = [0.0, 2.9, -30.0], [30.0, 3.1, 30.0]
    occlusion = maps_of(box(low, high), [low, high], lamp=LAMP.position, resolution=256)
    voxels = occupancy(occlusion, T([-30.0, -24.0, -28.0]), T([30.0, 36.0, 32.0]))
    kwargs = {"ground": -100.0, "maps": occlusion.lamp_maps, "grid": SMALL, "occupancy": voxels}
    refined = lamp_halo(medium, LAMP, beta, **kwargs)
    even = lamp_halo(medium, LAMP, beta, shadow_edges=False, **kwargs)
    assert refined.axes is not None and even.axes is None
    # Rays only go on beside objects where the grid is refined, as on the even grid, the voxels that stop them make up
    # for part of the light that it spreads into shadows
    unrefined = lamp_halo(medium, LAMP, beta, **{**kwargs, "grid": SMALL._replace(refinement=1)})
    assert unrefined.axes is None and torch.equal(unrefined.values, even.values)
    assert torch.equal(unrefined.peaks, even.peaks) and torch.equal(unrefined.glow, even.glow)

    # Points in the shadow, from 0.2 to 3 m beyond its edge, 2 to 6 m behind the wall
    x, y = torch.meshgrid(T([0.2, 0.7, 1.3, 2.1, 3.0]), T([0.9, -1.0, -3.1]), indexing="ij")
    points = torch.stack([x.reshape(-1), y.reshape(-1), torch.full((15,), 2.3, dtype=torch.float64)], -1)
    z, wz = (T(a) for a in np.polynomial.legendre.leggauss(64))
    phi = (torch.arange(128, dtype=torch.float64) + 0.5) / 128 * 2 * math.pi
    z, phi = torch.meshgrid(z, phi, indexing="ij")
    rays = torch.stack([(1 - z**2).sqrt() * torch.cos(phi), (1 - z**2).sqrt() * torch.sin(phi), z], -1).reshape(-1, 3)
    solid = (wz[:, None] * torch.full((128,), 2 * math.pi / 128, dtype=torch.float64)).reshape(-1)
    # Away from the lamp, and across the edge of the shadow, both ways
    outgoing = T([[0.0, -1.0, 0.0], [0.6, -0.8, 0.0], [-0.6, -0.8, 0.0]])
    expected = []
    for point in points:
        origins = point.expand_as(rays)
        # Rays stop at the wall itself
        lengths = torch.minimum(T(box(low, high)(origins.numpy(), rays.numpy())), _to_ground(origins, rays, -100.0))
        light = lamp_inscatter(medium, origins, rays, lengths, beta, LAMP, area_levels=(), maps=occlusion.lamp_maps)
        phase = henyey_greenstein(-(outgoing @ rays.T), medium.anisotropy)
        expected.append(medium.albedo * beta[0] * (phase * light[:, 0] * solid).sum(dim=-1))
    expected = torch.stack(expected)

    def error(halo):
        ours = _interpolate(halo, points.repeat_interleave(3, 0), outgoing.repeat(15, 1))[:, 0].reshape(15, 3)
        return float(((ours - expected).abs() / expected).mean())

    # About 16% against 185% here, with the coarse grid of these tests
    assert error(refined) < 0.25 and error(refined) < 0.2 * error(even)


def test_light_scattered_more_than_twice_is_gathered_in_shadows(monkeypatch):
    """On a refined grid, light scattered more than twice is gathered at the middle points of the cells of the even
    grid, from which its ratio to the light it is gathered from is interpolated elsewhere, and at the points in the
    lamp's shadows and next to them, where that ratio changes much faster: in the shadow of a wall, this is as close
    to gathering it at every point as the interpolation of tables allows, and much closer than interpolating it there
    too."""
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    low, high = [0.0, 2.9, -30.0], [30.0, 3.1, 30.0]
    occlusion = maps_of(box(low, high), [low, high], lamp=LAMP.position, resolution=256)
    voxels = occupancy(occlusion, T([-30.0, -24.0, -28.0]), T([30.0, 36.0, 32.0]))
    grid = SMALL._replace(radii=10, angles=(12, 6), gathered=(12, 12), orders=3)
    kwargs = {"ground": -100.0, "maps": occlusion.lamp_maps, "grid": grid, "occupancy": voxels}
    halo = lamp_halo(medium, LAMP, beta, **kwargs)
    assert halo.axes is not None and halo.lit is not None and halo.outside is not None
    nodes, middle, (rows, columns) = _gathering_points(halo.axes, grid.directions, halo.lit, grid.radii)
    shaded = (halo.lit < 1).reshape(-1)
    assert len(rows) == grid.directions[0] and len(columns) == grid.directions[1]
    assert torch.all(torch.isin(middle, nodes)) and torch.all(torch.isin(shaded.nonzero()[:, 0], nodes))
    assert len(middle) + int(shaded.sum()) < len(nodes) < 0.7 * len(shaded)

    # Gathered at every point, or at the middle points of the cells only
    original = halos_module._gathering_points
    monkeypatch.setattr(
        halos_module, "_gathering_points", lambda axes, even, lit, radii: original(axes, even, None, radii)
    )
    middles = lamp_halo(medium, LAMP, beta, **kwargs)
    monkeypatch.setattr(halos_module, "_middles", lambda cells: torch.arange(len(cells)))
    full = lamp_halo(medium, LAMP, beta, **kwargs)

    def errors(ours):
        """Relative difference of the light scattered towards every outgoing direction from gathering it at every point,
        in the shadow and elsewhere."""
        light, expected = (torch.exp(logs(h)).sum(dim=(3, 4)).reshape(-1) for h in (ours, full))
        error = (light - expected).abs() / expected
        outside = halo.outside.reshape(-1) > 0
        return float(error[shaded & outside].mean()), float(error[~shaded & outside].mean())

    # About 0 and 0.2% here, against 9% and 0.5%
    assert errors(halo)[0] < 0.01 and errors(halo)[0] < 0.2 * errors(middles)[0]
    assert errors(halo)[1] < 0.01


def test_lights_in_a_closed_room_keep_their_halo():
    """In a closed room, which the sky doesn't see, the medium that a lamp or an emissive surface lights is carved out of
    the visual hull of the objects by its map, so that the walls hardly stop its halo inside the room, and objects don't
    stop the halo of a lamp without a map, which casts no shadows, at all."""
    medium = Medium(extinction=0.15, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    walls, bounds = room()
    dark = maps_of(walls, bounds, resolution=128)
    position = (0.0, 0.0, 2.0)
    light = lamp_map(position, walls)
    ball = EmissiveSurface(
        position=position,
        positions=(position,),
        areas=(0.1257,),
        radiance=((2000.0,),),
        orientation=((1 / 6, 1 / 6, 1 / 6, 0.0, 0.0, 0.0),),
    )
    point = PointLight(position=position, power=(1000.0,), radius=0.1)
    # Rays across the room
    origins = T([[4.0, -4.0, 1.5], [4.0, 0.0, 1.0], [-4.0, 4.0, 2.5]])
    directions = T([[-1.0, 1.0, 0.0], [-1.0, 0.0, 0.3], [1.0, -0.5, -0.2]])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T([6.0, 6.0, 6.0])
    for lamp, maps, occlusion in (
        (ball, light, dark._replace(emissive_maps=light)),
        (point, light, dark._replace(lamp_maps=light)),
        (point, None, dark),
    ):
        free = cached_lamp_halo(medium, lamp, beta, 0.0, maps, SMALL)
        stopped = cached_lamp_halo(medium, lamp, beta, 0.0, maps, SMALL, occlusion=occlusion)
        ours = halo_inscatter(stopped, medium, origins, directions, distance, beta)
        theirs = halo_inscatter(free, medium, origins, directions, distance, beta)
        assert torch.allclose(ours, theirs, rtol=0.1) if maps is not None else torch.equal(ours, theirs)


def test_cached_halos_take_little_memory(monkeypatch):
    """Cached halos keep the map of their lamp alone, in single precision, store channels that only differ by the color
    of the lamp once, and the cache drops the least recently used halos beyond its memory."""
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    wall = box([-30.0, 2.9, -30.0], [30.0, 3.1, 30.0])
    # Maps of a lamp at two positions, in double precision
    singles = [lamp_map(position, wall) for position in (LAMP.position, (0.0, 8.0, 2.0))]
    maps = ShadowMaps(
        torch.cat([m.depths for m in singles]),
        torch.tensor([0, len(singles[0].depths)]),
        *(torch.cat([getattr(m, name) for m in singles]) for name in ShadowMaps._fields[2:]),
    )
    colored = LAMP.model_copy(update={"power": (1000.0, 500.0, 250.0)})
    white = cached_lamp_halo(medium, LAMP, T([0.08]), -100.0, maps, SMALL)
    halo = cached_lamp_halo(medium, colored, T([0.08] * 3), -100.0, maps, SMALL)
    assert halo.maps is not None and halo.maps.depths.dtype == torch.float32
    assert torch.equal(halo.maps.depths.double(), singles[0].depths.float().double())
    assert halo.values.dtype == torch.float16 and halo.values.shape[-1] == 1 and halo.peaks.shape[-1] == 3
    points = T([[0.0, 0.0, 1.5], [2.0, 9.0, 2.5], [-1.0, 5.0, 2.0]])
    outgoing = T([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    expected = _interpolate(white, points, outgoing) * T([1.0, 0.5, 0.25])
    assert torch.allclose(_interpolate(halo, points, outgoing), expected, rtol=1e-3)
    # Channels that differ by more than the color of the lamp are each stored
    monkeypatch.setattr("visionsim.medium.halos.HALO_CACHE_BYTES", _halo_bytes(halo) + 1)
    spread = cached_lamp_halo(medium, colored, T([0.08, 0.06, 0.04]), -100.0, maps, SMALL)
    assert spread.values.shape[-1] == 3 and list(_HALOS.values()) == [spread]


def test_interpolation_leaves_out_points_inside_objects():
    """Points of the grid inside an object, which hold the light of either side of it, aren't blended with the light of
    points outside of it."""
    peaks = torch.full((14, 8, 10, 1), math.log(2.0), dtype=torch.float64)
    peaks[6:8] = math.log(100.0)
    peaks[8:] = math.log(5.0)
    outside = torch.ones(14, 8, 10, dtype=torch.float64)
    outside[6:8] = 0
    radii = 0.05 * (30 / 0.05) ** (torch.arange(14, dtype=torch.float64) / 13)
    values = torch.zeros(14, 8, 10, 20, 8, 1, dtype=torch.float16)
    halo = LampHalo(T(LAMP.position), (0.05, 30.0), -100.0, values, peaks, outside)
    # Halfway between the last point of the grid before the object and the first one inside of it, and beyond it
    distances = torch.stack([(radii[5] * radii[6]).sqrt(), (radii[7] * radii[8]).sqrt()])
    points = T(LAMP.position) + distances[:, None] * T([0.3, 0.9, 0.1]) / T([0.3, 0.9, 0.1]).norm()
    light = _interpolate(halo, points, T([[0.0, 1.0, 0.0]] * 2))[:, 0]
    assert light.tolist() == pytest.approx([2.0, 5.0], rel=0.01)
    plain = _interpolate(halo._replace(outside=None), points, T([[0.0, 1.0, 0.0]] * 2))[:, 0]
    assert torch.all(plain > 10.0)


def test_glow_matches_gathering_at_points():
    """The irradiance of surfaces from the light of a lamp scattered once, which halos tabulate as their glow, against
    gathering it from directions concentrated towards the lamp, for normals facing the lamp, away from it and across,
    at points of the grid and between them."""
    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    halo = lamp_halo(medium, LAMP, beta, ground=-100.0, grid=SMALL)
    radii = 0.05 * (30 / 0.05) ** (torch.arange(SMALL.radii, dtype=torch.float64) / (SMALL.radii - 1))
    nodes = (T(LAMP.position) + radii[:, None, None] * _lat_long(*SMALL.directions, dtype=torch.float64)).reshape(-1, 3)
    on_grid = nodes[[(8 * 8 + 2) * 10 + 3, (10 * 8 + 5) * 10 + 7]]
    between = T(LAMP.position) + T([[1.3, -2.1, 0.4], [-4.0, 3.0, -1.5]])
    t, w = (T(a) for a in np.polynomial.legendre.leggauss(48))
    t, w = (t + 1) / 2, w / 2
    polar, azimuth = torch.meshgrid(
        math.pi * t**2, (torch.arange(48, dtype=torch.float64) + 0.5) / 48 * 2 * math.pi, indexing="ij"
    )
    solid = ((2 * math.pi * t * w)[:, None] * torch.sin(polar) * 2 * math.pi / 48).reshape(-1)
    # Within the error of the halo's coarser gathering directions at points of the grid, and of the interpolation between
    # them
    for points, tolerance in ((on_grid, 0.04), (between, 0.06)):
        towards = (T(LAMP.position) - points) / (T(LAMP.position) - points).norm(dim=-1, keepdim=True)
        across = torch.linalg.cross(towards, T([0.0, 0.0, 1.0]).expand_as(towards))
        across = across / across.norm(dim=-1, keepdim=True)
        for normals in (towards, -towards, across, (towards + across) / math.sqrt(2)):
            expected = []
            for point, normal, axis in zip(points, normals, towards):
                rays = _around(axis[None], polar.reshape(-1), azimuth.reshape(-1))[0]
                light = point_light_inscatter(
                    medium, point, rays, torch.full((len(rays),), math.inf, dtype=torch.float64), T(LAMP.position), beta
                )[:, 0]
                light = medium.albedo * LAMP.power[0] / (4 * math.pi) * light
                expected.append(float((light * (rays @ normal).clamp_min(0) * solid).sum()))
            ours, share = halo_glow(halo, points, normals)
            assert torch.all(share == 1)
            ours = ours[:, 0]
            # Normals facing away from the lamp receive little light, which is compared to that of the others
            scale = max(expected)
            assert ours.tolist() == pytest.approx(expected, rel=tolerance, abs=tolerance * scale)


def test_halos_relight_the_light_of_surfaces():
    """Surfaces lit directly by a lamp are lit by its glow too, in the shadow of a wall where the lamp doesn't light
    them, while surfaces lit with the similarity relation, which already counts part of their glow, aren't."""
    from tests.test_reflections import wall_points, wall_scene
    from visionsim.medium.reflections import lamp_reflection

    medium = Medium(extinction=0.08, anisotropy=0.6, components=[HeightFog(falloff=3.0)], sun_attenuation=True)
    beta = T([medium.extinction])
    lamp, _, maps = wall_scene()
    direct = lamp_reflection(medium, lamp, beta, ground_albedo=(0.4,), maps=maps, occluders=wall_points(), glow=True)
    halo = lamp_halo(medium, lamp, beta, maps=maps, grid=SMALL, reflection=direct)
    assert halo.reflection is not None and halo.glow is not None
    relit, texture = halo.reflection.texture[..., 0], direct.texture[..., 0]
    assert torch.all(relit >= texture) and float(relit.sum()) > 1.02 * float(texture.sum())
    # Behind the wall, towards -y from the lamp's foot from 3.5 m up to 20 m, within the halo's extent, the ground only
    # receives the glow
    assert float(texture[40:56, 42:54].max()) == 0.0 and float(relit[40:56, 42:54].min()) > 0.0
    similar = lamp_reflection(medium, lamp, beta, ground_albedo=(0.4,), maps=maps, occluders=wall_points())
    assert lamp_halo(medium, lamp, beta, maps=maps, grid=SMALL, reflection=similar).reflection is similar


def test_light_of_surfaces_lit_by_later_orders_is_scattered_again():
    """With four orders, the light of surfaces lit by the glow of the third, i.e. by the lamp's light scattered twice
    and the light of surfaces scattered once, is scattered again in the fourth, which adds a little light to the halo,
    and none with three orders, whose last glow only lights the light that camera rays gather once scattered."""
    from tests.test_reflections import wall_points, wall_scene
    from visionsim.medium.reflections import lamp_reflection

    medium = Medium(extinction=0.08, anisotropy=0.6, components=[HeightFog(falloff=3.0)], sun_attenuation=True)
    beta = T([medium.extinction])
    lamp, _, maps = wall_scene()
    direct = lamp_reflection(
        medium, lamp, beta, ground_albedo=(0.4,), object_albedo=(0.5,), maps=maps, occluders=wall_points(), glow=True
    )

    def light(grid, relit_scattering):
        halo = lamp_halo(medium, lamp, beta, maps=maps, grid=grid, reflection=direct, relit_scattering=relit_scattering)
        return torch.exp(logs(halo).double()).sum(), halo

    coarse = SMALL._replace(angles=(12, 6), gathered=(10, 10))
    (more, _), (fewer, _) = (light(coarse._replace(orders=4), on) for on in (True, False))
    assert fewer < more < 1.1 * fewer
    three = [light(coarse._replace(orders=3), on)[1] for on in (True, False)]
    assert torch.equal(three[0].values, three[1].values) and torch.equal(three[0].glow, three[1].glow)


def test_cached_halos_keep_the_light_of_surfaces():
    """Given the albedo of surfaces, cached halos keep the light that surfaces reflect, lit directly and by their glow,
    which they compute, keyed by the albedos rather than by any reflected light computed beforehand, and which counts
    towards the cache's memory."""
    from tests.test_reflections import wall_points, wall_scene
    from visionsim.medium.reflections import lamp_reflection, relit_reflection

    medium = Medium(extinction=0.08, anisotropy=0.6, components=[HeightFog(falloff=3.0)], sun_attenuation=True)
    beta = T([medium.extinction])
    lamp, _, maps = wall_scene()
    occluders = wall_points()
    halo = cached_lamp_halo(medium, lamp, beta, 0.0, maps, SMALL, albedos=((0.4,), (0.3,)), occluders=occluders)
    assert halo.reflection is not None and halo.reflection.albedos is not None
    # The same as relighting the light reflected by surfaces lit directly with the halo's glow
    direct = lamp_reflection(medium, lamp, beta, 0.0, (0.4,), (0.3,), maps, occluders, glow=True)
    relit = relit_reflection(direct, medium, beta, partial(halo_glow, halo))
    assert torch.allclose(halo.reflection.texture, relit.texture) and torch.allclose(
        halo.reflection.patches.power, relit.patches.power
    )
    # Equal albedos, whatever their containers, find the same halo
    again = cached_lamp_halo(medium, lamp, beta, 0.0, maps, SMALL, albedos=([0.4], [0.3]), occluders=occluders)
    assert again is halo
    reflected = sum(t.numel() * t.element_size() for t in _tensors(halo.reflection))
    assert _halo_bytes(halo) > reflected > 0


def _tensors(value):
    """Tensors of nested tuples."""
    if torch.is_tensor(value):
        return [value]
    return [t for v in value for t in _tensors(v)] if isinstance(value, tuple) else []


def test_glow_fades_out_at_the_halo_extent():
    """Across the outermost cell of the grid, the glow fades out, and the light of the lamp goes from the full extinction
    to the extinction reduced by the similarity relation, so that surfaces are lit continuously across the extent."""
    from visionsim.medium.surfaces import surface_irradiance

    medium = Medium(extinction=0.08, anisotropy=0.6, components=[Homogeneous()])
    beta = T([medium.extinction])
    halo = lamp_halo(medium, LAMP, beta, ground=-100.0, grid=SMALL)
    radii = torch.linspace(10.0, 40.0, 601, dtype=torch.float64)
    direction = T([0.6, 0.8, 0.0])
    points = T(LAMP.position) + radii[:, None] * direction
    normals = (-direction).expand_as(points).contiguous()
    lighting = Lighting(points=[LAMP])
    clear, similar = surface_irradiance(medium, lighting, beta, points, normals)
    _, through = surface_irradiance(medium, lighting, beta, points, normals, glows=[partial(halo_glow, halo)])
    ratio = (through / clear)[:, 0]
    _, share = halo_glow(halo, points, normals)
    outer = 30.0 * (0.05 / 30.0) ** (1 / (SMALL.radii - 1))
    assert torch.all(share[radii <= outer] == 1) and torch.all(share[radii >= 30.0] == 0)
    assert torch.all(share[1:] <= share[:-1])
    # Beyond the extent, the similarity relation, and within the grid, the full extinction and the glow
    assert torch.allclose(through[radii >= 30.0], similar[radii >= 30.0], rtol=1e-12)
    full = clear * torch.exp(-medium.extinction * radii)[:, None]
    assert torch.all(through[radii <= outer] > 1.1 * full[radii <= outer])
    # Steps of 5 cm change the light by much less than the gap between both models at the extent
    assert float((ratio[1:] / ratio[:-1] - 1).abs().max()) < 0.01


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
    # The light that the ground reflects is scattered more than once too, and the halo's glow lights surfaces, here the
    # ground, which the lamp's light dims less than with the similarity relation alone
    grey = lighting.model_copy(update={"ground_albedo": (0.4,) * 3})
    normals = np.broadcast_to([0.0, 0.0, 1.0], (4, 6, 3))
    frames = {
        (medium.multiple_scattering, lights.ground_albedo[0]): apply_medium(
            np.ones((4, 6, 3)), depth, camera, pose, medium, lights, normals=normals
        )
        for medium in (single, multiple)
        for lights in (lighting, grey)
    }
    reflected = {k: frames[(k, 0.4)].inscatter - frames[(k, 0.0)].inscatter for k in (False, True)}
    assert torch.all(reflected[True] > 0) and float(reflected[True].sum()) > float(reflected[False].sum())
    lit = {k: frames[(k, 0.0)].illumination for k in (False, True)}
    assert lit[True] is not None and lit[False] is not None and torch.all(lit[True] > lit[False])
