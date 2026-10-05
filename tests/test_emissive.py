import math

import numpy as np
import pytest
import torch

from visionsim.medium.lights import emitters, merge_patches, orientation_scale, patch_radius, projected_area
from visionsim.medium.model import AreaLight, EmissiveSurface, Homogeneous, Lighting, Medium, PointLight
from visionsim.medium.render import lamp_inscatter
from visionsim.simulate.blender import _emissive_lamps, _separate, _subdivide, _triangles

FOG = Medium(extinction=0.05, anisotropy=0.7, components=[Homogeneous()])


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def sphere_directions(count=20000):
    i = torch.arange(count, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / count
    phi = math.pi * (3 - math.sqrt(5)) * i
    s = (1 - z * z).sqrt()
    return torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], dim=-1)


def relative(emitter, directions):
    # Relative intensity of an emitter in given directions
    result = torch.ones(len(directions), dtype=torch.float64)
    if emitter.axis is not None:
        cosine = directions @ emitter.axis
        result = result * (cosine >= emitter.cone) * (emitter.profile(cosine) if emitter.profile else 1)
    return result * (emitter.pattern(directions) if emitter.pattern is not None else 1)


def intensity(lamp, directions):
    return sum(e.intensity[0] * relative(e, directions) for e in emitters(lamp, 1, dtype=torch.float64))


def moments(normals, front, back):
    # Orientation and facing of faces of equal areas
    s, d = (front + back) / 2, (front - back) / 2
    orientation = torch.einsum("k,ki,kj->ij", s, normals, normals) / len(normals)
    return orientation, (d[:, None] * normals).mean(dim=0)


def seen(directions, normals, front, back):
    # Area of faces of equal areas seen from each direction, counting those whose light leaves towards it
    cosines = directions @ normals.T
    return (front * cosines.clamp_min(0) + back * (-cosines).clamp_min(0)).mean(dim=-1)


@pytest.mark.parametrize("shape", ["two-sided", "one-sided", "sphere", "cylinder", "hemisphere"])
def test_projected_area_is_exact_for_simple_shapes(shape):
    many = sphere_directions(4000)
    ones, zeros = torch.ones(len(many), dtype=torch.float64), torch.zeros(len(many), dtype=torch.float64)
    if shape in ("two-sided", "one-sided"):
        normals = T([[0.3, -0.2, 0.9]]) / T([[0.3, -0.2, 0.9]]).norm()
        front, back = T([1.0]), T([1.0 if shape == "two-sided" else 0.0])
    elif shape == "cylinder":
        phi = (torch.arange(2000, dtype=torch.float64) + 0.5) / 2000 * 2 * math.pi
        normals = torch.stack([phi.cos(), phi.sin(), torch.zeros_like(phi)], dim=-1)
        front, back = torch.ones_like(phi), torch.zeros_like(phi)
    else:
        # Faces of a sphere, or of its upper half, whose light only leaves outwards
        keep = many[:, 2] > 0 if shape == "hemisphere" else ones.bool()
        normals, front, back = many[keep], ones[keep], zeros[keep]
    orientation, facing = moments(normals, front, back)
    directions = sphere_directions(3000)
    approximation = projected_area(directions, orientation, facing, orientation_scale(orientation))
    exact = seen(directions, normals, front, back)
    assert torch.allclose(approximation, exact, atol=2e-3 * exact.max().item())


def test_emissive_surfaces_emit_their_light():
    # A flat patch both sides of which emit, and a sphere, which emits outwards only
    radiance = 3.0
    lamp = EmissiveSurface(
        position=(0, 0, 0),
        positions=((0, 0, 0), (1, 0, 0)),
        areas=(2.0, 0.5),
        radiance=((radiance,), (radiance,)),
        orientation=((0, 0, 1, 0, 0, 0), (1 / 6, 1 / 6, 1 / 6, 0, 0, 0)),
    )
    power = intensity(lamp, sphere_directions()).mean().item() * 4 * math.pi
    assert power == pytest.approx(radiance * (2 * math.pi * 2.0 + math.pi * 0.5), rel=1e-3)
    assert lamp.area == 2.5
    with pytest.raises(ValueError, match="one per patch"):
        EmissiveSurface(position=(0, 0, 0), positions=((0, 0, 0),), areas=(1.0,), radiance=(), orientation=())
    with pytest.raises(ValueError, match="one per patch"):
        EmissiveSurface.model_validate(lamp.model_dump() | {"spread": ((0.1,) * 6,)})
    # Distances are clamped at the radius of each patch, by default half the square root of its area, otherwise the root
    # mean square distance between its points and its center
    assert [e.radius for e in emitters(lamp, 1)] == pytest.approx([math.sqrt(2.0) / 2, math.sqrt(0.5) / 2])
    spread = EmissiveSurface.model_validate(
        lamp.model_dump() | {"spread": ((0.05, 0.04, 0, 0, 0, 0), (0.01,) * 3 + (0,) * 3)}
    )
    assert [e.radius for e in emitters(spread, 1)] == pytest.approx([0.3, math.sqrt(0.03)])


def symmetric(matrix):
    return tuple(float(matrix[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))


def test_flat_patch_shines_as_an_area_light():
    # The front of a flat patch shines as an area light of the same area, and rays that pass close to it see it as a
    # grid over the rectangle of the same spread
    normal, area, radiance = np.array([0.2, -0.3, -1.0]) / np.linalg.norm([0.2, -0.3, -1.0]), 0.72, 40.0
    axis_u = np.cross(normal, [0.0, 0.0, 1.0])
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    spread = 1.2**2 / 12 * np.outer(axis_u, axis_u) + 0.6**2 / 12 * np.outer(axis_v, axis_v)
    center = (1.0, 6.0, 2.8)
    patch = EmissiveSurface(
        position=center,
        positions=(center,),
        areas=(area,),
        radiance=((radiance,),),
        orientation=(symmetric(np.outer(normal, normal) / 2),),
        facing=(tuple(normal / 2),),
        spread=(symmetric(spread),),
    )
    light = AreaLight(
        position=center, direction=tuple(normal), axis_u=tuple(axis_u), size=(1.2, 0.6), power=(math.pi * radiance * area,)
    )  # fmt: skip
    rng = np.random.default_rng(3)
    origin = T([0.0, 0.0, 1.5])
    # Rays towards points around the light, which mostly pass within a few radii of it
    targets = np.asarray(center) + rng.normal(size=(300, 3)) * rng.uniform(size=(300, 1)) ** 2 * 6
    directions = T(targets) - origin
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T(np.where(rng.uniform(size=300) < 0.3, np.inf, 2 * np.linalg.norm(targets - origin.numpy(), axis=-1)))
    beta = T([FOG.extinction])
    ours = lamp_inscatter(FOG, origin, directions, distance, beta, patch)
    reference = lamp_inscatter(FOG, origin, directions, distance, beta, light)
    assert torch.allclose(ours, reference, rtol=1e-6, atol=1e-12) and reference.sum() > 0
    # Without its spread, the patch is a single emitter, as is an area light without grids
    single = patch.model_copy(update={"spread": ()})
    far = lamp_inscatter(FOG, origin, directions, distance, beta, light, area_levels=())
    assert torch.allclose(lamp_inscatter(FOG, origin, directions, distance, beta, single), far, rtol=1e-3, atol=1e-12)


def test_closed_surface_shines_as_a_point_light():
    # A sphere, whose light only leaves outwards, shines as a point light of the same power
    radiance, area = 10.0, 0.5
    patch = EmissiveSurface(
        position=(2.0, 5.0, 2.0),
        positions=((2.0, 5.0, 2.0),),
        areas=(area,),
        radiance=((radiance,),),
        orientation=((1 / 6, 1 / 6, 1 / 6, 0, 0, 0),),
    )
    point = PointLight(position=(2.0, 5.0, 2.0), power=(math.pi * radiance * area,), radius=math.sqrt(area) / 2)
    rng = np.random.default_rng(4)
    origin = T([0.0, 0.0, 1.5])
    directions = T(rng.normal(size=(200, 3)) + [0.2, 1.0, 0.0])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T(rng.uniform(1, 30, size=200))
    beta = T([FOG.extinction])
    ours = lamp_inscatter(FOG, origin, directions, distance, beta, patch)
    reference = lamp_inscatter(FOG, origin, directions, distance, beta, point)
    assert torch.allclose(ours, reference, rtol=1e-3, atol=1e-12)


def tracer(corners):
    """Whether rays hit any of the triangles, by testing each of them (Möller-Trumbore)."""
    v0, e1, e2 = corners[:, 0], corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]

    def trace(origins, directions):
        hits = np.zeros(len(origins), dtype=bool)
        for i in range(0, len(origins), 1024):
            o, d = origins[i : i + 1024, None], directions[i : i + 1024, None]
            p = np.cross(d, e2)
            det = (e1 * p).sum(-1)
            inverse = 1 / np.where(np.abs(det) > 1e-12, det, np.inf)
            t = o - v0
            u = (t * p).sum(-1) * inverse
            q = np.cross(t, e1)
            v = (d * q).sum(-1) * inverse
            hit = (u >= 0) & (v >= 0) & (u + v <= 1) & ((e2 * q).sum(-1) * inverse > 0)
            hits[i : i + 1024] = hit.any(axis=-1)
        return hits

    return trace


def quad(center, u, v):
    center, u, v = (np.asarray(x, dtype=float) for x in (center, u, v))
    a, b, c, d = center - u - v, center + u - v, center + u + v, center - u + v
    return np.asarray([[a, b, c], [a, c, d]])


def cube(center, half):
    faces = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal, u, v = np.zeros(3), np.zeros(3), np.zeros(3)
            normal[axis], u[(axis + 1) % 3], v[(axis + 2) % 3] = sign, half, half
            faces.append(quad(np.asarray(center) + half * normal, sign * u, v))
    return np.concatenate(faces)


def emitted(lamp):
    # Light emitted by the patches of exported lamps, as their orientation and facing integrate to 2π trace(M) and 0
    return sum(
        r[0] * a * 2 * math.pi * sum(o[:3]) for r, a, o in zip(lamp["radiance"], lamp["areas"], lamp["orientation"])
    )


def test_subdivide_and_separate():
    corners = np.concatenate([quad([0, 0, 0], [1, 0, 0], [0, 1, 0]), quad([3, 0, 0], [0.2, 0, 0], [0, 0.2, 0])])
    fine, source = _subdivide(corners, 0.1)
    _, normals, areas = _triangles(fine)
    assert areas.max() <= 0.1 and areas.sum() == pytest.approx(4 + 0.16)
    assert np.allclose(normals, [0, 0, 1]) and np.bincount(source).tolist() == [64, 64, 1, 1]
    groups = _separate(fine.min(axis=1), fine.max(axis=1), 0.5)
    assert sorted(len(g) for g in groups) == [2, 128]
    assert len(_separate(fine.min(axis=1), fine.max(axis=1), 2.0)) == 1


def test_emissive_lamps_of_flat_and_closed_surfaces():
    # A dim panel and a small bright box, as bright as the panel overall
    dim, bright = 2.0, 50.0
    panel = quad([0, 0, 2], [1, 0, 0], [0, 0.5, 0])
    box = cube([3, 0, 1], 0.05)
    corners = np.concatenate([panel, box])
    radiance = np.repeat([[dim] * 3, [bright] * 3], [len(panel), len(box)], axis=0)
    lamps = _emissive_lamps(corners, radiance, tracer(corners), patches=16)
    panel_lamp, box_lamp = sorted(lamps, key=lambda lamp: lamp["position"][0])
    for lamp in lamps:
        EmissiveSurface.model_validate(lamp)

    # The panel emits from both sides, and is split into patches over it, from one of which it casts shadows
    assert emitted(panel_lamp) == pytest.approx(dim * 2 * math.pi * 2.0, rel=1e-6)
    assert len(panel_lamp["areas"]) > 4 and abs(panel_lamp["position"][2] - 2) < 1e-9
    assert np.allclose(np.average(panel_lamp["positions"], axis=0, weights=panel_lamp["areas"]), [0, 0, 2])
    # The box emits outwards only, and is split by the orientation of its faces, so that it shows exactly the area
    # of its faces seen from each direction, rather than as much in every direction. It casts shadows from its center,
    # which it surrounds
    assert emitted(box_lamp) == pytest.approx(bright * math.pi * 6 * 0.1**2, rel=0.02)
    assert np.allclose(box_lamp["position"], [3, 0, 1])
    surface = EmissiveSurface.model_validate(box_lamp)
    for direction in ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]):
        omega = T(direction) / T(direction).norm()
        seen = intensity(surface, omega[None])[0]
        assert seen.item() == pytest.approx(bright * 0.1**2 * omega.abs().sum().item(), rel=0.05)

    # Two boxes close to each other are a single lamp, which casts shadows from one of them rather than from between
    # them, where some other object could be
    pair = np.concatenate([cube([0, 0, 1], 0.05), cube([0.3, 0, 1], 0.05)])
    (lamp,) = _emissive_lamps(pair, np.full((len(pair), 3), dim), tracer(pair), patches=4)
    assert min(abs(lamp["position"][0]), abs(lamp["position"][0] - 0.3)) <= 0.05 + 1e-9

    # Light that surfaces block themselves, such as that of a box inside another, doesn't leave them
    inner = cube([3, 0, 1], 0.02)
    corners = np.concatenate([box, inner])
    (lamp,) = _emissive_lamps(corners, np.full((len(corners), 3), dim), tracer(corners), patches=16)
    assert emitted(lamp) == pytest.approx(dim * math.pi * 6 * 0.1**2, rel=0.02)


def test_emissive_lamps_are_gathered():
    # Four small panels far apart are four lamps, unless fewer are allowed, and a single small one is a single patch
    corners = np.concatenate([quad([x, 0, 2], [0.05, 0, 0], [0, 0.05, 0]) for x in (0, 2, 4, 6)])
    radiance = np.full((len(corners), 3), 1.0)
    assert len(_emissive_lamps(corners, radiance, tracer(corners))) == 4
    lamps = _emissive_lamps(corners, radiance, tracer(corners), lamps=2)
    assert len(lamps) == 2 and sorted(len(lamp["areas"]) for lamp in lamps) == [2, 2]
    assert sum(emitted(lamp) for lamp in lamps) == pytest.approx(4 * 2 * math.pi * 0.1**2)
    single = _emissive_lamps(corners[:2], radiance[:2], tracer(corners[:2]))
    assert len(single) == 1 and len(single[0]["areas"]) == 1
    # Its points spread over a square, whose root mean square distance to its center is its radius
    assert single[0]["spread"][0] == pytest.approx([0.1**2 / 12, 0.1**2 / 12, 0, 0, 0, 0], abs=1e-12)
    assert patch_radius(EmissiveSurface.model_validate(single[0]), 0) == pytest.approx(0.1 / math.sqrt(6))
    # Surfaces that don't emit are left out, and so are the faintest ones, as long as they make up at most 1% of the
    # light of all lamps, other lamps included
    assert _emissive_lamps(corners, np.zeros_like(radiance), tracer(corners)) == []
    faint = radiance.copy()
    faint[:2] = 1e-3
    assert len(_emissive_lamps(corners, faint, tracer(corners))) == 3
    assert _emissive_lamps(corners, radiance, tracer(corners), other_power=1e3) == []


def test_merged_patches():
    # Merged patches emit as much light in total, with the profile of all their faces, which is exact for flat ones
    normal = np.array([0.0, 0.6, 0.8])
    flat = symmetric(np.outer(normal, normal))
    lamp = EmissiveSurface(
        position=(0, 0, 1),
        positions=((0, 0, 1), (2, 0, 1), (4, 0, 1)),
        areas=(2.0, 0.5, 1.0),
        radiance=((3.0,), (1.0,), (2.0,)),
        orientation=(flat, flat, flat),
        spread=((0.1, 0.1, 0, 0, 0, 0),) * 3,
    )
    merged = merge_patches(lamp)
    directions = sphere_directions(4000)
    assert torch.allclose(intensity(merged, directions), intensity(lamp, directions), rtol=1e-9)
    assert merged.position == lamp.position and merged.areas == (3.5,)
    assert np.allclose(merged.positions[0], [(0 * 6 + 2 * 0.5 + 4 * 2) / 8.5, 0, 1])
    # Their spread includes that of the patches around the merged center
    offsets = np.array([0.0, 2.0, 4.0]) - merged.positions[0][0]
    assert merged.spread[0][0] == pytest.approx(0.1 + np.average(offsets**2, weights=[6, 0.5, 2]))


def test_emissive_lighting_round_trip():
    lamp = EmissiveSurface(
        position=(0, 0, 1),
        positions=((0, 0, 1),),
        areas=(1.0,),
        radiance=((1.0, 2.0, 3.0),),
        orientation=((0, 0, 0.5, 0, 0, 0),),
        facing=((0, 0, -0.5),),
    )
    lighting = Lighting(emissive=[lamp])
    assert Lighting.model_validate_json(lighting.model_dump_json()) == lighting and lighting.lamps == [lamp]
