import itertools
import math

import numpy as np
import pytest
import torch

from tests.test_medium import make_render
from visionsim.cli.medium import apply
from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, Sun, apply_medium, trace_shadows
from visionsim.medium.model import HeightFog
from visionsim.medium.occlusion import (
    Occlusion,
    ShadowMaps,
    _pack,
    _unpack,
    band_attenuation,
    cell_phases,
    load_occlusion,
    occluded_inscatter,
    sky_cell_directions,
    visibility,
)
from visionsim.medium.optics import height_fog_optical_depth, henyey_greenstein, sky_quadrature
from visionsim.medium.scattering import elevation_sources, tabulate_along_rays

FOG = Medium(extinction=0.08, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True)
SKY = (0.25, 0.35, 0.55)
ELEVATIONS = (0.0, 4.0, 9.0, 16.0, 26.0, 40.0, 60.0, 90.0)
CELLS = (24, 24, 20, 16, 12, 8, 3)
EDGES = tuple(math.sin(math.radians(e)) for e in ELEVATIONS)


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def basis(direction):
    """Orthonormal axes whose third one points along a direction, as the rows of a matrix."""
    z = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def render_maps(directions, bounds, first_hit, resolution=256):
    """Shadow maps of analytic occluders, laid out as by ``BlenderService.save_occlusion``, where ``first_hit`` gives
    the distance at which rays starting at points of shape (..., 3) and going along a direction hit something."""
    bounds = np.asarray(bounds, dtype=float)
    center = bounds.mean(axis=0)
    corners = np.asarray(list(itertools.product(*bounds.T)))
    radius = np.linalg.norm(corners - center, axis=1).max()
    depths, offsets, shapes, origins, axes, texels = [np.zeros(0)], [], [], [], [], []
    for direction in directions:
        rows = basis(direction)
        extent = np.abs((corners - center) @ rows[:2].T).max(axis=0)
        texel = 2 * extent.max() / resolution
        width, height = (max(1, int(np.ceil(2 * e / texel - 1e-6))) for e in extent)
        origin = center + rows[2] * (radius + 1)
        a = (np.arange(width) + 0.5 - width / 2) * texel
        b = (np.arange(height) + 0.5 - height / 2) * texel
        start = origin + a[None, :, None] * rows[0] + b[:, None, None] * rows[1]
        offsets.append(sum(d.size for d in depths))
        depths.append(first_hit(start, -rows[2]))
        shapes.append((height, width))
        origins.append(origin)
        axes.append(rows)
        texels.append(texel)
    return ShadowMaps(
        depths=T(np.concatenate([d.reshape(-1) for d in depths])),
        offsets=torch.as_tensor(offsets, dtype=torch.long),
        shapes=torch.as_tensor(shapes, dtype=torch.long).reshape(-1, 2),
        origins=T(origins).reshape(-1, 3),
        axes=T(axes).reshape(-1, 3, 3),
        texels=T(texels),
    )


def rectangle(height, x, y):
    """First hit of rays on a horizontal rectangle, or a huge distance if they miss it."""

    def first_hit(start, direction):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (height - start[..., 2]) / direction[..., 2]
            hit = start + t[..., None] * direction
        inside = (t > 0) & (hit[..., 0] >= x[0]) & (hit[..., 0] <= x[1]) & (hit[..., 1] >= y[0]) & (hit[..., 1] <= y[1])
        return np.where(inside, t, 1e10)

    return first_hit


def wall(position, y, z):
    """First hit of rays on a vertical wall at a given x, or a huge distance if they miss it."""

    def first_hit(start, direction):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (position - start[..., 0]) / direction[..., 0]
            hit = start + t[..., None] * direction
        inside = (t > 0) & (hit[..., 1] >= y[0]) & (hit[..., 1] <= y[1]) & (hit[..., 2] >= z[0]) & (hit[..., 2] <= z[1])
        return np.where(inside, t, 1e10)

    return first_hit


def occlusion_of(sun_maps, sun_directions, sky_maps, bounds):
    return Occlusion(
        sun_maps=sun_maps,
        sun_directions=T(sun_directions).reshape(-1, 3),
        sky_maps=sky_maps,
        sky_edges=EDGES,
        sky_counts=CELLS,
        bounds=T(bounds),
    )


def rays(n=7, azimuths=(-30, 30)):
    """Rays from 1.6m above the ground towards +y, from slightly down to well above the horizon."""
    elevations = np.radians(np.linspace(-3, 40, n))
    azimuths = np.radians(np.linspace(*azimuths, n))
    directions = np.stack(
        [np.cos(elevations) * np.sin(azimuths), np.cos(elevations) * np.cos(azimuths), np.sin(elevations)], -1
    )
    distance = np.where(directions[:, 2] < 0, -1.6 / np.minimum(directions[:, 2], -1e-9), np.inf)
    return T([0.0, 0.0, 1.6]), T(directions), T(distance)


def test_visibility_below_a_rectangle():
    sun = np.array([0.3, 0.2, 0.9])
    sun /= np.linalg.norm(sun)
    bounds = [[-1.0, -1.0, 5.0], [1.0, 1.0, 5.0]]
    maps = render_maps([sun], bounds, rectangle(5.0, (-1, 1), (-1, 1)), resolution=200)
    # Points straight below the rectangle along the sun's direction are in its shadow, others are lit
    below = np.array([0.0, 0.0, 5.0]) - np.outer([1.0, 3.0, 4.5], sun)
    beside = below + np.array([2.5, 0.0, 0.0])
    above = np.array([[0.0, 0.0, 6.0]])
    points = T(np.concatenate([below, beside, above]))
    origin = T([0.0, -20.0, 1.0])
    offset = points - origin
    distances = offset.norm(dim=-1)
    for filtered in (True, False):
        lit = visibility(maps, origin, offset / distances[:, None], distances[:, None], filtered=filtered)
        assert lit[:, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_cell_phases_add_up_to_the_phase_function():
    directions = T([[0.0, 1.0, 0.0], [0.6, 0.0, 0.8], [0.0, -0.8, -0.6], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    for g in (0.0, 0.6, 0.85):
        above = cell_phases(EDGES, CELLS, g, directions).sum(dim=-1)
        below = cell_phases(EDGES, CELLS, g, directions, below=True).sum(dim=-1)
        expected = sky_quadrature(directions, g)[1].sum(dim=-1)
        assert torch.allclose(above, expected, rtol=2e-3, atol=1e-4)
        assert torch.allclose(above + below, torch.ones_like(above), rtol=2e-3)


def test_band_attenuation():
    depth = T([1e-5, 0.05, 0.2, 1.0, 5.0])
    result = band_attenuation(EDGES, depth)
    mu = np.linspace(0, 1, 200_001)[1:]
    for b, (low, high) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        band = mu[(mu >= low) & (mu <= high)]
        expected = np.exp(-depth.numpy()[:, None] / band).mean(axis=-1)
        assert np.allclose(result[:, b].numpy(), expected, rtol=1e-2, atol=1e-6)


def save_occlusion(path, sun, sun_directions, sky, bounds, counts=CELLS):
    """Save shadow maps as ``BlenderService.save_occlusion`` does."""
    arrays = {
        f"{kind}_{name}": getattr(maps, name).numpy()
        for kind, maps in (("sun", sun), ("sky", sky))
        for name in ShadowMaps._fields
    }
    common = {"sun_directions": sun_directions, "sky_edges": EDGES, "bounds": bounds}
    np.savez(path, **arrays, **common, sky_counts=counts)


def test_load_occlusion(tmp_path):
    directions = sky_cell_directions(EDGES, CELLS)
    bounds = [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]
    sky = render_maps(directions, bounds, rectangle(3.0, (-5, 5), (-5, 5)), resolution=16)
    sun = render_maps([[0.0, 0.6, 0.8]], bounds, rectangle(3.0, (-5, 5), (-5, 5)), resolution=32)
    path = tmp_path / "occlusion.npz"
    save_occlusion(path, sun, [[0.0, 0.6, 0.8]], sky, bounds)
    occlusion = load_occlusion(path, dtype=torch.float64)
    assert occlusion.sky_counts == CELLS
    assert torch.equal(occlusion.sky_maps.depths, sky.depths)

    save_occlusion(path, sun, [[0.0, 0.6, 0.8]], sky, bounds, counts=(12, *CELLS[1:]))
    with pytest.raises(ValueError, match="do not match"):
        load_occlusion(path)


def test_cli_apply_with_occlusion(tmp_path):
    """``medium.apply`` uses the shadow maps saved alongside renders, for media that attenuate sunlight."""
    lighting = make_render(tmp_path / "render", n=1)
    sun = np.array(lighting.suns[0].direction) / np.linalg.norm(lighting.suns[0].direction)
    bounds = [[-50.0, -50.0, 6.0], [50.0, 50.0, 6.0]]
    roof = rectangle(6.0, (-50, 50), (-50, 50))
    sky = render_maps(sky_cell_directions(EDGES, CELLS), bounds, roof, 16)
    save_occlusion(tmp_path / "render" / "occlusion.npz", render_maps([sun], bounds, roof, 64), [sun], sky, bounds)
    medium = Medium(extinction=0.1, components=[HeightFog(density=1.0, falloff=5.0)], sun_attenuation=True)
    (tmp_path / "medium.json").write_text(medium.model_dump_json())
    apply(tmp_path / "render", tmp_path / "fog", tmp_path / "medium.json", device="cpu")

    (radiance, transform), (depth, _) = (
        Dataset.from_path(tmp_path / "render" / name)[0] for name in ("frames", "depths")
    )
    occlusion = load_occlusion(tmp_path / "render" / "occlusion.npz", dtype=torch.float64)
    pose = transform["transform_matrix"]
    expected = apply_medium(radiance, depth, transform, pose, medium, lighting, occlusion=occlusion)
    unoccluded = apply_medium(radiance, depth, transform, pose, medium, lighting)
    frame, _ = Dataset.from_path(tmp_path / "fog" / "frames")[0]
    assert np.allclose(frame, expected.radiance.numpy(), rtol=1e-6, atol=1e-7)
    assert not np.allclose(frame, unoccluded.radiance.numpy(), rtol=1e-3)

    # Media that don't attenuate sunlight can't be shadowed, and ignore the shadow maps
    (tmp_path / "medium.json").write_text(Medium(extinction=0.1).model_dump_json())
    apply(tmp_path / "render", tmp_path / "plain", tmp_path / "medium.json", device="cpu")


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"))],
)
def test_without_occluders_shadows_change_nothing(device):
    """Shadow maps that record nothing leave the closed form unchanged, for every source of light."""
    medium = FOG.model_copy(update={"multiple_scattering": True})
    lighting = Lighting(sky=SKY, suns=[Sun(direction=(0.4, 0.7, 0.58), irradiance=(4.0,))], ground_albedo=(0.3,) * 3)
    bounds = [[-10.0, 5.0, 0.0], [10.0, 50.0, 6.0]]
    nothing = lambda start, direction: np.full(start.shape[:-1], 1e10)
    sun = np.array(lighting.suns[0].direction) / np.linalg.norm(lighting.suns[0].direction)
    occlusion = occlusion_of(
        render_maps([sun], bounds, nothing, 64),
        [sun],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, nothing, 8),
        bounds,
    )
    camera = {"w": 24, "h": 16, "fl_x": 12.0, "fl_y": 12.0, "cx": 12.0, "cy": 8.0}
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # looking along +y
    pose[:3, 3] = [0.0, 0.0, 1.6]
    depth = np.full((16, 24), 30.0)
    depth[:4] = 1e10  # the top rows see the sky
    radiance = np.full((16, 24, 3), 0.2)
    expected = apply_medium(radiance, depth, camera, pose, medium, lighting, device=device)
    result = apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion, device=device)
    assert torch.allclose(result.radiance, expected.radiance, rtol=1e-10, atol=1e-12)


def test_pack_bits():
    bits = torch.rand(3, 5, 19) > 0.5
    packed = _pack(bits)
    assert packed.dtype == torch.uint8 and packed.shape == (3, 5, 3)
    assert torch.equal(_unpack(packed, 19), bits.to(torch.uint8))


def demo_scene():
    """Fog in front of a camera looking along +y at a box, lit by a low sun and the sky, with shadow maps."""
    lighting = Lighting(sky=SKY, suns=[Sun(direction=(0.6, -0.3, 0.5), irradiance=(4.0,))], ground_albedo=(0.3,) * 3)
    sun = np.array(lighting.suns[0].direction) / np.linalg.norm(lighting.suns[0].direction)
    bounds = [[-2.0, 10.0, 0.0], [2.0, 14.0, 4.0]]
    roof = rectangle(4.0, (-2, 2), (10, 14))
    occlusion = occlusion_of(
        render_maps([sun], bounds, roof, 256),
        [sun],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, roof, 64),
        bounds,
    )
    camera = {"w": 24, "h": 16, "fl_x": 12.0, "fl_y": 12.0, "cx": 12.0, "cy": 8.0}
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # looking along +y
    pose[:3, 3] = [0.0, 0.0, 1.6]
    depth = np.full((16, 24), 30.0)
    depth[:4] = 1e10  # the top rows see the sky
    return occlusion, lighting, camera, pose, depth, np.full((16, 24, 3), 0.2)


def test_reused_shadows():
    """Shadows traced once give the same frame as tracing them along with the medium, and most of the shadows of media
    of similar density."""
    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    shadows = trace_shadows(occlusion, depth, camera, pose, medium, lighting)
    traced = apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion)
    reused = apply_medium(radiance, depth, camera, pose, medium, lighting, shadows=shadows)
    assert torch.allclose(reused.radiance, traced.radiance, rtol=1e-10, atol=1e-12)
    unoccluded = apply_medium(radiance, depth, camera, pose, medium, lighting)
    assert not torch.allclose(reused.inscatter, unoccluded.inscatter, rtol=1e-2)

    # Fog that's a bit denser scatters light from about the same places, so the same shadows mostly fit it, but the
    # samples of the sky's visibility were placed for the other fog, here about 10% of the shadows' effect
    denser = medium.model_copy(update={"extinction": 1.5 * medium.extinction})
    own = apply_medium(radiance, depth, camera, pose, denser, lighting, occlusion=occlusion)
    borrowed = apply_medium(radiance, depth, camera, pose, denser, lighting, shadows=shadows)
    shadowed = (apply_medium(radiance, depth, camera, pose, denser, lighting).inscatter - own.inscatter).abs().sum()
    assert (borrowed.inscatter - own.inscatter).abs().sum() < 0.2 * shadowed

    with pytest.raises(ValueError, match="traced along"):
        apply_medium(radiance[:8], depth[:8], camera | {"h": 8}, pose, medium, lighting, shadows=shadows)


def test_occlusion_requires_attenuated_sunlight():
    medium = Medium(extinction=0.08, components=[HeightFog(density=1.0, falloff=2.5)])
    bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    maps = render_maps([[0.0, 0.0, 1.0]], bounds, rectangle(1.0, (0, 1), (0, 1)), resolution=4)
    occlusion = occlusion_of(maps, [[0.0, 0.0, 1.0]], maps, bounds)
    camera = {"w": 2, "h": 2, "fl_x": 1.0, "fl_y": 1.0, "cx": 1.0, "cy": 1.0}
    with pytest.raises(ValueError, match="sun_attenuation"):
        apply_medium(np.zeros((2, 2, 3)), np.ones((2, 2)), camera, np.eye(4), medium, Lighting(), occlusion=occlusion)


def brute_force(origin, directions, distance, source, steps=4000, max_depth=14.0):
    """Integral of ``σ(s) · T(s) · J(s)`` along rays, with a fine midpoint rule over the optical depth."""
    fog = FOG.components[0]
    beta = FOG.extinction
    k = fog.density * math.exp(-float(origin[2]) / fog.falloff)
    total = []
    for v, d in zip(directions, distance):
        tau_end = min(float(height_fog_optical_depth(fog, origin, v, d)) * beta, max_depth)
        tau = (np.arange(steps) + 0.5) / steps * tau_end
        # Distance along the ray at each optical depth, inverting the height fog's optical depth
        v_z = float(v[2])
        if abs(v_z) < 1e-12:
            s = tau / beta / k
        else:
            s = -fog.falloff / v_z * np.log1p(-np.minimum(tau / beta * v_z / (k * fog.falloff), 1 - 1e-12))
        points = origin.numpy() + s[:, None] * v.numpy()
        total.append((np.exp(-tau) * source(points, v.numpy())).sum() * tau_end / steps)
    return np.asarray(total)


def test_sun_shadow_of_a_rectangle():
    """A rectangle floating above the camera casts a shadow, which matches a brute-force integral."""
    sun = np.array([0.2, 0.5, 0.84])
    sun /= np.linalg.norm(sun)
    bounds = [[-4.0, 10.0, 4.0], [4.0, 30.0, 4.0]]
    occluder = rectangle(4.0, (-4, 4), (10, 30))
    occlusion = occlusion_of(
        render_maps([sun], bounds, occluder, 1024), [sun], render_maps([], bounds, occluder), bounds
    )
    origin, directions, distance = rays()
    beta = T([FOG.extinction])
    result = occluded_inscatter(
        occlusion, FOG, origin, directions, distance, beta, [(T(sun), T([1.0]))], {}, sun_step=1.0, bias=1.0
    )
    fog = FOG.components[0]

    def source(points, v, shadows=True):
        c = fog.density * np.exp(-points[:, 2] / fog.falloff) * fog.falloff * FOG.extinction
        lit = occluder(points, sun) >= 1e9 if shadows else 1.0
        phase = float(henyey_greenstein(T(v) @ T(sun), FOG.anisotropy))
        return phase * np.exp(-c / sun[2]) * lit

    expected = brute_force(origin, directions, distance, source)
    unoccluded = brute_force(origin, directions, distance, lambda p, v: source(p, v, shadows=False))
    assert (expected < 0.95 * unoccluded).sum() >= 2  # some rays do cross the shadow
    assert np.allclose(result[:, 0].numpy(), expected, rtol=5e-3, atol=1e-6)


def sky_table(beta):
    fog = FOG.components[0]
    source = elevation_sources(FOG, Lighting(sky=(1.0,)), beta, 1)["sky"]
    return tabulate_along_rays(source, beta * fog.density * math.exp(-1.6 / fog.falloff) * fog.falloff)


def test_sky_hidden_by_a_wall():
    """A wall beside the rays hides part of the sky, as a brute-force integral finds with the same cells of the sky,
    each of which is lit or not as its central direction is."""
    bounds = [[2.0, -40.0, 0.0], [2.0, 80.0, 20.0]]
    occluder = wall(2.0, (-40, 80), (0, 20))
    occlusion = occlusion_of(
        render_maps([], bounds, occluder),
        [],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 512),
        bounds,
    )
    origin, directions, distance = rays(azimuths=(-30, 0))  # which stay on the same side of the wall
    beta = T([FOG.extinction])
    result = occluded_inscatter(occlusion, FOG, origin, directions, distance, beta, [], {"sky": sky_table(beta)})
    fog = FOG.components[0]
    centers = sky_cell_directions(EDGES, CELLS)
    bands = np.repeat(np.arange(len(CELLS)), CELLS)

    def source(points, v):
        c = fog.density * np.exp(-points[:, 2] / fog.falloff) * fog.falloff * FOG.extinction
        phases = cell_phases(EDGES, CELLS, FOG.anisotropy, T(v)[None])[0].numpy()
        weights = phases * band_attenuation(EDGES, T(c)).numpy()[:, bands]
        lit = np.stack([occluder(np.broadcast_to(p, centers.shape), centers) >= 1e9 for p in points])
        # The light of the cells weighs the unoccluded source, which the table integrates more finely
        unoccluded = sky_source_at(points, v)
        return unoccluded * (weights * lit).sum(axis=-1) / weights.sum(axis=-1)

    expected = brute_force(origin, directions, distance, source, steps=400)
    unoccluded = brute_force(origin, directions, distance, sky_source_at, steps=400)
    assert (expected < 0.9 * unoccluded).sum() >= 3  # the wall hides part of the sky
    assert np.allclose(result[:, 0].numpy(), expected, rtol=0.01)


def sky_source_at(points, v):
    fog = FOG.components[0]
    c = fog.density * np.exp(-points[:, 2] / fog.falloff) * fog.falloff * FOG.extinction
    mu, weights = sky_quadrature(T(v)[None], FOG.anisotropy)
    return (weights.numpy() * np.exp(-c[:, None] / np.maximum(mu.numpy(), 1e-30))).sum(axis=-1)


def test_sky_hidden_by_a_roof():
    """Below a roof, the sky is entirely hidden, so only the medium beyond the roof scatters skylight."""
    bounds = [[-100.0, -100.0, 6.0], [100.0, 400.0, 6.0]]
    occluder = rectangle(6.0, (-100, 100), (-100, 400))
    occlusion = occlusion_of(
        render_maps([], bounds, occluder),
        [],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 1024),
        bounds,
    )
    origin, directions, distance = rays()
    beta = T([FOG.extinction])
    result = occluded_inscatter(occlusion, FOG, origin, directions, distance, beta, [], {"sky": sky_table(beta)})

    # Directions over the upper hemisphere, which are hidden when they hit the roof
    theta = (np.arange(150) + 0.5) / 150 * (np.pi / 2)
    phi = (np.arange(360) + 0.5) / 360 * 2 * np.pi
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    omega = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1).reshape(-1, 3)
    solid = (np.sin(theta) * (np.pi / 2 / 150) * (2 * np.pi / 360)).reshape(-1)
    fog = FOG.components[0]

    def source(points, v):
        c = fog.density * np.exp(-points[:, 2] / fog.falloff) * fog.falloff * FOG.extinction
        phase = henyey_greenstein(T(omega @ v), FOG.anisotropy).numpy() * solid
        lit = np.stack([occluder(np.broadcast_to(p, omega.shape), omega) >= 1e9 for p in points])
        return (phase * lit * np.exp(-c[:, None] / omega[None, :, 2])).sum(axis=-1)

    expected = brute_force(origin, directions, distance, source, steps=200)
    unoccluded = brute_force(origin, directions, distance, sky_source_at, steps=200)
    assert (expected < 0.5 * unoccluded).sum() >= 3  # the roof hides most of the sky from rays under it
    # Under the roof, only directions close to the horizon escape, through cells that are partly hidden
    assert np.allclose(unoccluded - result[:, 0].numpy(), unoccluded - expected, rtol=0.01)
