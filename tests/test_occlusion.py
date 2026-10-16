import functools
import gc
import itertools
import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

from tests.test_medium import make_render
from visionsim.cli.medium import apply
from visionsim.dataset import Dataset
from visionsim.medium import Effects, Lighting, Medium, Sun, apply_medium, render, scattering, surfaces, trace_shadows
from visionsim.medium import occlusion as occlusion_module
from visionsim.medium.halos import _HIT_DEPTH
from visionsim.medium.model import HeightFog, PointLight
from visionsim.medium.occlusion import (
    _DERIVED,
    _KEPT,
    Occlusion,
    Occupancy,
    ShadowMaps,
    _cell_centers,
    _depth_windows,
    _ground_shadowed,
    _object_light,
    _opposite_cells,
    _pack,
    _phase_sums,
    _ragged,
    _ragged_chunks,
    _sky_cells,
    _snap,
    _surely_lit,
    _surface_normals,
    _unpack,
    band_attenuation,
    band_radiance,
    cell_phases,
    cell_profile,
    hit_distances,
    lamp_shadow,
    lamp_visibility,
    load_occlusion,
    maps_to,
    object_reflections,
    occluded_inscatter,
    occlusion_to,
    occupancy,
    occupied,
    shade,
    sky_cell_directions,
    trace,
    visibility,
)
from visionsim.medium.optics import (
    height_fog_optical_depth,
    henyey_greenstein,
    point_light_inscatter,
    sky_quadrature,
)
from visionsim.medium.render import camera_rays, lamp_inscatter
from visionsim.medium.scattering import (
    cached_multiple_scattering,
    elevation_sources,
    environment_lit,
    tabulate_along_rays,
)

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
    """Shadows traced once give the same frame as tracing them along with the medium, given the shadow maps for the
    light that objects reflect, and most of the shadows of media of similar density."""
    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    shadows = trace_shadows(occlusion, depth, camera, pose, medium, lighting)
    traced = apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion)
    reused = apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion, shadows=shadows)
    assert torch.allclose(reused.radiance, traced.radiance, rtol=1e-10, atol=1e-12)
    unoccluded = apply_medium(radiance, depth, camera, pose, medium, lighting)
    assert not torch.allclose(reused.inscatter, unoccluded.inscatter, rtol=1e-2)
    # Without the maps, the roof reflects nothing into the fog
    unreflected = apply_medium(radiance, depth, camera, pose, medium, lighting, shadows=shadows)
    assert (unreflected.inscatter <= reused.inscatter + 1e-12).all()
    assert (unreflected.inscatter < reused.inscatter - 1e-6).any()

    # Fog that's a bit denser scatters light from about the same places, so the same shadows mostly fit it, but the
    # samples of the sky's visibility, where the ground's light is darkened by the sun's shadow and where the light that
    # objects reflect is gathered, were placed for the other fog, here about a fifth of the shadows' effect
    denser = medium.model_copy(update={"extinction": 1.5 * medium.extinction})
    own = apply_medium(radiance, depth, camera, pose, denser, lighting, occlusion=occlusion)
    borrowed = apply_medium(radiance, depth, camera, pose, denser, lighting, occlusion=occlusion, shadows=shadows)
    shadowed = (apply_medium(radiance, depth, camera, pose, denser, lighting).inscatter - own.inscatter).abs().sum()
    assert (borrowed.inscatter - own.inscatter).abs().sum() < 0.25 * shadowed

    with pytest.raises(ValueError, match="traced along"):
        apply_medium(radiance[:8], depth[:8], camera | {"h": 8}, pose, medium, lighting, shadows=shadows)

    # Shadows traced without the visibility of each cell only serve media of the same anisotropy
    origin, directions, scale = camera_rays(camera, pose, dtype=torch.float64)
    rays, lengths = directions.reshape(-1, 3), T(depth).reshape(-1) * scale.reshape(-1)
    lengths = torch.where(lengths > 1e9, torch.inf, lengths)
    lean = trace(occlusion, medium, origin, rays, lengths, [], ground_height=0.0, keep_cells=False)
    full = trace(occlusion, medium, origin, rays, lengths, [], ground_height=0.0)
    assert lean.visible.shape[-1] == 0 and full.visible.shape[-1] > 0
    beta = T([medium.extinction])
    tables = {"sky": sky_table(beta)}
    assert torch.allclose(
        shade(lean, medium, origin, rays, lengths, beta, [], tables),
        shade(full, medium, origin, rays, lengths, beta, [], tables),
    )
    other = medium.model_copy(update={"anisotropy": 0.5})
    shade(full, other, origin, rays, lengths, beta, [], tables)
    with pytest.raises(ValueError, match="anisotropy"):
        shade(lean, other, origin, rays, lengths, beta, [], tables)


def test_cells_towards_the_sun_bring_more_scattered_light():
    """Light scattered at least once arrives brighter from the cells of the sky towards the sun: for light scattered
    once, as much as the phase function averaged over each cell, relative to the mean of its band, which the profile of
    the cells is on average, and without suns, it doesn't vary."""
    sun = Sun(direction=(0.6, -0.3, 0.5), irradiance=(4.0,))
    towards = np.asarray(sun.direction) / np.linalg.norm(sun.direction)
    beta = T([FOG.extinction])
    # Light scattered once only, from the sun alone, with enough harmonics to be exact
    table = scattering.multiple_scattering(FOG, Lighting(suns=[sun]), beta, 1, higher_orders=False, modes=48)
    profile = cell_profile(table, EDGES, CELLS)
    assert profile is not None and profile.shape == (2, sum(CELLS))
    bands = np.repeat(np.arange(len(CELLS)), CELLS)
    for b in range(len(CELLS)):
        assert torch.allclose(profile[:, bands == b].mean(dim=1), T([1.0, 1.0]), atol=1e-9)
    # The phase function between the sun and directions spread over each cell, times the light of the sun scattered
    # once along the gathering rays of the cell's elevations, which the mean over azimuths gives
    t, w = np.polynomial.legendre.leggauss(8)
    phi = (np.arange(64) + 0.5) / 64
    gathered = table.gathered.sum(dim=(0, -1)).numpy()
    directions = table.directions.numpy()
    for sign, row in ((1.0, 0), (-1.0, 1)):
        expected = []
        for low, high, count in zip(EDGES[:-1], EDGES[1:], CELLS):
            mu = sign * ((low + high) / 2 + (high - low) / 2 * t)
            along = np.interp(mu, directions, gathered)
            mean = scattering.azimuthal_phase(T(mu), T([towards[2]]), FOG.anisotropy).numpy() / (2 * np.pi)
            for j in range(count):
                azimuths = 2 * np.pi * (j + phi) / count - math.atan2(towards[1], towards[0])
                cos = mu[:, None] * towards[2] + np.sqrt(1 - mu[:, None] ** 2) * np.hypot(*towards[:2]) * np.cos(
                    azimuths
                )
                phase = henyey_greenstein(T(cos), FOG.anisotropy).numpy().mean(axis=1)
                expected.append(((w / 2) * along * phase / mean).sum())
        expected = np.asarray(expected)
        for b in range(len(CELLS)):
            expected[bands == b] /= expected[bands == b].mean()
        assert np.allclose(profile[row].numpy(), expected, rtol=0.01, atol=1e-3)
        # Towards the sun, brighter than away from it
        low = bands == 0
        relative = np.radians(360 * (np.arange(CELLS[0]) + 0.5) / CELLS[0]) - math.atan2(towards[1], towards[0])
        closest = np.argmin(np.abs(np.angle(np.exp(1j * relative))))
        assert profile[row, low][closest] == profile[row, low].max() and profile[row, low].max() > 1.5
    # Light scattered more than once still follows the sun, less so
    table = cached_multiple_scattering(FOG, Lighting(sky=SKY, suns=[sun], ground_albedo=(0.3,)), beta, 3)
    both = cell_profile(table, EDGES, CELLS)
    assert both is not None and 1 < both[0, :24].max() < profile[0, :24].max()
    # Without suns, cells of a band all bring the same light
    assert cell_profile(cached_multiple_scattering(FOG, Lighting(sky=SKY), beta, 3), EDGES, CELLS) is None


def test_cells_towards_the_sun_hide_more_scattered_light():
    """Weighted by the light they bring, cells of the sky that a wall hides towards the sun hide more of the light
    scattered more than once than away from it, while equal weights change nothing, and the sums of the weighted cells
    traced along rays serve media of the same anisotropy as weighting cells again for another anisotropy does."""
    bounds = [[2.0, -40.0, 0.0], [2.0, 80.0, 20.0]]
    occluder = wall(2.0, (-40, 80), (0, 20))
    origin, directions, distance = rays(azimuths=(-30, 0))
    medium = FOG.model_copy(update={"multiple_scattering": True})
    beta = T([medium.extinction])
    fog = medium.components[0]
    depth_above = beta * fog.density * math.exp(-1.6 / fog.falloff) * fog.falloff
    scattered = {}
    for side in (1.0, -1.0):
        sun = Sun(direction=(0.8 * side, 0.0, 0.6), irradiance=(4.0,))
        lighting = Lighting(sky=SKY, suns=[sun], ground_albedo=(0.3,))
        towards = np.asarray(sun.direction) / np.linalg.norm(sun.direction)
        occlusion = occlusion_of(
            render_maps([towards], bounds, occluder),
            [towards],
            render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 256),
            bounds,
        )
        table = cached_multiple_scattering(medium, lighting, beta, 3)
        tables = {
            name: value
            for name, value in scattering.cached_camera_tables(medium, lighting, beta, 3, depth_above).items()
            if name.startswith("scattered")
        }
        profile = cell_profile(table, EDGES, CELLS)
        assert profile is not None
        args = (origin, directions, distance)
        plain = trace(occlusion, medium, *args, [], keep_cells=False)
        even = trace(occlusion, medium, *args, [], profile=torch.ones_like(profile), keep_cells=False)
        weighted = trace(occlusion, medium, *args, [], profile=profile, keep_cells=False)
        assert plain.scattered is None and even.scattered is not None
        for ours, theirs in zip(
            even.scattered[1:], (plain.visible_weights, plain.weights, plain.visible_weights_below, plain.weights_below)
        ):
            assert torch.equal(ours, theirs)
        shading = (medium, *args, beta, [], tables, table)
        assert torch.equal(shade(even, *shading), shade(plain, *shading))
        scattered[side] = shade(plain, *shading), shade(weighted, *shading)
        # Weighting again cells traced for another anisotropy
        other = trace(occlusion, medium.model_copy(update={"anisotropy": 0.5}), *args, [], profile=profile)
        assert torch.allclose(shade(other, *shading, profile=profile), scattered[side][1], rtol=1e-9)
    towards_wall, away = scattered[1.0], scattered[-1.0]
    hidden = towards_wall[0] < 0.99 * scattered[-1.0][0]  # rays whose scattered light the wall dims at all
    assert hidden.sum() >= 2
    assert (towards_wall[1] < towards_wall[0])[hidden].all() and (away[1] > away[0])[hidden].all()


def test_daylight_reflected_by_objects_can_be_left_out():
    """Without the light of suns and the sky that objects reflect, the frame is that of black objects, by day, and
    darker than with it."""
    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    reflected = apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion)
    effects = Effects(reflected_daylight=False)
    left_out = apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion, effects=effects)
    black = lighting.model_copy(update={"object_albedo": (0.0, 0.0, 0.0)})
    unlit = apply_medium(radiance, depth, camera, pose, medium, black, occlusion=occlusion)
    assert torch.equal(left_out.inscatter, unlit.inscatter)
    assert (left_out.inscatter <= reflected.inscatter).all()
    assert (left_out.inscatter < reflected.inscatter - 1e-6).any()


def test_daylight_reflected_by_objects_is_scattered_forward():
    """With multiple scattering, the light that objects reflect keeps the light that the medium scatters forward on its
    way, through the reduced extinction of the similarity relation, which brightens it, while it is scattered once
    only otherwise, as in a medium that scatters light once only."""
    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    black = lighting.model_copy(update={"object_albedo": (0.0, 0.0, 0.0)})
    args = (radiance, depth, camera, pose)
    unlit = apply_medium(*args, medium, black, occlusion=occlusion).inscatter
    forward = apply_medium(*args, medium, lighting, occlusion=occlusion).inscatter - unlit
    effects = Effects(scattered_daylight_reflections=False)
    once = apply_medium(*args, medium, lighting, occlusion=occlusion, effects=effects).inscatter - unlit
    assert (forward >= once - 1e-12).all() and (forward > 1.05 * once + 1e-9).any()
    # Through the reduced extinction
    origin, directions, scale = camera_rays(camera, pose, dtype=torch.float64)
    rays, lengths = directions.reshape(-1, 3), T(depth).reshape(-1) * scale.reshape(-1)
    shadows = trace(occlusion, medium, origin, rays, torch.where(lengths > 1e9, torch.inf, lengths), [])
    beta = T([medium.extinction])

    def surfaces(points, normals):
        return torch.ones_like(points[:, :1])

    reduced = beta * (1 - medium.albedo * medium.anisotropy)  # the similarity relation
    assert torch.equal(
        object_reflections(occlusion, medium, origin, rays, shadows, surfaces, beta, similarity=True),
        object_reflections(occlusion, medium, origin, rays, shadows, surfaces, reduced),
    )
    # The switch needs multiple scattering
    plain = FOG.model_copy(update={"multiple_scattering": False})
    assert torch.equal(
        apply_medium(*args, plain, lighting, occlusion=occlusion).inscatter,
        apply_medium(*args, plain, lighting, occlusion=occlusion, effects=effects).inscatter,
    )


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

    # Longer intervals are only sampled every texel where the shadow's edge may cross them, which is as accurate
    coarse = occluded_inscatter(occlusion, FOG, origin, directions, distance, beta, [(T(sun), T([1.0]))], {}, bias=1.0)
    assert np.allclose(coarse[:, 0].numpy(), expected, rtol=5e-3, atol=1e-6)


def test_surely_lit_intervals_are():
    """Intervals along rays found to be entirely lit or hidden from tiles of a shadow map are so at every point."""
    sun = np.array([0.2, 0.5, 0.84])
    sun /= np.linalg.norm(sun)
    bounds = [[-4.0, 10.0, 4.0], [4.0, 30.0, 4.0]]
    maps = render_maps([sun], bounds, rectangle(4.0, (-4, 4), (10, 30)), 256)
    generator = np.random.default_rng(0)
    origin = T([0.0, 0.0, 1.6])
    directions = T(generator.normal(size=(4000, 3)) * [1.0, 1.0, 0.3] + [0.0, 1.0, 0.3])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    texel = float(maps.texels[0])
    starts = T(generator.uniform(0, 40, 4000))
    ends = starts + T(generator.uniform(0, 8 * texel, 4000))
    lit = _surely_lit(maps, _depth_windows(maps, 10), 10, origin, directions, starts, ends, 0.5)
    points = torch.lerp(starts[:, None], ends[:, None], torch.linspace(0, 1, 65, dtype=torch.float64))
    along = visibility(maps, origin, directions, points)[..., 0]
    assert (lit == 1).sum() > 100 and (lit == 0).sum() > 100 and (lit < 0).sum() > 10
    assert (along[lit == 1] == 1).all() and (along[lit == 0] == 0).all()


def test_snapping_to_a_grid():
    generator = np.random.default_rng(1)
    low, high = T([-5.0, -3.0, 0.0]), T([5.0, 7.0, 2.0])
    points = T(generator.uniform(low.numpy(), high.numpy(), size=(1000, 3)))
    cell = _snap(points, low, high, 0.3)
    assert ((_cell_centers(cell, low, high, 0.3) - points).abs() <= 0.15 + 1e-12).all()
    # Points outside of the box go to its closest cell
    outside = _snap(T([[-50.0, 0.0, 1.0], [float("inf"), 0.0, 1.0]]), low, high, 0.3)
    centers = _cell_centers(outside, low, high, 0.3)
    assert centers[:, 0].tolist() == pytest.approx([-5.0 + 0.5 * 0.3, -5.0 + 33.5 * 0.3])  # the box has 34 cells


def test_ragged_layout():
    counts = torch.as_tensor([2, 0, 3, 1])
    ray, index = _ragged(counts, 6)
    assert ray.tolist() == [0, 0, 2, 2, 2, 3] and index.tolist() == [0, 1, 0, 1, 2, 0]
    assert list(_ragged_chunks(counts, 3)) == [(slice(0, 2), 2), (slice(2, 3), 3), (slice(3, 4), 1)]
    assert list(_ragged_chunks(torch.as_tensor([5, 1]), 2)) == [(slice(0, 1), 5), (slice(1, 2), 1)]


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


def test_objects_on_the_ground_hide_light_from_below_the_horizon():
    """A wall standing on the ground hides light from just below the horizon at points close to it, but not at points
    so far that this light starts on the ground before the wall, and a roof hides none of it."""
    origin = T([0.0, 0.0, 1.0])
    # Rays along the wall at x = 2, 2 m from it, and away from it, ending 59 m from it
    directions = T([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    distance = T([10.0, 57.0])

    def below_horizon(occluder, bounds):
        occlusion = occlusion_of(
            render_maps([], bounds, occluder),
            [],
            render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 512),
            bounds,
        )
        shadows = trace(occlusion, FOG, origin, directions, distance, [], sky_samples=32)
        lit = _unpack(shadows.visible[:, -1], sum(CELLS)).bool()
        lit_below = _unpack(shadows.visible_below[:, -1], sum(CELLS)).bool()
        return lit, lit_below

    lowest = torch.arange(sum(CELLS)) < CELLS[0]  # cells of the band just above the horizon
    lit, lit_below = below_horizon(wall(2.0, (-40, 80), (0, 20)), [[2.0, -40.0, 0.0], [2.0, 80.0, 20.0]])
    assert (lit_below | ~lit).all()  # what is visible above the horizon is visible below it
    # Close to the wall, it hides the same cells just below the horizon as above it
    assert not lit[0, lowest].all() and torch.equal(lit_below[0, lowest], lit[0, lowest])
    # Far from it, light from below the horizon starts on the ground before the wall
    assert not lit[1, lowest].all() and lit_below[1].all()
    lit, lit_below = below_horizon(
        rectangle(6.0, (-100, 100), (-100, 400)), [[-100.0, -100.0, 6.0], [100.0, 400.0, 6.0]]
    )
    assert not lit[0].all() and lit_below.all()


def test_objects_under_points_hide_the_ground():
    """Points above a box see its top along the cells mirrored below the horizon that meet it, as the maps of the cells
    half a turn around record it, which hides the ground beyond it, while lower cells pass over the box."""
    low, high = [-2.0, -2.0, 0.0], [2.0, 2.0, 3.0]
    occlusion = maps_of(box(low, high), [low, high])
    cells = _sky_cells(occlusion, T([[0.0, 0.0, 5.0]]), _opposite_cells(CELLS), 0.0, 0.5)
    band = torch.repeat_interleave(torch.arange(len(CELLS)), torch.as_tensor(CELLS))
    steep, low_band = band == 5, band == 0  # 40° to 60° above the horizon, and the lowest band
    assert cells.visible[0].all() and cells.visible_below[0, low_band].all()
    assert not cells.visible_below[0, steep].any()
    # The top of the box is 2 m below the point
    sine = (math.sin(math.radians(40)) + math.sin(math.radians(60))) / 2
    assert torch.allclose(cells.below[0, steep], torch.full((8,), 2.0 / sine, dtype=torch.float64), atol=0.05)
    assert torch.isinf(cells.below[0, low_band]).all()


def test_normals_of_recorded_surfaces():
    """Normals of the surfaces that maps record follow from the slopes of their depths, on a box's sides and top."""
    occlusion = maps_of(box([-2.0, -2.0, 0.0], [2.0, 2.0, 3.0]), [[-2.0, -2.0, 0.0], [2.0, 2.0, 3.0]])
    directions = T(sky_cell_directions(EDGES, CELLS))
    side = int((directions @ T([-1.0, 0.0, 0.2])).argmax())  # low above the horizon, towards -x
    top = int((directions @ T([0.3, 0.4, 0.9])).argmax())
    points = T([[-2.0, 0.3, 1.5], [0.5, -0.4, 3.0], [0.5, -0.4, 3.0]])
    normals = _surface_normals(occlusion.sky_maps, torch.tensor([side, top, side]), points)
    expected = T([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(normals, expected, atol=0.02)


def test_phase_sums():
    """Sums of the phase function over hidden cells of the sky, for each group of samples, from the hidden cells alone,
    as from all cells at once."""
    rng = np.random.default_rng(4)
    grid, k, d = 6, 10, 5
    local = torch.as_tensor(np.sort(np.concatenate([np.arange(grid), rng.integers(0, grid, 8)])))
    direction = torch.as_tensor(rng.integers(0, d, len(local)))
    phases = T(rng.uniform(size=(d, k)))
    hidden = torch.as_tensor(rng.uniform(size=(grid, k)) < 0.4)
    slots = torch.as_tensor(rng.integers(0, 3, k))
    rows, columns = hidden.nonzero(as_tuple=True)
    values = T(rng.uniform(size=(len(rows), 2)))
    one_hot = torch.nn.functional.one_hot(slots, 3).to(torch.float64)
    expected = (hidden[local] * phases[direction]) @ one_hot
    assert torch.allclose(_phase_sums(rows, columns, local, grid, direction, phases, slots, 3), expected)
    dense = torch.zeros(grid, k, 2, dtype=torch.float64).index_put_((rows, columns), values)
    expected = torch.einsum("gk,gkc,ks->gsc", phases[direction], dense[local], one_hot)
    assert torch.allclose(_phase_sums(rows, columns, local, grid, direction, phases, slots, 3, values), expected)


def test_the_ground_lacks_sunlight_in_shadows():
    """Light that the ground reflects into the fog under a roof, whose shadow the ground is in, lacks the share of the
    ground's light that the sun brings, in proportion to it."""
    sun = np.array([0.0, 0.3, 0.95])
    sun /= np.linalg.norm(sun)
    bounds = [[-3.0, 5.0, 6.0], [3.0, 25.0, 6.0]]
    roof = rectangle(6.0, (-3, 3), (5, 25))
    occlusion = occlusion_of(
        render_maps([sun], bounds, roof, 512),
        [sun],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, roof, 256),
        bounds,
    )
    origin = T([0.0, 0.0, 1.6])
    directions = T([[0.0, math.cos(0.03), -math.sin(0.03)], [0.0, -math.cos(0.03), -math.sin(0.03)]])
    distance = 1.6 / torch.sin(T([0.03, 0.03]))
    shadows = trace(occlusion, FOG, origin, directions, distance, [T(sun)])
    shadowed = shadows.shadowed_weights_below.sum(dim=(1, 2, 3))
    assert shadowed[0] > 0.02 and shadowed[0] > 3 * shadowed[1]
    beta = T([FOG.extinction])
    fog = FOG.components[0]
    source = elevation_sources(FOG, Lighting(ground_albedo=(0.4,), sky=(1.0,)), beta, 1)["ground"]
    table = tabulate_along_rays(source, beta * fog.density * math.exp(-1.6 / fog.falloff) * fog.falloff)
    suns = [(T(sun), T([0.0]))]

    def ground(share=None):
        shares = None if share is None else [T([share])]
        return shade(shadows, FOG, origin, directions, distance, beta, suns, {"ground": table}, ground_shares=shares)

    plain, none, half, full = ground(), ground(0.0), ground(0.5), ground(1.0)
    assert torch.allclose(none, plain) and torch.allclose(half, (none + full) / 2)
    assert full[0] < 0.97 * none[0] and full[1] > 0.99 * none[1]


def test_the_ground_sees_less_sky_next_to_a_wall():
    """Next to a long wall, the ground receives the light of the cells of the sky whose central direction misses the
    wall, each weighted by its solid angle times the mean cosine of its elevations, close to the share of a uniform sky
    that an infinitely long wall leaves, ``(1 + d / sqrt(d² + h²)) / 2`` at a distance ``d`` from a wall of height
    ``h``, but within a meter of it, where the few cells of the highest band are all hidden or seen, and about the whole
    sky far from it."""
    height = 4.0
    bounds = [[2.0, -40.0, 0.0], [2.0, 80.0, height]]
    occluder = wall(2.0, (-40, 80), (0, height))
    occlusion = occlusion_of(
        render_maps([], bounds, occluder),
        [],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 512),
        bounds,
    )
    ground = occlusion_module.ground_sky(occlusion)
    rows, columns = ground.share.shape
    # Reaches three times the wall's height around it
    assert ground.low.tolist() == pytest.approx([2.0 - 3 * height, -40.0 - 3 * height])
    assert columns * ground.size >= 6 * height and rows * ground.size >= 120 + 6 * height
    centers = sky_cell_directions(EDGES, CELLS)
    weights = np.repeat(
        [math.pi * (high**2 - low**2) / count for low, high, count in zip(EDGES[:-1], EDGES[1:], CELLS)], CELLS
    )
    row = int((20.0 - float(ground.low[1])) / ground.size)
    for distance in (0.3, 1.0, 2.0, 4.0, 8.0):
        column = int((2.0 - distance - float(ground.low[0])) / ground.size)
        x = float(ground.low[0]) + (column + 0.5) * ground.size
        # Cells whose central direction from the center of the grid's cell meets the wall
        with np.errstate(divide="ignore"):
            t = np.where(centers[:, 0] > 0, (2.0 - x) / centers[:, 0], np.inf)
        hidden = (t * centers[:, 2] <= height) & np.isfinite(t)
        expected = weights[~hidden].sum() / weights.sum()
        share = float(ground.share[row, column])
        assert share == pytest.approx(expected, abs=0.02), distance
        d = 2.0 - x
        if distance >= 1:
            assert share == pytest.approx((1 + d / math.hypot(d, height)) / 2, abs=0.04), distance
    # Far from the wall, the ground sees about the whole sky
    assert float(ground.share[0, 0]) > 0.99 and float(ground.share[-1, -1]) > 0.99
    # Kept for the next frames
    assert occlusion_module.ground_sky(occlusion) is ground


def test_the_ground_next_to_objects_lacks_skylight():
    """Light that the ground reflects into the fog next to a wall, which hides part of the sky from the ground, lacks
    the share of the ground's light that the sky and the glow of the fog bring, in proportion to it, much more along
    the wall than away from it, while shadows traced without the share of the sky that the ground sees change
    nothing."""
    bounds = [[2.0, -40.0, 0.0], [2.0, 80.0, 6.0]]
    occluder = wall(2.0, (-40, 80), (0, 6))
    occlusion = occlusion_of(
        render_maps([], bounds, occluder),
        [],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 512),
        bounds,
    )
    origin = T([0.0, 0.0, 1.6])
    # Along the wall, slightly down, and away from it
    directions = T([[0.0, math.cos(0.05), -math.sin(0.05)], [-math.cos(0.05), 0.0, -math.sin(0.05)]])
    distance = 1.6 / torch.sin(T([0.05, 0.05]))
    beta = T([FOG.extinction])
    fog = FOG.components[0]
    source = elevation_sources(FOG, Lighting(ground_albedo=(0.4,), sky=(1.0,)), beta, 1)["ground"]
    table = tabulate_along_rays(source, beta * fog.density * math.exp(-1.6 / fog.falloff) * fog.falloff)
    shadows = trace(occlusion, FOG, origin, directions, distance, [])
    assert shadows.unseen_weights_below is not None and float(shadows.unseen_weights_below[0].sum()) > 0.01

    def ground(traced, share=None):
        share = None if share is None else T([share])
        return shade(traced, FOG, origin, directions, distance, beta, [], {"ground": table}, ground_sky_share=share)

    plain, none, half, full = ground(shadows), ground(shadows, 0.0), ground(shadows, 0.5), ground(shadows, 1.0)
    assert torch.allclose(none, plain) and torch.allclose(half, (none + full) / 2)
    darker = none - full
    assert full[0] < 0.8 * none[0] and darker[0] > 5 * darker[1] and darker[1] >= 0
    lean = trace(occlusion, FOG, origin, directions, distance, [], ground_sky_occlusion=False)
    assert lean.unseen_weights_below is None and torch.equal(ground(lean, 1.0), plain)
    # Weighted again for a medium of another anisotropy, as much of each band
    other = FOG.model_copy(update={"anisotropy": 0.5})
    reweighted = [
        shade(shadows, other, origin, directions, distance, beta, [], {"ground": table}, ground_sky_share=T([s]))
        for s in (0.0, 1.0)
    ]
    assert reweighted[1][0] < 0.8 * reweighted[0][0]


def test_light_reflected_by_a_wall():
    """A wall of uniform radiance standing on the ground beside the rays lights the fog, as a brute-force integral
    finds with the same cells of the sky, above and mirrored below the horizon, each of which sees the wall or not as
    its central direction does, attenuated along it, although the light is gathered on a coarse grid."""
    bounds = [[2.0, -40.0, 0.0], [2.0, 80.0, 20.0]]
    occluder = wall(2.0, (-40, 80), (0, 20))
    occlusion = occlusion_of(
        render_maps([], bounds, occluder),
        [],
        render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, 512),
        bounds,
    )
    origin, directions, distance = rays(azimuths=(-30, 0))
    beta = T([FOG.extinction])
    shadows = trace(occlusion, FOG, origin, directions, distance, [])
    reflected = object_reflections(
        occlusion, FOG, origin, directions, shadows, lambda points, normals: torch.ones_like(points[:, :1])
    )
    result = shade(shadows, FOG, origin, directions, distance, beta, [], {}, reflected=reflected)[:, 0]
    fog = FOG.components[0]
    centers = sky_cell_directions(EDGES, CELLS)
    mirrored = centers * [1.0, 1.0, -1.0]
    expected = []
    for v, bound, middles in zip(directions, shadows.bounds, shadows.middles):
        light = []
        for t in middles.tolist():
            point = origin.numpy() + t * v.numpy()
            for cells, below in ((centers, False), (mirrored, True)):
                hit = occluder(np.broadcast_to(point, cells.shape), cells)
                seen = hit < 1e9
                depth = height_fog_optical_depth(fog, T(point), T(cells[seen]), T(hit[seen])) * FOG.extinction
                phases = cell_phases(EDGES, CELLS, FOG.anisotropy, v[None], below=below)[0][seen]
                light.append(float((phases * torch.exp(-depth)).sum()))
        scattered = torch.exp(-height_fog_optical_depth(fog, origin, v.expand(len(bound), 3), bound) * FOG.extinction)
        expected.append((T(light).reshape(-1, 2).sum(dim=-1) * -scattered.diff()).sum().item())
    assert np.all(np.asarray(expected) > 0)
    assert np.allclose(result.numpy(), expected, rtol=0.02)
    # The light depends on the medium, unlike shadows
    darker = FOG.model_copy(update={"albedo": 0.5})
    half = object_reflections(
        occlusion, darker, origin, directions, shadows, lambda points, normals: torch.ones_like(points[:, :1])
    )
    assert torch.allclose(half, reflected / 2)


def test_objects_reflect_light_from_their_side_facing_points():
    """A thick box hides cells of the sky from a point beside it, whose own maps only record the box's far side: the
    light reaching the point comes from the side facing it, which other maps of the sky find, whose normal faces the
    point, and so does the light from below the horizon that the box hides from a point above its top."""
    low, high = [2.0, -3.0, 0.0], [5.0, 3.0, 4.0]
    occlusion = maps_of(box(low, high), [low, high])
    points = T([[0.0, 0.0, 1.0], [3.5, -1.0, 5.0]])
    cells = _sky_cells(occlusion, points, _opposite_cells(CELLS), 0.0, 0.5)
    seen = []

    def surfaces(where, normals):
        seen.append((where, normals))
        return torch.ones_like(where[:, :1])

    rows, columns, _ = _object_light(occlusion, FOG.components[0], points, cells, surfaces, T([FOG.extinction]), 0.5)
    below = columns >= len(occlusion.sky_maps.texels)
    # Beside the box, above the horizon: just in front of its side at x = 2, which faces -x
    where, normals = (x[(rows == 0) & ~below] for x in seen[0])
    assert len(where) > 10
    assert (where[:, 0] > 1.9).all() and (where[:, 0] <= 2.0).all()
    assert torch.allclose(normals, T([-1.0, 0.0, 0.0]).expand_as(normals), atol=0.05)
    # Above the box, below the horizon: just above its top
    where, normals = (x[(rows == 1) & below] for x in seen[0])
    top = ((where[:, 0] > 2.0) & (where[:, 0] < 5.0)).nonzero()[:, 0]
    assert len(top) > 10
    assert (where[top, 2] > 3.99).all() and (where[top, 2] < 4.1).all()
    assert torch.allclose(normals[top], T([0.0, 0.0, 1.0]).expand(len(top), 3), atol=0.05)


def test_ground_in_the_shadow_of_a_sun():
    """The cells mirrored below the horizon whose light starts on the ground in the shadow of a sun are found from the
    coordinates of the ground in the sun's map, which are affine in the height of points, as from the points where their
    light starts on the ground."""
    sun = np.array([0.4, -0.3, 0.6])
    sun /= np.linalg.norm(sun)
    low, high = [-1.0, 2.0, 0.0], [1.0, 4.0, 3.0]
    occlusion = maps_of(box(low, high), [low, high])
    maps = render_maps([sun], [low, high], box(low, high), 512)
    rng = np.random.default_rng(1)
    points = T(rng.uniform([-8.0, -6.0, 0.2], [8.0, 12.0, 6.0], size=(200, 3)))
    cells = _sky_cells(occlusion, points, _opposite_cells(CELLS), 0.0, 0.5)
    towards = occlusion.sky_maps.axes[:, 2]
    slopes = towards[:, :2] / towards[:, 2:]
    height, width = (int(size) for size in maps.shapes[0])
    image = maps.depths[: height * width].reshape(height, width)
    shaded = _ground_shadowed(maps, image, points, slopes, cells.visible_below, 0.0, 0.5)
    ground = points[:, None, :] + cells.reach[..., None] * cells.mirrored
    lit = visibility(maps, T([0.0, 0.0, 0.0]), ground.reshape(-1, 3), torch.ones(ground.numel() // 3, 1), 0.5, False)
    expected = (lit[:, 0, 0] < 0.5).reshape(cells.visible_below.shape) & cells.visible_below
    assert expected.sum() > 50
    assert torch.equal(shaded, expected)


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


def window_map(height=64, width=128, radius=2.0):
    """Map of a lamp at the origin, enclosed by a sphere with a window around +x, through which it shines."""
    v, u = torch.meshgrid(
        (torch.arange(height, dtype=torch.float64) + 0.5) / height,
        (torch.arange(width, dtype=torch.float64) + 0.5) / width,
        indexing="ij",
    )
    # Inverse of the equirectangular mapping of `lamp_visibility`
    theta, phi = math.pi * (1 - v), math.pi - 2 * math.pi * u
    x = torch.sin(theta) * torch.cos(phi)
    depths = torch.where(x > 0.9, torch.full_like(x, 1e10), torch.full_like(x, radius))
    return ShadowMaps(
        depths=depths.reshape(-1),
        offsets=torch.zeros(1, dtype=torch.long),
        shapes=torch.tensor([[height, width]]),
        origins=torch.zeros(1, 3, dtype=torch.float64),
        axes=torch.eye(3, dtype=torch.float64)[None],
        texels=torch.tensor([2 * math.pi / width], dtype=torch.float64),
    )


def box(low, high):
    """First hit of rays on an axis-aligned box, from outside of it, or a huge distance if they miss it."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    def first_hit(start, direction):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (low - start) / direction, (high - start) / direction
        near = np.nanmax(np.minimum(t1, t2), axis=-1)
        far = np.nanmin(np.maximum(t1, t2), axis=-1)
        return np.where((far >= near) & (far > 0), np.maximum(near, 0), 1e10)

    return first_hit


def lamp_map(position, first_hit, height=128, width=256):
    """Equirectangular map of a lamp, as ``lamp_visibility`` reads it, of analytic occluders."""
    v, u = np.meshgrid((np.arange(height) + 0.5) / height, (np.arange(width) + 0.5) / width, indexing="ij")
    theta, phi = np.pi * (1 - v), np.pi - 2 * np.pi * u
    directions = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    depths = first_hit(np.broadcast_to(np.asarray(position, dtype=float), directions.shape), directions)
    return ShadowMaps(
        depths=T(depths.reshape(-1)),
        offsets=torch.zeros(1, dtype=torch.long),
        shapes=torch.tensor([[height, width]]),
        origins=T([position]),
        axes=torch.eye(3, dtype=torch.float64)[None],
        texels=T([2 * math.pi / width]),
    )


def union(*occluders):
    """First hit of rays on any of several occluders."""
    return lambda start, direction: np.minimum.reduce([occluder(start, direction) for occluder in occluders])


def maps_of(occluder, bounds, lamp=None, resolution=512):
    """Shadow maps of analytic occluders within bounds, along the cells of the sky and from a lamp."""
    sky = render_maps(sky_cell_directions(EDGES, CELLS), bounds, occluder, resolution)
    occlusion = occlusion_of(render_maps([], bounds, occluder), [], sky, bounds)
    return occlusion._replace(lamp_maps=lamp_map(lamp, occluder) if lamp is not None else None)


def test_occupancy_of_a_box():
    """The visual hull of a box seen from the sky and from a lamp is the box, not its shadow, give or take a voxel."""
    low, high = [1.0, -1.0, 0.0], [2.0, 1.0, 2.0]
    occlusion = maps_of(box(low, high), [[-3.0, -3.0, 0.0], [6.0, 3.0, 3.0]], lamp=[4.0, 0.0, 1.0])
    voxels = occupancy(occlusion, T([-5.0, -5.0, 0.0]), T([8.0, 5.0, 6.0]), size=0.05)
    assert voxels is not None and voxels.clearance.dtype == torch.uint8
    # Inside, in the lamp's shadow behind it, in front of it, above it and beside it
    points = T([[1.5, 0.0, 1.0], [1.1, 0.9, 0.1], [0.5, 0.0, 1.0], [-2.0, 0.0, 0.5], [3.0, 0.0, 1.0], [1.5, 0.0, 2.3]])
    assert occupied(voxels, points).tolist() == [True, True, False, False, False, False]
    # The voxels that cross the box are occupied, and those further than a voxel from it aren't
    cells = torch.meshgrid(*(torch.arange(n) for n in voxels.clearance.shape), indexing="ij")
    centers = voxels.low + (torch.stack(cells, -1) + 0.5) * voxels.size
    gap = torch.maximum(T(low) - centers, centers - T(high)).amax(dim=-1)
    full = voxels.clearance == 0
    assert full[gap < -voxels.size].all() and not full[gap > 1.5 * voxels.size].any()
    # Clearances count the voxels to the nearest occupied one, here 1 m away
    i, j, k = ((T([3.0, 0.0, 1.0]) - voxels.low) / voxels.size).long().tolist()
    assert 18 <= int(voxels.clearance[i, j, k]) <= 20

    # Rays from the shadow towards the box, away from it, beside it, over it, and from inside of it
    origins = T([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 2.0, 1.0], [-1.0, 0.0, 2.5], [1.5, 0.0, 1.0]])
    directions = T([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lengths = T([20.0, 20.0, math.inf, 20.0, 20.0])
    distances = hit_distances(voxels, origins, directions, lengths)
    assert distances[0].item() == pytest.approx(1.0, abs=2 * voxels.size)
    assert distances[1:4].tolist() == [20.0, math.inf, 20.0] and distances[4].item() == 0.0
    # Many rays in every direction from a point near the box hit it where the box does
    rng = np.random.default_rng(0)
    directions = T(rng.normal(size=(2000, 3)))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    origin = T([2.6, 0.3, 1.2])
    starts = np.broadcast_to(origin.numpy(), directions.shape)
    expected = T(box(low, high)(starts, directions.numpy()))
    # Rays that miss the box by more than two voxels
    wide = T(box(np.subtract(low, 2 * voxels.size), np.add(high, 2 * voxels.size))(starts, directions.numpy()))
    distances = hit_distances(voxels, origin.expand(2000, 3), directions, torch.full((2000,), 50.0, dtype=torch.float64))
    hit = expected < 1e9
    assert hit.sum() > 100 and torch.all(distances[wide >= 1e9] == 50)
    assert torch.allclose(distances[hit], expected[hit], atol=3 * voxels.size)


def test_rays_pass_right_beside_objects():
    """Given the depth of halos, a little less than a voxel, rays that cross the voxels occupied around a box over less
    than that go on, as they pass beside it, while rays that hit it stop where they enter the voxels around it, in
    front of it, as do rays that cross a wall thinner than a voxel."""
    low, high = [1.0, -1.0, 0.0], [2.0, 1.0, 2.0]
    occlusion = maps_of(box(low, high), [[-3.0, -3.0, 0.0], [6.0, 3.0, 3.0]], lamp=[4.0, 0.0, 1.0])
    voxels = occupancy(occlusion, T([-5.0, -5.0, 0.0]), T([8.0, 5.0, 6.0]), size=0.05)
    assert voxels is not None
    # Rays from beside the box towards its edges, within a few voxels of them
    rng = np.random.default_rng(1)
    origin = T([3.5, 2.5, 1.2])
    corners = T([[2.0, 1.0, 0.5], [1.0, 1.0, 1.5], [2.0, -1.0, 1.0], [1.5, 1.0, 2.0]])
    targets = corners[rng.integers(0, 4, 4000)] + T(rng.uniform(-0.15, 0.15, (4000, 3)))
    directions = (targets - origin) / (targets - origin).norm(dim=-1, keepdim=True)
    starts = np.broadcast_to(origin.numpy(), directions.shape)

    def missed_by(margin):
        """Whether rays miss the box grown by a margin, in voxels."""
        grown = box(np.subtract(low, margin * voxels.size), np.add(high, margin * voxels.size))
        return T(grown(starts, directions.numpy())) >= 1e9

    expected = T(box(low, high)(starts, directions.numpy()))
    lengths = torch.full((4000,), 50.0, dtype=torch.float64)
    plain = hit_distances(voxels, origin.expand(4000, 3), directions, lengths)
    deep = hit_distances(voxels, origin.expand(4000, 3), directions, lengths, depth=_HIT_DEPTH * voxels.size)
    hit, beside = expected < 1e9, missed_by(0.75) & ~missed_by(1.0)
    assert hit.sum() > 500 and beside.sum() > 100
    # Rays that hit the box stop in front of it, within the voxels around it, as without a depth
    assert torch.all(deep[hit] <= expected[hit] + 1e-9) and torch.all(deep[hit] >= expected[hit] - 3 * voxels.size)
    assert torch.all(plain[hit] <= deep[hit])
    # Voxels are occupied about a voxel around the box, which stops rays that miss it by less without a depth, but
    # hardly with it
    assert torch.all(plain[beside] < 50) and (deep[beside] < 50).double().mean() < 0.2
    assert torch.all(plain[missed_by(1.5)] == 50) and torch.all(deep[missed_by(1.0)] == 50)

    # A wall 2 cm thick, thinner than the voxels, stops every ray that crosses it, at any angle
    wall = box([-3.0, 0.0, 0.0], [3.0, 0.02, 2.0])
    occlusion = maps_of(wall, [[-4.0, -1.0, 0.0], [4.0, 1.0, 3.0]], lamp=[0.0, 2.0, 1.0])
    voxels = occupancy(occlusion, T([-5.0, -5.0, 0.0]), T([5.0, 5.0, 4.0]), size=0.05)
    assert voxels is not None
    directions = T(rng.normal(size=(4000, 3))) * T([1.0, 0.0, 1.0]) + T([0.0, 1.0, 0.0])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    origins = T([0.0, -1.0, 1.0]).expand(4000, 3)
    expected = T(wall(origins.numpy(), directions.numpy()))
    crossing = expected < 1e9
    lengths = torch.full((4000,), 50.0, dtype=torch.float64)
    deep = hit_distances(voxels, origins, directions, lengths, depth=_HIT_DEPTH * voxels.size)
    assert crossing.sum() > 1000 and torch.all(deep[crossing] <= expected[crossing] + 1e-9)


def test_rays_stop_at_a_single_layer_of_voxels():
    """Rays that go straight through a single layer of occupied voxels, the thinnest that objects fill, go through
    exactly a voxel, which the depth of halos, a little less, doesn't leave to rounding: they all stop where they enter
    the layer, in single and double precision, wherever the grid lies, as do rays that cross it at an angle, and they
    are chunked without changing anything."""
    layer = torch.zeros(120, 120, 60, dtype=torch.bool)
    layer[10:110, 61, 5:55] = True
    # Chebyshev distance to the layer, as `occupancy` computes it
    clearance = torch.full(layer.shape, 32, dtype=torch.uint8)
    reached = layer.float()[None, None]
    for distance in range(32):
        clearance[(reached[0, 0] > 0) & (clearance == 32)] = distance
        reached = torch.nn.functional.max_pool3d(reached, 3, stride=1, padding=1)
    rng, size = np.random.default_rng(2), 0.05
    for spread in (0.0, 0.5, 5.0):
        angles = rng.normal(0, math.radians(spread), (2, 2000))
        directions = T(np.stack([angles[0], np.ones(2000), angles[1]], -1))
        directions = directions / directions.norm(dim=-1, keepdim=True)
        for low in ([-3.0, -3.0, 0.0], [-2.987, -3.0131, 0.0]):
            # About 30 voxels in front of the layer, between the boundaries of voxels, within its extent
            origins = torch.stack(
                [
                    T(rng.uniform(-2.0, 2.0, 2000)),
                    torch.full((2000,), low[1] + 30.37 * size, dtype=torch.float64),
                    T(rng.uniform(0.6, 2.4, 2000)),
                ],
                -1,
            )
            entry = (low[1] + 61 * size - origins[:, 1]) / directions[:, 1]
            for dtype in (torch.float32, torch.float64):
                voxels = Occupancy(torch.tensor(low, dtype=dtype), size, clearance)
                lengths = torch.full((2000,), 50.0, dtype=dtype)
                deep = hit_distances(voxels, origins.to(dtype), directions.to(dtype), lengths, depth=_HIT_DEPTH * size)
                assert torch.all(deep < 50) and torch.allclose(deep.double(), entry, atol=0.01 * size)
                chunked = hit_distances(
                    voxels, origins.to(dtype), directions.to(dtype), lengths, depth=_HIT_DEPTH * size, chunk=300
                )
                assert torch.equal(chunked, deep)


def room():
    """First hit of rays on the walls, floor excepted, of a closed room of 10 by 10 by 3 meters, and its bounds."""
    walls = union(
        box([-5.0, -5.0, 3.0], [5.0, 5.0, 3.2]),
        box([-5.2, -5.0, 0.0], [-5.0, 5.0, 3.2]),
        box([5.0, -5.0, 0.0], [5.2, 5.0, 3.2]),
        box([-5.2, -5.2, 0.0], [5.2, -5.0, 3.2]),
        box([-5.2, 5.0, 0.0], [5.2, 5.2, 3.2]),
    )
    return walls, [[-5.2, -5.2, 0.0], [5.2, 5.2, 3.2]]


def test_occupancy_carves_what_lights_light_in_a_room():
    """Inside a closed room, the sky sees no place, but a lamp or an emissive surface in it does, so that only the walls
    are occupied."""
    walls, bounds = room()
    region = T([-8.0, -8.0, 0.0]), T([8.0, 8.0, 6.0])
    points = T([[0.0, 0.0, 3.1], [5.1, 0.0, 1.0], [3.0, 2.0, 1.0], [0.5, 0.0, 2.0], [7.0, 0.0, 1.0]])
    dark = maps_of(walls, bounds, resolution=128)
    light = lamp_map([0.0, 0.0, 2.0], walls)
    for occlusion in (dark._replace(lamp_maps=light), dark._replace(emissive_maps=light)):
        voxels = occupancy(occlusion, *region, size=0.1)
        assert voxels is not None and occupied(voxels, points).tolist() == [True, True, False, False, False]
    # Without lights, the room is taken for the inside of an object
    voxels = occupancy(dark, *region, size=0.1)
    assert voxels is not None and occupied(voxels, points).tolist() == [True, True, True, True, False]


def test_occupancy_needs_the_sky():
    occlusion = maps_of(box([1.0, -1.0, 0.0], [2.0, 1.0, 2.0]), [[1.0, -1.0, 0.0], [2.0, 1.0, 2.0]], resolution=128)
    assert occupancy(occlusion, T([3.0, 3.0, 3.0]), T([5.0, 5.0, 5.0])) is None  # away from every object
    no_sky = occlusion._replace(sky_maps=render_maps([], [[0.0] * 3, [1.0] * 3], box([0.0] * 3, [1.0] * 3)))
    assert occupancy(no_sky, T([-5.0, -5.0, 0.0]), T([5.0, 5.0, 5.0])) is None


def test_lamp_visibility():
    maps = window_map()
    points = torch.tensor([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 3.0, 0.2], [1.0, 0.5, -0.5], [0.0, 0.0, 5.0]])
    # Through the window, behind the sphere, and inside of it
    assert lamp_visibility(maps, 0, points.double()).tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_lamp_shadows_skip_hidden_rays():
    maps = window_map()
    origin = torch.tensor([6.0, -6.0, 0.0], dtype=torch.float64)
    rng = np.random.default_rng(3)
    directions = torch.as_tensor(rng.normal(size=(300, 3)) * [0.6, 0.6, 0.3] + [-0.7, 0.7, 0.0])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = torch.as_tensor(rng.uniform(3, 20, size=300))
    beta = torch.tensor([FOG.extinction], dtype=torch.float64)
    lamp = PointLight(position=(0.0, 0.0, 0.0), power=(4 * math.pi,))
    ours = lamp_inscatter(FOG, origin, directions, distance, beta, lamp, maps=maps)[:, 0]
    # The same quadrature, weighted by the visibility of every ray, including those entirely in the shadow
    shadow = lamp_shadow(maps, 0, origin, directions, distance)
    expected = point_light_inscatter(FOG, origin, directions, distance, maps.origins[0], beta, shadow=shadow.average)[
        :, 0
    ]
    assert 0 < int((~shadow.lit).sum()) < len(directions) and torch.all(ours[~shadow.lit] == 0)
    assert torch.allclose(ours, expected, rtol=1e-12)
    # Light only comes out of the window, around +x
    unshadowed = point_light_inscatter(FOG, origin, directions, distance, maps.origins[0], beta)[:, 0]
    assert 0 < ours.sum() < 0.5 * unshadowed.sum()


def test_maps_are_copied_once():
    """Maps moved to another precision or device are copied once and kept while they live, unless they change."""
    occlusion = demo_scene()[0]
    maps = occlusion.sky_maps
    single = maps_to(maps, dtype=torch.float32)
    assert single.depths.dtype == torch.float32 and torch.equal(single.depths, maps.depths.float())
    assert maps_to(maps, dtype=torch.float32) is single
    # Maps already in that precision aren't copied, nor kept
    assert maps_to(maps, dtype=torch.float64).depths is maps.depths
    moved = occlusion_to(occlusion, dtype=torch.float32)
    again = occlusion_to(occlusion, dtype=torch.float32)
    assert all(a is b for a, b in zip(again, moved)) and moved.sky_maps is single
    # Occlusions that share their maps keep their other fields, and their maps the fields that hold them
    tolerant = occlusion_to(occlusion._replace(lamp_tolerance=0.5), dtype=torch.float32)
    assert tolerant.lamp_tolerance == 0.5 and tolerant.sky_maps is single and moved.lamp_tolerance == 0.01
    lamps = occlusion_to(occlusion._replace(lamp_maps=maps), dtype=torch.float32)
    emissive = occlusion_to(occlusion._replace(emissive_maps=maps), dtype=torch.float32)
    assert lamps.lamp_maps is single and lamps.emissive_maps is None
    assert emissive.emissive_maps is single and emissive.lamp_maps is None
    # Maps modified in place are copied again
    maps.depths[0] = -1.0
    changed = maps_to(maps, dtype=torch.float32)
    assert changed is not single and float(changed.depths[0]) == -1.0
    # Copies are dropped along with their maps
    count = len(_KEPT)
    del occlusion, maps, moved, single, changed
    gc.collect()
    assert len(_KEPT) < count


def test_at_night_objects_dont_hide_the_sky(monkeypatch):
    """Without suns or skylight, nothing from the sky is traced, and shadow maps without lamps change nothing."""
    occlusion, _, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    # A lamp high above the fog, whose halo is too faint to compute, over a black ground, which reflects none of its
    # light into the fog, which objects would hide
    night = Lighting(points=[PointLight(position=(0.0, 20.0, 30.0), power=(5000.0,))])
    assert not environment_lit(night) and environment_lit(Lighting(sky=SKY))
    assert not environment_lit(Lighting(suns=[Sun(direction=(0.3, 0.0, -0.1), irradiance=(4.0,))]))
    assert "scattered" not in elevation_sources(medium, night, T([medium.extinction]), 3)
    normals = np.zeros((*depth.shape, 3))
    normals[..., 1] = -1.0
    expected = apply_medium(radiance, depth, camera, pose, medium, night, normals=normals)
    monkeypatch.setattr(render, "trace", None)  # which fails if it is called
    result = apply_medium(radiance, depth, camera, pose, medium, night, occlusion=occlusion, normals=normals)
    assert float(expected.inscatter.max()) > 0
    assert torch.equal(result.radiance, expected.radiance)


def test_band_radiance_is_kept_for_the_light_it_averages():
    """The radiance of each band is kept for the light gathered that it averages, not for the table that holds it."""
    medium = FOG.model_copy(update={"multiple_scattering": True})
    lighting = Lighting(sky=SKY, suns=[Sun(direction=(0.6, -0.3, 0.5), irradiance=(4.0,))], ground_albedo=(0.3,) * 3)
    table = cached_multiple_scattering(medium, lighting, T([medium.extinction]), 3)
    radiance = band_radiance(table, EDGES)
    assert band_radiance(table, EDGES) is radiance and radiance.shape == (len(table.gathered), 2, len(CELLS), 3)
    # Other light gathered from the same directions, which doubles exactly
    doubled = band_radiance(table._replace(gathered=2 * table.gathered), EDGES)
    assert torch.equal(doubled, 2 * radiance)
    # Light changed in place
    gathered = table.gathered.clone()
    copy = table._replace(gathered=gathered)
    assert torch.equal(band_radiance(copy, EDGES), radiance)
    gathered.mul_(2)
    assert torch.equal(band_radiance(copy, EDGES), doubled)
    # Part of the light gathered, the ground's scattered once, which the same table holds and which is kept apart
    assert table.from_ground is not None
    ground = band_radiance(table, EDGES, gathered=table.from_ground)
    assert not torch.equal(ground, radiance) and torch.all(ground <= radiance * (1 + 1e-12))
    assert band_radiance(table, EDGES) is radiance and band_radiance(table, EDGES, gathered=table.from_ground) is ground
    _DERIVED.clear()
    assert torch.equal(band_radiance(table, EDGES, gathered=table.from_ground), ground)
    assert torch.equal(band_radiance(table, EDGES), radiance)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_frames_under_inference_mode(dtype):
    """Frames rendered under inference mode, whose tensors don't tell their changes in place, from which nothing is
    kept, are those rendered outside of it, before and after, including with shadow maps made under it."""
    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    normals = np.zeros((*depth.shape, 3))
    normals[..., 1] = -1.0

    def frame(occlusion: Occlusion) -> torch.Tensor:
        kwargs = {"occlusion": occlusion, "normals": normals, "dtype": dtype}
        return apply_medium(radiance, depth, camera, pose, medium, lighting, **kwargs).radiance

    # Light scattered more than once, whose table is kept, is first computed under inference mode, as is everything
    # kept for the maps, which are then used outside of it
    scattering._cached_multiple_scattering.cache_clear()
    with torch.inference_mode():
        inferred = frame(occlusion)
        made_under = frame(demo_scene()[0])
        assert made_under.is_inference()
    assert torch.equal(made_under, inferred)
    assert torch.equal(frame(occlusion), inferred)
    with torch.no_grad():
        assert torch.equal(frame(occlusion), inferred)
    # And outside of it, once more
    scattering._cached_multiple_scattering.cache_clear()
    assert torch.equal(frame(occlusion), inferred)


def _forget_kept_results() -> None:
    """Drop every result that the library keeps from one frame to the next."""
    for module in (scattering, surfaces, occlusion_module, render):
        for value in vars(module).values():
            if isinstance(value, functools._lru_cache_wrapper):
                value.cache_clear()
            elif isinstance(value, OrderedDict):
                value.clear()


def test_kept_results_follow_the_lighting():
    """What is kept from one frame to the next, the copies of maps, the tables along camera rays and the radiance of
    bands, is kept for the albedos of the ground and of objects too, that the light objects reflect and the ground's
    light depend on: frames rendered after changing them are those rendered with nothing kept."""
    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"multiple_scattering": True})
    normals = np.zeros((*depth.shape, 3))
    normals[..., 1] = -1.0
    lightings = [
        lighting,
        lighting.model_copy(update={"ground_albedo": (0.15,) * 3}),
        lighting.model_copy(update={"object_albedo": (0.6, 0.5, 0.4)}),
    ]

    def frame(lighting: Lighting) -> torch.Tensor:
        kwargs = {"occlusion": occlusion, "normals": normals, "dtype": torch.float32}
        return apply_medium(radiance, depth, camera, pose, medium, lighting, **kwargs).radiance

    frames = [frame(lighting) for lighting in lightings]
    assert not torch.equal(frames[0], frames[1]) and not torch.equal(frames[0], frames[2])
    for lighting, kept in zip(lightings, frames):
        _forget_kept_results()
        assert torch.equal(frame(lighting), kept)
