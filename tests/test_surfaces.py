import math

import numpy as np
import pytest
import torch

from tests.test_occlusion import occlusion_of, rectangle, render_maps
from visionsim.medium import Effects, Lighting, Medium, Sun, apply_medium, camera_rays
from visionsim.medium.model import HeightFog, Homogeneous, PointLight
from visionsim.medium.scattering import (
    _gather_nodes,
    cached_multiple_scattering,
    direct_ground_radiance,
    sun_scattered_source,
)
from visionsim.medium.surfaces import (
    _glow_visibility,
    ground_shares,
    lamp_irradiance,
    object_radiance,
    surface_attenuation,
    surface_irradiance,
    surface_tables,
)

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


def test_glow_of_light_scattered_more_than_once_follows_suns():
    """Surfaces facing the sun receive more of its light scattered more than once than those facing away, as from the
    integral of its variations with azimuth along every direction, by brute force."""
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.3,)})
    medium = FOG.model_copy(update={"multiple_scattering": True})
    beta = T([medium.extinction])
    tables = surface_tables(medium, lighting, beta, 3)
    varying = tables.suns[0] - surface_tables(FOG, lighting, beta, 3).suns[0]
    table = cached_multiple_scattering(medium, lighting, beta, 3)

    h = 3  # about 2 m above the ground
    mu, mu_weights = _gather_nodes(64, dtype=torch.float64)
    phi = (torch.arange(128, dtype=torch.float64) + 0.5) / 128 * 2 * math.pi
    t, w = (T(a) for a in np.polynomial.legendre.leggauss(32))
    t, w = (t + 1) / 2, w / 2
    c0, depth_ground = tables.depths[h], table.depth_ground
    v = mu[:, None]
    end = torch.where(v > 0, torch.exp(-c0 / v.abs()), torch.exp(-(depth_ground - c0) / v.abs()))  # (d, 1)
    depth = (c0 + v[..., None] * torch.log1p(-(1 - end)[:, None, :] * t[None, :, None])).clamp_min(0)  # (d, steps, 1)
    arriving = torch.stack(
        [
            (sun_scattered_source(table, 0, mu, torch.full_like(mu, min(p, 2 * math.pi - p)), depth) * w[:, None]).sum(1)
            * (1 - end)
            for p in phi.tolist()
        ],
        dim=1,
    )  # (d, phi, 3)
    for normal, column in (((1.0, 0.0, 0.0), 0), ((-1.0, 0.0, 0.0), -1)):  # facing the sun's azimuth, and away from it
        cosine = (torch.sqrt(1 - mu[:, None] ** 2) * torch.cos(phi) * normal[0]).clamp_min(0)
        expected = ((cosine * mu_weights[:, None] * 2 * math.pi / 128)[..., None] * arriving).sum((0, 1))
        assert torch.allclose(varying[h, 16, column], expected, rtol=0.01)
    assert torch.all(varying[h, 16, 0] > 0) and torch.all(varying[h, 16, -1] < 0)


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


def test_glow_of_lamps_lights_surfaces():
    """With the glow of a lamp, its light is attenuated with the full extinction, and the glow added, in its shadows
    too, while lamps without a glow keep the reduced extinction, and so does the share of the lamp's light whose glow
    isn't modeled, as beyond the extent of halos."""
    medium = Medium(extinction=0.1, anisotropy=0.7, albedo=0.9, components=[Homogeneous()])
    lamps = [PointLight(position=(0.0, 0.0, 5.0), power=(1000.0,)), PointLight(position=(2.0, 0.0, 4.0), power=(500.0,))]
    lighting = Lighting(points=lamps)
    points, normals = T([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]]), T([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    beta = T([medium.extinction])
    _, similar = surface_irradiance(medium, lighting, beta, points, normals)
    full = lamp_irradiance(medium.model_copy(update={"albedo": 0.0}), lamps[0], beta, points, normals)[1]
    reduced = [lamp_irradiance(medium, lamp, beta, points, normals)[1] for lamp in lamps]
    for share in (T([1.0, 1.0]), T([1.0, 0.25])):
        _, through = surface_irradiance(
            medium,
            lighting,
            beta,
            points,
            normals,
            glows=[lambda p, n, share=share: (p.new_full((len(p), 1), 0.25), share), None],
        )
        blended = share[:, None] * full + (1 - share[:, None]) * reduced[0]
        assert torch.allclose(through, blended + 0.25 + reduced[1], rtol=1e-12)
    assert torch.all(through - 0.25 < similar)
    assert torch.allclose(lamp_irradiance(medium, lamps[0], beta, points, normals, similarity=False)[1], full)
    # A share of the light of each point is attenuated with the reduced extinction, the rest with the full one
    mixed = lamp_irradiance(medium, lamps[0], beta, points, normals, similarity=T([1.0, 0.0]))[1][:, 0]
    assert mixed.tolist() == pytest.approx([float(reduced[0][0, 0]), float(full[1, 0])], rel=1e-12)


def test_radiance_of_objects():
    """Objects are lit as surfaces are, by suns, the sky and the glow but not by lamps, at the nearest tabulated
    normal, which those on the grid of the tables are."""
    lamp = PointLight(position=(2.0, 3.0, 4.0), power=(500.0,))
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.3,), "object_albedo": (0.5, 0.4, 0.3), "points": [lamp]})
    medium = FOG.model_copy(update={"multiple_scattering": True})
    beta = T([medium.extinction] * 3)
    towards = math.atan2(0.43, 0.75)
    # Vertical components and azimuths relative to the sun's on the grid of the tables
    normal_z, azimuth = np.meshgrid(np.linspace(-1, 1, 5), towards + np.linspace(0, np.pi, 5), indexing="ij")
    horizontal = np.sqrt(1 - normal_z**2)
    normals = T(np.stack([horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), normal_z], -1).reshape(-1, 3))
    points = T(np.stack([np.zeros(25), np.zeros(25), np.linspace(0.1, 9.0, 25)], -1))
    radiance = object_radiance(medium, lighting, beta, points, normals)
    daylight = lighting.model_copy(update={"points": []})
    _, through = surface_irradiance(medium, daylight, beta, points, normals)
    assert torch.allclose(radiance, T([0.5, 0.4, 0.3]) / math.pi * through, rtol=1e-6)


def test_ground_shares():
    """The sun brings the share of the ground's radiance that it lights directly, and with multiple scattering, through
    the glow of its light scattered once too, of the ground's radiance that the glow brightens."""
    setting = Sun(direction=(1.0, 0.0, -0.2), irradiance=(1.0,))
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.3,), "suns": [*LIGHTING.suns, setting]})
    beta = T([FOG.extinction])
    single, below = ground_shares(FOG, lighting, beta, 3)
    sky = lighting.model_copy(update={"suns": []})
    direct = direct_ground_radiance(FOG, lighting, beta, 3)
    sunlit = direct - direct_ground_radiance(FOG, sky, beta, 3)
    assert below is None and single is not None and torch.allclose(single, sunlit / direct)
    medium = FOG.model_copy(update={"multiple_scattering": True})
    multiple, _ = ground_shares(medium, lighting, beta, 3)
    ground = cached_multiple_scattering(medium, lighting, beta, 3).ground
    assert multiple is not None and torch.all(multiple > sunlit / ground) and torch.all(multiple < 1)


def test_surfaces_in_the_shadow_of_a_sun_lack_its_glow():
    """Under a wide roof, the ground lacks the glow of the sun's light scattered once from the medium in the roof's
    shadow: all of it deep in the shadow, less of it closer to the shadow's edge, and none of it beyond, where it is
    lit as without the roof."""
    sun = np.array([0.3, 0.2, 0.93])
    sun /= np.linalg.norm(sun)
    lighting = LIGHTING.model_copy(update={"suns": [Sun(direction=tuple(sun), irradiance=(4.0,))]})
    bounds = [[-50.0, -50.0, 6.0], [50.0, 50.0, 6.0]]
    roof = rectangle(6.0, (-50, 50), (-50, 50))
    occlusion = occlusion_of(render_maps([sun], bounds, roof, 1024), [sun], render_maps([], bounds, roof), bounds)
    # Deep in the shadow, 1 m and 0.2 m inside its edge, which the roof casts 6 · 0.32 m away from its own, and
    # beyond it
    edge = 50.0 - 6 * sun[0] / sun[2]
    points = T([[0.0, 0.0, 0.0], [edge - 1.0, 0.0, 0.0], [edge - 0.2, 0.0, 0.0], [60.0, 0.0, 0.0]])
    normals = T([[0.0, 0.0, 1.0]] * 4)
    beta = T([FOG.extinction] * 3)
    seen = _glow_visibility(occlusion.sun_maps, FOG, beta, tuple(sun), 0.0, points, normals)[:, 0]
    assert seen[0] < 0.01 and seen[0] < seen[1] < seen[2] < 1 and seen[3] == 1
    _, open_air = surface_irradiance(FOG, lighting, beta, points, normals)
    edges = Effects(glow_shadow_edges=False)
    _, deep = surface_irradiance(FOG, lighting, beta, points, normals, occlusion=occlusion, effects=edges)
    _, shadowed = surface_irradiance(FOG, lighting, beta, points, normals, occlusion=occlusion)
    assert torch.allclose(open_air[3], shadowed[3]) and torch.allclose(open_air[3], deep[3])
    assert torch.all(deep[1:3] < shadowed[1:3]) and torch.all(shadowed[:3] < open_air[:3])
    glow = surface_tables(FOG, lighting, beta[:1], 3).once[0][0, -1, 0]
    assert torch.allclose(shadowed[:3] - deep[:3], seen[:3, None] * glow, rtol=1e-2)  # looked up in single precision


def test_surface_attenuation_falls_back_to_the_mean():
    clear = T([[1.0], [1.0], [0.0]])
    through = T([[0.5], [0.7], [0.0]])
    ratio = surface_attenuation(clear, through)
    # Lit surfaces are mostly scaled by their own ratio, unlit ones by the mean ratio
    assert ratio[2, 0].item() == pytest.approx(0.6)
    assert ratio[0, 0].item() == pytest.approx((0.5 + 0.6 * 0.2 * 2 / 3) / (1 + 0.2 * 2 / 3))
    assert torch.equal(surface_attenuation(T(np.zeros((0, 1))), T(np.zeros((0, 1)))), T(np.zeros((0, 1))))


def test_surface_attenuation_with_the_light_surfaces_received():
    """Light that surfaces received beyond their direct irradiance reaches them indirectly, and is dimmed as the frame is
    on average, so that the glow that reaches a surface lit by other surfaces only, here the third, adds its albedo
    over π times it to its radiance, rather than being divided by the ambient share alone."""
    clear, through = T([[1.0], [1.0], [0.0]]), T([[0.5], [0.7], [0.3]])
    albedo = 0.4
    received = T([[0.9], [2.0], [1.5]])
    radiance = albedo / math.pi * received
    ratio = surface_attenuation(clear, through, received=received)
    mean = 1.5 / 2
    assert float(ratio[0, 0]) == pytest.approx((0.5 + mean * 0.2 * 2 / 3) / (1 + 0.2 * 2 / 3))
    assert float(ratio[1, 0]) == pytest.approx((0.7 + mean * 1.0) / 2.0)
    assert float(radiance[2, 0] * ratio[2, 0]) == pytest.approx(float(radiance[2, 0]) * mean + albedo / math.pi * 0.3)
    # Without it, the glow of the third surface is divided by the ambient share of the frame's direct light
    assert float(surface_attenuation(clear, through)[2, 0]) > 3 * float(ratio[2, 0])
    assert torch.equal(
        surface_attenuation(clear, through, received=torch.zeros_like(clear)), surface_attenuation(clear, through)
    )


def test_surface_attenuation_with_the_light_surfaces_received_from_their_own_albedo():
    """Light that surfaces received, as measured from their own albedo, replaces the ambient share of the frame's direct
    light, even where it is less: the surface lit by others then reflects its albedo over π times the glow on top of
    the rest of its light, dimmed as the frame is on average, however little light it received. Surfaces that received
    less than their direct irradiance are scaled by its ratio alone, and those whose light isn't known, or that
    received none, keep the ambient share."""
    clear, through = T([[1.0], [1.0], [0.0], [0.0], [0.0]]), T([[0.5], [0.7], [0.3], [0.3], [0.3]])
    albedo = 0.4
    received = T([[0.9], [2.0], [0.01], [0.01], [0.0]])
    measured = torch.tensor([[True], [True], [True], [False], [True]])
    radiance = albedo / math.pi * received
    ratio = surface_attenuation(clear, through, received=received, measured=measured)
    mean = 2.1 / 2
    assert float(ratio[0, 0]) == pytest.approx(0.5)
    assert float(ratio[1, 0]) == pytest.approx((0.7 + mean * 1.0) / 2.0)
    assert float(radiance[2, 0] * ratio[2, 0]) == pytest.approx(float(radiance[2, 0]) * mean + albedo / math.pi * 0.3)
    # As the ambient share, 0.2 · 2 / 5, is more than the light measured, it bounds it from below where it isn't
    # measured, and is kept where none was received
    floor = surface_attenuation(clear, through, received=received)
    assert torch.equal(ratio[3:], floor[3:]) and float(floor[2, 0]) < float(ratio[2, 0]) / 3
    assert float(ratio[4, 0]) == pytest.approx((0.3 + mean * 0.2 * 2 / 5) / (0.2 * 2 / 5))
    nowhere = torch.zeros_like(measured)
    assert torch.equal(surface_attenuation(clear, through, received=received, measured=nowhere), floor)
    # Some of the frame's surfaces, given its mean irradiances, are scaled as among all of them
    frame = (clear.mean(dim=0), through.mean(dim=0))
    for mask in (measured, None):
        some = surface_attenuation(
            clear[1:4], through[1:4], received=received[1:4], measured=None if mask is None else mask[1:4], frame=frame
        )
        every = surface_attenuation(clear, through, received=received, measured=mask)
        assert torch.allclose(some, every[1:4], rtol=1e-12)


def test_received_irradiance_from_the_albedo_of_the_ground_and_objects():
    from visionsim.medium.render import received_irradiance

    points = T([[0.0, 0.0, 0.01], [0.0, 0.0, 0.5], [1.0, 2.0, 3.0]])
    reflected = T([[0.1, 0.2], [0.1, 0.2], [-0.1, 0.2]])
    lighting = Lighting(ground_albedo=(0.4, 0.5), object_albedo=(0.2, 0.0))
    received = received_irradiance(lighting, points, reflected)
    assert received is not None
    expected = [[math.pi * 0.1 / 0.4, math.pi * 0.2 / 0.5], [math.pi * 0.1 / 0.2, 0.0], [0.0, 0.0]]
    assert torch.allclose(received, T(expected))
    assert received_irradiance(Lighting(), points, reflected) is None
    # The albedo of each surface replaces those of the lighting, and is unknown where it is zero
    albedo = T([[0.2, 0.0], [0.5, 0.25], [0.1, 0.1]])
    received = received_irradiance(Lighting(), points, reflected, albedo=albedo)
    assert received is not None
    expected = [[math.pi * 0.1 / 0.2, 0.0], [math.pi * 0.1 / 0.5, math.pi * 0.2 / 0.25], [0.0, math.pi * 0.2 / 0.1]]
    assert torch.allclose(received, T(expected))
    assert torch.equal(received_irradiance(lighting, points, reflected, albedo=albedo), received)
    assert received_irradiance(lighting, points, reflected, albedo=torch.zeros_like(albedo)) is None
    # Points only tell the ground from objects, which the albedo of each surface doesn't need
    assert torch.equal(received_irradiance(lighting, None, reflected, albedo=albedo), received)
    with pytest.raises(ValueError, match="Points"):
        received_irradiance(lighting, None, reflected)


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


def test_albedo_of_each_pixel():
    """Given the albedo of each pixel, the light that a textured ground received doesn't follow its texture, and
    neither does the ratio by which the fog scales its light, as on a plain ground that received the same light, while
    the mean albedo of the ground takes its dark squares for less lit, and its light ones for more, which changes it."""
    intrinsics, pose = camera(16, 12)
    origin, directions, scale = (x.numpy() for x in camera_rays(intrinsics, pose))
    with np.errstate(divide="ignore"):
        depth = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2] / scale, np.inf)
    normals = np.broadcast_to([0.0, 0.0, 1.0], (*depth.shape, 3))
    checker = np.where((np.arange(16)[None, :] + np.arange(12)[:, None]) % 2 == 0, 0.15, 0.6)[..., None]
    checker = np.repeat(checker, 3, axis=-1)
    # The lamp's light, dimmed more far from it, makes the ratio of surfaces vary over the ground
    medium, lamp = FOG, PointLight(position=(0.5, 5.0, 1.0), power=(300.0,), radius=0.05)
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.375,), "points": [lamp]})

    def run(texture, **kwargs):
        # Surfaces that received more light than their direct irradiance and the ambient share
        args = (2.0 * texture, depth, intrinsics, pose, medium, lighting)
        return apply_medium(*args, normals=normals, **kwargs).illumination

    textured, plain = run(checker, albedo=checker), run(np.full_like(checker, 0.6), albedo=np.full_like(checker, 0.6))
    ground = torch.as_tensor(np.isfinite(depth))
    assert torch.allclose(textured, plain, rtol=1e-12) and not torch.allclose(textured[ground], torch.ones(1).double())
    assert not torch.allclose(run(checker), textured, rtol=2e-3)
    # A grey albedo serves every channel, and the albedo of each pixel serves without those of the lighting
    assert torch.equal(run(checker, albedo=checker[..., :1]), textured)
    black = lighting.model_copy(update={"ground_albedo": (0.0,)})
    args = (2.0 * checker, depth, intrinsics, pose, medium, black)
    known = apply_medium(*args, normals=normals, albedo=checker).illumination
    assert not torch.allclose(apply_medium(*args, normals=normals).illumination, known)
    with pytest.raises(ValueError, match="Albedo"):
        run(checker, albedo=checker[:-1])
    # Alpha is left out, and grayscale frames take the mean of a color albedo
    assert torch.equal(run(checker, albedo=np.concatenate([checker, np.ones_like(checker[..., :1])], axis=-1)), textured)
    colored = checker * [1.2, 1.0, 0.8]
    grey = (2.0 * checker[..., :1], depth, intrinsics, pose, medium, lighting.model_copy(update={"sky": (0.35,)}))
    assert torch.allclose(
        apply_medium(*grey, normals=normals, albedo=colored).illumination,
        apply_medium(*grey, normals=normals, albedo=checker[..., :1]).illumination,
        rtol=1e-12,
    )


def test_irradiance_of_each_pixel():
    """The irradiance that each pixel's surface received measures the same light as its albedo and radiance do, and takes
    precedence over the albedo. The light that reaches surfaces indirectly is then the rest of it beyond their direct
    irradiance, as the medium's result gives it, or at least a share of the frame's direct light without
    ``measured_indirect``."""
    intrinsics, pose = camera(16, 12)
    origin, directions, scale = (x.numpy() for x in camera_rays(intrinsics, pose))
    with np.errstate(divide="ignore"):
        depth = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2] / scale, np.inf)
    normals = np.broadcast_to([0.0, 0.0, 1.0], (*depth.shape, 3))
    checker = np.where((np.arange(16)[None, :] + np.arange(12)[:, None]) % 2 == 0, 0.15, 0.6)[..., None]
    checker = np.repeat(checker, 3, axis=-1)
    lamp = PointLight(position=(0.5, 5.0, 1.0), power=(300.0,), radius=0.05)
    lighting = LIGHTING.model_copy(update={"ground_albedo": (0.375,), "points": [lamp]})
    # Surfaces on the right received more light than their direct irradiance, those on the left less
    received = np.broadcast_to(np.linspace(1.0, 12.0, 16)[None, :, None], checker.shape).copy()
    args = (checker * received / math.pi, depth, intrinsics, pose, FOG, lighting)
    by_albedo = apply_medium(*args, normals=normals, albedo=checker)
    by_irradiance = apply_medium(*args, normals=normals, irradiance=received)
    assert torch.allclose(by_irradiance.illumination, by_albedo.illumination, rtol=1e-12)
    assert torch.equal(
        apply_medium(*args, normals=normals, irradiance=received, albedo=0.5 * checker).radiance, by_irradiance.radiance
    )
    # The ratio, from the irradiance of surfaces with and without the medium, which the result holds, and the frame's
    # mean dimming
    ground = torch.as_tensor(np.isfinite(depth))
    clear, through = by_irradiance.clear_irradiance[ground], by_irradiance.surface_irradiance[ground]
    assert (by_irradiance.clear_irradiance[~ground] == 0).all() and (clear > 0).all()
    mean = through.sum(dim=0) / clear.sum(dim=0)
    indirect = (torch.as_tensor(received)[ground] - clear).clamp_min(0)
    assert (indirect == 0).any() and (indirect > 0).any()
    expected = (through + mean * indirect) / (clear + indirect)
    assert torch.allclose(by_irradiance.illumination[ground], expected, rtol=1e-12)
    # Without measured_indirect, a share of the frame's direct light bounds the light that reaches them indirectly
    floor = apply_medium(*args, normals=normals, irradiance=received, effects=Effects(measured_indirect=False))
    indirect = torch.maximum(indirect, 0.2 * clear.mean(dim=0))
    assert torch.allclose(floor.illumination[ground], (through + mean * indirect) / (clear + indirect), rtol=1e-12)
    assert not torch.allclose(floor.illumination, by_irradiance.illumination)


def test_mean_albedos_of_the_ground_and_objects():
    """The mean albedos of the ground and of objects, weighted by the light that each pixel's surface received, are
    those of grey surfaces that would reflect as much of it, which leave out the background and surfaces of no albedo,
    and add up over frames."""
    from visionsim.medium import mean_albedos

    intrinsics, pose = camera(16, 12)
    origin, directions, scale = (x.numpy() for x in camera_rays(intrinsics, pose))
    with np.errstate(divide="ignore"):
        depth = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2] / scale, np.inf)
    # A wall 3 m away in front of the middle columns, whose left half is red, lit less, and its right half blue
    wall = (np.arange(16) >= 6) & (np.arange(16) < 10) & (depth > 3)
    depth = np.where(wall, 3.0, depth)
    checker = np.where((np.arange(16)[None, :] + np.arange(12)[:, None]) % 2 == 0, 0.2, 0.6)[..., None]
    albedo = np.repeat(checker, 3, axis=-1)
    red, blue = wall & (np.arange(16) < 8), wall & (np.arange(16) >= 8)
    albedo[red], albedo[blue] = (0.8, 0.2, 0.2), (0.2, 0.2, 0.8)
    received = np.where(red, 1.0, np.where(blue, 3.0, 2.0))[..., None]
    radiance = albedo * received / np.pi
    # A metal, which reflects no diffuse light, but whose specular reflection is bright
    albedo[wall.nonzero()[0][0], 6], radiance[wall.nonzero()[0][0], 6] = 0.0, 5.0
    ground, objects = mean_albedos([(albedo, depth, intrinsics, pose, radiance)])
    lit = wall & (albedo[..., 0] > 0)
    assert objects == pytest.approx(tuple((albedo[lit] * received[lit]).sum(axis=0) / received[lit].sum()))
    assert objects[0] < 0.4 and objects[2] > 0.6
    on_ground = np.isfinite(depth) & ~wall
    assert ground == pytest.approx((checker[on_ground].mean(),) * 3)
    # Without the radiance, every pixel counts as much, and the same frames twice give the same albedos
    _, plain = mean_albedos([(albedo, depth, intrinsics, pose, None)] * 2)
    assert plain == pytest.approx(tuple(albedo[lit].mean(axis=0)))
    # A grey albedo serves every channel of the radiance, and no pixel sees objects without the wall
    flat = np.where(wall, np.inf, depth)
    grey, nothing = mean_albedos([(checker, flat, intrinsics, pose, np.repeat(checker, 3, axis=-1))])
    assert nothing is None and grey == pytest.approx((checker[on_ground].mean(),) * 3)
    # Alpha is left out, and a grey radiance takes the mean of a color albedo, as apply_medium does
    rgba = np.concatenate([albedo, np.ones_like(albedo[..., :1])], axis=-1)
    assert mean_albedos([(rgba, depth, intrinsics, pose, radiance)]) == mean_albedos(
        [(albedo, depth, intrinsics, pose, radiance)]
    )
    single, mean = radiance.mean(axis=-1, keepdims=True), albedo.mean(axis=-1)
    _, objects = mean_albedos([(albedo, depth, intrinsics, pose, single)])
    weights = single[lit, 0] / mean[lit]
    assert objects == pytest.approx(((mean[lit] * weights).sum() / weights.sum(),))


@pytest.mark.parametrize("at_once", [False, True])
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_interpolation_of_each_corner(dims, at_once):
    """The corners of tables around each point are interpolated at once, as on a GPU, as each of them in turn."""
    from visionsim.medium.surfaces import _interpolate

    rng = np.random.default_rng(dims)
    table = T(rng.uniform(size=(5, 4, 6)[:dims] + (3,)))
    indices = [T(rng.uniform(-0.5, size - 0.5, size=40)) for size in table.shape[:-1]]
    lows = [index.floor().clamp(0, size - 2).long() for index, size in zip(indices, table.shape)]
    fractions = [(index - low)[:, None] for index, low in zip(indices, lows)]
    expected = torch.zeros(40, 3, dtype=torch.float64)
    for corner in range(2**dims):
        weight = torch.ones_like(fractions[0])
        for d, fraction in enumerate(fractions):
            weight = weight * (fraction if (corner >> d) & 1 else 1 - fraction)
        expected = expected + weight * table[tuple(low + ((corner >> d) & 1) for d, low in enumerate(lows))]
    assert torch.equal(_interpolate(table, *indices, at_once=at_once), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_interpolation_of_each_corner_on_a_gpu(dims):
    """On a GPU, where the corners of tables around each point are interpolated at once, they add up as in turn."""
    from visionsim.medium.surfaces import _interpolate

    rng = np.random.default_rng(dims)
    kwargs = {"dtype": torch.float32, "device": "cuda"}
    table = torch.as_tensor(rng.uniform(size=(33, 17, 9)[:dims] + (3,)), **kwargs)
    indices = [torch.as_tensor(rng.uniform(-0.5, size - 0.5, size=100_000), **kwargs) for size in table.shape[:-1]]
    assert torch.equal(_interpolate(table, *indices), _interpolate(table, *indices, at_once=False))
