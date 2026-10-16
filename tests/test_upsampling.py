import numpy as np
import pytest
import torch

from visionsim.medium import HeightFog, Lighting, Medium, Sun, apply_medium, camera_rays
from visionsim.medium.model import PointLight
from visionsim.medium.upsampling import (
    _joint_upsample,
    apply_medium_upsampled,
    joint_upsample,
    low_resolution_camera,
    subsample,
)

FOG = Medium(
    extinction=0.08,
    anisotropy=0.8,
    components=[HeightFog(density=1.0, falloff=2.5)],
    sun_attenuation=True,
    multiple_scattering=True,
)
LIGHTING = Lighting(
    sky=(0.25, 0.35, 0.55),
    suns=[Sun(direction=(0.4, 0.7, 0.58), irradiance=(4.0,))],
    points=[PointLight(position=(1.0, 12.0, 2.0), power=(300.0,), radius=0.05)],
    ground_albedo=(0.4,),
)


def camera(w=64, h=48, height=1.6):
    """Camera looking horizontally along +Y, which sees the sky and the ground."""
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    pose[:3, 3] = [0.0, 0.0, height]
    return {"w": w, "h": h, "fl_x": 40.0, "fl_y": 40.0, "cx": w / 2, "cy": h / 2}, pose


def ground_depth(cam, pose):
    """Depth of the ground plane at z = 0, or infinity for rays that go up."""
    origin, directions, scale = camera_rays(cam, pose)
    along = torch.where(directions[..., 2] < 0, -origin[2] / directions[..., 2].clamp(max=-1e-9), torch.inf)
    return (along / scale).numpy()


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_low_resolution_camera_sees_the_full_resolution_rays(factor):
    cam, pose = camera()
    _, full, _ = camera_rays(cam, pose)
    _, small, _ = camera_rays(low_resolution_camera(cam, factor), pose)
    assert torch.allclose(small, subsample(full, factor), atol=1e-12)


def test_joint_upsample_keeps_constants_and_respects_depth_edges():
    depth = torch.ones(16, 16, dtype=torch.float64)
    depth[:, 8:] = 10.0  # two surfaces at very different depths
    values = torch.where(depth[..., None] > 5, 2.0, 1.0).expand(16, 16, 3).to(torch.float64).contiguous()
    low = subsample(values, 4)
    upsampled = joint_upsample(low, subsample(depth, 4), depth, 4)
    # Each side of the edge only takes the samples of its own depth
    assert torch.allclose(upsampled, values)
    constant = torch.full((4, 4, 2), 0.7, dtype=torch.float64)
    expected = torch.full((16, 16, 2), 0.7, dtype=torch.float64)
    assert torch.allclose(joint_upsample(constant, subsample(depth, 4), depth, 4), expected)


def test_factor_one_is_apply_medium():
    cam, pose = camera(16, 12)
    depth = ground_depth(cam, pose)
    radiance = np.full((*depth.shape, 3), 0.3)
    expected = apply_medium(radiance, depth, cam, pose, FOG, LIGHTING)
    result = apply_medium_upsampled(radiance, depth, cam, pose, FOG, LIGHTING, factor=1)
    assert torch.equal(result.radiance, expected.radiance)


@pytest.mark.parametrize("factor", [2, 4])
def test_upsampled_medium_is_close_to_the_full_resolution(factor):
    """The light that the medium scatters is smooth over the ground and the sky, and the transmittance exact."""
    cam, pose = camera()
    depth = ground_depth(cam, pose)
    radiance = np.full((*depth.shape, 3), 0.3)
    normals = np.zeros((*depth.shape, 3))
    normals[..., 2] = 1.0
    expected = apply_medium(radiance, depth, cam, pose, FOG, LIGHTING, normals=normals)
    result = apply_medium_upsampled(radiance, depth, cam, pose, FOG, LIGHTING, factor=factor, normals=normals)
    assert torch.equal(result.transmittance, expected.transmittance)
    error = (result.inscatter - expected.inscatter).abs().sum() / expected.inscatter.sum()
    assert error < 0.01
    # Interpolating across the horizon, between the far ground and the sky, is the worst case
    assert (result.radiance - expected.radiance).abs().sum() / expected.radiance.sum() < 0.005
    assert torch.allclose(result.radiance, expected.radiance, rtol=0.05)


def test_upsampled_surfaces_with_the_albedo_of_each_pixel():
    """Given the albedo of each pixel, the ratio by which the medium scales the light of a textured ground doesn't follow
    its texture, so that it is upsampled from samples that all see its dark squares as well as from any others."""
    cam, pose = camera()
    depth = ground_depth(cam, pose)
    checker = np.where((np.arange(64)[None, :] + np.arange(48)[:, None]) % 2 == 0, 0.15, 0.6)[..., None]
    radiance, normals = 2.0 * np.repeat(checker, 3, axis=-1), np.broadcast_to([0.0, 0.0, 1.0], (*depth.shape, 3))
    args = (radiance, depth, cam, pose, FOG.model_copy(update={"multiple_scattering": False}), LIGHTING)
    expected = apply_medium(*args, normals=normals, albedo=checker)
    every = apply_medium_upsampled(*args, factor=1, normals=normals, albedo=checker)
    assert torch.equal(every.radiance, expected.radiance)
    result = apply_medium_upsampled(*args, factor=2, normals=normals, albedo=checker)
    assert (result.radiance - expected.radiance).abs().sum() / expected.radiance.sum() < 0.002
    assert not torch.equal(result.radiance, apply_medium_upsampled(*args, factor=2, normals=normals).radiance)


def test_upsampled_surfaces_that_received_little_light():
    """Given the light that surfaces received, the ratio by which the medium scales their light grows without bound where
    they received little, such as in a sun's shadow on a dark ground, which the medium's glow lights. Upsampled, it would
    carry onto the lit pixels next to them: the ratio is formed at every pixel from upsampled irradiances instead, and
    the light of surfaces stays close to that of every pixel."""
    from tests.test_effects import MULTIPLE, SUN, albedo_of, scene

    occlusion, cam, pose, depth, normals, radiance = scene(96, 64)
    lighting = Lighting(suns=[SUN], ground_albedo=(0.3,))
    args = (cam, pose, MULTIPLE, lighting)
    options = {"occlusion": occlusion, "normals": normals}
    # The ground in the sun's shadow received a hundredth of the light of the rest
    shadow = (apply_medium(radiance, depth, *args, **options).clear_irradiance[..., 0] == 0).numpy()
    shadow &= depth < 1e9
    radiance = np.where(shadow[..., None], 0.01 * radiance, radiance)
    for measured in ({"albedo": albedo_of(96, 64)}, {"irradiance": np.pi * radiance / albedo_of(96, 64).clip(1e-3)}):
        expected = apply_medium(radiance, depth, *args, **options, **measured)
        result = apply_medium_upsampled(radiance, depth, *args, factor=4, **options, **measured)
        light = (expected.radiance - expected.inscatter) / expected.transmittance
        upsampled = (result.radiance - result.inscatter) / result.transmittance
        error = ((upsampled - light).abs() / light.clamp_min(1e-12)).mean(dim=-1)
        lit = torch.as_tensor(~shadow & (depth < 1e9))
        # The ratio of the samples themselves, interpolated, would make lit pixels next to the shadow much brighter
        low = apply_medium(
            subsample(radiance, 4),
            subsample(depth, 4),
            low_resolution_camera(cam, 4),
            *args[1:],
            occlusion=occlusion,
            normals=subsample(normals, 4),
            **{k: subsample(v, 4) for k, v in measured.items()},
        )
        guide = torch.as_tensor(np.minimum(depth, 1e9))
        assert low.illumination is not None and expected.illumination is not None
        ratio = joint_upsample(low.illumination, subsample(guide, 4), guide, 4)
        interpolated = ((ratio - expected.illumination).abs() / expected.illumination).mean(dim=-1)
        assert float(interpolated[lit].max()) > 3 and float(error[lit].max()) < 0.5
        assert float(error[torch.as_tensor(shadow)].max()) < 1
        assert (upsampled - light).abs().sum() / light.sum() < 0.02


def test_upsampling_refuses_shadows_of_every_pixel():
    cam, pose = camera(16, 12)
    depth = ground_depth(cam, pose)
    with pytest.raises(ValueError, match="Shadows"):
        apply_medium_upsampled(np.zeros((*depth.shape, 3)), depth, cam, pose, FOG, LIGHTING, factor=2, shadows=object())


@pytest.mark.parametrize("at_once", [False, True])
def test_joint_upsample_of_each_sample(at_once):
    """The four samples around each pixel are weighted at once, as on a GPU, as each of them in turn."""
    rng = np.random.default_rng(2)
    factor, low_shape = 3, (5, 7)
    depth = torch.as_tensor(rng.uniform(1, 3, size=(15, 21)))
    depth[:, 10:] *= 10  # an edge
    depth[:, 5] *= 30  # a thin object between samples, whose pixels take the closest sample
    low_depth = subsample(depth, factor) * torch.as_tensor(rng.uniform(0.9, 1.1, size=low_shape))
    values = torch.as_tensor(rng.uniform(size=(*low_shape, 2)))
    normals = torch.nn.functional.normalize(torch.as_tensor(rng.normal(size=(15, 21, 3))), dim=-1)
    low_normals = torch.nn.functional.normalize(subsample(normals, factor) + 0.3, dim=-1)

    def expected(low_normals=None, normals=None):
        fy = ((torch.arange(15.0, dtype=torch.float64) - factor // 2) / factor).clamp(0, low_shape[0] - 1)
        fx = ((torch.arange(21.0, dtype=torch.float64) - factor // 2) / factor).clamp(0, low_shape[1] - 1)
        y0, x0 = fy.floor().clamp(max=low_shape[0] - 2).long(), fx.floor().clamp(max=low_shape[1] - 2).long()
        wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
        total, weights, best, best_gap = 0, 0, None, None
        for dy in (0, 1):
            for dx in (0, 1):
                yi = (y0 + dy).clamp(max=low_shape[0] - 1)[:, None].expand(15, 21)
                xi = (x0 + dx).clamp(max=low_shape[1] - 1)[None, :].expand(15, 21)
                gap = (torch.log(low_depth)[yi, xi] - torch.log(depth)).abs()
                weight = (wy if dy else 1 - wy) * (wx if dx else 1 - wx) * torch.exp(-((gap / 0.05) ** 2) / 2)
                if normals is not None:
                    cosine = (low_normals[yi, xi] * normals).sum(dim=-1)
                    weight, gap = weight * torch.exp(-(1 - cosine) / 0.1), gap + (1 - cosine)
                total, weights = total + weight[..., None] * values[yi, xi], weights + weight
                if best is None:
                    best, best_gap = values[yi, xi], gap
                else:
                    closer = gap < best_gap
                    best, best_gap = (
                        torch.where(closer[..., None], values[yi, xi], best),
                        torch.where(closer, gap, best_gap),
                    )
        assert bool((weights <= 1e-3).any()) and bool((weights > 1e-3).any())
        return torch.where((weights > 1e-3)[..., None], total / weights.clamp_min(1e-30)[..., None], best)

    assert torch.equal(_joint_upsample(values, low_depth, depth, factor, None, None, at_once), expected())
    guided = _joint_upsample(values, low_depth, depth, factor, low_normals, normals, at_once)
    assert torch.equal(guided, expected(low_normals, normals))


def test_upsampled_shadows_and_surfaces():
    """Light shafts and surfaces lit through the medium, upsampled, stay close to those at every pixel."""
    from tests.test_occlusion import demo_scene

    occlusion, lighting, cam, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"anisotropy": 0.8})
    normals = np.zeros((*depth.shape, 3))
    normals[..., 1] = -1.0
    expected = apply_medium(radiance, depth, cam, pose, medium, lighting, occlusion=occlusion, normals=normals)
    result = apply_medium_upsampled(
        radiance, depth, cam, pose, medium, lighting, factor=2, occlusion=occlusion, normals=normals
    )
    assert torch.equal(result.transmittance, expected.transmittance)
    assert (result.inscatter - expected.inscatter).abs().sum() / expected.inscatter.sum() < 0.02
    assert (result.radiance - expected.radiance).abs().sum() / expected.radiance.sum() < 0.02


def test_upsampling_is_part_of_the_package():
    import visionsim.medium

    assert visionsim.medium.apply_medium_upsampled is apply_medium_upsampled


def guides(factor: int, shape: tuple[int, int], seed: int, **kwargs) -> tuple[torch.Tensor, ...]:
    """Values at samples, and the depths and normals of samples and pixels, with an edge and a thin object."""
    rng = np.random.default_rng(seed)
    low_shape = (shape[0] // factor, shape[1] // factor)
    depth = torch.as_tensor(rng.uniform(1, 3, size=shape), **kwargs)
    depth[:, shape[1] // 2 :] *= 10
    depth[:, shape[1] // 4 + 1] *= 30
    low_depth = subsample(depth, factor) * torch.as_tensor(rng.uniform(0.9, 1.1, size=low_shape), **kwargs)
    values = torch.as_tensor(rng.uniform(size=(*low_shape, 3)), **kwargs)
    normals = torch.nn.functional.normalize(torch.as_tensor(rng.normal(size=(*shape, 3)), **kwargs), dim=-1)
    low_normals = torch.nn.functional.normalize(subsample(normals, factor) + 0.3, dim=-1)
    return values, low_depth, depth, low_normals, normals


def test_closest_sample_where_guides_are_not_numbers():
    """Samples whose guides aren't numbers, e.g. normals of zero length, count as the farthest in either way of
    weighting samples."""
    values, low_depth, depth, low_normals, normals = guides(3, (15, 21), 3, dtype=torch.float64)
    low_normals[::2, ::3] = torch.nan
    at_once = _joint_upsample(values, low_depth, depth, 3, low_normals, normals, at_once=True)
    assert torch.equal(at_once, _joint_upsample(values, low_depth, depth, 3, low_normals, normals, at_once=False))
    assert bool(torch.isfinite(at_once).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_upsample_of_each_sample_on_a_gpu(dtype):
    """On a GPU, where the four samples around each pixel are weighted at once, they add up as each of them in turn,
    up to the order in which the cosines between normals are summed."""
    values, low_depth, depth, low_normals, normals = guides(4, (180, 320), 8, dtype=dtype, device="cuda")
    for low, full in ((None, None), (low_normals, normals)):
        at_once = joint_upsample(values, low_depth, depth, 4, low, full)
        in_turn = _joint_upsample(values, low_depth, depth, 4, low, full, at_once=False)
        assert torch.allclose(at_once, in_turn, rtol=1e-5 if dtype == torch.float32 else 1e-12, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_upsampled_shadows_and_surfaces_on_a_gpu():
    """On a GPU, which gathers samples, corners and rows at once and looks up the maps of the sky in larger chunks,
    frames are those of the CPU."""
    from tests.test_occlusion import demo_scene

    occlusion, lighting, cam, pose, depth, radiance = demo_scene()
    medium = FOG.model_copy(update={"anisotropy": 0.8, "multiple_scattering": True})
    normals = np.zeros((*depth.shape, 3))
    normals[..., 1] = -1.0
    for factor in (1, 2):
        kwargs = {"factor": factor, "occlusion": occlusion, "normals": normals}
        cpu = apply_medium_upsampled(radiance, depth, cam, pose, medium, lighting, device="cpu", **kwargs)
        gpu = apply_medium_upsampled(radiance, depth, cam, pose, medium, lighting, device="cuda", **kwargs)
        assert torch.allclose(gpu.transmittance.cpu(), cpu.transmittance, rtol=1e-12, atol=0)
        for field in ("inscatter", "radiance", "illumination"):
            a, b = getattr(gpu, field).cpu(), getattr(cpu, field)
            assert float((a - b).abs().sum() / b.abs().sum()) < 1e-6
