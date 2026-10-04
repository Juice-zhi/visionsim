import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import cumulative_trapezoid, quad, trapezoid

from visionsim.cli.emulate import spad
from visionsim.cli.medium import _write_exr, apply
from visionsim.dataset import Dataset, Metadata
from visionsim.medium import (
    Blob,
    HeightFog,
    Homogeneous,
    Lighting,
    Medium,
    Sun,
    apply_medium,
    camera_rays,
    kim_exponent,
)
from visionsim.medium.optics import (
    component_optical_depth,
    height_fog_sun_inscatter,
    henyey_greenstein,
    sky_quadrature,
)
from visionsim.medium.raymarch import ray_march_medium

T = torch.tensor
COMPONENTS = [
    Homogeneous(density=0.7),
    HeightFog(density=1.3, base_height=1.0, falloff=4.0),
    Blob(density=2.0, center=(1.0, 6.0, 1.5), radius=1.5, velocity=(0.5, -0.2, 0.1)),
]


def density(component, points, time=0.0):
    """Reference density of a component, evaluated at points of shape (..., 3)."""
    if isinstance(component, Homogeneous):
        return np.full(points.shape[:-1], component.density)
    if isinstance(component, HeightFog):
        return component.density * np.exp(-(points[..., 2] - component.base_height) / component.falloff)
    center = np.array(component.center) + time * np.array(component.velocity)
    return component.density * np.exp(-((points - center) ** 2).sum(-1) / (2 * component.radius**2))


def random_rays(rng, n):
    directions = rng.normal(size=(n, 3))
    return rng.uniform(-3, 3, size=3), directions / np.linalg.norm(directions, axis=-1, keepdims=True)


@pytest.mark.parametrize("component", COMPONENTS, ids=lambda c: c.type)
def test_optical_depth_matches_quadrature(component):
    rng = np.random.default_rng(0)
    origin, directions = random_rays(rng, 64)
    distance = rng.uniform(0.1, 30, size=64)
    directions[0] = [1.0, 0.0, 0.0]  # horizontal ray

    closed = component_optical_depth(component, T(origin), T(directions), T(distance), time=0.7).numpy()
    for o, v, d, tau in zip([origin] * 64, directions, distance, closed):

        def integrand(s, o=o, v=v):
            return density(component, o + s * v, time=0.7)

        ref, _ = quad(integrand, 0, d, epsabs=0, epsrel=1e-12, limit=200)
        assert tau == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_optical_depth_to_infinity():
    origin = T([0.0, 0.0, 2.0], dtype=torch.float64)
    directions = T([[0.0, 0.6, 0.8], [0.0, 0.6, -0.8], [1.0, 0.0, 0.0]], dtype=torch.float64)
    inf = torch.full((3,), torch.inf, dtype=torch.float64)

    fog, blob = COMPONENTS[1], COMPONENTS[2]
    k = fog.density * math.exp(-(2.0 - fog.base_height) / fog.falloff)
    up, down, horizontal = component_optical_depth(fog, origin, directions, inf).tolist()
    assert up == pytest.approx(k * fog.falloff / 0.8)
    assert down == math.inf and horizontal == math.inf

    # Towards its center, a blob integrates to sqrt(2*pi) * radius * density, minus the part behind the camera
    offset = T(blob.center, dtype=torch.float64) - origin
    total = component_optical_depth(blob, origin, (offset / offset.norm())[None], inf[:1]).item()
    behind = math.erfc(offset.norm().item() / (blob.radius * math.sqrt(2))) / 2
    assert total == pytest.approx(math.sqrt(2 * math.pi) * blob.radius * blob.density * (1 - behind), rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_height_fog_sun_inscatter_matches_quadrature(seed):
    rng = np.random.default_rng(seed)
    fog = HeightFog(density=rng.uniform(0.2, 2), base_height=rng.uniform(-2, 2), falloff=rng.uniform(1, 20))
    extinction, sun_z = rng.uniform(0.01, 0.3), rng.uniform(0.05, 1)
    origin, directions = random_rays(rng, 32)
    directions[0] = [np.sqrt(1 - sun_z**2), 0, sun_z]  # removable singularity, rays parallel to the sun's elevation
    directions[1] = [1.0, 0.0, 0.0]  # horizontal ray
    distance = rng.uniform(0.1, 60, size=32)

    def sigma(o, v, s):
        return extinction * density(fog, o + s[:, None] * v)

    tau = extinction * component_optical_depth(fog, T(origin), T(directions), T(distance)).numpy()
    k_ext = extinction * density(fog, origin)
    closed = height_fog_sun_inscatter(
        T(tau)[:, None], T([k_ext]), fog.falloff, T(directions[:, 2:3]), T(distance)[:, None], sun_z
    ).numpy()[:, 0]

    for v, d, value in zip(directions, distance, closed):
        s = np.linspace(0, d, 200_001)
        sig = sigma(origin, v, s)
        tau_s = cumulative_trapezoid(sig, s, initial=0)
        tau_sun = sig * fog.falloff / sun_z  # exponential fog above each point
        ref = trapezoid(sig * np.exp(-tau_s - tau_sun), s)
        assert value == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_height_fog_sun_inscatter_edge_cases():
    fog = HeightFog(density=1.0, falloff=5.0)
    k_ext, sun_z = T([0.2], dtype=torch.float64), 0.5
    dirs_z = T([[0.5], [0.3], [0.0], [-0.5], [0.3], [0.0], [-0.5]], dtype=torch.float64)
    dist = T([[10.0], [10.0], [10.0], [10.0], [torch.inf], [torch.inf], [torch.inf]], dtype=torch.float64)
    directions = torch.cat([torch.sqrt(1 - dirs_z**2), torch.zeros_like(dirs_z), dirs_z], dim=-1)
    tau = 0.2 * component_optical_depth(fog, T([0.0, 0.0, 0.0], dtype=torch.float64), directions, dist[:, 0])

    values = height_fog_sun_inscatter(tau[:, None], k_ext, fog.falloff, dirs_z, dist, sun_z)
    assert torch.isfinite(values).all() and (values >= 0).all()

    # Looking down into infinitely dense fog, all sunlight entering the fog along the ray gets scattered
    c0 = 0.2 * fog.falloff / sun_z
    assert values[-1].item() == pytest.approx(sun_z / (sun_z + 0.5) * math.exp(-c0))
    # No sunlight reaches the fog when the sun is below the horizon, and no fog means no scattering
    assert (height_fog_sun_inscatter(tau[:, None], k_ext, fog.falloff, dirs_z, dist, -0.1) == 0).all()
    assert (height_fog_sun_inscatter(tau[:, None] * 0, k_ext * 0, fog.falloff, dirs_z, dist, sun_z) == 0).all()


def test_henyey_greenstein_is_normalized():
    theta = np.linspace(0, np.pi, 200_001)
    for g in (-0.5, 0.0, 0.85):
        phase = henyey_greenstein(T(np.cos(theta)), g).numpy()
        assert trapezoid(2 * np.pi * phase * np.sin(theta), theta) == pytest.approx(1.0, rel=1e-8)


def test_visibility():
    medium = Medium.from_visibility(100.0, components=[Homogeneous()])
    # The contrast of a black object at the visibility distance is the threshold
    assert math.exp(-medium.extinction * 100.0) == pytest.approx(0.02)
    assert Medium.from_visibility(100.0, contrast_threshold=0.05).extinction == pytest.approx(math.log(20) / 100)
    assert [kim_exponent(v) for v in (100, 800, 3000, 10_000, 60_000)] == pytest.approx([0, 0.3, 0.82, 1.3, 1.6])


def test_medium_validation():
    with pytest.raises(ValueError, match="single `HeightFog`"):
        Medium(extinction=0.1, components=[Homogeneous()], sun_attenuation=True)
    assert Medium.model_validate_json(Medium(extinction=0.1, components=COMPONENTS).model_dump_json()).components == (
        COMPONENTS
    )


def small_camera(w=7, h=5):
    angle = 0.3
    pose = np.eye(4)
    # Look along +Y, slightly upwards, from 1.5m above the ground
    pose[:3, :3] = [[1, 0, 0], [0, -np.sin(angle), np.cos(angle)], [0, np.cos(angle), np.sin(angle)]]
    pose[:3, 3] = [0.3, -4.0, 1.5]
    return {"w": w, "h": h, "fl_x": 6.0, "fl_y": 6.0, "cx": 3.2, "cy": 2.6}, pose


def reference_render(medium, lighting, camera, pose, radiance, depth, wavelengths, time=0.0):
    """Brute-force single scattering along each pixel's ray, independently of the closed forms."""
    origin, directions, scale = (x.numpy() for x in camera_rays(camera, pose))
    beta = np.array(medium.extinction_at(wavelengths))
    result = np.zeros_like(radiance)

    for i, j in np.ndindex(depth.shape):
        v, d = directions[i, j], depth[i, j] * scale[i, j]
        s = np.linspace(0, d, 100_001)
        points = origin + s[:, None] * v
        rho = sum(density(c, points, time) for c in medium.components)
        tau = cumulative_trapezoid(rho, s, initial=0)[:, None] * beta
        source = medium.albedo * np.broadcast_to(lighting.ambient, (len(s), len(beta)))

        for sun in lighting.suns:
            towards = np.array(sun.direction) / np.linalg.norm(sun.direction)
            phase = henyey_greenstein(T(v @ towards), medium.anisotropy).item()
            sun_tau = 0.0
            if medium.sun_attenuation:
                fog = medium.components[0]
                sun_tau = density(fog, points)[:, None] * beta * fog.falloff / towards[2]
            source = source + medium.albedo * phase * np.array(sun.irradiance) * np.exp(-sun_tau)

        inscatter = trapezoid(rho[:, None] * beta * np.exp(-tau) * source, s, axis=0)
        result[i, j] = np.exp(-tau[-1]) * radiance[i, j] + inscatter
    return result


@pytest.mark.parametrize(
    "medium",
    [
        Medium(extinction=0.08, angstrom=1.3, albedo=0.9, anisotropy=0.6, components=COMPONENTS),
        Medium(extinction=0.1, components=[HeightFog(density=2.0, falloff=3.0)], sun_attenuation=True),
    ],
    ids=["mixture", "attenuated-sun"],
)
def test_apply_medium_matches_brute_force(medium):
    rng = np.random.default_rng(1)
    camera, pose = small_camera()
    radiance = rng.uniform(0, 2, size=(5, 7, 3))
    depth = rng.uniform(1, 25, size=(5, 7))
    lighting = Lighting(
        ambient=(0.3, 0.4, 0.6),
        suns=[
            Sun(direction=(0.2, 1.0, 0.6), irradiance=(3.0, 2.8, 2.5)),
            Sun(direction=(-1, 0, 0.3), irradiance=(1.0,)),
        ],
    )

    result = apply_medium(radiance, depth, camera, pose, medium, lighting, time=0.4)
    reference = reference_render(medium, lighting, camera, pose, radiance, depth, (610.0, 550.0, 465.0), time=0.4)
    assert np.allclose(result.radiance.numpy(), reference, rtol=1e-6, atol=1e-9)

    # Ground truth is consistent with the composited radiance
    assert torch.allclose(result.transmittance, torch.exp(-result.optical_depth))
    assert torch.allclose(result.radiance, result.transmittance * T(radiance) + result.inscatter)


def sky_source(v, g, c_values, n_theta=300, n_phi=600):
    """Brute-force skylight scattered along direction v, for a unit sky above the horizon, after it was attenuated
    by an optical depth of ``c / ω_z`` along each direction ω, for each value of c."""
    theta = (np.arange(n_theta) + 0.5) / n_theta * np.pi / 2
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    omega = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    weights = (np.sin(th) * (np.pi / 2 / n_theta) * (2 * np.pi / n_phi)).reshape(-1)
    weights = weights * henyey_greenstein(T(omega @ v), g).numpy()
    return np.exp(-np.outer(np.atleast_1d(c_values), 1 / omega[:, 2])) @ weights


@pytest.mark.parametrize("g", [0.0, 0.6, 0.85, -0.3])
def test_sky_quadrature_matches_brute_force(g):
    for v_z in (-0.9, -0.3, -0.05, 0.0, 0.1, 0.5, 1.0):
        v = np.array([math.sqrt(1 - v_z * v_z), 0.0, v_z])
        elevations, weights = sky_quadrature(T(v)[None], g)
        for c, expected in zip((0.0, 0.05, 0.5, 5.0), sky_source(v, g, (0.0, 0.05, 0.5, 5.0))):
            value = (weights * torch.exp(-c / elevations.clamp_min(1e-300)) * (elevations > 0)).sum().item()
            assert value == pytest.approx(expected, rel=2e-3, abs=1e-6)


@pytest.mark.parametrize(
    "medium",
    [
        Medium(extinction=0.1, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True),
        Medium(extinction=0.08, angstrom=1.3, albedo=0.9, anisotropy=0.6, components=COMPONENTS),
    ],
    ids=["attenuated", "unattenuated"],
)
def test_apply_medium_sky_matches_brute_force(medium):
    rng = np.random.default_rng(3)
    camera, pose = small_camera(w=3, h=2)
    camera = camera | {"fl_x": 1.5, "fl_y": 1.5, "cx": 1.5, "cy": 1.0}
    origin, directions, scale = (x.numpy() for x in camera_rays(camera, pose))

    # Rays going down stop on the ground, which is what blocks light from below the horizon
    with np.errstate(divide="ignore"):
        to_ground = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2], np.inf)
    depth = np.minimum(rng.uniform(2, 40, size=(2, 3)), to_ground / scale)
    sky = np.array([0.2, 0.3, 0.5])
    result = apply_medium(np.zeros((2, 3, 3)), depth, camera, pose, medium, Lighting(sky=tuple(sky)))

    beta = np.array(medium.extinction_at((610.0, 550.0, 465.0)))
    for i, j in np.ndindex(depth.shape):
        v = directions[i, j]
        s = np.linspace(0, depth[i, j] * scale[i, j], 4001)
        points = origin + s[:, None] * v
        rho = sum(density(c, points) for c in medium.components)
        tau = cumulative_trapezoid(rho, s, initial=0)[:, None] * beta

        if medium.sun_attenuation:
            # Skylight is attenuated by the fog above each point, i.e. an optical depth of density * falloff / ω_z
            fog = medium.components[0]
            c_grid = np.linspace(0, rho.max() * beta.max() * fog.falloff, 400)
            table = sky_source(v, medium.anisotropy, c_grid)
            source = np.stack([np.interp(rho * b * fog.falloff, c_grid, table) for b in beta], -1)
        else:
            source = np.broadcast_to(sky_source(v, medium.anisotropy, 0.0), (len(s), len(beta)))

        inscatter = trapezoid(rho[:, None] * beta * np.exp(-tau) * medium.albedo * sky * source, s, axis=0)
        assert result.inscatter[i, j].numpy() == pytest.approx(inscatter, rel=3e-3)


@pytest.mark.parametrize(
    "medium",
    [
        Medium(extinction=0.1, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True),
        Medium(extinction=0.08, angstrom=1.3, albedo=0.9, anisotropy=0.6, components=COMPONENTS),
    ],
    ids=["attenuated", "unattenuated"],
)
def test_ray_marching_converges_to_closed_form(medium):
    rng = np.random.default_rng(4)
    camera, pose = small_camera()
    lighting = Lighting(sky=(0.2, 0.3, 0.5), ambient=(0.05,), suns=[Sun(direction=(0.2, 1.0, 0.6), irradiance=(3.0,))])
    if medium.sun_attenuation:
        # Light from the ground and multiple scattering are tabulated like skylight, and marched through alike
        medium = medium.model_copy(update={"multiple_scattering": True})
        lighting = lighting.model_copy(update={"ground_albedo": (0.4,), "ground_height": -0.5})
    args = (rng.uniform(0, 2, size=(5, 7, 3)), rng.uniform(1, 25, size=(5, 7)), camera, pose, medium)
    exact = apply_medium(*args, lighting, time=0.4).radiance

    def error(**kwargs):
        marched = ray_march_medium(*args, lighting, time=0.4, **kwargs).radiance
        return ((marched - exact).abs().sum() / exact.abs().sum()).item()

    errors = [error(steps=steps) for steps in (4, 16, 64, 256)]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-4
    if medium.sun_attenuation:
        # Marching towards the sun converges too, but adds its own error
        assert errors[-1] < error(steps=256, shadow_steps=64) < 1e-3


@pytest.mark.parametrize(
    "medium",
    [
        Medium(extinction=0.1, anisotropy=0.8, components=[HeightFog(density=1.0, falloff=2.5)], sun_attenuation=True),
        Medium(extinction=0.08, angstrom=1.3, albedo=0.9, anisotropy=0.6, components=COMPONENTS),
    ],
    ids=["attenuated", "unattenuated"],
)
def test_chunks_of_pixels_give_the_same_result(medium, monkeypatch):
    rng = np.random.default_rng(5)
    camera, pose = small_camera()
    args = (rng.uniform(0, 2, size=(5, 7, 3)), rng.uniform(1, 25, size=(5, 7)), camera, pose, medium)
    lighting = Lighting(sky=(0.2, 0.3, 0.5), ambient=(0.05,), suns=[Sun(direction=(0.2, 1.0, 0.6), irradiance=(3.0,))])
    whole = [apply_medium(*args, lighting), ray_march_medium(*args, lighting, steps=8)]

    # Skylight is otherwise integrated in chunks of pixels at high resolutions only
    monkeypatch.setattr("visionsim.medium.render._CHUNK_ELEMENTS", 4000)
    chunked = [apply_medium(*args, lighting), ray_march_medium(*args, lighting, steps=8)]
    for expected, result in zip(whole, chunked):
        for a, b in zip(expected, result):
            # Surfaces aren't lit through the medium without their normals
            assert (a is None and b is None) or torch.allclose(a, b, rtol=1e-12, atol=0)


def test_apply_medium_background_and_extra_channels():
    camera, pose = small_camera()
    radiance = np.ones((5, 7, 4))
    depth = np.full((5, 7), 1e10)  # Blender's depth for the background
    lighting = Lighting(ambient=(0.5,))

    # In a homogeneous medium the background is hidden behind the airlight, as in Koschmieder's model
    fog = apply_medium(radiance, depth, camera, pose, Medium(extinction=0.05, albedo=0.8), lighting)
    assert torch.allclose(fog.radiance[..., :3], T(0.8 * 0.5, dtype=torch.float64))
    assert (fog.radiance[..., 3] == 1).all() and fog.transmittance.shape == (5, 7, 3)

    # Whereas with a height fog, the sky is seen through a finite amount of fog, but not the ground
    medium = Medium(extinction=0.05, components=[HeightFog(density=1.0, falloff=10.0)])
    fog = apply_medium(radiance, depth, camera, pose, medium, lighting)
    upwards = camera_rays(camera, pose)[1][..., 2] > 0
    assert upwards.any() and (~upwards).any() and torch.isfinite(fog.radiance).all()
    assert ((fog.transmittance[upwards] > 0) & (fog.transmittance[upwards] < 1)).all()
    assert (fog.transmittance[~upwards] == 0).all()

    with pytest.raises(ValueError, match="1 or 3 values"):
        apply_medium(radiance, depth, camera, pose, medium, Lighting(ambient=(1.0, 2.0)))


def make_render(root, n=3, gray=False, keyframe_scale=1.0):
    """Synthetic render, with linear EXR frames, depth maps (including background pixels) and lighting."""
    camera, pose = small_camera()
    rng = np.random.default_rng(2)
    frames, depths = [], []

    for i in range(n):
        path = Path("0000") / f"{i:03}.exr"
        depth = rng.uniform(1, 20, size=(5, 7, 1))
        depth[0, :2] = 1e10
        radiance = rng.uniform(0, 2, size=(5, 7, 1 if gray else 3))
        _write_exr(root / "frames" / path, np.repeat(radiance, 3, axis=-1) if gray else radiance)
        _write_exr(root / "depths" / path, depth)
        transform = camera | {"file_path": path, "transform_matrix": pose.tolist(), "fps": 24.0}
        transform["keyframe_scale"] = keyframe_scale
        frames.append(transform | {"c": 3})
        depths.append(transform | {"c": 1})

    Metadata.from_dense_transforms(frames).save(root / "frames" / "transforms.json")
    Metadata.from_dense_transforms(depths).save(root / "depths" / "transforms.json")
    lighting = Lighting(ambient=(0.3, 0.4, 0.5), suns=[Sun(direction=(0, 1, 1), irradiance=(2.0,))])
    (root / "lighting.json").write_text(lighting.model_dump_json())
    return lighting


@pytest.mark.parametrize("keyframe_scale", [1.0, 5.0])
def test_cli_apply(tmp_path, keyframe_scale):
    lighting = make_render(tmp_path / "render", keyframe_scale=keyframe_scale)
    medium = Medium(
        extinction=0.1,
        components=[Homogeneous(density=0.5), Blob(center=(0, 5, 1.5), radius=2, velocity=(4, 0, 0))],
    )
    (tmp_path / "medium.json").write_text(medium.model_dump_json())
    apply(tmp_path / "render", tmp_path / "fog", tmp_path / "medium.json", device="cpu")

    frames = Dataset.from_path(tmp_path / "render" / "frames")
    depths = Dataset.from_path(tmp_path / "render" / "depths")
    # Output directories, in the same order as the fields of `MediumResult`
    names = ("frames", "transmittance", "optical-depth", "inscatter")
    outputs = {name: Dataset.from_path(tmp_path / "fog" / name) for name in names}
    assert Medium.model_validate_json((tmp_path / "fog" / "medium.json").read_text()) == medium

    for i, ((radiance, transform), (depth, _)) in enumerate(zip(frames, depths)):
        pose = transform["transform_matrix"]
        # Frames rendered with a keyframe multiplier are closer in time
        expected = apply_medium(radiance, depth, transform, pose, medium, lighting, time=i / (24 * keyframe_scale))

        for name, value in zip(names, expected):
            data, saved = outputs[name][i]
            assert np.allclose(data, value.numpy(), rtol=1e-6, atol=1e-7)
            assert np.allclose(saved["transform_matrix"], pose) and saved["fl_x"] == transform["fl_x"]

    # The blob moves, so frames differ even though the scene doesn't
    assert not np.allclose(outputs["frames"][0][0], outputs["frames"][-1][0])

    # Frames with the medium can be used as is by emulators
    spad(tmp_path / "fog" / "frames", tmp_path / "spad", seed=1)
    binary, _ = Dataset.from_path(tmp_path / "spad")[0]
    assert binary.shape == (5, 7, 3) and set(np.unique(binary)) <= {0, 1}


def test_cli_apply_lights_surfaces_through_the_medium(tmp_path):
    lighting = make_render(tmp_path / "render", n=2)
    rng = np.random.default_rng(4)
    frames = Dataset.from_path(tmp_path / "render" / "frames")
    for name in ("normals", "emission"):
        transforms = []
        for i, (_, transform) in enumerate(frames):
            path = Path("0000") / f"{i:03}.exr"
            data = rng.normal(size=(5, 7, 3)) if name == "normals" else rng.uniform(0, 0.3, size=(5, 7, 3))
            _write_exr(tmp_path / "render" / name / path, data)
            transforms.append(transform | {"file_path": path, "c": 3})
        Metadata.from_dense_transforms(transforms).save(tmp_path / "render" / name / "transforms.json")
    medium = Medium(extinction=0.1, components=[HeightFog(falloff=3.0)], sun_attenuation=True)
    (tmp_path / "medium.json").write_text(medium.model_dump_json())
    apply(tmp_path / "render", tmp_path / "fog", tmp_path / "medium.json", device="cpu")

    normals, emission = (Dataset.from_path(tmp_path / "render" / name) for name in ("normals", "emission"))
    for i, (radiance, transform) in enumerate(frames):
        depth = Dataset.from_path(tmp_path / "render" / "depths")[i][0]
        # Normals are saved in the camera's space, and emitted light isn't dimmed
        expected = apply_medium(
            radiance,
            depth,
            transform,
            transform["transform_matrix"],
            medium,
            lighting,
            normals=normals[i][0],
            normals_space="camera",
            emission=emission[i][0],
        )
        saved = Dataset.from_path(tmp_path / "fog" / "frames")[i][0]
        illumination = Dataset.from_path(tmp_path / "fog" / "illumination")[i][0]
        assert np.allclose(saved, expected.radiance.numpy(), rtol=1e-6, atol=1e-6)
        assert np.allclose(illumination, expected.illumination.numpy(), rtol=1e-6)
        assert not np.allclose(illumination, 1)

    # Without normals, or when asked not to, surfaces keep their light
    apply(tmp_path / "render", tmp_path / "plain", tmp_path / "medium.json", device="cpu", surfaces=False)
    assert not (tmp_path / "plain" / "illumination").exists()


def test_cli_apply_requires_lighting(tmp_path):
    make_render(tmp_path / "render")
    (tmp_path / "render" / "lighting.json").unlink()
    (tmp_path / "medium.json").write_text(Medium(extinction=0.1).model_dump_json())

    with pytest.raises(FileNotFoundError, match="--include-lighting"):
        apply(tmp_path / "render", tmp_path / "fog", tmp_path / "medium.json", device="cpu")


def test_cli_apply_gray_frames(tmp_path):
    # RGB frames of a gray scene are loaded as a single channel, but should still be lit in color
    make_render(tmp_path / "render", n=1, gray=True)
    (tmp_path / "medium.json").write_text(Medium(extinction=0.1, angstrom=1.3).model_dump_json())
    apply(tmp_path / "render", tmp_path / "fog", tmp_path / "medium.json", device="cpu")

    frame, transform = Dataset.from_path(tmp_path / "fog" / "frames")[0]
    assert frame.shape == (5, 7, 3) and transform["c"] == 3
    assert not np.allclose(frame[..., 0], frame[..., 2])  # extinction depends on wavelength
