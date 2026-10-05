import itertools
import math
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import OpenEXR
import pytest
import torch
from peewee import SqliteDatabase

from visionsim.dataset import Dataset, Metadata
from visionsim.medium import AnimatedLighting, Lighting
from visionsim.medium.occlusion import lamp_shadow, lamp_visibility, load_occlusion, visibility
from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER, BlenderClients
from visionsim.simulate.schema import _MODELS, _Data


@pytest.mark.parametrize(
    "gt_type",
    [
        "composites",
        "frames",
        "depths",
        "normals",
        "flows",
        "segmentations",
        "previews/depths",
        "previews/normals",
        "previews/flows/forward",
        "previews/segmentations",
        "previews/materials",
        "materials",
        "diffuse/color",
        "diffuse/direct",
        "diffuse/indirect",
        # "diffuse/light",
        "specular/color",
        "specular/direct",
        "specular/indirect",
        # "specular/light",
        "points",
        "previews/points",
        "emission",
    ],
)
def test_render_layout(cube_dataset, gt_type):
    assert not (cube_dataset / "transforms.json").exists()
    assert not (cube_dataset / "transforms.db").exists()

    subdir = cube_dataset / gt_type
    assert subdir.exists()
    assert not (subdir / "transforms.json").exists()
    assert (subdir / "transforms.db").exists()

    if gt_type in ("frames", "composites") or "previews" in gt_type:
        assert len(list(subdir.glob("**/*.png"))) == 5
    else:
        assert len(list(subdir.glob("**/*.exr"))) == 5


@pytest.mark.parametrize(
    "subdir, channels",
    [
        ("depths", ["V"]),
        ("normals", ["RGB"]),
        ("flows", ["RGBA"]),
        ("segmentations", ["V"]),
        ("materials", ["V"]),
        ("diffuse/color", ["RGB"]),
        ("diffuse/direct", ["RGB"]),
        ("diffuse/indirect", ["RGB"]),
        # ("diffuse/light", ["RGB"]),
        ("specular/color", ["RGB"]),
        ("specular/direct", ["RGB"]),
        ("specular/indirect", ["RGB"]),
        # ("specular/light", ["RGB"]),
        ("points", ["RGB"]),
        ("emission", ["RGB"]),
    ],
)
def test_groundtruth_exrs(cube_dataset, subdir, channels):
    for file in cube_dataset.glob(f"{subdir}/**/*.exr"):
        with OpenEXR.File(str(file)) as f:
            # Before v4 exr's couldn't be single channel, they were saved as
            # RGB with duplicated channels.
            if channels == ["V"] and "V" not in f.channels():
                assert "RGB" in f.channels()
                data = f.channels()["RGB"].pixels.transpose(2, 0, 1)
                assert all(np.allclose(a, b) for a, b in itertools.pairwise(data))
                channels = ["RGB"]
            else:
                assert list(f.channels().keys()) == channels

            for channel in channels:
                assert np.issubdtype(f.channels()[channel].pixels.dtype, np.floating)


@pytest.mark.parametrize(
    "subdir, shape, auto_collapse",
    [
        ("depths", (50, 50, 1), True),
        ("normals", (50, 50, 3), False),
        ("flows", (50, 50, 4), False),
        ("segmentations", (50, 50, 1), True),
        ("materials", (50, 50, 1), True),
        ("diffuse/color", (50, 50, 3), False),
        ("diffuse/direct", (50, 50, 3), False),
        ("diffuse/indirect", (50, 50, 3), False),
        # ("diffuse/light", (50, 50, 3)),
        ("specular/color", (50, 50, 3), False),
        ("specular/direct", (50, 50, 3), False),
        ("specular/indirect", (50, 50, 3), False),
        # ("specular/light", (50, 50, 3)),
        ("points", (50, 50, 3), False),
        ("emission", (50, 50, 3), False),
    ],
)
def test_load_exrs(cube_dataset, subdir, shape, auto_collapse):
    for file in cube_dataset.glob(f"{subdir}/**/*.exr"):
        assert Dataset.load_data(file, auto_collapse=auto_collapse).shape == shape


def test_transforms_schema(cube_dataset):
    for path in cube_dataset.glob("**/*.db"):
        Metadata.load(path)


def test_data_paths_exist(cube_dataset):
    for db_path in cube_dataset.glob("**/*.db"):
        db = SqliteDatabase(db_path)
        with db.connection_context(), db.bind_ctx(_MODELS):
            for data in _Data.select():
                assert (db_path.parent / data.path).exists()


def test_database_threading(tmp_path_factory, executable):
    tmpdir = tmp_path_factory.mktemp("renders")
    log_dir = tmp_path_factory.mktemp("logs")
    scene = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"

    # Spoof frames to bypass render, only save metadata, from a bunch of blender instances.
    # This forces a lot of database writes, which helps test for any potential "Database is locked" errors.
    with BlenderClients.spawn(jobs=os.cpu_count() or 5, executable=executable, timeout=30, log=log_dir) as clients:
        clients.initialize(scene.resolve(), tmpdir.resolve())
        clients.include_frames()
        clients.move_keyframes(scale=5)

        for idx in clients.common_animation_range():
            folder_index = f"{idx // ITEMS_PER_SUBFOLDER:04}"
            frame_index = f"{idx % ITEMS_PER_SUBFOLDER:0{INDEX_PADDING}}"
            frame = tmpdir / "frames" / folder_index / f"{frame_index}.png"
            frame.parent.mkdir(exist_ok=True, parents=True)
            frame.touch()
        clients.render_animation()


def test_metadata_roundtrip_from_db(cube_dataset):
    for path in cube_dataset.glob("**/*.db"):
        meta = Metadata.load(path)
        meta.save(path.parent / "transforms.json")

        assert Metadata.load(path.parent / "transforms.json").model_dump() == meta.model_dump()


def _run_blender_script(executable, name, *args):
    # Some checks need to access Blender's API directly, so they run from within Blender itself
    script = Path(__file__).parent / "blender_scripts" / name
    scene = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"
    cmd = shlex.split(f"{executable or 'blender'} -b --factory-startup --python-exit-code 1")
    result = subprocess.run(
        [*cmd, "--python", str(script), "--", str(scene), *map(str, args)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_camera_intrinsics(executable):
    # Checks intrinsics against Blender's own projection
    _run_blender_script(executable, "camera_intrinsics.py")


def test_save_occlusion(executable, tmp_path):
    # Shadow maps are rendered from within Blender, which checks that the scene is left as it was, then the maps of
    # the test scene's cube, lit by a sun that leans towards -y, and of a glowing ball above it, are checked here
    _run_blender_script(executable, "occlusion.py", tmp_path / "occlusion.npz")
    occlusion = load_occlusion(tmp_path / "occlusion.npz", dtype=torch.float64)
    assert occlusion.sky_counts == (4, 2) and len(occlusion.sky_maps.texels) == 6 and len(occlusion.sun_maps.texels) == 1
    low, high = occlusion.bounds.numpy()
    # Only the cube and the ball above it cast shadows
    assert np.allclose(low, -1, atol=0.05) and np.allclose(high, [1, 1, 4.25], atol=0.05)

    # Points right below the cube, along the sun's direction, are in its shadow, unlike points beside it
    towards = occlusion.sun_directions[0]
    center = torch.zeros(3, dtype=torch.float64)
    below = center - towards * 2 * math.sqrt(3)
    beside = below + torch.as_tensor([4.0, 0.0, 0.0], dtype=torch.float64)
    origin = center + torch.as_tensor([0.0, 0.0, 50.0], dtype=torch.float64)
    points = torch.stack([below, beside])
    offset = points - origin
    lit = visibility(
        occlusion.sun_maps, origin, offset / offset.norm(dim=-1, keepdim=True), offset.norm(dim=-1)[:, None]
    )
    assert lit[:, 0, 0].tolist() == [0.0, 1.0]

    # The scene's point light has a map too, behind the cube from which points are in its shadow, unlike points
    # in front of the light or beside the cube
    (lamp,), (ball,) = occlusion.lamp_maps.origins, occlusion.emissive_maps.origins
    assert tuple(occlusion.lamp_maps.shapes[0].tolist()) == (128, 256)
    away = (center - lamp) / (center - lamp).norm()
    side = torch.linalg.cross(away, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    points = torch.stack([center + 3 * away, lamp - 3 * away, center + 3 * away + 4 * side / side.norm()])
    assert lamp_visibility(occlusion.lamp_maps, 0, points).tolist() == [0.0, 1.0, 1.0]

    # Averaged along a ray that crosses the shadow, the visibility matches the mean of many samples, evenly spread in
    # the angle at which the lamp sees them
    origin = center + 3 * away - 6 * side / side.norm()
    end = center + 3 * away + 6 * side / side.norm()
    direction = (end - origin) / (end - origin).norm()
    length = (end - origin).norm()[None]
    shadow = lamp_shadow(occlusion.lamp_maps, 0, origin, direction[None], length, samples=1024)
    mean = shadow.average(torch.stack([torch.zeros_like(length), length], dim=-1))
    along = float(direction @ (lamp - origin))
    closest = float((origin + along * direction - lamp).norm())
    angles = torch.linspace(math.atan2(-along, closest), math.atan2(length.item() - along, closest), 20001)
    samples = origin + (along + closest * torch.tan(angles.double()))[:, None] * direction
    assert 0.1 < mean.item() < 0.9
    assert mean.item() == pytest.approx(lamp_visibility(occlusion.lamp_maps, 0, samples).mean().item(), abs=0.01)

    # The glowing ball casts shadows from its center, through itself but not through the cube
    assert (ball - torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)).norm().item() < 0.01
    through = ball + torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
    under = ball + 2 * (center - ball)
    assert lamp_visibility(occlusion.emissive_maps, 0, torch.stack([through, under])).tolist() == [1.0, 0.0]


def test_animated_lighting(executable, tmp_path):
    # A point light rises and brightens over frames, which are all saved as they all change, and gets maps along its way,
    # wherever it moved further than the spacing from where its maps were rendered
    _run_blender_script(executable, "animated.py", tmp_path)
    animated = AnimatedLighting.model_validate_json((tmp_path / "animated-lighting.json").read_text())
    assert sorted(animated.frames) == [1, 2, 3, 4, 5]
    assert Lighting.model_validate_json((tmp_path / "lighting.json").read_text()) == animated.frames[1]
    heights = [animated.frames[frame].points[0].position[2] for frame in range(1, 6)]
    powers = [animated.frames[frame].points[0].power[0] for frame in range(1, 6)]
    assert np.allclose(np.diff(heights), 0.5)
    assert np.allclose(np.asarray(powers) / powers[0], [1, 2.5, 4, 4, 4])

    occlusion = load_occlusion(tmp_path / "occlusion.npz", dtype=torch.float64)
    assert occlusion.lamp_tolerance == 0.75 and occlusion.emissive_maps is not None
    assert np.allclose(occlusion.lamp_maps.origins[:, 2].numpy() - heights[0], [0, 1, 2])


def test_lighting_info(executable, tmp_path):
    # Lights are added and checked from within Blender, the saved lighting is then validated here
    _run_blender_script(executable, "lighting.py", tmp_path / "lighting.json")
    lighting = Lighting.model_validate_json((tmp_path / "lighting.json").read_text())
    latitudes = (np.arange(16) + 0.5) / 32 * np.pi
    expected_sky = (np.array((1.0, 0.5, 0.25)) + 100 * np.cos(latitudes[-1]) / np.cos(latitudes).sum()) * 2.0
    assert len(lighting.suns) == 1 and len(lighting.points) == 1 and lighting.sky == pytest.approx(expected_sky)
    assert len(lighting.emissive) == 2
