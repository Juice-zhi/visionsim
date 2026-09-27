import inspect
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from docstring_parser import parse_from_object

from visionsim.cli import _cli_modules, _run, dataset, ffmpeg, transforms
from visionsim.cli.medium import _write_exr
from visionsim.dataset.models import Metadata


@pytest.mark.skipif("win" in sys.platform, reason="No autocomplete on windows")
def test_completions(tmpdir):
    # Note: If this test fails, it most likely means one of the arguments of a CLI method is
    #   not annotated properly. Logs will be saved to the temp dir, and should show what's going on.
    shell_name = Path(_run("echo $SHELL", shell=True, check=True, hide=True).stdout.strip()).stem

    if shell_name not in ("bash", "zsh", "tcsh"):
        pytest.skip(f"Unsupported shell, got {shell_name}, expected on of 'bash', 'zsh', 'tcsh'.")

    _run(rf"visionsim --tyro-write-completion {shell_name} {tmpdir}/visionsim", check=True, shell=True, log_path=tmpdir)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths only")
def test_run_keeps_windows_paths(tmp_path):
    # Commands such as ffmpeg's are built as strings containing Windows paths, which must survive being run
    path = tmp_path / "frames" / "%09d.png"
    result = _run(f'"{sys.executable}" -c "import sys; print(sys.argv[1])" {path}', hide=True, check=True)
    assert result.stdout.strip() == str(path)


def test_tonemap_single_channel_frames(tmp_path):
    # Gray EXRs are loaded with a single channel, and should be saved as grayscale images
    _write_exr(tmp_path / "frames" / "0000" / "000.exr", np.full((4, 6, 3), 0.5))
    transforms.tonemap_frames(tmp_path / "frames", tmp_path / "tonemapped")
    assert iio.imread(tmp_path / "tonemapped" / "000.png").shape == (4, 6)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_combine_overwrites_with_force(tmp_path):
    for name in ("a", "b"):
        _run(f"ffmpeg -f lavfi -i testsrc=size=64x48:rate=5 -t 1 {tmp_path / name}.mp4", hide=True, check=True)

    # Each video is scaled to QVGA before being stacked, so the output's width tells how many were combined
    for row in (["a", "b"], ["a", "b", "a"]):
        ffmpeg.combine(str([[str(tmp_path / f"{name}.mp4") for name in row]]), tmp_path / "combined.mp4", force=True)
        assert ffmpeg.dimensions(tmp_path / "combined.mp4") == (320 * len(row), 240)


@pytest.mark.parametrize("module", _cli_modules)
def test_help_is_full(module):
    for func_name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ == module.__name__ and not func_name.startswith("_"):
            docs = parse_from_object(func)
            documented_params = {param.arg_name for param in docs.params if param.description}
            all_params = set(inspect.getfullargspec(func).args)

            assert documented_params == all_params


def test_dataset_merge(cube_dataset):
    # Rename single dataset
    dataset.merge(
        input_files=[cube_dataset / "frames"],
        names=["custom_file_path"],
        output_file=(frames_path := cube_dataset / "frames" / "transforms.json"),
    )
    Metadata.load(frames_path)

    # Merge with other dataset that is already renamed
    dataset.merge(
        input_files=[cube_dataset / "depths", frames_path],
        names=None,
        output_file=(ds_path := cube_dataset / "combined.json"),
    )
    Metadata.load(ds_path)
    frames_path.unlink()
    ds_path.unlink()

    # Merge with renames
    dataset.merge(
        input_files=[cube_dataset / "frames", cube_dataset / "depths"],
        names=["file_path", "depth_file_path"],
        output_file=(ds_path := cube_dataset / "combined.json"),
    )
    Metadata.load(ds_path)
    ds_path.unlink()
