from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt


def _write_exr(path: Path, data: npt.NDArray) -> None:
    import OpenEXR  # type: ignore

    channels = {1: "V", 3: "RGB", 4: "RGBA"}
    if (c := data.shape[-1]) not in channels:
        raise ValueError(f"Cannot save an EXR with {c} channels, expected one of {tuple(channels)}.")

    pixels = np.ascontiguousarray(data[..., 0] if c == 1 else data, dtype=np.float32)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    path.parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File(header, {channels[c]: pixels}).write(str(path))


def apply(
    input_dir: Path,
    output_dir: Path,
    medium: Path,
    lighting: Path | None = None,
    frames: str = "frames",
    depths: str = "depths",
    wavelengths: tuple[float, ...] | None = None,
    fps: float | None = None,
    device: str | None = None,
    force: bool = False,
) -> None:
    """Add a participating medium, such as fog or haze, to rendered frames

    The medium is computed in closed form from the linear frames and depth maps of a render (see :mod:`visionsim.medium`),
    and the resulting frames can be used as the input of any sensor emulator. Ground truth transmittance, optical depth
    and in-scattered radiance are saved alongside them, as well as the medium and lighting that were used.

    Args:
        input_dir: directory containing rendered frames and depth maps, such as the output of ``blender.render-animation``
        output_dir: directory in which to save frames with the medium, as linear EXRs, and ground truths
        medium: path to a JSON file describing the medium, see :class:`Medium <visionsim.medium.model.Medium>`
        lighting: path to a JSON file describing the lighting, see :class:`Lighting <visionsim.medium.model.Lighting>`.
            Defaults to the ``lighting.json`` saved in ``input_dir`` when rendering with ``--include-lighting``
        frames: name of the directory containing frames within ``input_dir``, these should be linear (EXR/HDR)
            as tonemapped frames have clipped highlights
        depths: name of the directory containing depth maps within ``input_dir``
        wavelengths: effective wavelength, in nm, of each color channel. Defaults to 550 for grayscale frames,
            and to (610, 550, 465) otherwise
        fps: frame rate of the sequence, only needed for moving media, inferred from the dataset if possible
        device: torch device to run on, defaults to "cuda" if available
        force: if true, overwrite output directory if present
    """
    import torch

    from visionsim.cli import _log
    from visionsim.dataset import Dataset, Metadata
    from visionsim.medium import Blob, Lighting, Medium, apply_medium
    from visionsim.utils.color import LINEAR_EXTENSIONS, to_linearrgb
    from visionsim.utils.progress import ElapsedProgress

    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Input and output directory cannot be the same!")
    if output_dir.exists() and not force:
        raise FileExistsError("Output directory already exists.")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    lighting = lighting or input_dir / "lighting.json"
    if not lighting.exists():
        raise FileNotFoundError(
            f"No lighting found at {lighting}, either render with `--include-lighting` or provide it with `--lighting`."
        )
    medium_spec = Medium.model_validate_json(Path(medium).read_text())
    lighting_spec = Lighting.model_validate_json(lighting.read_text())

    if lighting_spec.points:
        _log.warning(f"Ignoring {len(lighting_spec.points)} point light(s), as these are not yet supported.")

    ds_frames = Dataset.from_path(input_dir / frames)
    ds_depths = Dataset.from_path(input_dir / depths)

    if len(ds_frames) != len(ds_depths):
        raise ValueError(f"Found {len(ds_frames)} frames but {len(ds_depths)} depth maps.")
    if fps is None and ds_frames.cameras and len(framerates := {cam.fps for cam in ds_frames.cameras}) == 1:
        fps = framerates.pop()
    if not fps and any(isinstance(c, Blob) and any(c.velocity) for c in medium_spec.components):
        raise ValueError("The medium moves but the frame rate is unknown, please specify it with `--fps`.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outputs: dict[str, list[dict]] = {"frames": [], "transmittance": [], "optical-depth": [], "inscatter": []}

    with ElapsedProgress() as progress:
        task = progress.add_task("Adding medium", total=len(ds_frames))

        for i, ((frame, transform), (depth, depth_transform)) in enumerate(zip(ds_frames, ds_depths)):
            if not np.allclose(transform["transform_matrix"], depth_transform["transform_matrix"]):
                raise ValueError(f"Frame {i} and its depth map were not rendered from the same pose.")

            path = Path(transform["file_path"])
            if i == 0 and path.suffix.lower() not in LINEAR_EXTENSIONS:
                _log.warning("Frames are tonemapped and have clipped highlights, prefer linear EXR frames.")

            result = apply_medium(
                to_linearrgb(frame, path),
                depth,
                transform,
                transform["transform_matrix"],
                medium_spec,
                lighting_spec,
                wavelengths=wavelengths,
                time=i / fps if fps else 0.0,
                device=device,
            )
            relative = path.relative_to(ds_frames.root or "").with_suffix(".exr")
            pose = np.asarray(transform["transform_matrix"]).tolist()

            for (name, transforms), data in zip(outputs.items(), result):
                _write_exr(output_dir / name / relative, data.cpu().numpy())
                transforms.append(transform | {"file_path": relative, "c": data.shape[-1], "transform_matrix": pose})
            progress.update(task, advance=1)

    for name, transforms in outputs.items():
        Metadata.from_dense_transforms(transforms).save(output_dir / name / "transforms.json")
    (output_dir / "medium.json").write_text(medium_spec.model_dump_json(indent=2))
    (output_dir / "lighting.json").write_text(lighting_spec.model_dump_json(indent=2))
