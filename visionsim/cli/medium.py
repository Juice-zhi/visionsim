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
    occlusion: Path | None = None,
    frames: str = "frames",
    depths: str = "depths",
    normals: str = "normals",
    emission: str = "emission",
    surfaces: bool = True,
    wavelengths: tuple[float, ...] | None = None,
    fps: float | None = None,
    device: str | None = None,
    force: bool = False,
) -> None:
    """Add a participating medium, such as fog or haze, to rendered frames

    The medium is computed in closed form from the linear frames and depth maps of a render (see :mod:`visionsim.medium`),
    and the resulting frames can be used as the input of any sensor emulator. When the render includes normal maps,
    surfaces are also lit through the medium (see :mod:`visionsim.medium.surfaces`). Ground truth transmittance, optical
    depth, in-scattered radiance and the illumination of surfaces are saved alongside them, as well as the medium and
    lighting that were used.

    Args:
        input_dir: directory containing rendered frames and depth maps, such as the output of ``blender.render-animation``
        output_dir: directory in which to save frames with the medium, as linear EXRs, and ground truths
        medium: path to a JSON file describing the medium, see :class:`Medium <visionsim.medium.model.Medium>`
        lighting: path to a JSON file describing the lighting, see :class:`Lighting <visionsim.medium.model.Lighting>`.
            Defaults to the ``lighting.json`` saved in ``input_dir`` when rendering with ``--include-lighting``
        occlusion: path to the shadow maps through which objects cast shadows onto the medium, see
            :mod:`visionsim.medium.occlusion`. Defaults to the ``occlusion.npz`` saved in ``input_dir`` when rendering
            with ``--include-occlusion``, if any. Only used by media that attenuate sunlight
        frames: name of the directory containing frames within ``input_dir``, these should be linear (EXR/HDR)
            as tonemapped frames have clipped highlights
        depths: name of the directory containing depth maps within ``input_dir``
        normals: name of the directory containing normal maps within ``input_dir``, saved in the camera's space as
            with ``--include-normals``, through which surfaces are lit through the medium. Without them, surfaces keep
            the light they were rendered with
        emission: name of the directory containing the light emitted by surfaces within ``input_dir``, as saved with
            ``--include-emission``, which the medium doesn't dim. Without it, surfaces are assumed not to emit light
        surfaces: if false, surfaces keep the light they were rendered with, even with normal maps
        wavelengths: effective wavelength, in nm, of each color channel. Defaults to 550 for grayscale frames,
            and to (610, 550, 465) otherwise
        fps: frame rate of the sequence, only needed for moving media. Inferred from the dataset if possible,
            accounting for the keyframe multiplier used when rendering
        device: torch device to run on, defaults to "cuda" if available
        force: if true, overwrite output directory if present
    """
    import torch

    from visionsim.cli import _log
    from visionsim.dataset import Dataset, Metadata
    from visionsim.medium import Blob, Lighting, Medium, apply_medium
    from visionsim.medium.occlusion import load_occlusion
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if occlusion is None and (input_dir / "occlusion.npz").exists():
        occlusion = input_dir / "occlusion.npz"
    if occlusion is not None and not medium_spec.sun_attenuation:
        _log.warning("Ignoring shadow maps, as the medium does not attenuate sunlight, see `sun_attenuation`.")
        occlusion = None
    shadows = load_occlusion(occlusion, device=device, dtype=torch.float64) if occlusion is not None else None

    ds_frames = Dataset.from_path(input_dir / frames)
    ds_depths = Dataset.from_path(input_dir / depths)
    ds_normals = Dataset.from_path(input_dir / normals) if surfaces and (input_dir / normals).exists() else None
    ds_emission = Dataset.from_path(input_dir / emission) if ds_normals and (input_dir / emission).exists() else None
    if surfaces and ds_normals is None:
        _log.info("No normal maps found, surfaces keep the light they were rendered with.")

    if len(ds_frames) != len(ds_depths):
        raise ValueError(f"Found {len(ds_frames)} frames but {len(ds_depths)} depth maps.")
    for name, ds in (("normal maps", ds_normals), ("emission maps", ds_emission)):
        if ds is not None and len(ds) != len(ds_frames):
            raise ValueError(f"Found {len(ds_frames)} frames but {len(ds)} {name}.")
    if fps is None and ds_frames.cameras:
        # A keyframe multiplier slows the animation down, so consecutive frames are closer in time
        framerates = {cam.fps * (getattr(cam, "keyframe_scale", None) or 1) for cam in ds_frames.cameras if cam.fps}
        fps = framerates.pop() if len(framerates) == 1 else None
    if not fps and any(isinstance(c, Blob) and any(c.velocity) for c in medium_spec.components):
        raise ValueError("The medium moves but the frame rate is unknown, please specify it with `--fps`.")

    outputs: dict[str, list[dict]] = {"frames": [], "transmittance": [], "optical-depth": [], "inscatter": []}
    if ds_normals is not None:
        outputs["illumination"] = []

    with ElapsedProgress() as progress:
        task = progress.add_task("Adding medium", total=len(ds_frames))

        for i, ((frame, transform), (depth, depth_transform)) in enumerate(zip(ds_frames, ds_depths)):
            if not np.allclose(transform["transform_matrix"], depth_transform["transform_matrix"]):
                raise ValueError(f"Frame {i} and its depth map were not rendered from the same pose.")

            path = Path(transform["file_path"])
            if i == 0 and path.suffix.lower() not in LINEAR_EXTENSIONS:
                _log.warning("Frames are tonemapped and have clipped highlights, prefer linear EXR frames.")

            radiance = to_linearrgb(frame, path)
            if radiance.shape[-1] == 1 and (c := transform.get("c") or 1) > 1:
                # EXRs whose channels are all identical, as in gray scenes, are collapsed when loaded
                radiance = np.repeat(radiance, c, axis=-1)
            emitted = np.asarray(ds_emission[i][0], dtype=float) if ds_emission is not None else None

            result = apply_medium(
                radiance,
                depth,
                transform,
                transform["transform_matrix"],
                medium_spec,
                lighting_spec,
                wavelengths=wavelengths,
                time=i / fps if fps else 0.0,
                occlusion=shadows,
                normals=np.asarray(ds_normals[i][0], dtype=float) if ds_normals is not None else None,
                normals_space="camera",
                emission=emitted,
                device=device,
            )
            relative = path.relative_to(ds_frames.root or "").with_suffix(".exr")
            pose = np.asarray(transform["transform_matrix"]).tolist()

            for (name, transforms), data in zip(outputs.items(), result):
                assert data is not None
                _write_exr(output_dir / name / relative, data.cpu().numpy())
                transforms.append(transform | {"file_path": relative, "c": data.shape[-1], "transform_matrix": pose})
            progress.update(task, advance=1)

    for name, transforms in outputs.items():
        Metadata.from_dense_transforms(transforms).save(output_dir / name / "transforms.json")
    (output_dir / "medium.json").write_text(medium_spec.model_dump_json(indent=2))
    (output_dir / "lighting.json").write_text(lighting_spec.model_dump_json(indent=2))
