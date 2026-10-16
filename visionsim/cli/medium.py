from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt

_ALBEDO_FRAMES = 16
"""Number of frames, spread over the sequence, from which the mean albedos of the ground and of objects are estimated"""


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
    effects: Path | None = None,
    frames: str = "frames",
    depths: str = "depths",
    normals: str = "normals",
    emission: str = "emission",
    albedo: str = "diffuse/color",
    irradiance: tuple[str, ...] = ("diffuse/direct", "diffuse/indirect"),
    estimate_albedos: bool = False,
    surfaces: bool = True,
    wavelengths: tuple[float, ...] | None = None,
    fps: float | None = None,
    factor: int = 1,
    device: str | None = None,
    force: bool = False,
) -> None:
    """Add a participating medium, such as fog or haze, to rendered frames

    The medium is computed in closed form from the linear frames and depth maps of a render (see :mod:`visionsim.medium`),
    and the resulting frames can be used as the input of any sensor emulator. When the render includes normal maps,
    surfaces are also lit through the medium (see :mod:`visionsim.medium.surfaces`). Ground truth transmittance, optical
    depth, in-scattered radiance and the illumination of surfaces are saved alongside them, as well as the medium,
    lighting and terms of the model that were used. The illumination is the ratio by which the medium scales the light
    of surfaces, which, given the light they received (see ``irradiance`` and ``albedo``), can be large where they
    received almost no light, such as deep in the shadow of a lamp, which the medium's glow lights.

    Args:
        input_dir: directory containing rendered frames and depth maps, such as the output of ``blender.render-animation``
        output_dir: directory in which to save frames with the medium, as linear EXRs, and ground truths
        medium: path to a JSON file describing the medium, see :class:`Medium <visionsim.medium.model.Medium>`
        lighting: path to a JSON file describing the lighting, see :class:`Lighting <visionsim.medium.model.Lighting>`,
            or the lighting of each frame at which it changes, see :class:`AnimatedLighting
            <visionsim.medium.model.AnimatedLighting>`, in which case frames are matched by the number of their file
            name. Defaults to the ``animated-lighting.json`` saved in ``input_dir`` when rendering with
            ``--include-lighting`` and lights change over the frames, or else to its ``lighting.json``
        occlusion: path to the shadow maps through which objects cast shadows onto the medium, see
            :mod:`visionsim.medium.occlusion`. Defaults to the ``occlusion.npz`` saved in ``input_dir`` when rendering
            with ``--include-occlusion``, if any. Only used by media that attenuate sunlight
        effects: path to a JSON file of the terms of the model to compute, see :class:`Effects
            <visionsim.medium.model.Effects>`, e.g. ``{"halos": false}`` to leave out the halos of lamps. Defaults to
            every term that the inputs allow
        frames: name of the directory containing frames within ``input_dir``, these should be linear (EXR/HDR)
            as tonemapped frames have clipped highlights
        depths: name of the directory containing depth maps within ``input_dir``
        normals: name of the directory containing normal maps within ``input_dir``, saved in the camera's space as
            with ``--include-normals``, through which surfaces are lit through the medium. Without them, surfaces keep
            the light they were rendered with
        emission: name of the directory containing the light emitted by surfaces within ``input_dir``, as saved with
            ``--include-emission``, which the medium doesn't dim. Without it, surfaces are assumed not to emit light
        albedo: name of the directory containing the albedo of the surfaces seen by each pixel within ``input_dir``, i.e.
            Blender's diffuse color pass, as saved with ``--include-diffuse-pass``. With normal maps, and without the
            ``irradiance`` passes, it gives the light that surfaces received, by which the glow of the medium that lights
            them is weighed, see :func:`apply_medium <visionsim.medium.render.apply_medium>`, though too much of it on
            glossy surfaces, whose specular reflection the albedo leaves out. Without it, the albedos of the lighting
            are used
        irradiance: names of the directories containing the diffuse light passes within ``input_dir``, whose sum,
            times π, is the irradiance that the surfaces seen by each pixel received, as saved with
            ``--include-diffuse-pass``: Cycles' diffuse direct and indirect light (use ``diffuse/light`` for EEVEE's).
            With normal maps, they give the light that surfaces received, glossy ones too, rather than the ``albedo``,
            if all of them are found
        estimate_albedos: if true, the mean albedos of the ground and of objects, with which they reflect light into the
            medium, are estimated from the ``albedo`` of the pixels that see them, where the lighting doesn't give them,
            i.e. for a ground of zero albedo, as Blender's lighting exports, and objects of no albedo of their own, once,
            over frames spread over the sequence, as weighted by the light that their surfaces received, see
            :func:`mean_albedos <visionsim.medium.render.mean_albedos>`, and saved with the lighting. A diffuse color
            pass leaves out the specular reflection of glossy surfaces, which reflects light into the medium too, so
            that the albedos estimated are too low for them
        surfaces: if false, surfaces keep the light they were rendered with, even with normal maps
        wavelengths: effective wavelength, in nm, of each color channel. Defaults to 550 for grayscale frames,
            and to (610, 550, 465) otherwise
        fps: frame rate of the sequence, only needed for moving media. Inferred from the dataset if possible,
            accounting for the keyframe multiplier used when rendering
        factor: if greater than 1, the medium is added along the rays of every ``factor``-th pixel along each axis
            and upsampled to every pixel, along with the exact transmittance of each pixel, which saves most of the
            time, see :mod:`visionsim.medium.upsampling`. Defaults to 1, i.e. every pixel
        device: torch device to run on, defaults to "cuda" if available
        force: if true, overwrite output directory if present
    """
    import torch

    from visionsim.cli import _log
    from visionsim.dataset import Dataset, Metadata
    from visionsim.medium import AnimatedLighting, Blob, Effects, Lighting, Medium, apply_medium_upsampled, mean_albedos
    from visionsim.medium.occlusion import load_occlusion
    from visionsim.utils.color import LINEAR_EXTENSIONS, to_linearrgb
    from visionsim.utils.progress import ElapsedProgress

    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Input and output directory cannot be the same!")
    if output_dir.exists() and not force:
        raise FileExistsError("Output directory already exists.")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    if lighting is None and (input_dir / "animated-lighting.json").exists():
        lighting = input_dir / "animated-lighting.json"
    lighting = lighting or input_dir / "lighting.json"
    if not lighting.exists():
        raise FileNotFoundError(
            f"No lighting found at {lighting}, either render with `--include-lighting` or provide it with `--lighting`."
        )
    medium_spec = Medium.model_validate_json(Path(medium).read_text())
    effects_spec = Effects.model_validate_json(Path(effects).read_text()) if effects is not None else Effects()
    data = json.loads(lighting.read_text())
    animated = AnimatedLighting.model_validate(data) if "frames" in data else None
    lighting_spec = animated.frames[min(animated.frames)] if animated is not None else Lighting.model_validate(data)
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
    ds_albedo = Dataset.from_path(input_dir / albedo) if (input_dir / albedo).exists() else None
    found = bool(irradiance) and all((input_dir / name).exists() for name in irradiance)
    ds_irradiance = [Dataset.from_path(input_dir / name) for name in irradiance] if found else []
    if surfaces and ds_normals is None:
        _log.info("No normal maps found, surfaces keep the light they were rendered with.")
    if estimate_albedos and ds_albedo is None:
        raise FileNotFoundError(f"No albedo maps found at {input_dir / albedo}, from which to estimate albedos.")

    if len(ds_frames) != len(ds_depths):
        raise ValueError(f"Found {len(ds_frames)} frames but {len(ds_depths)} depth maps.")
    maps = [("normal maps", ds_normals), ("emission maps", ds_emission), ("albedo maps", ds_albedo)]
    for name, ds in maps + [("diffuse light passes", ds) for ds in ds_irradiance]:
        if ds is not None and len(ds) != len(ds_frames):
            raise ValueError(f"Found {len(ds_frames)} frames but {len(ds)} {name}.")

    def linear(frame: npt.NDArray, transform: dict) -> npt.NDArray:
        radiance = to_linearrgb(frame, Path(transform["file_path"]))
        if radiance.shape[-1] == 1 and (c := transform.get("c") or 1) > 1:
            # EXRs whose channels are all identical, as in gray scenes, are collapsed when loaded
            radiance = np.repeat(radiance, c, axis=-1)
        return radiance

    def received(i: int) -> npt.NDArray | None:
        # Irradiance that the surfaces of frame i received, π times the sum of the diffuse light passes
        return np.pi * sum(np.asarray(ds[i][0], dtype=float) for ds in ds_irradiance) if ds_irradiance else None

    if estimate_albedos and ds_albedo is not None:
        # The ground and objects reflect light into the medium with the same albedo in every frame, estimated once from
        # frames spread over the sequence, as weighted by the light that their surfaces received
        count = min(len(ds_frames), _ALBEDO_FRAMES)
        picks = np.unique(np.linspace(0, len(ds_frames) - 1, count).round().astype(int)).tolist()

        def sampled():
            for i in picks:
                frame, transform = ds_frames[i]
                radiance = linear(frame, transform)
                n = 3 if radiance.shape[-1] >= 3 else 1
                albedos, light = np.asarray(ds_albedo[i][0], dtype=float), received(i)
                if light is not None:
                    # The light that surfaces reflect diffusely, which leaves out the specular reflection of glossy ones,
                    # with the albedo of each channel, as apply_medium takes it
                    diffuse = albedos[..., None] if albedos.ndim == 2 else albedos
                    diffuse = (
                        diffuse[..., :3].mean(axis=-1, keepdims=True) if n == 1 and diffuse.shape[-1] >= 3 else diffuse
                    )
                    reflected = diffuse[..., :n] * light[..., :n] / np.pi
                else:
                    emitted = np.asarray(ds_emission[i][0])[..., :n] if ds_emission is not None else 0.0
                    reflected = radiance[..., :n] - emitted
                yield albedos, ds_depths[i][0], transform, transform["transform_matrix"], reflected

        ground, objects = mean_albedos(sampled(), lighting_spec.ground_height)
        for name, value in (("the ground", ground), ("objects", objects)):
            if value is not None:
                # Glossy surfaces reflect more light into the medium than their diffuse albedo
                _log.info(
                    f"Mean diffuse albedo of {name}, without specular reflection: {tuple(round(a, 3) for a in value)}."
                )

        def estimated(spec: Lighting) -> Lighting:
            update: dict[str, tuple[float, ...]] = {}
            if ground is not None and max(spec.ground_albedo, default=0.0) <= 0:
                update["ground_albedo"] = ground
            if objects is not None and spec.object_albedo is None:
                update["object_albedo"] = objects
            return spec.model_copy(update=update)

        lighting_spec = estimated(lighting_spec)
        if animated is not None:
            animated = AnimatedLighting(frames={k: estimated(spec) for k, spec in animated.frames.items()})
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

            radiance = linear(frame, transform)
            emitted = np.asarray(ds_emission[i][0], dtype=float) if ds_emission is not None else None
            # The light that surfaces received, measured by the diffuse light passes, or else by the albedo
            light = received(i) if ds_normals is not None else None
            albedos = None
            if light is None and ds_albedo is not None and ds_normals is not None:
                albedos = np.asarray(ds_albedo[i][0], dtype=float)
            frame_lighting = lighting_spec
            if animated is not None:
                # Frames are saved by their number, e.g. as frames/0000/255.exr, or else follow the lighting's first
                frame_lighting = animated.at(int(path.stem) if path.stem.isdigit() else min(animated.frames) + i)

            result = apply_medium_upsampled(
                radiance,
                depth,
                transform,
                transform["transform_matrix"],
                medium_spec,
                frame_lighting,
                factor=factor,
                wavelengths=wavelengths,
                time=i / fps if fps else 0.0,
                occlusion=shadows,
                normals=np.asarray(ds_normals[i][0], dtype=float) if ds_normals is not None else None,
                normals_space="camera",
                emission=emitted,
                albedo=albedos,
                irradiance=light,
                effects=effects_spec,
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
    (output_dir / "effects.json").write_text(effects_spec.model_dump_json(indent=2))
    (output_dir / "lighting.json").write_text(lighting_spec.model_dump_json(indent=2))
    if animated is not None:
        (output_dir / "animated-lighting.json").write_text(animated.model_dump_json(indent=2))
