"""Controlled comparison of ways to add fog to a scene, as seen by four emulated sensors.

Every method starts from the same animated scene (``demo_scene.py``) and the same ground fog
(``media/ground_fog.json``), and is compared against a Cycles reference, i.e. volumetric path tracing with 4096 spp,
without adaptive sampling nor denoising:

- ``clear``: no fog at all, the baseline.
- ``cycles_default``: Cycles volumetric path tracing with visionsim's default settings (256 spp, adaptive sampling
  and denoising), i.e. what one gets by adding the fog volume to the scene and rendering it as usual.
- ``cycles_nodenoise``: the same without denoising, i.e. with Cycles' raw Monte Carlo noise.
- ``raymarch_16`` and ``raymarch_64s16``: ray marching (visionsim.medium.raymarch), with 16 steps and light
  attenuation in closed form, or with 64 steps and 16 steps towards the sun, as generic volume renderers do.
- ``closed_form_m1``: the closed form, with the lighting model of the first version (isotropic, unattenuated sky).
- ``closed_form``: the closed form, with the current lighting model.
- ``reference_seed1``: the reference rendered with another seed, which measures the reference's own noise.

The four sensors are a conventional RGB camera (25 fps), a passive single photon camera (125 Hz binary frames), an
event camera (v2e, without noise) and an active single photon lidar (flash, 1000 time bins). Renders are expected in
``runs/fog-comparison`` (see ``render_all.ps1`` and ``render_refs.ps1``), and results are written to its ``results``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sensors import SPAD, EventCamera, RGBCamera, bernoulli_kl, event_f1, event_preview, psnr, relative_l1, ssim

from visionsim.dataset import Dataset
from visionsim.emulate.spc import spc_avg_to_rgb
from visionsim.medium import Lighting, Medium, apply_medium, camera_rays
from visionsim.medium.raymarch import ray_march_medium
from visionsim.medium.transient import Flash, capture_histogram, estimate_distance, flash_transient
from visionsim.utils.color import linearrgb_to_srgb, to_linearrgb

ROOT = Path("runs/fog-comparison")
BOX_MIN, BOX_MAX = np.array([-200.0, -200.0, -1.0]), np.array([200.0, 200.0, 25.0])
FPS, DEVICE, DTYPE = 125.0, "cuda", torch.float32
FLASH = Flash(wavelength=905.0, bins=1000, bin_width=0.4e-9, pulse_width=1e-9, signal=0.5, background=2.0)
CYCLES = 500
FRAME = 250  # frame used for still images, clamped to the number of frames
GROUND_ALBEDO = 0.39  # area average of the checkered ground's base colors, 0.6 and 0.18

LABELS = {
    "clear": "无雾 baseline",
    "cycles_default": "Cycles 默认设置",
    "cycles_nodenoise": "Cycles 默认（不降噪）",
    "raymarch_16": "Ray marching 16 步",
    "raymarch_64s16": "Ray marching 64+16 步",
    "closed_form_m1": "闭式解（修正前）",
    "closed_form": "闭式解",
    "reference": "参考 Cycles 4096spp",
    "reference_seed1": "参考（另一种子）",
    "cycles_default_ss": "Cycles 默认（单次散射）",
    "closed_form_ms": "闭式解（多次散射）",
}
FOLDERS = {  # rendered method -> folder of its render in ROOT
    "cycles_default": "cycles_default",
    "cycles_nodenoise": "cycles_nodenoise",
    "reference": "cycles_ref",
    "reference_seed1": "cycles_ref_seed1",
}
PASSIVE = [
    "clear",
    "cycles_default",
    "cycles_nodenoise",
    "raymarch_16",
    "raymarch_64s16",
    "closed_form_m1",
    "closed_form",
    "reference_seed1",
]
TOF = {  # ToF method -> (passive method providing the ambient background, ray marching steps or None)
    "clear": ("clear", None),
    "raymarch_16": ("raymarch_16", 128),
    "raymarch_64s16": ("raymarch_64s16", 1024),
    "closed_form": ("closed_form", None),
    "reference_seed1": ("reference_seed1", None),
}
TOF_LABELS = LABELS | {"raymarch_16": "Ray marching 128 步", "raymarch_64s16": "Ray marching 1024 步"}
COLORS = {name: f"C{i}" for i, name in enumerate(LABELS)}  # the same color for a method in every plot
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
STILLS: tuple[str, ...] = ("clear", "cycles_default", "cycles_nodenoise", "raymarch_16", "closed_form_m1", "closed_form")
VIDEO: tuple[str, ...] = ("clear", "cycles_default", "cycles_nodenoise", "raymarch_16", "closed_form")


def use_multiple_scattering() -> None:
    """Compare against Cycles renders in which light may scatter many times in the fog (``render_ms.ps1``), instead
    of Blender's default single scattering, alongside Cycles' single scattering render with default settings."""
    global STILLS, VIDEO
    FOLDERS.update({name: f"ms/{folder}" for name, folder in FOLDERS.items()} | {"cycles_default_ss": "cycles_default"})
    PASSIVE.insert(PASSIVE.index("cycles_nodenoise") + 1, "cycles_default_ss")
    # The closed form's approximation of multiple scattering, including light reflected by the ground
    PASSIVE.insert(PASSIVE.index("closed_form") + 1, "closed_form_ms")
    TOF["closed_form_ms"] = ("closed_form_ms", None)
    names = {
        "cycles_default": "Cycles 默认（多次散射）",
        "cycles_nodenoise": "Cycles 不降噪（多次散射）",
        "closed_form": "闭式解（单次散射）",
        "reference": "参考（多次散射）",
        "reference_seed1": "参考（多次散射，另一种子）",
    }
    LABELS.update(names)
    TOF_LABELS.update(names)
    STILLS = ("clear", "cycles_default_ss", "cycles_default", "raymarch_16", "closed_form", "closed_form_ms")
    VIDEO = ("clear", "cycles_default", "closed_form", "closed_form_ms")


# --------------------------------------------------------------------------------------------------------------------
# Inputs


def load_sequence(path: Path, channels: int = 3, limit: int | None = None) -> tuple[np.ndarray, list[dict]]:
    frames, transforms = [], []
    dataset = Dataset.from_path(path)
    for index in range(min(len(dataset), limit or len(dataset))):
        data, transform = dataset[index]
        assert isinstance(transform, dict)
        data = np.asarray(data)
        if channels == 3:
            data = to_linearrgb(data, transform["file_path"])
            data = np.repeat(data, 3, axis=-1) if data.shape[-1] == 1 else data[..., :3]
        frames.append(data.astype(np.float32))
        transforms.append(transform)
    return np.stack(frames), transforms


def box_depth(depth: np.ndarray, transform: dict) -> np.ndarray:
    """Depth where rays that see the sky stop when leaving the fog volume, as in Cycles."""
    origin, directions, scale = (x.numpy() for x in camera_rays(transform, transform["transform_matrix"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        bounds = np.where(directions > 0, BOX_MAX, BOX_MIN)
        exit_distance = np.where(directions != 0, (bounds - origin) / directions, np.inf).min(axis=-1)
    return np.where(depth >= 1e9, exit_distance / scale, depth).astype(np.float32)


def passive_sequences(cache: Path, limit: int | None) -> tuple[dict[str, np.ndarray], list[dict], dict[str, float]]:
    """Radiance seen by the camera for every method, cached as .npy files, and the time taken per frame."""
    clear, transforms = load_sequence(ROOT / "clear" / "frames", limit=limit)
    depths = load_sequence(ROOT / "clear" / "depths", 1, limit=limit)[0]
    depth = np.stack([box_depth(d[..., 0], t) for d, t in zip(depths, transforms)])
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    medium_ms = medium.model_copy(update={"multiple_scattering": True})
    lighting_ms = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    computed = {
        "closed_form": lambda i: apply_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium,
            lighting,
            time=i / FPS,
            device=DEVICE,
            dtype=DTYPE,
        ),
        "closed_form_ms": lambda i: apply_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium_ms,
            lighting_ms,
            time=i / FPS,
            device=DEVICE,
            dtype=DTYPE,
        ),
        "closed_form_m1": lambda i: apply_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium,
            Lighting(ambient=lighting.sky, suns=lighting.suns),
            time=i / FPS,
            device=DEVICE,
            dtype=DTYPE,
        ),
        "raymarch_16": lambda i: ray_march_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium,
            lighting,
            steps=16,
            time=i / FPS,
            device=DEVICE,
            dtype=DTYPE,
        ),
        "raymarch_64s16": lambda i: ray_march_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium,
            lighting,
            steps=64,
            shadow_steps=16,
            time=i / FPS,
            device=DEVICE,
            dtype=DTYPE,
        ),
    }
    sequences, seconds = {"clear": clear}, {}
    for name, folder in FOLDERS.items():
        if (ROOT / folder / "frames").exists():
            sequences[name] = load_sequence(ROOT / folder / "frames", limit=limit)[0]
    if len(sequences.get("cycles_nodenoise", [])) < len(clear):
        print("Skipping Cycles without denoising, as its render is missing or incomplete.")
        sequences.pop("cycles_nodenoise", None)
        PASSIVE.remove("cycles_nodenoise")
    if len(sequences.get("reference_seed1", [])) < len(clear):
        # The reference's noise floor is optional
        print("Skipping the reference's noise floor, as its second render is missing or incomplete.")
        sequences.pop("reference_seed1", None)
        PASSIVE.remove("reference_seed1")
        TOF.pop("reference_seed1")
    for name, fn in computed.items():
        path = cache / f"{name}_{len(clear)}.npy"
        if not path.exists():
            np.save(path, np.stack([fn(i).radiance.cpu().numpy() for i in range(len(clear))]).astype(np.float32))
        sequences[name] = np.load(path)
        # Time a few frames, with the GPU otherwise idle
        fn(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(0, len(clear), len(clear) // 10):
            fn(i)
        torch.cuda.synchronize()
        seconds[name] = (time.perf_counter() - start) / len(range(0, len(clear), len(clear) // 10))
    return sequences, transforms, seconds


# --------------------------------------------------------------------------------------------------------------------
# Sensors


def rgb_results(seqs: dict[str, np.ndarray], camera: RGBCamera) -> dict:
    n = len(seqs["reference"]) // camera.exposure
    groups = lambda seq: [seq[k * camera.exposure : (k + 1) * camera.exposure] for k in range(n)]
    reference = [camera.expected(g) for g in groups(seqs["reference"])]
    reference_linear = [g.mean(axis=0) for g in groups(seqs["reference"])]
    results = {}
    for name in PASSIVE:
        expected = [camera.expected(g) for g in groups(seqs[name])]
        noisy = [camera.noisy(g, np.random.default_rng(1000 + k)) for k, g in enumerate(groups(seqs[name]))]
        results[name] = {
            "psnr": [psnr(e, r) for e, r in zip(expected, reference)],
            "ssim": [ssim(e, r) for e, r in zip(expected, reference)],
            "rel_l1": [relative_l1(g.mean(axis=0), r) for g, r in zip(groups(seqs[name]), reference_linear)],
            "noisy": noisy,
        }
    results["reference"] = {
        "noisy": [camera.noisy(g, np.random.default_rng(1000 + k)) for k, g in enumerate(groups(seqs["reference"]))]
    }
    return results


def spad_results(seqs: dict[str, np.ndarray], spad: SPAD, window: int = 32) -> dict:
    results = {}
    reference = [spad.probability(f) for f in seqs["reference"]]
    for name in [*PASSIVE, "reference"]:
        rng = np.random.default_rng(2000)
        binary = np.stack([spad.noisy(f, rng) for f in seqs[name]])
        averages = [
            spc_avg_to_rgb(binary[max(0, i - window + 1) : i + 1].mean(axis=0), factor=spad.factor)
            for i in range(2, len(binary), 5)
        ]
        entry = {"binary": binary[2::5], "average": averages}
        if name != "reference":
            probabilities = [spad.probability(f) for f in seqs[name]]
            entry["mae"] = [float(np.abs(p - r).mean()) for p, r in zip(probabilities, reference)]
            entry["rel_l1"] = [relative_l1(p, r) for p, r in zip(probabilities, reference)]
            entry["kl_bits"] = [bernoulli_kl(r, p) for p, r in zip(probabilities, reference)]
        results[name] = entry
    return results


def dvs_results(seqs: dict[str, np.ndarray], camera: EventCamera) -> dict:
    reference = camera.counts(seqs["reference"])
    results = {"reference": {"counts": reference, "events": int(reference.sum())}}
    for name in PASSIVE:
        counts = camera.counts(seqs[name])
        results[name] = {
            "counts": counts,
            "events": int(counts.sum()),
            "f1": [event_f1(c, r) for c, r in zip(counts, reference)],
            # Pooled over all windows, so that windows with few events, when the camera barely moves, weigh less
            "f1_pooled": event_f1(counts, reference),
        }
    return results


def tof_results(seqs: dict[str, np.ndarray], transforms: list[dict]) -> tuple[dict, dict[str, float]]:
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    depths = Dataset.from_path(ROOT / "clear" / "depths")
    normals = Dataset.from_path(ROOT / "clear" / "normals")
    albedos = Dataset.from_path(ROOT / "clear" / "diffuse" / "color")
    keys = ("rel_l1", "rel_l1_laser", "disagree", "disagree_expected", "valid", "distance", "pixels")
    results: dict[str, dict[str, list]] = {name: {key: [] for key in keys} for name in [*TOF, "reference"]}
    seconds: dict[str, list[float]] = {name: [] for name in TOF}
    pixels = [(165, 160), (100, 300), (70, 180)]  # near ground, green cube, sky near the horizon

    for k, i in enumerate(range(2, len(transforms), 5)):
        t = transforms[i]
        pose = np.asarray(t["transform_matrix"])
        depth = np.asarray(depths[i][0], dtype=np.float32)[..., 0]
        albedo = np.asarray(albedos[i][0], dtype=np.float32)[..., 1]
        # Normals are saved rotated by the camera's rotation, rather than its inverse, so undo that to get world normals
        normal_world = np.asarray(normals[i][0], dtype=np.float64) @ pose[:3, :3]
        _, directions, scale = (x.numpy() for x in camera_rays(t, pose))
        cos_incidence = np.clip(-(normal_world * directions).sum(-1), 0, 1)
        true_distance = np.where(depth < 1e9, depth * scale, np.inf)
        frame = {"depth": depth, "albedo": albedo, "cos_incidence": cos_incidence, "camera": t, "transform_matrix": pose}

        def transient(name, steps, background, frame, time, nodes=4):
            return flash_transient(
                **frame,
                medium=None if name == "clear" else medium,
                flash=FLASH,
                ambient_radiance=background[..., 1],
                steps=steps,
                nodes=nodes,
                time=time,
                device=DEVICE,
                dtype=DTYPE,
            )

        def expected_distance(photons):
            # Without photon noise, undoing pile-up recovers the expected flux, whose peak gives the distance
            return (photons.argmax(dim=-1).to(DTYPE) + 0.5) * FLASH.bin_range

        reference_transient = transient("reference", None, seqs["reference"][i], frame, i / FPS, nodes=16)
        reference, reference_laser = (
            reference_transient.photons,
            reference_transient.surface + reference_transient.backscatter,
        )
        generator = torch.Generator(DEVICE).manual_seed(3000 + k)
        reference_distance = estimate_distance(capture_histogram(reference, CYCLES, generator), CYCLES, FLASH)
        reference_expected = expected_distance(reference)
        in_range = torch.as_tensor(true_distance < FLASH.max_range, device=DEVICE)
        true_distance_t = torch.as_tensor(true_distance, device=DEVICE, dtype=DTYPE)
        for name, (passive, steps) in [*TOF.items(), ("reference", ("reference", None))]:
            if name == "reference":
                photons, distance = reference, reference_distance
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = transient(name, steps, seqs[passive][i], frame, i / FPS)
                torch.cuda.synchronize()
                seconds[name].append(time.perf_counter() - start)
                photons, laser = result.photons, result.surface + result.backscatter
                generator = torch.Generator(DEVICE).manual_seed(3000 + k)
                distance = estimate_distance(capture_histogram(photons, CYCLES, generator), CYCLES, FLASH)
                results[name]["rel_l1"].append(float((photons - reference).abs().sum() / reference.sum()))
                results[name]["rel_l1_laser"].append(
                    float((laser - reference_laser).abs().sum() / reference_laser.sum())
                )
                results[name]["disagree"].append(float(((distance - reference_distance).abs() > 0.15).float().mean()))
                disagree = (expected_distance(photons) - reference_expected).abs() > 0.15
                results[name]["disagree_expected"].append(float(disagree.float().mean()))
            valid = ((distance - true_distance_t).abs() < 0.15) & in_range
            results[name]["valid"].append(float(valid.sum() / in_range.sum()))
            results[name]["distance"].append(distance.cpu().numpy())
            if k == FRAME // 5:
                results[name]["pixels"] = [photons[r, c].cpu().numpy() for r, c in pixels]
    return results, {name: float(np.mean(s[1:])) for name, s in seconds.items()}


def ray_marching_convergence(seqs: dict[str, np.ndarray], transforms: list[dict], reference: np.ndarray) -> dict:
    """Error and time per frame of ray marching as the number of steps increases, on the still frame."""
    clear = seqs["clear"][FRAME]
    t = transforms[FRAME]
    depth = box_depth(np.asarray(Dataset.from_path(ROOT / "clear" / "depths")[FRAME][0])[..., 0], t)
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    args = (clear, depth, t, t["transform_matrix"], medium, lighting)

    def timed(fn, **kwargs):
        fn(*args, device=DEVICE, dtype=DTYPE, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, device=DEVICE, dtype=DTYPE, **kwargs).radiance.cpu().numpy()
        torch.cuda.synchronize()
        return result, time.perf_counter() - start

    exact, exact_seconds = timed(apply_medium)
    marched = []
    for shadow, steps_list in ((0, (2, 4, 8, 16, 32, 64, 128, 256)), (16, (16, 64, 256))):
        for steps in steps_list:
            result, seconds = timed(ray_march_medium, steps=steps, shadow_steps=shadow)
            marched.append(
                {
                    "steps": steps,
                    "shadow_steps": shadow,
                    "seconds": seconds,
                    "error_vs_closed_form": relative_l1(result, exact),
                    "error_vs_reference": relative_l1(result, reference),
                }
            )
    return {
        "closed_form_seconds": exact_seconds,
        "closed_form_error_vs_reference": relative_l1(exact, reference),
        "ray_marching": marched,
    }


# --------------------------------------------------------------------------------------------------------------------
# Figures and videos


def tonemap(linear: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    return (np.asarray(linearrgb_to_srgb(np.clip(linear * exposure, 0, 1))) * 255).astype(np.uint8)


def colormap(values: np.ndarray, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    normalized = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    return (plt.get_cmap(cmap)(normalized)[..., :3] * 255).astype(np.uint8)


def labeled_grid(rows: list[list[np.ndarray]], columns: list[str], row_labels: list[str] | None = None) -> np.ndarray:
    h, w = rows[0][0].shape[:2]
    left, top, gap = (110 if row_labels else 0), 24, 4
    canvas = Image.new("RGB", (left + len(columns) * (w + gap), top + len(rows) * (h + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(columns):
        draw.text((left + j * (w + gap) + w // 2, top // 2), label, font=FONT, fill="black", anchor="mm")
    for i, row in enumerate(rows):
        if row_labels:
            draw.text((left // 2, top + i * (h + gap) + h // 2), row_labels[i], font=FONT, fill="black", anchor="mm")
        for j, image in enumerate(row):
            canvas.paste(Image.fromarray(image), (left + j * (w + gap), top + i * (h + gap)))
    array = np.asarray(canvas)
    return array[: array.shape[0] // 2 * 2, : array.shape[1] // 2 * 2]


def placeholder(text: str, shape: tuple[int, ...]) -> np.ndarray:
    """Gray tile with a centered note, for sensors a method cannot emulate."""
    image = Image.new("RGB", (shape[1], shape[0]), (60, 64, 68))
    ImageDraw.Draw(image).text((shape[1] // 2, shape[0] // 2), text, font=FONT, fill=(220, 224, 228), anchor="mm")
    return np.asarray(image)


def write_video(frames: list[np.ndarray], path: Path, fps: int = 25) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            iio.imwrite(Path(tmp) / f"{i:05d}.png", frame)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(tmp) / "%05d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "28",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
        )


def main(args):
    if args.truth == "multiple":
        use_multiple_scattering()
    out = ROOT / ("results" if args.truth == "single" else "results_ms")
    cache = ROOT / "results" / "cache"  # the closed form and ray marching don't depend on the reference
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    seqs, transforms, medium_seconds = passive_sequences(cache, args.frames)
    print("passive sequences ready", {k: v.shape for k, v in seqs.items()}, medium_seconds, flush=True)
    global FRAME
    FRAME = min(FRAME, 5 * (len(transforms) // 10))

    rgb = rgb_results(seqs, RGBCamera())
    print("rgb done", flush=True)
    spad = spad_results(seqs, SPAD())
    print("spad done", flush=True)
    dvs = dvs_results(seqs, EventCamera())
    print("dvs done", flush=True)
    tof, tof_seconds = tof_results(seqs, transforms)
    print("tof done", flush=True)

    # Losses of the rendered frames themselves, before any sensor
    frames_loss = {
        name: {
            "rel_l1": [relative_l1(f, r) for f, r in zip(seqs[name], seqs["reference"])],
            "psnr": [psnr(tonemap(f) / 255.0, tonemap(r) / 255.0) for f, r in zip(seqs[name], seqs["reference"])],
        }
        for name in PASSIVE
    }

    summary = {
        "frames": {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in frames_loss.items()},
        "rgb": {name: {k: float(np.mean(rgb[name][k])) for k in ("psnr", "ssim", "rel_l1")} for name in PASSIVE},
        "spad": {name: {k: float(np.mean(spad[name][k])) for k in ("mae", "rel_l1", "kl_bits")} for name in PASSIVE},
        "dvs": {
            name: {
                "f1": float(np.mean(dvs[name]["f1"])),
                "f1_pooled": dvs[name]["f1_pooled"],
                "events": dvs[name]["events"],
            }
            for name in PASSIVE
        }
        | {"reference": {"events": dvs["reference"]["events"]}},
        "tof": {
            name: {
                k: float(np.mean(tof[name][k]))
                for k in ("rel_l1", "rel_l1_laser", "disagree", "disagree_expected", "valid")
                if tof[name][k]
            }
            for name in [*TOF, "reference"]
        },
        "seconds": {"medium": medium_seconds, "tof": tof_seconds},
    }
    curves = {
        "frames_rel_l1": {n: frames_loss[n]["rel_l1"] for n in PASSIVE},
        "rgb_psnr": {n: rgb[n]["psnr"] for n in PASSIVE},
        "spad_mae": {n: spad[n]["mae"] for n in PASSIVE},
        "dvs_f1": {n: dvs[n]["f1"] for n in PASSIVE},
        "tof_disagree": {n: tof[n]["disagree"] for n in TOF},
        "tof_disagree_expected": {n: tof[n]["disagree_expected"] for n in TOF},
        "tof_valid": {n: tof[n]["valid"] for n in [*TOF, "reference"]},
    }
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "curves": curves}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)

    # Still images of the chosen frame
    columns = [n for n in STILLS if n in PASSIVE] + ["reference"]
    k = FRAME // 5
    reference = seqs["reference"][FRAME]
    rows = [
        [tonemap(seqs[n][FRAME]) for n in columns],
        [
            colormap(np.abs(seqs[n][FRAME] - reference).mean(-1) / (reference.mean(-1) + 1e-3), 0, 0.5, "magma")
            for n in columns
        ],
        [(rgb[n]["noisy"][k] * 255).astype(np.uint8) for n in columns],
        [(np.repeat(spad[n]["binary"][k][..., 1:2], 3, -1) * 255).astype(np.uint8) for n in columns],
        [tonemap(spad[n]["average"][k]) for n in columns],
        [event_preview(dvs[n]["counts"][k]) for n in columns],
    ]
    labels = ["渲染帧", "相对误差 0–50%", "RGB 25fps", "SPAD 单帧", "SPAD 32帧平均", "DVS 40ms"]
    iio.imwrite(out / "images" / "overview.png", labeled_grid(rows, [LABELS[n] for n in columns], labels))
    tof_columns = ["clear", "raymarch_16", "raymarch_64s16", "closed_form", "reference"]
    tof_rows = [[colormap(tof[n]["distance"][k], 0, FLASH.max_range, "turbo") for n in tof_columns]]
    iio.imwrite(
        out / "images" / "tof_depth.png",
        labeled_grid(tof_rows, [TOF_LABELS[n] for n in tof_columns]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    names = ["像素 A：近处地面", "像素 B：绿色方块", "像素 C：地平线附近天空"]
    r_axis = (np.arange(FLASH.bins) + 0.5) * FLASH.bin_range
    for ax, title, idx in zip(axes, names, range(3)):
        for n, style in zip(tof_columns, ("-", ":", "--", "-.", "-")):
            ax.plot(
                r_axis,
                tof[n]["pixels"][idx],
                style,
                lw=1.2 if n != "reference" else 2.5,
                label=TOF_LABELS[n],
                color=COLORS[n],
                alpha=0.9 if n != "reference" else 0.35,
            )
        ax.set_yscale("log")
        ax.set_xlabel("距离 (m)")
        ax.set_ylabel("每个脉冲每个 bin 的期望光子数")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "images" / "tof_transients.png", dpi=110)
    plt.close(fig)

    # Loss over time, per sensor
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    plots = [
        ("rgb_psnr", "RGB：PSNR (dB)，越高越好", 25),
        ("spad_mae", "SPAD：检测概率 MAE，越低越好", 125),
        ("dvs_f1", "DVS：事件 F1，越高越好", 25),
        ("tof_disagree_expected", "ToF：与参考测距不一致的像素比例（无噪声），越低越好", 25),
    ]
    for ax, (key, title, fps) in zip(axes.flat, plots):
        labels = TOF_LABELS if key.startswith("tof") else LABELS
        for n, values in curves[key].items():
            ax.plot(np.arange(len(values)) / fps, values, label=labels[n], color=COLORS[n], lw=1.3)
        ax.set_title(title)
        ax.set_xlabel("时间 (s)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "images" / "loss_curves.png", dpi=100)
    plt.close(fig)

    convergence = ray_marching_convergence(seqs, transforms, reference)
    (out / "convergence.json").write_text(json.dumps(convergence, indent=1))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for shadow, marker in ((0, "o"), (16, "s")):
        points = [c for c in convergence["ray_marching"] if c["shadow_steps"] == shadow]
        ax.loglog(
            [c["seconds"] for c in points],
            [c["error_vs_closed_form"] for c in points],
            marker=marker,
            label=f"Ray marching（{'光照衰减用闭式' if shadow == 0 else '朝太阳 16 步'}）",
        )
        for c in points:
            ax.annotate(
                str(c["steps"]),
                (c["seconds"], c["error_vs_closed_form"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.axvline(convergence["closed_form_seconds"], color="k", ls="--", lw=1, label="闭式解耗时")
    ax.set_xlabel("每帧耗时 (s)")
    ax.set_ylabel("与闭式解的相对 L1 误差")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "images" / "convergence.png", dpi=110)
    plt.close(fig)

    # Videos
    video_columns = [n for n in VIDEO if n in PASSIVE] + ["reference"]
    frames = []
    for k in range(len(rgb["reference"]["noisy"])):
        i = 5 * k + 2
        grid = [
            [tonemap(seqs[n][i]) for n in video_columns],
            [(rgb[n]["noisy"][k] * 255).astype(np.uint8) for n in video_columns],
            [tonemap(spad[n]["average"][k]) for n in video_columns],
            [event_preview(dvs[n]["counts"][k]) for n in video_columns],
            [
                colormap(tof[n]["distance"][k], 0, FLASH.max_range, "turbo")
                if n in tof
                else placeholder("不适用：Cycles 无法渲染瞬态", seqs[n][i].shape)
                for n in video_columns
            ],
        ]
        frames.append(
            labeled_grid(
                grid,
                [LABELS[n] for n in video_columns],
                ["渲染帧", "RGB 25fps", "SPAD 32帧平均", "DVS 40ms", "ToF 距离"],
            )
        )
    write_video(frames, out / "videos" / "sensors.mp4", fps=12)
    print("done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, help="only use the first frames, for quick tests")
    parser.add_argument(
        "--truth",
        choices=("single", "multiple"),
        default="single",
        help="compare against Cycles with single scattering (Blender's default) or multiple scattering, whose "
        "results are saved to results_ms",
    )
    main(parser.parse_args())
