"""Time and error of the closed form against Cycles, by day and at night, for the final comparison.

The closed form (``apply_medium``, through ``apply_medium_upsampled`` at 1/``factor`` of the resolution) adds fog to
frames rendered without it, so that it is compared as a pipeline: Cycles renders the frame without fog at a few samples
per pixel, denoised by OIDN, and the closed form adds the fog, against Cycles rendering the frame with fog, denoised,
on the same hardware (renders of ``gpu_final.ps1``). Depth and normals, which converge in a sample or two, are those
of the frames without fog rendered at many samples.

- Errors at 320x180 on frames 255 and 345 (relative L1, as is and blurred, and the ratio of light, see
  ``common.errors``): by day against Cycles 4096 spp with 16 volume bounces (``ms/cycles_ref``), at night against
  Cycles 65536 spp with 16 volume bounces without clamping (``gpu/night_full_spp65536``), whose noise is negligible.
  The closed form's own error is that given the frames without fog rendered at many samples (``clear``).
- By day, flicker over frames 255 to 262 (``common.flicker``): the temporal error of log intensity drives false events
  in event cameras.
- Times per frame at 320x180 and 1920x1080: the closed form's first call, which includes one-off costs (tables, halos
  and the light that surfaces reflect, computed once per scene), and the fastest of the next ones, with the inputs
  already on the device. Cycles' times are read from its logs (mean of the frames after the first, which warms up
  Blender), and include denoising.
- Errors at 1920x1080 on frames 255 and 256, when its references were rendered (``hd_refs``, by ``gpu_final.ps1 -All``
  on the GPU): against the mean of two Cycles renders of different seeds (2x2048 spp by day, 2x8192 spp unclamped at
  night, 16 volume bounces), whose own noise is reported; the depth of ``hd/clear`` and the normals of ``hd_refs``.
  These references keep a few percent of noise per pixel (night: 6.5%), so that errors are also given as relative RMSE
  without the reference's noise (``rel_rmse_corrected``): the mean square error to a noisy reference is that to the
  truth plus the variance of the reference's noise, a quarter of the mean square difference of its two halves. The
  same is given at 320x180, by day with the noise measured against ``ms/cycles_ref_seed1``.

Renders and logs of Cycles are read from ``--renders`` (``gpu``, or e.g. ``gpu2`` and ``cpu2`` made in one quiet
session by ``final_session.ps1``), and the closed form runs on ``--device``, so that both use the same hardware.

The time Cycles needs to reach the pipeline's error, or its flicker, is interpolated log-log along its curve, and the
saving is its ratio to the pipeline's time. Results are saved to ``results/final_benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visionsim.medium.upsampling import apply_medium_upsampled

from common import (
    DTYPE,
    GROUND_ALBEDO,
    NIGHT_FRAMES,
    OUT,
    ROOT,
    box_depth,
    day_scene,
    errors,
    flicker,
    load_images,
    load_transforms,
    night_scene,
)

GPU = OUT / "gpu"
"""Cycles' renders and logs, set by ``--renders``"""
NIGHT_REFERENCE = OUT / "gpu" / "night_full_spp65536"
FRAMES = (255, 345)
SEQUENCE = tuple(range(255, 263))
SPP = (1, 2, 4, 16, 64, 256, 1024)
CLEAR_SPP = (1, 4, 16, 64)
HD_FRAMES = (255, 256)
HD_CLEAR_SPP = (1, 4, 16)
HD_REFS = OUT / "hd_refs"


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def frame_times(name: str) -> float | None:
    path = GPU / "logs" / f"{name}.log"
    if not path.exists():
        return None
    times = [float(t) for t in re.findall(r"FRAME_TIME frame=\d+ seconds=([\d.]+)", path.read_text(errors="replace"))]
    return float(np.mean(times[1:])) if len(times) > 1 else None


def timed(fn, device: str, repeats: int) -> tuple[object, float, float]:
    """Result, time of the first call, and fastest time of the next ones."""
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    first = time.perf_counter() - start
    fastest = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        fastest = min(fastest, time.perf_counter() - start)
    return result, first, fastest


class Case:
    """Inputs of the closed form for frames of a scene, already on the device."""

    def __init__(self, name, frames, scene, lighting, medium, clear, normals, reference, device, noise_mse=0.0):
        self.name, self.frames, self.scene, self.device = name, tuple(frames), scene, device
        self.lighting, self.medium, self.reference, self.noise_mse = lighting, medium, reference, noise_mse
        self.clear, self.normals = self.on_device(clear), self.on_device(normals)
        self.depth = self.on_device(scene.depth)

    def on_device(self, images: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(images), dtype=DTYPE, device=self.device)

    def clear_frames(self, spp: int | None) -> torch.Tensor:
        """Frames without fog: rendered at many samples, or by Cycles at a few, denoised."""
        if spp is None:
            return self.clear
        return self.on_device(load_images(GPU / f"clear_{self.name}_spp{spp}" / "frames", self.frames))

    def run(self, i: int, factor: int, clear=None, depth=None, transform=None, normals=None):
        t = transform if transform is not None else self.scene.transforms[i]
        return apply_medium_upsampled(
            self.clear[i] if clear is None else clear,
            self.depth[i] if depth is None else depth,
            t,
            t["transform_matrix"],
            self.medium,
            self.lighting,
            factor=factor,
            occlusion=self.scene.occlusion,
            normals=self.normals[i] if normals is None else normals,
            dtype=DTYPE,
            device=self.device,
        )

    def measure(self, images: np.ndarray) -> dict[str, float]:
        """Errors against the reference: relative L1 (common.errors) and relative RMSE without the reference's noise."""
        return errors(images, self.reference) | rmse(images, self.reference, self.noise_mse)

    def images(self, factor: int, clear: torch.Tensor) -> np.ndarray:
        return np.stack([self.run(i, factor, clear=clear[i]).radiance.cpu().numpy() for i in range(len(self.frames))])


def day_case(frames, device: str) -> Case:
    scene = day_scene(frames)
    # The reference's noise, from a second render of another seed: both have it, so that their mean square difference
    # is twice its variance
    seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", frames)
    noise_mse = float(np.mean((scene.reference.astype(np.float64) - seed1) ** 2) / 2)
    return Case(
        "day",
        frames,
        scene,
        scene.lighting,
        scene.medium,
        scene.clear,
        scene.extra["normals"],
        scene.reference,
        device,
        noise_mse,
    )


def night_case(device: str) -> Case:
    scene = night_scene(NIGHT_FRAMES)
    lighting = scene.lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    medium = scene.medium.model_copy(update={"multiple_scattering": True})
    clear = load_images(OUT / "night_clear" / "frames", NIGHT_FRAMES)
    normals = day_scene(NIGHT_FRAMES).extra["normals"]
    reference = load_images(NIGHT_REFERENCE / "frames", NIGHT_FRAMES)
    return Case("night", NIGHT_FRAMES, scene, lighting, medium, clear, normals, reference, device)


def closed_form(case: Case, factors, repeats: int) -> dict:
    """Error and time of the closed form, given frames without fog rendered at many samples or at a few."""
    results = {}
    for factor in factors:
        firsts, fastest = [], []
        for i in range(len(case.frames)):
            _, first, best = timed(lambda i=i, factor=factor: case.run(i, factor), case.device, repeats)
            firsts.append(first)
            fastest.append(best)
        entry = {
            "seconds": float(np.mean(fastest)),
            "first_seconds": firsts[0],
            "whole": case.measure(case.images(factor, case.clear)),
        }
        for spp in CLEAR_SPP:
            if (GPU / f"clear_{case.name}_spp{spp}").exists():
                entry[f"clear_spp{spp}"] = case.measure(case.images(factor, case.clear_frames(spp)))
        results[factor] = entry
        print(case.name, factor, json.dumps(entry), flush=True)
    return results


def rmse(ours: np.ndarray, reference: np.ndarray, noise_mse: float = 0.0) -> dict[str, float]:
    """Relative root mean square error, as is and with the reference's own noise taken out: the mean square error to a
    noisy reference is that to the truth plus the variance of the reference's noise, when both are independent."""
    reference = reference.astype(np.float64)
    mse = float(np.mean((ours.astype(np.float64) - reference) ** 2))
    scale = float(np.sqrt(np.mean(reference**2)))
    return {"rel_rmse": np.sqrt(mse) / scale, "rel_rmse_corrected": np.sqrt(max(mse - noise_mse, 0.0)) / scale}


def hd_inputs(case: Case) -> dict:
    """Inputs of the closed form at 1920x1080 on frames 255 and 256, and the reference when it was rendered.

    Depth is the day scene's (the night scene has the same geometry and cameras), and so are the normals, from
    ``hd_refs/normals`` (upsampled from 320x180 if missing, which only serves timing). Frames without fog are the day's
    in ``hd/clear``, or ``hd_refs/clear_night``. The reference is the mean of two halves of different seeds, whose
    difference gives its own noise: the mean is off by half their relative L1 difference.
    """
    hd = ROOT / "hd" / "clear"
    transforms = load_transforms(hd / "frames", HD_FRAMES)
    depths = load_images(hd / "depths", HD_FRAMES)[..., 0]
    inputs: dict = {
        "transforms": transforms,
        "depth": case.on_device(np.stack([box_depth(d, t) for d, t in zip(depths, transforms)])),
    }
    h, w = depths.shape[1:]
    if (HD_REFS / "normals" / "normals").exists():
        normals = load_images(HD_REFS / "normals" / "normals", HD_FRAMES)
        # Saved rotated by the camera's rotation rather than its inverse, see common.day_scene
        rotations = [np.asarray(t["transform_matrix"])[:3, :3] for t in transforms]
        normals = np.stack([n @ (r / np.linalg.norm(r, axis=0)) for n, r in zip(normals, rotations)])
        inputs["exact_normals"] = True
    else:
        small = case.normals[0].cpu().numpy()
        rows = (np.arange(h) * small.shape[0] // h).clip(0, small.shape[0] - 1)
        columns = (np.arange(w) * small.shape[1] // w).clip(0, small.shape[1] - 1)
        normals = np.stack([small[rows][:, columns]] * len(HD_FRAMES))
        inputs["exact_normals"] = False
    inputs["normals"] = case.on_device(normals)
    clear = hd / "frames" if case.name == "day" else HD_REFS / "clear_night" / "frames"
    inputs["clear"] = case.on_device(load_images(clear, HD_FRAMES)) if clear.exists() else None
    halves = [HD_REFS / f"{case.name}_seed{seed}" / "frames" for seed in (0, 1)]
    if all(half.exists() for half in halves):
        first, second = (load_images(half, HD_FRAMES) for half in halves)
        inputs["reference"] = (first + second) / 2
        inputs["reference_noise"] = errors(first, second)["rel_l1"] / 2
        # The mean of two independent halves has a quarter of the mean square of their difference as its noise
        inputs["noise_mse"] = float(np.mean((first.astype(np.float64) - second) ** 2) / 4)
    return inputs


def hd(case: Case, factors, repeats: int) -> dict:
    """Times per frame at 1920x1080 (frame 255) and, given the reference, errors on frames 255 and 256 of the closed
    form, with frames without fog rendered at many samples and at a few, and of Cycles with fog."""
    inputs = hd_inputs(case)
    clear = inputs["clear"] if inputs["clear"] is not None else torch.zeros_like(inputs["normals"])

    def run(i: int, factor: int, frames: torch.Tensor):
        t = inputs["transforms"][i]
        return case.run(i, factor, clear=frames[i], depth=inputs["depth"][i], transform=t, normals=inputs["normals"][i])

    results: dict = {"exact_normals": inputs["exact_normals"], "closed_form": {}}
    reference = inputs.get("reference")
    if reference is not None:
        results["reference_noise"] = inputs["reference_noise"]
        results["reference_noise_rmse"] = float(
            np.sqrt(inputs["noise_mse"]) / np.sqrt(np.mean(reference.astype(np.float64) ** 2))
        )

    def measure(images: np.ndarray) -> dict:
        return errors(images, reference) | rmse(images, reference, inputs["noise_mse"])

    for factor in factors:
        _, first, best = timed(lambda f=factor: run(0, f, clear), case.device, repeats)
        entry: dict = {"seconds": best, "first_seconds": first}
        if reference is not None and inputs["clear"] is not None:
            images = np.stack([run(i, factor, clear).radiance.cpu().numpy() for i in range(len(HD_FRAMES))])
            entry["whole"] = measure(images)
            for spp in HD_CLEAR_SPP:
                path = GPU / f"clear_{'night_' if case.name == 'night' else ''}hd_spp{spp}" / "frames"
                if path.exists():
                    frames = case.on_device(load_images(path, HD_FRAMES))
                    images = np.stack([run(i, factor, frames).radiance.cpu().numpy() for i in range(len(HD_FRAMES))])
                    entry[f"clear_spp{spp}"] = measure(images)
        results["closed_form"][factor] = entry
        print(case.name, "1080p", factor, json.dumps(entry), flush=True)
    if reference is not None:
        results["cycles"] = {}
        for spp in SPP:
            path = GPU / (f"night_full_hd_spp{spp}" if case.name == "night" else f"hd_spp{spp}") / "frames"
            if path.exists():
                results["cycles"][spp] = measure(load_images(path, HD_FRAMES))
        print(case.name, "1080p cycles", json.dumps(results["cycles"]), flush=True)
    return results


def cycles(case: Case) -> dict:
    """Error and time of Cycles with fog, and time of Cycles without fog, at each number of samples per pixel."""
    night = case.name == "night"
    results = {}
    for spp in SPP:
        name = f"night_full_spp{spp}" if night else f"day_spp{spp}"
        entry: dict = {}
        if (GPU / name).exists():
            entry["seconds"] = frame_times(name)
            entry["seconds_1080p"] = frame_times(f"night_full_hd_spp{spp}" if night else f"hd_spp{spp}")
            entry["denoised"] = case.measure(load_images(GPU / name / "frames", case.frames))
            entry["noisy"] = case.measure(load_images(GPU / name / "noisy", case.frames))
        if spp in CLEAR_SPP:
            entry["clear_seconds"] = frame_times(f"clear_{case.name}_spp{spp}")
            entry["clear_seconds_1080p"] = frame_times(f"clear_{'night_' if night else ''}hd_spp{spp}")
        if entry:
            results[spp] = entry
    return results


def interpolate(points: list[tuple[float, float]], value: float) -> float | None:
    """Time at which a curve of (error, time) points reaches an error, log-log, or None outside it."""
    points = sorted(p for p in points if p[0] > 0 and p[1])
    for (e0, t0), (e1, t1) in zip(points[:-1], points[1:]):
        if e0 <= value <= e1:
            f = (np.log(value) - np.log(e0)) / (np.log(e1) - np.log(e0))
            return float(np.exp(np.log(t0) + f * (np.log(t1) - np.log(t0))))
    return None


def comparison(case_name: str, entry: dict, flickers: dict | None) -> list[dict]:
    """Each pipeline (samples of the frames without fog, factor) against Cycles at the same error, and flicker, at
    320x180, and at 1080p with errors measured at 1080p when its reference was rendered (else those of 320x180)."""
    curve = entry["cycles"]
    rows = []
    for factor, values in entry["closed_form"].items():
        for spp in CLEAR_SPP:
            if f"clear_spp{spp}" not in values or not curve[spp].get("clear_seconds"):
                continue
            error = values[f"clear_spp{spp}"]["rel_l1"]
            row = {
                "case": case_name,
                "resolution": "320x180",
                "factor": factor,
                "clear_spp": spp,
                "rel_l1": error,
                "rel_l1_blurred": values[f"clear_spp{spp}"]["rel_l1_blurred"],
                "seconds": curve[spp]["clear_seconds"] + values["seconds"],
                "cycles_seconds_at_equal_error": interpolate(
                    [(v["denoised"]["rel_l1"], v["seconds"]) for v in curve.values() if "denoised" in v], error
                ),
                "rel_rmse_corrected": values[f"clear_spp{spp}"]["rel_rmse_corrected"],
                "cycles_seconds_at_equal_rmse": interpolate(
                    [(v["denoised"]["rel_rmse_corrected"], v["seconds"]) for v in curve.values() if "denoised" in v],
                    values[f"clear_spp{spp}"]["rel_rmse_corrected"],
                ),
            }
            if flickers is not None and f"{factor}_clear_spp{spp}" in flickers:
                temporal = flickers[f"{factor}_clear_spp{spp}"]["temporal"]
                cycles_flicker = [
                    (flickers[f"cycles_spp{s}"]["temporal"], v["seconds"])
                    for s, v in curve.items()
                    if f"cycles_spp{s}" in flickers
                ]
                row["temporal_flicker"] = temporal
                row["cycles_seconds_at_equal_flicker"] = interpolate(cycles_flicker, temporal)
            rows.append(row)

    # At 1080p, with errors measured at 1080p when its reference was rendered (else those of 320x180). Its references
    # keep a few percent of noise per pixel, so the RMSE without the reference's noise is the error to compare there
    hd = entry["1080p"]
    measured = "cycles" in hd

    def cycles_curve(metric: str) -> list[tuple[float, float]]:
        return [
            (hd["cycles"][s][metric] if measured else v["denoised"][metric], v["seconds_1080p"])
            for s, v in curve.items()
            if v.get("seconds_1080p") and (s in hd.get("cycles", {}) or not measured)
        ]

    for factor, values in hd["closed_form"].items():
        for spp in HD_CLEAR_SPP:
            if not curve.get(spp, {}).get("clear_seconds_1080p"):
                continue
            if measured and f"clear_spp{spp}" in values:
                found = values[f"clear_spp{spp}"]
            elif not measured and f"clear_spp{spp}" in entry["closed_form"].get(factor, {}):
                found = entry["closed_form"][factor][f"clear_spp{spp}"]
            else:
                continue
            rows.append(
                {
                    "case": case_name,
                    "resolution": "1920x1080",
                    "errors_measured_at": "1920x1080" if measured else "320x180",
                    "factor": factor,
                    "clear_spp": spp,
                    "rel_l1": found["rel_l1"],
                    "rel_l1_blurred": found["rel_l1_blurred"],
                    "rel_rmse_corrected": found["rel_rmse_corrected"],
                    "seconds": curve[spp]["clear_seconds_1080p"] + values["seconds"],
                    "cycles_seconds_at_equal_error": interpolate(cycles_curve("rel_l1"), found["rel_l1"]),
                    "cycles_seconds_at_equal_rmse": interpolate(
                        cycles_curve("rel_rmse_corrected"), found["rel_rmse_corrected"]
                    ),
                }
            )
    return rows


def day_flicker(factors, device: str) -> dict:
    case = day_case(SEQUENCE, device)
    results = {}
    for factor in factors:
        results[f"{factor}_clear_ref"] = flicker(case.images(factor, case.clear), case.reference)
        for spp in CLEAR_SPP:
            if (GPU / f"clear_day_spp{spp}").exists():
                images = case.images(factor, case.clear_frames(spp))
                results[f"{factor}_clear_spp{spp}"] = flicker(images, case.reference)
    for spp in SPP:
        if (GPU / f"day_spp{spp}").exists():
            results[f"cycles_spp{spp}"] = flicker(
                load_images(GPU / f"day_spp{spp}" / "frames", SEQUENCE), case.reference
            )
    for spp in CLEAR_SPP:
        if (GPU / f"clear_day_spp{spp}").exists():
            clear = load_images(GPU / f"clear_day_spp{spp}" / "frames", SEQUENCE)
            results[f"clear_spp{spp}_vs_clear_ref"] = flicker(clear, case.clear.cpu().numpy())
    reference_seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", SEQUENCE)
    results["reference_seed1"] = flicker(reference_seed1, case.reference)
    print("flicker", json.dumps(results), flush=True)
    return results


def main(args):
    global GPU
    GPU = OUT / args.renders
    device = args.device
    results: dict = {
        "device": torch.cuda.get_device_name(0)
        if device.startswith("cuda")
        else f"cpu, {torch.get_num_threads()} threads",
        "renders": args.renders,
    }
    flickers = day_flicker(args.factors, device) if "day" in args.cases else None
    rows = []
    for name in args.cases:
        case = day_case(FRAMES, device) if name == "day" else night_case(device)
        entry = {
            "closed_form": closed_form(case, args.factors, args.repeats),
            "1080p": hd(case, args.hd_factors or args.factors, args.repeats),
            "cycles": cycles(case),
        }
        results[name] = entry
        rows += comparison(name, entry, flickers if name == "day" else None)
    results["day_flicker"] = flickers
    results["comparison"] = rows
    out = OUT / "results" / (args.out or "final_benchmark.json")
    out.write_text(json.dumps(results, indent=1))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", nargs="+", default=["day", "night"], choices=["day", "night"])
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hd-factors", nargs="+", type=int, help="factors timed at 1080p, defaults to --factors")
    parser.add_argument("--renders", default="gpu", help="folder of Cycles' renders and logs, in the acceleration study")
    parser.add_argument("--out", help="name of the JSON file in results/, defaults to final_benchmark.json")
    main(parser.parse_args())
