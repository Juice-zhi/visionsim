"""Speed first: the fastest configurations of the closed form, their error, and Cycles at the same time or error.

Where ``final_benchmark.py`` asks how long Cycles takes for the closed form's error, this asks the reverse, for the
configurations that favor speed: the Effects presets of the library (``complete`` and ``fast``), the upsampling at
factors 4 and 8 of ``apply_medium_upsampled``, and frames without fog that Cycles renders at 1, 4 or 16 samples per
pixel (denoised), or that are already there (``reused``: rendered anyway, e.g. once for several fogs, so that only the
closed form's own time counts). For each, at 320x180 (frames 255 and 345) and 1920x1080 (frames 255 and 256, against
``hd_refs``, with the RMSE without the reference's noise, see ``final_benchmark.py``), it gives the time per frame of the
pipeline, its errors, and, along Cycles' curve of error against time with fog (renders of the same session, read from
``--benchmark``, the JSON of ``final_benchmark.py``), the error Cycles reaches in the same time and the time it needs for
the same error. At 1080p on the GPU, Cycles' curve is extended with its renders at 64 and 256 samples per pixel
(``gpu/``, errors from ``--extra-1080p``). Results go to ``runs/fog-comparison/acceleration/results/<--out>``::

    python examples/medium/experiment/acceleration/analysis/speed_first.py --device cuda --renders gpu2 \\
        --benchmark final_gpu.json --out speed_first_gpu.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_benchmark as fb
from visionsim.medium import EFFECT_PRESETS
from visionsim.medium.upsampling import apply_medium_upsampled

from common import OUT, errors, load_images

CLEAR = ("reused", 1, 4, 16)


def frame_seconds(log: Path) -> float | None:
    """Mean time of the frames after the first in a log of render_mc.py."""
    if not log.exists():
        return None
    times = [float(t) for t in re.findall(r"FRAME_TIME frame=\d+ seconds=([\d.]+)", log.read_text(errors="replace"))]
    return float(np.mean(times[1:])) if len(times) > 1 else None


def run(case, frames, depth, transform, normals, factor, effects):
    return apply_medium_upsampled(
        frames,
        depth,
        transform,
        transform["transform_matrix"],
        case.medium,
        case.lighting,
        factor=factor,
        occlusion=case.scene.occlusion,
        normals=normals,
        effects=effects,
        dtype=torch.float32,
        device=case.device,
    )


def along(points: list[tuple[float, float]], x: float) -> float | None:
    """Log-log interpolation of y at x along (x, y) points sorted by x, or None outside them."""
    points = sorted(p for p in points if p[0] > 0 and p[1] > 0)
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 <= x <= x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
    return None


def cycles_curves(case_name: str, benchmark: dict, extra: dict | None, hd: bool) -> dict[str, list[tuple[float, float]]]:
    """Cycles' (seconds, error) points per metric, at 320x180 or 1080p."""
    entry = benchmark[case_name]
    curves: dict[str, list[tuple[float, float]]] = {"rel_l1": [], "rel_rmse_corrected": []}
    for spp, v in entry["cycles"].items():
        seconds = v.get("seconds_1080p") if hd else v.get("seconds")
        found = entry["1080p"].get("cycles", {}).get(spp) if hd else v.get("denoised")
        if seconds and found:
            for metric, points in curves.items():
                points.append((seconds, found[metric]))
    if hd and extra is not None:
        for spp in ("64", "256"):
            found = extra[case_name]["1080p"].get("cycles", {}).get(spp)
            name = f"night_full_hd_spp{spp}" if case_name == "night" else f"hd_spp{spp}"
            seconds = frame_seconds(OUT / "gpu" / "logs" / f"{name}.log")
            if found and seconds:
                for metric, points in curves.items():
                    points.append((seconds, found[metric]))
    return curves


def compare(seconds: float, error: dict, curves: dict) -> dict:
    """Cycles' error at the same time, and time for the same error, per metric."""
    out = {}
    for metric, points in curves.items():
        by_time = sorted(points)
        same_time = along(by_time, seconds)
        if same_time is None and by_time:
            same_time = "faster than Cycles' fastest" if seconds < by_time[0][0] else by_time[-1][1]
        same_error = along([(e, t) for t, e in points], error[metric])
        if same_error is None and points:
            worst = max(points, key=lambda p: p[1])
            best = min(points, key=lambda p: p[1])
            same_error = f"<{worst[0]:.3f}" if error[metric] >= worst[1] else f">{best[0]:.3f}"
        out[metric] = {"cycles_error_at_same_time": same_time, "cycles_seconds_at_same_error": same_error}
    return out


def measure(case, frames_of, depths, transforms, normals, reference, noise_mse, factor, effects, repeats, timed_frames):
    """Errors for every input of frames without fog, and warm and first-call times of the closed form."""
    device = case.device
    fb.synchronize(device)
    start = time.perf_counter()
    run(case, frames_of["reused"][0], depths[0], transforms[0], normals[0], factor, effects)
    fb.synchronize(device)
    first = time.perf_counter() - start
    fastest = []
    for i in range(timed_frames):
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            run(case, frames_of["reused"][i], depths[i], transforms[i], normals[i], factor, effects)
            fb.synchronize(device)
            best = min(best, time.perf_counter() - start)
        fastest.append(best)
    results = {"seconds": float(np.mean(fastest)), "first_seconds": first, "inputs": {}}
    for name, frames in frames_of.items():
        images = np.stack(
            [
                run(case, frames[i], depths[i], transforms[i], normals[i], factor, effects).radiance.cpu().numpy()
                for i in range(len(transforms))
            ]
        )
        results["inputs"][str(name)] = errors(images, reference) | fb.rmse(images, reference, noise_mse)
    return results


def main(args):
    fb.GPU = OUT / args.renders
    benchmark = json.loads((OUT / "results" / args.benchmark).read_text())
    extra_path = OUT / "results" / args.extra_1080p
    extra = json.loads(extra_path.read_text()) if args.device.startswith("cuda") and extra_path.exists() else None
    results: dict = {
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else f"cpu, {torch.get_num_threads()}",
        "renders": args.renders,
        "rows": [],
    }
    for case_name in args.cases:
        case = fb.day_case(fb.FRAMES, args.device) if case_name == "day" else fb.night_case(args.device)
        night = case_name == "night"
        for hd in (False, True):
            if hd:
                inputs = fb.hd_inputs(case)
                transforms, depths, normals = inputs["transforms"], inputs["depth"], inputs["normals"]
                reference, noise_mse = inputs["reference"], inputs["noise_mse"]
                frames_of = {"reused": inputs["clear"]}
                for spp in CLEAR[1:]:
                    folder = fb.GPU / f"clear_{'night_' if night else ''}hd_spp{spp}" / "frames"
                    frames_of[spp] = case.on_device(load_images(folder, fb.HD_FRAMES))
                timed = 1
            else:
                transforms, depths, normals = case.scene.transforms, case.depth, case.normals
                reference, noise_mse = case.reference, case.noise_mse
                frames_of = {"reused": case.clear} | {spp: case.clear_frames(spp) for spp in CLEAR[1:]}
                timed = len(case.frames)
            curves = cycles_curves(case_name, benchmark, extra, hd)
            for preset in args.presets:
                effects = EFFECT_PRESETS[preset][0]
                for factor in args.factors:
                    measured = measure(
                        case,
                        frames_of,
                        depths,
                        transforms,
                        normals,
                        reference,
                        noise_mse,
                        factor,
                        effects,
                        args.repeats,
                        timed,
                    )
                    for name, error in measured["inputs"].items():
                        if name == "reused":
                            clear_seconds = 0.0
                        else:
                            log = (
                                f"clear_{'night_' if night else ''}hd_spp{name}"
                                if hd
                                else f"clear_{case_name}_spp{name}"
                            )
                            clear_seconds = frame_seconds(fb.GPU / "logs" / f"{log}.log")
                            if clear_seconds is None:
                                continue
                        seconds = clear_seconds + measured["seconds"]
                        row = {
                            "case": case_name,
                            "resolution": "1920x1080" if hd else "320x180",
                            "preset": preset,
                            "factor": factor,
                            "clear": name,
                            "seconds": seconds,
                            "closed_form_seconds": measured["seconds"],
                            "first_seconds": measured["first_seconds"],
                            "rel_l1": error["rel_l1"],
                            "rel_l1_blurred": error["rel_l1_blurred"],
                            "rel_rmse_corrected": error["rel_rmse_corrected"],
                            "versus": compare(seconds, error, curves),
                        }
                        results["rows"].append(row)
                        v = row["versus"]
                        print(
                            f"{case_name:5s} {row['resolution']:9s} {preset:8s} f{factor} clear={name:>6}:"
                            f" {1e3 * seconds:7.1f} ms  L1 {100 * error['rel_l1']:5.2f}%"
                            f" RMSE {100 * error['rel_rmse_corrected']:5.2f}%  | Cycles same time: L1"
                            f" {fmt(v['rel_l1']['cycles_error_at_same_time'])} RMSE"
                            f" {fmt(v['rel_rmse_corrected']['cycles_error_at_same_time'])}",
                            flush=True,
                        )
    (OUT / "results" / args.out).write_text(json.dumps(results, indent=1))


def fmt(value) -> str:
    return f"{100 * value:5.2f}%" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renders", default="gpu2", help="folder of the session's renders and logs")
    parser.add_argument("--benchmark", default="final_gpu.json", help="final_benchmark.py's JSON of the same session")
    parser.add_argument("--extra-1080p", default="merged2_hd_errors_gpu.json", help="errors of Cycles at 64, 256 spp")
    parser.add_argument("--cases", nargs="+", default=["day", "night"], choices=["day", "night"])
    parser.add_argument("--presets", nargs="+", default=["complete", "fast"])
    parser.add_argument("--factors", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="speed_first_gpu.json")
    main(parser.parse_args())
