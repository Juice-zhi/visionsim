"""Evaluate the closed form against Cycles in three test cases, with the same metrics for every version of the library.

The library is imported as usual, so that a version under development, e.g. in a git worktree, is evaluated by putting
it first on ``PYTHONPATH``; this script and the data it reads stay in the main checkout, and it runs from its root::

    PYTHONPATH=E:/visionsim-wt/halos python examples/medium/experiment/acceleration/analysis/eval_closed_form.py \\
        --cases day night --out runs/fog-comparison/acceleration/analysis/eval/halos.json

Cases, all at 320x180 on frames 255 and 345:

- ``day``: the day scene with objects, with multiple scattering, shadow maps and surfaces lit through the fog, against
  Cycles 4096 spp with 16 volume bounces (``ms/cycles_ref``): whole frames, the fog's light (its volume passes) and
  surfaces (frames minus volume passes); the fog's light by azimuth of rays from the sun and by elevation; and the time
  of a frame.
- ``noobjects``: the same scene without objects, where only the fog's light is compared, with the ground's mean albedo,
  against Cycles with 32 volume bounces without clamping (``noclamp/noobjects32``), overall, by elevation and by
  azimuth from the sun, which isolates the model of light scattered more than once from shadows; the light of suns and
  the sky scattered once against Cycles with no volume bounces, and the rest (light reflected by the ground, and light
  scattered more than once) against the difference of both.
- ``night``: the night scene, with a lamp behind a cube and the ground's mean albedo, with multiple scattering, shadow
  maps and surfaces lit through the fog, against Cycles 65536 spp with 16 volume bounces without clamping on the GPU
  (``gpu/night_full_spp65536``, whose noise is negligible; ``--night-reference night_ms_noclamp`` uses the earlier
  2048 spp render on the CPU, with about 10% of noise per pixel): whole frames, the
  fog's light, and the fog's light in pixels where the cube shadows the lamp and elsewhere; the light it scatters
  once against the light scattered once (direct pass); the rest against the indirect pass, made of light scattered
  more than once by the fog alone (the indirect pass of ``night_ms_black_noclamp``, whose surfaces are black) and of
  light reflected by surfaces; the light that surfaces reflect and the fog scatters once (the closed form with the
  ground's albedo, minus without it, both without multiple scattering) against all the light that surfaces reflect;
  the halo against the fog's own light scattered more than once; and the time of a frame.

Errors are relative L1 errors over all pixels and channels of both frames, as is and once both images are blurred
(5x5), with the ratio of their light, see ``common.errors``. Results are printed and saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import visionsim
from visionsim.medium import Lighting, Medium, apply_medium, camera_rays

from common import DTYPE, NIGHT_FRAMES, OUT, ROOT, day_scene, errors, load_images, night_scene

FRAMES = (255, 345)
NIGHT_REFERENCE = "gpu/night_full_spp65536"
AZIMUTHS = (0, 20, 40, 60, 90, 120, 180)
ELEVATIONS = (-90, -10, -4, -1.5, 0, 1.5, 4, 10, 25, 90)


def fastest(fn, repeats: int) -> tuple[object, float]:
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, min(times)


def angles(transforms: list[dict], sun: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elevation of rays, and their azimuth from the sun's, in degrees, of shape (frames, h, w)."""
    directions = np.stack([camera_rays(t, t["transform_matrix"])[1].numpy() for t in transforms])
    elevation = np.degrees(np.arcsin(directions[..., 2].clip(-1, 1)))
    azimuth = np.degrees(np.arctan2(directions[..., 1], directions[..., 0]) - math.atan2(sun[1], sun[0]))
    return elevation, np.abs((azimuth + 180) % 360 - 180)


def binned(ours: np.ndarray, theirs: np.ndarray, values: np.ndarray, bounds, keep=None) -> dict:
    """Ratio of the light of both, per bin of a per-pixel value."""
    result = {}
    keep = np.ones(values.shape, dtype=bool) if keep is None else keep
    for low, high in zip(bounds[:-1], bounds[1:]):
        mask = keep & (values >= low) & (values < high)
        if mask.any():
            result[f"{low:g}..{high:g}"] = float(ours[mask].sum() / theirs[mask].sum())
    return result


def evaluate_day(repeats: int) -> dict:
    scene = day_scene(FRAMES)
    reference = ROOT / "ms" / "cycles_ref"
    fog = load_images(reference / "volume" / "direct", FRAMES) + load_images(reference / "volume" / "indirect", FRAMES)
    frames, inscatter, seconds = [], [], []
    for i in range(len(FRAMES)):
        result, s = fastest(lambda i=i: scene.closed_form(i), repeats)
        frames.append(result.radiance.cpu().numpy())
        inscatter.append(result.inscatter.cpu().numpy())
        seconds.append(s)
    frames, inscatter = np.stack(frames), np.stack(inscatter)
    sun = np.asarray(scene.lighting.suns[0].direction, dtype=float)
    elevation, azimuth = angles(scene.transforms, sun)
    sky = np.stack([d >= 1e9 for d in load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]])
    return {
        "seconds": min(seconds),
        "whole": errors(frames, scene.reference),
        "fog": errors(inscatter, fog),
        "surfaces": errors(frames - inscatter, scene.reference - fog),
        "fog_by_azimuth": binned(inscatter, fog, azimuth, AZIMUTHS),
        "fog_by_azimuth_sky": binned(inscatter, fog, azimuth, AZIMUTHS, sky),
        "fog_by_elevation": binned(inscatter, fog, elevation, ELEVATIONS),
    }


def evaluate_noobjects(repeats: int) -> dict:
    from breakdown import distances
    from run_experiment import GROUND_ALBEDO

    scene = day_scene(FRAMES)
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    lighting = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    medium = medium.model_copy(update={"multiple_scattering": True})
    reference = ROOT / "noclamp" / "noobjects32"
    fog = load_images(reference / "volume" / "direct", FRAMES) + load_images(reference / "volume" / "indirect", FRAMES)
    # With no volume bounces, Cycles' paths end at the fog, which then scatters no light reflected by the ground
    once_reference = ROOT / "bounces_noobjects" / "0" / "volume"
    single_reference = load_images(once_reference / "direct", FRAMES) + load_images(once_reference / "indirect", FRAMES)
    ours, once, seconds = [], [], []
    for t in scene.transforms:
        depth = distances(t)[0]
        zeros = np.zeros((*depth.shape, 3))
        result, s = fastest(
            lambda zeros=zeros, depth=depth, t=t: apply_medium(
                zeros, depth, t, t["transform_matrix"], medium, lighting, dtype=DTYPE
            ),
            repeats,
        )
        ours.append(result.inscatter.cpu().numpy())
        single, black = (
            medium.model_copy(update={"multiple_scattering": False}),
            lighting.model_copy(update={"ground_albedo": (0.0,)}),
        )
        once.append(apply_medium(zeros, depth, t, t["transform_matrix"], single, black, dtype=DTYPE).inscatter.numpy())
        seconds.append(s)
    ours, once = np.stack(ours), np.stack(once)
    sun = np.asarray(lighting.suns[0].direction, dtype=float)
    elevation, azimuth = angles(scene.transforms, sun)
    near_horizon = np.abs(elevation) < 10
    rest_reference = fog - single_reference
    return {
        "seconds": min(seconds),
        "fog": errors(ours, fog),
        "single": errors(once, single_reference),
        # Light reflected by the ground and light scattered more than once, against Cycles' fog light minus the light it
        # scatters once, which is noisier
        "rest_ratio": float((ours - once).sum() / rest_reference.sum()),
        "fog_by_elevation": binned(ours, fog, elevation, ELEVATIONS),
        "fog_by_azimuth_near_horizon": binned(ours, fog, azimuth, AZIMUTHS, near_horizon),
        "rest_by_azimuth_near_horizon": binned(ours - once, rest_reference, azimuth, AZIMUTHS, near_horizon),
        "rest_by_elevation": binned(ours - once, rest_reference, elevation, ELEVATIONS),
    }


def evaluate_night(repeats: int) -> dict:
    from run_experiment import GROUND_ALBEDO

    scene = night_scene(NIGHT_FRAMES)
    multiple = scene.medium.model_copy(update={"multiple_scattering": True})
    # The night's lighting has no ground albedo, which only matters to light that the ground reflects into the fog
    lighting = scene.lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    clear = load_images(OUT / "night_clear" / "frames", NIGHT_FRAMES)
    normals = day_scene(NIGHT_FRAMES).extra["normals"]
    frames, inscatter, once, reflected, seconds = [], [], [], [], []
    for i, t in enumerate(scene.transforms):

        def run(medium: Medium, i=i, t=t):
            return apply_medium(
                clear[i],
                scene.depth[i],
                t,
                t["transform_matrix"],
                medium,
                lighting,
                occlusion=scene.occlusion,
                normals=normals[i],
                dtype=DTYPE,
            )

        result, s = fastest(lambda: run(multiple), repeats)
        frames.append(result.radiance.cpu().numpy())
        inscatter.append(result.inscatter.cpu().numpy())
        # Light scattered once, without the light that surfaces reflect, and the light that surfaces reflect and the
        # fog scatters once
        once.append(scene.closed_form(i).inscatter.cpu().numpy())
        reflected.append(scene.closed_form(i, lighting=lighting).inscatter.cpu().numpy() - once[-1])
        seconds.append(s)
    frames, inscatter, once, reflected = np.stack(frames), np.stack(inscatter), np.stack(once), np.stack(reflected)
    reference = OUT / NIGHT_REFERENCE
    direct = load_images(reference / "volume" / "direct", NIGHT_FRAMES)
    indirect = load_images(reference / "volume" / "indirect", NIGHT_FRAMES)
    fog_only = load_images(OUT / "night_ms_black_noclamp" / "volume" / "indirect", NIGHT_FRAMES)
    reference_frames = load_images(reference / "frames", NIGHT_FRAMES)
    fog = direct + indirect
    luma = direct.mean(axis=-1)
    shadow = luma < 0.02 * np.median(luma[luma > 0])
    rest = inscatter - once
    return {
        "seconds": min(seconds),
        "whole": errors(frames, reference_frames),
        "fog": errors(inscatter, fog),
        "surfaces": errors(frames - inscatter, reference_frames - fog),
        "fog_shadow_pixels": errors(inscatter, fog, shadow),
        "fog_lit_pixels": errors(inscatter, fog, ~shadow),
        "single_vs_direct": errors(once, direct),
        "rest_vs_indirect": errors(rest, indirect),
        # Light reflected by surfaces and scattered once, against all the light that surfaces reflect into the fog
        "reflected_once_over_reflected": float(reflected.sum() / (indirect - fog_only).sum()),
        # The rest without it, i.e. light scattered more than once (the halo, which also holds light reflected by
        # surfaces when the halo gathers it), against the fog's own light scattered more than once
        "halo_over_fog_only": float((rest - reflected).sum() / fog_only.sum()),
        "halo_over_fog_only_shadow_pixels": float((rest - reflected)[shadow].sum() / fog_only[shadow].sum()),
        "indirect_over_fog_only": float(indirect.sum() / fog_only.sum()),
    }


CASES = {"day": evaluate_day, "noobjects": evaluate_noobjects, "night": evaluate_night}


def main(args):
    global NIGHT_REFERENCE
    NIGHT_REFERENCE = args.night_reference
    torch.set_num_threads(args.threads)
    results = {
        "library": str(Path(visionsim.__file__).parent.parent),
        "device": "cpu",
        "threads": args.threads,
        "night_reference": args.night_reference,
    }
    for case in args.cases:
        results[case] = CASES[case](args.repeats)
        print(case, json.dumps(results[case], indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--repeats", type=int, default=1, help="runs of each frame, of which the fastest is timed")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads")
    parser.add_argument("--out", help="JSON file to save the results to")
    parser.add_argument(
        "--night-reference",
        default=NIGHT_REFERENCE,
        help="night reference, relative to runs/fog-comparison/acceleration",
    )
    main(parser.parse_args())
