"""Direction 5, with Unreal Engine 5.8: its exponential height fog's volumetric fog, against Cycles.

Unreal's frames are rendered by ``unreal/run_unreal.ps1`` with Movie Render Queue (one sample per frame, as in real
time, along the camera's path so that the volumetric fog's temporal accumulation has a history), at its High, Epic and
Cinematic qualities of volumetric fog, and without fog. Their outputs are linear and their exposure is fixed, and the
sky and ground of the frames without fog match Cycles' within a few percent, so they are compared as is:

- whole images against Cycles' references (multiple scattering by day, single and multiple scattering at night), and
  their flicker over frames 255 to 262;
- the light scattered by the fog, i.e. the frame with fog minus the frame without fog times the fog's transmittance,
  computed exactly, against Cycles' volume passes and the closed form's in-scattering.

Frame times come from Unreal's CSV profiler in game (``-csvGpuStats``), through the camera at frame 345, as the mean
GPU time (median, which leaves out hitches) over the last 600 of 900 frames, with and without fog. Results are saved to ``results/unreal_<tag>.json``.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import OpenEXR

from common import NIGHT_FRAMES, OUT, ROOT, day_scene, errors, flicker, load_images, night_scene

UNREAL = OUT / "unreal"
csv.field_size_limit(2**31 - 1)  # the last rows hold long metadata
QUALITIES = ("high", "epic", "cinematic")


def read(level: str, preset: str, frames) -> np.ndarray:
    images = []
    for frame in frames:
        with OpenEXR.File(str(UNREAL / "renders" / f"{level}_{preset}" / f"{frame:04d}.exr")) as f:
            images.append(np.asarray(f.channels()["RGBA"].pixels, dtype=np.float64)[..., :3])
    return np.stack(images)


def timings(tag: str) -> dict:
    """Mean frame and GPU times, in ms, over the last 600 frames of each CSV capture, and the volumetric fog's own GPU
    time when the profiler reports it."""
    result = {}
    for path in sorted((UNREAL / f"logs_{tag}").glob("time_*.csv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        frame = header.index("FrameTime")

        def number(text: str) -> bool:
            try:
                float(text)
                return True
            except ValueError:
                return False

        # The first column holds events, and the last rows metadata
        data = [r for r in rows[1:] if len(r) == len(header) and number(r[frame])][-600:]
        entry = {}
        for column in header:
            name = column.strip()
            if (
                name in ("FrameTime", "GPUTime", "RenderThreadTime", "GameThreadTime")
                or "VolumetricFog" in name
                or name
                in (
                    "GPU/Total",
                    "GPU/VolumetricFog",
                    "GPU/Fog",
                )
            ):
                k = header.index(column)
                try:
                    entry[name] = float(np.median([float(r[k]) for r in data]))
                except ValueError:
                    continue
        result[path.stem[len("time_") :]] = entry
    return result


def main(args):
    results: dict = {"day": {}, "night": {}}
    still = (255, 345)
    sequence = tuple(range(255, 263))
    day = day_scene(frames=still + sequence[1:-1] + (262,))
    index = {f: k for k, f in enumerate(day.frames)}
    transmittance = {f: day.transmittance(index[f]) for f in day.frames}
    volume = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "direct", still) + load_images(
        ROOT / "ms" / "cycles_ref" / "volume" / "indirect", still
    )
    single = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "direct", still)
    closed_form = np.stack([day.closed_form(index[f]).inscatter.cpu().numpy() for f in still])
    clear = read("Day", "nofog", still)
    clear_cycles = load_images(ROOT / "clear" / "frames", still)
    results["day"]["nofog_vs_cycles_clear"] = errors(clear, clear_cycles)
    for quality in QUALITIES:
        frames = read("Day", quality, still)
        fog_light = frames - np.stack([transmittance[f] for f in still]) * clear
        seq = read("Day", quality, sequence)
        results["day"][quality] = {
            "whole": errors(frames, day.reference[[index[f] for f in still]]),
            "flicker": flicker(seq, day.reference[[index[f] for f in sequence]]),
            "fog_light_vs_multiple": errors(fog_light, volume),
            "fog_light_vs_single": errors(fog_light, single),
            "fog_light_vs_closed_form": errors(fog_light, closed_form),
        }
        np.save(OUT / "results" / f"unreal_day_{quality}.npy", frames[1])
        print("day", quality, json.dumps(results["day"][quality]), flush=True)

    night = night_scene(NIGHT_FRAMES)
    transmittance = [night.transmittance(i) for i in range(len(NIGHT_FRAMES))]
    clear = read("Night", "nofog", NIGHT_FRAMES)
    whole_ref = load_images(ROOT / "point_shadow" / "bounces0" / "frames", NIGHT_FRAMES)
    closed_form = np.stack([night.closed_form(i).inscatter.cpu().numpy() for i in range(len(NIGHT_FRAMES))])
    # Unreal's default global illumination (Lumen) also lights the fog with light reflected by surfaces, which Cycles
    # only includes with volume bounces
    multiple = OUT / "night_ms_ref"
    multiple_light = load_images(multiple / "volume" / "direct", NIGHT_FRAMES) + load_images(
        multiple / "volume" / "indirect", NIGHT_FRAMES
    )
    multiple_whole = load_images(multiple / "frames", NIGHT_FRAMES)
    for quality in QUALITIES:
        frames = read("Night", quality, NIGHT_FRAMES)
        fog_light = frames - np.stack(transmittance) * clear
        results["night"][quality] = {
            "whole": errors(frames, whole_ref),
            "fog_light": errors(fog_light, night.reference),
            "fog_light_vs_closed_form": errors(fog_light, closed_form),
            "whole_vs_multiple": errors(frames, multiple_whole),
            "fog_light_vs_multiple": errors(fog_light, multiple_light),
        }
        np.save(OUT / "results" / f"unreal_night_{quality}.npy", fog_light[1])
        print("night", quality, json.dumps(results["night"][quality]), flush=True)
    results["timings"] = timings(args.tag)
    print(json.dumps(results["timings"], indent=1))
    (OUT / "results" / f"unreal_{args.tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="igpu", help="machine whose timings to read, see run_unreal.ps1")
    main(parser.parse_args())
