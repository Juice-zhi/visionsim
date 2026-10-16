"""Direction 1: Monte Carlo, i.e. Cycles' path tracing at several budgets, with denoising and path guiding.

For each render of ``render_all.ps1``, measures the time per frame (leaving out the first frame, which loads the scene),
the error against the reference on frames 255 and 345, as is and once blurred (bias without noise), and how the error
changes from one frame to the next over frames 255 to 262 (flicker, which event cameras turn into false events).

Also measures a hybrid of the closed form and Monte Carlo, in which the closed form computes the light scattered once
by the fog, and path tracing only the rest (surfaces, and light scattered more than once): Cycles' combined image
minus its volume direct pass, plus the closed form's single scattering. Results are saved to ``results/mc.json``.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

from common import DAY_FRAMES, NIGHT_FRAMES, OUT, ROOT, day_scene, errors, flicker, load_images, night_scene

STILL = (255, 345)
SEQUENCE = tuple(range(255, 263))


def frame_times(name: str) -> list[float]:
    text = (OUT / "logs" / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    return [float(s) for f, s in re.findall(r"FRAME_TIME frame=(\d+) seconds=([\d.]+)", text) if int(f) != 254]


def main():
    results: dict = {"day": {}, "night": {}}

    # Day scene: whole images against the reference with multiple scattering
    day = day_scene(DAY_FRAMES)
    reference = {f: day.reference[i] for i, f in enumerate(DAY_FRAMES)}
    still = np.stack([reference[f] for f in STILL])
    sequence = np.stack([reference[f] for f in SEQUENCE])
    volume_ref = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "direct", STILL) + load_images(
        ROOT / "ms" / "cycles_ref" / "volume" / "indirect", STILL
    )
    seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", STILL)
    results["day"]["reference_seed1"] = {"still": errors(seed1, still), "seconds": 4.422}

    # Light scattered once, by the closed form, on the frames where Monte Carlo's single scattering is replaced
    single = day.medium.model_copy(update={"multiple_scattering": False})
    lighting = day.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    start = time.perf_counter()
    single_scattering = {}
    for f in STILL + SEQUENCE:
        if f not in single_scattering:
            i = DAY_FRAMES.index(f)
            single_scattering[f] = day.closed_form(i, normals=None, medium=single, lighting=lighting).inscatter.numpy()
    results["day"]["closed_form_single_seconds"] = (time.perf_counter() - start) / len(single_scattering)

    names = [f"day_spp{s}" for s in (4, 16, 64, 256, 1024)] + [f"day_guided{s}" for s in (16, 64, 256)]
    names += ["day_default"]
    for name in names:
        folder = OUT / name
        entry: dict = {"seconds": float(np.mean(frame_times(name)))}
        for kind, sub in (("denoised", "frames"), ("noisy", "noisy")):
            frames = {f: img for f, img in zip(DAY_FRAMES, load_images(folder / sub, DAY_FRAMES))}
            entry[kind] = {
                "still": errors(np.stack([frames[f] for f in STILL]), still),
                "flicker": flicker(np.stack([frames[f] for f in SEQUENCE]), sequence),
            }
            if kind == "noisy":
                noisy = frames
        direct = {
            f: img for f, img in zip(STILL + SEQUENCE, load_images(folder / "volume" / "direct", STILL + SEQUENCE))
        }
        indirect = load_images(folder / "volume" / "indirect", STILL)
        entry["fog_light"] = errors(np.stack([direct[f] for f in STILL]) + indirect, volume_ref)
        hybrid = {f: noisy[f] - direct[f] + single_scattering[f] for f in STILL + SEQUENCE}
        entry["hybrid"] = {
            "still": errors(np.stack([hybrid[f] for f in STILL]), still),
            "flicker": flicker(np.stack([hybrid[f] for f in SEQUENCE]), sequence),
        }
        results["day"][name] = entry
        print(name, json.dumps(entry), flush=True)
    seed1_64 = load_images(OUT / "day_spp64_seed1" / "noisy", STILL)
    results["day"]["spp64_seed_noise"] = errors(seed1_64, load_images(OUT / "day_spp64" / "noisy", STILL))

    # Night scene: light scattered by the fog (volume direct pass, noisy), and whole images, denoised or not
    night = night_scene(NIGHT_FRAMES)
    whole_ref = load_images(ROOT / "point_shadow" / "bounces0" / "frames", NIGHT_FRAMES)
    for name in [f"night_spp{s}" for s in (16, 64, 256, 1024)] + [f"night_guided{s}" for s in (64, 256)]:
        folder = OUT / name
        entry = {
            "seconds": float(np.mean(frame_times(name))),
            "fog_light": errors(load_images(folder / "volume" / "direct", NIGHT_FRAMES), night.reference),
            "whole_noisy": errors(load_images(folder / "noisy", NIGHT_FRAMES), whole_ref),
            "whole_denoised": errors(load_images(folder / "frames", NIGHT_FRAMES), whole_ref),
        }
        results["night"][name] = entry
        print(name, json.dumps(entry), flush=True)

    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "mc.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
