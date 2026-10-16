"""Direction 2: the closed form (``apply_medium``), i.e. the current approach, as a baseline for the others.

Measures, on the CPU, the time per frame without shadows, with the shadows of objects (shadow maps), and with surfaces
lit through the fog, at 320x180 and 1920x1080, the errors against Cycles, the flicker over consecutive frames, and the
storage of the shadow maps. Frames are timed several times, and the fastest time is kept, as other programs share the
CPU. The closed form's results at 1080p are saved for ``exp_less.py``. Results are saved to ``results/closed_form.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from visionsim.medium import apply_medium

from common import (
    DAY_FRAMES,
    DTYPE,
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


def fastest(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def storage(path: Path) -> dict:
    """Size of the shadow maps on disk (compressed) and in memory (32-bit floats)."""
    with np.load(path) as data:
        in_memory = sum(data[k].nbytes for k in data.files if k.endswith("_depths"))
        texels = {k[: -len("_depths")]: int(data[k].size) for k in data.files if k.endswith("_depths")}
    return {"file_bytes": path.stat().st_size, "memory_bytes": int(in_memory), "texels": texels}


def main():
    results: dict = {}
    day = day_scene(DAY_FRAMES)
    still = [DAY_FRAMES.index(f) for f in (255, 345)]
    sequence = [DAY_FRAMES.index(f) for f in range(255, 263)]

    outputs = {i: day.closed_form(i).radiance.numpy() for i in range(len(DAY_FRAMES))}
    inscatter = {i: day.closed_form(i).inscatter.numpy() for i in still}
    volume = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "direct", (255, 345)) + load_images(
        ROOT / "ms" / "cycles_ref" / "volume" / "indirect", (255, 345)
    )
    results["day"] = {
        "still": errors(np.stack([outputs[i] for i in still]), day.reference[still]),
        "flicker": flicker(np.stack([outputs[i] for i in sequence]), day.reference[sequence]),
        "fog_light": errors(np.stack([inscatter[i] for i in still]), volume),
        "seconds": {
            "no_shadows": fastest(lambda: day.closed_form(still[1], occlusion=None, normals=None)),
            "shadows": fastest(lambda: day.closed_form(still[1], normals=None)),
            "shadows_surfaces": fastest(lambda: day.closed_form(still[1])),
        },
        "storage": storage(ROOT / "occlusion" / "occlusion.npz"),
        "occlusion_export_seconds": 3.74,
    }
    np.save(OUT / "results" / "closed_form_day.npy", np.stack([outputs[i] for i in range(len(DAY_FRAMES))]))
    print("day", json.dumps(results["day"]), flush=True)

    # 1080p, timing, and its results for the low resolution experiments
    hd = ROOT / "hd" / "clear"
    frames = (255, 256)
    transforms = load_transforms(hd / "frames", frames)
    clear = load_images(hd / "frames", frames)
    depth = np.stack([box_depth(d, t) for d, t in zip(load_images(hd / "depths", frames)[..., 0], transforms)])
    t = transforms[0]
    kwargs = {"occlusion": day.occlusion, "dtype": DTYPE}

    def hd_frame(occlusion: bool = True):
        options = kwargs if occlusion else {"dtype": DTYPE}
        return apply_medium(clear[0], depth[0], t, t["transform_matrix"], day.medium, day.lighting, **options)

    full = hd_frame()
    np.savez_compressed(
        OUT / "results" / "closed_form_hd.npz",
        radiance=full.radiance.numpy(),
        inscatter=full.inscatter.numpy(),
        transmittance=full.transmittance.numpy(),
        depth=depth[0],
    )
    results["day_1080p"] = {
        "seconds": {"no_shadows": fastest(lambda: hd_frame(False), 2), "shadows": fastest(hd_frame, 2)}
    }
    print("day_1080p", json.dumps(results["day_1080p"]), flush=True)

    # Night scene: light scattered by the fog, and whole images from the render without fog, lit through the fog
    night = night_scene(NIGHT_FRAMES)
    light = np.stack([night.closed_form(i).inscatter.numpy() for i in range(len(NIGHT_FRAMES))])
    clear_night = load_images(OUT / "night_clear" / "frames", NIGHT_FRAMES)
    normals = day_scene(NIGHT_FRAMES).extra["normals"]
    whole = np.stack(
        [
            apply_medium(
                clear_night[i],
                night.depth[i],
                night.transforms[i],
                night.transforms[i]["transform_matrix"],
                night.medium,
                night.lighting,
                occlusion=night.occlusion,
                normals=normals[i],
                dtype=DTYPE,
            ).radiance.numpy()
            for i in range(len(NIGHT_FRAMES))
        ]
    )
    results["night"] = {
        "fog_light": errors(light, night.reference),
        "whole": errors(whole, load_images(ROOT / "point_shadow" / "bounces0" / "frames", NIGHT_FRAMES)),
        "seconds": {
            "no_shadows": fastest(lambda: night.closed_form(1, occlusion=None)),
            "shadows": fastest(lambda: night.closed_form(1)),
        },
        "storage": storage(ROOT / "point_shadow" / "occlusion.npz"),
    }
    np.save(OUT / "results" / "closed_form_night.npy", light)
    print("night", json.dumps(results["night"]), flush=True)
    results["threads"] = torch.get_num_threads()
    (OUT / "results" / "closed_form.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
