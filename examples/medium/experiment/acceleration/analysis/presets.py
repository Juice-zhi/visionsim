"""Errors and times of the Effects presets of the library (``EFFECT_PRESETS``), by day and at night, at factors 1 and 4.

With the library on ``PYTHONPATH``, runs the closed form with each preset on the inputs and references of
``final_benchmark.py`` (frames 255 and 345 at 320x180, frames without fog rendered at many samples), and reports whole
frames, fog and surfaces errors (``common.errors``), the warm time per frame (fastest of ``--repeats`` calls, averaged
over both frames), the first call (which computes what is kept from one frame to the next), and, at night, the memory the
halo keeps. Results are saved to ``runs/fog-comparison/acceleration/results/<--out>``. Run from the repository root::

    PYTHONPATH=<library> python examples/medium/experiment/acceleration/analysis/presets.py --device cuda --out p.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_benchmark import day_case, night_case, synchronize
from visionsim.medium import EFFECT_PRESETS
from visionsim.medium.upsampling import apply_medium_upsampled

from common import OUT, errors

FRAMES = (255, 345)


def run(case, i, factor, effects):
    t = case.scene.transforms[i]
    return apply_medium_upsampled(
        case.clear[i],
        case.depth[i],
        t,
        t["transform_matrix"],
        case.medium,
        case.lighting,
        factor=factor,
        occlusion=case.scene.occlusion,
        normals=case.normals[i],
        effects=effects,
        dtype=torch.float32,
        device=case.device,
    )


def measure(case, factor, effects, repeats):
    """Errors over both frames, warm time per frame and the first call's time."""
    from visionsim.medium import halos

    if hasattr(halos, "_HALOS"):
        halos._HALOS.clear()
    synchronize(case.device)
    start = time.perf_counter()
    first = run(case, 0, factor, effects)
    synchronize(case.device)
    first_seconds = time.perf_counter() - start
    images, fastest = [], []
    for i in range(len(case.frames)):
        best, result = float("inf"), first
        for _ in range(repeats):
            start = time.perf_counter()
            result = run(case, i, factor, effects)
            synchronize(case.device)
            best = min(best, time.perf_counter() - start)
        fastest.append(best)
        images.append(result)
    radiance = np.stack([r.radiance.cpu().numpy() for r in images])
    inscatter = np.stack([r.inscatter.cpu().numpy() for r in images])
    entry = {"seconds": float(np.mean(fastest)), "first_seconds": first_seconds}
    entry |= {"whole": errors(radiance, case.reference)}
    fog = getattr(case, "fog", None)
    if fog is not None:
        entry["fog"] = errors(inscatter, fog)
        entry["surfaces"] = errors(radiance - inscatter, case.reference - fog)
    if hasattr(halos, "_HALOS") and hasattr(halos, "_halo_bytes"):
        entry["halo_mb"] = [halos._halo_bytes(h) / 2**20 for h in halos._HALOS.values()]
    return entry


def main(args):
    from final_benchmark import NIGHT_REFERENCE

    from common import ROOT, load_images

    results: dict = {"device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"}
    for name in args.cases:
        case = day_case(FRAMES, args.device) if name == "day" else night_case(args.device)
        passes = (ROOT / "ms" / "cycles_ref" / "volume") if name == "day" else (NIGHT_REFERENCE / "volume")
        case.fog = load_images(passes / "direct", case.frames) + load_images(passes / "indirect", case.frames)
        results[name] = {}
        for preset, (effects, _) in EFFECT_PRESETS.items():
            for factor in args.factors:
                entry = measure(case, factor, effects, args.repeats)
                results[name][f"{preset}, factor {factor}"] = entry
                parts = " / ".join(f"{100 * entry[k]['rel_l1']:.2f}" for k in ("whole", "fog", "surfaces"))
                print(
                    f"{name} {preset:9s} factor {factor}: {1e3 * entry['seconds']:7.1f} ms/frame, first "
                    f"{entry['first_seconds']:6.2f} s, errors {parts}%, halo MB {entry.get('halo_mb')}",
                    flush=True,
                )
    out = OUT / "results" / args.out
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", nargs="+", default=["day", "night"], choices=["day", "night"])
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="presets.json")
    main(parser.parse_args())
