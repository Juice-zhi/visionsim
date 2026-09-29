"""Measure how much light multiple scattering adds, from frames rendered with more and more volume bounces.

Frames 255 and 345 of the animation (dataset indices 250 and 340, the latter being where the closed form is furthest
from the reference) are rendered by Cycles with at most 0, 1, 2, ... volume scattering events, where 0 is single
scattering as modeled by the closed form, e.g. with::

    blender -b runs/fog-comparison/fog_ref.blend --python render_passes.py -- runs/fog-comparison/bounces/16 \\
        --samples 1024 --frame 255 345 --volume-bounces 16

Each render is compared with the single scattering one and with the one with the most bounces, as is the closed form,
and results are saved to ``results/bounces.json``.
"""

import json
import re

import numpy as np
from run_experiment import ROOT

from visionsim.dataset import Dataset

INDICES = (250, 340)


def load(path):
    frames = [np.asarray(Dataset.from_path(path)[i][0], dtype=float) for i in range(len(INDICES))]
    return np.stack([np.repeat(f, 3, -1) if f.shape[-1] == 1 else f[..., :3] for f in frames])


def relative_l1(x, reference):
    return float(np.abs(x - reference).sum() / np.abs(reference).sum())


def main():
    counts = sorted(int(p.name) for p in (ROOT / "bounces").iterdir() if p.is_dir() and p.name.isdigit())
    frames = {n: load(ROOT / "bounces" / str(n) / "frames") for n in counts}
    direct = {n: load(ROOT / "bounces" / str(n) / "volume" / "direct") for n in counts}
    indirect = {n: load(ROOT / "bounces" / str(n) / "volume" / "indirect") for n in counts}
    closed_form = np.load(ROOT / "results" / "cache" / "closed_form_500.npy")[list(INDICES)]
    depths = Dataset.from_path(ROOT / "clear" / "depths")
    sky = np.stack([np.asarray(depths[i][0])[..., 0] >= 1e9 for i in INDICES])
    single, most = frames[counts[0]], frames[counts[-1]]

    results = {"frames": [i + 5 for i in INDICES], "bounces": []}
    for n in counts:
        seconds = re.search(r"per_frame=([\d.]+)s", (ROOT / "bounces" / f"{n}.log").read_text(errors="replace"))
        results["bounces"].append(
            {
                "volume_bounces": n,
                "seconds_per_frame": float(seconds[1]) if seconds else None,
                # Light reaching the camera, relative to single scattering, over the whole frame and per region
                "brightness": float(frames[n].sum() / single.sum()),
                "brightness_sky": float(frames[n][sky].sum() / single[sky].sum()),
                "brightness_surfaces": float(frames[n][~sky].sum() / single[~sky].sum()),
                # Fraction of the light scattered by the fog towards the camera that was scattered more than once
                "indirect_share": float(indirect[n].sum() / (direct[n].sum() + indirect[n].sum())),
                "vs_single": relative_l1(frames[n], single),
                "vs_most": relative_l1(frames[n], most),
                "closed_form_error": relative_l1(closed_form, frames[n]),
            }
        )
        print(results["bounces"][-1], flush=True)

    # The single scattering render at 1024 spp against the 4096 spp reference, to check that they agree
    reference = np.stack(
        [
            np.asarray(d[0], dtype=float)[..., :3]
            for d in (Dataset.from_path(ROOT / "cycles_ref" / "frames")[i] for i in INDICES)
        ]
    )
    results["single_vs_reference"] = relative_l1(single, reference)
    print("single scattering at 1024 spp vs the reference:", results["single_vs_reference"])
    (ROOT / "results" / "bounces.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
