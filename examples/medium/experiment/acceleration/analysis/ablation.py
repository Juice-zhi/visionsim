"""Ablation of the closed form: what each term of the model costs per frame, and what error it removes.

The library is imported as usual, so that a version under development, e.g. in a git worktree, is evaluated by putting
it first on ``PYTHONPATH``; this script and the data it reads stay in the main checkout, whose root it runs from::

    PYTHONPATH=E:/visionsim-wt/effects python examples/medium/experiment/acceleration/analysis/ablation.py \\
        --threads 4 --out effects-api_ablation.json

The switches are read from the library's :class:`Effects <visionsim.medium.model.Effects>`, so that new switches are
included as they are added. The closed form is evaluated with every switch on, with each switch off in turn, and with
``Medium.multiple_scattering`` off, at factors 1 and 4 of :func:`apply_medium_upsampled
<visionsim.medium.upsampling.apply_medium_upsampled>`, on the same inputs and references as ``eval_closed_form.py``, at
320x180 on frames 255 and 345:

- ``day``: the day scene with objects, multiple scattering, shadow maps and surfaces lit through the fog, against
  Cycles 4096 spp with 16 volume bounces (``ms/cycles_ref``);
- ``night``: the night scene, a lamp behind a cube and the ground's mean albedo, with the same inputs, against Cycles
  65536 spp with 16 volume bounces without clamping (``gpu/night_full_spp65536``).

For each, whole frames, the fog's light (its volume passes) and surfaces (frames minus volume passes) are compared with
``common.errors`` (relative L1 over all pixels and channels of both frames, as is and blurred, and the ratio of light).
A switch that isn't effective in a case, as its prerequisites aren't met (e.g. the shadows of suns at night, see
``Effects.missing``), leaves the frames exactly as they are with every switch on, as the library's tests check: it isn't
run again, and is reported as ineffective, with what it misses.

Times are the warm time per frame: after a first call, which computes what is kept from one frame to the next (tables,
halos, the light that surfaces reflect), the fastest of ``--repeats`` calls of each frame, averaged over both frames,
with the inputs already on the device. As times measured on a loaded CPU are noisy, variants are timed back to back, in
the given order and then in the reverse order (``--orders``), and each order is reported, along with the fastest of
both. Results are saved as JSON to ``runs/fog-comparison/acceleration/results/<--out>``, and a table of the time saved
and the error added by turning each switch off is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import visionsim
from visionsim.medium import Effects
from visionsim.medium.upsampling import apply_medium_upsampled

from common import DTYPE, GROUND_ALBEDO, NIGHT_FRAMES, OUT, ROOT, day_scene, errors, load_images, night_scene

FRAMES = (255, 345)
NIGHT_REFERENCE = OUT / "gpu" / "night_full_spp65536"
RESULTS = OUT / "results"
ALL_ON = "all on"
MULTIPLE_OFF = "multiple_scattering off"


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class Case:
    """Inputs of the closed form for the frames of a scene, already on the device, and Cycles' references."""

    def __init__(self, name, scene, medium, lighting, clear, normals, reference, fog, device):
        self.name, self.scene, self.medium, self.lighting, self.device = name, scene, medium, lighting, device
        self.clear, self.normals = self.on_device(clear), self.on_device(normals)
        self.depth = self.on_device(scene.depth)
        self.reference, self.fog = reference, fog

    def on_device(self, images) -> torch.Tensor:
        return torch.as_tensor(np.asarray(images), dtype=DTYPE, device=self.device)

    def run(self, i: int, factor: int, effects: Effects, medium=None):
        t = self.scene.transforms[i]
        return apply_medium_upsampled(
            self.clear[i],
            self.depth[i],
            t,
            t["transform_matrix"],
            medium or self.medium,
            self.lighting,
            factor=factor,
            occlusion=self.scene.occlusion,
            normals=self.normals[i],
            effects=effects,
            dtype=DTYPE,
            device=self.device,
        )

    def missing(self, effects: Effects) -> dict[str, list[str]]:
        return effects.missing(self.medium, self.lighting, occlusion=True, normals=True)


def day_case(device: str) -> Case:
    scene = day_scene(FRAMES)
    reference = ROOT / "ms" / "cycles_ref"
    fog = load_images(reference / "volume" / "direct", FRAMES) + load_images(reference / "volume" / "indirect", FRAMES)
    return Case(
        "day", scene, scene.medium, scene.lighting, scene.clear, scene.extra["normals"], scene.reference, fog, device
    )


def night_case(device: str) -> Case:
    scene = night_scene(NIGHT_FRAMES)
    medium = scene.medium.model_copy(update={"multiple_scattering": True})
    # The night's lighting has no ground albedo, which only matters to light that the ground reflects into the fog
    lighting = scene.lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    clear = load_images(OUT / "night_clear" / "frames", NIGHT_FRAMES)
    normals = day_scene(NIGHT_FRAMES).extra["normals"]
    reference = load_images(NIGHT_REFERENCE / "frames", NIGHT_FRAMES)
    fog = load_images(NIGHT_REFERENCE / "volume" / "direct", NIGHT_FRAMES)
    fog = fog + load_images(NIGHT_REFERENCE / "volume" / "indirect", NIGHT_FRAMES)
    return Case("night", scene, medium, lighting, clear, normals, reference, fog, device)


CASES = {"day": day_case, "night": night_case}


def variants(case: Case, names: list[str]) -> dict[str, tuple[Effects, object]]:
    """Switches and medium of each variant: every switch on, each effective switch off, and multiple scattering off."""
    result = {ALL_ON: (Effects(), None)}
    missing = case.missing(Effects())
    for name in names:
        if not missing[name]:
            result[f"{name} off"] = (Effects(**{name: False}), None)
    result[MULTIPLE_OFF] = (Effects(), case.medium.model_copy(update={"multiple_scattering": False}))
    return result


def evaluate(case: Case, factor: int, effects: Effects, medium) -> tuple[dict, float]:
    """Errors of a variant against Cycles, and the time of its first call of both frames."""
    synchronize(case.device)
    start = time.perf_counter()
    results = [case.run(i, factor, effects, medium) for i in range(len(case.scene.transforms))]
    synchronize(case.device)
    first = time.perf_counter() - start
    frames = np.stack([r.radiance.cpu().numpy() for r in results])
    inscatter = np.stack([r.inscatter.cpu().numpy() for r in results])
    metrics = {
        "whole": errors(frames, case.reference),
        "fog": errors(inscatter, case.fog),
        "surfaces": errors(frames - inscatter, case.reference - case.fog),
    }
    return metrics, first


def warm_time(case: Case, factor: int, effects: Effects, medium, repeats: int) -> list[float]:
    """Fastest of ``repeats`` calls of each frame, once what is kept from one frame to the next is computed."""
    fastest = []
    for i in range(len(case.scene.transforms)):
        best = float("inf")
        for _ in range(repeats):
            synchronize(case.device)
            start = time.perf_counter()
            case.run(i, factor, effects, medium)
            synchronize(case.device)
            best = min(best, time.perf_counter() - start)
        fastest.append(best)
    return fastest


def table(results: dict) -> str:
    """Time saved and error added by turning each switch off, against every switch on, per case and factor."""
    lines = []
    for case, by_factor in results["cases"].items():
        for factor, entry in by_factor["factors"].items():
            on = entry["variants"][ALL_ON]
            lines.append(
                f"\n{case}, factor {factor}: all on {on['seconds'] * 1000:.0f} ms/frame, whole "
                f"{on['whole']['rel_l1']:.2%}, fog {on['fog']['rel_l1']:.2%}, surfaces {on['surfaces']['rel_l1']:.2%}"
            )
            header = f"{'variant':32s} {'saved ms':>9s} {'d whole':>8s} {'d fog':>8s} {'d surf':>8s} {'fog ratio':>9s}"
            lines.append(header)
            for name, variant in entry["variants"].items():
                if name == ALL_ON:
                    continue
                saved = (on["seconds"] - variant["seconds"]) * 1000
                added = {k: (variant[k]["rel_l1"] - on[k]["rel_l1"]) * 100 for k in ("whole", "fog", "surfaces")}
                lines.append(
                    f"{name:32s} {saved:9.1f} {added['whole']:+8.2f} {added['fog']:+8.2f} {added['surfaces']:+8.2f} "
                    f"{variant['fog']['ratio']:9.3f}"
                )
        if by_factor["ineffective"]:
            ineffective = (f"{k} ({', '.join(v)})" for k, v in by_factor["ineffective"].items())
            lines.append(f"{case}, ineffective: " + ", ".join(ineffective))
    return "\n".join(lines)


def write(results: dict, name: str) -> None:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1))


def main(args):
    # Paths are relative to the root of the repository
    os.chdir(Path(__file__).resolve().parents[5])
    torch.set_num_threads(args.threads)
    names = list(Effects.model_fields)
    if args.switches:
        names = [name for name in names if name in args.switches]
    declared = Effects.prerequisites()
    results: dict = {
        "library": str(Path(visionsim.__file__).parent.parent),
        "device": args.device,
        "threads": args.threads,
        "repeats": args.repeats,
        "orders": args.orders,
        "frames": {"day": list(FRAMES), "night": list(NIGHT_FRAMES)},
        "switches": {
            name: {"description": Effects.model_fields[name].description, **declared[name]}
            for name in Effects.model_fields
        },
        "cases": {},
    }
    for case_name in args.cases:
        case = CASES[case_name](args.device)
        chosen = variants(case, names)
        missing = case.missing(Effects())
        entry: dict = {"ineffective": {name: missing[name] for name in names if missing[name]}, "factors": {}}
        results["cases"][case_name] = entry
        for factor in args.factors:
            by_variant: dict = {}
            # Errors, and the first call, which computes what is kept from one frame to the next for each variant
            for name, (effects, medium) in chosen.items():
                metrics, first = evaluate(case, factor, effects, medium)
                by_variant[name] = {**metrics, "first_seconds": first, "times": {}}
                print(case_name, factor, name, json.dumps(metrics["whole"]), f"first {first:.2f} s", flush=True)
            # Warm times, back to back, in each order
            for order in args.orders:
                sequence = list(chosen.items()) if order == "forward" else list(reversed(chosen.items()))
                for name, (effects, medium) in sequence:
                    seconds = warm_time(case, factor, effects, medium, args.repeats)
                    by_variant[name]["times"][order] = seconds
                    print(case_name, factor, order, name, [f"{s * 1000:.0f} ms" for s in seconds], flush=True)
            for variant in by_variant.values():
                per_frame = np.min([variant["times"][order] for order in args.orders], axis=0)
                variant["seconds"] = float(np.mean(per_frame))
                variant["seconds_by_order"] = {o: float(np.mean(variant["times"][o])) for o in args.orders}
            entry["factors"][str(factor)] = {"variants": by_variant}
            if args.out:
                write(results, args.out)
    print(table(results))
    if args.out:
        write(results, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 4], help="factors of the upsampling")
    parser.add_argument("--switches", nargs="+", help="switches to turn off, all of them by default")
    parser.add_argument("--repeats", type=int, default=3, help="calls of each frame, of which the fastest is timed")
    parser.add_argument("--orders", nargs="+", default=["forward", "reverse"], choices=["forward", "reverse"])
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--device", default="cpu", help="device of the closed form, e.g. cuda")
    parser.add_argument(
        "--out", help="JSON file to save the results to, within runs/fog-comparison/acceleration/results"
    )
    main(parser.parse_args())
