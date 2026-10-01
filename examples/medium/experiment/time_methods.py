"""Time the closed form and ray marching per frame, at every resolution rendered for the experiment.

Frames are taken from the clear renders at 320x180 (``render_all.ps1``), 800x800 (``timing_resolution.ps1``) and
1920x1080 (``timing_1080p.ps1``). As other programs may use the GPU in bursts, each configuration is timed many times,
interleaved with the others, each time once the GPU is otherwise idle, and its time is the minimum over these samples;
the median is saved too, to show how much the measurements were disturbed. Results are saved to
``results/timings.json``, which ``make_report.py`` uses for the time taken by each method.
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch
from run_experiment import FRAME, GROUND_ALBEDO, OCCLUSION, ROOT, box_depth

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium.occlusion import load_occlusion
from visionsim.medium.raymarch import ray_march_medium
from visionsim.medium.scattering import multiple_scattering

FOLDERS = {"320x180": ROOT / "clear", "800x800": ROOT / "res800x800" / "clear", "1920x1080": ROOT / "hd" / "clear"}


def load(folder: Path, index: int) -> tuple:
    frame, transform = Dataset.from_path(folder / "frames")[index]
    assert isinstance(transform, dict)
    depth = box_depth(np.asarray(Dataset.from_path(folder / "depths")[index][0])[..., 0], transform)
    return np.asarray(frame, dtype=np.float32)[..., :3], depth.astype(np.float32), transform


def wait_for_idle_gpu(threshold: int = 10, timeout: float = 120.0) -> None:
    """Wait until other programs stop using the GPU, to start timing at the beginning of an idle period."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        if int(subprocess.run(query, capture_output=True, text=True, check=True).stdout.split()[0]) < threshold:
            return
        time.sleep(0.2)


def sample(fn: Callable[[], object], wait: float = 120.0) -> float:
    wait_for_idle_gpu(timeout=wait)
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main(args):
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    lighting_ms = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    medium_ms = medium.model_copy(update={"multiple_scattering": True})
    kwargs = {"device": "cuda", "dtype": torch.float32}
    m1 = Lighting(ambient=lighting.sky, suns=lighting.suns)
    occlusion = load_occlusion(OCCLUSION, **kwargs)
    methods = {
        "closed_form": lambda f, d, t: apply_medium(f, d, t, t["transform_matrix"], medium, lighting, **kwargs),
        "closed_form_ms": lambda f, d, t: apply_medium(f, d, t, t["transform_matrix"], medium_ms, lighting_ms, **kwargs),
        "closed_form_occ": lambda f, d, t: apply_medium(
            f, d, t, t["transform_matrix"], medium, lighting, occlusion=occlusion, **kwargs
        ),
        "closed_form_ms_occ": lambda f, d, t: apply_medium(
            f, d, t, t["transform_matrix"], medium_ms, lighting_ms, occlusion=occlusion, **kwargs
        ),
        "closed_form_m1": lambda f, d, t: apply_medium(f, d, t, t["transform_matrix"], medium, m1, **kwargs),
        "raymarch_16": lambda f, d, t: ray_march_medium(
            f, d, t, t["transform_matrix"], medium, lighting, steps=16, **kwargs
        ),
        "raymarch_64s16": lambda f, d, t: ray_march_medium(
            f, d, t, t["transform_matrix"], medium, lighting, steps=64, shadow_steps=16, **kwargs
        ),
    }
    # (resolution, method, number of samples), leaving out ray marching at high resolutions, which takes long and
    # isn't used by the report
    plan = [
        (r, m, 6)
        for r in FOLDERS
        for m in ("closed_form", "closed_form_ms", "closed_form_m1", "closed_form_occ", "closed_form_ms_occ")
    ]
    plan += [("320x180", "raymarch_16", 6), ("320x180", "raymarch_64s16", 4), ("800x800", "raymarch_16", 4)]
    inputs = {
        r: load(folder, min(FRAME, len(Dataset.from_path(folder / "frames")) - 1)) for r, folder in FOLDERS.items()
    }

    # Ray marching against its number of steps, on the frame of run_experiment.py's convergence study
    for shadow, steps_list in ((0, (2, 4, 8, 16, 32, 64, 128, 256)), (16, (16, 64, 256))):
        for steps in steps_list:
            name = f"raymarch_{steps}" + (f"s{shadow}" if shadow else "")
            methods.setdefault(
                name,
                lambda f, d, t, steps=steps, shadow=shadow: ray_march_medium(
                    f, d, t, t["transform_matrix"], medium, lighting, steps=steps, shadow_steps=shadow, **kwargs
                ),
            )
            plan.append(("convergence", name, 4))
    inputs["convergence"] = inputs["320x180"]

    beta = torch.tensor([medium.extinction], device="cuda", dtype=torch.float32)
    methods["table"] = lambda f, d, t: multiple_scattering(medium_ms, lighting_ms, beta, 3)
    plan.append(("table", "table", 6))
    inputs["table"] = inputs["320x180"]

    if args.only:
        # Only time some methods at every resolution, and update their times in the existing results
        plan = [(r, m, args.samples) for r in FOLDERS for m in args.only]

    samples: dict = defaultdict(list)
    for resolution, name, _ in plan:
        methods[name](*inputs[resolution])  # warm up, and build the tables that are cached across frames
    for index in range(max(n for *_, n in plan)):
        for resolution, name, count in plan:
            if index < count:
                samples[resolution, name].append(sample(partial(methods[name], *inputs[resolution])))
        print(f"round {index + 1} done", flush=True)

    results: dict = json.loads((ROOT / "results" / "timings.json").read_text()) if args.only else {"median": {}}
    for (resolution, name), values in samples.items():
        results.setdefault(resolution, {})[name] = min(values)
        results["median"][f"{resolution}/{name}"] = float(np.median(values))
    if args.only:
        (ROOT / "results" / "timings.json").write_text(json.dumps(results, indent=1))
        print(json.dumps({r: {m: results[r][m] for m in args.only} for r in FOLDERS}, indent=1))
        return
    results["multiple_scattering_table"] = results.pop("table")["table"]
    convergence = results.pop("convergence")
    results["convergence"] = {
        "closed_form": results["320x180"]["closed_form"],
        "ray_marching": [
            {
                "steps": steps,
                "shadow_steps": shadow,
                "seconds": convergence[f"raymarch_{steps}" + (f"s{shadow}" if shadow else "")],
            }
            for shadow, steps_list in ((0, (2, 4, 8, 16, 32, 64, 128, 256)), (16, (16, 64, 256)))
            for steps in steps_list
        ],
    }
    (ROOT / "results" / "timings.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "median"}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="only time these methods, and update them in results/timings.json")
    parser.add_argument("--samples", type=int, default=6, help="number of timings of each method, with --only")
    parser.add_argument("--wait", type=float, default=120.0, help="longest wait for the GPU to be idle, in seconds")
    args = parser.parse_args()
    sample = partial(sample, wait=args.wait)
    main(args)
