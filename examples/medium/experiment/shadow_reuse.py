"""Reuse the shadows of a frame for fogs of other densities, as when generating a dataset with several fogs.

Shadows are traced once per frame with the fog of ``media/ground_fog.json`` (multiple scattering included), then reused
for fogs whose density is scaled by a range of factors, and compared with tracing shadows for each fog, on every
tenth frame, with all the lights of the scene and with the sun or the sky alone (single scattering). The time to trace
shadows and to add each fog with them is measured on the three resolutions of the
experiment, with the GPU otherwise idle as in ``time_methods.py``, along with the memory taken by the shadows of a
frame. Results are saved to ``results/shadow_reuse.json``.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from run_experiment import GROUND_ALBEDO, OCCLUSION, ROOT, box_depth
from time_methods import FOLDERS, load, sample

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium, load_occlusion, trace_shadows

SCALES = (0.25, 0.5, 0.75, 1.5, 2.0, 4.0)


def nbytes(value) -> int:
    """Memory taken by the tensors of shadows, or of any of their fields."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, tuple):
        return sum(nbytes(v) for v in value)
    return 0


def main(args):
    kwargs = {"device": "cuda", "dtype": torch.float32}
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    lighting = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    medium = medium.model_copy(update={"multiple_scattering": True})
    occlusion = load_occlusion(OCCLUSION, **kwargs)
    results: dict = {"errors": {}, "seconds": {}, "bytes": {}}
    single = medium.model_copy(update={"multiple_scattering": False})

    # Error of reusing shadows for other densities, relative to the light that shadows remove and to the light
    # scattered by the fog, summed over frames
    sources = {
        "all": (medium, lighting),
        "sun": (single, Lighting(suns=lighting.suns)),
        "sky": (single, Lighting(sky=lighting.sky)),
    }
    frames = Dataset.from_path(ROOT / "clear" / "frames")
    depths = Dataset.from_path(ROOT / "clear" / "depths")
    sums = {source: {scale: np.zeros(3) for scale in SCALES} for source in sources}
    for index in range(0, len(frames), 10):
        clear, transform = frames[index]
        assert isinstance(transform, dict)
        clear = np.asarray(clear, dtype=np.float32)[..., :3]
        depth = box_depth(np.asarray(depths[index][0])[..., 0], transform)
        args_ = (clear, depth, transform, transform["transform_matrix"])
        for source, (fog, light) in sources.items():
            shadows = trace_shadows(occlusion, depth, transform, transform["transform_matrix"], fog, light, **kwargs)
            for scale in SCALES:
                other = fog.model_copy(update={"extinction": scale * fog.extinction})
                plain = apply_medium(*args_, other, light, **kwargs).inscatter
                own = apply_medium(*args_, other, light, occlusion=occlusion, **kwargs).inscatter
                reused = apply_medium(*args_, other, light, shadows=shadows, **kwargs).inscatter
                sums[source][scale] += [float(t.abs().sum()) for t in (reused - own, plain - own, own)]
    for source, by_scale in sums.items():
        results["errors"][source] = {
            str(scale): {"of_shadowed_light": float(error / shadowed), "of_inscatter": float(error / inscatter)}
            for scale, (error, shadowed, inscatter) in by_scale.items()
        }
    print(json.dumps(results["errors"], indent=1), flush=True)

    # Time to trace shadows once, and to add a fog with them, with single or multiple scattering
    plain_lighting = lighting.model_copy(update={"ground_albedo": (0.0,) * 3})
    for resolution, folder in FOLDERS.items():
        frame, depth, transform = load(folder, min(250, len(Dataset.from_path(folder / "frames")) - 1))
        pose = transform["transform_matrix"]
        steps = {"trace": partial(trace_shadows, occlusion, depth, transform, pose, medium, lighting, **kwargs)}
        shadows = steps["trace"]()
        results["bytes"][resolution] = nbytes(shadows)
        for name, fog, light in (("closed_form_occ", single, plain_lighting), ("closed_form_ms_occ", medium, lighting)):
            steps[name] = partial(apply_medium, frame, depth, transform, pose, fog, light, shadows=shadows, **kwargs)
            steps[name]()
        timed: dict = {name: [] for name in steps}
        for _ in range(args.samples):
            for name, step in steps.items():
                timed[name].append(sample(step, wait=args.wait))
        results["seconds"][resolution] = {name: min(values) for name, values in timed.items()}
        print(resolution, results["seconds"][resolution], flush=True)
    (ROOT / "results" / "shadow_reuse.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=6, help="number of timings of each step")
    parser.add_argument("--wait", type=float, default=120.0, help="longest wait for the GPU to be idle, in seconds")
    main(parser.parse_args())
