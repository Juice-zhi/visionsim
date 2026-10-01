"""Time the closed form and ray marching, and measure every method's error, on frames rendered by
``timing_resolution.ps1`` at another resolution, e.g. VisionSIM-50's 800x800.

Errors are relative to the multiple scattering reference rendered at that resolution. The closed form is also timed
with the lighting model before the sky was fixed, which skips integrating skylight over directions, as a stand-in for
the cost of the closed form once skylight is precomputed in a lookup table. Results are saved to ``results.json`` in
the folder of the renders.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from run_experiment import GROUND_ALBEDO, OCCLUSION, ROOT, box_depth, psnr, relative_l1, tonemap

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium.occlusion import load_occlusion
from visionsim.medium.raymarch import ray_march_medium


def load(path: Path) -> np.ndarray:
    dataset = Dataset.from_path(path)
    return np.stack([np.asarray(dataset[i][0], dtype=np.float32)[..., :3] for i in range(len(dataset))])


def main(args):
    folder = ROOT / args.folder
    dataset = Dataset.from_path(folder / "clear" / "frames")
    clear = load(folder / "clear" / "frames")
    transforms = [dataset[i][1] for i in range(len(dataset))]
    depths = Dataset.from_path(folder / "clear" / "depths")
    depth = [box_depth(np.asarray(depths[i][0])[..., 0], t) for i, t in enumerate(transforms)]
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    medium_ms = medium.model_copy(update={"multiple_scattering": True})
    lighting_ms = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    kwargs = {"device": "cuda", "dtype": torch.float32}
    # Shadow maps only depend on the scene, not on the camera's resolution
    occlusion = load_occlusion(OCCLUSION, **kwargs)
    methods = {
        "closed_form": lambda i: apply_medium(
            clear[i], depth[i], transforms[i], transforms[i]["transform_matrix"], medium, lighting, **kwargs
        ),
        "closed_form_ms": lambda i: apply_medium(
            clear[i], depth[i], transforms[i], transforms[i]["transform_matrix"], medium_ms, lighting_ms, **kwargs
        ),
        "closed_form_occ": lambda i: apply_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium,
            lighting,
            occlusion=occlusion,
            **kwargs,
        ),
        "closed_form_ms_occ": lambda i: apply_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium_ms,
            lighting_ms,
            occlusion=occlusion,
            **kwargs,
        ),
        "closed_form_without_sky_quadrature": lambda i: apply_medium(
            clear[i],
            depth[i],
            transforms[i],
            transforms[i]["transform_matrix"],
            medium,
            Lighting(ambient=lighting.sky, suns=lighting.suns),
            **kwargs,
        ),
        "raymarch_16": lambda i: ray_march_medium(
            clear[i], depth[i], transforms[i], transforms[i]["transform_matrix"], medium, lighting, steps=16, **kwargs
        ),
    }

    reference = load(folder / "ms_cycles_ref" / "frames")
    results = {"resolution": list(clear.shape[1:3]), "frames": len(clear), "seconds_per_frame": {}, "vs_multiple": {}}
    outputs = {"clear": clear}
    for name, fn in methods.items():
        fn(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        frames = [fn(i).radiance.cpu().numpy() for i in range(len(clear))]
        torch.cuda.synchronize()
        results["seconds_per_frame"][name] = (time.perf_counter() - start) / len(clear)
        outputs[name] = np.stack(frames)
    for name in ("cycles_default", "cycles_default_ms"):
        outputs[name] = load(folder / name / "frames")

    for name, frames in outputs.items():
        results["vs_multiple"][name] = {
            "rel_l1": float(np.mean([relative_l1(f, r) for f, r in zip(frames, reference)])),
            "psnr": float(np.mean([psnr(tonemap(f) / 255.0, tonemap(r) / 255.0) for f, r in zip(frames, reference)])),
        }

    # Blender render times: total wall time (including Blender's startup) and the rendering itself, from the log
    for line in (folder / "timing.txt").read_text(encoding="utf-8-sig").splitlines():
        if match := re.match(r"(\w+) exit=0 ([\d,.]+)s for (\d+) frames", line):
            results["seconds_per_frame"][f"{match[1]}_wall"] = float(match[2].replace(",", "")) / int(match[3])
    if match := re.search(r"per_frame=([\d.]+)s", (folder / "ms_cycles_ref.log").read_text(errors="replace")):
        results["seconds_per_frame"]["ms_cycles_ref_render"] = float(match[1])
    (folder / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default="res800x800", help="folder of the renders in runs/fog-comparison")
    main(parser.parse_args())
