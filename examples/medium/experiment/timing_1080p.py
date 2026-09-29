"""Time the closed form and ray marching on 1920x1080 frames rendered by ``timing_1080p.ps1``, and compare the
closed form with Cycles' fog at that resolution. Results are appended to ``timing_1080p.txt``."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium, camera_rays
from visionsim.medium.raymarch import ray_march_medium

ROOT = Path("runs/fog-comparison")
BOX_MIN, BOX_MAX = np.array([-200.0, -200.0, -1.0]), np.array([200.0, 200.0, 25.0])

frames, depths = Dataset.from_path(ROOT / "hd" / "clear" / "frames"), Dataset.from_path(ROOT / "hd" / "clear" / "depths")
lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())


def inputs(i):
    frame, t = frames[i]
    depth = np.asarray(depths[i][0])[..., 0]
    origin, directions, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        exit_distance = np.where(
            directions != 0, (np.where(directions > 0, BOX_MAX, BOX_MIN) - origin) / directions, np.inf
        ).min(-1)
    depth = np.where(depth >= 1e9, exit_distance / scale, depth)
    return np.asarray(frame, dtype=np.float32)[..., :3], depth.astype(np.float32), t


results = {}
for name, fn in (
    (
        "closed_form",
        lambda f, d, t: apply_medium(
            f, d, t, t["transform_matrix"], medium, lighting, device="cuda", dtype=torch.float32
        ),
    ),
    (
        "raymarch_16",
        lambda f, d, t: ray_march_medium(
            f, d, t, t["transform_matrix"], medium, lighting, steps=16, device="cuda", dtype=torch.float32
        ),
    ),
):
    fn(*inputs(0))
    elapsed = []
    for i in range(len(frames)):
        args = inputs(i)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args).radiance.cpu()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    results[name] = float(np.mean(elapsed))

# How close the closed form is to Cycles' default settings at this resolution, for reference
cycles = Dataset.from_path(ROOT / "hd" / "cycles_default" / "frames")
f, d, t = inputs(5)
ours = (
    apply_medium(f, d, t, t["transform_matrix"], medium, lighting, device="cuda", dtype=torch.float32)
    .radiance.cpu()
    .numpy()
)
reference = np.asarray(cycles[5][0], dtype=np.float32)[..., :3]
results["closed_form_vs_cycles_default_rel_l1"] = float(np.abs(ours - reference).sum() / np.abs(reference).sum())
with open(ROOT / "timing_1080p.txt", "a", encoding="utf-8") as log:
    log.write(json.dumps({"seconds_per_frame": results, "resolution": list(f.shape[:2])}) + "\n")
print(results)
