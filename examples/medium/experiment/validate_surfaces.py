"""Validate how surfaces are lit through the fog, against Cycles, on frames 255 and 345.

The light that surfaces send through the fog in Cycles is the combined image minus the volume passes, divided by the
transmittance along camera rays. It is compared with the frame rendered without fog, as is, and scaled by the closed
form's illumination (``apply_medium`` given the normals of surfaces, see :mod:`visionsim.medium.surfaces`):

- by day, lit by the sun and the sky, with Cycles' renders with 0 and 32 volume bounces (``bounces.py``) and a render
  without fog with world-space normals (``render_passes.py --light-groups`` into ``runs/fog-comparison/lightgroups``);
- at night, lit by a point, spot or area light (see ``validate_lighting.py``), with renders of the same variants
  without their fog volume, e.g.::

      blender -b runs/fog-comparison/variants/clear_spot.blend --python render_passes.py -- \\
          runs/fog-comparison/spot/clear --samples 512 --frame 255 345 --no-volume-passes --light-groups

Note that Cycles' renders with 0 volume bounces still include light scattered once by the fog on its way to surfaces,
which then reflect it towards the camera, as the light is sampled from the point where it scatters. Results are saved to
``results/surfaces.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from run_experiment import GROUND_ALBEDO, ROOT

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium.optics import optical_depth
from visionsim.medium.render import camera_rays

FRAMES = {255: 250, 345: 340}
"""Frames of the animation rendered with their passes, and their index in the dataset"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(path: Path, k: int) -> np.ndarray:
    return np.asarray(Dataset.from_path(path)[k][0], dtype=np.float64)[..., :3]


def compare(clear_dir: Path, fog_dir: Path, medium: Medium, lighting: Lighting) -> dict:
    """Light of surfaces through the fog, Cycles' against the render without fog as is, and lit through the fog."""
    regions: dict[str, dict[str, float]] = {}
    for k, index in enumerate(FRAMES.values()):
        clear, normals = load(clear_dir / "frames", k), load(clear_dir / "normals", k)
        volume = load(fog_dir / "volume" / "direct", k) + load(fog_dir / "volume" / "indirect", k)
        depth, transform = Dataset.from_path(ROOT / "clear" / "depths")[index]
        depth = np.asarray(depth, dtype=np.float64)[..., 0]
        result = apply_medium(
            clear, depth, transform, transform["transform_matrix"], medium, lighting, normals=normals, device=DEVICE
        )
        origin, directions, scale = camera_rays(transform, transform["transform_matrix"])
        surface = torch.as_tensor(depth < 1e9)
        distance = torch.where(surface, torch.as_tensor(depth) * scale, torch.zeros_like(scale))
        tau = optical_depth(medium, origin, directions, distance) * medium.extinction
        measured = (torch.as_tensor(load(fog_dir / "frames", k) - volume)) / torch.exp(-tau)[..., None]
        lit = torch.as_tensor(clear) * result.illumination.cpu()
        mask = surface & (measured.sum(dim=-1) > 0.01) & (distance < 15)
        normal_z = torch.as_tensor(normals[..., 2])
        for name, region in (
            ("all", mask),
            ("ground", mask & (normal_z > 0.95)),
            ("walls", mask & (normal_z.abs() < 0.3)),
        ):
            entry = regions.setdefault(
                name, {"clear": 0.0, "lit": 0.0, "measured": 0.0, "clear_error": 0.0, "lit_error": 0.0}
            )
            for key, image in (("clear", torch.as_tensor(clear)), ("lit", lit), ("measured", measured)):
                entry[key] += float(image[region].sum())
            entry["clear_error"] += float((torch.as_tensor(clear) - measured)[region].abs().sum())
            entry["lit_error"] += float((lit - measured)[region].abs().sum())
    return {
        name: {
            "clear_ratio": e["clear"] / e["measured"],
            "lit_ratio": e["lit"] / e["measured"],
            "clear_rel_l1": e["clear_error"] / e["measured"],
            "lit_rel_l1": e["lit_error"] / e["measured"],
        }
        for name, e in regions.items()
        if e["measured"] > 0
    }


def main():
    fog = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    day = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    day = day.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    results = {
        "day_vs_0_bounces": compare(ROOT / "lightgroups", ROOT / "bounces" / "0", fog, day),
        "day_vs_32_bounces": compare(
            ROOT / "lightgroups", ROOT / "bounces" / "32", fog.model_copy(update={"multiple_scattering": True}), day
        ),
    }
    for lamp in ("point", "spot", "area"):
        # Cycles clamps indirect light by default, which darkens the glow lighting surfaces near lamps
        fog_dir = ROOT / lamp / "noclamp_bounces0"
        fog_dir = fog_dir if fog_dir.exists() else ROOT / lamp / "bounces0"
        if (ROOT / lamp / "clear").exists() and fog_dir.exists():
            lighting = Lighting.model_validate_json((ROOT / lamp / "lighting.json").read_text())
            results[f"{lamp}_vs_0_bounces"] = compare(ROOT / lamp / "clear", fog_dir, fog, lighting)
    (ROOT / "results" / "surfaces.json").write_text(json.dumps(results, indent=1))
    for name, regions in results.items():
        for region, entry in regions.items():
            print(name, region, {k: round(v, 3) for k, v in entry.items()})


if __name__ == "__main__":
    sys.exit(main())
