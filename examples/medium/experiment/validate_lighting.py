"""Validate the closed form's multiple scattering and point lights against Cycles, where objects don't get in the way.

- Multiple scattering: frames 255 and 345 of the scene without objects (the ``noobjects`` variant of
  ``breakdown_scenes.py``) are rendered by Cycles with 0 and 32 volume bounces, e.g.::

      blender -b runs/fog-comparison/variants/fog_noobjects.blend --python render_passes.py -- \\
          runs/fog-comparison/bounces_noobjects/32 --samples 1024 --frame 255 345 --volume-bounces 32

  and the light scattered by the fog towards the camera (Cycles' volume passes) is compared with the closed form's
  in-scattering, without and with multiple scattering and light reflected by the ground, by elevation of the rays.
- Lamps: the same frames of night variants lit by a single lamp (``lamp_scene.py``), rendered by Cycles with 0 volume
  bounces into ``runs/fog-comparison/<lamp>/bounces0``, with their lighting exported by ``export_lighting.py`` to
  ``runs/fog-comparison/<lamp>/lighting.json``, for a point light (``point``), a spot light (``spot``), an area light
  (``area``), a point light whose light is smoothed near it by a Light Falloff node (``point_smooth``), and a point
  light right behind a cube, which casts its shadow onto the fog in front of it (``point_shadow``), e.g.::

      blender -b runs/fog-comparison/fog_ref.blend --python lamp_scene.py -- runs/fog-comparison/variants/fog_spot.blend \\
          --type SPOT --position 1.5 22 4 --direction -0.35 -0.3 -0.89 --power 4000 --radius 0.05 --angle 0.9
      blender -b runs/fog-comparison/variants/fog_spot.blend --python render_passes.py -- \\
          runs/fog-comparison/spot/bounces0 --samples 4096 --frame 255 345 --volume-bounces 0

  Lamps whose directory holds shadow maps, exported by ``export_occlusion.py`` to ``occlusion.npz``, are also compared
  with the shadows that objects cast onto the fog, e.g. for a point light behind a cube::

      blender -b runs/fog-comparison/fog_ref.blend --python lamp_scene.py -- \\
          runs/fog-comparison/variants/fog_point_shadow.blend --position 4 21 1.5 --power 3000 --radius 0.05
      blender -b runs/fog-comparison/variants/fog_point_shadow.blend --python export_occlusion.py -- \\
          runs/fog-comparison/point_shadow/occlusion.npz --exclude Plane

Results are saved to ``results/validation.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from breakdown import distances
from run_experiment import GROUND_ALBEDO, ROOT, box_depth

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, Occlusion, apply_medium, camera_rays, load_occlusion

INDICES = (250, 340)
ELEVATIONS = (-90, -10, -4, -1.5, 0, 1.5, 4, 10, 25, 90)
LAMPS = ("point", "spot", "area", "point_smooth", "point_shadow")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def blur(images: np.ndarray, size: int = 5) -> np.ndarray:
    """Box blur of a stack of images, of shape (n, h, w, c), over windows of ``size`` pixels."""
    kernel = np.ones(size) / size
    images = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, images)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 2, images)


def load(path: Path) -> np.ndarray:
    dataset = Dataset.from_path(path)
    frames = [np.asarray(dataset[i][0], dtype=np.float64) for i in range(len(INDICES))]
    return np.stack([np.repeat(f, 3, -1) if f.shape[-1] == 1 else f[..., :3] for f in frames])


def fog_light(path: Path) -> np.ndarray:
    return load(path / "volume" / "direct") + load(path / "volume" / "indirect")


def in_scattering(
    medium: Medium, lighting: Lighting, depth: np.ndarray, transforms: list[dict], occlusion: Occlusion | None = None
) -> np.ndarray:
    return np.stack(
        [
            apply_medium(
                np.zeros((*d.shape, 3)),
                d,
                t,
                t["transform_matrix"],
                medium,
                lighting,
                occlusion=occlusion,
                device=DEVICE,
            )
            .inscatter.cpu()
            .numpy()
            for d, t in zip(depth, transforms)
        ]
    )


def by_elevation(ours: np.ndarray, theirs: np.ndarray, elevation: np.ndarray) -> dict:
    """Ratio of the total light of both, overall and for rays within ranges of elevations, in degrees."""
    bounds = np.sin(np.radians(ELEVATIONS))
    ratios = {"all": float(ours.sum() / theirs.sum())}
    for lo, hi, a, b in zip(bounds[:-1], bounds[1:], ELEVATIONS[:-1], ELEVATIONS[1:]):
        mask = (elevation >= lo) & (elevation < hi)
        if mask.any():
            ratios[f"{a:+g}..{b:+g}"] = float(ours[mask].sum() / theirs[mask].sum())
    return ratios


def main():
    frames = Dataset.from_path(ROOT / "clear" / "frames")
    transforms = [frames[i][1] for i in INDICES]
    elevation = np.stack([camera_rays(t, t["transform_matrix"])[1][..., 2].numpy() for t in transforms])
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    results = {"frames": [i + 5 for i in INDICES], "ground_albedo": GROUND_ALBEDO}

    # Without objects, rays either hit the ground or leave the fog volume
    depth = np.stack([distances(t)[0] for t in transforms])
    results["single_vs_0_bounces"] = by_elevation(
        in_scattering(medium, lighting, depth, transforms), fog_light(ROOT / "bounces_noobjects" / "0"), elevation
    )
    results["multiple_vs_32_bounces"] = by_elevation(
        in_scattering(
            medium.model_copy(update={"multiple_scattering": True}),
            lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3}),
            depth,
            transforms,
        ),
        fog_light(ROOT / "bounces_noobjects" / "32"),
        elevation,
    )

    # Lamps, in the scene with objects, which barely shadow the fog from lamps in front of the camera
    depths = Dataset.from_path(ROOT / "clear" / "depths")
    depth = np.stack([box_depth(np.asarray(depths[i][0])[..., 0], t) for i, t in zip(INDICES, transforms)])
    for lamp in LAMPS:
        if not (ROOT / lamp / "bounces0").exists():
            continue
        lighting = Lighting.model_validate_json((ROOT / lamp / "lighting.json").read_text())
        ours = in_scattering(medium, lighting, depth, transforms)
        theirs = load(ROOT / lamp / "bounces0" / "volume" / "direct")
        results[f"{lamp}_vs_0_bounces"] = {
            "ratio": float(ours.sum() / theirs.sum()),
            "rel_l1": float(np.abs(ours - theirs).sum() / theirs.sum()),
            # Errors of blurred images, which leave out Cycles' noise
            "rel_l1_blurred": float(np.abs(blur(ours) - blur(theirs)).sum() / theirs.sum()),
        }
        np.savez_compressed(ROOT / lamp / "comparison.npz", ours=ours, theirs=theirs)
        if (ROOT / lamp / "occlusion.npz").exists():
            # Objects cast the lamp's shadows onto the fog
            occlusion = load_occlusion(ROOT / lamp / "occlusion.npz", dtype=torch.float64)
            shadowed = in_scattering(medium, lighting, depth, transforms, occlusion)
            results[f"{lamp}_shadows_vs_0_bounces"] = {
                "ratio": float(shadowed.sum() / theirs.sum()),
                "rel_l1": float(np.abs(shadowed - theirs).sum() / theirs.sum()),
                "rel_l1_blurred": float(np.abs(blur(shadowed) - blur(theirs)).sum() / theirs.sum()),
            }
            np.savez_compressed(ROOT / lamp / "comparison.npz", ours=ours, theirs=theirs, shadowed=shadowed)
        if (ROOT / lamp / "bounces32").exists():
            results[f"{lamp}_vs_32_bounces"] = {"ratio": float(ours.sum() / fog_light(ROOT / lamp / "bounces32").sum())}

    (ROOT / "results" / "validation.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
