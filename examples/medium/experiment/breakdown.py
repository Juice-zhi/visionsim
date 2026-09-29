"""Break down the difference between the closed form and Cycles on a single frame, by source of light.

Compares the light scattered towards the camera by the fog, i.e. Cycles' volume direct pass, with the closed form's
in-scattering for each variant saved by ``breakdown_scenes.py``, and estimates how much surfaces are dimmed by being
lit through the fog, which the closed form doesn't model. Results are saved to ``results/breakdown.json``.
"""

import json
from pathlib import Path

import numpy as np

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium, camera_rays

ROOT = Path("runs/fog-comparison")
INDEX = 250  # frame 255 of the animation
BOX_MIN, BOX_MAX = np.array([-200.0, -200.0, -1.0]), np.array([200.0, 200.0, 25.0])


def load(path):
    data = np.asarray(Dataset.from_path(path)[0][0], dtype=float)
    return np.repeat(data, 3, -1) if data.shape[-1] == 1 else data[..., :3]


def distances(transform, depth=None):
    """Depth to surfaces, or to where rays leave the fog volume, and where the ground is hit when no objects remain."""
    origin, directions, scale = (x.numpy() for x in camera_rays(transform, transform["transform_matrix"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        exit_distance = np.where(
            directions != 0, (np.where(directions > 0, BOX_MAX, BOX_MIN) - origin) / directions, np.inf
        ).min(-1)
        ground = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2], np.inf)
    if depth is None:
        hits = ground < exit_distance
        return np.where(hits, ground, exit_distance) / scale, ~hits
    background = depth >= 1e9
    return np.where(background, exit_distance / scale, depth), background


def main():
    clear, transform = Dataset.from_path(ROOT / "clear" / "frames")[INDEX]
    clear = np.asarray(clear, dtype=float)[..., :3]
    depth, background = distances(transform, np.asarray(Dataset.from_path(ROOT / "clear" / "depths")[INDEX][0])[..., 0])
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    pose = transform["transform_matrix"]

    def ours(lighting, radiance=clear, depth=depth):
        return apply_medium(radiance, depth, transform, pose, medium, lighting)

    def ratios(mine, cycles, sky):
        return {
            region: float(mine[mask][:, 1].sum() / cycles[mask][:, 1].sum())
            for region, mask in (("sky", sky), ("surfaces", ~sky))
        }

    results = {}
    full = ours(Lighting(sky=lighting.sky, suns=lighting.suns))
    old = ours(Lighting(ambient=lighting.sky, suns=lighting.suns))
    cycles_full = load(ROOT / "breakdown" / "full" / "volume" / "direct")
    results["full"] = ratios(full.inscatter.numpy(), cycles_full, background)
    results["full_before_fix"] = ratios(old.inscatter.numpy(), cycles_full, background)
    for variant, subset in (
        ("sun", Lighting(suns=lighting.suns)),
        ("sun_noshadow", Lighting(suns=lighting.suns)),
        ("sky", Lighting(sky=lighting.sky)),
    ):
        results[variant] = ratios(
            ours(subset).inscatter.numpy(), load(ROOT / "breakdown" / variant / "volume" / "direct"), background
        )

    # Without objects, rays either hit the ground or leave the fog volume
    empty_depth, empty_sky = distances(transform)
    empty = ours(Lighting(sky=lighting.sky), radiance=np.zeros_like(clear), depth=empty_depth)
    results["sky_noobjects"] = ratios(
        empty.inscatter.numpy(), load(ROOT / "breakdown" / "sky_noobjects" / "volume" / "direct"), empty_sky
    )

    # Surfaces lit through the fog: what remains of the reference once its in-scattering is removed, relative to the
    # clear render seen through the fog. Sky pixels check that transmittances agree, as the sky is not lit
    seen = load(ROOT / "breakdown" / "full" / "frames") - cycles_full
    expected = full.transmittance.numpy() * clear
    dimming = seen[~background][:, 1] / np.maximum(expected[~background][:, 1], 1e-6)
    results["surface_dimming"] = {
        "median": float(np.median(dimming)),
        "p10": float(np.percentile(dimming, 10)),
        "p90": float(np.percentile(dimming, 90)),
    }
    results["sky_transmittance"] = float(seen[background][:, 1].sum() / expected[background][:, 1].sum())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "breakdown.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
