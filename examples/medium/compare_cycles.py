"""Compare the closed-form medium against the Cycles volume rendered by ``cycles_reference.py``.

Run ``cycles_reference.py`` first, then this script with the same output directory::

    python compare_cycles.py output/
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from visionsim.dataset import Dataset
from visionsim.medium import HeightFog, Lighting, Medium, apply_medium, camera_rays
from visionsim.utils.color import linearrgb_to_srgb

# Must match the scene of `cycles_reference.py`
EXTINCTION, FALLOFF, ANISOTROPY = 0.04, 6.0, 0.6
BOX_MIN, BOX_MAX = np.array([-300.0, -300.0, -1.0]), np.array([300.0, 300.0, 99.0])


def load(path):
    data, transform = Dataset.from_path(path)[0]
    # Gray renders are collapsed to a single channel when loaded
    return np.broadcast_to(data, (*data.shape[:2], 3)) if data.shape[-1] == 1 else data[..., :3], transform


def blocks(x, k=4):
    h, w = x.shape[:2]
    return x[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def tonemap(x, scale):
    return (linearrgb_to_srgb(np.clip(x * scale, 0, 1)) * 255).astype(np.uint8)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    radiance, transform = load(root / "clear" / "frames")
    depth = Dataset.from_path(root / "clear" / "depths")[0][0][..., 0].copy()
    lighting = Lighting.model_validate_json((root / "clear" / "lighting.json").read_text())
    medium = Medium(
        extinction=EXTINCTION,
        anisotropy=ANISOTROPY,
        components=[HeightFog(density=1.0, falloff=FALLOFF)],
        sun_attenuation=True,
    )

    # The fog volume is finite in Cycles, so rays that don't hit a surface stop where they leave it
    origin, directions, scale = (x.numpy() for x in camera_rays(transform, transform["transform_matrix"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        exit_distance = np.where(
            directions != 0, (np.where(directions > 0, BOX_MAX, BOX_MIN) - origin) / directions, np.inf
        ).min(axis=-1)
    background = depth >= 1e9
    depth[background] = (exit_distance / scale)[background]

    result = apply_medium(radiance, depth, transform, transform["transform_matrix"], medium, lighting)
    inscatter, total = result.inscatter.numpy(), result.radiance.numpy()
    reference_total, _ = load(root / "reference" / "frames")
    reference_direct, _ = load(root / "reference" / "volume" / "direct")
    reference_indirect, _ = load(root / "reference" / "volume" / "indirect")

    # Block averages reduce Cycles' Monte Carlo noise, edges still differ as depth maps are not anti-aliased
    error = np.abs(blocks(inscatter) - blocks(reference_direct)) / blocks(reference_direct)
    print(f"Multiple scattering in reference (should be zero): {np.abs(reference_indirect).max():.2e}")
    print(
        f"In-scattering, relative error over 4x4 blocks: median {np.median(error):.2%}, p90 {np.percentile(error, 90):.2%}"
    )
    print(f"In-scattering, mean ratio to Cycles: {inscatter.mean() / reference_direct.mean():.4f}")

    panel = (radiance[..., 1] > 4.5) & ~background
    for name, mask in (("emissive panel", panel), ("sky", background)):
        ratio = total[mask].mean() / reference_total[mask].mean()
        print(f"Radiance through the fog, {name} ({mask.sum()} pixels): ratio to Cycles {ratio:.4f}")

    scale = 1 / np.percentile(reference_direct, 99)
    relative = np.abs(inscatter - reference_direct) / (reference_direct + 1e-6)
    error_image = (np.clip(relative * 10, 0, 1) * 255).astype(np.uint8)
    iio.imwrite(
        root / "comparison.png",
        np.concatenate([tonemap(inscatter, scale), tonemap(reference_direct, scale), error_image], axis=1),
    )
    print(f"Saved in-scattering (closed form, Cycles, 10x relative error) to {root / 'comparison.png'}")
