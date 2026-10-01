"""Check the shadows that objects cast onto the fog against Cycles, on a single frame and per source of light.

Compares the light scattered towards the camera by the fog with Cycles' volume direct pass (single scattering) for the
variants saved by ``breakdown_scenes.py``: lit by the sun only, with objects casting shadows (light shafts), and lit by
the sky only, with objects hiding part of it. The closed form is evaluated with and without the shadow maps saved by
``export_occlusion.py``. Results are saved to ``results/occlusion.json``.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from breakdown import ROOT, distances, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium.occlusion import load_occlusion

INDEX = 250  # frame 255 of the animation


def errors(mine: np.ndarray, cycles: np.ndarray, sky: np.ndarray) -> dict:
    """Ratio of the fog's light to Cycles' over sky and surface pixels, and mean relative error per pixel."""
    result = {
        region: float(mine[mask][:, 1].sum() / cycles[mask][:, 1].sum())
        for region, mask in (("sky", sky), ("surfaces", ~sky))
    }
    result["rel_l1"] = float(np.abs(mine - cycles).sum() / cycles.sum())
    return result


def main(args):
    clear, transform = Dataset.from_path(ROOT / "clear" / "frames")[INDEX]
    clear = np.asarray(clear, dtype=float)[..., :3]
    depth, background = distances(transform, np.asarray(Dataset.from_path(ROOT / "clear" / "depths")[INDEX][0])[..., 0])
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    occlusion = load_occlusion(args.occlusion)
    kwargs = {"device": args.device, "dtype": torch.float64 if args.device == "cpu" else torch.float32}

    results: dict = {}
    images = {}
    for variant, subset in (("sun", Lighting(suns=lighting.suns)), ("sky", Lighting(sky=lighting.sky))):
        cycles = load(ROOT / "breakdown" / variant / "volume" / "direct")
        for name, maps in (("without", None), ("with", occlusion)):
            start = time.perf_counter()
            result = apply_medium(
                clear, depth, transform, transform["transform_matrix"], medium, subset, occlusion=maps, **kwargs
            )
            seconds = time.perf_counter() - start
            mine = result.inscatter.cpu().numpy()
            results[f"{variant}_{name}"] = errors(mine, cycles, background) | {"seconds": seconds}
            images[f"{variant}_{name}"] = mine
        images[f"{variant}_cycles"] = cycles
        print(variant, json.dumps({k: v for k, v in results.items() if k.startswith(variant)}, indent=1), flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "occlusion.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(ROOT / "occlusion" / "frame255.npz", background=background, **images)
    figure(images, ROOT / "results" / "images" / "occlusion.png")


def figure(images: dict[str, np.ndarray], path: Path) -> None:
    """Light scattered by the fog in Cycles and in the closed form, with and without shadows, and their errors."""
    fig, axes = plt.subplots(2, 5, figsize=(17, 4.6), layout="constrained")
    titles = {"cycles": "Cycles（参考）", "without": "闭式解，无阴影", "with": "闭式解 + 阴影"}
    for row, (variant, name) in enumerate((("sun", "只有太阳"), ("sky", "只有天光"))):
        cycles = images[f"{variant}_cycles"]
        scale = np.percentile(cycles, 99.5)
        for col, kind in enumerate(titles):
            axes[row, col].imshow(np.clip(images[f"{variant}_{kind}"] / scale, 0, 1) ** (1 / 2.2))
            axes[row, col].set_title(f"{name}：{titles[kind]}", fontsize=10)
        for col, kind in ((3, "without"), (4, "with")):
            error = (images[f"{variant}_{kind}"] - cycles).mean(-1) / np.maximum(cycles.mean(-1), 1e-6)
            shown = axes[row, col].imshow(error, cmap="coolwarm", vmin=-0.3, vmax=0.3)
            axes[row, col].set_title(f"{name}：{titles[kind]}的相对误差", fontsize=10)
        fig.colorbar(shown, ax=axes[row, :], shrink=0.9, pad=0.01, format=lambda v, _: f"{v:+.0%}")
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--occlusion", default=str(ROOT / "occlusion" / "occlusion.npz"), help="shadow maps")
    parser.add_argument("--device", default="cpu", help="torch device")
    main(parser.parse_args())
