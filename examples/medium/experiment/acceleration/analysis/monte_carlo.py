"""Where and why Monte Carlo (Cycles' path tracing) errs, on the day and the night scene of the acceleration study.

Answers, from the renders of ``render_all.ps1`` (320x180, frames 255 and 345, and 255 to 262 for flicker):

1. noise or bias: the relative L1 error as is and once blurred, against the reference's own noise (two seeds), and
   whether the noise falls as 1/sqrt(spp), with the L1 error and with the mean squared error minus the reference's;
2. which light the noise comes from: the residual split into surfaces, light scattered once (volume direct pass) and
   the rest of the light scattered by the fog (volume indirect pass), by L1 share and by variance share, which tells
   how much replacing the volume direct pass by the closed form (the "hybrid") could gain at best;
3. where OpenImageDenoise leaves bias: blurred errors of denoised images by image region (sky, distance, ground or
   objects, depth edges, texture edges, dark pixels, pixels the shadows of objects change), against the mean of the
   two reference seeds, which halves the reference's noise;
4. why path guiding flickers more: the correlation of the noise (render minus reference) between consecutive frames,
   for plain and guided renders, and per pass;
5. the same for the night scene's fog light (volume direct pass) and its denoised whole images, by lamp halo and
   shadow wedge.

Saves ``analysis/monte_carlo/monte_carlo.json`` and PNG figures alongside it. CPU only.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *

torch.set_num_threads(3)  # other jobs share the CPU

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

SAVE = OUT / "analysis" / "monte_carlo"
STILL = (255, 345)
SEQUENCE = tuple(range(255, 263))
DAY_SPP = (4, 16, 64, 256, 1024)
GUIDED_SPP = (16, 64, 256)
NIGHT_SPP = (16, 64, 256, 1024)
PASSES = ("surfaces", "single", "multiple")


# --------------------------------------------------------------------------------------------------------------------
# Helpers


def fast_blur(images: np.ndarray, size: int = 5) -> np.ndarray:
    """Same 5x5 box blur with zero padding as :func:`common.blur`, but vectorized."""
    return ndimage.uniform_filter(np.asarray(images, dtype=np.float64), size=(1, size, size, 1), mode="constant")


def luma(images: np.ndarray) -> np.ndarray:
    return np.asarray(images, dtype=np.float64) @ LUMA.astype(np.float64)


def rounded(value):
    """Floats rounded to 5 significant digits, recursively, for a readable JSON."""
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.5g}") if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def l1(ours, ref, total=None, keep=None, blurred=False) -> float:
    """Sum of |ours - ref| over a region (all pixels by default), over the sum of the reference over ``total``."""
    ours, ref = np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    diff = np.abs(fast_blur(ours) - fast_blur(ref)) if blurred else np.abs(ours - ref)
    total = np.asarray(ref if total is None else total, dtype=np.float64).sum()
    if keep is not None:
        diff = diff[keep]
    return float(diff.sum() / total) if total > 0 else float("nan")


def mse(ours, ref) -> float:
    return float(np.mean((np.asarray(ours, dtype=np.float64) - np.asarray(ref, dtype=np.float64)) ** 2))


def slope(spp, values) -> float:
    """Slope of log(value) against log(spp), -0.5 for noise falling as 1/sqrt(spp)."""
    return float(np.polyfit(np.log(spp), np.log(values), 1)[0])


def correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / norm) if norm > 0 else float("nan")


def tail_share(residual, fraction) -> float:
    """Share of the L1 error coming from the given fraction of pixels with the largest error."""
    per_pixel = np.abs(residual).sum(axis=-1).ravel()
    top = np.sort(per_pixel)[::-1][: max(1, int(len(per_pixel) * fraction))]
    return float(top.sum() / per_pixel.sum())


def excess_from_top(ours, ref, fraction=0.001) -> float:
    """Share of the excess light (sum ours - sum ref) that the given fraction of pixels with the largest positive
    error carries: close to 1 or above when a few outliers (fireflies) make the whole image brighter."""
    residual = (np.asarray(ours, dtype=np.float64) - ref).sum(axis=-1).ravel()
    top = np.sort(residual)[::-1][: max(1, int(len(residual) * fraction))]
    return float(top.sum() / residual.sum())


def where_top_errors(residual, masks, fraction=0.01) -> dict:
    """Share of the pixels with the largest |error| (given fraction) that fall in each region."""
    per_pixel = np.abs(residual).sum(axis=-1)
    top = per_pixel >= np.quantile(per_pixel, 1 - fraction)
    return {name: float((top & keep).sum() / top.sum()) for name, keep in masks.items()}


def region_table(ours, ref, masks, truth_floor=None) -> dict:
    """Per region: share of pixels and of the reference's light, contribution to the whole image's error (as is and
    blurred), the region's own relative error (as is and blurred), and its signed bias (sum ours - ref over the
    region's reference), all over the frames given."""
    ours, ref = np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    diff, diff_blurred = np.abs(ours - ref).sum(-1), np.abs(fast_blur(ours) - fast_blur(ref)).sum(-1)
    signed = (ours - ref).sum(-1)
    ref_pixels, total = ref.sum(-1), ref.sum()
    table = {}
    for name, keep in masks.items():
        if not keep.any():
            continue
        local = ref_pixels[keep].sum()
        table[name] = {
            "pixels": float(keep.mean()),
            "light": float(local / total),
            "share_of_error": float(diff[keep].sum() / total),
            "share_of_error_blurred": float(diff_blurred[keep].sum() / total),
            "local_rel_l1": float(diff[keep].sum() / local),
            "local_rel_l1_blurred": float(diff_blurred[keep].sum() / local),
            "local_bias": float(signed[keep].sum() / local),
        }
        if truth_floor is not None:
            table[name]["local_floor_blurred"] = float(truth_floor[keep].sum() / local)
    return table


def tonemap(image: np.ndarray) -> np.ndarray:
    scaled = image / (np.percentile(luma(image), 99) + 1e-12)
    return np.clip(scaled / (1 + scaled) * 2, 0, 1) ** (1 / 2.2)


def error_map(ours, ref, mean_ref, signed=False) -> np.ndarray:
    """Blurred error over the mean of the reference, summed (or averaged when signed) over channels."""
    diff = fast_blur(ours[None]) - fast_blur(ref[None])
    return (diff[0].mean(-1) if signed else np.abs(diff[0]).mean(-1)) / mean_ref


def outline(ax, mask, color):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.6)


# --------------------------------------------------------------------------------------------------------------------
# Day scene


def day_passes(folder, frames) -> dict[str, np.ndarray]:
    """Noisy image split into surfaces, single scattering (volume direct) and the rest of the fog's light (indirect)."""
    combined = load_images(folder / ("frames" if folder.name.startswith("cycles_ref") else "noisy"), frames)
    direct = load_images(folder / "volume" / "direct", frames)
    indirect = load_images(folder / "volume" / "indirect", frames)
    return {"total": combined, "surfaces": combined - direct - indirect, "single": direct, "multiple": indirect}


def day_masks(scene: Scene) -> dict[str, np.ndarray]:
    """Regions of the day scene's frames, each of shape (n, h, w)."""
    raw = load_images(ROOT / "clear" / "depths", scene.frames)[..., 0]
    sky = raw >= 1e9
    distance = np.stack(
        [scene.depth[i] * camera_rays(t, t["transform_matrix"])[2].numpy() for i, t in enumerate(scene.transforms)]
    )
    ground = ~sky & (scene.extra["normals"][..., 2] > 0.95)
    log_depth = np.log(np.maximum(scene.depth, 1e-3))
    jump = ndimage.maximum_filter(log_depth, size=(1, 3, 3)) - ndimage.minimum_filter(log_depth, size=(1, 3, 3))
    edges = ndimage.binary_dilation(jump > 0.1, structure=np.ones((1, 5, 5), dtype=bool))
    log_clear = np.log(luma(scene.clear) + 1e-4)
    contrast = ndimage.maximum_filter(log_clear, size=(1, 3, 3)) - ndimage.minimum_filter(log_clear, size=(1, 3, 3))
    texture = ndimage.binary_dilation(contrast > 0.5, structure=np.ones((1, 3, 3), dtype=bool)) & ~edges & ~sky
    reference = luma(scene.reference)
    dark = reference < np.percentile(reference, 20, axis=(1, 2), keepdims=True)

    # Pixels whose closed form changes by more than 2% (or 20%) of the reference when objects stop casting shadows:
    # 2% marks nearly every pixel, since objects hide part of the sky from all the fog, 20% the sun's shadows
    change = []
    for i in range(len(scene.frames)):
        with_shadows = scene.closed_form(i).radiance.numpy()
        without = scene.closed_form(i, occlusion=None).radiance.numpy()
        change.append(np.abs(with_shadows - without).sum(-1) / np.maximum(scene.reference[i].sum(-1), 1e-12))
    change = np.stack(change)
    return {
        "all": np.ones_like(sky),
        "sky": sky,
        "surface_0_5m": ~sky & (distance < 5),
        "surface_5_15m": ~sky & (distance >= 5) & (distance < 15),
        "surface_over_15m": ~sky & (distance >= 15),
        "ground": ground,
        "object_faces": ~sky & ~ground,
        "depth_edges": edges,
        "texture_edges": texture,
        "dark_20pct": dark,
        "shadow_affected_2pct": change > 0.02,
        "shadow_affected_20pct": change > 0.2,
        "shadow_affected_20pct_not_edge": (change > 0.2) & ~edges,
    }


def component_split(ours: dict, ref: dict) -> dict:
    """How the error of a noisy render splits between surfaces, single and multiple scattering."""
    total = ref["total"]
    residual = {c: ours[c].astype(np.float64) - ref[c] for c in PASSES}
    whole = sum(residual.values())
    squares = (whole**2).sum()
    split = {
        "rel_l1": l1(ours["total"], total),
        "rel_l1_blurred": l1(ours["total"], total, blurred=True),
        "passes": {},
        "covariance_share": {},
    }
    present = [c for c in PASSES if np.abs(ref[c]).sum() > 0 or np.abs(ours[c]).sum() > 0]
    for c in present:
        split["passes"][c] = {
            "light": float(ref[c].sum() / total.sum()),
            "share_of_error": l1(ours[c], ref[c], total=total),
            "share_of_error_blurred": l1(ours[c], ref[c], total=total, blurred=True),
            "bias": float((ours[c].sum() - ref[c].sum()) / total.sum()),
            "own_rel_l1": l1(ours[c], ref[c]),
            "variance_share": float((residual[c] ** 2).sum() / squares),
        }
    for a, b in (("surfaces", "single"), ("surfaces", "multiple"), ("single", "multiple")):
        if a not in present or b not in present:
            continue
        split["covariance_share"][f"{a}_{b}"] = float(2 * (residual[a] * residual[b]).sum() / squares)
        split["covariance_share"][f"{a}_{b}_correlation"] = correlation(residual[a], residual[b])
    # Error left if single scattering were exact (a perfect hybrid), keeping Monte Carlo's other passes
    oracle = total + residual["surfaces"] + residual["multiple"]
    split["perfect_single_scattering"] = {
        "rel_l1": l1(oracle, total),
        "rel_l1_blurred": l1(oracle, total, blurred=True),
    }
    return split


def analyze_day(results: dict) -> dict:
    figures = {}
    scene = day_scene(STILL)
    ref = day_passes(ROOT / "ms" / "cycles_ref", STILL)
    seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", STILL)
    truth = (ref["total"] + seed1) / 2  # half the reference's noise variance, for denoised images
    total = ref["total"]
    mean_ref = float(total.mean())

    # Reference's own noise: two seeds differ by sqrt(2) times one reference's noise, and by 2 times their mean's
    floor = {
        "seed_pair": errors(seed1, total),
        "one_reference_rel_l1": l1(seed1, total) / np.sqrt(2),
        "one_reference_rel_l1_blurred": l1(seed1, total, blurred=True) / np.sqrt(2),
        "mean_of_seeds_rel_l1": l1(seed1, total) / 2,
        "mean_of_seeds_rel_l1_blurred": l1(seed1, total, blurred=True) / 2,
        "one_reference_mse": mse(seed1, total) / 2,
        "pixel_noise_blur_factor": l1(seed1, total, blurred=True) / l1(seed1, total),
    }
    results["day"]["reference_floor"] = floor
    var_ref = floor["one_reference_mse"]
    print("day floor", json.dumps(rounded(floor)), flush=True)

    # 1. Noise against bias, per budget, and how the noise falls with spp
    renders = {f"spp{s}": OUT / f"day_spp{s}" for s in DAY_SPP} | {
        f"guided{s}": OUT / f"day_guided{s}" for s in GUIDED_SPP
    }
    renders["default"] = OUT / "day_default"
    budget, passes = {}, {}
    for name, folder in renders.items():
        passes[name] = day_passes(folder, STILL)
        noisy = passes[name]["total"]
        denoised = load_images(folder / "frames", STILL)
        noise_mse = mse(noisy, total) - var_ref
        budget[name] = {
            "noisy": errors(noisy, total),
            "noisy_against_seed1": errors(noisy, seed1),
            "noisy_noise_rel_rmse": float(np.sqrt(max(noise_mse, 0)) / mean_ref),
            "noisy_blurred_over_raw": l1(noisy, total, blurred=True) / l1(noisy, total),
            "noisy_top1pct_pixels_share": tail_share(noisy - total, 0.01),
            "denoised": errors(denoised, total),
            "denoised_against_mean_of_seeds": {
                "rel_l1": l1(denoised, truth),
                "rel_l1_blurred": l1(denoised, truth, blurred=True),
                "ratio": float(denoised.sum() / truth.sum()),
            },
        }
        if name.startswith("spp"):
            spp = int(name[3:])
            budget[name]["spp_times_noise_mse_over_mean2"] = spp * noise_mse / mean_ref**2
        passes[name]["denoised"] = denoised
        print(name, json.dumps(rounded(budget[name]["noisy"])), flush=True)
    plain = [budget[f"spp{s}"] for s in DAY_SPP]
    one = floor["one_reference_rel_l1"]
    scaling = {
        "slope_rel_l1": slope(DAY_SPP, [b["noisy"]["rel_l1"] for b in plain]),
        "slope_rel_l1_minus_reference_noise": slope(
            DAY_SPP, [np.sqrt(max(b["noisy"]["rel_l1"] ** 2 - one**2, 1e-12)) for b in plain]
        ),
        "slope_noise_rmse": slope(DAY_SPP, [b["noisy_noise_rel_rmse"] for b in plain]),
        "slope_noise_rmse_64_to_1024": slope(DAY_SPP[2:], [b["noisy_noise_rel_rmse"] for b in plain[2:]]),
        "slope_rel_l1_64_to_1024_minus_reference_noise": slope(
            DAY_SPP[2:], [np.sqrt(b["noisy"]["rel_l1"] ** 2 - one**2) for b in plain[2:]]
        ),
        "slope_denoised_rel_l1_blurred_vs_mean_of_seeds": slope(
            DAY_SPP, [b["denoised_against_mean_of_seeds"]["rel_l1_blurred"] for b in plain]
        ),
        "slope_guided_noise_rmse": slope(GUIDED_SPP, [budget[f"guided{s}"]["noisy_noise_rel_rmse"] for s in GUIDED_SPP]),
    }
    results["day"]["budget"] = budget
    results["day"]["scaling"] = scaling
    print("scaling", json.dumps(rounded(scaling)), flush=True)

    # 2. Which passes the error comes from, and what the hybrid gains at best
    single_medium = scene.medium.model_copy(update={"multiple_scattering": False})
    no_ground = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    closed_single = np.stack(
        [
            scene.closed_form(i, normals=None, medium=single_medium, lighting=no_ground).inscatter.numpy()
            for i in range(len(STILL))
        ]
    )
    results["day"]["closed_form_single_vs_volume_direct"] = {
        "share_of_error": l1(closed_single, ref["single"], total=total),
        "share_of_error_blurred": l1(closed_single, ref["single"], total=total, blurred=True),
        "bias": float((closed_single.sum() - ref["single"].sum()) / total.sum()),
        "own_rel_l1_blurred": l1(closed_single, ref["single"], blurred=True),
    }
    components = {}
    for name in [f"spp{s}" for s in DAY_SPP] + [f"guided{s}" for s in GUIDED_SPP]:
        components[name] = component_split(passes[name], ref)
        p = passes[name]
        hybrid = p["surfaces"] + p["multiple"] + closed_single
        components[name]["hybrid"] = {"rel_l1": l1(hybrid, total), "rel_l1_blurred": l1(hybrid, total, blurred=True)}
    # The reference's own passes, whose noise is in every residual above
    results["day"]["components"] = components
    results["day"]["reference_passes_light"] = {c: float(ref[c].sum() / total.sum()) for c in PASSES}
    print("components spp64", json.dumps(rounded(components["spp64"])), flush=True)

    components["default"] = component_split(passes["default"], ref)
    # Each pass's squared error times spp, over the squared mean of the whole image: constant when its noise falls
    # as 1/sqrt(spp) (it includes the reference's own noise in that pass, about 1/4 of it at 1024 spp)
    results["day"]["pass_spp_times_mse"] = {
        c: {str(s): s * mse(passes[f"spp{s}"][c], ref[c]) / mean_ref**2 for s in DAY_SPP} for c in PASSES
    }
    results["day"]["excess_light_from_top_0.1pct_pixels"] = {
        name: excess_from_top(passes[name]["total"], total) for name in ("spp4", "spp16", "default")
    }

    # 3. Where denoising leaves bias, by region, against the mean of both seeds
    masks = day_masks(scene)
    floor_map = np.abs(fast_blur(seed1) - fast_blur(total)).sum(-1) / 2  # noise left in the mean of the seeds
    regions = {"pixel_fraction": {k: float(m.mean()) for k, m in masks.items()}}
    for name in ("spp4", "spp16", "spp64", "spp256", "guided16"):
        regions[f"denoised_{name}"] = region_table(passes[name]["denoised"], truth, masks, truth_floor=floor_map)
    for name in ("spp16", "spp64"):
        regions[f"noisy_{name}"] = region_table(passes[name]["total"], total, masks)
    regions["noisy_spp16_top1pct_error_pixels_in"] = where_top_errors(passes["spp16"]["total"] - total, masks)
    # By brightness and by local contrast of the reference: deciles of pixels
    lum = luma(fast_blur(truth))
    grad = ndimage.morphological_gradient(np.log(lum + 1e-4), size=(1, 5, 5))
    for label, key in (("luminance", lum), ("contrast", grad)):
        edges = np.percentile(key, np.linspace(0, 100, 11))
        bins = [(key >= lo) & (key <= hi) for lo, hi in zip(edges[:-1], edges[1:])]
        for name in ("spp4", "spp16", "spp64"):
            table = region_table(passes[name]["denoised"], truth, {f"decile{k}": b for k, b in enumerate(bins)})
            regions[f"denoised_{name}_by_{label}"] = {
                k: {
                    "local_rel_l1_blurred": v["local_rel_l1_blurred"],
                    "local_bias": v["local_bias"],
                    "share_of_error_blurred": v["share_of_error_blurred"],
                }
                for k, v in table.items()
            }
    results["day"]["regions"] = regions
    for name in ("denoised_spp4", "denoised_spp16"):
        print(
            name,
            json.dumps(
                rounded(
                    {
                        k: (v["share_of_error_blurred"], v["local_rel_l1_blurred"], v["local_bias"])
                        for k, v in regions[name].items()
                    }
                )
            ),
            flush=True,
        )

    # Error maps of frame 345
    i = STILL.index(345)
    mean_frame = float(truth[i].mean())
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.4), constrained_layout=True)
    axes[0, 0].imshow(tonemap(truth[i]))
    axes[0, 0].set_title("reference (mean of 2 seeds), frame 345\ncyan: depth edges, yellow: shadow-affected (>20%)")
    outline(axes[0, 0], masks["depth_edges"][i], "cyan")
    outline(axes[0, 0], masks["shadow_affected_20pct"][i], "yellow")
    panels = [
        (axes[0, 1], passes["spp4"]["denoised"][i], "OIDN 4 spp: |blurred error| / mean(ref)", False),
        (axes[0, 2], passes["spp16"]["denoised"][i], "OIDN 16 spp: |blurred error| / mean(ref)", False),
        (axes[1, 0], passes["spp16"]["total"][i], "noisy 16 spp: |blurred error| / mean(ref)", False),
        (axes[1, 1], passes["spp4"]["denoised"][i], "OIDN 4 spp: signed blurred error / mean(ref)", True),
        (axes[1, 2], passes["spp16"]["denoised"][i], "OIDN 16 spp: signed blurred error / mean(ref)", True),
    ]
    for ax, image, title, signed in panels:
        m = error_map(image, truth[i], mean_frame, signed)
        if signed:
            shown = ax.imshow(m, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        else:
            shown = ax.imshow(m, cmap="magma", vmin=0, vmax=0.3)
        ax.set_title(f"{title}\nmean {np.abs(m).mean():.3f}")
        fig.colorbar(shown, ax=ax, shrink=0.8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Day: where OpenImageDenoise leaves bias (5x5 box blur removes most pixel noise)")
    figures["oidn_day"] = SAVE / "day_oidn_bias_345.png"
    fig.savefig(figures["oidn_day"], dpi=110)
    plt.close(fig)

    # 4. Why path guiding flickers: correlation of the noise between consecutive frames
    sequence_ref = day_passes(ROOT / "ms" / "cycles_ref", SEQUENCE)
    sequence_ref_luma = log_luma(sequence_ref["total"])
    flick = {}
    examples = {}
    for name in [f"spp{s}" for s in DAY_SPP] + [f"guided{s}" for s in GUIDED_SPP] + ["default"]:
        folder = renders[name]
        seq = day_passes(folder, SEQUENCE)
        denoised = load_images(folder / "frames", SEQUENCE)
        entry = {}
        for c in ("total",) + PASSES:
            residual = seq[c].astype(np.float64) - sequence_ref[c]
            entry[f"noise_correlation_{c}"] = float(
                np.mean([correlation(residual[k], residual[k + 1]) for k in range(len(SEQUENCE) - 1)])
            )
        log_error = log_luma(seq["total"]) - sequence_ref_luma
        rho = float(np.mean([correlation(log_error[k], log_error[k + 1]) for k in range(len(SEQUENCE) - 1)]))
        entry["log_luma_error_correlation"] = rho
        entry["flicker"] = flicker(seq["total"], sequence_ref["total"])
        entry["temporal_over_static"] = entry["flicker"]["temporal"] / entry["flicker"]["static"]
        entry["predicted_temporal_over_static"] = float(np.sqrt(2 * (1 - rho)))
        den_residual = denoised.astype(np.float64) - sequence_ref["total"]
        entry["denoised_error_correlation"] = float(
            np.mean([correlation(den_residual[k], den_residual[k + 1]) for k in range(len(SEQUENCE) - 1)])
        )
        entry["denoised_flicker"] = flicker(denoised, sequence_ref["total"])
        flick[name] = entry
        if name in ("spp64", "guided64"):
            examples[name] = (seq["total"][:2] - sequence_ref["total"][:2]) @ LUMA
        print(name, json.dumps(rounded({k: v for k, v in entry.items() if "correlation" in k})), flush=True)
    # Control: noise of two seeds on the same frame, which shares no sample
    seed_64 = load_images(OUT / "day_spp64_seed1" / "noisy", STILL)
    flick["control_seed0_vs_seed1_spp64_same_frame"] = correlation(passes["spp64"]["total"] - total, seed_64 - total)
    flick["control_seed0_vs_seed1_spp64_rel_l1"] = l1(seed_64, passes["spp64"]["total"])
    results["day"]["flicker"] = flick

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1, 1]})
    x = np.arange(len(GUIDED_SPP))
    for offset, (label, prefix, color) in enumerate((("plain", "spp", "#4c78a8"), ("guided", "guided", "#e45756"))):
        values = [flick[f"{prefix}{s}"]["noise_correlation_total"] for s in GUIDED_SPP]
        axes[0].bar(x + (offset - 0.5) * 0.38, values, 0.36, label=label, color=color)
        for xx, v in zip(x + (offset - 0.5) * 0.38, values):
            axes[0].text(xx, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].axhline(
        flick["control_seed0_vs_seed1_spp64_same_frame"], color="k", ls=":", lw=1, label="seed 0 vs 1, same frame"
    )
    axes[0].set_xticks(x, [f"{s} spp" for s in GUIDED_SPP])
    axes[0].set_ylabel("correlation of (render - reference)\nbetween consecutive frames")
    axes[0].set_ylim(0, 1.25)
    axes[0].legend(fontsize=8, ncol=3, loc="upper center")
    axes[0].set_title("Day, frames 255..262")
    limit = np.percentile(np.abs(examples["spp64"]), 99)
    for ax, name in zip(axes[1:], ("spp64", "guided64")):
        pair = np.concatenate([examples[name][0][40:140, 60:260], examples[name][1][40:140, 60:260]], axis=1)
        shown = ax.imshow(pair, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.axvline(199.5, color="k", lw=1)
        ax.set_title(
            f"{name}: noise (luma) of frames 255 | 256, crop\ncorrelation {flick[name]['noise_correlation_total']:.2f}"
        )
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(shown, ax=axes[2], shrink=0.8)
    figures["guiding"] = SAVE / "day_guiding_noise_correlation.png"
    fig.savefig(figures["guiding"], dpi=110)
    plt.close(fig)

    # Components figure: variance share of each pass, and the hybrid's best case
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    names = [f"spp{s}" for s in DAY_SPP]
    bottom = np.zeros(len(names))
    colors = {"surfaces": "#4c78a8", "single": "#f58518", "multiple": "#54a24b"}
    for c in PASSES:
        values = np.array([components[n]["passes"][c]["variance_share"] for n in names])
        axes[0].bar(
            names,
            values,
            bottom=bottom,
            color=colors[c],
            label=f"{c} (light {results['day']['reference_passes_light'][c]:.0%})",
        )
        bottom += values
    covariance = np.array(
        [sum(v for k, v in components[n]["covariance_share"].items() if not k.endswith("correlation")) for n in names]
    )
    axes[0].bar(names, covariance, bottom=np.where(covariance < 0, 0, bottom), color="#bbbbbb", label="covariances")
    axes[0].set_ylabel("share of squared error of the whole image")
    axes[0].set_title("Day noisy: which pass the noise comes from (frames 255, 345)")
    axes[0].legend(fontsize=8)
    width = 0.27
    x = np.arange(len(names))
    for k, (label, get) in enumerate(
        (
            ("Monte Carlo", lambda n: components[n]["rel_l1"]),
            ("exact single scattering", lambda n: components[n]["perfect_single_scattering"]["rel_l1"]),
            ("hybrid (closed form single)", lambda n: components[n]["hybrid"]["rel_l1"]),
        )
    ):
        axes[1].bar(x + (k - 1) * width, [get(n) for n in names], width, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("relative L1 error of the whole image")
    axes[1].set_title("Replacing the volume direct pass: best case and actual")
    axes[1].legend(fontsize=8)
    figures["components"] = SAVE / "day_noise_by_pass.png"
    fig.savefig(figures["components"], dpi=110)
    plt.close(fig)
    return figures


# --------------------------------------------------------------------------------------------------------------------
# Night scene


def night_masks(scene: Scene, whole_ref: np.ndarray, surfaces_ref: np.ndarray) -> dict[str, np.ndarray]:
    raw = load_images(ROOT / "clear" / "depths", scene.frames)[..., 0]
    sky = raw >= 1e9
    fog = luma(scene.reference)
    core = fog >= np.percentile(fog, 99, axis=(1, 2), keepdims=True)
    halo = (fog >= np.percentile(fog, 90, axis=(1, 2), keepdims=True)) & ~core
    # The cube hides the lamp from part of almost every ray (the closed form's fog light without occlusion is 2.4
    # times higher at the median pixel), so the wedge is where it hides the lamp from (nearly) the whole ray
    kept = []
    for i in range(len(scene.frames)):
        shadowed = luma(scene.closed_form(i).inscatter.numpy())
        free = luma(scene.closed_form(i, occlusion=None).inscatter.numpy())
        kept.append(shadowed / np.maximum(free, 1e-30))
    kept = np.stack(kept)
    wedge = kept < 0.1
    structure = np.ones((1, 7, 7), dtype=bool)
    band = ndimage.binary_dilation(wedge, structure) & ~ndimage.binary_erosion(wedge, structure)
    log_depth = np.log(np.maximum(scene.depth, 1e-3))
    jump = ndimage.maximum_filter(log_depth, size=(1, 3, 3)) - ndimage.minimum_filter(log_depth, size=(1, 3, 3))
    edges = ndimage.binary_dilation(jump > 0.1, structure=np.ones((1, 5, 5), dtype=bool))
    lit = luma(surfaces_ref) > 0.05 * np.percentile(luma(surfaces_ref), 99, axis=(1, 2), keepdims=True)
    whole = luma(whole_ref)
    dark = whole < np.percentile(whole, 20, axis=(1, 2), keepdims=True)
    return {
        "all": np.ones_like(sky),
        "sky": sky,
        "halo_core_top1pct": core,
        "halo_ring_top1_10pct": halo,
        "shadow_wedge": wedge,
        "partly_shadowed": (kept >= 0.1) & (kept < 0.5),
        "shadow_wedge_boundary": band,
        "lit_surfaces": lit & ~sky,
        "depth_edges": edges,
        "dark_20pct": dark,
    }


def analyze_night(results: dict) -> dict:
    figures = {}
    scene = night_scene(STILL)
    fog_ref = scene.reference.astype(np.float64)
    ref_folder = ROOT / "point_shadow" / "bounces0"
    whole_ref = load_images(ref_folder / "frames", STILL)
    ref = {
        "total": whole_ref,
        "single": load_images(ref_folder / "volume" / "direct", STILL),
        "multiple": load_images(ref_folder / "volume" / "indirect", STILL),
    }
    ref["surfaces"] = ref["total"] - ref["single"] - ref["multiple"]
    mean_fog = float(fog_ref.mean())

    # Reference noise: from a separate 2048 spp render's volume direct pass (16 bounces), and by extrapolating 1024 spp
    other = load_images(OUT / "night_ms_ref" / "volume" / "direct", STILL)
    spp1024 = load_images(OUT / "night_spp1024" / "volume" / "direct", STILL)
    floor = {
        "ms_ref_2048_vs_ref": errors(other, fog_ref),
        # var(2048 - 4096) = 3 var(4096) when independent; var(1024 - 4096) = 5 var(4096)
        "one_reference_mse_from_2048": mse(other, fog_ref) / 3,
        "one_reference_mse_from_1024": mse(spp1024, fog_ref) / 5,
        "one_reference_rel_l1_from_1024": l1(spp1024, fog_ref) / np.sqrt(5),
        "one_reference_rel_l1_blurred_from_1024": l1(spp1024, fog_ref, blurred=True) / np.sqrt(5),
    }
    results["night"]["reference_floor"] = floor
    var_ref = floor["one_reference_mse_from_1024"]

    budget, renders = {}, {}
    names = [f"spp{s}" for s in NIGHT_SPP] + [f"guided{s}" for s in (64, 256)]
    for name in names:
        folder = OUT / f"night_{name}"
        noisy = load_images(folder / "noisy", STILL)
        direct = load_images(folder / "volume" / "direct", STILL)
        indirect = load_images(folder / "volume" / "indirect", STILL)
        render = {"total": noisy, "single": direct, "multiple": indirect, "surfaces": noisy - direct - indirect}
        render["denoised"] = load_images(folder / "frames", STILL)
        renders[name] = render
        noise_mse = mse(direct, fog_ref) - var_ref
        budget[name] = {
            "fog_light": errors(direct, fog_ref),
            "fog_light_blurred_over_raw": l1(direct, fog_ref, blurred=True) / l1(direct, fog_ref),
            "fog_light_noise_rel_rmse": float(np.sqrt(max(noise_mse, 0)) / mean_fog),
            "fog_light_top1pct_pixels_share": tail_share(direct - fog_ref, 0.01),
            "whole_noisy": errors(noisy, whole_ref),
            "whole_denoised": errors(render["denoised"], whole_ref),
            "components_whole_noisy": component_split(render, ref),
        }
        if name.startswith("spp"):
            budget[name]["spp_times_noise_mse_over_mean2"] = int(name[3:]) * noise_mse / mean_fog**2
        print("night", name, json.dumps(rounded(budget[name]["fog_light"])), flush=True)
    plain = [budget[f"spp{s}"] for s in NIGHT_SPP]
    one = floor["one_reference_rel_l1_from_1024"]
    results["night"]["budget"] = budget
    results["night"]["scaling"] = {
        "slope_rel_l1": slope(NIGHT_SPP, [b["fog_light"]["rel_l1"] for b in plain]),
        "slope_rel_l1_minus_reference_noise": slope(
            NIGHT_SPP, [np.sqrt(max(b["fog_light"]["rel_l1"] ** 2 - one**2, 1e-12)) for b in plain]
        ),
        "slope_noise_rmse_16_to_256": slope(NIGHT_SPP[:3], [b["fog_light_noise_rel_rmse"] for b in plain[:3]]),
    }

    # Regions: noisy fog light, and denoised whole images
    masks = night_masks(scene, whole_ref, ref["surfaces"])
    regions = {"pixel_fraction": {k: float(m.mean()) for k, m in masks.items()}}
    for name in ("spp16", "spp64", "spp256"):
        regions[f"fog_light_noisy_{name}"] = region_table(renders[name]["single"], fog_ref, masks)
        regions[f"fog_light_noisy_{name}_top1pct_error_pixels_in"] = where_top_errors(
            renders[name]["single"] - fog_ref, masks
        )
    for name in ("spp16", "spp64", "spp256", "guided64", "guided256"):
        regions[f"whole_denoised_{name}"] = region_table(renders[name]["denoised"], whole_ref, masks)
    for name in ("guided256", "spp256"):
        regions[f"whole_noisy_{name}"] = region_table(renders[name]["total"], whole_ref, masks)
    results["night"]["regions"] = regions
    results["night"]["excess_light_from_top_0.1pct_pixels"] = {
        f"whole_noisy_{name}": excess_from_top(renders[name]["total"], whole_ref)
        for name in ("spp256", "guided64", "guided256")
    }
    for name in ("fog_light_noisy_spp16", "whole_denoised_spp16", "whole_noisy_guided256"):
        print(
            name,
            json.dumps(
                rounded(
                    {
                        k: (v["share_of_error_blurred"], v["local_rel_l1_blurred"], v["local_bias"])
                        for k, v in regions[name].items()
                    }
                )
            ),
            flush=True,
        )

    # Maps of frame 345
    i = STILL.index(345)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.4), constrained_layout=True)
    axes[0, 0].imshow(tonemap(fog_ref[i]))
    axes[0, 0].set_title("reference fog light, frame 345\nyellow: shadow wedge, cyan: halo core (top 1%)")
    outline(axes[0, 0], masks["shadow_wedge"][i], "yellow")
    outline(axes[0, 0], masks["halo_core_top1pct"][i], "cyan")
    axes[1, 0].imshow(tonemap(whole_ref[i]))
    axes[1, 0].set_title("reference whole image, frame 345")
    mean_whole = float(whole_ref[i].mean())
    panels = [
        (
            axes[0, 1],
            renders["spp16"]["single"][i],
            fog_ref[i],
            mean_fog,
            "noisy 16 spp fog light: |blurred error|",
            False,
        ),
        (
            axes[0, 2],
            renders["spp64"]["single"][i],
            fog_ref[i],
            mean_fog,
            "noisy 64 spp fog light: |blurred error|",
            False,
        ),
        (
            axes[1, 1],
            renders["spp16"]["denoised"][i],
            whole_ref[i],
            mean_whole,
            "OIDN 16 spp whole: |blurred error|",
            False,
        ),
        (
            axes[1, 2],
            renders["guided256"]["total"][i],
            whole_ref[i],
            mean_whole,
            "guided 256 spp noisy whole: signed blurred",
            True,
        ),
    ]
    for ax, image, reference, mean_value, title, signed in panels:
        m = error_map(image, reference, mean_value, signed)
        shown = ax.imshow(m, cmap="RdBu_r" if signed else "magma", vmin=-0.5 if signed else 0, vmax=0.5)
        ax.set_title(f"{title} / mean(ref)\nmean {np.abs(m).mean():.3f}")
        fig.colorbar(shown, ax=ax, shrink=0.8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Night: where Monte Carlo errs (5x5 box blur)")
    figures["night"] = SAVE / "night_error_maps_345.png"
    fig.savefig(figures["night"], dpi=110)
    plt.close(fig)
    return figures


# --------------------------------------------------------------------------------------------------------------------


def convergence_figure(results: dict) -> Path:
    day, night = results["day"], results["night"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    ax = axes[0]
    b = day["budget"]
    ax.loglog(DAY_SPP, [b[f"spp{s}"]["noisy"]["rel_l1"] for s in DAY_SPP], "o-", c="#4c78a8", label="noisy")
    ax.loglog(
        DAY_SPP, [b[f"spp{s}"]["noisy"]["rel_l1_blurred"] for s in DAY_SPP], "o--", c="#4c78a8", label="noisy, blurred"
    )
    ax.loglog(
        GUIDED_SPP, [b[f"guided{s}"]["noisy"]["rel_l1"] for s in GUIDED_SPP], "s-", c="#e45756", label="guided noisy"
    )
    ax.loglog(DAY_SPP, [b[f"spp{s}"]["denoised"]["rel_l1"] for s in DAY_SPP], "o-", c="#54a24b", label="OIDN")
    ax.loglog(
        DAY_SPP,
        [b[f"spp{s}"]["denoised_against_mean_of_seeds"]["rel_l1_blurred"] for s in DAY_SPP],
        "o--",
        c="#54a24b",
        label="OIDN, blurred, vs mean of seeds",
    )
    ax.loglog(
        DAY_SPP,
        [b[f"spp{s}"]["noisy_noise_rel_rmse"] for s in DAY_SPP],
        "^:",
        c="#72b7b2",
        label="noise RMSE (ref noise removed)",
    )
    guide = b["spp64"]["noisy_noise_rel_rmse"] * np.sqrt(64 / np.array(DAY_SPP))
    ax.loglog(DAY_SPP, guide, c="#999", lw=0.8, label="1/sqrt(spp)")
    floor = day["reference_floor"]
    ax.axhline(floor["one_reference_rel_l1"], c="k", ls=":", lw=1, label="reference noise, as is")
    ax.axhline(floor["one_reference_rel_l1_blurred"], c="k", ls="-.", lw=1, label="reference noise, blurred")
    ax.set_xlabel("samples per pixel")
    ax.set_ylabel("relative error")
    ax.set_title("Day, whole image, frames 255 and 345")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    b = night["budget"]
    ax.loglog(
        NIGHT_SPP, [b[f"spp{s}"]["fog_light"]["rel_l1"] for s in NIGHT_SPP], "o-", c="#4c78a8", label="fog light, noisy"
    )
    ax.loglog(
        NIGHT_SPP,
        [b[f"spp{s}"]["fog_light"]["rel_l1_blurred"] for s in NIGHT_SPP],
        "o--",
        c="#4c78a8",
        label="fog light, blurred",
    )
    ax.loglog(
        (64, 256),
        [b[f"guided{s}"]["fog_light"]["rel_l1"] for s in (64, 256)],
        "s-",
        c="#e45756",
        label="guided fog light",
    )
    ax.loglog(
        NIGHT_SPP, [b[f"spp{s}"]["whole_denoised"]["rel_l1"] for s in NIGHT_SPP], "o-", c="#54a24b", label="OIDN whole"
    )
    ax.loglog(
        NIGHT_SPP,
        [b[f"spp{s}"]["whole_denoised"]["rel_l1_blurred"] for s in NIGHT_SPP],
        "o--",
        c="#54a24b",
        label="OIDN whole, blurred",
    )
    ax.loglog(
        (64, 256),
        [b[f"guided{s}"]["whole_denoised"]["rel_l1_blurred"] for s in (64, 256)],
        "s--",
        c="#b279a2",
        label="guided OIDN whole, blurred",
    )
    guide = b["spp64"]["fog_light"]["rel_l1"] * np.sqrt(64 / np.array(NIGHT_SPP))
    ax.loglog(NIGHT_SPP, guide, c="#999", lw=0.8, label="1/sqrt(spp)")
    ax.axhline(
        night["reference_floor"]["one_reference_rel_l1_from_1024"], c="k", ls=":", lw=1, label="reference noise (est.)"
    )
    ax.set_xlabel("samples per pixel")
    ax.set_title("Night, frames 255 and 345")
    ax.legend(fontsize=7, ncol=2)
    path = SAVE / "convergence.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    check = np.random.default_rng(0).random((1, 20, 30, 3)).astype(np.float32)
    assert np.allclose(fast_blur(check), blur(check), atol=1e-6)
    results: dict = {"day": {}, "night": {}}
    figures = analyze_day(results)
    figures |= analyze_night(results)
    figures["convergence"] = convergence_figure(results)
    results["figures"] = {k: str(v) for k, v in figures.items()}
    (SAVE / "monte_carlo.json").write_text(json.dumps(rounded(results), indent=1))
    print("saved", SAVE / "monte_carlo.json", flush=True)


if __name__ == "__main__":
    main()
