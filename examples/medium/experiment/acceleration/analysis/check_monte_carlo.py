"""Independent check of ``analysis/monte_carlo.py``: re-derives its key numbers with separate code, and tests
alternative explanations of its findings.

- noise or bias: errors of noisy and denoised renders with two seeds at 64 spp, where the half difference of the two
  seeds has the same noise as their mean but no bias, so that the blurred error of the mean only exceeds that of the
  half difference by the bias;
- the anti-correlation of the passes' noise, from the difference of two seeds, which holds no reference noise;
- the hybrid's best case, and the significance of small brightness biases (4 spp, adaptive sampling) against the
  spread of the error summed over blocks of pixels;
- the noise of consecutive frames, for plain and guided renders;
- night: noise scaling, where the largest errors are, OpenImageDenoise's bias, and whether path guiding's extra light
  may come from clamping (Cycles clamps indirect samples by default), from a clamped and an unclamped render of the
  point lamp scene.

Saves ``analysis/check_monte_carlo/check_monte_carlo.json`` and a figure. CPU only.
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

SAVE = OUT / "analysis" / "check_monte_carlo"
STILL = (255, 345)
SEQ = tuple(range(255, 263))
DAY_SPP = (4, 16, 64, 256, 1024)
NIGHT_SPP = (16, 64, 256, 1024)
PASSES = ("surfaces", "single", "multiple")


def b5(x: np.ndarray) -> np.ndarray:
    """5x5 box blur with zero padding over (n, h, w, c), as common.blur."""
    return ndimage.uniform_filter(np.asarray(x, dtype=np.float64), size=(1, 5, 5, 1), mode="constant")


def f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def rl1(ours, ref, blurred=False, keep=None, total=None) -> float:
    """Sum |ours - ref| (blurred or not) over a region, over the sum of ``total`` (default ref) over the region."""
    ours, ref = f64(ours), f64(ref)
    diff = np.abs(b5(ours) - b5(ref)) if blurred else np.abs(ours - ref)
    total = ref if total is None else f64(total)
    if keep is not None:
        return float(diff[keep].sum() / total[keep].sum())
    return float(diff.sum() / total.sum())


def signed(ours, ref, keep=None) -> float:
    ours, ref = f64(ours), f64(ref)
    if keep is not None:
        ours, ref = ours[keep], ref[keep]
    return float((ours.sum() - ref.sum()) / ref.sum())


def mse(a, b) -> float:
    return float(np.mean((f64(a) - f64(b)) ** 2))


def corr(a, b) -> float:
    a, b = f64(a).ravel(), f64(b).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def slope(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def luma(x) -> np.ndarray:
    return f64(x) @ np.array([0.2126, 0.7152, 0.0722])


def block_test(ours, ref, size=16) -> dict:
    """Signed error summed over blocks of size x size pixels: fraction of blocks that are too bright, and the total
    signed error over its standard deviation estimated from the spread of the blocks (z), which holds for heavy tails
    as long as blocks are independent."""
    residual = (f64(ours) - f64(ref)).sum(-1)
    n, h, w = residual.shape
    hh, ww = h // size * size, w // size * size
    blocks = residual[:, :hh, :ww].reshape(n, hh // size, size, ww // size, size).sum(axis=(2, 4)).ravel()
    total = blocks.sum()
    return {
        "fraction_blocks_brighter": float((blocks > 0).mean()),
        "z": float(total / (blocks.std() * np.sqrt(len(blocks)))),
        "relative_bias_cropped": float(total / f64(ref)[:, :hh, :ww].sum()),
    }


def trimmed_excess(ours, ref, fraction=0.001) -> dict:
    """Excess light left once the given fraction of pixels with the largest and smallest error are dropped."""
    residual = (f64(ours) - f64(ref)).sum(-1).ravel()
    order = np.sort(residual)
    k = max(1, int(len(order) * fraction))
    total = f64(ref).sum()
    return {
        "excess": float(order.sum() / total),
        "without_top": float(order[:-k].sum() / total),
        "without_top_and_bottom": float(order[k:-k].sum() / total),
        "median_pixel_residual_over_mean": float(np.median(residual) / (f64(ref).sum(-1).mean())),
    }


def load_day(name: str, frames=STILL) -> dict:
    folder = OUT / name
    combined = load_images(folder / "noisy", frames)
    direct = load_images(folder / "volume" / "direct", frames)
    indirect = load_images(folder / "volume" / "indirect", frames)
    return {
        "total": combined,
        "single": direct,
        "multiple": indirect,
        "surfaces": combined - direct - indirect,
        "denoised": load_images(folder / "frames", frames),
    }


def load_day_reference(frames=STILL) -> dict:
    folder = ROOT / "ms" / "cycles_ref"
    combined = load_images(folder / "frames", frames)
    direct = load_images(folder / "volume" / "direct", frames)
    indirect = load_images(folder / "volume" / "indirect", frames)
    return {"total": combined, "single": direct, "multiple": indirect, "surfaces": combined - direct - indirect}


def depth_edges(depth: np.ndarray, threshold=0.1, dilation=5) -> np.ndarray:
    log_depth = np.log(np.maximum(depth, 1e-3))
    jump = ndimage.maximum_filter(log_depth, size=(1, 3, 3)) - ndimage.minimum_filter(log_depth, size=(1, 3, 3))
    return ndimage.binary_dilation(jump > threshold, structure=np.ones((1, dilation, dilation), dtype=bool))


# --------------------------------------------------------------------------------------------------------------------


def check_day(out: dict) -> None:
    ref = load_day_reference()
    seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", STILL)
    total = ref["total"]
    truth = (f64(total) + seed1) / 2
    mean_ref = float(f64(total).mean())
    var_ref = mse(seed1, total) / 2
    out["floor"] = {
        "seed_pair_rel_l1": rl1(seed1, total),
        "seed_pair_rel_l1_blurred": rl1(seed1, total, True),
        "one_reference_rel_l1": rl1(seed1, total) / np.sqrt(2),
        "var_ref_over_mean2": var_ref / mean_ref**2,
    }
    renders = {f"spp{s}": load_day(f"day_spp{s}") for s in DAY_SPP}
    renders |= {f"guided{s}": load_day(f"day_guided{s}") for s in (16, 64, 256)}
    renders["default"] = load_day("day_default")
    renders["spp64_seed1"] = load_day("day_spp64_seed1")

    # 1-2. Noise against bias, and how the noise falls
    budget = {}
    for name, r in renders.items():
        noise = mse(r["total"], total) - var_ref
        budget[name] = {
            "ratio": float(f64(r["total"]).sum() / f64(total).sum()),
            "rel_l1": rl1(r["total"], total),
            "rel_l1_blurred": rl1(r["total"], total, True),
            "rel_l1_vs_seed1": rl1(r["total"], seed1),
            "blurred_over_raw": rl1(r["total"], total, True) / rl1(r["total"], total),
            "noise_rel_rmse": float(np.sqrt(max(noise, 0)) / mean_ref),
            "denoised_vs_mean_of_seeds": rl1(r["denoised"], truth),
            "denoised_vs_mean_of_seeds_blurred": rl1(r["denoised"], truth, True),
            "denoised_signed_vs_mean_of_seeds": signed(r["denoised"], truth),
        }
        if name.startswith("spp") and "_" not in name:
            budget[name]["spp_times_noise_mse_over_mean2"] = int(name[3:]) * noise / mean_ref**2
        print(name, json.dumps({k: round(v, 5) for k, v in budget[name].items()}), flush=True)
    out["budget"] = budget
    rmse = [budget[f"spp{s}"]["noise_rel_rmse"] for s in DAY_SPP]
    out["slopes"] = {
        "rel_l1": slope(DAY_SPP, [budget[f"spp{s}"]["rel_l1"] for s in DAY_SPP]),
        "noise_rmse": slope(DAY_SPP, rmse),
        "noise_rmse_64_1024": slope(DAY_SPP[2:], rmse[2:]),
        "denoised_blurred_vs_mean_of_seeds": slope(
            DAY_SPP, [budget[f"spp{s}"]["denoised_vs_mean_of_seeds_blurred"] for s in DAY_SPP]
        ),
    }

    # Two seeds at 64 spp: the mean of the seeds against the mean of the references holds bias + noise, the half
    # difference of the seeds (plus that of the references) the same noise and no bias
    s0, s1 = renders["spp64"], renders["spp64_seed1"]
    half_ref = (f64(total) - seed1) / 2
    seeds = {}
    for kind in ("total", "denoised"):
        mean_error = (f64(s0[kind]) + s1[kind]) / 2 - truth
        half_diff = (f64(s0[kind]) - s1[kind]) / 2 + half_ref
        e0, e1 = b5(f64(s0[kind]) - truth), b5(f64(s1[kind]) - truth)
        seeds[kind] = {
            "seed0_blurred": float(np.abs(e0).sum() / truth.sum()),
            "seed1_blurred": float(np.abs(e1).sum() / truth.sum()),
            "mean_of_seeds_blurred": float(np.abs(b5(mean_error)).sum() / truth.sum()),
            "half_difference_blurred (noise only)": float(np.abs(b5(half_diff)).sum() / truth.sum()),
            "correlation_of_blurred_error_maps": corr(e0, e1),
            "seed1_ratio": float(f64(s1[kind]).sum() / truth.sum()),
            "seed0_ratio": float(f64(s0[kind]).sum() / truth.sum()),
        }
    out["two_seeds_64spp"] = seeds
    print("two seeds", json.dumps(seeds), flush=True)

    # 3. Passes: variance shares from residuals against the reference, and from seed differences (no reference noise)
    def shares(res: dict) -> dict:
        whole = sum(res.values())
        sq = (whole**2).sum()
        result = {c: float((res[c] ** 2).sum() / sq) for c in PASSES}
        for a, b in (("surfaces", "single"), ("surfaces", "multiple"), ("single", "multiple")):
            result[f"cov_{a}_{b}"] = float(2 * (res[a] * res[b]).sum() / sq)
            result[f"corr_{a}_{b}"] = corr(res[a], res[b])
        return result

    passes = {}
    for name in [f"spp{s}" for s in DAY_SPP]:
        passes[name] = shares({c: f64(renders[name][c]) - ref[c] for c in PASSES})
    passes["seed0_minus_seed1_64spp"] = shares({c: f64(s0[c]) - s1[c] for c in PASSES})
    passes["light"] = {c: float(f64(ref[c]).sum() / f64(total).sum()) for c in PASSES}
    passes["own_rel_l1_64"] = {c: rl1(s0[c], ref[c]) for c in PASSES}
    passes["bias_64"] = {c: float((f64(s0[c]).sum() - f64(ref[c]).sum()) / f64(total).sum()) for c in PASSES}
    out["passes"] = passes
    print("passes", json.dumps(passes["spp64"]), json.dumps(passes["seed0_minus_seed1_64spp"]), flush=True)

    # 4. Hybrid: exact single scattering (reference pass), the closed form's, and an independent seed's
    scene = day_scene(STILL)
    single_medium = scene.medium.model_copy(update={"multiple_scattering": False})
    no_ground = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    closed_single = np.stack(
        [
            scene.closed_form(i, normals=None, medium=single_medium, lighting=no_ground).inscatter.numpy()
            for i in range(len(STILL))
        ]
    )
    hybrid = {
        "closed_form_single_vs_volume_direct_share": rl1(closed_single, ref["single"], total=total),
        "closed_form_single_vs_volume_direct_share_blurred": rl1(closed_single, ref["single"], True, total=total),
        "closed_form_single_bias": float((closed_single.sum() - f64(ref["single"]).sum()) / f64(total).sum()),
    }
    for s in DAY_SPP:
        r = renders[f"spp{s}"]
        rest = f64(r["surfaces"]) + r["multiple"]
        hybrid[f"spp{s}"] = {
            "plain": rl1(r["total"], total),
            "oracle": rl1(rest + ref["single"], total),
            "closed_form": rl1(rest + closed_single, total),
        }
    # Swap in the single scattering of the other seed: loses the cancellation and adds independent noise
    hybrid["spp64_single_from_seed1"] = rl1(f64(s0["surfaces"]) + s0["multiple"] + s1["single"], total)
    out["hybrid"] = hybrid
    print("hybrid", json.dumps(hybrid), flush=True)

    # 5-6. Where OIDN errs, by region, with my own masks
    raw = load_images(ROOT / "clear" / "depths", STILL)[..., 0]
    sky = raw >= 1e9
    normals = scene.extra["normals"]
    ground = ~sky & (normals[..., 2] > 0.95)
    edges = depth_edges(scene.depth)
    edges_tight = depth_edges(scene.depth, 0.05, 3)
    log_clear = np.log(luma(scene.clear) + 1e-4)
    contrast = ndimage.maximum_filter(log_clear, size=(1, 3, 3)) - ndimage.minimum_filter(log_clear, size=(1, 3, 3))
    texture = ndimage.binary_dilation(contrast > 0.5, structure=np.ones((1, 3, 3), dtype=bool)) & ~edges & ~sky
    lum = luma(total)
    dark = lum < np.percentile(lum, 20, axis=(1, 2), keepdims=True)
    change = []
    for i in range(len(STILL)):
        with_shadows = scene.closed_form(i).radiance.numpy()
        without = scene.closed_form(i, occlusion=None).radiance.numpy()
        change.append(np.abs(with_shadows - without).sum(-1) / np.maximum(f64(total[i]).sum(-1), 1e-12))
    change = np.stack(change)
    masks = {
        "all": np.ones_like(sky),
        "sky": sky,
        "ground": ground,
        "objects": ~sky & ~ground,
        "depth_edges": edges,
        "depth_edges_tight": edges_tight,
        "texture_edges": texture,
        "ground_texture": texture & ground,
        "ground_flat": ground & ~texture & ~edges,
        "dark_20pct": dark,
        "shadow_20pct": change > 0.2,
        "shadow_2pct": change > 0.02,
    }
    regions = {"pixels": {k: float(m.mean()) for k, m in masks.items()}}
    for name in ("spp4", "spp16", "spp64", "spp64_seed1"):
        d = renders[name]["denoised"]
        regions[name] = {
            k: {"local_blurred": rl1(d, truth, True, keep=m), "bias": signed(d, truth, keep=m)} for k, m in masks.items()
        }
    mean64 = (f64(s0["denoised"]) + s1["denoised"]) / 2
    regions["mean_of_seeds_64"] = {
        k: {"local_blurred": rl1(mean64, truth, True, keep=m), "bias": signed(mean64, truth, keep=m)}
        for k, m in masks.items()
    }
    # Contrast deciles of the blurred reference at 16 spp
    grad = ndimage.morphological_gradient(np.log(luma(b5(truth)) + 1e-4), size=(1, 5, 5))
    cuts = np.percentile(grad, np.linspace(0, 100, 11))
    regions["spp16_contrast_deciles"] = [
        {
            "local_blurred": rl1(renders["spp16"]["denoised"], truth, True, keep=(grad >= lo) & (grad <= hi)),
            "bias": signed(renders["spp16"]["denoised"], truth, keep=(grad >= lo) & (grad <= hi)),
        }
        for lo, hi in zip(cuts[:-1], cuts[1:])
    ]
    out["regions"] = regions
    for name in ("spp4", "spp16", "spp64", "spp64_seed1", "mean_of_seeds_64"):
        print(
            name, json.dumps({k: (round(v["local_blurred"], 4), round(v["bias"], 4)) for k, v in regions[name].items()})
        )

    # Excess light at low spp and with adaptive sampling: is it outliers, or many pixels?
    small = {}
    for name in ("spp4", "spp16", "spp64", "spp256", "default", "guided64", "spp64_seed1"):
        r = renders[name]
        small[name] = {"trim": trimmed_excess(r["total"], total), "blocks": block_test(r["total"], total)}
        for c in PASSES:
            small[name][f"bias_{c}"] = float((f64(r[c]).sum() - f64(ref[c]).sum()) / f64(total).sum())
        small[name]["denoised_ratio_vs_mean_of_seeds"] = float(f64(r["denoised"]).sum() / truth.sum())
    out["small_biases"] = small
    print("small biases", json.dumps(small), flush=True)

    # 7. Consecutive frames: correlation of the noise, and flicker
    seq_ref = load_day_reference(SEQ)
    flick = {}
    for name in ("spp4", "spp64", "spp256", "spp1024", "guided16", "guided64", "guided256", "default"):
        seq = load_day(
            name.replace("spp", "day_spp").replace("guided", "day_guided").replace("default", "day_default"), SEQ
        )
        entry = {}
        for c in ("total",) + PASSES:
            res = f64(seq[c]) - seq_ref[c]
            entry[c] = float(np.mean([corr(res[k], res[k + 1]) for k in range(len(SEQ) - 1)]))
        entry["flicker"] = flicker(seq["total"], seq_ref["total"])
        entry["ratio_per_frame"] = [
            float(f64(seq["total"][k]).sum() / f64(seq_ref["total"][k]).sum()) for k in range(len(SEQ))
        ]
        den = f64(seq["denoised"]) - seq_ref["total"]
        entry["denoised"] = float(np.mean([corr(den[k], den[k + 1]) for k in range(len(SEQ) - 1)]))
        entry["denoised_flicker"] = flicker(seq["denoised"], seq_ref["total"])
        if name == "spp4":
            # Do the brightest outliers stay on the same pixels from frame to frame (frozen seed)?
            per_pixel = (f64(seq["total"]) - seq_ref["total"]).sum(-1).reshape(len(SEQ), -1)
            top = [set(np.argsort(p)[-58:]) for p in per_pixel]  # top 0.1% of 57600 pixels
            entry["top_outliers_shared_with_next_frame"] = float(
                np.mean([len(top[k] & top[k + 1]) / 58 for k in range(len(SEQ) - 1)])
            )
        flick[name] = entry
        print("flicker", name, json.dumps(entry), flush=True)
    # Reference's own frame-to-frame correlation is unknown, but the seed-0 reference vs the seed-1 64 spp render's
    # noise on the same frame shares no sample
    flick["control_same_frame_two_seeds"] = corr(f64(s0["total"]) - total, f64(s1["total"]) - total)
    # Camera motion between consecutive frames
    poses = [np.asarray(t["transform_matrix"]) for t in load_transforms(ROOT / "clear" / "frames", SEQ)]
    flick["camera_step_m"] = float(
        np.mean([np.linalg.norm(poses[k + 1][:3, 3] - poses[k][:3, 3]) for k in range(len(SEQ) - 1)])
    )
    out["flicker"] = flick

    # Figure: blurred signed error of denoised 64 spp, seed 0 vs seed 1 vs their mean, frame 345
    i = STILL.index(345)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), constrained_layout=True)
    mean_frame = float(truth[i].mean())
    for ax, image, title in (
        (axes[0], s0["denoised"][i], "OIDN 64 spp, seed 0"),
        (axes[1], s1["denoised"][i], "OIDN 64 spp, seed 1"),
        (axes[2], mean64[i], "mean of both seeds"),
    ):
        m = (b5(f64(image)[None]) - b5(truth[i][None]))[0].mean(-1) / mean_frame
        shown = ax.imshow(m, cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        ax.set_title(f"{title}: signed blurred error / mean(ref)\nmean |.| {np.abs(m).mean():.4f}")
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(shown, ax=axes, shrink=0.8)
    fig.savefig(SAVE / "day_oidn_two_seeds_345.png", dpi=110)
    plt.close(fig)


def check_night(out: dict) -> None:
    scene = night_scene(STILL)
    ref_folder = ROOT / "point_shadow" / "bounces0"
    fog_ref = f64(load_images(ref_folder / "volume" / "direct", STILL))
    whole_ref = f64(load_images(ref_folder / "frames", STILL))
    ref = {
        "total": whole_ref,
        "single": fog_ref,
        "multiple": f64(load_images(ref_folder / "volume" / "indirect", STILL)),
    }
    ref["surfaces"] = ref["total"] - ref["single"] - ref["multiple"]
    mean_fog = float(fog_ref.mean())
    renders = {}
    for name in [f"spp{s}" for s in NIGHT_SPP] + ["guided64", "guided256"]:
        folder = OUT / f"night_{name}"
        r = {
            "total": f64(load_images(folder / "noisy", STILL)),
            "single": f64(load_images(folder / "volume" / "direct", STILL)),
            "multiple": f64(load_images(folder / "volume" / "indirect", STILL)),
            "denoised": f64(load_images(folder / "frames", STILL)),
        }
        r["surfaces"] = r["total"] - r["single"] - r["multiple"]
        renders[name] = r
    other = load_images(OUT / "night_ms_ref" / "volume" / "direct", STILL)
    var_2048 = mse(other, fog_ref) / 3
    var_1024 = mse(renders["spp1024"]["single"], fog_ref) / 5
    out["floor"] = {"var_ref_from_2048": var_2048, "var_ref_from_1024": var_1024}
    budget = {}
    for name, r in renders.items():
        noise = mse(r["single"], fog_ref) - var_1024
        budget[name] = {
            "fog_ratio": float(r["single"].sum() / fog_ref.sum()),
            "fog_rel_l1": rl1(r["single"], fog_ref),
            "fog_rel_l1_blurred": rl1(r["single"], fog_ref, True),
            "fog_rmse": float(np.sqrt(max(noise, 0)) / mean_fog),
            "whole_ratio": float(r["total"].sum() / whole_ref.sum()),
            "whole_denoised_ratio": float(r["denoised"].sum() / whole_ref.sum()),
            "whole_denoised_blurred": rl1(r["denoised"], whole_ref, True),
            "multiple_pass_sum": float(r["multiple"].sum()),
        }
        if name.startswith("spp"):
            budget[name]["spp_times_noise_mse_over_mean2"] = int(name[3:]) * noise / mean_fog**2
            budget[name]["spp_times_noise_mse_over_mean2_var2048"] = (
                int(name[3:]) * (mse(r["single"], fog_ref) - var_2048) / mean_fog**2
            )
        # Variance shares of the whole noisy image
        res = {c: r[c] - ref[c] for c in PASSES}
        whole = sum(res.values())
        budget[name]["surfaces_share_of_squared_error"] = float((res["surfaces"] ** 2).sum() / (whole**2).sum())
        budget[name]["surfaces_light"] = float(ref["surfaces"].sum() / whole_ref.sum())
        budget[name]["bias_surfaces"] = float((r["surfaces"].sum() - ref["surfaces"].sum()) / whole_ref.sum())
        budget[name]["bias_single"] = float((r["single"].sum() - ref["single"].sum()) / whole_ref.sum())
        print("night", name, json.dumps(budget[name]), flush=True)
    out["budget"] = budget
    rmse = [budget[f"spp{s}"]["fog_rmse"] for s in NIGHT_SPP]
    out["slopes"] = {
        "fog_rmse_16_256": slope(NIGHT_SPP[:3], rmse[:3]),
        "fog_rmse_256_1024": slope(NIGHT_SPP[2:], rmse[2:]),
    }
    # Same seed at different budgets: do renders share samples (prefix of the same sequence)?
    out["residual_correlation_spp64_spp256"] = corr(
        renders["spp64"]["single"] - fog_ref, renders["spp256"]["single"] - fog_ref
    )
    out["residual_correlation_spp256_spp1024"] = corr(
        renders["spp256"]["single"] - fog_ref, renders["spp1024"]["single"] - fog_ref
    )

    # Masks
    raw = load_images(ROOT / "clear" / "depths", STILL)[..., 0]
    sky = raw >= 1e9
    fog_l = luma(fog_ref)
    core = fog_l >= np.percentile(fog_l, 99, axis=(1, 2), keepdims=True)
    ring = (fog_l >= np.percentile(fog_l, 90, axis=(1, 2), keepdims=True)) & ~core
    kept = []
    for i in range(len(STILL)):
        shadowed = luma(scene.closed_form(i).inscatter.numpy())
        free = luma(scene.closed_form(i, occlusion=None).inscatter.numpy())
        kept.append(shadowed / np.maximum(free, 1e-30))
    kept = np.stack(kept)
    wedge = kept < 0.1
    st = np.ones((1, 7, 7), dtype=bool)
    band = ndimage.binary_dilation(wedge, st) & ~ndimage.binary_erosion(wedge, st)
    edges = depth_edges(scene.depth)
    surf_l = luma(ref["surfaces"])
    lit = (surf_l > 0.05 * np.percentile(surf_l, 99, axis=(1, 2), keepdims=True)) & ~sky
    whole_l = luma(whole_ref)
    dark = whole_l < np.percentile(whole_l, 20, axis=(1, 2), keepdims=True)
    masks = {
        "all": np.ones_like(sky),
        "sky": sky,
        "core": core,
        "ring": ring,
        "wedge": wedge,
        "band": band,
        "edges": edges,
        "lit": lit,
        "dark": dark,
    }
    out["kept_median"] = float(np.median(kept))
    out["pixels"] = {k: float(m.mean()) for k, m in masks.items()}

    # Lamp's pixel: the ray closest to the direction towards it
    lamp = np.array([4.0, 21.0, 1.5])
    lamp_pixels = []
    for t in scene.transforms:
        origin, directions, _ = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
        towards = (lamp - origin) / np.linalg.norm(lamp - origin)
        cosine = directions @ towards
        lamp_pixels.append((np.unravel_index(cosine.argmax(), cosine.shape), float(cosine.max())))
    out["lamp_pixels"] = [[int(p[0]), int(p[1]), c] for p, c in lamp_pixels]
    rows, cols = np.mgrid[: scene.depth.shape[1], : scene.depth.shape[2]]
    distance_to_lamp = np.stack([np.hypot(rows - p[0], cols - p[1]) for p, _ in lamp_pixels])

    where = {}
    for name in ("spp16", "spp64", "spp256"):
        per_pixel = np.abs(renders[name]["single"] - fog_ref).sum(-1)
        top = per_pixel >= np.quantile(per_pixel, 0.99)
        where[name] = {k: float((top & m).sum() / top.sum()) for k, m in masks.items()}
        where[name]["share_of_l1_from_top1pct"] = float(per_pixel[top].sum() / per_pixel.sum())
        where[name]["top_on_edges_within_30px_of_lamp"] = float(
            (top & edges & (distance_to_lamp < 30)).sum() / (top & edges).sum()
        )
        where[name]["top_within_30px_of_lamp"] = float((top & (distance_to_lamp < 30)).sum() / top.sum())
        where[name]["median_distance_to_lamp_px_of_top"] = float(np.median(distance_to_lamp[top]))
        where[name]["local_blurred"] = {k: rl1(renders[name]["single"], fog_ref, True, keep=m) for k, m in masks.items()}
    out["where_fog_noise"] = where
    print("night where", json.dumps(where), flush=True)

    denoised = {}
    for name in ("spp16", "spp64", "spp256", "guided64", "guided256"):
        d = renders[name]["denoised"]
        denoised[name] = {
            k: {"local_blurred": rl1(d, whole_ref, True, keep=m), "bias": signed(d, whole_ref, keep=m)}
            for k, m in masks.items()
        }
    for name in ("guided64", "guided256", "spp256"):
        denoised[f"noisy_{name}"] = {k: signed(renders[name]["total"], whole_ref, keep=m) for k, m in masks.items()}
        denoised[f"noisy_{name}_blocks"] = block_test(renders[name]["total"], whole_ref)
        denoised[f"noisy_{name}_trim"] = trimmed_excess(renders[name]["total"], whole_ref)
    out["denoised_regions"] = denoised
    print("night denoised", json.dumps(denoised), flush=True)

    # Clamping: the point lamp scene (same power, no cube) rendered with Cycles' default clamping and without it
    clamp = {}
    point = ROOT / "point"
    clamped = f64(load_images(point / "bounces0" / "frames", STILL))
    unclamped = f64(load_images(point / "noclamp_bounces0" / "frames", STILL))
    clamp["ratio_unclamped_over_clamped"] = float(unclamped.sum() / clamped.sum())
    for kind in ("direct", "indirect"):
        a = f64(load_images(point / "bounces0" / "volume" / kind, STILL)).sum()
        b = f64(load_images(point / "noclamp_bounces0" / "volume" / kind, STILL)).sum()
        clamp[f"volume_{kind}_ratio"] = float(b / a) if a > 0 else None
    cl = luma(clamped)
    dark_point = cl < np.percentile(cl, 20, axis=(1, 2), keepdims=True)
    bright_point = cl >= np.percentile(cl, 90, axis=(1, 2), keepdims=True)
    clamp["dark20_excess"] = signed(unclamped, clamped, keep=dark_point)
    clamp["top10_excess"] = signed(unclamped, clamped, keep=bright_point)
    clamp["blocks"] = block_test(unclamped, clamped)
    out["clamping_point_scene"] = clamp
    print("clamping", json.dumps(clamp), flush=True)


def matched_excess(error: np.ndarray, ref: np.ndarray, region: np.ndarray, others: np.ndarray, bins=20) -> float:
    """|error| summed over a region, over what pixels outside it with the same reference brightness would give:
    above 1 when the region is noisier than its brightness alone explains. Arrays are per pixel, (n, h, w)."""
    cuts = np.quantile(ref[others], np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(cuts, ref, side="right") - 1, 0, bins - 1)
    rate = np.array(
        [error[others & (index == k)].sum() / max(ref[others & (index == k)].sum(), 1e-30) for k in range(bins)]
    )
    return float(error[region].sum() / (rate[index[region]] * ref[region]).sum())


def followup(out: dict) -> None:
    """Checks added after the first pass: which part of OIDN's regional errors is noise, whether the 4 spp excess
    is spatially independent noise, where the night's largest errors are, and selection bias of masks defined from
    the reference's own noisy values."""
    # (a) Day, OIDN at 64 spp: error of the mean of two seeds against their half difference, per region
    ref = load_day_reference()
    seed1 = f64(load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", STILL))
    total = f64(ref["total"])
    truth = (total + seed1) / 2
    s0, s1 = load_day("day_spp64"), load_day("day_spp64_seed1")
    scene = day_scene(STILL)
    raw = load_images(ROOT / "clear" / "depths", STILL)[..., 0]
    sky = raw >= 1e9
    ground = ~sky & (scene.extra["normals"][..., 2] > 0.95)
    edges = depth_edges(scene.depth)
    log_clear = np.log(luma(scene.clear) + 1e-4)
    contrast = ndimage.maximum_filter(log_clear, size=(1, 3, 3)) - ndimage.minimum_filter(log_clear, size=(1, 3, 3))
    texture = ndimage.binary_dilation(contrast > 0.5, structure=np.ones((1, 3, 3), dtype=bool)) & ~edges & ~sky
    dark_seed0 = luma(total) < np.percentile(luma(total), 20, axis=(1, 2), keepdims=True)
    dark_seed1 = luma(seed1) < np.percentile(luma(seed1), 20, axis=(1, 2), keepdims=True)
    masks = {
        "all": np.ones_like(sky),
        "sky": sky,
        "depth_edges": edges,
        "texture_edges": texture,
        "ground_flat": ground & ~texture & ~edges,
        "dark_20pct": dark_seed0,
    }
    e0 = b5(f64(s0["denoised"]) - truth).sum(-1)
    e1 = b5(f64(s1["denoised"]) - truth).sum(-1)
    half = b5((f64(s0["denoised"]) - s1["denoised"]) / 2 + (total - seed1) / 2).sum(-1)
    light = truth.sum(-1)
    regions = {}
    for name, m in masks.items():
        regions[name] = {
            "seed0": float(np.abs(e0[m]).sum() / light[m].sum()),
            "mean_of_seeds": float(np.abs((e0 + e1)[m] / 2).sum() / light[m].sum()),
            "half_difference": float(np.abs(half[m]).sum() / light[m].sum()),
            "error_map_correlation": corr(e0[m], e1[m]),
            "signed_seed0": float(e0[m].sum() / light[m].sum()),
            "signed_seed1": float(e1[m].sum() / light[m].sum()),
        }
    out["day_oidn64_regions_two_seeds"] = regions
    print("a", json.dumps(regions), flush=True)

    # Dark pixels chosen from one reference and measured against the other: no selection bias
    dark = {}
    for spp in (4, 16, 64):
        r = load_day(f"day_spp{spp}")
        dark[f"spp{spp}"] = {
            "denoised_vs_seed0_dark_by_seed0": signed(r["denoised"], total, keep=dark_seed0),
            "denoised_vs_seed0_dark_by_seed1": signed(r["denoised"], total, keep=dark_seed1),
            "noisy_vs_seed0_dark_by_seed0": signed(r["total"], total, keep=dark_seed0),
            "noisy_vs_seed0_dark_by_seed1": signed(r["total"], total, keep=dark_seed1),
        }
    dark["seed1_vs_seed0_dark_by_seed0"] = signed(seed1, total, keep=dark_seed0)
    out["day_dark_selection"] = dark
    print("dark day", json.dumps(dark), flush=True)

    # (b) Day 4 spp: is the noise independent between neighbouring blocks of 16x16 pixels?
    blocks = {}
    for spp in (4, 64):
        residual = (f64(load_day(f"day_spp{spp}")["total"]) - total).sum(-1)
        n, h, w = residual.shape
        sums = residual[:, : h // 16 * 16, : w // 16 * 16].reshape(n, h // 16, 16, w // 16, 16).sum(axis=(2, 4))
        blocks[f"spp{spp}"] = {
            "horizontal_neighbour_correlation": corr(sums[:, :, :-1], sums[:, :, 1:]),
            "vertical_neighbour_correlation": corr(sums[:, :-1], sums[:, 1:]),
            "pixel_lag1_correlation": corr(residual[:, :, :-1], residual[:, :, 1:]),
            "pixel_lag8_correlation": corr(residual[:, :, :-8], residual[:, :, 8:]),
        }
    out["day_block_independence"] = blocks
    print("b", json.dumps(blocks), flush=True)

    # (c, d, e) Night
    night = night_scene(STILL)
    ref_folder = ROOT / "point_shadow" / "bounces0"
    fog_ref = f64(load_images(ref_folder / "volume" / "direct", STILL))
    whole_ref = f64(load_images(ref_folder / "frames", STILL))
    fog_l = luma(fog_ref)
    edges_n = depth_edges(night.depth)
    kept = []
    for i in range(len(STILL)):
        shadowed = luma(night.closed_form(i).inscatter.numpy())
        free = luma(night.closed_form(i, occlusion=None).inscatter.numpy())
        kept.append(shadowed / np.maximum(free, 1e-30))
    wedge = np.stack(kept) < 0.1
    st = np.ones((1, 7, 7), dtype=bool)
    band = ndimage.binary_dilation(wedge, st) & ~ndimage.binary_erosion(wedge, st)
    # The green cube, from the day scene's render without fog (same camera)
    clear = scene.clear
    green = (clear[..., 1] > 1.3 * clear[..., 0]) & (clear[..., 1] > 1.3 * clear[..., 2]) & ~sky
    cube_silhouette = edges_n & ndimage.binary_dilation(green, structure=np.ones((1, 5, 5), dtype=bool))
    noise = {}
    for spp in (16, 64, 256):
        single = f64(load_images(OUT / f"night_spp{spp}" / "volume" / "direct", STILL))
        error = np.abs(single - fog_ref).sum(-1)
        ref_pixels = fog_ref.sum(-1)
        top = error >= np.quantile(error, 0.99)
        noise[f"spp{spp}"] = {
            "green_cube_pixels": float(green.mean()),
            "top_on_edges": float((top & edges_n).sum() / top.sum()),
            "top_on_cube_silhouette": float((top & cube_silhouette).sum() / top.sum()),
            "edges_matched_excess": matched_excess(error, ref_pixels, edges_n, ~edges_n),
            "band_matched_excess": matched_excess(error, ref_pixels, band, ~band),
            "cube_silhouette_matched_excess": matched_excess(error, ref_pixels, cube_silhouette, ~edges_n),
        }
        for size in (9, 13):  # wider bands around the cube, in case the edge band reaches further from it
            wide = edges_n & ndimage.binary_dilation(green, structure=np.ones((1, size, size), dtype=bool))
            noise[f"spp{spp}"][f"top_on_cube_silhouette_{size}px"] = float((top & wide).sum() / top.sum())
        if spp == 16:
            # Figure: where the 1% largest errors of the fog light are, on the render without fog
            fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
            for k, ax in enumerate(axes):
                ax.imshow(np.clip(clear[k], 0, 1) ** (1 / 2.2))
                ax.contour(edges_n[k].astype(float), [0.5], colors="c", linewidths=0.5)
                ax.contour(green[k].astype(float), [0.5], colors="lime", linewidths=0.8)
                for keep, color, label in (
                    (top[k] & edges_n[k], "r", "on depth edges"),
                    (top[k] & ~edges_n[k], "orange", "elsewhere"),
                ):
                    yy, xx = np.nonzero(keep)
                    ax.scatter(xx, yy, s=2, c=color, label=label)
                ax.set_title(
                    f"frame {STILL[k]}: 1% largest fog-light errors at 16 spp\ncyan: depth-edge mask, green: cube"
                )
                ax.legend(fontsize=7, loc="lower left")
            fig.savefig(SAVE / "night_top_error_pixels_16spp.png", dpi=100)
            plt.close(fig)
    out["night_noise_matched"] = noise
    print("c/d", json.dumps(noise), flush=True)

    # Night dark pixels: chosen from the reference itself, from its 5x5 blur, or from an independent render
    other = luma(f64(load_images(OUT / "night_ms_ref" / "frames", STILL)))
    own = luma(whole_ref)
    choices = {
        "by_reference": own < np.percentile(own, 20, axis=(1, 2), keepdims=True),
        "by_blurred_reference": luma(b5(whole_ref)) < np.percentile(luma(b5(whole_ref)), 20, axis=(1, 2), keepdims=True),
        "by_night_ms_ref_2048": other < np.percentile(other, 20, axis=(1, 2), keepdims=True),
    }
    lit_surfaces = {}
    for spp in (16, 64, 256, 1024):
        folder = OUT / f"night_spp{spp}"
        noisy, denoised = f64(load_images(folder / "noisy", STILL)), f64(load_images(folder / "frames", STILL))
        lit_surfaces[f"spp{spp}"] = {
            f"{kind}_{name}": signed(image, whole_ref, keep=m)
            for name, m in choices.items()
            for kind, image in (("noisy", noisy), ("denoised", denoised))
        }
    for name in ("guided64", "guided256"):
        noisy = f64(load_images(OUT / f"night_{name}" / "noisy", STILL))
        lit_surfaces[name] = {f"noisy_{k}": signed(noisy, whole_ref, keep=m) for k, m in choices.items()}
    out["night_dark_selection"] = lit_surfaces
    print("e", json.dumps(lit_surfaces), flush=True)


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    if "--followup" in sys.argv:
        out: dict = {}
        followup(out)
        (SAVE / "check_monte_carlo_followup.json").write_text(json.dumps(out, indent=1, default=float))
        print("saved", flush=True)
        return
    out = {"day": {}, "night": {}}
    check_day(out["day"])
    check_night(out["night"])
    (SAVE / "check_monte_carlo.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
