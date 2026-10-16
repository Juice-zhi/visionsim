"""Independent check of compute_less.py's findings on the "compute less" shortcuts of ``exp_less.py``.

Re-derives the key numbers with separate code (own masks, own splits) and runs extra tests meant to refute the
analyzer's causal claims:

- low resolution: own depth-edge masks (4-neighbour jumps rather than a 3x3 range), the fraction of pixels where the
  guided upsampler's four bilateral weights collapse and it falls back on the nearest depth, and how far apart in
  depth neighbouring 1/4 samples are on the far ground; at night, the error of the lamp's cube against other objects'
  silhouettes, the sign of the error on and next to the cube, and whether the mean source J is smoother than the fog
  light between samples of similar depth (the reason guided_source gains nothing at night);
- reuse: that the reprojection lands on the same world points, the method's error split again, and, by marching the
  closed form's lighting with every cell of the sky visible, whether the path effect of the light from the sky, the
  ground and multiple scattering is caused by objects hiding the sky; the noise floor of the 48-node march;
- density: the sky-sample swap, and whether 64 sky samples are converged (against 128).

Outputs in ``runs/fog-comparison/acceleration/analysis/check_compute_less/check.json``.

Run from the repository root, on the CPU, optionally naming sections (masks, lowres, reuse, paths, density)::

    CUDA_VISIBLE_DEVICES="" PYTHONUTF8=1 python examples/medium/experiment/acceleration/analysis/check_compute_less.py
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *
from common import OUT, ROOT, DayLight, day_nodes, day_scene, errors, load_images, night_scene

torch.set_num_threads(3)  # other jobs share the CPU

from scipy import ndimage
from visionsim.medium import apply_medium, trace_shadows
from visionsim.medium.optics import henyey_greenstein
from visionsim.medium.render import camera_rays

import exp_less

SAVE = OUT / "analysis" / "check_compute_less"
FRAMES = (255, 345)
LAMP = np.array([4.0, 21.0, 1.5])


def f64(x):
    return np.asarray(x, dtype=np.float64)


def ray_geometry(transform, depth):
    """Distance along each pixel's ray to its surface (or the fog box), and the world point it reaches."""
    origin, directions, scale = (x.numpy() for x in camera_rays(transform, transform["transform_matrix"]))
    distance = depth * scale
    return distance, origin + directions * distance[..., None]


def jumps(distance, threshold=0.1):
    """Pixels on either side of a 4-neighbour jump of the log distance larger than the threshold."""
    log = np.log(distance)
    out = np.zeros(distance.shape, dtype=bool)
    dx = np.abs(np.diff(log, axis=1)) > threshold
    dy = np.abs(np.diff(log, axis=0)) > threshold
    out[:, 1:] |= dx
    out[:, :-1] |= dx
    out[1:] |= dy
    out[:-1] |= dy
    return out


def dilate(mask, r):
    return ndimage.binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool)) if r else mask


def share(ours, exact, mask, total):
    """Sum over the mask of |ours - exact| (all channels) over total; mask (n, h, w)."""
    return float((np.abs(f64(ours) - f64(exact)).sum(-1) * mask).sum() / total)


def signed(ours, exact, mask, total):
    return float(((f64(ours) - f64(exact)).sum(-1) * mask).sum() / total)


def guided_with_diagnostics(values, low_depth, depth, s, sigma=0.05):
    """exp_less.upsample(guided=True), re-written, also returning the bilateral weight sum and the fallback mask."""
    v, d, high = (np.asarray(x, dtype=np.float64) for x in (values, low_depth, depth))
    hl, wl = d.shape
    H, W = high.shape
    fy = np.clip((np.arange(H) - s // 2) / s, 0, hl - 1)
    fx = np.clip((np.arange(W) - s // 2) / s, 0, wl - 1)
    y0, x0 = np.minimum(np.floor(fy), hl - 2).astype(int), np.minimum(np.floor(fx), wl - 2).astype(int)
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    total, wsum = 0.0, 0.0
    gaps, weights = [], []
    best, best_gap = None, None
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = np.broadcast_to((y0 + dy)[:, None], (H, W)), np.broadcast_to((x0 + dx)[None, :], (H, W))
            w = (wy if dy else 1 - wy) * (wx if dx else 1 - wx)
            gap = np.abs(np.log(d[yi, xi]) - np.log(high))
            gaps.append(gap)
            w = w * np.exp(-((gap / sigma) ** 2) / 2)
            weights.append(w)
            total = total + w[..., None] * v[yi, xi]
            wsum = wsum + w
            if best is None:
                best, best_gap = v[yi, xi].copy(), gap.copy()
            else:
                closer = gap < best_gap
                best[closer], best_gap[closer] = v[yi, xi][closer], gap[closer]
    fallback = wsum < 1e-3
    result = np.where(fallback[..., None], best, total / np.maximum(wsum, 1e-12)[..., None])
    dominance = np.max(weights, axis=0) / np.maximum(wsum, 1e-300)
    return result, wsum, fallback, np.min(gaps, axis=0), dominance


# --------------------------------------------------------------------------------------------------------------------


def lowres_section() -> dict:
    out: dict = {}
    raw = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    transforms = load_transforms(ROOT / "clear" / "frames", FRAMES)
    normals_cam = load_images(ROOT / "clear" / "normals", FRAMES)
    rotations = [np.asarray(t["transform_matrix"])[:3, :3] for t in transforms]
    normals = np.stack([n @ (r / np.linalg.norm(r, axis=0)) for n, r in zip(normals_cam, rotations)])
    for scene in (day_scene(frames=FRAMES), night_scene(FRAMES)):
        is_day = scene.name == "day"
        n = len(FRAMES)
        exact = np.stack([scene.closed_form(i, normals=None).inscatter.numpy() for i in range(n)])
        unshadowed = np.stack([scene.closed_form(i, normals=None, occlusion=None).inscatter.numpy() for i in range(n)])
        total = f64(exact).sum()
        geo = [ray_geometry(scene.transforms[i], scene.depth[i]) for i in range(n)]
        distance = np.stack([g[0] for g in geo])
        points = np.stack([g[1] for g in geo])
        sky = raw >= 1e9
        ground = (normals[..., 2] > 0.95) & ~sky
        objects = ~sky & ~ground
        jump = np.stack([jumps(distance[i]) for i in range(n)])
        near_object_jump = np.stack([jump[i] & dilate(objects[i], 1) for i in range(n)])
        masks = {f"silhouette_r{r}": np.stack([dilate(near_object_jump[i], r) for i in range(n)]) for r in (2, 4, 8)}
        any_edge = np.stack([dilate(jump[i], 4) for i in range(n)])
        masks["other_depth_edges_r4"] = any_edge & ~masks["silhouette_r4"]
        masks["far_ground_gt15m"] = ground & ~any_edge & (distance > 15)
        masks["near_ground_le15m"] = ground & ~any_edge & (distance <= 15)
        masks["sky_no_edge"] = sky & ~any_edge
        masks["object_faces_no_edge"] = objects & ~any_edge
        entry: dict = {
            "total_fog_light": total,
            "unshadowed_over_shadowed_fog_light": float(f64(unshadowed).sum() / total),
            "mask_pixels": {k: float(m.mean()) for k, m in masks.items()},
        }
        if not is_day:
            # The lamp's cube: object pixels whose world point lies within 2.5 m horizontally of the lamp
            horizontal = np.linalg.norm(points[..., :2] - LAMP[:2], axis=-1)
            cube = objects & (horizontal < 2.5)
            cube_sil = np.stack([dilate(jump[i] & dilate(cube[i], 1), 4) for i in range(n)])
            masks["lamp_cube_silhouette_r4"] = cube_sil
            masks["other_silhouette_r4"] = masks["silhouette_r4"] & ~cube_sil
            masks["cube_pixels_in_silhouette"] = masks["silhouette_r4"] & cube
            masks["non_cube_pixels_in_silhouette"] = masks["silhouette_r4"] & ~cube
            entry["lamp_cube_pixels"] = float(cube.mean())
            lum = f64(exact) @ LUMA
            entry["cube_fog_light_over_mean"] = float(lum[cube].mean() / lum.mean())
        for s in (2, 4):
            lows = [
                exp_less.low_resolution(
                    scene.clear[i], scene.depth[i], scene.transforms[i], scene.medium, scene.lighting, scene.occlusion, s
                )
                for i in range(n)
            ]
            res: dict = {}
            for name in ("bilinear", "guided", "guided_source"):
                ours = np.stack([low["inscatter"][name] for low in lows])
                res[name] = {
                    "rel_l1": errors(ours, exact)["rel_l1"],
                    "rel_l1_blurred": errors(ours, exact)["rel_l1_blurred"],
                    "shares": {k: share(ours, exact, m, total) for k, m in masks.items()},
                    "bias": {k: signed(ours, exact, m, total) for k, m in masks.items()},
                }
            # Guided fallback diagnostics, on the fog light (as exp_less upsamples it)
            diag = {
                "fallback_pixels": {},
                "error_share_fallback": {},
                "error_share_reweighted": {},
                "median_min_log_gap": {},
            }
            guided_ours, fallbacks = [], []
            for i in range(n):
                low_depth = exp_less.subsample(scene.depth[i], s)
                # The low-resolution fog light, recomputed exactly as exp_less does, to feed our re-implementation
                small = exp_less.low_camera(scene.transforms[i], s)
                r = apply_medium(
                    exp_less.subsample(scene.clear[i], s),
                    low_depth,
                    small,
                    scene.transforms[i]["transform_matrix"],
                    scene.medium,
                    scene.lighting,
                    occlusion=scene.occlusion,
                    dtype=DTYPE,
                )
                g, wsum, fb, gap, _ = guided_with_diagnostics(r.inscatter.numpy(), low_depth, scene.depth[i], s)
                guided_ours.append(g)
                fallbacks.append((fb, gap))
            guided_ours = np.stack(guided_ours)
            fb = np.stack([f[0] for f in fallbacks])
            gap = np.stack([f[1] for f in fallbacks])
            diag["max_abs_diff_vs_exp_less_guided"] = float(
                np.abs(guided_ours - np.stack([low["inscatter"]["guided"] for low in lows])).max() / f64(exact).mean()
            )
            for k in ("silhouette_r4", "other_depth_edges_r4", "far_ground_gt15m", "near_ground_le15m", "sky_no_edge"):
                m = masks[k]
                diag["fallback_pixels"][k] = float(fb[m].mean())
                diag["error_share_fallback"][k] = share(guided_ours, exact, m & fb, total)
                diag["error_share_reweighted"][k] = share(guided_ours, exact, m & ~fb, total)
                diag["median_min_log_gap"][k] = float(np.median(gap[m]))
            res["guided_diagnostics"] = diag
            if s == 4:
                # Upsampling without shadows, normalised by the shadowed fog light (comparable with the method's)
                free = [
                    exp_less.low_resolution(
                        scene.clear[i], scene.depth[i], scene.transforms[i], scene.medium, scene.lighting, None, s
                    )["inscatter"]
                    for i in range(n)
                ]
                res["without_shadows_over_shadowed_total"] = {
                    name: share(np.stack([f[name] for f in free]), unshadowed, np.ones(exact.shape[:3], bool), total)
                    for name in ("bilinear", "guided", "guided_source")
                }
            out_key = f"{scene.name}_1/{s}"
            out[out_key] = res
            print(
                out_key,
                json.dumps({k: (v["rel_l1"] if "rel_l1" in v else None) for k, v in res.items() if isinstance(v, dict)}),
                flush=True,
            )
        # Is J smoother than the fog light between samples s=4 px apart with similar depth? Transfer error of
        # copying a neighbour's value (fog light) or its J times this pixel's 1 - T (guided_source's assumption).
        T = np.stack([scene.transmittance(i) for i in range(n)])
        J = np.where(1 - T > 1e-6, exact / np.maximum(1 - T, 1e-6), 0.0)
        transfer = {}
        for step in (2, 4):
            a, b = [], []
            for axis in (1, 2):
                sl0 = [slice(None)] * 4
                sl1 = [slice(None)] * 4
                sl0[axis], sl1[axis] = slice(0, -step), slice(step, None)
                ld = np.log(distance)[..., None]
                similar = np.abs(ld[tuple(sl0)] - ld[tuple(sl1)]) < 0.05
                I0, I1 = f64(exact[tuple(sl0)]), f64(exact[tuple(sl1)])
                J1, T0 = f64(J[tuple(sl1)]), f64(T[tuple(sl0)])
                a.append((np.abs(I1 - I0) * similar).sum())
                b.append((np.abs(J1 * (1 - T0) - I0) * similar).sum())
            transfer[f"{step}px"] = {
                "copy_fog_light": float(sum(a) / total),
                "copy_J_times_own_opacity": float(sum(b) / total),
            }
        entry["neighbour_transfer_error_similar_depth"] = transfer
        out[scene.name] = entry
    return out


# --------------------------------------------------------------------------------------------------------------------


def sun_only(scene):
    medium = scene.medium.model_copy(update={"multiple_scattering": False})
    lighting = scene.lighting.model_copy(update={"sky": (0.0, 0.0, 0.0), "ground_albedo": (0.0, 0.0, 0.0)})
    return medium, lighting


def reuse_section() -> dict:
    offsets = (1, 16, 32)
    frames = (255,) + tuple(255 + k for k in offsets)
    day = day_scene(frames=frames)
    sm, sl = sun_only(day)
    raw = load_images(ROOT / "clear" / "depths", frames)[..., 0]

    def parts(i, **options):
        full = day.closed_form(i, normals=None, **options)
        sun = day.closed_form(i, medium=sm, lighting=sl, normals=None, **options)
        return f64(full.inscatter.numpy()), f64(sun.inscatter.numpy()), f64(full.transmittance.numpy())

    key_all, key_sun, key_T = parts(0)
    key_free, key_free_sun, _ = parts(0, occlusion=None)
    op_key = np.maximum(1 - key_T, 1e-6)
    _, key_dirs, _ = camera_rays(day.transforms[0], day.transforms[0]["transform_matrix"])
    towards = torch.as_tensor(day.lighting.suns[0].direction, dtype=torch.float64)
    towards = towards / towards.norm()
    hg_key = henyey_greenstein(key_dirs @ towards, day.medium.anisotropy).numpy()[..., None]
    _, key_points = ray_geometry(day.transforms[0], day.depth[0])
    key = {
        "camera": day.transforms[0],
        "depth": day.depth[0],
        "mean": np.concatenate(
            [key_all / op_key, key_sun / op_key, key_sun / op_key / hg_key, key_free / op_key, key_points], axis=-1
        ),
    }
    out: dict = {}
    for j, k in enumerate(offsets, start=1):
        target = {"camera": day.transforms[j], "depth": day.depth[j]}
        mean, seen = exp_less.reproject(key, target)
        mean = f64(mean)
        all_j, sun_j, sun_over_hg, free_j, pts = (mean[..., a:b] for a, b in ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15)))
        dist_t, pts_t = ray_geometry(day.transforms[j], day.depth[j])
        miss = np.linalg.norm(pts - pts_t, axis=-1)[seen]
        rel_miss = (np.linalg.norm(pts - pts_t, axis=-1) / dist_t)[seen]
        exact, exact_sun, T = parts(j)
        free, _, _ = parts(j, occlusion=None)
        total = exact.sum()
        _, dirs, _ = camera_rays(day.transforms[j], day.transforms[j]["transform_matrix"])
        hg = henyey_greenstein(dirs @ towards, day.medium.anisotropy).numpy()[..., None]
        op = 1 - T
        S, U = seen[None].astype(float)[0], (~seen).astype(float)
        used = np.where(seen[..., None], all_j * op, free)
        corrected = np.where(seen[..., None], (all_j - sun_j + sun_over_hg * hg) * op, free)
        edges = dilate(jumps(dist_t), 4)
        sky = raw[j] >= 1e9
        out[k] = {
            "reprojected_point_miss_m_median": float(np.median(miss)),
            "reprojected_point_miss_m_p99": float(np.percentile(miss, 99)),
            "reprojected_point_miss_relative_p99": float(np.percentile(rel_miss, 99)),
            "seen_pixels": float(seen.mean()),
            "rel_l1": share(used[None], exact[None], np.ones((1, *seen.shape)), total),
            "unseen_share": share(free[None], exact[None], U[None], total),
            "seen_share": share((all_j * op)[None], exact[None], S[None], total),
            "seen_share_sun_part": share((sun_j * op)[None], exact_sun[None], S[None], total),
            "seen_share_sun_part_phase_corrected": share((sun_over_hg * hg * op)[None], exact_sun[None], S[None], total),
            "seen_share_phase_corrected": share(corrected[None], exact[None], S[None], total),
            "seen_share_no_shadows_over_unshadowed": share((free_j * op)[None], free[None], S[None], free.sum()),
            "seen_bias": signed((all_j * op)[None], exact[None], S[None], total),
            "seen_bias_rest_part": signed(((all_j - sun_j) * op)[None], (exact - exact_sun)[None], S[None], total),
            "seen_bias_sun_part": signed((sun_j * op)[None], exact_sun[None], S[None], total),
            "unseen_near_depth_edges_fraction": float((~seen & edges).sum() / max((~seen).sum(), 1)),
            "unseen_sky_fraction": float((~seen & sky).sum() / max((~seen).sum(), 1)),
        }
        print("reuse", k, json.dumps(out[k]), flush=True)
    return out


# --------------------------------------------------------------------------------------------------------------------


def paths_section(offsets=(32, 1)) -> dict:
    """Rest part (sky, ground, multiple scattering) marched along the key's and the target's rays, with the sky's
    visibility as the closed form models it and with every cell visible, plus the mean visible fraction of the sky."""
    frames = (255,) + tuple(255 + k for k in offsets)
    day = day_scene(frames=frames)
    light = DayLight(day)
    key_origin = day.rays(0)[0]
    blank = {"camera": day.transforms[0], "depth": day.depth[0], "mean": np.zeros((*day.depth[0].shape, 1))}
    cells = len(light.towards)

    def march(origin, dirs, dist, look, nodes, occluded):
        t, w = day_nodes(light.fog, light.beta, origin, dirs, dist, nodes)
        pts = (origin + dirs[:, None] * t[..., None]).reshape(-1, 3)
        L = look[:, None].expand(-1, nodes, -1).reshape(-1, 3)
        c = light.depth_above(pts)
        if occluded:
            lit = light.sky_visibility(pts)
        else:
            ones = torch.ones(len(pts), cells, dtype=torch.bool)
            lit = (ones, ones)
        ratios = light.ratios(pts, L, *lit)
        value = sum(light.source(name, L[:, 2], c) * ratios[name] for name in light.tables)
        mean = (value.reshape(-1, nodes, 3) * w[..., None]).sum(1) / (w * nodes).clamp_min(1e-12)
        sky_fraction = ratios["sky"].reshape(-1, nodes).mean(1, keepdim=True)
        return torch.cat([mean, sky_fraction], dim=-1)

    def run(origin, dirs, dist, look, nodes, occluded, chunk=1024):
        return by_chunks(
            lambda d, r, k: march(origin, d, r, k, nodes, occluded), len(dirs), chunk, dirs, dist, look
        ).numpy()

    out: dict = {}
    for j, k in enumerate(offsets, start=1):
        origin, dirs, dist = day.rays(j)
        pts = origin + dirs * dist[:, None]
        towards = pts - key_origin
        key_dist = towards.norm(dim=-1)
        key_dirs = towards / key_dist[:, None]
        _, seen = exp_less.reproject(blank, {"camera": day.transforms[j], "depth": day.depth[j]})
        seen = seen.reshape(-1)
        op = 1 - exp_less.transmittance(day.medium, day.transforms[j], day.depth[j]).reshape(-1, 1)
        entry = {}
        start = time.perf_counter()
        for occluded in (True, False):
            tgt = run(origin, dirs, dist, dirs, 48, occluded)
            kp = run(key_origin, key_dirs, key_dist, dirs, 48, occluded)
            kk = run(key_origin, key_dirs, key_dist, key_dirs, 48, occluded)
            total = float((tgt[:, :3] * op).sum())
            S = seen[:, None]

            def sh(a, b, sign=False):
                d = (a[:, :3] - b[:, :3]) * op * S
                return float((d if sign else np.abs(d)).sum() / total)

            # Shares below are of the rest part's fog light (about 80% of all fog light by day), not of all of it
            name = "sky_occluded" if occluded else "sky_all_visible"
            entry[name] = {
                "path_only": sh(kp, tgt),
                "path_only_bias": sh(kp, tgt, True),
                "direction_only": sh(kk, kp),
                "direction_only_bias": sh(kk, kp, True),
                "path_and_direction": sh(kk, tgt),
                "path_and_direction_bias": sh(kk, tgt, True),
                "normaliser_rest_fog_light_all_pixels": total,
            }
            if occluded:
                wt = tgt[:, :3].sum(-1) * op[:, 0] * seen
                entry[name]["mean_visible_sky_fraction_target_ray"] = float((tgt[:, 3] * wt).sum() / wt.sum())
                entry[name]["mean_visible_sky_fraction_key_ray"] = float((kp[:, 3] * wt).sum() / wt.sum())
                more = (kp[:, 3] > tgt[:, 3] + 1e-3) & seen
                less = (kp[:, 3] < tgt[:, 3] - 1e-3) & seen
                entry[name]["pixels_key_sees_more_sky"] = float(more.mean())
                entry[name]["pixels_key_sees_less_sky"] = float(less.mean())
                # How much of the positive path bias comes from pixels where the key's ray sees more sky
                d = ((kp[:, :3] - tgt[:, :3]) * op).sum(-1)
                entry[name]["path_bias_where_key_sees_more_sky"] = float(d[more].sum() / total)
                entry[name]["path_bias_elsewhere"] = float(d[seen & ~more].sum() / total)
                if k == 1:
                    # Noise floor of the 48-node march: target marched with 48 against 192 nodes
                    fine = run(origin, dirs, dist, dirs, 192, True, chunk=256)
                    kp_fine = run(key_origin, key_dirs, key_dist, dirs, 192, True, chunk=256)
                    entry[name]["target_48_vs_192_nodes"] = sh(tgt, fine)
                    entry[name]["path_only_192_nodes"] = sh(kp_fine, fine)
        entry["seconds"] = time.perf_counter() - start
        out[k] = entry
        print("paths", k, json.dumps(entry, default=float), flush=True)
    return out


# --------------------------------------------------------------------------------------------------------------------


def density_section() -> dict:
    day = day_scene(frames=(345,))
    t = day.transforms[0]
    pose = t["transform_matrix"]
    args = (day.clear[0], day.depth[0], t, pose)

    def trace(medium, samples=8):
        return trace_shadows(
            day.occlusion, day.depth[0], t, pose, medium, day.lighting, shadow_samples=samples, dtype=DTYPE
        )

    def render(medium, shadows):
        return f64(apply_medium(*args, medium, day.lighting, shadows=shadows, dtype=DTYPE).radiance.numpy())

    shared = trace(day.medium)
    out: dict = {}
    for scale in (0.5, 4.0):
        medium = day.medium.model_copy(update={"extinction": day.medium.extinction * scale})
        own, own64, own128 = trace(medium), trace(medium, 64), trace(medium, 128)
        sep = render(medium, own)
        via_occlusion = f64(
            apply_medium(*args, medium, day.lighting, occlusion=day.occlusion, dtype=DTYPE).radiance.numpy()
        )
        shr = render(medium, shared)
        r64, r128 = render(medium, own64), render(medium, own128)
        e = lambda a, b: errors(a[None], b[None])["rel_l1"]
        out[str(scale)] = {
            "own_render_vs_apply_medium_with_occlusion": e(sep, via_occlusion),
            "shared_vs_own": e(shr, sep),
            "sky_swapped_vs_own": e(render(medium, shared._replace(suns=own.suns)), sep),
            "sun_swapped_vs_own": e(render(medium, own._replace(suns=shared.suns)), sep),
            "own8_vs_64": e(sep, r64),
            "shared8_vs_64": e(shr, r64),
            "own64_vs_128": e(r64, r128),
            "own8_vs_128": e(sep, r128),
            "shared8_vs_128": e(shr, r128),
        }
        print("density", scale, json.dumps(out[str(scale)]), flush=True)
    return out


def masks_section() -> dict:
    """The analyzer's own region masks (``compute_less.regions``, read only) for the guided upsampler's fallback test
    by day at 1/4, and the extent of the night lamp's cube mask used above, saved as a PNG for inspection."""
    sys.path.insert(0, "examples/medium/experiment/acceleration/analysis")
    import compute_less as cl
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    day = day_scene(frames=FRAMES)
    raw = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    sm, sl = sun_only(day)
    out: dict = {"day_1/4_guided": {}}
    acc: dict = {}
    for i in range(len(FRAMES)):
        exact = f64(day.closed_form(i, normals=None).inscatter.numpy())
        ratio = cl.ratio(
            *(
                day.closed_form(i, medium=sm, lighting=sl, normals=None, occlusion=o).inscatter.numpy()
                for o in (day.occlusion, None)
            )
        )
        masks = cl.regions(day.transforms[i], day.depth[i], raw[i], day.extra["normals"][i], ratio)
        low_depth = exp_less.subsample(day.depth[i], 4)
        r = apply_medium(
            exp_less.subsample(day.clear[i], 4),
            low_depth,
            exp_less.low_camera(day.transforms[i], 4),
            day.transforms[i]["transform_matrix"],
            day.medium,
            day.lighting,
            occlusion=day.occlusion,
            dtype=DTYPE,
        )
        guided, wsum, fb, gap, dominance = guided_with_diagnostics(r.inscatter.numpy(), low_depth, day.depth[i], 4)
        bilinear = exp_less.upsample(r.inscatter.numpy(), low_depth, day.depth[i], 4, guided=False)
        error_g, error_b = np.abs(guided - exact).sum(-1), np.abs(bilinear - exact).sum(-1)
        for name, m in masks.items():
            a = acc.setdefault(
                name, {"n": 0, "fb": 0, "dom90": 0, "gap": [], "eg_fb": 0.0, "eg": 0.0, "eb": 0.0, "eg_dom90": 0.0}
            )
            a["n"] += int(m.sum())
            a["fb"] += int((m & fb).sum())
            a["dom90"] += int((m & (dominance > 0.9)).sum())
            a["gap"].append(gap[m])
            a["eg_fb"] += float(error_g[m & fb].sum())
            a["eg_dom90"] += float(error_g[m & (dominance > 0.9) & ~fb].sum())
            a["eg"] += float(error_g[m].sum())
            a["eb"] += float(error_b[m].sum())
        acc["_total"] = acc.get("_total", 0.0) + float(exact.sum())
    total = acc.pop("_total")
    for name, a in acc.items():
        out["day_1/4_guided"][name] = {
            "pixels": a["n"] / (2 * 180 * 320),
            "fallback_fraction": a["fb"] / max(a["n"], 1),
            "dominant_weight_over_90pct_fraction": a["dom90"] / max(a["n"], 1),
            "median_min_log_gap": float(np.median(np.concatenate(a["gap"]))),
            "guided_error_share": a["eg"] / total,
            "guided_error_share_from_fallback": a["eg_fb"] / total,
            "guided_error_share_from_dominant_weight": a["eg_dom90"] / total,
            "bilinear_error_share": a["eb"] / total,
        }
    # Night lamp cube mask: object pixels within 2.5 m horizontally of the lamp
    night = night_scene(FRAMES)
    normals_ok = day.extra["normals"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
    cube_info = {}
    for i in range(len(FRAMES)):
        dist, pts = ray_geometry(night.transforms[i], night.depth[i])
        sky = raw[i] >= 1e9
        objects = ~sky & ~((normals_ok[i][..., 2] > 0.95) & ~sky)
        cube = objects & (np.linalg.norm(pts[..., :2] - LAMP[:2], axis=-1) < 2.5)
        ys, xs = np.nonzero(cube)
        cube_info[FRAMES[i]] = {
            "pixels": int(cube.sum()),
            "columns": [int(xs.min()), int(xs.max())] if len(xs) else None,
            "rows": [int(ys.min()), int(ys.max())] if len(ys) else None,
            "world_min": pts[cube].min(0).tolist() if len(xs) else None,
            "world_max": pts[cube].max(0).tolist() if len(xs) else None,
            "distance_median_m": float(np.median(dist[cube])) if len(xs) else None,
        }
        fog = night.closed_form(i).inscatter.numpy() @ LUMA
        axes[i].imshow(np.clip(fog / np.percentile(fog, 99.5), 0, 1) ** (1 / 2.2), cmap="gray")
        axes[i].contour(cube, levels=[0.5], colors="r", linewidths=0.8)
        axes[i].contour(objects, levels=[0.5], colors="c", linewidths=0.5)
        axes[i].set_title(f"night fog light, frame {FRAMES[i]}: red = lamp cube mask, cyan = objects", fontsize=9)
    fig.savefig(SAVE / "night_lamp_cube_mask.png", dpi=100)
    plt.close(fig)
    out["night_lamp_cube"] = cube_info
    print(json.dumps(out, indent=1), flush=True)
    return out


SECTIONS = {
    "masks": masks_section,
    "lowres": lowres_section,
    "reuse": reuse_section,
    "paths": paths_section,
    "density": density_section,
}


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    path = SAVE / "check.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in sys.argv[1:] or SECTIONS:
        start = time.perf_counter()
        results[name] = SECTIONS[name]()
        results[name + "_seconds"] = time.perf_counter() - start
        path.write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
