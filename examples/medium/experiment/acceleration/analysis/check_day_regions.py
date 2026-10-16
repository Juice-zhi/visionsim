"""Independent check of day_regions.py's findings: where and why each method's error of the day scene arises.

Re-derives the key numbers from the saved images with separate code (own masks, own splits), and runs extra tests
meant to refute the analyzer's causal claims:

- noise of the reference's own lights, estimated from 1024 spp (its noise is 1/sqrt(5) of 1024 spp vs 4096 spp), to
  tell whether the closed form's single scattering and surfaces are really at the noise floor;
- the closed form's light scattered more than once against azimuth at fixed elevation, both its own and Cycles', and
  outside the light shafts, to test the azimuth explanation;
- shadowed against sunlit ground at the same distance and the same angle to the sun, to test the shadow explanation;
- the caches against their own marching without a cache (key ``exact``) rather than the analytic closed form;
- edge profiles normalized by the fog's light rather than the whole image (fog's share confounds them);
- a silhouette mask from objects' normals (object/non-object boundary) instead of depth jumps;
- Unreal's fog light against Cycles' single scattering and the closed form's sun-only and sky-only single scattering,
  by angle to the sun;
- denoised Monte Carlo at 64 spp with two seeds, to tell residual noise from deterministic bias.

Outputs in ``runs/fog-comparison/acceleration/analysis/check_day_regions``: ``check.json`` and a cache of arrays.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *
from common import OUT, ROOT, DayLight, blur, day_scene, errors, load_images

torch.set_num_threads(3)  # other jobs share the CPU

import OpenEXR
from scipy import ndimage
from visionsim.medium.render import camera_rays

FRAMES = (255, 345)
SAVE = OUT / "analysis" / "check_day_regions"
CYCLES = ROOT / "ms" / "cycles_ref"


def f64(x):
    return np.asarray(x, dtype=np.float64)


def prepare(scene, light) -> dict[str, np.ndarray]:
    """Closed-form variants and froxels for both frames, cached on disk."""
    path = SAVE / "arrays.npz"
    if path.exists():
        return dict(np.load(path))
    no_ms = scene.medium.model_copy(update={"multiple_scattering": False})
    lights = {
        "single": scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)}),
        "single_sun": scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0), "sky": (0.0, 0.0, 0.0)}),
        "single_sky": scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0), "suns": []}),
    }
    out: dict[str, list] = {k: [] for k in ("radiance", "fog", "single", "single_sun", "single_sky", "froxel")}
    from exp_froxel import froxels

    for k in range(len(FRAMES)):
        full = scene.closed_form(k)
        out["radiance"].append(full.radiance.numpy())
        out["fog"].append(full.inscatter.numpy())
        for name, lighting in lights.items():
            out[name].append(scene.closed_form(k, medium=no_ms, lighting=lighting).inscatter.numpy())
        out["froxel"].append(froxels(scene, light, k, 8, 64, 1)[0])
        print("prepared frame", FRAMES[k], flush=True)
    arrays = {k: np.stack(v) for k, v in out.items()}
    np.savez_compressed(path, **arrays)
    return arrays


def masks_for(scene, light) -> dict[str, np.ndarray]:
    raw = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    towards = np.asarray(scene.lighting.suns[0].direction, dtype=np.float64)
    towards /= np.linalg.norm(towards)
    sun_azimuth = np.degrees(np.arctan2(towards[1], towards[0]))
    m: dict[str, list] = {
        k: []
        for k in (
            "sky",
            "distance",
            "elevation",
            "azimuth",
            "sun_angle",
            "ground",
            "objects",
            "ground_shadow",
            "object_silhouette",
            "object_edge_distance",
        )
    }
    for i in range(len(FRAMES)):
        t = scene.transforms[i]
        origin, directions, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
        sky = raw[i] >= 1e9
        distance = scene.depth[i] * scale
        ground = (scene.extra["normals"][i][..., 2] > 0.95) & ~sky
        objects = ~sky & ~ground
        # Object silhouettes: object pixels next to non-object pixels (ground or sky), or object pixels next to object
        # pixels much farther (one object in front of another)
        boundary = np.zeros_like(objects)
        for axis in (0, 1):
            a, b = np.delete(objects, -1, axis), np.delete(objects, 0, axis)
            da, db = np.delete(distance, -1, axis), np.delete(distance, 0, axis)
            pair = (a != b) | (a & b & (np.abs(np.log(da / db)) > np.log(1.15)))
            for pad in ((0, 1), (1, 0)):
                boundary |= np.pad(pair, [pad if x == axis else (0, 0) for x in (0, 1)])
        points = torch.as_tensor(origin + directions * distance[..., None], dtype=torch.float32).reshape(-1, 3)
        sunlit = light.sun_visibility(points).numpy().reshape(sky.shape)
        azimuth = np.degrees(np.arctan2(directions[..., 1], directions[..., 0]))
        m["sky"].append(sky)
        m["distance"].append(distance)
        m["elevation"].append(np.degrees(np.arcsin(np.clip(directions[..., 2], -1, 1))))
        m["azimuth"].append(np.abs((azimuth - sun_azimuth + 180) % 360 - 180))
        m["sun_angle"].append(np.degrees(np.arccos(np.clip(directions @ towards, -1, 1))))
        m["ground"].append(ground)
        m["objects"].append(objects)
        m["ground_shadow"].append(ground & (sunlit < 0.5))
        m["object_silhouette"].append(ndimage.binary_dilation(boundary, iterations=2))
        m["object_edge_distance"].append(ndimage.distance_transform_edt(~boundary))
    return {k: np.stack(v) for k, v in m.items()}


def analyzer_edges(scene) -> tuple[np.ndarray, np.ndarray]:
    """The analyzer's silhouette mask, rebuilt the same way, to see what it contains."""
    edges, dist = [], []
    for i in range(len(FRAMES)):
        t = scene.transforms[i]
        _, _, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
        distance = scene.depth[i] * scale
        values = np.log(scene.depth[i])
        found = np.zeros(values.shape, dtype=bool)
        for axis in (0, 1):
            pairs = np.abs(np.diff(values, axis=axis)) > np.log(1.15)
            pairs &= np.minimum(np.delete(distance, -1, axis=axis), np.delete(distance, 0, axis=axis)) < 100
            for pad in ((0, 1), (1, 0)):
                found |= np.pad(pairs, [pad if a == axis else (0, 0) for a in (0, 1)])
        edges.append(ndimage.binary_dilation(found, iterations=2))
        dist.append(ndimage.distance_transform_edt(~found))
    return np.stack(edges), np.stack(dist)


def rel(o, r, mask=None):
    """Relative bias of sums over a mask (all channels)."""
    o, r = f64(o), f64(r)
    if mask is not None:
        o, r = o[mask], r[mask]
    return float(o.sum() / r.sum() - 1)


def contrib(o, r, total, mask=None, blurred=False):
    o, r = f64(o), f64(r)
    e = np.abs(blur(o) - blur(r)) if blurred else np.abs(o - r)
    e = e.sum(-1)
    return float((e if mask is None else e[mask]).sum() / total)


def binned(values, key, bins, mask, o, r):
    out = {}
    for low, high in bins:
        m = mask & (key >= low) & (key < high)
        out[f"{low}-{high}"] = round(rel(o, r, m) * 100, 1) if m.sum() > 100 else None
    return out


def read_exr(path) -> np.ndarray:
    with OpenEXR.File(str(path)) as f:
        return np.asarray(f.channels()["RGBA"].pixels, dtype=np.float32)[..., :3]


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    scene = day_scene(FRAMES)
    light = DayLight(scene)
    a = prepare(scene, light)
    saved = np.load(OUT / "results" / "closed_form_day.npy")[[DAY_FRAMES.index(f) for f in FRAMES]]
    report: dict = {"closed_form_matches_saved": float(np.abs(saved - a["radiance"]).sum() / saved.sum())}
    m = masks_for(scene, light)
    ref = f64(scene.reference)
    total = ref.sum()
    direct = f64(load_images(CYCLES / "volume" / "direct", FRAMES))
    indirect = f64(load_images(CYCLES / "volume" / "indirect", FRAMES))
    ref_parts = {"surfaces": ref - direct - indirect, "single": direct, "multiple": indirect, "fog": direct + indirect}
    cf = {
        "surfaces": a["radiance"] - a["fog"],
        "single": a["single"],
        "multiple": a["fog"] - a["single"],
        "fog": a["fog"],
    }
    seed1 = f64(load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", FRAMES))
    near = ~m["sky"] & (m["distance"] < 5)
    mid = ~m["sky"] & (m["distance"] >= 5) & (m["distance"] < 15)
    far = ~m["sky"] & (m["distance"] >= 15)
    edges_a, edge_dist_a = analyzer_edges(scene)

    # ---------------------------------------------------------------- noise floor
    report["noise_floor"] = errors(seed1, ref) | {
        "mid_share_of_error": contrib(seed1, ref, total, mid & ~edges_a) / errors(seed1, ref)["rel_l1"],
        "mid_share_of_light": float(ref.sum(-1)[mid & ~edges_a].sum() / total),
    }

    # ---------------------------------------------------------------- closed form by light, against noise of each light
    spp1024 = f64(load_images(OUT / "day_spp1024" / "noisy", FRAMES))
    d1024 = f64(load_images(OUT / "day_spp1024" / "volume" / "direct", FRAMES))
    i1024 = f64(load_images(OUT / "day_spp1024" / "volume" / "indirect", FRAMES))
    mc_parts = {"surfaces": spp1024 - d1024 - i1024, "single": d1024, "multiple": i1024}
    # The reference's own noise is that of 1024 spp vs 4096 spp scaled by sqrt((1/4096) / (1/1024 + 1/4096)) = 1/sqrt(5)
    by_light = {}
    for part in ("surfaces", "single", "multiple"):
        noise = (
            contrib(mc_parts[part], ref_parts[part], total),
            contrib(mc_parts[part], ref_parts[part], total, blurred=True),
        )
        by_light[part] = {
            "cf_contribution_raw_pct": 100 * contrib(cf[part], ref_parts[part], total),
            "cf_contribution_blurred_pct": 100 * contrib(cf[part], ref_parts[part], total, blurred=True),
            "cf_signed_bias_pct_of_total": 100 * float((f64(cf[part]) - ref_parts[part]).sum() / total),
            "cf_relative_bias_pct": 100 * rel(cf[part], ref_parts[part]),
            "ref_noise_estimate_raw_pct": 100 * noise[0] / 5**0.5,
            "ref_noise_estimate_blurred_pct": 100 * noise[1] / 5**0.5,
            "share_of_reference_light_pct": 100 * float(ref_parts[part].sum() / total),
        }
    report["closed_form_by_light"] = by_light
    report["closed_form_whole"] = errors(a["radiance"], ref)
    report["closed_form_multiple_relative_bias_by_region"] = {
        name: round(100 * rel(cf["multiple"], ref_parts["multiple"], mask), 1)
        for name, mask in {
            "near": near & ~edges_a,
            "mid": mid & ~edges_a,
            "far": far & ~edges_a,
            "sky": m["sky"] & ~edges_a,
            "objects": m["objects"],
            "horizon": np.abs(m["elevation"]) < 2,
        }.items()
    }
    report["closed_form_single_relative_bias_by_region"] = {
        name: round(100 * rel(cf["single"], ref_parts["single"], mask), 1)
        for name, mask in {
            "sky": m["sky"],
            "ground": m["ground"],
            "objects": m["objects"],
            "near": near,
            "mid": mid,
            "far": far,
        }.items()
    }
    report["closed_form_surfaces_relative_bias_by_region"] = {
        name: round(100 * rel(cf["surfaces"], ref_parts["surfaces"], mask), 1)
        for name, mask in {
            "sky": m["sky"],
            "ground_sunlit": m["ground"] & ~m["ground_shadow"],
            "ground_shadow": m["ground_shadow"],
            "objects": m["objects"],
        }.items()
    }
    # Where the surfaces' blurred error is: shadowed ground vs the rest
    report["closed_form_surfaces_blurred_contribution_pct"] = {
        name: round(100 * contrib(cf["surfaces"], ref_parts["surfaces"], total, mask, blurred=True), 3)
        for name, mask in {
            "all": None,
            "ground_shadow": m["ground_shadow"],
            "objects": m["objects"],
            "sky": m["sky"],
            "ground_sunlit": m["ground"] & ~m["ground_shadow"],
        }.items()
    }

    # ---------------------------------------------------------------- azimuth at fixed elevation
    multiple_cf, multiple_ref = f64(cf["multiple"]).sum(-1), ref_parts["multiple"].sum(-1)
    shafts_free = sun_shaft_free(scene, light, ref)
    az_bins = ((0, 30), (30, 50), (50, 70), (70, 90), (90, 180))
    azimuth = {}
    for low, high in ((2, 8), (8, 15)):
        band = m["sky"] & ~edges_a & (m["elevation"] >= low) & (m["elevation"] < high)
        row = {"bias_pct": {}, "bias_no_shafts_pct": {}, "cf_mean": {}, "cycles_mean": {}, "pixels": {}}
        for x, y in az_bins:
            sel = band & (m["azimuth"] >= x) & (m["azimuth"] < y)
            sel2 = sel & shafts_free
            key = f"{x}-{y}"
            row["pixels"][key] = int(sel.sum())
            if sel.sum() > 100:
                row["bias_pct"][key] = round(100 * (multiple_cf[sel].sum() / multiple_ref[sel].sum() - 1), 1)
                row["cf_mean"][key] = round(float(multiple_cf[sel].mean()), 4)
                row["cycles_mean"][key] = round(float(multiple_ref[sel].mean()), 4)
            if sel2.sum() > 100:
                row["bias_no_shafts_pct"][key] = round(100 * (multiple_cf[sel2].sum() / multiple_ref[sel2].sum() - 1), 1)
        azimuth[f"elevation {low}-{high}"] = row
    report["closed_form_multiple_by_azimuth_sky"] = azimuth
    sun_bins = ((0, 40), (40, 55), (55, 70), (70, 85), (85, 180))
    report["closed_form_multiple_by_sun_angle"] = {
        "sky": binned(None, m["sun_angle"], sun_bins, m["sky"] & ~edges_a, cf["multiple"], ref_parts["multiple"]),
        "surfaces": binned(None, m["sun_angle"], sun_bins, ~m["sky"] & ~edges_a, cf["multiple"], ref_parts["multiple"]),
        "surfaces_without_shadowed_ground": binned(
            None,
            m["sun_angle"],
            sun_bins,
            ~m["sky"] & ~edges_a & ~m["ground_shadow"],
            cf["multiple"],
            ref_parts["multiple"],
        ),
    }
    report["closed_form_single_by_sun_angle"] = {
        "sky": binned(None, m["sun_angle"], sun_bins, m["sky"] & ~edges_a, cf["single"], ref_parts["single"]),
        "surfaces": binned(None, m["sun_angle"], sun_bins, ~m["sky"] & ~edges_a, cf["single"], ref_parts["single"]),
    }

    # ---------------------------------------------------------------- shadowed vs sunlit ground, same distance and sun angle
    shadow = {}
    for name, (lo, hi) in {"0-5": (0, 5), "5-15": (5, 15)}.items():
        band = m["ground"] & (m["distance"] >= lo) & (m["distance"] < hi)
        for angle_lo, angle_hi in ((55, 70), (70, 85), (85, 180)):
            same = band & (m["sun_angle"] >= angle_lo) & (m["sun_angle"] < angle_hi)
            lit, dark = same & ~m["ground_shadow"], same & m["ground_shadow"]
            if lit.sum() > 100 and dark.sum() > 100:
                shadow[f"{name} m, sun angle {angle_lo}-{angle_hi}"] = {
                    "sunlit_pct": round(100 * rel(cf["multiple"], ref_parts["multiple"], lit), 1),
                    "shadowed_pct": round(100 * rel(cf["multiple"], ref_parts["multiple"], dark), 1),
                    "pixels": [int(lit.sum()), int(dark.sum())],
                }
    report["shadowed_vs_sunlit_ground_multiple"] = shadow
    gs = m["ground_shadow"]
    report["ground_shadow"] = {
        "pixels_pct": 100 * float(gs.mean()),
        "light_pct": 100 * float(ref.sum(-1)[gs].sum() / total),
        "local_blurred_pct": 100 * contrib(a["radiance"], ref, ref.sum(-1)[gs].sum(), gs, blurred=True),
        "contribution_pct": 100 * contrib(a["radiance"], ref, total, gs),
        "fog_bias_pct": 100 * rel(cf["fog"], ref_parts["fog"], gs),
        "multiple_bias_pct": 100 * rel(cf["multiple"], ref_parts["multiple"], gs),
        "single_bias_pct": 100 * rel(cf["single"], ref_parts["single"], gs),
        "surfaces_bias_pct": 100 * rel(cf["surfaces"], ref_parts["surfaces"], gs),
    }

    # ---------------------------------------------------------------- the analyzer's silhouette mask
    ground_far = m["ground"] & (m["distance"] > 30)
    report["analyzer_edges"] = {
        "pixels_pct": 100 * float(edges_a.mean()),
        "share_object_pixels": float((edges_a & m["objects"]).sum() / edges_a.sum()),
        "share_ground_pixels": float((edges_a & m["ground"]).sum() / edges_a.sum()),
        "share_sky_pixels": float((edges_a & m["sky"]).sum() / edges_a.sum()),
        "share_ground_beyond_30m": float((edges_a & ground_far).sum() / edges_a.sum()),
        "share_not_near_object_silhouette": float((edges_a & ~m["object_silhouette"]).sum() / edges_a.sum()),
        "fog_share_of_light_in_edges": float(ref_parts["fog"].sum(-1)[edges_a].sum() / ref.sum(-1)[edges_a].sum()),
        "object_silhouette_pixels_pct": 100 * float(m["object_silhouette"].mean()),
    }

    # ---------------------------------------------------------------- methods
    methods = load_methods(scene, a)
    method_report = {}
    for key, (images, parts, frames) in methods.items():
        idx = [FRAMES.index(f) for f in frames]
        o, r = f64(images), ref[idx]
        tot = r.sum()
        entry = errors(o, r)
        entry["analyzer_edges_share_of_l1"] = contrib(o, r, tot, edges_a[idx]) / entry["rel_l1"]
        entry["object_silhouette_share_of_l1"] = contrib(o, r, tot, m["object_silhouette"][idx]) / entry["rel_l1"]
        entry["object_silhouette_share_of_light"] = float(r.sum(-1)[m["object_silhouette"][idx]].sum() / tot)
        regions_ = {
            "near": (near & ~edges_a)[idx],
            "mid": (mid & ~edges_a)[idx],
            "far": (far & ~edges_a)[idx],
            "sky": (m["sky"] & ~edges_a)[idx],
            "edges": edges_a[idx],
            "ground_shadow": gs[idx],
            "objects": m["objects"][idx],
            "horizon": (np.abs(m["elevation"]) < 2)[idx],
        }
        entry["contribution_pct"] = {n: round(100 * contrib(o, r, tot, x), 2) for n, x in regions_.items()}
        entry["local_blurred_pct"] = {
            n: round(100 * contrib(o, r, r.sum(-1)[x].sum(), x, blurred=True), 1) for n, x in regions_.items()
        }
        entry["region_bias_pct"] = {n: round(100 * rel(o, r, x), 1) for n, x in regions_.items()}
        if parts is not None:
            fog_o = f64(parts)
            fog_r = ref_parts["fog"][idx]
            entry["fog_relative_bias_pct"] = round(100 * rel(fog_o, fog_r), 1)
            entry["fog_bias_by_region_pct"] = {n: round(100 * rel(fog_o, fog_r, x), 1) for n, x in regions_.items()}
            surf_o, surf_r = o - fog_o, r - fog_r
            entry["surfaces_relative_bias_pct"] = round(100 * rel(surf_o, surf_r), 1)
            entry["surfaces_bias_by_region_pct"] = {
                n: round(100 * rel(surf_o, surf_r, x), 1) for n, x in regions_.items()
            }
            entry["fog_contribution_pct"] = 100 * contrib(fog_o, fog_r, tot)
            entry["surfaces_contribution_pct"] = 100 * contrib(surf_o, surf_r, tot)
            entry["fog_vs_cycles_single"] = rel(fog_o, ref_parts["single"][idx]) + 1
            entry["fog_bias_by_sun_angle_sky"] = binned(
                None, m["sun_angle"][idx], sun_bins, (m["sky"] & ~edges_a)[idx], fog_o, fog_r
            )
            entry["fog_bias_by_object_edge_distance"] = binned(
                None,
                m["object_edge_distance"][idx],
                ((0, 1.5), (1.5, 3.5), (3.5, 6.5), (6.5, 12.5), (12.5, 1e9)),
                np.ones_like(edges_a[idx]),
                fog_o,
                fog_r,
            )
            entry["fog_bias_by_analyzer_edge_distance"] = binned(
                None,
                edge_dist_a[idx],
                ((0, 1.5), (1.5, 3.5), (3.5, 6.5), (6.5, 12.5), (12.5, 1e9)),
                np.ones_like(edges_a[idx]),
                fog_o,
                fog_r,
            )
        method_report[key] = entry
    report["methods"] = method_report

    # ---------------------------------------------------------------- Unreal's fog against single scattering by sun angle
    unreal_fog = f64(methods["unreal_high"][1])
    sky_mask = m["sky"] & ~edges_a
    report["unreal_fog_by_sun_angle_sky"] = {
        "vs_cycles_total_fog_pct": binned(None, m["sun_angle"], sun_bins, sky_mask, unreal_fog, ref_parts["fog"]),
        "vs_cycles_single_pct": binned(None, m["sun_angle"], sun_bins, sky_mask, unreal_fog, ref_parts["single"]),
        "vs_cf_sun_single_pct": binned(None, m["sun_angle"], sun_bins, sky_mask, unreal_fog, a["single_sun"]),
        "cf_sun_share_of_cycles_fog_pct": binned(
            None, m["sun_angle"], sun_bins, sky_mask, a["single_sun"], ref_parts["fog"]
        ),
        "cf_sky_share_of_cycles_fog_pct": binned(
            None, m["sun_angle"], sun_bins, sky_mask, a["single_sky"], ref_parts["fog"]
        ),
    }
    report["unreal_fog_by_sun_angle_surfaces"] = {
        "vs_cycles_total_fog_pct": binned(
            None, m["sun_angle"], sun_bins, ~m["sky"] & ~edges_a, unreal_fog, ref_parts["fog"]
        ),
        "vs_cycles_single_pct": binned(
            None, m["sun_angle"], sun_bins, ~m["sky"] & ~edges_a, unreal_fog, ref_parts["single"]
        ),
    }
    report["unreal_overall"] = {
        "fog_vs_cycles_fog": rel(unreal_fog, ref_parts["fog"]) + 1,
        "fog_vs_cycles_single": rel(unreal_fog, ref_parts["single"]) + 1,
        "fog_vs_cf_single_sun": rel(unreal_fog, a["single_sun"]) + 1,
        "fog_vs_cf_single_sun_plus_sky": rel(unreal_fog, a["single"]) + 1,
        "cf_single_sun_vs_cycles_single": rel(a["single_sun"], ref_parts["single"]) + 1,
    }

    # ---------------------------------------------------------------- own errors of froxels and caches
    report["own_errors"] = own_errors(a, ref, m, edge_dist_a, ~shafts_free)

    # ---------------------------------------------------------------- Monte Carlo
    report["monte_carlo"] = monte_carlo_checks(ref, ref_parts, m, edges_a, near, mid, far)

    (SAVE / "check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


def sun_shaft_free(scene, light, ref) -> np.ndarray:
    """Pixels whose single-scattered sunlight the objects' shadows change by less than 1% of the reference's light."""
    out = []
    for i in range(len(FRAMES)):
        fog = scene.medium.components[0]
        origin, directions, distance = scene.rays(i)
        t, w = day_nodes(fog, light.beta, origin, directions, distance, 32)
        points = (origin + directions[:, None, :] * t[..., None]).reshape(-1, 3)
        towards = directions[:, None, :].expand(-1, 32, -1).reshape(-1, 3)
        shadowed = by_chunks(lambda p, d: light.sun(p, d), len(points), 2**16, points, towards)
        unshadowed = by_chunks(
            lambda p, d: light.sun(p, d, torch.ones_like(p[:, 0])), len(points), 2**16, points, towards
        )
        change = (
            ((unshadowed - shadowed).reshape(-1, 32, 3).sum(1) * w).numpy().reshape(*scene.depth[i].shape, 3).sum(-1)
        )
        out.append(change < 0.01 * ref[i].sum(-1))
    return np.stack(out)


def load_methods(scene, a) -> dict:
    """Whole images, fog light (or None) and frames of each method."""
    surfaces = a["radiance"] - a["fog"]
    methods = {}
    for name in ("day_spp4", "day_spp16"):
        methods[f"{name}_denoised"] = (load_images(OUT / name / "frames", FRAMES), None, FRAMES)
    for name in ("day_spp16", "day_spp1024"):
        noisy = load_images(OUT / name / "noisy", FRAMES)
        fog = sum(load_images(OUT / name / "volume" / x, FRAMES) for x in ("direct", "indirect"))
        methods[f"{name}_noisy"] = (noisy, fog, FRAMES)
    for name in ("eevee_day_default", "eevee_day_1sample"):
        methods[name] = (
            load_images(OUT / name / "frames", FRAMES),
            load_images(OUT / name / "volume" / "light", FRAMES),
            FRAMES,
        )
    for preset in ("high", "epic"):
        frames = np.stack([read_exr(OUT / "unreal" / "renders" / f"Day_{preset}" / f"{f:04d}.exr") for f in FRAMES])
        clear = np.stack([read_exr(OUT / "unreal" / "renders" / "Day_nofog" / f"{f:04d}.exr") for f in FRAMES])
        transmittance = np.stack([scene.transmittance(k) for k in range(len(FRAMES))])
        methods[f"unreal_{preset}"] = (frames, frames - transmittance * clear, FRAMES)
    methods["closed_form"] = (a["radiance"], a["fog"], FRAMES)
    methods["froxel_8_64"] = (surfaces + a["froxel"], a["froxel"], FRAMES)
    cache = np.load(OUT / "results" / "cache_day_images.npz")
    for key in ("sh_1.0_2", "sun_1.0", "exact"):
        methods[f"cache_{key}"] = ((surfaces[1] + cache[key])[None], cache[key][None], (345,))
    return methods


def own_errors(a, ref, m, edge_dist_a, shafts) -> dict:
    """Froxels and caches against what they approximate, normalized by Cycles' whole light (as the analyzer) and by the
    fog's light of what they approximate (not confounded by the fog's share), by distance to silhouettes and in shafts."""
    out = {}
    cache = np.load(OUT / "results" / "cache_day_images.npz")
    saved_froxel = np.load(OUT / "results" / "froxel_images.npz")["day_8_64_1"]
    out["froxel_345_rerun_vs_saved"] = float(np.abs(a["froxel"][1] - saved_froxel).sum() / saved_froxel.sum())
    k = FRAMES.index(345)
    cases = {
        "froxel_vs_closed_form": (a["froxel"], a["fog"], [0, 1]),
        "sun_1.0_vs_closed_form": (cache["sun_1.0"][None], a["fog"][[k]], [k]),
        "sun_1.0_vs_exact_march": (cache["sun_1.0"][None], cache["exact"][None], [k]),
        "sh_1.0_2_vs_closed_form": (cache["sh_1.0_2"][None], a["fog"][[k]], [k]),
        "sh_1.0_2_vs_exact_march": (cache["sh_1.0_2"][None], cache["exact"][None], [k]),
        "exact_march_vs_closed_form": (cache["exact"][None], a["fog"][[k]], [k]),
    }
    bins = ((0, 1.5), (1.5, 3.5), (3.5, 6.5), (6.5, 12.5), (12.5, 1e9))
    for name, (ours, theirs, idx) in cases.items():
        ours, theirs = f64(ours), f64(theirs)
        whole_ref = ref[idx].sum()
        fog_ref = theirs.sum()
        e = np.abs(ours - theirs).sum(-1)
        eb = np.abs(blur(ours) - blur(theirs)).sum(-1)
        fog_sum = theirs.sum(-1)
        entry = {
            "l1_pct_of_cycles_whole": 100 * float(e.sum() / whole_ref),
            "l1_blurred_pct_of_cycles_whole": 100 * float(eb.sum() / whole_ref),
            "bias_pct_of_cycles_whole": 100 * float((ours - theirs).sum() / whole_ref),
            "l1_pct_of_fog": 100 * float(e.sum() / fog_ref),
            "bias_pct_of_fog": 100 * float((ours - theirs).sum() / fog_ref),
        }
        d = edge_dist_a[idx]
        whole_local, fog_local, share = {}, {}, {}
        for lo, hi in bins:
            sel = (d >= lo) & (d < hi)
            whole_local[f"{lo}-{hi}"] = round(100 * float(eb[sel].sum() / ref[idx].sum(-1)[sel].sum()), 2)
            fog_local[f"{lo}-{hi}"] = round(100 * float(eb[sel].sum() / fog_sum[sel].sum()), 2)
            share[f"{lo}-{hi}"] = round(float(fog_sum[sel].sum() / ref[idx].sum(-1)[sel].sum()), 2)
        entry["edge_profile_local_blurred_of_whole_pct"] = whole_local
        entry["edge_profile_local_blurred_of_fog_pct"] = fog_local
        entry["edge_profile_fog_share"] = share
        sh = shafts[idx]
        entry["shafts_fog_local_pct"] = round(100 * float(e[sh].sum() / fog_sum[sh].sum()), 2)
        entry["outside_shafts_fog_local_pct"] = round(100 * float(e[~sh].sum() / fog_sum[~sh].sum()), 2)
        elev = m["elevation"][idx]
        sky = m["sky"][idx]
        by_elev = {}
        for lo, hi in ((-90, -10), (-10, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 10), (10, 90)):
            for where, mask in (("sky", sky), ("surf", ~sky)):
                sel = mask & (elev >= lo) & (elev < hi)
                if sel.sum() > 100:
                    by_elev[f"{where} {lo}..{hi}"] = round(
                        100 * float((ours - theirs).sum(-1)[sel].sum() / fog_sum[sel].sum()), 1
                    )
        entry["fog_bias_by_elevation_pct"] = by_elev
        objs = m["objects"][idx]
        entry["fog_bias_objects_pct"] = round(100 * float((ours - theirs).sum(-1)[objs].sum() / fog_sum[objs].sum()), 1)
        entry["fog_bias_ground_pct"] = round(
            100 * float((ours - theirs).sum(-1)[m["ground"][idx]].sum() / fog_sum[m["ground"][idx]].sum()), 1
        )
        out[name] = entry
    return out


def monte_carlo_checks(ref, ref_parts, m, edges_a, near, mid, far) -> dict:
    out = {}
    regions_ = {
        "near": near & ~edges_a,
        "mid": mid & ~edges_a,
        "far": far & ~edges_a,
        "sky": m["sky"] & ~edges_a,
        "edges": edges_a,
        "ground_shadow": m["ground_shadow"],
        "objects": m["objects"],
    }
    for name in ("day_spp16", "day_spp1024"):
        noisy = f64(load_images(OUT / name / "noisy", FRAMES))
        direct = f64(load_images(OUT / name / "volume" / "direct", FRAMES))
        indirect = f64(load_images(OUT / name / "volume" / "indirect", FRAMES))
        parts = {"surfaces": noisy - direct - indirect, "single": direct, "multiple": indirect}
        total = ref.sum()
        out[name + "_noisy"] = {
            "whole": errors(noisy, ref),
            "region_bias_pct": {n: round(100 * rel(noisy, ref, x), 2) for n, x in regions_.items()},
            "noise_per_light_relative": {
                p: round(contrib(parts[p], ref_parts[p], total) / float(ref_parts[p].sum() / total), 3) for p in parts
            },
        }
    # Clipping/negatives in denoised output and in the texture-edge mask
    clear = f64(load_images(ROOT / "clear" / "frames", FRAMES)) @ np.array([0.2126, 0.7152, 0.0722])
    texture = []
    for i in range(len(FRAMES)):
        values = np.log(clear[i] + 1e-4)
        found = np.zeros(values.shape, dtype=bool)
        for axis in (0, 1):
            pairs = np.abs(np.diff(values, axis=axis)) > np.log(1.6)
            for pad in ((0, 1), (1, 0)):
                found |= np.pad(pairs, [pad if a == axis else (0, 0) for a in (0, 1)])
        texture.append(ndimage.binary_dilation(found & ~m["sky"][i]) & ~edges_a[i])
    texture = np.stack(texture)
    for name in ("day_spp4", "day_spp16", "day_spp64"):
        den = f64(load_images(OUT / name / "frames", FRAMES))
        total = ref.sum()
        whole = errors(den, ref)
        out[name + "_denoised"] = {
            "whole": whole,
            "texture_pixels_pct": 100 * float(texture.mean()),
            "texture_light_pct": 100 * float(ref.sum(-1)[texture].sum() / total),
            "texture_contribution_raw_pct": 100 * contrib(den, ref, total, texture),
            "texture_contribution_blurred_pct": 100 * contrib(den, ref, total, texture, blurred=True),
            "texture_share_of_raw": contrib(den, ref, total, texture) / whole["rel_l1"],
            "texture_share_of_blurred": contrib(den, ref, total, texture, blurred=True) / whole["rel_l1_blurred"],
            "mid_contribution_blurred_pct": 100 * contrib(den, ref, total, mid & ~edges_a, blurred=True),
            "region_bias_pct": {n: round(100 * rel(den, ref, x), 2) for n, x in regions_.items()},
        }
    # Two seeds of 64 spp, denoised: if the blurred error is deterministic bias, the two agree with each other far more
    # than with the reference
    a64 = f64(load_images(OUT / "day_spp64" / "frames", FRAMES))
    b64 = f64(load_images(OUT / "day_spp64_seed1" / "frames", FRAMES))
    out["day_spp64_denoised_seed0_vs_seed1"] = errors(a64, b64)
    out["day_spp64_denoised_seed1_vs_ref"] = errors(b64, ref)
    # Consecutive frames of 4 spp denoised: same seed across frames or not (correlated noise patterns)
    return out


if __name__ == "__main__":
    main()
