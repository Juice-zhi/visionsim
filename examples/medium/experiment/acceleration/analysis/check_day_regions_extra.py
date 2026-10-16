"""Extra tests of day_regions.py's mechanisms, alongside check_day_regions.py.

1. Ground light in the closed form's multiple-scattering table: the table gathers, as "light scattered once", the
   ground's radiance seen from below too, and scatters it a second time; the separate "ground" source also scatters
   the ground's light once. If the two agree, the ground's singly scattered light is counted twice when both multiple
   scattering and the ground's albedo are on.
2. Sun-visibility cache: visibility of the sun near the ground, in the open, as the 1 m grid stores it at its corners.
3. Order-2 spherical harmonics of the smooth light (sky, ground, multiple scattering): their error against direction,
   at a point in the open and at points in front of objects' faces, and the SH cache's bias by kind of pixel at matched
   elevation and share of the ray's fog inside the cache's box.

Writes ``runs/fog-comparison/acceleration/analysis/check_day_regions/extra.json``.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *
from common import OUT, ROOT, DayLight, day_scene, load_images

torch.set_num_threads(3)  # other jobs share the CPU

from visionsim.medium.render import camera_rays
from visionsim.medium.scattering import (
    _gather_nodes,
    azimuthal_phase,
    depth_above_ground,
    elevation_grid,
    elevation_sources,
    ground_radiance,
    multiple_scattering,
)

SAVE = OUT / "analysis" / "check_day_regions"
DAY_BOX = ((-40.0, 0.0, 0.0), (40.0, 140.0, 8.5))


def ground_double_count(scene) -> dict:
    medium, lighting = scene.medium, scene.lighting
    beta = torch.tensor([float(medium.extinction)], dtype=torch.float64)
    kw = {"dtype": torch.float64}
    table = multiple_scattering(medium, lighting, beta, 3, higher_orders=False)
    no_ground = lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    table0 = multiple_scattering(medium, no_ground, beta, 3, higher_orders=False)
    full = multiple_scattering(medium, lighting, beta, 3, higher_orders=True)
    n_heights, height_scale = table.values.shape[1], table.height_scale
    heights = torch.linspace(0, height_scale, n_heights, **kw)
    depth_ground = depth_above_ground(medium, lighting, beta)
    depth_at = depth_ground * torch.exp(-heights)  # (h,)
    elevations = elevation_grid(table.values.shape[0], **kw)
    # The part of the table's second order that comes from the ground's radiance seen directly from below
    omega, w = _gather_nodes(48, **kw)
    ground = ground_radiance(medium, lighting, beta, 3)  # (3,)
    v = omega[:, None]
    end = torch.where(v > 0, torch.zeros_like(v), torch.exp(-(depth_ground - depth_at[None, :]) / v.abs()))  # (d, h)
    kernel = azimuthal_phase(omega[None, :], elevations[:, None], medium.anisotropy)  # (e, d)
    direct = medium.albedo * torch.einsum("ed,dh,d->eh", kernel, end, w)[..., None] * ground  # (e, h, 3)
    # The separate "ground" source of the closed form at the same elevations and heights
    source = elevation_sources(medium, lighting, beta, 3)["ground"]
    e_grid = elevations[:, None].expand(-1, n_heights).reshape(-1)
    c_grid = depth_at[None, :].expand(len(elevations), -1).reshape(-1, 1, 1)
    ground_once = source(e_grid, c_grid).reshape(len(elevations), n_heights, 3)
    from_ground = table.values - table0.values  # second order due to the ground (direct + its light scattered once)
    out = {}
    for h in (0, 5, 10):
        for mu in (-0.9, -0.5, -0.2, 0.0, 0.2, 0.5):
            e = int(torch.argmin((elevations - mu).abs()))
            key = f"height {float(heights[h]):.1f} falloffs, mu {float(elevations[e]):+.2f}"
            out[key] = {
                "ground_source_once": round(float(ground_once[e, h, 1]), 5),
                "table_part_from_direct_ground": round(float(direct[e, h, 1]), 5),
                "table_second_order_due_to_ground": round(float(from_ground[e, h, 1]), 5),
                "table_full_with_ground": round(float(full.values[e, h, 1]), 5),
            }
    ratio = direct[..., 1] / ground_once[..., 1].clamp_min(1e-12)
    out["ratio_direct_part_to_ground_source_median"] = float(ratio.median())
    out["ratio_direct_part_to_ground_source_range"] = [float(ratio.min()), float(ratio.max())]
    return out


def sun_visibility_near_ground(light) -> dict:
    out = {}
    for x, y in ((-30.0, 5.0), (30.0, 120.0), (0.0, 60.0), (-20.0, 100.0)):
        zs = torch.tensor([0.0, 0.05, 0.25, 0.5, 1.0, 2.0])
        points = torch.stack([torch.full_like(zs, x), torch.full_like(zs, y), zs], dim=-1)
        out[f"({x:.0f}, {y:.0f})"] = [round(float(v), 3) for v in light.sun_visibility(points)]
    # Over the whole grid's lowest two layers: mean visibility at z=0 and z=1 m
    for z in (0.0, 1.0, 2.0):
        xs, ys = torch.meshgrid(torch.arange(-40.0, 40.5, 1.0), torch.arange(0.0, 140.5, 1.0), indexing="ij")
        points = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.full((xs.numel(),), z)], dim=-1)
        out[f"mean visibility of 1 m grid corners at z={z}"] = round(float(light.sun_visibility(points).mean()), 4)
    return out


def sh_directional_error(scene, light) -> dict:
    """Order-2 SH projection of light.smooth at single points, against the exact value, by direction elevation."""
    directions, weights = sphere_quadrature(16, 32)
    basis = sh_basis(directions, 2)
    test_dirs, _ = sphere_quadrature(24, 48)
    test_basis = sh_basis(test_dirs, 2)
    elev = torch.rad2deg(torch.asin(test_dirs[:, 2].clamp(-1, 1)))
    t = scene.transforms[0]
    origin, dirs, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
    distance = scene.depth[0] * scale
    normals = scene.extra["normals"][0]
    objects = (normals[..., 2] <= 0.95) & (load_images(ROOT / "clear" / "depths", (345,))[0, ..., 0] < 1e9)
    points = {"open (0, 60, 0.5)": np.array([0.0, 60.0, 0.5]), "open (-30, 20, 1.5)": np.array([-30.0, 20.0, 1.5])}
    rows, cols = np.nonzero(objects & (distance < 30))
    rng = np.random.default_rng(0)
    for k in rng.choice(len(rows), 4, replace=False):
        r, c = rows[k], cols[k]
        hit = origin + dirs[r, c] * distance[r, c]
        points[f"0.3 m before object pixel ({r},{c}) at {distance[r, c]:.0f} m"] = hit - dirs[r, c] * 0.3
    out = {}
    for name, p in points.items():
        p = torch.as_tensor(p, dtype=DTYPE)
        exact = light.smooth(p[None].expand(len(directions), 3).contiguous(), directions)  # (M, 3)
        coefficients = (basis * weights[:, None]).T @ exact  # (9, 3)
        truth = light.smooth(p[None].expand(len(test_dirs), 3).contiguous(), test_dirs)[:, 1]
        approx = (test_basis @ coefficients)[:, 1]
        row = {}
        for lo, hi in ((-90, -30), (-30, -10), (-10, -2), (-2, 2), (2, 10), (10, 30), (30, 90)):
            sel = (elev >= lo) & (elev < hi)
            row[f"{lo}..{hi}"] = round(100 * float(approx[sel].sum() / truth[sel].sum() - 1), 1)
        out[name] = row
    return out


def sh_cache_by_pixel_kind(scene, light) -> dict:
    """The SH cache's bias against the exact march, for ground, objects and sky pixels at matched elevations, among
    pixels whose fog light mostly comes from inside the cache's box."""
    cache = np.load(OUT / "results" / "cache_day_images.npz")
    sh, exact = cache["sh_1.0_2"].astype(np.float64).sum(-1), cache["exact"].astype(np.float64).sum(-1)
    fog = scene.medium.components[0]
    origin, directions, distance = scene.rays(0)
    tt, _ = day_nodes(fog, light.beta, origin, directions, distance, 48)
    pts = origin + directions[:, None, :] * tt[..., None]
    inside_share = inside(*DAY_BOX, pts.reshape(-1, 3)).reshape(tt.shape).float().mean(1).numpy().reshape(sh.shape)
    t = scene.transforms[0]
    _, dirs, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
    elev = np.degrees(np.arcsin(np.clip(dirs[..., 2], -1, 1)))
    raw = load_images(ROOT / "clear" / "depths", (345,))[0, ..., 0]
    sky = raw >= 1e9
    ground = (scene.extra["normals"][0][..., 2] > 0.95) & ~sky
    objects = ~sky & ~ground
    out = {
        "inside_share_mean": {
            "sky": float(inside_share[sky].mean()),
            "ground": float(inside_share[ground].mean()),
            "objects": float(inside_share[objects].mean()),
        }
    }
    for lo, hi in ((-6, -2), (-2, 0), (0, 2), (2, 6), (6, 20)):
        row = {}
        for kind, mask in (("sky", sky), ("ground", ground), ("objects", objects)):
            sel = mask & (elev >= lo) & (elev < hi) & (inside_share > 0.9)
            if sel.sum() > 50:
                row[kind] = [round(100 * float(sh[sel].sum() / exact[sel].sum() - 1), 1), int(sel.sum())]
        out[f"elevation {lo}..{hi}, >90% inside box"] = row
    return out


def double_count_in_image() -> dict:
    """Bias of the closed form's light scattered more than once against Cycles' volume indirect, as is and with one
    copy of the ground's singly scattered light removed (the closed form's ground term with multiple scattering off,
    times 1 and times the table's higher-order factor 1 / (1 - f) at the ground), by region and angle to the sun."""
    frames = (255, 345)
    scene = day_scene(frames)
    arrays = np.load(SAVE / "arrays.npz")
    no_ms = scene.medium.model_copy(update={"multiple_scattering": False})
    ground_term = (
        np.stack(
            [scene.closed_form(k, medium=no_ms).inscatter.numpy() - arrays["single"][k] for k in range(len(frames))]
        )
        .astype(np.float64)
        .sum(-1)
    )
    multiple = (arrays["fog"] - arrays["single"]).astype(np.float64).sum(-1)
    cycles = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "indirect", frames).astype(np.float64).sum(-1)
    # Higher-order factor of the table at the ground: 1 / (1 - f), f = albedo * mean over directions of 1 - escape
    beta = torch.tensor([float(scene.medium.extinction)], dtype=torch.float64)
    depth_ground = float(depth_above_ground(scene.medium, scene.lighting, beta))
    omega, w = _gather_nodes(48, dtype=torch.float64)
    # At the ground, light going down reaches the ground at once (escapes), light going up crosses the whole fog
    escape = torch.where(omega > 0, torch.exp(-depth_ground / omega.clamp_min(1e-9)), torch.ones_like(omega))
    f = float(scene.medium.albedo * ((1 - escape) * w).sum() / 2)
    raw = load_images(ROOT / "clear" / "depths", frames)[..., 0]
    sky = raw >= 1e9
    ground = (scene.extra["normals"][..., 2] > 0.95) & ~sky
    objects = ~sky & ~ground
    towards = np.asarray(scene.lighting.suns[0].direction, dtype=np.float64)
    towards /= np.linalg.norm(towards)
    angle, distance = [], []
    for i, t in enumerate(scene.transforms):
        _, dirs, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
        angle.append(np.degrees(np.arccos(np.clip(dirs @ towards, -1, 1))))
        distance.append(scene.depth[i] * scale)
    angle, distance = np.stack(angle), np.stack(distance)
    regions = {
        "all": np.ones_like(sky),
        "sky": sky,
        "objects": objects,
        "ground 0-5 m": ground & (distance < 5),
        "ground 5-15 m": ground & (distance >= 5) & (distance < 15),
        "ground >15 m": ground & (distance >= 15),
    }
    for lo, hi in ((0, 40), (40, 55), (55, 70), (70, 85), (85, 180)):
        regions[f"sky, sun angle {lo}-{hi}"] = sky & (angle >= lo) & (angle < hi)
        regions[f"sunlit-or-not ground 0-15 m, sun angle {lo}-{hi}"] = (
            ground & (distance < 15) & (angle >= lo) & (angle < hi)
        )
    out = {
        "higher_order_factor_at_ground": 1 / (1 - f),
        "ground_term_share_of_cycles_indirect": float(ground_term.sum() / cycles.sum()),
    }
    for name, m in regions.items():
        if m.sum() < 100:
            continue
        out[name] = {
            "as_is_pct": round(100 * float(multiple[m].sum() / cycles[m].sum() - 1), 1),
            "minus_one_copy_pct": round(100 * float((multiple - ground_term)[m].sum() / cycles[m].sum() - 1), 1),
            "minus_amplified_copy_pct": round(
                100 * float((multiple - ground_term / (1 - f))[m].sum() / cycles[m].sum() - 1), 1
            ),
            "ground_term_over_cycles_pct": round(100 * float(ground_term[m].sum() / cycles[m].sum()), 1),
        }
    return out


def sun_grid_near_objects() -> dict:
    """Split the 1 m sun grid's error at lit fog points by whether a corner of their cell lies inside an object (it
    sees none of the sky's cells), which interpolation then mixes in as shadow."""
    scene = day_scene((345,))
    light = DayLight(scene)
    low, high = DAY_BOX
    points, shape = grid_points(low, high, 1.0)
    grid = (light.sun_visibility(points) * 255).round().reshape(*shape, 1) / 255
    seen = by_chunks(lambda q: light.sky_visibility(q)[0].float().mean(-1, keepdim=True), len(points), 2**15, points)
    buried = (seen[:, 0] < 0.02).reshape(*shape, 1).float()
    fog = scene.medium.components[0]
    origin, directions, distance = scene.rays(0)
    t, w = day_nodes(fog, light.beta, origin, directions, distance, 64)
    p = (origin + directions[:, None, :] * t[..., None]).reshape(-1, 3)
    weight = w.expand_as(t).reshape(-1)
    keep = inside(low, high, p)
    p, weight = p[keep], weight[keep]
    exact = light.sun_visibility(p)
    diff = trilinear(grid, low, 1.0, p)[:, 0] - exact
    near_buried = trilinear(buried, low, 1.0, p)[:, 0] > 0
    lit = exact > 0.999
    total = weight.sum()
    return {
        "buried_corner_share": float(buried.mean()),
        "lit_points_next_to_buried_corner": {
            "weight_share": float(weight[lit & near_buried].sum() / total),
            "contribution_to_mean": float((diff * weight)[lit & near_buried].sum() / total),
        },
        "lit_points_elsewhere": {
            "weight_share": float(weight[lit & ~near_buried].sum() / total),
            "contribution_to_mean": float((diff * weight)[lit & ~near_buried].sum() / total),
        },
        "shadowed_or_partial_points": float((diff * weight)[~lit].sum() / total),
    }


def sun_grid_points() -> dict:
    """Sun visibility of the 1 m grid (trilinear between corners) against the exact shadow map, at the points where
    the march samples the fog along frame 345's camera rays, by the exact visibility, and the share of the darkening
    that comes from cells with a corner inside an object (corner visibility 0 while the point is lit)."""
    scene = day_scene((345,))
    light = DayLight(scene)
    low, high = DAY_BOX
    points, shape = grid_points(low, high, 1.0)
    grid = (light.sun_visibility(points) * 255).round().reshape(*shape, 1) / 255
    fog = scene.medium.components[0]
    origin, directions, distance = scene.rays(0)
    t, w = day_nodes(fog, light.beta, origin, directions, distance, 64)
    p = (origin + directions[:, None, :] * t[..., None]).reshape(-1, 3)
    weight = w.expand_as(t).reshape(-1)
    keep = inside(low, high, p)
    p, weight = p[keep], weight[keep]
    exact = light.sun_visibility(p)
    cached = trilinear(grid, low, 1.0, p)[:, 0]
    diff = cached - exact
    out = {"weighted_mean_difference": float((diff * weight).sum() / weight.sum())}
    for name, sel in (
        ("exact lit (=1)", exact > 0.999),
        ("exact shadow (=0)", exact < 0.001),
        ("exact partial", (exact >= 0.001) & (exact <= 0.999)),
    ):
        out[name] = {
            "weight_share": float(weight[sel].sum() / weight.sum()),
            "mean_difference": float((diff[sel] * weight[sel]).sum() / weight[sel].sum()),
            "contribution_to_mean": float((diff[sel] * weight[sel]).sum() / weight.sum()),
        }
    return out


def main():
    scene = day_scene((345,))
    light = DayLight(scene)
    report = {
        "double_count_in_image": double_count_in_image(),
        "sun_grid_points": sun_grid_points(),
        "sun_grid_near_objects": sun_grid_near_objects(),
        "ground_double_count": ground_double_count(scene),
        "sun_visibility_near_ground": sun_visibility_near_ground(light),
        "sh_directional_error_pct": sh_directional_error(scene, light),
        "sh_cache_by_pixel_kind": sh_cache_by_pixel_kind(scene, light),
    }
    (SAVE / "extra.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
