"""Independent checks of the findings of ``analysis/cache_learning.py`` (world-space caches and learning), 320x180,
frames 255 and 345, written without reusing that script's code.

Parts (all without arguments, or name some on the command line):

- ``day_sun``: sun-visibility grids against the same marching with the shadow map: error per frame and voxel size,
  the share of frame 345's error in the band of pixels whose rays graze the plane of the green cube's shadow, grids
  shifted by a fraction of a voxel (is the 1 m grid's darkening systematic?), and the share of sample errors near shadow
  edges with a different set of 26 probe directions;
- ``day_sh``: order-4 spherical harmonics near the horizon at the sample points of camera rays, with and without the
  objects' occlusion of the sky, to explain why truncation is larger in images than at open grid corners; per-source
  truncation against elevation at open points;
- ``day_cycles``: whole images of the order-4 cache vs Cycles, and the closed form's fog light split into single
  scattering (Cycles' volume direct) and the rest (volume indirect) by ray elevation;
- ``night_lit``: lit grids, grids shifted by a fraction of a voxel, visibility at corners lying on the shadow planes,
  where in space the 1 m grid's error is, and the phase function's spherical-harmonics truncation by scattering angle;
- ``night_field``: the saved neural-field image vs exact marching; samples in shadow vs lit, near edges vs not;
- ``night_residual``: the residual U-Net at night vs the closed form and Cycles; darkened vs brightened pixels, and how
  the choice of the "dark pixels" mask (noisy reference vs blurred reference vs closed form) changes the conclusion.

Results go to ``runs/fog-comparison/acceleration/analysis/check_cache_learning/check.json``.
"""

from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *

torch.set_num_threads(3)

import numpy as np
import torch
from visionsim.medium import apply_medium

from exp_cache import DAY_BOX, NIGHT_BOX, day_sh_grid

RESULTS = OUT / "results"
SAVE = OUT / "analysis" / "check_cache_learning"
JSON = SAVE / "check.json"
FRAMES = (255, 345)
CUBES = ((-3.0, 10.0, 2.0), (4.0, 18.0, 3.0), (-6.0, 30.0, 4.0), (7.0, 45.0, 6.0))  # demo_scene.py: x, y, size
POLES = tuple((-10.0 if i % 2 else 10.0, float(y)) for i, y in enumerate(range(15, 120, 15)))  # radius 0.2, height 8
LUMA = np.array([0.2126, 0.7152, 0.0722])


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def rel(a, b) -> float:
    return float(a / b) if b else float("nan")


def sdf(points: torch.Tensor) -> torch.Tensor:
    """Signed distance to the nearest cube or pole."""
    best = torch.full((len(points),), math.inf, dtype=points.dtype)
    for x, y, size in CUBES:
        q = (points - torch.tensor([x, y, size / 2], dtype=points.dtype)).abs() - size / 2
        best = torch.minimum(best, q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0))
    for x, y in POLES:
        q = torch.stack([torch.hypot(points[:, 0] - x, points[:, 1] - y) - 0.2, (points[:, 2] - 4).abs() - 4], dim=-1)
        best = torch.minimum(best, q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0))
    return best


# 26 probe directions: the neighbours of a voxel (faces, edges and corners), unlike the analyzer's 14
PROBES = torch.tensor(
    [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)],
    dtype=torch.float32,
)
PROBES = PROBES / PROBES.norm(dim=-1, keepdim=True)


def near_edge(visible, points: torch.Tensor, radius: float) -> torch.Tensor:
    """Whether a probe at the given radius, outside objects and above the ground, is on the other side of visibility
    0.5 than the point."""
    side = visible(points) >= 0.5
    found = torch.zeros(len(points), dtype=torch.bool)
    for chunk in torch.arange(len(points)).split(2**15):
        around = (points[chunk, None, :] + radius * PROBES.to(points.dtype)).reshape(-1, 3)
        flips = (
            ((visible(around) >= 0.5) != side[chunk].repeat_interleave(len(PROBES)))
            & (sdf(around) > 0)
            & (around[:, 2] > 0)
        )
        found[chunk] = flips.reshape(len(chunk), -1).any(dim=1)
    return found


def corner_grid(fn, box, size: float, shift=(0.0, 0.0)) -> tuple[torch.Tensor, tuple]:
    """Values of fn at the corners of a grid of voxels, of shape (nx, ny, nz, c), and its low corner. Unshifted, the
    grid is the box's, as in exp_cache.py; shifted by (sx, sy) in x and y, it is extended by one voxel to cover the
    box."""
    low, high = box
    if any(shift):
        low = (low[0] - size + shift[0], low[1] - size + shift[1], low[2])
        high = (high[0] + size, high[1] + size, high[2])
    points, shape = grid_points(low, high, size)
    values = by_chunks(fn, len(points), 2**18, points)
    return values.reshape(*shape, -1), tuple(low)


def lookup(grid, low, size, box, points, outside: torch.Tensor) -> torch.Tensor:
    """Grid value at points inside the box (trilinear), ``outside`` elsewhere, of shape (n,)."""
    value = outside.clone()
    within = inside(*box, points)
    value[within] = trilinear(grid, low, size, points[within])[:, 0]
    return value


def lum(x: np.ndarray) -> np.ndarray:
    return x @ LUMA


def merge(name: str, part: dict):
    results = json.loads(JSON.read_text()) if JSON.exists() else {}
    results[name] = part
    JSON.write_text(json.dumps(results, indent=1))


# --------------------------------------------------------------------------------------------------------------------
# Day: sun grids


DAY_SUN_GRIDS = {
    "1m": (1.0, (0.0, 0.0)),
    "1m_shift(0.37,0.29)": (1.0, (0.37, 0.29)),
    "1m_shift(0.71,0.53)": (1.0, (0.71, 0.53)),
    "0.5m": (0.5, (0.0, 0.0)),
    "0.5m_shift(0.19,0.13)": (0.5, (0.19, 0.13)),
    "0.25m": (0.25, (0.0, 0.0)),
}


def day_sun_images(scene, light, grids, n=512, chunk=2048) -> dict[str, np.ndarray]:
    images: dict[str, list] = {}
    for i in range(len(scene.frames)):
        origin, directions, distance = scene.rays(i)
        parts: dict[str, list] = {}
        for s in range(0, len(directions), chunk):
            d, l = directions[s : s + chunk], distance[s : s + chunk]
            t, w = day_nodes(light.fog, light.beta, origin, d, l, n)
            points = (origin + d[:, None] * t[..., None]).reshape(-1, 3)
            dirs = d[:, None].expand(-1, n, -1).reshape(-1, 3)
            ones = torch.ones(len(points))
            unit = light.sun(points, dirs, visible=ones) * w.expand(-1, n).reshape(-1, 1)
            vis = {"unshadowed": ones, "exact": light.sun_visibility(points)}
            for name, (grid, low, size) in grids.items():
                vis[name] = lookup(grid, low, size, DAY_BOX, points, ones)
            for name, v in vis.items():
                parts.setdefault(name, []).append((unit * v[:, None]).reshape(len(d), n, 3).sum(1))
        for name, v in parts.items():
            images.setdefault(name, []).append(torch.cat(v).reshape(*scene.depth[i].shape, 3).numpy())
        log("day sun images, frame", scene.frames[i])
    return {k: np.stack(v) for k, v in images.items()}


def day_sun() -> dict:
    scene = day_scene(frames=FRAMES)
    light = DayLight(scene)
    quantized = lambda p: (light.sun_visibility(p) * 255).round()[:, None] / 255
    grids = {}
    for name, (size, shift) in DAY_SUN_GRIDS.items():
        grid, low = corner_grid(quantized, DAY_BOX, size, shift)
        grids[name] = (grid, low, size)
        log("grid", name, tuple(grid.shape))
    images = day_sun_images(scene, light, grids)
    np.savez_compressed(SAVE / "day_sun_images.npz", **images)
    exact = images["exact"]
    shadowed = 1 - exact.mean(-1) / np.maximum(images["unshadowed"].mean(-1), 1e-12)
    fog_345 = float(np.load(RESULTS / "cache_day_images.npz")["exact"].sum())
    out: dict = {
        "sun_share_of_fog_light_345": float(exact[1].sum() / fog_345),
        "shadowed_share_of_sun_light": {
            str(f): float(1 - exact[k].sum() / images["unshadowed"][k].sum()) for k, f in enumerate(FRAMES)
        },
    }
    # The plane of the face of the green cube's shadow that starts at its top-left edge (x = 2.5, z = 3), which
    # contains the sun's direction (0.75, 0.433, 0.5): 0.5 x - 0.75 z + 1 = 0
    normal = np.array([0.5, 0.0, -0.75]) / np.linalg.norm([0.5, 0.0, -0.75])
    bands = {}
    for k in range(2):
        origin, directions, _ = scene.rays(k)
        o = origin.numpy()
        d = directions.reshape(180, 320, 3).numpy()
        bands[k] = (np.abs(d @ normal) < 0.1) & (d[..., 0] > 0)
        out.setdefault("camera_to_plane_m", {})[str(FRAMES[k])] = float(
            (0.5 * o[0] - 0.75 * o[2] + 1) / np.linalg.norm([0.5, 0.75])
        )
    for name in DAY_SUN_GRIDS:
        g = images[name]
        error = np.abs(g - exact)
        entry = {
            "rel_l1_of_sun_light_both": float(error.sum() / exact.sum()),
            "ratio_of_sun_light_both": float(g.sum() / exact.sum()),
            "rel_l1_of_sun_light": {str(f): float(error[k].sum() / exact[k].sum()) for k, f in enumerate(FRAMES)},
            "ratio_of_sun_light": {str(f): float(g[k].sum() / exact[k].sum()) for k, f in enumerate(FRAMES)},
            "rel_l1_of_fog_light_345": float(error[1].sum() / fog_345),
            "no_shadow_pixels_345_local_ratio": rel(g[1][shadowed[1] < 0.01].sum(), exact[1][shadowed[1] < 0.01].sum()),
            "grazing_band": {
                str(f): {
                    "pixels": float(bands[k].mean()),
                    "share_of_frame_abs_error": rel(error[k][bands[k]].sum(), error[k].sum()),
                }
                for k, f in enumerate(FRAMES)
            },
        }
        difference = blur((g[1] - exact[1])[None])[0] @ LUMA / (exact[1] @ LUMA).mean()
        entry["signed_map_345_min_max"] = [float(difference.min()), float(difference.max())]
        out[name] = entry
        log(name, json.dumps(entry))

    # Samples along random rays: share of the absolute error of samples within one voxel of a shadow edge (26 probes)
    # or within one voxel of an object
    generator = torch.Generator().manual_seed(11)
    samples: dict[str, list] = {}
    for k in range(2):
        origin, directions, distance = scene.rays(k)
        select = torch.randperm(len(directions), generator=generator)[:1500]
        d, l = directions[select], distance[select]
        t, w = day_nodes(light.fog, light.beta, origin, d, l, 128)
        points = (origin + d[:, None] * t[..., None]).reshape(-1, 3)
        dirs = d[:, None].expand(-1, 128, -1).reshape(-1, 3)
        unit = (light.sun(points, dirs, visible=torch.ones(len(points))) * w.expand(-1, 128).reshape(-1, 1)).sum(-1)
        exact_v = light.sun_visibility(points)
        samples.setdefault("unit", []).append(unit)
        samples.setdefault("object", []).append(sdf(points))
        for name in ("1m", "0.5m", "0.25m"):
            grid, low, size = grids[name]
            samples.setdefault(f"error_{name}", []).append(
                unit * (lookup(grid, low, size, DAY_BOX, points, torch.ones(len(points))) - exact_v)
            )
            samples.setdefault(f"edge_{name}", []).append(near_edge(light.sun_visibility, points, size))
        log("day samples, frame", FRAMES[k])
    x = {key: torch.cat(v) for key, v in samples.items()}
    split = {}
    for name, size in (("1m", 1.0), ("0.5m", 0.5), ("0.25m", 0.25)):
        e = x[f"error_{name}"].abs()
        objects = x["object"] <= size
        edges = x[f"edge_{name}"] & ~objects
        split[name] = {
            "share_near_edges(within one voxel, not near objects)": rel(e[edges].sum(), e.sum()),
            "share_near_objects(within one voxel)": rel(e[objects].sum(), e.sum()),
            "share_elsewhere": rel(e[~edges & ~objects].sum(), e.sum()),
            "samples_near_edges": float(edges.float().mean()),
        }
    out["sample_split_26_probes"] = split
    return out


# --------------------------------------------------------------------------------------------------------------------
# Day: spherical harmonics near the horizon


def unoccluded(light, points, directions):
    c = light.depth_above(points)
    return sum(light.source(name, directions[:, 2], c) for name in light.tables)


def pointwise_truncation(light, points, view, order, quadrature, occluded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact light towards view directions and its spherical harmonics of an order projected at the point itself (no
    interpolation between corners), with or without the objects' occlusion of the sky, (n, 3) each."""
    directions, weights = quadrature
    m = len(directions)
    basis = sh_basis(directions, order) * weights[:, None]
    exact, rebuilt = [], []
    for chunk in torch.arange(len(points)).split(16):
        p, v = points[chunk], view[chunk]
        flat_p, flat_d = p.repeat_interleave(m, dim=0), directions.repeat(len(p), 1)
        if occluded:
            lit = light.sky_visibility(p)
            values = light.smooth(
                flat_p, flat_d, lit=(lit[0].repeat_interleave(m, dim=0), lit[1].repeat_interleave(m, dim=0))
            )
            exact.append(light.smooth(p, v, lit=lit))
        else:
            values = unoccluded(light, flat_p, flat_d)
            exact.append(unoccluded(light, p, v))
        coefficients = torch.einsum("nmc,mk->nkc", values.reshape(len(p), m, 3), basis)
        rebuilt.append(torch.einsum("nkc,nk->nc", coefficients, sh_basis(v, order)))
    return torch.cat(exact), torch.cat(rebuilt)


def day_sh() -> dict:
    scene = day_scene(frames=FRAMES)
    light = DayLight(scene)
    grid = day_sh_grid(light, 1.0, 4)
    log("sh grid", tuple(grid.shape))
    cache_quadrature = sphere_quadrature(16, 32)
    fine = sphere_quadrature(64, 64)
    out: dict = {}
    # 1. Samples along camera rays of frame 345 by elevation band: the cache, the truncation at the point itself with
    # occlusion, and the same without occlusion (the open-field function at the same height)
    origin, directions, distance = scene.rays(1)
    elevation = torch.rad2deg(torch.asin(directions[:, 2].clamp(-1, 1)))
    generator = torch.Generator().manual_seed(5)
    low, high = DAY_BOX
    for band, (lo, hi) in {"-5..+5deg": (-5, 5), "-20..-10deg": (-20, -10)}.items():
        candidates = ((elevation >= lo) & (elevation < hi)).nonzero()[:, 0]
        select = candidates[torch.randperm(len(candidates), generator=generator)[:120]]
        d, l = directions[select], distance[select]
        t, w = day_nodes(light.fog, light.beta, origin, d, l, 48)
        points = (origin + d[:, None] * t[..., None]).reshape(-1, 3)
        view = d[:, None].expand(-1, 48, -1).reshape(-1, 3)
        weight = w.expand(-1, 48).reshape(-1, 1)
        keep = inside(low, high, points)
        points, view, weight = points[keep], view[keep], weight[keep]
        stored = trilinear(grid, low, 1.0, points).reshape(len(points), -1, 3)
        cache = (stored * sh_basis(view, 4)[..., None]).sum(1)
        exact_o, trunc_o = pointwise_truncation(light, points, view, 4, cache_quadrature, True)
        exact_u, trunc_u = pointwise_truncation(light, points, view, 4, cache_quadrature, False)
        _, trunc_u_fine = pointwise_truncation(light, points, view, 4, fine, False)
        s = lambda x: float((x * weight).sum())
        near = sdf(points) < 3.0
        out[f"frame345_rays_{band}"] = {
            "samples": len(points),
            "occluded_over_unoccluded_light": s(exact_o) / s(exact_u),
            "cache_bias": s(cache) / s(exact_o) - 1,
            "pointwise_truncation_bias_occluded": s(trunc_o) / s(exact_o) - 1,
            "pointwise_truncation_bias_unoccluded(16x32)": s(trunc_u) / s(exact_u) - 1,
            "pointwise_truncation_bias_unoccluded(64x64)": s(trunc_u_fine) / s(exact_u) - 1,
            "share_of_light_within_3m_of_objects": s(exact_o * near[:, None]) / s(exact_o),
            "pointwise_truncation_bias_occluded_within_3m_of_objects": rel(
                s((trunc_o - exact_o) * near[:, None]), s(exact_o * near[:, None])
            ),
            "pointwise_truncation_bias_occluded_farther": rel(
                s((trunc_o - exact_o) * ~near[:, None]), s(exact_o * ~near[:, None])
            ),
            "mean_height_m": float((points[:, 2:3] * weight).sum() / weight.sum()),
        }
        log(band, json.dumps(out[f"frame345_rays_{band}"]))
    # 2. Open field, per source, at several heights: order-4 truncation (64x64 projection) by elevation, as % of the
    # total smooth light at that elevation
    directions_f, weights_f = fine
    basis = sh_basis(directions_f, 4) * weights_f[:, None]
    elevations = torch.tensor([-25.0, -15.0, -7.5, -3.5, -1.0, 1.0, 3.5, 7.5, 15.0, 25.0, 45.0])
    view = torch.stack(
        [torch.deg2rad(elevations).cos(), torch.zeros_like(elevations), torch.deg2rad(elevations).sin()], -1
    )
    for z in (0.5, 1.6, 3.0):
        point = torch.tensor([[-30.0, 70.0, z]])
        c = light.depth_above(point)
        per = {}
        totals = sum(light.source(name, view[:, 2], c.expand(len(view))) for name in light.tables).mean(-1)
        for name in light.tables:
            values = light.source(name, directions_f[:, 2], c.expand(len(directions_f)))  # (m, 3)
            coefficients = basis.T @ values
            rebuilt = sh_basis(view, 4) @ coefficients
            exact = light.source(name, view[:, 2], c.expand(len(view)))
            per[name] = [round(float(v), 4) for v in ((rebuilt - exact).mean(-1) / totals)]
        total_values = unoccluded(light, point.expand(len(directions_f), -1), directions_f)
        rebuilt_total = sh_basis(view, 4) @ (basis.T @ total_values)
        exact_total = unoccluded(light, point.expand(len(view), -1), view)
        per["total"] = [round(float(v), 4) for v in ((rebuilt_total - exact_total).mean(-1) / exact_total.mean(-1))]
        per["exact_total"] = [round(float(v), 4) for v in exact_total.mean(-1)]
        out[f"open_field_z{z:g}m_order4_by_elevation"] = {"elevations_deg": elevations.tolist(), **per}
    return out


# --------------------------------------------------------------------------------------------------------------------
# Day: order-4 cache and closed form vs Cycles


def day_cycles() -> dict:
    scene = day_scene(frames=FRAMES)
    light = DayLight(scene)
    sun = np.load(SAVE / "day_sun_images.npz")["exact"]
    grid = day_sh_grid(light, 1.0, 4)
    low, high = DAY_BOX
    smooth, sh4 = [], []
    for i in range(2):
        origin, directions, distance = scene.rays(i)
        parts = {"exact": [], "sh4": []}
        for s in range(0, len(directions), 1024):
            d, l = directions[s : s + 1024], distance[s : s + 1024]
            t, w = day_nodes(light.fog, light.beta, origin, d, l, 48)
            points = (origin + d[:, None] * t[..., None]).reshape(-1, 3)
            dirs = d[:, None].expand(-1, 48, -1).reshape(-1, 3)
            weight = w.expand(-1, 48).reshape(-1, 1)
            exact = light.smooth(points, dirs)
            within = inside(low, high, points)
            cached = unoccluded(light, points, dirs)
            stored = trilinear(grid, low, 1.0, points[within]).reshape(-1, 25, 3)
            cached[within] = (stored * sh_basis(dirs[within], 4)[..., None]).sum(1)
            parts["exact"].append((exact * weight).reshape(len(d), 48, 3).sum(1))
            parts["sh4"].append((cached * weight).reshape(len(d), 48, 3).sum(1))
        smooth.append(torch.cat(parts["exact"]).reshape(180, 320, 3).numpy())
        sh4.append(torch.cat(parts["sh4"]).reshape(180, 320, 3).numpy())
        log("smooth light, frame", FRAMES[i])
    smooth, sh4 = np.stack(smooth), np.stack(sh4)
    surfaces = np.stack([scene.surfaces(i) for i in range(2)])
    reference = scene.reference
    direct = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "direct", FRAMES)
    indirect = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "indirect", FRAMES)
    exact_whole, cache_whole = surfaces + sun + smooth, surfaces + sun + sh4
    out: dict = {
        "exact_marching_vs_cycles": errors(exact_whole, reference),
        "sh4_vs_cycles": errors(cache_whole, reference),
        "fog_light_exact_vs_cycles": errors(sun + smooth, direct + indirect),
        "fog_light_sh4_vs_cycles": errors(sun + sh4, direct + indirect),
    }
    e, dd = blur(exact_whole - reference), blur(cache_whole - exact_whole)
    change = np.abs(e + dd) - np.abs(e)
    total = float(reference.sum())
    out["blurred_change"] = {
        "total": float(change.sum() / total),
        "towards_cycles": float(change[e * dd < 0].sum() / total),
        "away": float(change[e * dd >= 0].sum() / total),
    }
    # The closed form's fog light split into single scattering and the rest, against Cycles' passes, by elevation
    single_medium = scene.medium.model_copy(update={"multiple_scattering": False})
    no_ground = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    cf_fog, cf_single = [], []
    for i in range(2):
        cf_fog.append(scene.closed_form(i).inscatter.numpy())
        cf_single.append(scene.closed_form(i, medium=single_medium, lighting=no_ground).inscatter.numpy())
        log("closed form components, frame", FRAMES[i])
    cf_fog, cf_single = np.stack(cf_fog), np.stack(cf_single)
    cf_rest = cf_fog - cf_single
    elevation = np.stack(
        [np.degrees(np.arcsin(np.clip(scene.rays(i)[1][:, 2].numpy(), -1, 1))).reshape(180, 320) for i in range(2)]
    )
    table = {}
    for lo, hi in ((-30, -20), (-20, -10), (-10, -5), (-5, -2), (-2, 2), (2, 5), (5, 10), (10, 20)):
        m = np.broadcast_to(((elevation >= lo) & (elevation < hi))[..., None], reference.shape)
        if not m.any():
            continue
        fog = (direct + indirect)[m].sum()
        table[f"{lo}..{hi}deg"] = {
            "cf_fog/cycles_fog": rel(cf_fog[m].sum(), fog),
            "cf_single/cycles_direct": rel(cf_single[m].sum(), direct[m].sum()),
            "cf_rest/cycles_indirect": rel(cf_rest[m].sum(), indirect[m].sum()),
            "cf_single_minus_direct_%fog": 100 * rel(cf_single[m].sum() - direct[m].sum(), fog),
            "cf_rest_minus_indirect_%fog": 100 * rel(cf_rest[m].sum() - indirect[m].sum(), fog),
            "sh4_change_%fog": 100 * rel((sh4 - smooth)[m].sum(), fog),
            "marching_fog/cycles_fog": rel((sun + smooth)[m].sum(), fog),
            "sh4_fog/cycles_fog": rel((sun + sh4)[m].sum(), fog),
        }
    out["by_elevation"] = table
    return out


# --------------------------------------------------------------------------------------------------------------------
# Night: lit grids, spherical harmonics


NIGHT_GRIDS = {
    "1m": (1.0, (0.0, 0.0)),
    "1m_shift(0.37,0.29)": (1.0, (0.37, 0.29)),
    "1m_shift(0.71,0.53)": (1.0, (0.71, 0.53)),
    "1m_shift(0.5,0)": (1.0, (0.5, 0.0)),
    "1m_shift(0.5,0.5)": (1.0, (0.5, 0.5)),
    "0.5m": (0.5, (0.0, 0.0)),
    "0.5m_shift(0.19,0.13)": (0.5, (0.19, 0.13)),
}


def night_render(scene, light, source, frames=(0, 1)) -> np.ndarray:
    images = []
    for i in frames:
        origin, directions, distance = scene.rays(i)
        image = by_chunks(
            lambda d, l: integrate_lamp(light, origin, d, l, source, n=256), len(directions), 4096, directions, distance
        )
        images.append(image.reshape(180, 320, 3).numpy())
    return np.stack(images)


def grid_source(light, grid, low, size):
    def source(points, directions):
        result = light(points, directions)
        within = inside(*NIGHT_BOX, points)
        result[within] = light.falloff(points[within], directions[within]) * trilinear(grid, low, size, points[within])
        return result

    return source


def closest_approach(scene, lamp: np.ndarray, frames=(0, 1)) -> np.ndarray:
    out = []
    for i in frames:
        origin, directions, distance = (x.numpy() for x in scene.rays(i))
        t = np.clip(directions @ (lamp - origin), 0, distance)
        out.append(np.linalg.norm(origin + directions * t[:, None] - lamp, axis=-1).reshape(180, 320))
    return np.stack(out)


def lamp_sample_values(light, origin, d, l, n=128):
    """Points (r*n, 3), weight sigma T dt times falloff and phase (r*n,), lit value (r*n,), cosine (r*n,)."""
    t, w = lamp_nodes(origin, d, l, light.position, n)
    points = origin + d[:, None, :] * t[..., None]
    fog = light.medium.components[0]
    sigma = light.beta * fog.density * torch.exp(-(points[..., 2] - fog.base_height) / fog.falloff)
    tau = height_fog_optical_depth(fog, origin, d[:, None, :].expand_as(points), t) * light.beta
    flat, dirs = points.reshape(-1, 3), d[:, None, :].expand_as(points).reshape(-1, 3)
    base = light.falloff(flat, dirs).mean(-1) * (sigma * torch.exp(-tau) * w).reshape(-1)
    to_light = light.position - flat
    cos = (dirs * to_light).sum(-1) / to_light.norm(dim=-1).clamp_min(1e-12)
    return flat, base, light.lit(flat), cos


def truncated_hg(cos: np.ndarray, g: float, order: int) -> np.ndarray:
    coefficients = [(2 * l + 1) / (4 * math.pi) * g**l for l in range(order + 1)]
    return np.polynomial.legendre.legval(cos, coefficients)


def hg(cos: np.ndarray, g: float) -> np.ndarray:
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cos) ** 1.5)


def night_lit() -> dict:
    scene = night_scene(FRAMES)
    light = NightLight(scene)
    lamp = light.position.numpy()
    exact = night_render(scene, light, light)
    np.save(SAVE / "night_exact.npy", exact)
    approach = closest_approach(scene, lamp)
    regions = {
        f"{lo:g}-{hi:g}m": (approach >= lo) & (approach < hi) for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, np.inf))
    }
    total = float(exact.sum())
    out: dict = {"lamp": lamp.tolist(), "variants": {}}
    grids = {}
    for name, (size, shift) in NIGHT_GRIDS.items():
        grid, low = corner_grid(lambda p: light.lit(p)[:, None], NIGHT_BOX, size, shift)
        grids[name] = (grid, low, size)
        image = night_render(scene, light, grid_source(light, grid, low, size))
        entry = {"total": errors(image, exact)}
        entry["by_closest_approach_to_lamp"] = {
            k: {
                "pixels": float(m.mean()),
                "error_points": 100 * float(np.abs(image - exact)[m].sum() / total),
                "bias_points": 100 * float((image - exact)[m].sum() / total),
            }
            for k, m in regions.items()
        }
        out["variants"][name] = entry
        log(name, json.dumps(entry["total"]))
    # Visibility of the lamp at corners lying on the planes of the shadow that start at the cube's back vertical edges:
    # x - y = -17 (left) and x + y = 25 (right), and 5 cm to either side (shadow side first)
    planes = {
        "left(x-y=-17)": ([(2, 19), (1, 18), (0, 17), (-1, 16)], np.array([1.0, -1.0, 0]) / math.sqrt(2)),
        "right(x+y=25)": ([(6, 19), (7, 18), (8, 17), (9, 16)], np.array([-1.0, -1.0, 0]) / math.sqrt(2)),
    }
    corners = {}
    for name, (xy, toward_shadow) in planes.items():
        rows = []
        for x, y in xy:
            for z in (1.0, 2.0):
                p = torch.tensor([[x, y, z]], dtype=torch.float32)
                offset = torch.tensor(toward_shadow * 0.05, dtype=torch.float32)
                vis = [float(lamp_visibility(light.maps, light.index, q)[0]) for q in (p + offset, p, p - offset)]
                rows.append({"corner": [x, y, z], "visibility(shadow side, on plane, lit side)": vis})
        corners[name] = rows
    out["corners_on_shadow_planes"] = corners

    # Where in space the error of the 1 m grids lies: samples along random rays of both frames
    generator = torch.Generator().manual_seed(7)
    acc: dict[str, list] = {}
    for k in range(2):
        origin, directions, distance = scene.rays(k)
        select = torch.randperm(len(directions), generator=generator)[:3000]
        points, base, lit, _ = lamp_sample_values(light, origin, directions[select], distance[select])
        base = base * (len(directions) / 3000)
        acc.setdefault("points", []).append(points)
        acc.setdefault("light", []).append(base * lit)
        acc.setdefault("visible", []).append(lamp_visibility(light.maps, light.index, points) >= 0.5)
        for name in ("1m", "1m_shift(0.37,0.29)", "1m_shift(0.5,0.5)", "0.5m", "0.5m_shift(0.19,0.13)"):
            grid, low, size = grids[name]
            acc.setdefault(name, []).append(base * (lookup(grid, low, size, NIGHT_BOX, points, lit) - lit))
    x = {key: torch.cat(v) for key, v in acc.items()}
    light_total = float(x["light"].sum())
    by_side = {}
    for name in ("1m", "1m_shift(0.37,0.29)", "1m_shift(0.5,0.5)", "0.5m", "0.5m_shift(0.19,0.13)"):
        e = x[name]
        by_side[name] = {
            "bias_lit_side": float(e[x["visible"]].sum() / light_total),
            "bias_shadow_side": float(e[~x["visible"]].sum() / light_total),
            "abs": float(e.abs().sum() / light_total),
        }
    out["samples_by_side"] = by_side
    # The 1 m cells of space holding most of the 1 m grid's absolute error
    e = x["1m"].abs()
    cells = torch.floor(x["points"]).long()
    keys, inverse = torch.unique(cells, dim=0, return_inverse=True)
    sums = torch.zeros(len(keys)).index_add_(0, inverse, e)
    top = sums.argsort(descending=True)[:8]
    out["top_cells_1m"] = [
        {
            "cell_min_corner": keys[j].tolist(),
            "share_of_abs_error": float(sums[j] / e.sum()),
            "signed_share": float(x["1m"][inverse == j].sum() / e.sum()),
        }
        for j in top
    ]
    # Angle between camera rays passing the cube's back-left edge and the left shadow plane
    o = scene.rays(1)[0].numpy()
    to_edge = np.array([2.5, 19.5, 1.5]) - o
    to_edge /= np.linalg.norm(to_edge)
    out["angle_ray_to_left_shadow_plane_deg_345"] = float(
        np.degrees(np.arcsin(abs(to_edge @ np.array([1, -1, 0]) / math.sqrt(2))))
    )
    out["camera_345"] = o.tolist()

    # Spherical harmonics: share of frame 345's light by scattering angle and the truncated phase function
    origin, directions, distance = scene.rays(1)
    bins = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120, 180])
    light_by_angle = np.zeros(len(bins) - 1)
    truncated = {o: np.zeros(len(bins) - 1) for o in (2, 4, 8)}
    for s in range(0, len(directions), 2048):
        _, base, lit, cos = lamp_sample_values(light, origin, directions[s : s + 2048], distance[s : s + 2048], 256)
        value = (base * lit).double().numpy()
        c = cos.double().numpy()
        index = np.clip(np.digitize(np.degrees(np.arccos(np.clip(c, -1, 1))), bins) - 1, 0, len(bins) - 2)
        light_by_angle += np.bincount(index, value, len(bins) - 1)
        for order in truncated:
            truncated[order] += np.bincount(
                index, value * (truncated_hg(c, light.g, order) / hg(c, light.g) - 1), len(bins) - 1
            )
    out["sh_light_share_by_angle_%"] = dict(
        zip(
            [f"{a}..{b}" for a, b in zip(bins[:-1], bins[1:])],
            (100 * light_by_angle / light_by_angle.sum()).round(2).tolist(),
        )
    )
    out["sh_share_20_90deg_%"] = float(100 * light_by_angle[3:7].sum() / light_by_angle.sum())
    out["sh_signed_error_by_angle_%"] = {
        o: (100 * v / light_by_angle.sum()).round(1).tolist() for o, v in truncated.items()
    }
    angles = np.array([0.0, 5, 10, 15, 25, 37.5, 52.5, 75])
    out["truncated_over_hg"] = {
        str(o): dict(
            zip(
                angles.tolist(),
                (truncated_hg(np.cos(np.radians(angles)), light.g, o) / hg(np.cos(np.radians(angles)), light.g))
                .round(3)
                .tolist(),
            )
        )
        for o in (4, 8)
    }
    # Relative L1 error of the truncated phase function if light came from every direction alike (weighted by the
    # phase function itself), and only from the forward cone (< 20 degrees)
    mu = np.linspace(-1, 1, 200001)
    for order in (4, 8):
        f, t = hg(mu, light.g), truncated_hg(mu, light.g, order)
        forward = mu > math.cos(math.radians(20))
        out.setdefault("phase_truncation_rel_l1", {})[str(order)] = {
            "whole_sphere": float(np.abs(t - f).sum() / f.sum()),
            "forward_cone_20deg": float(np.abs(t - f)[forward].sum() / f[forward].sum()),
            "20..90deg": float(
                np.abs(t - f)[(mu < math.cos(math.radians(20))) & (mu > 0)].sum()
                / f[(mu < math.cos(math.radians(20))) & (mu > 0)].sum()
            ),
        }
    return out


# --------------------------------------------------------------------------------------------------------------------
# Night: neural field


def night_field() -> dict:
    import exp_learning

    scene = night_scene(FRAMES)
    light = NightLight(scene)
    lamp = light.position.numpy()
    exact = np.load(SAVE / "night_exact.npy")[1]
    saved = np.load(RESULTS / "learning_field_10945.npy")
    approach = closest_approach(scene, lamp, (1,))[0]
    total = float(exact.sum())
    out: dict = {"saved_vs_exact_345": errors(saved[None], exact[None])}
    out["by_closest_approach_points"] = {
        f"{lo:g}-{hi:g}m": 100 * float(np.abs(saved - exact)[(approach >= lo) & (approach < hi)].sum() / total)
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, np.inf))
    }
    # The analyzer's retrained weights, checked against the saved image by rendering frame 345 again
    net = exp_learning.Field(6, 64, 3)
    net.load_state_dict(torch.load(OUT / "analysis" / "cache_learning" / "night_field_10945.pt"))
    low, high = (torch.as_tensor(b, dtype=torch.float32) for b in NIGHT_BOX)

    def stored(points):
        value = light.lit(points)
        within = inside(*NIGHT_BOX, points)
        with torch.no_grad():
            value[within] = net((points[within] - low) / (high - low) * 2 - 1)[:, 0]
        return value

    def source(points, directions):
        return light.falloff(points, directions) * stored(points)[:, None]

    rendered = night_render(scene, light, source, (1,))[0]
    out["weights_render_vs_saved"] = errors(rendered[None], saved[None])
    generator = torch.Generator().manual_seed(13)
    origin, directions, distance = scene.rays(1)
    select = torch.randperm(len(directions), generator=generator)[:3000]
    points, base, lit, _ = lamp_sample_values(light, origin, directions[select], distance[select])
    error = base * (stored(points) - lit)
    light_total = float((base * lit).sum())
    shadow = lit < 0.05
    edge = near_edge(lambda p: lamp_visibility(light.maps, light.index, p), points, 0.125)
    lamp_distance = (points - light.position).norm(dim=-1)
    out["samples"] = {
        "bias_shadowed(lit<0.05)": float(error[shadow].sum() / light_total),
        "bias_lit": float(error[~shadow].sum() / light_total),
        "abs": float(error.abs().sum() / light_total),
        "share_abs_within_0.125m_of_edge(26 probes)": rel(error[edge].abs().sum(), error.abs().sum()),
        "samples_within_0.125m_of_edge": float(edge.float().mean()),
        "share_abs_2-4m_from_lamp": rel(
            error[(lamp_distance >= 2) & (lamp_distance < 4)].abs().sum(), error.abs().sum()
        ),
    }
    return out


# --------------------------------------------------------------------------------------------------------------------
# Night: residual network


def night_residual() -> dict:
    scene = night_scene((345,))
    clear = load_images(OUT / "night_clear" / "frames", (345,))
    t = scene.transforms[0]
    medium = scene.medium.model_copy(update={"multiple_scattering": True})
    cf = apply_medium(
        clear[0],
        scene.depth[0],
        t,
        t["transform_matrix"],
        medium,
        scene.lighting,
        occlusion=scene.occlusion,
        dtype=torch.float32,
    ).radiance.numpy()
    residual = np.load(RESULTS / "learning_residual_night.npy")
    reference = load_images(OUT / "night_ms_ref" / "frames", (345,))[0]
    out: dict = {"closed_form": errors(cf[None], reference[None]), "residual": errors(residual[None], reference[None])}
    total = float(reference.sum())
    change = np.abs(residual - reference) - np.abs(cf - reference)
    darker = lum(residual) < lum(cf)
    out["pixels_darkened"] = float(darker.mean())
    out["change_points"] = {
        "total": 100 * float(change.sum() / total),
        "in_darkened_pixels": 100 * float(change[darker].sum() / total),
        "in_brightened_pixels": 100 * float(change[~darker].sum() / total),
    }
    # Change of the absolute error split into the net darkening of each region and the rest (redistribution)
    level = lum(cf)
    edges = np.quantile(level, np.linspace(0, 1, 11))
    deciles = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (level >= lo) & (level <= hi)
        deciles.append(
            {
                "decile": k + 1,
                "residual/cf": rel(residual[m].sum(), cf[m].sum()),
                "cycles/cf": rel(reference[m].sum(), cf[m].sum()),
                "change_points": 100 * float(change[m].sum() / total),
                "share_of_decile_pixels_darkened": float(darker[m].mean()),
            }
        )
    out["deciles_by_closed_form_luminance"] = deciles
    # The "darkest half of pixels": selected on the noisy reference, on the blurred reference, on the closed form
    blurred = blur(reference[None])[0]
    for name, values in (
        ("noisy_reference", lum(reference)),
        ("blurred_reference", lum(blurred)),
        ("closed_form", level),
    ):
        m = values <= np.median(values)
        out.setdefault("darkest_half_local_ratio", {})[name] = {
            "cf/cycles": rel(cf[m].sum(), reference[m].sum()),
            "residual/cycles": rel(residual[m].sum(), reference[m].sum()),
            "cf/blurred_cycles": rel(cf[m].sum(), blurred[m].sum()),
        }
    return out


PARTS = {
    "day_sun": day_sun,
    "day_sh": day_sh,
    "day_cycles": day_cycles,
    "night_lit": night_lit,
    "night_field": night_field,
    "night_residual": night_residual,
}


if __name__ == "__main__":
    SAVE.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(PARTS):
        start = time.perf_counter()
        log("part", name)
        part = PARTS[name]()
        part["seconds"] = time.perf_counter() - start
        merge(name, part)
        log("part", name, "done in", round(part["seconds"]), "s")
