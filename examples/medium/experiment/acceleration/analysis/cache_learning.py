"""Where and why the world-space caches and the learned methods of the acceleration study err (320x180, frames 255
and 345), from the outputs of ``exp_cache.py`` and ``exp_learning.py``, some of which are recomputed here.

What each method is:

- sun grid (day): the sun's visibility stored at the corners of a voxel grid (1, 0.5, 0.25 or 0.125 m) and
  interpolated trilinearly along camera rays, instead of looking up the shadow map at every sample;
- spherical-harmonics sky cache (day): the light of the sky, the ground and multiple scattering that each grid corner
  scatters towards every direction, stored as spherical harmonics of order 1 to 4 (1 m grid);
- lit grid (night): the lamp's transmittance times visibility stored at grid corners, the 1/r² falloff and the phase
  function being exact; spherical harmonics (night): the light reaching each corner as spherical harmonics, which
  the phase function convolves by scaling each order l by g^l;
- neural field (night): an MLP of Fourier features (10945 parameters) storing what the lit grid stores;
- residual network: a U-Net trained on day frames that corrects the closed form's image, applied to the night scene.

The caches use the closed form's lighting, so they are compared with the same marching without a cache ("exact"),
which isolates the error that the cache itself adds. Parts:

- ``day_sun``: error of sun grids against the distance of samples to shadow boundaries and to objects;
- ``day_sh``: error of spherical harmonics against the elevation of view directions, per source of light;
- ``day_sh_cycles``: why the order-4 cache is closer to Cycles than the closed form is;
- ``night_lit``: error of lit grids at the boundary of the cube's shadow wedge, near the lamp, and elsewhere, and
  error of spherical harmonics against the scattering angle;
- ``night_field``: where the neural field errs;
- ``night_residual``: where the residual network makes the night image worse than the closed form alone.

Run from the repository root, e.g. ``python examples/medium/experiment/acceleration/analysis/cache_learning.py
day_sun night_lit`` (all parts without arguments). Results go to ``runs/fog-comparison/acceleration/analysis/
cache_learning/``: ``cache_learning.json`` and PNG maps and plots.
"""

from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *

torch.set_num_threads(3)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from visionsim.medium import apply_medium

from exp_cache import (
    DAY_BOX,
    LAMP_NODES,
    NIGHT_BOX,
    SMOOTH_NODES,
    SUN_NODES,
    day_sh_grid,
    day_sun_grid,
    night_grid,
)

RESULTS = OUT / "results"
SAVE = OUT / "analysis" / "cache_learning"
JSON = SAVE / "cache_learning.json"
FRAMES = (255, 345)
SUN_SIZES = (1.0, 0.5, 0.25, 0.125)
LIT_SIZES = (1.0, 0.5, 0.25, 0.125)
SH_ORDERS = (1, 2, 3, 4)
NIGHT_SH_ORDERS = (1, 2, 3, 4, 6, 8)
RADII = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)

# Objects of the scene (examples/medium/demo_scene.py): cubes (x, y, size) standing on the ground, and poles
CUBES = ((-3.0, 10.0, 2.0), (4.0, 18.0, 3.0), (-6.0, 30.0, 4.0), (7.0, 45.0, 6.0))
POLES = tuple((-10.0 if i % 2 else 10.0, float(y)) for i, y in enumerate(range(15, 120, 15)))
POLE_RADIUS, POLE_HEIGHT = 0.2, 8.0


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def ratio(a, b) -> float:
    return float(a / b) if b else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# Geometry


def object_distance(points: torch.Tensor) -> torch.Tensor:
    """Signed distance of points to the nearest object, negative inside, of shape (n,)."""
    distance = torch.full((len(points),), math.inf, dtype=points.dtype)
    for x, y, size in CUBES:
        center = torch.tensor([x, y, size / 2], dtype=points.dtype)
        q = (points - center).abs() - size / 2
        box = q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0)
        distance = torch.minimum(distance, box)
    for x, y in POLES:
        radial = torch.hypot(points[:, 0] - x, points[:, 1] - y) - POLE_RADIUS
        vertical = (points[:, 2] - POLE_HEIGHT / 2).abs() - POLE_HEIGHT / 2
        q = torch.stack([radial, vertical], dim=-1)
        cylinder = q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0)
        distance = torch.minimum(distance, cylinder)
    return distance


_axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
_diagonals = [[x, y, z] for x in (1, -1) for y in (1, -1) for z in (1, -1)]
PROBES = torch.tensor(_axes + _diagonals, dtype=torch.float32)
PROBES = PROBES / PROBES.norm(dim=-1, keepdim=True)


def boundary_distance(visible, points: torch.Tensor, radii=RADII) -> torch.Tensor:
    """Distance from points to the nearest boundary of a shadow in the fog, of shape (n,): the smallest radius at which
    one of 14 probes around a point (along the axes and the diagonals) is on the other side of visibility 0.5; probes
    inside objects or below the ground are ignored, so that objects' own surfaces don't count. Infinite when no
    boundary is found within the largest radius. Probing along 14 directions overestimates distances by up to ~40%."""
    side = visible(points) >= 0.5
    distance = torch.full((len(points),), math.inf, dtype=points.dtype)
    probes = PROBES.to(points.dtype)
    for radius in radii:
        todo = torch.isinf(distance).nonzero()[:, 0]
        if not len(todo):
            break
        for chunk in todo.split(2**16):
            around = (points[chunk, None, :] + radius * probes).reshape(-1, 3)
            flips = (visible(around) >= 0.5) != side[chunk].repeat_interleave(len(probes))
            flips &= (object_distance(around) > 0) & (around[:, 2] > 0)
            found = flips.reshape(len(chunk), -1).any(dim=1)
            distance[chunk[found]] = radius
    return distance


def ray_geometry(scene, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin (3,), unit directions (h, w, 3) and length (h, w) of the camera rays of frame i."""
    origin, directions, distance = scene.rays(i)
    h, w = scene.depth[i].shape
    return origin.numpy(), directions.reshape(h, w, 3).numpy(), distance.reshape(h, w).numpy()


def band(mask: np.ndarray, width: int) -> np.ndarray:
    """Pixels within ``width`` pixels of the edge of a mask, on both sides."""
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_dilation(mask, structure, width) & ~ndimage.binary_erosion(mask, structure, width)


# --------------------------------------------------------------------------------------------------------------------
# Error decompositions


def regions_table(ours: np.ndarray, ref: np.ndarray, regions: dict[str, np.ndarray], total: float | None = None) -> dict:
    """For each region (masks of shape (n, h, w)): share of pixels, share of the reference's light, share of the total
    error (sum of |ours - ref| over the region / sum of ref over the whole image), signed bias over the whole image's
    light, and the region's own relative error, as is and after a 5x5 box blur (which removes Monte Carlo noise)."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    total = ref.sum() if total is None else total
    error, blurred = np.abs(ours - ref), np.abs(blur(ours) - blur(ref))
    table = {}
    for name, mask in regions.items():
        m = np.broadcast_to(mask[..., None], ours.shape)
        local = ref[m].sum()
        table[name] = {
            "pixels": float(mask.mean()),
            "light": float(local / total),
            "error_share": float(error[m].sum() / total),
            "error_share_blurred": float(blurred[m].sum() / total),
            "bias": float((ours[m] - ref[m]).sum() / total),
            "local_rel_l1": ratio(error[m].sum(), local),
            "local_rel_l1_blurred": ratio(blurred[m].sum(), local),
            "local_ratio": ratio(ours[m].sum(), local),
        }
    return table


def node_table(classes: dict[str, torch.Tensor], error: torch.Tensor, light_total: float) -> dict:
    """Signed and absolute error of samples along rays, before they cancel each other within pixels, per class of
    samples, relative to the total light: share of the absolute error and signed bias."""
    absolute = error.abs().sum()
    return {
        name: {
            "samples": float(mask.float().mean()),
            "abs_error": float(error[mask].abs().sum() / light_total),
            "abs_share": ratio(float(error[mask].abs().sum()), float(absolute)),
            "bias": float(error[mask].sum() / light_total),
        }
        for name, mask in classes.items()
    }


def distance_classes(distance: torch.Tensor, prefix: str, radii=RADII) -> dict[str, torch.Tensor]:
    edges = (0.0, *radii)
    classes = {f"{prefix}<={hi:g}m": (distance > lo) & (distance <= hi) for lo, hi in zip(edges[:-1], edges[1:])}
    classes[f"{prefix}>{radii[-1]:g}m"] = distance > radii[-1]
    return classes


# --------------------------------------------------------------------------------------------------------------------
# Plots


def show(image: np.ndarray, exposure: float) -> np.ndarray:
    """Linear RGB as displayable sRGB-like values."""
    return np.clip(image * exposure, 0, 1) ** (1 / 2.2)


def luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def error_map(ours: np.ndarray, ref: np.ndarray, signed: bool = False) -> np.ndarray:
    """Error of one frame (h, w, 3), blurred 5x5, relative to the mean of the reference."""
    difference = blur((ours - ref)[None])[0]
    value = luminance(difference) / luminance(ref).mean()
    return value if signed else np.abs(value)


def panel(ax, data, title, cmap="magma", vmax=None, signed=False, contour=None):
    if signed:
        vmax = vmax or float(np.percentile(np.abs(data), 99.5)) or 1.0
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    elif cmap is None:
        im = ax.imshow(data)
    else:
        vmax = vmax or float(np.percentile(data, 99.5)) or 1.0
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    if contour is not None:
        ax.contour(contour.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if cmap is not None:
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(SAVE / name, dpi=130)
    plt.close(fig)
    log("saved", SAVE / name)


# --------------------------------------------------------------------------------------------------------------------
# Day: marching along camera rays, as exp_cache.py does


def day_march(scene, light, i: int, n: int, fn, chunk: int, select: torch.Tensor | None = None):
    """Light scattered along camera rays of frame i, given fn(points, directions, weights) -> {name: (r, n, 3)} of
    weighted sample values, summed per ray, as {name: (rays, 3)}. ``select`` restricts the rays to some pixels."""
    origin, directions, distance = scene.rays(i)
    if select is not None:
        directions, distance = directions[select], distance[select]
    sums: dict[str, list[torch.Tensor]] = {}
    for s in range(0, len(directions), chunk):
        d, l = directions[s : s + chunk], distance[s : s + chunk]
        t, w = day_nodes(light.fog, light.beta, origin, d, l, n)
        points = origin + d[:, None] * t[..., None]
        for name, value in fn(points, d[:, None].expand_as(points), w[..., None]).items():
            sums.setdefault(name, []).append(value.sum(dim=1))
    return {name: torch.cat(values) for name, values in sums.items()}


def grid_visibility(grid: torch.Tensor, size: float, points: torch.Tensor) -> torch.Tensor:
    """Sun visibility interpolated in a grid (uint8) inside the cache's box, 1 outside, of shape (n,)."""
    low, high = DAY_BOX
    visible = torch.ones(len(points), dtype=points.dtype)
    within = inside(low, high, points)
    visible[within] = trilinear(grid, low, size, points[within])[:, 0] / 255
    return visible


def day_sun(save_maps: bool = True) -> dict:
    """Sun visibility grids: where along rays their error lies, against the distance to shadow boundaries (in the
    fog) and to objects, whose inside is in shadow and leaks into corners next to their lit faces."""
    scene = day_scene(frames=FRAMES)
    light = DayLight(scene)
    grids = {}
    for size in SUN_SIZES:
        grids[size] = day_sun_grid(light, size)
        log("sun grid", size, tuple(grids[size].shape))

    def samples(points, directions, w):
        flat, dirs = points.reshape(-1, 3), directions.reshape(-1, 3)
        unit = light.sun(flat, dirs, visible=torch.ones(len(flat))).reshape(*points.shape) * w
        values = {"unshadowed": unit, "exact": unit * light.sun_visibility(flat).reshape(*points.shape[:2], 1)}
        for size, grid in grids.items():
            values[f"grid_{size:g}"] = unit * grid_visibility(grid, size, flat).reshape(*points.shape[:2], 1)
        return values

    # Whole images, with the same 512 samples per ray as exp_cache.py
    images = {}
    for i in range(len(FRAMES)):
        for name, value in day_march(scene, light, i, SUN_NODES, samples, 2048).items():
            images.setdefault(name, []).append(value.reshape(*scene.depth[i].shape, 3).numpy())
        log("sun images of frame", FRAMES[i])
    images = {name: np.stack(value) for name, value in images.items()}
    saved = np.load(RESULTS / "cache_day_images.npz")
    exact_fog = saved["exact"]  # sun and smooth light of frame 345, without cache
    fog_total = float(exact_fog.sum())  # fog light of frame 345, the denominator of image errors here
    sun_total = images["exact"][1].sum()
    results: dict = {
        "what": "sun visibility grids vs the same marching with the exact shadow map; errors relative to the exact "
        "fog light (sun + sky/ground/multiple scattering) of frame 345 unless stated",
        "sun_share_of_fog_light_345": float(sun_total / fog_total),
        "shadowed_share_of_sun_light_345": float(1 - sun_total / images["unshadowed"][1].sum()),
        "check_saved_images": {
            f"grid_{s:g}": float(
                np.abs(images[f"grid_{s:g}"][1] - images["exact"][1] + saved["exact"] - saved[f"sun_{s}"]).sum()
                / fog_total
            )
            for s in (1.0, 0.25)
        },
        "images": {},
    }
    # How close each camera is to a shadow boundary: rays leaving it graze that boundary
    cameras = torch.stack([scene.rays(i)[0] for i in range(len(FRAMES))])
    fine_radii = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    results["camera_to_shadow_boundary_m"] = dict(
        zip(map(str, FRAMES), boundary_distance(light.sun_visibility, cameras, fine_radii).tolist())
    )
    # Pixels by the share of their sun light that shadows remove
    shadowed = 1 - images["exact"].mean(-1) / np.maximum(images["unshadowed"].mean(-1), 1e-12)
    pixel_regions = {
        "no_shadow(<1%)": shadowed < 0.01,
        "partly_shadowed(1-50%)": (shadowed >= 0.01) & (shadowed < 0.5),
        "mostly_shadowed(>=50%)": shadowed >= 0.5,
    }
    depths = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    edges = np.stack([band(np.hypot(*np.gradient(np.log(np.clip(d, 0.1, 1e4)))) > 0.1, 2) for d in depths])
    pixel_regions["depth_edges(2px)"] = edges
    for size in SUN_SIZES:
        grid_image = images[f"grid_{size:g}"]
        entry = {
            "both_frames": {
                "rel_l1_of_sun_light": float(np.abs(grid_image - images["exact"]).sum() / images["exact"].sum()),
                "ratio_of_sun_light": float(grid_image.sum() / images["exact"].sum()),
            },
            "frame_345": {
                "rel_l1_of_fog_light": float(np.abs(grid_image[1] - images["exact"][1]).sum() / fog_total),
                "bias_of_fog_light": float((grid_image[1] - images["exact"][1]).sum() / fog_total),
            },
            "per_frame_rel_l1_of_sun_light": {
                str(f): float(np.abs(grid_image[k] - images["exact"][k]).sum() / images["exact"][k].sum())
                for k, f in enumerate(FRAMES)
            },
            "pixel_regions_345": regions_table(
                grid_image[1:], images["exact"][1:], {k: v[1:] for k, v in pixel_regions.items()}, total=fog_total
            ),
        }
        results["images"][f"grid_{size:g}"] = entry

    # Samples along a random subset of rays, classified by their distance to shadow boundaries and to objects
    generator = torch.Generator().manual_seed(0)
    pixels, n = 2500, 128
    parts: dict[str, list[torch.Tensor]] = {}
    for i in range(len(FRAMES)):
        select = torch.randperm(scene.depth[i].size, generator=generator)[:pixels]
        origin, directions, distance = scene.rays(i)
        d, l = directions[select], distance[select]
        t, w = day_nodes(light.fog, light.beta, origin, d, l, n)
        points = (origin + d[:, None] * t[..., None]).reshape(-1, 3)
        dirs = d[:, None].expand(-1, n, -1).reshape(-1, 3)
        # Unshadowed sun light of each sample, scaled so that sums estimate whole-image sums
        unit = light.sun(points, dirs, visible=torch.ones(len(points))) * w.repeat_interleave(n)[:, None]
        unit = unit.sum(-1) * (scene.depth[i].size / pixels)
        exact = light.sun_visibility(points)
        values = {"unit": unit, "exact": exact}
        for size, grid in grids.items():
            values[f"error_{size:g}"] = unit * (grid_visibility(grid, size, points) - exact)
        values["boundary"] = boundary_distance(light.sun_visibility, points)
        values["object"] = object_distance(points)
        for name, value in values.items():
            parts.setdefault(name, []).append(value.reshape(pixels, n))
        log("sun samples of frame", FRAMES[i])
    x = {name: torch.cat(value) for name, value in parts.items()}  # (rays, n) each
    exact_light = float((x["unit"] * x["exact"]).sum())
    results["samples"] = {
        "what": "error of samples along 2500 random rays per frame (128 per ray) before cancelling within pixels; "
        "abs_error and bias relative to the sun light, abs_share = share of the samples' absolute error; samples "
        "> 1 m from objects are split by their distance to the nearest shadow boundary in the fog and by side",
        "sampled_sun_light_vs_image": float(exact_light / images["exact"].sum()),
    }
    distance, near_object, lit_side = x["boundary"].reshape(-1), x["object"].reshape(-1), x["exact"].reshape(-1) >= 0.5
    away = near_object > 1.0
    classes = {"object<=0.25m": near_object <= 0.25, "object_0.25-1m": (near_object > 0.25) & (near_object <= 1.0)}
    classes |= {f"away_from_objects,{k}": v & away for k, v in distance_classes(distance, "shadow_boundary").items()}
    for size in SUN_SIZES:
        sides = {
            f"away,boundary<={size:g}m,lit_side": away & (distance <= size) & lit_side,
            f"away,boundary<={size:g}m,shadow_side": away & (distance <= size) & ~lit_side,
        }
        results["samples"][f"grid_{size:g}"] = node_table(classes | sides, x[f"error_{size:g}"].reshape(-1), exact_light)

    # Rays, by the share of their unshadowed sun light that comes from samples within one voxel of a shadow boundary:
    # rays that graze a boundary stay near it for long and carry most of the error of their pixel
    rays: dict = {
        "what": "rays by the fraction f of their unshadowed sun light within one voxel (h) of a shadow "
        "boundary (> 1 m from objects); error summed along each ray (as in its pixel), relative to the "
        "sampled sun light"
    }
    for size in SUN_SIZES:
        near = ((x["boundary"] <= size) & (x["object"] > 1.0)).float()
        f = (x["unit"] * near).sum(1) / x["unit"].sum(1).clamp_min(1e-30)
        ray_error = x[f"error_{size:g}"].sum(1)
        total_abs = float(ray_error.abs().sum())
        table = {}
        for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 1.01)):
            sel = (f >= lo) & (f < hi)
            table[f"f={lo}-{min(hi, 1)}"] = {
                "rays": float(sel.float().mean()),
                "pixel_abs_error": float(ray_error[sel].abs().sum() / exact_light),
                "share_of_pixel_abs_error": ratio(float(ray_error[sel].abs().sum()), total_abs),
                "bias": float(ray_error[sel].sum() / exact_light),
            }
        rays[f"grid_{size:g}"] = table
    results["rays"] = rays
    if save_maps:
        plot_day_sun(images, results)
    return results


def plot_day_sun(images: dict, results: dict):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.4))
    exact = images["exact"][1]
    panel(axes[0, 0], show(exact, 0.5 / exact.mean()), "sun's light scattered by the fog, frame 345 (exact)", cmap=None)
    vmax = float(np.percentile(np.abs(error_map(images["grid_1"][1], exact)), 99.5))
    for ax, size in zip([axes[0, 1], axes[0, 2], axes[1, 0]], SUN_SIZES[:3]):
        e = results["images"][f"grid_{size:g}"]["frame_345"]
        panel(
            ax,
            error_map(images[f"grid_{size:g}"][1], exact, signed=True),
            f"sun grid {size:g} m - exact, / mean sun light (blurred)\n"
            f"|err| {100 * e['rel_l1_of_fog_light']:.2f}% of fog light, bias {100 * e['bias_of_fog_light']:+.2f}%",
            signed=True,
            vmax=vmax,
        )
    ax = axes[1, 1]
    for size in SUN_SIZES:
        table = results["samples"][f"grid_{size:g}"]
        names = [k for k in table if k.startswith("away_from_objects")]
        ax.plot(range(len(names)), [100 * table[k]["abs_error"] for k in names], "o-", label=f"{size:g} m")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(",")[1].replace("shadow_boundary", "") for n in names], fontsize=6, rotation=30)
    ax.set_ylabel("|error| of samples, % of sun light", fontsize=7)
    ax.set_title("sample error vs distance to the nearest\nshadow boundary (samples > 1 m from objects)", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_yscale("log")
    ax = axes[1, 2]
    rays = results["rays"]
    bins = list(rays["grid_1"])
    for k, size in enumerate(SUN_SIZES):
        table = rays[f"grid_{size:g}"]
        ax.bar(
            np.arange(len(bins)) + 0.2 * (k - 1.5),
            [100 * table[b]["share_of_pixel_abs_error"] for b in bins],
            0.2,
            label=f"{size:g} m",
        )
        for j, b in enumerate(bins):
            ax.text(j + 0.2 * (k - 1.5), 2, f"{100 * table[b]['rays']:.0f}", fontsize=5, ha="center", rotation=90)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, fontsize=7)
    ax.set_xlabel("f = share of a ray's sun light within one voxel of a shadow boundary", fontsize=7)
    ax.set_ylabel("% of pixel |error| (numbers: % of rays)", fontsize=7)
    ax.set_title("rays grazing shadow boundaries carry the error", fontsize=8)
    ax.legend(fontsize=6)
    save(fig, "day_sun_grid.png")


# --------------------------------------------------------------------------------------------------------------------
# Day: spherical harmonics of the sky, the ground and multiple scattering


def unoccluded_smooth(light, points, directions):
    c = light.depth_above(points)
    total = torch.zeros(len(points), 3, dtype=points.dtype)
    for name in light.tables:
        total = total + light.source(name, directions[:, 2], c)
    return total


def sh_from_grid(grid: torch.Tensor, order: int, points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Light that the cache's spherical harmonics of an order give at points towards directions, (n, 3)."""
    low, _ = DAY_BOX
    k = (order + 1) ** 2
    stored = trilinear(grid, low, 1.0, points).reshape(len(points), -1, 3)[:, :k]
    return (stored * sh_basis(directions, order)[..., None]).sum(dim=1)


def sh_images(scene, light, grid) -> dict[str, np.ndarray]:
    """Smooth light (sky, ground, multiple scattering) scattered along camera rays, exact and from the spherical
    harmonics of each order, with the same 48 samples per ray as exp_cache.py, (n, h, w, 3) each."""
    low, high = DAY_BOX

    def samples(points, directions, w):
        flat, dirs = points.reshape(-1, 3), directions.reshape(-1, 3)
        within = inside(low, high, flat)
        exact = light.smooth(flat, dirs)
        values = {"exact": exact.reshape(*points.shape) * w}
        outside = unoccluded_smooth(light, flat[~within], dirs[~within])
        for order in SH_ORDERS:
            cached = torch.empty_like(exact)
            cached[~within] = outside
            cached[within] = sh_from_grid(grid, order, flat[within], dirs[within])
            values[f"sh{order}"] = cached.reshape(*points.shape) * w
        return values

    images: dict = {}
    for i in range(len(scene.frames)):
        for name, value in day_march(scene, light, i, SMOOTH_NODES, samples, 1024).items():
            images.setdefault(name, []).append(value.reshape(*scene.depth[i].shape, 3).numpy())
        log("smooth images of frame", scene.frames[i])
    return {name: np.stack(value) for name, value in images.items()}


ELEVATION_BINS = (-90, -20, -10, -5, -2, 0, 2, 5, 10, 20, 90)


def day_sh(save_maps: bool = True) -> dict:
    """Spherical-harmonics cache of the smooth light: its error against the elevation of view directions, at grid
    corners (truncation alone) and in images (with interpolation), and per source (sky, ground, multiple scattering)
    from an exact projection on a fine set of directions, as exp_sh.py does."""
    scene = day_scene(frames=FRAMES)
    light = DayLight(scene)
    grid = day_sh_grid(light, 1.0, max(SH_ORDERS))
    log("sh grid", tuple(grid.shape))
    results: dict = {"what": "spherical harmonics of the smooth light (sky + ground + multiple scattering), 1 m grid"}

    # 1. At grid corners near the ground (no interpolation), towards directions of every elevation
    low, high = DAY_BOX
    corners, shape = grid_points(low, high, 1.0)
    generator = torch.Generator().manual_seed(1)
    keep = (corners[:, 2] <= 4) & (object_distance(corners) > 0.5)
    chosen = keep.nonzero()[:, 0]
    chosen = chosen[torch.randperm(len(chosen), generator=generator)[:300]]
    points = corners[chosen]
    elevations = torch.arange(-89.0, 90.0, 1.0)
    azimuths = torch.arange(24) * (2 * math.pi / 24)
    el, az = torch.meshgrid(torch.deg2rad(elevations), azimuths, indexing="ij")
    directions = torch.stack([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], dim=-1).reshape(-1, 3)
    m = len(directions)
    exact_sum = torch.zeros(len(elevations), 3, dtype=torch.float64)
    cached_sum = {order: torch.zeros_like(exact_sum) for order in SH_ORDERS}
    cached_abs = {order: torch.zeros_like(exact_sum) for order in SH_ORDERS}
    for chunk in points.split(10):
        flat_points, flat_dirs = chunk.repeat_interleave(m, dim=0), directions.repeat(len(chunk), 1)
        exact = light.smooth(flat_points, flat_dirs).reshape(len(chunk), len(elevations), -1, 3)
        exact_sum += exact.sum(dim=(0, 2)).double()
        for order in SH_ORDERS:
            cached = sh_from_grid(grid, order, flat_points, flat_dirs).reshape(exact.shape)
            cached_sum[order] += cached.sum(dim=(0, 2)).double()
            cached_abs[order] += (cached - exact).abs().sum(dim=(0, 2)).double()
    curves = {
        "elevation_deg": elevations.tolist(),
        "exact_mean": (exact_sum.mean(-1) / (len(points) * len(azimuths))).tolist(),
    }
    for order in SH_ORDERS:
        curves[f"sh{order}_signed"] = ((cached_sum[order] - exact_sum).sum(-1) / exact_sum.sum(-1)).tolist()
        curves[f"sh{order}_abs"] = (cached_abs[order].sum(-1) / exact_sum.sum(-1)).tolist()
    results["corner_truncation_vs_elevation"] = curves
    binned = {}
    for lo, hi in zip(ELEVATION_BINS[:-1], ELEVATION_BINS[1:]):
        sel = (elevations >= lo) & (elevations < hi)
        e = exact_sum[sel].sum()
        binned[f"{lo}..{hi}deg"] = {
            f"sh{o}": {
                "signed": float((cached_sum[o][sel].sum() - e) / e),
                "abs": float(cached_abs[o][sel].sum() / e),
            }
            for o in SH_ORDERS
        }
    results["corner_truncation_binned"] = binned
    log("corner truncation done")

    # 2. Per source, exact projection on 64 x 64 directions (as exp_sh.py, finer in elevation), error per elevation
    # bin relative to the total smooth light of that bin, which shows which source the truncation fails on
    fine, weights = sphere_quadrature(64, 64)
    sources: dict[str, list[torch.Tensor]] = {}
    for chunk in points.split(25):
        n, mf = len(chunk), len(fine)
        lit = light.sky_visibility(chunk)
        c = light.depth_above(chunk)
        flat_points, flat_dirs = chunk.repeat_interleave(mf, dim=0), fine.repeat(n, 1)
        ratios = light.ratios(
            flat_points, flat_dirs, lit[0].repeat_interleave(mf, dim=0), lit[1].repeat_interleave(mf, dim=0)
        )
        for name in light.tables:
            value = light.source(name, flat_dirs[:, 2], c.repeat_interleave(mf)) * ratios[name]
            sources.setdefault(name, []).append(value.reshape(n, mf, 3))
    sources = {name: torch.cat(v).mean(-1) for name, v in sources.items()}  # (points, directions)
    total = sum(sources.values())
    fine_elevation = torch.rad2deg(torch.asin(fine[:, 2].clamp(-1, 1)))
    per_source: dict = {"share_of_light": {name: float(v.sum() / total.sum()) for name, v in sources.items()}}
    for order in SH_ORDERS:
        basis = sh_basis(fine, order)
        entry = {}
        for name, values in sources.items():
            rebuilt = (values @ (basis * weights[:, None])) @ basis.T
            difference = rebuilt - values
            entry[name] = {}
            for lo, hi in zip(ELEVATION_BINS[:-1], ELEVATION_BINS[1:]):
                sel = (fine_elevation >= lo) & (fine_elevation < hi)
                if sel.any():
                    t = total[:, sel].sum()
                    entry[name][f"{lo}..{hi}deg"] = {
                        "signed": float(difference[:, sel].sum() / t),
                        "abs": float(difference[:, sel].abs().sum() / t),
                    }
        per_source[f"sh{order}"] = entry
    results["per_source_exact_projection"] = per_source
    log("per source done")

    # 3. In images: smooth light by elevation of the camera ray, against the exact marching
    images = sh_images(scene, light, grid)
    np.savez_compressed(SAVE / "day_smooth_images.npz", **images)
    elevation = np.stack([np.degrees(np.arcsin(np.clip(ray_geometry(scene, i)[1][..., 2], -1, 1))) for i in range(2)])
    saved = np.load(RESULTS / "cache_day_images.npz")
    fog_total_345 = float(saved["exact"].sum())
    pixel_bins = {
        f"{lo}..{hi}deg": (elevation >= lo) & (elevation < hi) for lo, hi in zip(ELEVATION_BINS[:-1], ELEVATION_BINS[1:])
    }
    pixel_bins = {k: v for k, v in pixel_bins.items() if v.any()}
    results["images"] = {
        "smooth_share_of_fog_light_345": float(images["exact"][1].sum() / fog_total_345),
        "pixel_elevation_range_deg": [float(elevation.min()), float(elevation.max())],
    }
    for order in SH_ORDERS:
        results["images"][f"sh{order}"] = {
            "smooth_rel_l1": float(np.abs(images[f"sh{order}"] - images["exact"]).sum() / images["exact"].sum()),
            "smooth_ratio": float(images[f"sh{order}"].sum() / images["exact"].sum()),
            "by_ray_elevation_relative_to_smooth_light": regions_table(
                images[f"sh{order}"], images["exact"], pixel_bins, total=float(images["exact"].sum())
            ),
        }

    # 4. Truncation vs interpolation, on samples of a subset of rays: the exact light interpolated trilinearly from the
    # 8 corners around each sample, towards the same direction (a cache of infinite order), against the order-4 cache
    results["interpolation_vs_truncation"] = interpolation_split(scene, light, grid, elevation)
    if save_maps:
        plot_day_sh(results)
    return results


def interpolation_split(scene, light, grid, elevation: np.ndarray, pixels: int = 1500) -> dict:
    """Error of the order-4 cache split into interpolation between corners (exact light at the 8 corners around each
    sample, interpolated) and truncation (the rest), per elevation bin of the camera rays, relative to the bin's
    exact smooth light, on 48 samples along random rays of both frames."""
    low, high = DAY_BOX
    offsets = torch.tensor([[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=DTYPE)
    generator = torch.Generator().manual_seed(3)
    sums: dict[str, list[torch.Tensor]] = {}
    ray_elevations = []
    for i in range(len(scene.frames)):
        select = torch.randperm(scene.depth[i].size, generator=generator)[:pixels]
        ray_elevations.append(torch.as_tensor(elevation[i].reshape(-1))[select])

        def samples(points, directions, w):
            flat, dirs = points.reshape(-1, 3), directions.reshape(-1, 3)
            exact = light.smooth(flat, dirs)
            interpolated, cached = exact.clone(), exact.clone()
            within = inside(low, high, flat)
            p, d = flat[within], dirs[within]
            f = ((p - torch.tensor(low, dtype=p.dtype)) / 1.0).clamp_min(0)
            base = f.floor()
            corners = (torch.tensor(low, dtype=p.dtype) + (base[:, None] + offsets)).reshape(-1, 3)
            weights = torch.prod(torch.where(offsets.bool(), (f - base)[:, None], 1 - (f - base)[:, None]), dim=-1)
            at_corners = light.smooth(corners, d.repeat_interleave(8, dim=0)).reshape(len(p), 8, 3)
            interpolated[within] = (at_corners * weights[..., None]).sum(dim=1)
            cached[within] = sh_from_grid(grid, 4, p, d)
            shape = points.shape
            return {
                "exact": exact.reshape(shape) * w,
                "interpolated": interpolated.reshape(shape) * w,
                "sh4": cached.reshape(shape) * w,
            }

        for name, value in day_march(scene, light, i, SMOOTH_NODES, samples, 256, select=select).items():
            sums.setdefault(name, []).append(value)
        log("interpolation split of frame", scene.frames[i])
    x = {name: torch.cat(v).double() for name, v in sums.items()}
    e = torch.cat(ray_elevations)
    table = {}
    bins = {"all": torch.ones_like(e, dtype=torch.bool)}
    bins |= {f"{lo}..{hi}deg": (e >= lo) & (e < hi) for lo, hi in zip(ELEVATION_BINS[:-1], ELEVATION_BINS[1:])}
    for name, sel in bins.items():
        if not sel.any():
            continue
        exact = x["exact"][sel].sum()
        table[name] = {
            "light_share": float(exact / x["exact"].sum()),
            "interpolation_bias": float((x["interpolated"][sel].sum() - exact) / exact),
            "interpolation_rel_l1": float((x["interpolated"][sel] - x["exact"][sel]).abs().sum() / exact),
            "sh4_bias": float((x["sh4"][sel].sum() - exact) / exact),
            "sh4_rel_l1": float((x["sh4"][sel] - x["exact"][sel]).abs().sum() / exact),
            "truncation_bias(sh4-interpolated)": float((x["sh4"][sel].sum() - x["interpolated"][sel].sum()) / exact),
        }
    return table


def plot_day_sh(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    curves = results["corner_truncation_vs_elevation"]
    e = np.array(curves["elevation_deg"])
    ax = axes[0]
    for order in SH_ORDERS:
        ax.plot(e, 100 * np.array(curves[f"sh{order}_signed"]), label=f"order {order}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(-25, 18, color="0.9", zorder=-1, label="camera rays")
    ax.set_xlabel("elevation of the view direction (deg)")
    ax.set_ylabel("cache - exact, % (azimuth-averaged)")
    ax.set_title("SH truncation at grid corners (z<=4 m)", fontsize=9)
    ax.set_ylim(-12, 12)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(e, curves["exact_mean"], "k")
    ax.set_xlabel("elevation of the view direction (deg)")
    ax.set_ylabel("exact smooth light (per unit optical depth)")
    ax.set_title("the light the cache must represent: a notch at the horizon", fontsize=9)
    ax = axes[2]
    per = results["per_source_exact_projection"]["sh4"]
    bins = list(per["sky"])
    x = np.arange(len(bins))
    for k, name in enumerate(per):
        ax.bar(x + 0.27 * (k - 1), [100 * per[name][b]["signed"] for b in bins], 0.27, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, fontsize=6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("signed error, % of the smooth light in the bin")
    ax.set_title("order 4, per source (exact projection)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "day_sh_elevation.png")


def day_sh_cycles(save_maps: bool = True) -> dict:
    """Why the order-4 cache's whole images are closer to Cycles (7.5%) than the closed form's (8.2%): where its
    truncation moves the image towards or away from Cycles, against the closed form's own error there."""
    scene = day_scene(frames=FRAMES)
    light = DayLight(scene)
    path = SAVE / "day_smooth_images.npz"
    if path.exists():
        images = dict(np.load(path))
    else:
        images = sh_images(scene, light, day_sh_grid(light, 1.0, max(SH_ORDERS)))
        np.savez_compressed(path, **images)

    def sun_samples(points, directions, w):
        flat, dirs = points.reshape(-1, 3), directions.reshape(-1, 3)
        return {"sun": light.sun(flat, dirs).reshape(*points.shape) * w}

    sun = np.stack(
        [day_march(scene, light, i, SUN_NODES, sun_samples, 2048)["sun"].reshape(180, 320, 3).numpy() for i in range(2)]
    )
    surfaces = np.stack([scene.surfaces(i) for i in range(2)])
    reference = scene.reference
    cycles_fog = sum(load_images(ROOT / "ms" / "cycles_ref" / "volume" / p, FRAMES) for p in ("direct", "indirect"))
    closed_form = np.load(RESULTS / "closed_form_day.npy")[[DAY_FRAMES.index(f) for f in FRAMES]]
    exact = surfaces + sun + images["exact"]
    results: dict = {
        "what": "whole images (surfaces + sun + smooth light); e = exact marching - Cycles, d = order-4 cache - exact; "
        "all shares relative to the sum of Cycles' whole images of frames 255 and 345",
        "totals": {
            "closed_form_vs_cycles": errors(closed_form, reference),
            "exact_marching_vs_cycles": errors(exact, reference),
            "exact_marching_vs_closed_form": errors(exact, closed_form),
        },
    }
    for order in SH_ORDERS:
        results["totals"][f"sh{order}_vs_cycles"] = errors(exact - images["exact"] + images[f"sh{order}"], reference)
    # The same split into the light of surfaces and the fog's light, against Cycles' passes (fog light = volume
    # direct + indirect, surfaces = the rest)
    results["components_vs_cycles"] = {
        "fog_light_exact": errors(sun + images["exact"], cycles_fog),
        "fog_light_sh4": errors(sun + images["sh4"], cycles_fog),
        "surfaces": errors(surfaces, reference - cycles_fog),
        "fog_light_share_of_cycles": float(cycles_fog.sum() / reference.sum()),
    }
    cached = exact - images["exact"] + images["sh4"]
    e, d = blur(exact - reference), blur(cached - exact)
    total = float(reference.sum())
    change = np.abs(e + d) - np.abs(e)
    corrective = (e * d) < 0
    results["blurred_change_of_abs_error"] = {
        "total": float(change.sum() / total),
        "where_cache_moves_towards_cycles": float(change[corrective].sum() / total),
        "where_cache_moves_away": float(change[~corrective].sum() / total),
        "cache_darker_where_exact_too_bright": float(change[(d < 0) & (e > 0)].sum() / total),
        "cache_brighter_where_exact_too_dark": float(change[(d > 0) & (e < 0)].sum() / total),
        "cache_darker_where_exact_too_dark": float(change[(d < 0) & (e < 0)].sum() / total),
        "cache_brighter_where_exact_too_bright": float(change[(d > 0) & (e > 0)].sum() / total),
        "light_moved_darker": float(d[d < 0].sum() / total),
        "light_moved_brighter": float(d[d > 0].sum() / total),
    }
    # Regions
    depths = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    sky = depths >= 1e9
    normals = scene.extra["normals"]
    ground = (normals[..., 2] > 0.95) & ~sky
    geometry = [ray_geometry(scene, i) for i in range(2)]
    distance = np.stack([g[2] for g in geometry])
    elevation = np.degrees(np.arcsin(np.clip(np.stack([g[1][..., 2] for g in geometry]), -1, 1)))
    regions = {
        "sky": sky,
        "ground_0-5m": ground & (distance < 5),
        "ground_5-15m": ground & (distance >= 5) & (distance < 15),
        "ground_>15m": ground & (distance >= 15),
        "objects": ~sky & ~ground,
    }
    elevation_regions = {
        f"ray_elevation_{lo}..{hi}deg": (elevation >= lo) & (elevation < hi)
        for lo, hi in zip(ELEVATION_BINS[:-1], ELEVATION_BINS[1:])
        if ((elevation >= lo) & (elevation < hi)).any()
    }
    table = {}
    for name, mask in (regions | elevation_regions).items():
        m = np.broadcast_to(mask[..., None], e.shape)
        local = float(reference[m].sum())
        table[name] = {
            "pixels": float(mask.mean()),
            "exact_bias": float(e[m].sum() / total),
            "exact_fog_light_bias": float((sun + images["exact"] - cycles_fog)[m].sum() / total),
            "exact_surfaces_bias": float((surfaces - reference + cycles_fog)[m].sum() / total),
            "exact_abs_error": float(np.abs(e[m]).sum() / total),
            "cache_change": float(d[m].sum() / total),
            "change_of_abs_error": float(change[m].sum() / total),
            "local_exact_ratio": ratio(float((exact)[m].sum()), local),
            "local_cache_ratio": ratio(float(cached[m].sum()), local),
            "fog_light_ratio_exact_vs_cycles": ratio(
                float((sun + images["exact"])[m].sum()), float(cycles_fog[m].sum())
            ),
            "fog_light_ratio_cache_vs_cycles": ratio(float((sun + images["sh4"])[m].sum()), float(cycles_fog[m].sum())),
        }
    results["regions"] = table
    if save_maps:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        ref = reference[1]
        panel(axes[0, 0], show(ref, 0.6 / ref.mean()), "Cycles reference, frame 345", cmap=None)
        signed_e = luminance(e[1]) / luminance(ref).mean()
        signed_d = luminance(d[1]) / luminance(ref).mean()
        vmax = float(np.percentile(np.abs(signed_e), 99))
        panel(axes[0, 1], signed_e, "exact marching (= closed form) - Cycles, blurred, / mean", signed=True, vmax=vmax)
        panel(axes[1, 0], signed_d, "order-4 SH cache - exact, blurred, / mean", signed=True, vmax=vmax)
        panel(
            axes[1, 1],
            (np.abs(e[1] + d[1]) - np.abs(e[1])).mean(-1) / luminance(ref).mean(),
            "change of |error| vs Cycles with the cache (blue: closer)",
            signed=True,
            vmax=vmax,
        )
        save(fig, "day_sh4_vs_cycles.png")
    return results


# --------------------------------------------------------------------------------------------------------------------
# Night: the lamp's light, sample by sample


def lamp_samples(light, origin, directions, distance, n: int = LAMP_NODES) -> dict[str, torch.Tensor]:
    """Samples along rays as in integrate_lamp: points (r, n, 3), weight σ T dt times the lamp's falloff and phase
    function (r, n, 3), transmittance towards the lamp and visibility (r, n), cosine of the scattering angle (r, n)."""
    t, w = lamp_nodes(origin, directions, distance, light.position, n)
    points = origin + directions[:, None, :] * t[..., None]
    fog = light.medium.components[0]
    sigma = light.beta * fog.density * torch.exp(-(points[..., 2] - fog.base_height) / fog.falloff)
    tau = height_fog_optical_depth(fog, origin, directions[:, None, :].expand_as(points), t) * light.beta
    flat, dirs = points.reshape(-1, 3), directions[:, None, :].expand_as(points).reshape(-1, 3)
    base = light.falloff(flat, dirs) * (sigma * torch.exp(-tau) * w).reshape(-1, 1)
    to_light = light.position - flat
    r = to_light.norm(dim=-1).clamp_min(1e-6)
    depth = relative_optical_depth(light.medium, flat, to_light / r[:, None], r) * light.beta
    shape = t.shape
    return {
        "points": points,
        "base": base.reshape(*shape, 3),
        "transmittance": torch.exp(-depth).reshape(shape),
        "visible": lamp_visibility(light.maps, light.index, flat).reshape(shape),
        "cos": ((dirs * to_light).sum(-1) / r).reshape(shape),
        "lamp_distance": r.reshape(shape),
    }


def lit_from_grid(grid, size, points: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """What the lit grid gives at points (r, n, 3): interpolated inside the box, the exact value outside, (r, n)."""
    low, high = NIGHT_BOX
    flat = points.reshape(-1, 3)
    value = exact.reshape(-1).clone()
    within = inside(low, high, flat)
    value[within] = trilinear(grid, low, size, flat[within])[:, 0]
    return value.reshape(points.shape[:-1])


def legendre(x: torch.Tensor, order: int) -> list[torch.Tensor]:
    p = [torch.ones_like(x), x]
    for l in range(1, order):
        p.append(((2 * l + 1) * x * p[l] - l * p[l - 1]) / (l + 1))
    return p[: order + 1]


def truncated_phase(cos: torch.Tensor, g: float, order: int) -> torch.Tensor:
    """Henyey-Greenstein phase function as spherical harmonics up to an order: Σ (2l+1)/4π g^l P_l(cos θ)."""
    return sum((2 * l + 1) / (4 * math.pi) * g**l * p for l, p in enumerate(legendre(cos, order)))


ANGLE_BINS = (0, 5, 10, 20, 30, 45, 60, 90, 120, 180)


def night_images(scene, light, grids: dict, field=None) -> tuple[dict, dict]:
    """Night fog light of each frame, exact, without shadows, with each lit grid, with the phase function truncated
    to each order of spherical harmonics, and with the neural field if given; plus sums over samples of the error of
    grids per class of samples (distance to the lamp), and of the truncation per scattering angle."""
    images: dict = {}
    by_angle = {"exact": np.zeros(len(ANGLE_BINS) - 1)}
    by_angle |= {f"sh{o}": np.zeros(len(ANGLE_BINS) - 1) for o in NIGHT_SH_ORDERS}
    by_angle |= {f"sh{o}_abs": np.zeros(len(ANGLE_BINS) - 1) for o in NIGHT_SH_ORDERS}
    angle_edges = torch.cos(torch.deg2rad(torch.tensor(ANGLE_BINS, dtype=torch.float32)))
    for i in range(len(scene.frames)):
        origin, directions, distance = scene.rays(i)
        sums: dict[str, list[torch.Tensor]] = {}
        for s in range(0, len(directions), 2048):
            d, l = directions[s : s + 2048], distance[s : s + 2048]
            x = lamp_samples(light, origin, d, l)
            lit = x["transmittance"] * x["visible"]
            exact = x["base"] * lit[..., None]
            values = {"exact": exact, "unshadowed": x["base"] * x["transmittance"][..., None]}
            for size, grid in grids.items():
                values[f"lit_{size:g}"] = x["base"] * lit_from_grid(grid, size, x["points"], lit)[..., None]
            phase = henyey_greenstein(x["cos"], light.g)
            # The angle between the ray and the direction towards the lamp, from 0 (looking at it) to 180 degrees
            bins = torch.bucketize(-x["cos"], -angle_edges[1:-1]).reshape(-1)
            if i == 1:
                by_angle["exact"] += np.bincount(
                    bins.numpy(), exact.mean(-1).reshape(-1).double().numpy(), len(ANGLE_BINS) - 1
                )
            for order in NIGHT_SH_ORDERS:
                truncated = exact * (truncated_phase(x["cos"], light.g, order) / phase)[..., None]
                values[f"sh{order}"] = truncated
                if i == 1:
                    difference = (truncated - exact).mean(-1).reshape(-1).double().numpy()
                    by_angle[f"sh{order}"] += np.bincount(bins.numpy(), difference, len(ANGLE_BINS) - 1)
                    by_angle[f"sh{order}_abs"] += np.bincount(bins.numpy(), np.abs(difference), len(ANGLE_BINS) - 1)
            if field is not None:
                values["field"] = x["base"] * field(x["points"])[..., None]
            for name, value in values.items():
                sums.setdefault(name, []).append(value.sum(dim=1))
        for name, value in sums.items():
            images.setdefault(name, []).append(torch.cat(value).reshape(*scene.depth[i].shape, 3).numpy())
        log("night images of frame", scene.frames[i])
    return {name: np.stack(v) for name, v in images.items()}, by_angle


def closest_to_lamp(scene, lamp: np.ndarray) -> np.ndarray:
    """Distance from the lamp to the nearest point of each camera ray (up to its surface), (n, h, w)."""
    closest = []
    for i in range(len(scene.frames)):
        origin, directions, distance = ray_geometry(scene, i)
        t = np.clip(directions @ (lamp - origin), 0, distance)
        closest.append(np.linalg.norm(origin + directions * t[..., None] - lamp, axis=-1))
    return np.stack(closest)


def approach_regions(scene, lamp: np.ndarray) -> dict[str, np.ndarray]:
    """Pixels by how close their ray passes to the lamp."""
    closest = closest_to_lamp(scene, lamp)
    edges = (0, 1.5, 3, 6, np.inf)
    return {
        f"ray_passes_lamp_at_{lo:g}-{hi:g}m": (closest >= lo) & (closest < hi) for lo, hi in zip(edges[:-1], edges[1:])
    }


def night_regions(scene, images: dict, lamp: np.ndarray) -> dict[str, np.ndarray]:
    """Exclusive regions of night images, (n, h, w): the band of 3 pixels around the edge of the shadow wedge (the
    pixels where shadows remove more than half of the fog light); rays passing within 1.5 m of the lamp; the rest of
    the mostly shadowed pixels; and the rest, mostly lit."""
    removed = 1 - images["exact"].mean(-1) / np.maximum(images["unshadowed"].mean(-1), 1e-12)
    wedge = removed > 0.5
    edge = np.stack([band(w, 3) for w in wedge])
    near_lamp = closest_to_lamp(scene, lamp) < 1.5
    return {
        "wedge_edge(3px)": edge,
        "near_lamp(<1.5m)": near_lamp & ~edge,
        "inside_wedge": wedge & ~edge & ~near_lamp,
        "lit_elsewhere": ~wedge & ~edge & ~near_lamp,
    }


def night_lit(save_maps: bool = True) -> dict:
    """Lit grids (transmittance times visibility at corners) against voxel size: at the edge of the cube's shadow
    wedge, near the lamp, and elsewhere; and per sample, against the distance to shadow boundaries in the fog. Also
    spherical harmonics, whose phase function is truncated, against the scattering angle."""
    scene = night_scene(FRAMES)
    light = NightLight(scene)
    grids = {size: night_grid(light, size, "lit") for size in LIT_SIZES}
    images, by_angle = night_images(scene, light, grids)
    np.savez_compressed(SAVE / "night_images.npz", **images)
    exact = images["exact"]
    total = float(exact.sum())
    saved = np.load(RESULTS / "cache_night_images.npz")
    regions = night_regions(scene, images, light.position.numpy())
    results: dict = {
        "what": "night fog light (single scattering); caches vs the same marching without cache ('exact'); error "
        "shares relative to the exact fog light of frames 255 and 345",
        "check_saved_exact_345": float(np.abs(exact[1] - saved["exact"]).sum() / saved["exact"].sum()),
        "check_saved_lit_1": float(np.abs(images["lit_1"][1] - saved["lit_1.0_0"]).sum() / saved["exact"].sum()),
        "shadow_removes": float(1 - total / images["unshadowed"].sum()),
        "lit": {},
    }
    approach = approach_regions(scene, light.position.numpy())
    for size in LIT_SIZES:
        name = f"lit_{size:g}"
        results["lit"][name] = {
            "total": errors(images[name], exact),
            "regions": regions_table(images[name], exact, regions),
            "by_ray_distance_to_lamp": regions_table(images[name], exact, approach),
        }

    # Samples along random rays: error of the grids against the distance to shadow boundaries and to the lamp
    generator = torch.Generator().manual_seed(0)
    pixels = 2000
    errors_by_size = {size: [] for size in LIT_SIZES}
    distances, lamp_distances, object_distances, exact_values, sides = [], [], [], [], []
    visible = lambda p: lamp_visibility(light.maps, light.index, p)
    for i in range(len(FRAMES)):
        select = torch.randperm(scene.depth[i].size, generator=generator)[:pixels]
        origin, directions, distance = scene.rays(i)
        x = lamp_samples(light, origin, directions[select], distance[select], 128)
        base = x["base"].mean(-1).reshape(-1) * (scene.depth[i].size / pixels)
        lit = (x["transmittance"] * x["visible"]).reshape(-1)
        for size, grid in grids.items():
            errors_by_size[size].append(base * (lit_from_grid(grid, size, x["points"], lit).reshape(-1) - lit))
        exact_values.append(base * lit)
        points = x["points"].reshape(-1, 3)
        distances.append(boundary_distance(visible, points))
        lamp_distances.append(x["lamp_distance"].reshape(-1))
        object_distances.append(object_distance(points))
        sides.append(x["visible"].reshape(-1) >= 0.5)
        log("night samples of frame", FRAMES[i])
    distance, lamp_distance, near_object = torch.cat(distances), torch.cat(lamp_distances), torch.cat(object_distances)
    lit_side = torch.cat(sides)
    light_total = float(torch.cat(exact_values).sum())
    results["samples"] = {
        "what": "error of samples along 2000 random rays per frame (128 per ray) before cancelling within pixels; "
        "abs_error and bias relative to the exact fog light",
        "sampled_light_vs_image": light_total / total,
    }
    for size in LIT_SIZES:
        error = torch.cat(errors_by_size[size])
        classes = distance_classes(distance, "shadow_boundary")
        classes |= {
            "lamp<1m": lamp_distance < 1,
            "lamp_1-2m": (lamp_distance >= 1) & (lamp_distance < 2),
            "lamp_2-4m": (lamp_distance >= 2) & (lamp_distance < 4),
            "lamp>=4m": lamp_distance >= 4,
            "object<=1m": near_object <= 1.0,
            "boundary<=1m_and_lamp<4m": (distance <= 1) & (lamp_distance < 4),
            f"boundary<={size:g}m,lit_side": (distance <= size) & lit_side,
            f"boundary<={size:g}m,shadow_side": (distance <= size) & ~lit_side,
        }
        results["samples"][f"lit_{size:g}"] = node_table(classes, error, light_total)

    # Spherical harmonics: the phase function truncated to each order, against the scattering angle
    sh: dict = {
        "what": "phase function truncated to spherical harmonics of each order (no grid), frames 255 and 345; per "
        "scattering-angle bin (frame 345): share of the exact light, and the truncation's signed and absolute error "
        "relative to the frame's exact light",
        "check_saved_sh2_0.25m_vs_truncation_345": float(
            np.abs(images["sh2"][1] - saved["sh_0.25_2"]).sum() / saved["exact"].sum()
        ),
        "angle_bins_deg": [f"{lo}..{hi}" for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])],
        "light_share": (by_angle["exact"] / by_angle["exact"].sum()).tolist(),
    }
    centers = torch.deg2rad(torch.tensor([(a + b) / 2 for a, b in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])]))
    for order in NIGHT_SH_ORDERS:
        sh[f"sh{order}"] = {
            "total": errors(images[f"sh{order}"], exact),
            "signed_by_angle": (by_angle[f"sh{order}"] / by_angle["exact"].sum()).tolist(),
            "abs_by_angle": (by_angle[f"sh{order}_abs"] / by_angle["exact"].sum()).tolist(),
            "phase_ratio_at_bin_centers": (
                truncated_phase(centers.cos(), light.g, order) / henyey_greenstein(centers.cos(), light.g)
            ).tolist(),
            "regions": regions_table(images[f"sh{order}"], exact, regions),
        }
    results["sh"] = sh
    if save_maps:
        plot_night_lit(images, regions, results, light)
    return results


def plot_night_lit(images, regions, results, light):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
    exact = images["exact"][1]
    edge = regions["wedge_edge(3px)"][1]
    wedge = (1 - images["exact"][1].mean(-1) / np.maximum(images["unshadowed"][1].mean(-1), 1e-12)) > 0.5
    panel(
        axes[0, 0],
        show(exact, 0.7 / exact.mean()),
        "night fog light, frame 345 (exact); cyan: shadow wedge edge",
        cmap=None,
        contour=wedge,
    )
    vmax = float(np.percentile(np.abs(error_map(images["lit_0.5"][1], exact)), 99.5))
    for ax, size in zip(axes.flat[1:5], LIT_SIZES):
        name = f"lit_{size:g}"
        e = results["lit"][name]["regions"]
        tot = results["lit"][name]["total"]
        panel(
            ax,
            error_map(images[name][1], exact, signed=True),
            f"lit grid {size:g} m - exact, / mean (blurred); total |err| {100 * tot['rel_l1']:.1f}%\n"
            f"edge band {100 * e['wedge_edge(3px)']['error_share']:.1f}%, near lamp {100 * e['near_lamp(<1.5m)']['error_share']:.1f}%, "
            f"elsewhere {100 * (e['inside_wedge']['error_share'] + e['lit_elsewhere']['error_share']):.1f}%",
            signed=True,
            vmax=vmax,
            contour=wedge,
        )
    ax = axes[1, 2]
    sh = results["sh"]
    centers = [(a + b) / 2 for a, b in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])]
    ax.bar(range(len(centers)), 100 * np.array(sh["light_share"]), color="0.7", label="share of exact light")
    for order in (2, 4, 8):
        ax.plot(
            range(len(centers)),
            100 * np.array(sh[f"sh{order}"]["signed_by_angle"]),
            "o-",
            label=f"SH order {order}: signed error",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels(sh["angle_bins_deg"], rotation=45, fontsize=6)
    ax.set_xlabel("scattering angle (deg, 0 = looking at the lamp)", fontsize=7)
    ax.set_ylabel("% of frame 345's fog light", fontsize=7)
    ax.set_title("SH: phase function truncation vs scattering angle", fontsize=8)
    ax.legend(fontsize=6)
    save(fig, "night_lit_grid_and_sh.png")


# --------------------------------------------------------------------------------------------------------------------
# Night: neural field


def night_field(save_maps: bool = True) -> dict:
    """The neural field of exp_learning.py (10945 parameters), retrained with the same settings, to find where its
    error lies in images (saved image vs exact marching) and in space (against the exact lit value)."""
    import exp_learning

    scene = night_scene(FRAMES)
    light = NightLight(scene)
    low, high = (torch.as_tensor(b, dtype=DTYPE) for b in NIGHT_BOX)
    torch.manual_seed(0)
    net = exp_learning.Field(6, 64, 3)
    weights = SAVE / "night_field_10945.pt"
    if weights.exists():
        net.load_state_dict(torch.load(weights))
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
        steps = 3000
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        generator = torch.Generator().manual_seed(0)
        for step in range(steps):
            points = low + (high - low) * torch.rand(16384, 3, generator=generator)
            loss = (net((points - low) / (high - low) * 2 - 1) - light.lit(points)[:, None]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        log("field trained, last loss", loss.item())
        torch.save(net.state_dict(), weights)

    def field(points: torch.Tensor) -> torch.Tensor:
        flat = points.reshape(-1, 3)
        value = light.lit(flat)
        within = inside(low, high, flat)
        with torch.no_grad():
            value[within] = net((flat[within] - low) / (high - low) * 2 - 1)[:, 0]
        return value.reshape(points.shape[:-1])

    path = SAVE / "night_images.npz"
    images = dict(np.load(path)) if path.exists() else night_images(scene, light, {})[0]
    saved_field = np.load(RESULTS / "learning_field_10945.npy")
    exact = images["exact"]
    regions = night_regions(scene, images, light.position.numpy())
    regions_345 = {k: v[1:] for k, v in regions.items()}
    results: dict = {
        "what": "saved field image (frame 345) vs the same marching without cache; shares relative to the exact fog "
        "light of frame 345",
        "saved_vs_exact": errors(saved_field[None], exact[1:]),
        "saved_regions": regions_table(saved_field[None], exact[1:], regions_345),
        "saved_by_ray_distance_to_lamp": regions_table(
            saved_field[None], exact[1:], {k: v[1:] for k, v in approach_regions(scene, light.position.numpy()).items()}
        ),
    }
    # The retrained field's image, to check that it behaves like the saved one, then its error in space
    retrained, _ = night_images(night_scene((345,)), light, {}, field=field)
    results["retrained_vs_exact"] = errors(retrained["field"], exact[1:])
    results["retrained_vs_saved"] = errors(retrained["field"], saved_field[None])
    generator = torch.Generator().manual_seed(2)
    pixels = 3000
    origin, directions, distance = scene.rays(1)
    select = torch.randperm(scene.depth[1].size, generator=generator)[:pixels]
    x = lamp_samples(light, origin, directions[select], distance[select], 128)
    base = x["base"].mean(-1).reshape(-1) * (scene.depth[1].size / pixels)
    lit = (x["transmittance"] * x["visible"]).reshape(-1)
    error = base * (field(x["points"]).reshape(-1) - lit)
    points = x["points"].reshape(-1, 3)
    visible = lambda p: lamp_visibility(light.maps, light.index, p)
    boundary = boundary_distance(visible, points)
    lamp_distance = x["lamp_distance"].reshape(-1)
    classes = distance_classes(boundary, "shadow_boundary")
    classes |= {
        "lamp<1m": lamp_distance < 1,
        "lamp_1-2m": (lamp_distance >= 1) & (lamp_distance < 2),
        "lamp_2-4m": (lamp_distance >= 2) & (lamp_distance < 4),
        "lamp>=4m": lamp_distance >= 4,
        "in_shadow(lit<0.05)": lit < 0.05,
        "lit(lit>=0.05)": lit >= 0.05,
    }
    results["samples"] = node_table(classes, error, float((base * lit).sum()))
    # Error of the field's stored value itself, uniformly in the box
    uniform = low + (high - low) * torch.rand(200000, 3, generator=generator)
    with torch.no_grad():
        stored_error = (net((uniform - low) / (high - low) * 2 - 1)[:, 0] - light.lit(uniform)).abs()
    near = boundary_distance(visible, uniform[:20000]) <= 0.5
    results["stored_value_error"] = {
        "mean_abs_uniform_in_box": float(stored_error.mean()),
        "mean_abs_within_0.5m_of_boundary": float(stored_error[:20000][near].mean()),
        "mean_abs_farther": float(stored_error[:20000][~near].mean()),
        "fraction_of_box_within_0.5m": float(near.float().mean()),
    }
    if save_maps:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        wedge = (1 - exact[1].mean(-1) / np.maximum(images["unshadowed"][1].mean(-1), 1e-12)) > 0.5
        panel(
            axes[0],
            show(exact[1], 0.7 / exact[1].mean()),
            "night fog light, frame 345 (exact)",
            cmap=None,
            contour=wedge,
        )
        r = results["saved_regions"]
        panel(
            axes[1],
            error_map(saved_field, exact[1], signed=True),
            f"neural field - exact, / mean (blurred); total {100 * results['saved_vs_exact']['rel_l1']:.2f}%\n"
            f"edge band {100 * r['wedge_edge(3px)']['error_share']:.2f}%, near lamp {100 * r['near_lamp(<1.5m)']['error_share']:.2f}%, "
            f"wedge {100 * r['inside_wedge']['error_share']:.2f}%, lit {100 * r['lit_elsewhere']['error_share']:.2f}%",
            signed=True,
            contour=wedge,
        )
        save(fig, "night_field.png")
    return results


# --------------------------------------------------------------------------------------------------------------------
# Night: residual network out of distribution


def night_residual(save_maps: bool = True) -> dict:
    """Residual U-Net trained on day frames, applied to the night frame 345 (whole image, lamp lighting surfaces and
    fog, light scattered many times), against Cycles (night_ms_ref) and the closed form it corrects."""
    scene = night_scene((345,))
    clear = load_images(OUT / "night_clear" / "frames", (345,))
    medium = scene.medium.model_copy(update={"multiple_scattering": True})
    t = scene.transforms[0]
    closed_form = apply_medium(
        clear[0],
        scene.depth[0],
        t,
        t["transform_matrix"],
        medium,
        scene.lighting,
        occlusion=scene.occlusion,
        dtype=DTYPE,
    ).radiance.numpy()[None]
    residual = np.load(RESULTS / "learning_residual_night.npy")[None]
    image = np.load(RESULTS / "learning_image_night.npy")[None]
    reference = load_images(OUT / "night_ms_ref" / "frames", (345,))
    fog = sum(load_images(OUT / "night_ms_ref" / "volume" / p, (345,)) for p in ("direct", "indirect"))
    surfaces = reference - fog
    total = float(reference.sum())
    results: dict = {
        "what": "frame 345, whole images; shares relative to Cycles' whole image",
        "closed_form": errors(closed_form, reference),
        "residual": errors(residual, reference),
        "image_net": errors(image, reference),
        "reference_fog_share": float(fog.sum() / total),
    }
    depths = load_images(ROOT / "clear" / "depths", (345,))[..., 0]
    sky = depths >= 1e9
    lum = luminance(reference)
    fog_lum, surface_lum = luminance(fog), luminance(surfaces)
    halo = fog_lum >= np.percentile(fog_lum, 98)
    lit_surfaces = (surface_lum > fog_lum) & (surface_lum >= np.percentile(surface_lum[~sky], 75)) & ~halo
    dark = lum <= np.percentile(lum, 50)
    regions = {
        "halo(top 2% fog light)": halo,
        "lit_surfaces": lit_surfaces & ~sky,
        "fog_dominated": (fog_lum >= surface_lum) & ~halo & ~dark,
        "dark(half of pixels)": dark & ~halo & ~lit_surfaces,
    }
    regions["other"] = ~np.any(np.stack(list(regions.values())), axis=0)
    results["closed_form_regions"] = regions_table(closed_form, reference, regions)
    results["residual_regions"] = regions_table(residual, reference, regions)
    change = np.abs(residual - reference) - np.abs(closed_form - reference)
    results["change_of_abs_error"] = {
        name: float(change[np.broadcast_to(mask[..., None], change.shape)].sum() / total)
        for name, mask in regions.items()
    }
    results["change_of_abs_error"]["total"] = float(change.sum() / total)
    # The correction it applies, against the brightness of the closed form it is given (its input)
    level = luminance(closed_form)
    edges = np.quantile(level, np.linspace(0, 1, 11))
    corrections = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (level >= lo) & (level <= hi)
        m = np.broadcast_to(sel[..., None], closed_form.shape)
        corrections.append(
            {
                "closed_form_luminance": [float(lo), float(hi)],
                "residual_over_closed_form": ratio(float(residual[m].sum()), float(closed_form[m].sum())),
                "cycles_over_closed_form": ratio(float(reference[m].sum()), float(closed_form[m].sum())),
                "change_of_abs_error": float(change[m].sum() / total),
            }
        )
    results["correction_by_input_brightness_deciles"] = corrections
    if save_maps:
        fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
        ref = reference[0]
        panel(axes[0, 0], show(ref, 0.3 / ref.mean()), "Cycles (night, multiple scattering), frame 345", cmap=None)
        panel(axes[0, 1], show(closed_form[0], 0.3 / ref.mean()), "closed form (multiple scattering)", cmap=None)
        panel(axes[0, 2], show(residual[0], 0.3 / ref.mean()), "residual U-Net (trained on day frames)", cmap=None)
        cf_map, res_map = error_map(closed_form[0], ref, signed=True), error_map(residual[0], ref, signed=True)
        vmax = float(np.percentile(np.abs(cf_map), 99))
        panel(
            axes[1, 0],
            cf_map,
            f"closed form - Cycles, / mean (blurred), |err| {100 * results['closed_form']['rel_l1']:.1f}%",
            signed=True,
            vmax=vmax,
        )
        panel(
            axes[1, 1],
            res_map,
            f"residual - Cycles, / mean (blurred), |err| {100 * results['residual']['rel_l1']:.1f}%",
            signed=True,
            vmax=vmax,
        )
        panel(
            axes[1, 2],
            luminance(blur(change)[0]) / luminance(ref).mean(),
            "change of |error| with the residual net (red: worse)",
            signed=True,
            vmax=vmax,
        )
        save(fig, "night_residual.png")
    return results


PARTS = {
    "day_sun": day_sun,
    "day_sh": day_sh,
    "day_sh_cycles": day_sh_cycles,
    "night_lit": night_lit,
    "night_field": night_field,
    "night_residual": night_residual,
}


def main(names: list[str]):
    SAVE.mkdir(parents=True, exist_ok=True)
    for name in names or list(PARTS):
        start = time.perf_counter()
        log("part", name)
        part = PARTS[name]()
        part["seconds"] = time.perf_counter() - start
        # Read again before writing, as parts may run in separate processes
        results = json.loads(JSON.read_text()) if JSON.exists() else {}
        results[name] = part
        JSON.write_text(json.dumps(results, indent=1))
        log("part", name, "done in", round(time.perf_counter() - start), "s")


if __name__ == "__main__":
    main(sys.argv[1:])
