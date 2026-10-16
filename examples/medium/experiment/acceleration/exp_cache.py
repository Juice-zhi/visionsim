"""Direction 3: lighting precomputed once in a world-space grid, shared by every frame and camera.

Each frame then only integrates along its camera rays what it looks up in the grid, with the same samples as the
pointwise evaluation of the closed form's lighting, so that differences come from the cache alone.

Night scene, the point light behind the cube, in a 32 x 32 x 8 m grid around it (the exact lighting is used outside):

- ``lit``: the light's direction and intensity are known, and each voxel corner stores its transmittance towards the
  light times its visibility (1 value), which the 1/r² falloff and the phase function then multiply exactly;
- ``irradiance``: each corner stores the light reaching it, falloff included (1 value), as a generic irradiance cache;
- ``sh``: each corner stores the light reaching it as spherical harmonics up to an order, ``(order + 1)²`` values,
  which the phase function convolves by scaling each order l by g^l, as a cache of all lights in one representation.

Day scene, in an 80 x 140 x 8.5 m grid over the objects, below which objects can hide the sun and the sky (outside,
nothing is hidden, and the closed form's unoccluded lighting is exact):

- ``sun``: each corner stores the sun's visibility (1 value), which only depends on the geometry;
- ``sh``: each corner stores the light from the sky, the ground and multiple scattering that it scatters towards every
  direction, i.e. after the phase function, as spherical harmonics, ``3 (order + 1)²`` values, which depend on the
  fog's density;
- ``cells``: each corner stores which cells of the sky it sees (2 bits per cell, above and below the horizon), which
  only depends on the geometry, and the closed form's tables then give the light of the visible cells exactly.

Results are saved to ``results/cache.json``.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from visionsim.medium.occlusion import band_attenuation, cell_phases
from visionsim.medium.optics import henyey_greenstein

from common import (
    DEVICE,
    DTYPE,
    OUT,
    DayLight,
    NightLight,
    by_chunks,
    day_nodes,
    day_scene,
    errors,
    grid_points,
    inside,
    integrate_lamp,
    night_scene,
    sh_basis,
    sphere_quadrature,
    trilinear,
)

NIGHT_BOX = ((-12.0, 5.0, 0.0), (20.0, 37.0, 8.0))
DAY_BOX = ((-40.0, 0.0, 0.0), (40.0, 140.0, 8.5))
SUN_NODES, SMOOTH_NODES, LAMP_NODES = 512, 48, 256


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


# --------------------------------------------------------------------------------------------------------------------
# Night


def night_grid(light: NightLight, size: float, kind: str, order: int = 0) -> torch.Tensor:
    low, high = NIGHT_BOX
    points, shape = grid_points(low, high, size)

    def values(p: torch.Tensor) -> torch.Tensor:
        lit = light.lit(p)
        if kind == "lit":
            return lit[:, None]
        to_light = light.position - p
        r = to_light.norm(dim=-1).clamp_min(max(light.emitter.radius, 1e-4))
        irradiance = (lit / (r * r) * light.emitter.intensity[1])[:, None]
        if kind == "irradiance":
            return irradiance
        return irradiance * sh_basis(to_light / to_light.norm(dim=-1, keepdim=True).clamp_min(1e-12), order)

    return by_chunks(values, len(points), 2**18, points).reshape(*shape, -1)


def night_source(light: NightLight, grid: torch.Tensor, size: float, kind: str, order: int = 0):
    low, high = NIGHT_BOX
    factors = torch.tensor([light.g**l for l in range(order + 1) for _ in range(2 * l + 1)], dtype=DTYPE, device=DEVICE)
    color = light.emitter.intensity / light.emitter.intensity[1]

    def source(points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        result = light(points, directions)  # exact outside of the grid
        within = inside(low, high, points)
        p, d = points[within], directions[within]
        stored = trilinear(grid, low, size, p)
        if kind == "lit":
            value = light.falloff(p, d) * stored
        elif kind == "irradiance":
            to_light = light.position - p
            phase = henyey_greenstein((d * to_light).sum(-1) / to_light.norm(dim=-1).clamp_min(1e-12), light.g)
            value = light.albedo * (phase[:, None] * stored) * color
        else:
            value = light.albedo * (stored * factors * sh_basis(d, order)).sum(-1, keepdim=True) * color
        result[within] = value
        return result

    return source


def run_night() -> dict:
    scene = night_scene()
    light = NightLight(scene)
    cf = (
        np.load(OUT / "results" / "closed_form_night.npy")
        if (OUT / "results" / "closed_form_night.npy").exists()
        else None
    )
    if cf is None:
        cf = np.stack([scene.closed_form(i).inscatter.numpy() for i in range(len(scene.frames))])
    rays = [scene.rays(i) for i in range(len(scene.frames))]

    def render(source) -> tuple[np.ndarray, float]:
        images, seconds = [], []
        for (origin, directions, distance), depth in zip(rays, scene.depth):
            image, s = timed(
                lambda: by_chunks(
                    lambda d, l: integrate_lamp(light, origin, d, l, source, n=LAMP_NODES),
                    len(directions),
                    4096,
                    directions,
                    distance,
                )
            )
            images.append(image.reshape(*depth.shape, 3).numpy())
            seconds.append(s)
        return np.stack(images), min(seconds)

    exact, exact_seconds = render(light)
    results: dict = {
        "box": NIGHT_BOX,
        "exact": {
            "vs_closed_form": errors(exact, cf),
            "vs_cycles": errors(exact, scene.reference),
            "seconds": exact_seconds,
        },
        "closed_form_vs_cycles": errors(cf, scene.reference),
        "variants": [],
    }
    print("night exact", json.dumps(results["exact"]), flush=True)
    configs = [("lit", s, 0) for s in (1.0, 0.5, 0.25, 0.125)]
    configs += [("irradiance", s, 0) for s in (0.5, 0.25, 0.125)]
    configs += [("sh", 0.25, order) for order in (8, 6, 4, 3, 2, 1)]
    saved = {}
    highest = None
    for kind, size, order in configs:
        if kind == "sh":
            # Coefficients of lower orders are the first ones of higher orders
            if highest is None:
                highest, build = timed(lambda: night_grid(light, size, kind, order))
            grid = highest[..., : (order + 1) ** 2].contiguous()
        else:
            grid, build = timed(lambda: night_grid(light, size, kind, order))
        image, seconds = render(night_source(light, grid, size, kind, order))
        entry = {
            "kind": kind,
            "size": size,
            "order": order,
            "values_per_corner": grid.shape[-1],
            "corners": int(np.prod(grid.shape[:3])),
            "bytes_fp32": int(grid.numel() * 4),
            "build_seconds": build,
            "seconds": seconds,
            "vs_closed_form": errors(image, cf),
            "vs_cycles": errors(image, scene.reference),
        }
        results["variants"].append(entry)
        print("night", json.dumps(entry), flush=True)
        if (kind, size, order) in (("lit", 0.25, 0), ("irradiance", 0.25, 0), ("sh", 0.25, 2), ("lit", 1.0, 0)):
            saved[f"{kind}_{size}_{order}"] = image[1]
    np.savez_compressed(OUT / "results" / "cache_night_images.npz", exact=exact[1], **saved)
    return results


# --------------------------------------------------------------------------------------------------------------------
# Day


def day_sun_grid(light: DayLight, size: float) -> torch.Tensor:
    points, shape = grid_points(*DAY_BOX, size)
    visible = by_chunks(lambda p: light.sun_visibility(p), len(points), 2**20, points)
    return (visible * 255).round().to(torch.uint8).reshape(*shape, 1)


def day_cells_grid(light: DayLight, size: float) -> torch.Tensor:
    points, shape = grid_points(*DAY_BOX, size)

    def bits(p: torch.Tensor) -> torch.Tensor:
        above, below = light.sky_visibility(p)
        return torch.cat([above, below], dim=-1)

    return by_chunks(bits, len(points), 2**17, points).reshape(*shape, -1)


def day_sh_grid(light: DayLight, size: float, order: int, n_theta: int = 16, n_phi: int = 32) -> torch.Tensor:
    """Spherical harmonics of the light from the sky, the ground and multiple scattering scattered towards every
    direction at each corner, from a quadrature over directions, with the visible fraction of each source computed
    for all directions at once as products of matrices."""
    points, shape = grid_points(*DAY_BOX, size)
    directions, weights = sphere_quadrature(n_theta, n_phi)
    basis = sh_basis(directions, order) * weights[:, None]  # (M, K)
    # Sources only depend on the height of corners, through the optical depth above them
    low_z = DAY_BOX[0][2]
    heights = torch.arange(shape[2], dtype=DTYPE, device=DEVICE) * size + low_z
    depth_z = light.depth_above(torch.stack([torch.zeros_like(heights)] * 2 + [heights], dim=-1))
    mu = directions[:, 2]
    sources = {
        name: light.source(name, mu.repeat(len(heights)), depth_z.repeat_interleave(len(directions))).reshape(
            len(heights), len(directions), 3
        )
        for name in light.tables
    }
    phases = cell_phases(light.edges, light.counts, light.g, directions)  # (M, k)
    phases_below = cell_phases(light.edges, light.counts, light.g, directions, below=True)
    counts = torch.as_tensor(light.counts)
    band = torch.repeat_interleave(torch.arange(len(light.counts)), counts).to(DEVICE)  # (k,)

    def coefficients(p: torch.Tensor) -> torch.Tensor:
        c = light.depth_above(p)
        zi = ((p[:, 2] - low_z) / size).round().long()
        lit, lit_below = (x.to(DTYPE) for x in light.sky_visibility(p))
        total = torch.zeros(len(p), len(directions), 3, dtype=DTYPE, device=DEVICE)
        for name in light.tables:
            source = sources[name][zi]
            if name == "sky":
                weight = band_attenuation(light.edges, c)[:, band]  # (n, k)
                ratio = ((lit * weight) @ phases.T) / (weight @ phases.T).clamp_min(1e-30)
            elif name == "ground":
                ratio = (lit_below @ phases_below.T) / phases_below.sum(-1).clamp_min(1e-30)
            else:
                heights = torch.log(light.depth_ground / c.clamp_min(1e-30))
                scale = light.scattering.height_scale
                fh = heights.clamp(0, scale) / scale * (len(light.band_light) - 1)
                h = fh.floor().clamp(0, len(light.band_light) - 2).long()
                arriving = torch.lerp(light.band_light[h], light.band_light[h + 1], (fh - h)[:, None, None])
                up, down = arriving[:, 0][:, band], arriving[:, 1][:, band]  # (n, k)
                numerator = (lit * up) @ phases.T + (lit_below * down) @ phases_below.T
                ratio = numerator / (up @ phases.T + down @ phases_below.T).clamp_min(1e-30)
            total = total + source * ratio[..., None]
        return torch.einsum("nmc,mk->nkc", total, basis).reshape(len(p), -1)

    return by_chunks(coefficients, len(points), 4096, points).reshape(*shape, -1)


def run_day() -> dict:
    scene = day_scene(frames=(255, 345))
    light = DayLight(scene)
    fog, beta = light.fog, light.beta
    cf = [scene.closed_form(i) for i in range(2)]
    cf_inscatter = np.stack([r.inscatter.numpy() for r in cf])
    rays = [scene.rays(i) for i in range(2)]
    low, high = DAY_BOX

    def march(origin, directions, distance, n, fn):
        t, w = day_nodes(fog, beta, origin, directions, distance, n)
        points = origin + directions[:, None] * t[..., None]
        values = fn(points.reshape(-1, 3), directions[:, None].expand_as(points).reshape(-1, 3))
        return (values.reshape(*t.shape, 3) * w[..., None]).sum(dim=1)

    def render(fn, nodes: int, chunk: int) -> tuple[np.ndarray, float]:
        """One part of the light, the sun's or the smooth light, scattered along the rays of both frames."""
        images, seconds = [], []
        for (origin, directions, distance), depth in zip(rays, scene.depth):
            image, s = timed(
                lambda: by_chunks(
                    lambda d, l: march(origin, d, l, nodes, fn), len(directions), chunk, directions, distance
                )
            )
            images.append(image.reshape(*depth.shape, 3).numpy())
            seconds.append(s)
        return np.stack(images), min(seconds)

    def render_sun(fn):
        return render(fn, SUN_NODES, 2048)

    def render_smooth(fn):
        return render(fn, SMOOTH_NODES, 1024)

    def unoccluded(points, directions):
        c = light.depth_above(points)
        total = torch.zeros(len(points), 3, dtype=DTYPE, device=DEVICE)
        for name in light.tables:
            total = total + light.source(name, directions[:, 2], c)
        return total

    def sun_from(grid, size):
        def fn(points, directions):
            visible = torch.ones(len(points), dtype=DTYPE, device=DEVICE)
            within = inside(low, high, points)
            visible[within] = trilinear(grid, low, size, points[within])[:, 0] / 255
            return light.sun(points, directions, visible=visible)

        return fn

    def smooth_from_sh(grid, size, order):
        def fn(points, directions):
            result = unoccluded(points, directions)
            within = inside(low, high, points)
            stored = trilinear(grid, low, size, points[within]).reshape(-1, (order + 1) ** 2, 3)
            result[within] = (stored * sh_basis(directions[within], order)[..., None]).sum(dim=1)
            return result

        return fn

    def smooth_from_cells(grid, size):
        k = grid.shape[-1] // 2

        def fn(points, directions):
            result = unoccluded(points, directions)
            within = inside(low, high, points)
            p = points[within]
            index = ((p - torch.as_tensor(low, dtype=DTYPE, device=DEVICE)) / size).round().long()
            bits = grid[index[:, 0], index[:, 1], index[:, 2]]
            result[within] = light.smooth(p, directions[within], lit=(bits[:, :k], bits[:, k:]))
            return result

        return fn

    def exact_sun(points, directions):
        return light.sun(points, directions)

    def exact_smooth(points, directions):
        return light.smooth(points, directions)

    def entry_for(image, seconds, **info):
        return info | {
            "seconds": seconds,
            "vs_closed_form": errors(image, cf_inscatter),
            "vs_cycles": errors(np.stack([scene.output(i, image[i]) for i in range(2)]), scene.reference),
        }

    # Each variant caches one part of the light, the other being computed exactly once for all
    exact_sun_image, sun_seconds = render_sun(exact_sun)
    exact_smooth_image, smooth_seconds = render_smooth(exact_smooth)
    exact = exact_sun_image + exact_smooth_image
    results: dict = {
        "box": DAY_BOX,
        "exact": entry_for(exact, sun_seconds + smooth_seconds, sun_seconds=sun_seconds, smooth_seconds=smooth_seconds),
        "closed_form_vs_cycles": errors(np.stack([r.radiance.numpy() for r in cf]), scene.reference),
        "variants": [],
    }
    print("day exact", json.dumps(results["exact"]), flush=True)

    saved = {"exact": exact[1]}
    for size in (1.0, 0.5, 0.25, 0.125):
        grid, build = timed(lambda: day_sun_grid(light, size))
        image, seconds = render_sun(sun_from(grid, size))
        entry = entry_for(
            image + exact_smooth_image,
            seconds,
            kind="sun",
            size=size,
            corners=int(np.prod(grid.shape[:3])),
            bytes=int(grid.numel()),
            build_seconds=build,
        )
        results["variants"].append(entry)
        print("day", json.dumps(entry), flush=True)
        if size in (1.0, 0.25):
            saved[f"sun_{size}"] = image[1] + exact_smooth_image[1]
        del grid
    for size in (1.0, 0.5):
        grid, build = timed(lambda: day_cells_grid(light, size))
        image, seconds = render_smooth(smooth_from_cells(grid, size))
        bits = grid.shape[-1]
        entry = entry_for(
            exact_sun_image + image,
            seconds,
            kind="cells",
            size=size,
            corners=int(np.prod(grid.shape[:3])),
            bytes=int(np.prod(grid.shape[:3]) * math.ceil(bits / 8)),
            build_seconds=build,
        )
        results["variants"].append(entry)
        print("day", json.dumps(entry), flush=True)
        del grid
    built = {}
    for size, order in ((1.0, 4), (1.0, 3), (1.0, 2), (1.0, 1), (0.5, 3), (0.5, 2)):
        # Coefficients of lower orders are the first ones of higher orders, so each size is built once
        if size not in built:
            built.clear()
            built[size] = timed(lambda: day_sh_grid(light, size, order))
        full, build = built[size]
        grid = full.reshape(*full.shape[:3], -1, 3)[..., : (order + 1) ** 2, :].reshape(*full.shape[:3], -1)
        image, seconds = render_smooth(smooth_from_sh(grid, size, order))
        entry = entry_for(
            exact_sun_image + image,
            seconds,
            kind="sh",
            size=size,
            order=order,
            corners=int(np.prod(grid.shape[:3])),
            bytes=int(grid.numel() * 4),
            build_seconds=build,
        )
        results["variants"].append(entry)
        print("day", json.dumps(entry), flush=True)
        if (size, order) in ((1.0, 2),):
            saved[f"sh_{size}_{order}"] = exact_sun_image[1] + image[1]
        del grid, full
    np.savez_compressed(OUT / "results" / "cache_day_images.npz", **saved)
    return results


def main():
    results = {"night": run_night(), "day": run_day()}
    (OUT / "results" / "cache.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
