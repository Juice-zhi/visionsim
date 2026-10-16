"""Time the deterministic methods of the acceleration study on the GPU, once it is free.

Run with ``ACCEL_DEVICE=cuda``. Their results are the same as on the CPU (``exp_*.py``), up to rounding, so only times
are measured here: the closed form at 320x180 and 1920x1080, with and without shadows, at lower resolutions with
upsampling, froxels, world-space caches, a neural field and the image networks. Each time is the fastest of a few
runs, after a warm-up run. Results are saved to ``results/gpu.json``.
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np
import torch

from common import (
    DEVICE,
    DTYPE,
    OUT,
    ROOT,
    DayLight,
    NightLight,
    Scene,
    box_depth,
    by_chunks,
    day_nodes,
    day_scene,
    inside,
    integrate_lamp,
    load_images,
    load_transforms,
    night_scene,
    sh_basis,
    synchronize,
    trilinear,
)
from exp_cache import DAY_BOX, NIGHT_BOX, SMOOTH_NODES, SUN_NODES, day_sh_grid, day_sun_grid, night_grid, night_source
from exp_froxel import froxels
from exp_learning import Field, UNet
from exp_less import low_resolution

assert DEVICE.startswith("cuda"), "Set ACCEL_DEVICE=cuda to time on the GPU."


def fastest(fn, repeats: int = 3) -> float:
    fn()
    synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


def hd_scene() -> Scene:
    hd = ROOT / "hd" / "clear"
    frames = (255,)
    transforms = load_transforms(hd / "frames", frames)
    day = day_scene(frames=frames)
    depth = np.stack([box_depth(d, t) for d, t in zip(load_images(hd / "depths", frames)[..., 0], transforms)])
    return Scene(
        name="day",
        medium=day.medium,
        lighting=day.lighting,
        occlusion=day.occlusion,
        frames=frames,
        clear=load_images(hd / "frames", frames),
        depth=depth,
        transforms=transforms,
        reference=day.reference,
        whole=True,
    )


def march(light, origin, directions, distance, n, fn):
    t, w = day_nodes(light.fog, light.beta, origin, directions, distance, n)
    points = origin + directions[:, None] * t[..., None]
    values = fn(points.reshape(-1, 3), directions[:, None].expand_as(points).reshape(-1, 3))
    return (values.reshape(*t.shape, 3) * w[..., None]).sum(dim=1)


def main():
    results: dict = {"device": torch.cuda.get_device_name(0)}

    def record(name, fn, repeats=3):
        try:
            results[name] = fastest(fn, repeats)
            print(name, f"{results[name]:.4f}", flush=True)
        except Exception:
            results[name] = None
            print(name, "failed", traceback.format_exc(), flush=True)

    day = day_scene(frames=(345,))
    night = night_scene((345,))
    hd = hd_scene()
    record("cf_day_320_no_shadows", lambda: day.closed_form(0, occlusion=None, normals=None))
    record("cf_day_320_shadows", lambda: day.closed_form(0, normals=None))
    record("cf_day_320_shadows_surfaces", lambda: day.closed_form(0))
    record("cf_night_320_no_shadows", lambda: night.closed_form(0, occlusion=None))
    record("cf_night_320_shadows", lambda: night.closed_form(0))
    record("cf_day_1080_no_shadows", lambda: hd.closed_form(0, occlusion=None))
    record("cf_day_1080_shadows", lambda: hd.closed_form(0))
    for s in (2, 4, 8):
        camera = hd.transforms[0]
        record(
            f"cf_day_1080_low{s}",
            lambda s=s, camera=camera: low_resolution(
                hd.clear[0], hd.depth[0], camera, hd.medium, hd.lighting, hd.occlusion, s
            ),
        )

    lights = {"day": DayLight(day), "night": NightLight(night)}
    for tile, slices, jitter in ((8, 64, 1), (4, 128, 4)):
        record(
            f"froxel_day_320_{tile}_{slices}_{jitter}",
            lambda t=tile, s=slices, j=jitter: froxels(day, lights["day"], 0, t, s, j),
        )
        record(
            f"froxel_night_320_{tile}_{slices}_{jitter}",
            lambda t=tile, s=slices, j=jitter: froxels(night, lights["night"], 0, t, s, j),
        )
    hd_light = DayLight(hd)
    record("froxel_day_1080_8_64_1", lambda: froxels(hd, hd_light, 0, 8, 64, 1), repeats=2)

    # Caches: building them, and integrating what they store along the rays of a frame
    light = lights["night"]
    origin, directions, distance = night.rays(0)
    record("cache_night_lit_0.25_build", lambda: night_grid(light, 0.25, "lit"))
    grid = night_grid(light, 0.25, "lit")
    source = night_source(light, grid, 0.25, "lit")
    record(
        "cache_night_lit_0.25_320",
        lambda: by_chunks(
            lambda d, l: integrate_lamp(light, origin, d, l, source, n=256), len(directions), 2**15, directions, distance
        ),
    )
    record(
        "exact_night_320",
        lambda: by_chunks(
            lambda d, l: integrate_lamp(light, origin, d, l, light, n=256), len(directions), 2**15, directions, distance
        ),
    )

    light = lights["day"]
    origin, directions, distance = day.rays(0)
    low, high = DAY_BOX
    sun_grid = day_sun_grid(light, 0.5)
    sh_grid = day_sh_grid(light, 1.0, 4)

    def sun_cached(points, dirs):
        visible = torch.ones(len(points), dtype=DTYPE, device=DEVICE)
        within = inside(low, high, points)
        visible[within] = trilinear(sun_grid, low, 0.5, points[within])[:, 0] / 255
        return light.sun(points, dirs, visible=visible)

    def smooth_cached(points, dirs):
        c = light.depth_above(points)
        result = sum(light.source(name, dirs[:, 2], c) for name in light.tables)
        within = inside(low, high, points)
        stored = trilinear(sh_grid, low, 1.0, points[within]).reshape(-1, 25, 3)
        result[within] = (stored * sh_basis(dirs[within], 4)[..., None]).sum(dim=1)
        return result

    record(
        "cache_day_sun0.5_320",
        lambda: by_chunks(
            lambda d, l: march(light, origin, d, l, SUN_NODES, sun_cached), len(directions), 2**13, directions, distance
        ),
    )
    record(
        "cache_day_sh4_1m_320",
        lambda: by_chunks(
            lambda d, l: march(light, origin, d, l, SMOOTH_NODES, smooth_cached),
            len(directions),
            2**13,
            directions,
            distance,
        ),
    )
    record(
        "exact_day_sun_320",
        lambda: by_chunks(
            lambda d, l: march(light, origin, d, l, SUN_NODES, light.sun), len(directions), 2**13, directions, distance
        ),
    )
    record(
        "exact_day_smooth_320",
        lambda: by_chunks(
            lambda d, l: march(light, origin, d, l, SMOOTH_NODES, light.smooth),
            len(directions),
            2**11,
            directions,
            distance,
        ),
        repeats=1,
    )
    record("cache_day_sun0.5_build", lambda: day_sun_grid(light, 0.5), repeats=1)
    record("cache_day_sh4_1m_build", lambda: day_sh_grid(light, 1.0, 4), repeats=1)

    # Neural field of the lamp's transmittance times visibility, trained briefly, then queried along the rays
    light = lights["night"]
    origin, directions, distance = night.rays(0)
    lo, hi = (torch.as_tensor(b, dtype=DTYPE, device=DEVICE) for b in NIGHT_BOX)
    net = Field(6, 64, 3).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)

    def train():
        for _ in range(3000):
            points = lo + (hi - lo) * torch.rand(16384, 3, device=DEVICE)
            loss = (net((points - lo) / (hi - lo) * 2 - 1) - light.lit(points)[:, None]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    record("field_train_3000_steps", train, repeats=1)

    def field_source(points, dirs):
        result = light(points, dirs)
        within = ((points >= lo) & (points <= hi)).all(dim=-1)
        with torch.no_grad():
            stored = net((points[within] - lo) / (hi - lo) * 2 - 1)
        result[within] = light.falloff(points[within], dirs[within]) * stored
        return result

    record(
        "field_night_320",
        lambda: by_chunks(
            lambda d, l: integrate_lamp(light, origin, d, l, field_source, n=256),
            len(directions),
            2**14,
            directions,
            distance,
        ),
    )

    unet = UNet(4).to(DEVICE).eval()
    for h, w in ((180, 320), (1080, 1920)):
        x = torch.rand(1, 4, h, w, device=DEVICE)
        with torch.no_grad():
            record(f"unet_{w}", lambda x=x: unet(x))
    print(json.dumps(results, indent=1))
    (OUT / "results" / "gpu.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
