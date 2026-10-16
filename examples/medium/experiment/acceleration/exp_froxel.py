"""Direction 5: real-time volumetric fog, as game engines do it, with froxels (cells of a grid aligned with the camera).

- ``froxel``: a prototype of Unreal's volumetric fog, with the closed form's lighting evaluated at the center of each
  froxel (tiles of ``tile`` pixels, ``slices`` slices up to 100 m, spread as EEVEE spreads them), integrated from the
  camera outwards per column, and looked up by each pixel at its depth. Beyond 100 m, the closed form without shadows
  adds the rest, as engines add their analytic height fog. Optionally, the lighting is averaged over jittered positions
  within each froxel, as temporal accumulation does for a still camera. This separates the error of froxels from that
  of a different lighting model.
- ``eevee``: Blender's EEVEE, an off-the-shelf real-time renderer whose volumetrics use froxels too, rendered by
  ``render_all.ps1`` on the integrated GPU, as is, and with its fog's light added to the closed form's surfaces.

Results are saved to ``results/froxel.json``.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import torch
from visionsim.medium import apply_medium
from visionsim.medium.render import camera_rays

from common import (
    DAY_FRAMES,
    DEVICE,
    DTYPE,
    NIGHT_FRAMES,
    OUT,
    ROOT,
    DayLight,
    NightLight,
    by_chunks,
    day_scene,
    errors,
    flicker,
    load_images,
    night_scene,
)

START, END, DISTRIBUTION = 0.1, 100.0, 0.8


def slice_depth(u: torch.Tensor) -> torch.Tensor:
    """View-space depth at a fraction of the slices, as EEVEE spreads them between linear and exponential."""
    linear = START + u * (END - START)
    exponential = START * (END / START) ** u
    return torch.lerp(linear, exponential, DISTRIBUTION)


def depth_fraction(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`slice_depth`, by bisection."""
    low, high = torch.zeros_like(z), torch.ones_like(z)
    for _ in range(40):
        mid = (low + high) / 2
        above = slice_depth(mid) < z
        low, high = torch.where(above, mid, low), torch.where(above, high, mid)
    return (low + high) / 2


def froxels(scene, light, i: int, tile: int, slices: int, jitter: int) -> tuple[np.ndarray, float]:
    """Light scattered towards each pixel, of shape (h, w, 3), with froxels, and the time taken."""
    camera = scene.transforms[i]
    start = time.perf_counter()
    h, w = int(camera["h"]), int(camera["w"])
    rows, cols = -(-h // tile), -(-w // tile)
    fog = scene.medium.components[0]
    beta = float(scene.medium.extinction)
    generator = torch.Generator().manual_seed(0)
    total = torch.zeros(rows, cols, slices + 1, 4, dtype=DTYPE, device=DEVICE)
    for j in range(jitter):
        # Sample positions within each froxel: its center, or jittered as temporal accumulation spreads them
        offset = torch.rand(3, generator=generator) if jitter > 1 else torch.full((3,), 0.5)
        sub = dict(camera) | {
            "fl_x": camera["fl_x"] / tile,
            "fl_y": camera["fl_y"] / tile,
            "cx": camera["cx"] / tile - (float(offset[0]) - 0.5),
            "cy": camera["cy"] / tile - (float(offset[1]) - 0.5),
            "w": cols,
            "h": rows,
        }
        origin, directions, scale = camera_rays(sub, camera["transform_matrix"], dtype=DTYPE, device=DEVICE)
        bounds = slice_depth(torch.arange(slices + 1, dtype=DTYPE, device=DEVICE) / slices)
        centers = slice_depth((torch.arange(slices, dtype=DTYPE, device=DEVICE) + float(offset[2])) / slices)
        distance = centers[None, None, :] * scale[..., None]  # (rows, cols, slices)
        points = origin + directions[:, :, None, :] * distance[..., None]
        flat_dirs = directions[:, :, None, :].expand_as(points).reshape(-1, 3)
        source = by_chunks(lambda p, d: light(p, d), flat_dirs.shape[0], 2**15, points.reshape(-1, 3), flat_dirs)
        source = source.reshape(rows, cols, slices, 3)
        sigma = beta * fog.density * torch.exp(-(points[..., 2] - fog.base_height) / fog.falloff)
        thickness = (bounds[1:] - bounds[:-1])[None, None, :] * scale[..., None]
        step = torch.exp(-sigma * thickness)
        scattered = source * (1 - step)[..., None]
        before = torch.cumprod(torch.cat([torch.ones_like(step[..., :1]), step], dim=-1), dim=-1)  # (r, c, s+1)
        light_sum = torch.cat(
            [torch.zeros_like(scattered[..., :1, :]), torch.cumsum(before[..., :-1, None] * scattered, dim=-2)], dim=-2
        )
        total = total + torch.cat([light_sum, before[..., None]], dim=-1)
    volume = total / jitter

    # Each pixel looks the volume up at its depth, bilinearly across columns and linearly across slice boundaries
    _, _, scale = camera_rays(camera, camera["transform_matrix"], dtype=DTYPE, device=DEVICE)
    z = torch.as_tensor(scene.depth[i], dtype=DTYPE, device=DEVICE)
    fk = (depth_fraction(z.clamp(START, END)) * slices).clamp(0, slices - 1e-4)
    py = ((torch.arange(h, dtype=DTYPE, device=DEVICE) + 0.5) / tile - 0.5).clamp(0, rows - 1)
    px = ((torch.arange(w, dtype=DTYPE, device=DEVICE) + 0.5) / tile - 0.5).clamp(0, cols - 1)
    y0, x0 = py.floor().clamp(max=max(rows - 2, 0)).long(), px.floor().clamp(max=max(cols - 2, 0)).long()
    wy, wx = (py - y0)[:, None], (px - x0)[None, :]
    k0 = fk.floor().long()
    wk = fk - k0
    result = 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = (y0 + dy)[:, None].expand(h, w), (x0 + dx)[None, :].expand(h, w)
            weight = (wy if dy else 1 - wy) * (wx if dx else 1 - wx)
            near, far = volume[yi, xi, k0, :3], volume[yi, xi, (k0 + 1).clamp(max=slices), :3]
            result = result + weight[..., None] * torch.lerp(near, far, wk[..., None])
    inscatter = result.cpu().numpy()

    # Beyond the last slice, the closed form without shadows adds the light scattered between it and the surface
    far = scene.depth[i] > END
    if far.any():
        whole = scene.closed_form(i, occlusion=None, normals=None).inscatter.cpu().numpy()
        clamped = np.minimum(scene.depth[i], END)
        t = scene.transforms[i]
        near_part = (
            apply_medium(
                scene.clear[i],
                clamped,
                t,
                t["transform_matrix"],
                scene.medium,
                scene.lighting,
                dtype=DTYPE,
                device=DEVICE,
            )
            .inscatter.cpu()
            .numpy()
        )
        inscatter = inscatter + np.where(far[..., None], whole - near_part, 0.0)
    return inscatter, time.perf_counter() - start


def frame_times(name: str) -> list[float]:
    text = (OUT / "logs" / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    return [float(s) for f, s in re.findall(r"FRAME_TIME frame=(\d+) seconds=([\d.]+)", text) if int(f) != 254]


def main():
    results: dict = {"froxel": {"day": [], "night": []}, "eevee": {}}
    day = day_scene(frames=(255, 345))
    night = night_scene(NIGHT_FRAMES)
    lights = {"day": DayLight(day), "night": NightLight(night)}
    references = {"day": np.stack([day.closed_form(i).inscatter.numpy() for i in range(2)])}
    references["night"] = np.stack([night.closed_form(i).inscatter.numpy() for i in range(2)])
    saved = {}
    for scene in (day, night):
        for tile, slices, jitter in ((8, 64, 1), (8, 64, 8), (8, 128, 8), (4, 128, 4), (2, 128, 1)):
            images, seconds = [], []
            for i in range(2):
                image, s = froxels(scene, lights[scene.name], i, tile, slices, jitter)
                images.append(image)
                seconds.append(s)
            images = np.stack(images)
            entry = {
                "tile": tile,
                "slices": slices,
                "jitter": jitter,
                "froxels": int(-(-180 // tile) * -(-320 // tile) * slices),
                "seconds": min(seconds),
                "vs_closed_form": errors(images, references[scene.name]),
                "vs_cycles": errors(np.stack([scene.output(i, images[i]) for i in range(2)]), scene.reference),
            }
            results["froxel"][scene.name].append(entry)
            print(scene.name, json.dumps(entry), flush=True)
            if (tile, slices, jitter) in ((8, 64, 1), (4, 128, 4)):
                saved[f"{scene.name}_{tile}_{slices}_{jitter}"] = images[1]

    # EEVEE, as rendered
    day_all = day_scene(DAY_FRAMES)
    still = [DAY_FRAMES.index(f) for f in (255, 345)]
    sequence = [DAY_FRAMES.index(f) for f in range(255, 263)]
    volume = load_images(ROOT / "ms" / "cycles_ref" / "volume" / "direct", (255, 345))
    volume_all = volume + load_images(ROOT / "ms" / "cycles_ref" / "volume" / "indirect", (255, 345))
    for name in ("eevee_day_default", "eevee_day_fine", "eevee_day_1sample"):
        frames = DAY_FRAMES if name != "eevee_day_fine" else (255, 345)
        images = load_images(OUT / name / "frames", frames)
        fog_light = load_images(OUT / name / "volume" / "light", frames)
        index = {f: k for k, f in enumerate(frames)}
        entry = {
            "seconds": float(np.mean(frame_times(name))),
            "whole": errors(images[[index[255], index[345]]], day_all.reference[still]),
            "fog_light_vs_single": errors(fog_light[[index[255], index[345]]], volume),
            "fog_light_vs_multiple": errors(fog_light[[index[255], index[345]]], volume_all),
            "composited": errors(
                np.stack([day_all.output(i, fog_light[index[f]]) for i, f in zip(still, (255, 345))]),
                day_all.reference[still],
            ),
        }
        if len(frames) > 2:
            entry["flicker"] = flicker(images[[index[f] for f in range(255, 263)]], day_all.reference[sequence])
        results["eevee"][name] = entry
        print(name, json.dumps(entry), flush=True)
        if name == "eevee_day_default":
            saved["eevee_day"] = images[index[345]]
    for name in ("eevee_night_default", "eevee_night_fine"):
        fog_light = load_images(OUT / name / "volume" / "light", NIGHT_FRAMES)
        entry = {
            "seconds": float(np.mean(frame_times(name))),
            "fog_light": errors(fog_light, night.reference),
            "whole": errors(
                load_images(OUT / name / "frames", NIGHT_FRAMES),
                load_images(ROOT / "point_shadow" / "bounces0" / "frames", NIGHT_FRAMES),
            ),
        }
        results["eevee"][name] = entry
        print(name, json.dumps(entry), flush=True)
        if name == "eevee_night_default":
            saved["eevee_night"] = fog_light[1]
    np.savez_compressed(OUT / "results" / "froxel_images.npz", **saved)
    (OUT / "results" / "froxel.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
