"""Direction 4: computing less, by exploiting how smooth and redundant the fog's light is.

- Low resolution: the closed form at 1/s of the resolution in each direction, through a subset of the camera's rays,
  upsampled to the full resolution, either bilinearly, or guided by depth (joint bilateral), and, best, by upsampling
  the mean source along rays ``J = inscatter / (1 - T)``, guided by depth, then multiplying it by the transmittance
  ``1 - T`` computed exactly at the full resolution, which is cheap. At 1920x1080, against the closed form at full
  resolution (``exp_closed_form.py``), and at 320x180 against Cycles.
- Reuse across frames: the mean source of a key frame reprojected to later frames through depth, where it was seen
  (elsewhere, the closed form without shadows fills in), against computing each frame.
- Shadows traced once, for fogs of several densities.

Results are saved to ``results/less.json``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from visionsim.medium import apply_medium, trace_shadows
from visionsim.medium.optics import height_fog_optical_depth
from visionsim.medium.render import camera_rays

from common import (
    DEVICE,
    DTYPE,
    NIGHT_FRAMES,
    OUT,
    ROOT,
    day_scene,
    errors,
    load_images,
    load_transforms,
    night_scene,
)


def low_camera(camera: dict, s: int) -> dict:
    """Intrinsics of a camera with 1/s of the pixels along each axis, whose rays are those through the full
    resolution's pixels (s i + s/2, s j + s/2)."""
    return dict(camera) | {
        "fl_x": camera["fl_x"] / s,
        "fl_y": camera["fl_y"] / s,
        "cx": (camera["cx"] - 0.5) / s,
        "cy": (camera["cy"] - 0.5) / s,
        "w": int(camera["w"]) // s,
        "h": int(camera["h"]) // s,
    }


def subsample(image: np.ndarray, s: int) -> np.ndarray:
    return image[s // 2 :: s, s // 2 :: s][: image.shape[0] // s, : image.shape[1] // s]


def transmittance(scene_medium, camera: dict, depth: np.ndarray) -> np.ndarray:
    origin, directions, scale = camera_rays(camera, camera["transform_matrix"], dtype=torch.float64)
    distance = torch.as_tensor(depth, dtype=torch.float64) * scale
    tau = height_fog_optical_depth(scene_medium.components[0], origin, directions, distance) * scene_medium.extinction
    return torch.exp(-tau)[..., None].float().numpy()


def upsample(values: np.ndarray, low_depth: np.ndarray, depth: np.ndarray, s: int, guided: bool, sigma=0.05):
    """Upsample values at the subsampled pixels to every pixel, bilinearly, and, if guided, weighted by how close the
    depths are (joint bilateral), falling back on the neighbor of closest depth."""
    v, d, high = (torch.as_tensor(x, dtype=torch.float32, device=DEVICE) for x in (values, low_depth, depth))
    hl, wl = d.shape
    H, W = high.shape
    fy = ((torch.arange(H, dtype=torch.float32, device=DEVICE) - s // 2) / s).clamp(0, hl - 1)
    fx = ((torch.arange(W, dtype=torch.float32, device=DEVICE) - s // 2) / s).clamp(0, wl - 1)
    y0, x0 = fy.floor().clamp(max=hl - 2).long(), fx.floor().clamp(max=wl - 2).long()
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    total, weight_sum = 0.0, 0.0
    best, best_gap = None, None
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = (y0 + dy)[:, None].expand(H, W), (x0 + dx)[None, :].expand(H, W)
            w = (wy if dy else 1 - wy) * (wx if dx else 1 - wx)
            gap = (torch.log(d[yi, xi]) - torch.log(high)).abs()
            if guided:
                w = w * torch.exp(-((gap / sigma) ** 2) / 2)
            total = total + w[..., None] * v[yi, xi]
            weight_sum = weight_sum + w
            if best is None:
                best, best_gap = v[yi, xi].clone(), gap.clone()
            else:
                closer = gap < best_gap
                best[closer], best_gap[closer] = v[yi, xi][closer], gap[closer]
    result = total / weight_sum.clamp_min(1e-12)[..., None]
    if guided:
        result = torch.where((weight_sum < 1e-3)[..., None], best, result)
    return result.cpu().numpy()


def low_resolution(clear, depth, camera, medium, lighting, occlusion, s: int, normals=None) -> dict:
    """The closed form at 1/s of the resolution, upsampled in three ways, and its time."""
    small = low_camera(camera, s)
    low_depth = subsample(depth, s)
    start = time.perf_counter()
    result = apply_medium(
        subsample(clear, s),
        low_depth,
        small,
        camera["transform_matrix"],
        medium,
        lighting,
        occlusion=occlusion,
        dtype=DTYPE,
        device=DEVICE,
    )
    inscatter, t_low = result.inscatter.cpu().numpy(), result.transmittance.cpu().numpy()
    seconds = {"low": time.perf_counter() - start}
    start = time.perf_counter()
    t_full = transmittance(medium, camera, depth)
    seconds["transmittance"] = time.perf_counter() - start
    start = time.perf_counter()
    mean = np.where(1 - t_low > 1e-6, inscatter / np.maximum(1 - t_low, 1e-6), 0.0)
    out = {
        "bilinear": upsample(inscatter, low_depth, depth, s, guided=False),
        "guided": upsample(inscatter, low_depth, depth, s, guided=True),
        "guided_source": upsample(mean, low_depth, depth, s, guided=True) * (1 - t_full),
    }
    seconds["upsample"] = (time.perf_counter() - start) / 3
    return {"inscatter": out, "transmittance": t_full, "seconds": seconds}


def hd_experiment() -> dict:
    day = day_scene(frames=(255,))
    full = np.load(OUT / "results" / "closed_form_hd.npz")
    hd = ROOT / "hd" / "clear"
    camera = load_transforms(hd / "frames", (255,))[0]
    clear = load_images(hd / "frames", (255,))[0]
    depth = full["depth"]
    results = {}
    for s in (2, 4, 8):
        low = low_resolution(clear, depth, camera, day.medium, day.lighting, day.occlusion, s)
        entry = {"seconds": low["seconds"]}
        for name, inscatter in low["inscatter"].items():
            image = low["transmittance"] * clear + inscatter
            entry[name] = {
                "inscatter": errors(inscatter[None], full["inscatter"][None]),
                "image": errors(image[None], full["radiance"][None]),
            }
        results[s] = entry
        print("hd", s, json.dumps(entry), flush=True)
        if s == 4:
            np.savez_compressed(
                OUT / "results" / "less_hd_images.npz",
                full=full["inscatter"][300:620, 700:1260],
                bilinear=low["inscatter"]["bilinear"][300:620, 700:1260],
                guided_source=low["inscatter"]["guided_source"][300:620, 700:1260],
            )
    return results


def small_experiment() -> dict:
    """At 320x180, against Cycles, for the day and night scenes."""
    results: dict = {}
    day = day_scene(frames=(255, 345))
    night = night_scene(NIGHT_FRAMES)
    for scene in (day, night):
        entry: dict = {}
        cf = np.stack([scene.closed_form(i, normals=None).inscatter.numpy() for i in range(2)])
        for s in (2, 4):
            out = {name: [] for name in ("bilinear", "guided", "guided_source")}
            seconds = []
            for i in range(2):
                camera = scene.transforms[i]
                low = low_resolution(
                    scene.clear[i], scene.depth[i], camera, scene.medium, scene.lighting, scene.occlusion, s
                )
                seconds.append(low["seconds"])
                for name, inscatter in low["inscatter"].items():
                    out[name].append(inscatter)
            entry[s] = {
                name: {
                    "vs_closed_form": errors(np.stack(images), cf),
                    "vs_cycles": errors(np.stack([scene.output(i, images[i]) for i in range(2)]), scene.reference),
                }
                for name, images in out.items()
            }
            entry[s]["seconds"] = seconds[1]
        results[scene.name] = entry
        print("small", scene.name, json.dumps(entry), flush=True)
    return results


def reproject(key: dict, target: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean source of a key frame looked up at the points seen by a target frame, and where it was seen."""
    origin_t, dirs_t, scale_t = camera_rays(target["camera"], target["camera"]["transform_matrix"], dtype=torch.float64)
    points = origin_t + dirs_t * (torch.as_tensor(target["depth"], dtype=torch.float64) * scale_t)[..., None]
    pose = torch.as_tensor(np.asarray(key["camera"]["transform_matrix"], dtype=float), dtype=torch.float64)
    rotation = pose[:3, :3] / pose[:3, :3].norm(dim=0)
    local = (points - pose[:3, 3]) @ rotation
    camera = key["camera"]
    u = camera["fl_x"] * local[..., 0] / -local[..., 2] + camera["cx"] - 0.5
    v = camera["cy"] - camera["fl_y"] * local[..., 1] / -local[..., 2] - 0.5
    distance = (points - pose[:3, 3]).norm(dim=-1)
    h, w = key["depth"].shape
    _, key_dirs, key_scale = camera_rays(camera, camera["transform_matrix"], dtype=torch.float64)
    key_distance = torch.as_tensor(key["depth"], dtype=torch.float64) * key_scale
    mean = torch.as_tensor(key["mean"], dtype=torch.float64)
    x0, y0 = u.floor().long(), v.floor().long()
    total, weights = 0.0, 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (local[..., 2] < 0)
            xc, yc = xi.clamp(0, w - 1), yi.clamp(0, h - 1)
            weight = (1 - (u - xi).abs()).clamp_min(0) * (1 - (v - yi).abs()).clamp_min(0)
            # Only where the key frame saw the same surface, or sky
            same = ((key_distance[yc, xc] - distance).abs() / distance) < 0.03
            weight = weight * (valid & same)
            total = total + weight[..., None] * mean[yc, xc]
            weights = weights + weight
    seen = weights > 0.25
    return (total / weights.clamp_min(1e-12)[..., None]).float().numpy(), seen.numpy()


def reuse_experiment() -> dict:
    offsets = (1, 2, 4, 8, 16, 32)
    frames = (255,) + tuple(255 + k for k in offsets)
    day = day_scene(frames=frames)
    results: dict = {}
    key_result = day.closed_form(0, normals=None)
    t_key = key_result.transmittance.numpy()
    key = {
        "camera": day.transforms[0],
        "depth": day.depth[0],
        "mean": np.where(1 - t_key > 1e-6, key_result.inscatter.numpy() / np.maximum(1 - t_key, 1e-6), 0.0),
    }
    for j, k in enumerate(offsets, start=1):
        exact = day.closed_form(j, normals=None)
        start = time.perf_counter()
        mean, seen = reproject(key, {"camera": day.transforms[j], "depth": day.depth[j]})
        t = transmittance(day.medium, day.transforms[j], day.depth[j])
        seconds = time.perf_counter() - start
        # Where the key frame didn't see the point, the closed form without shadows fills in
        start = time.perf_counter()
        unshadowed = day.closed_form(j, normals=None, occlusion=None).inscatter.numpy()
        fill_seconds = time.perf_counter() - start
        inscatter = np.where(seen[..., None], mean * (1 - t), unshadowed)
        results[k] = {
            "seen": float(seen.mean()),
            "seconds": seconds,
            "fill_seconds": fill_seconds,
            "vs_closed_form": errors(inscatter[None], exact.inscatter.numpy()[None]),
            "vs_cycles": errors(day.output(j, inscatter)[None], day.reference[j : j + 1]),
            "closed_form_vs_cycles": errors(day.output(j, exact.inscatter.numpy())[None], day.reference[j : j + 1]),
        }
        print("reuse", k, json.dumps(results[k]), flush=True)
    return results


def density_experiment() -> dict:
    """Shadows traced once per frame, then fogs of several densities, against tracing them for each fog."""
    day = day_scene(frames=(345,))
    t = day.transforms[0]
    scales = (0.5, 1.0, 2.0, 4.0)
    media = [day.medium.model_copy(update={"extinction": day.medium.extinction * s}) for s in scales]
    args = (day.clear[0], day.depth[0], t, t["transform_matrix"])
    start = time.perf_counter()
    separate = [apply_medium(*args, m, day.lighting, occlusion=day.occlusion, dtype=DTYPE).radiance for m in media]
    separate_seconds = time.perf_counter() - start
    start = time.perf_counter()
    shadows = trace_shadows(day.occlusion, day.depth[0], t, t["transform_matrix"], day.medium, day.lighting, dtype=DTYPE)
    trace_seconds = time.perf_counter() - start
    start = time.perf_counter()
    shared = [apply_medium(*args, m, day.lighting, shadows=shadows, dtype=DTYPE).radiance for m in media]
    shade_seconds = time.perf_counter() - start
    return {
        "scales": scales,
        "separate_seconds": separate_seconds,
        "trace_seconds": trace_seconds,
        "shade_seconds": shade_seconds,
        "errors": [errors(a.numpy()[None], b.numpy()[None]) for a, b in zip(shared, separate)],
    }


def main():
    results = {"density": density_experiment()}
    print("density", json.dumps(results["density"]), flush=True)
    results["hd"] = hd_experiment()
    results["small"] = small_experiment()
    results["reuse"] = reuse_experiment()
    (OUT / "results" / "less.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
