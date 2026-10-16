"""Where and why the "compute less" shortcuts of ``exp_less.py`` depart from the closed form they approximate.

All images are 320x180. Every error here is against the closed form computed for every pixel and frame (the method
being accelerated), not against Cycles, so it isolates what each shortcut adds; the closed form's own error against
Cycles is given alongside for scale.

1. Low resolution: the closed form through 1/s of the camera's rays along each axis (s = 2: 160x90 rays, s = 4: 80x45),
   upsampled to 320x180 bilinearly, guided by depth (joint bilateral), or by upsampling the mean source along rays
   ``J = inscatter / (1 - T)`` guided by depth and multiplying by the exact ``1 - T``. Errors are located by region
   (object silhouettes, horizon, light shaft edges, sky, flat ground, object faces, and at night the lamp's halo core)
   and by light component (sunlight scattered once, and the rest: sky, ground and multiple scattering, day only), and
   compared with the same upsampling of the fog's light computed without any shadows.
2. Reuse across frames: the mean source of key frame 255 reprojected to later frames. The error is split into pixels
   the key frame didn't see (filled with the closed form without shadows) and seen pixels, whose error is tested
   against three causes: the sun's phase function changes with the view direction (corrected by reprojecting
   ``J_sun / HG(key direction)`` and multiplying by ``HG(target direction)``), the path of the ray changes with the
   camera's position (shadows along the extra or missing stretch of ray, seen by comparing with and without shadows),
   and the coarse sampling of the sky's visibility along rays (8 samples, compared with 32).
3. Shadows shared across densities: shadows traced for the original fog reused for fogs 0.5x and 4x as dense. The
   difference is split into the sun's shadow segments and the sky's visibility samples, by swapping one or the other,
   and both are compared with shadows traced with 64 sky samples, which is close to converged.

Outputs go to ``runs/fog-comparison/acceleration/analysis/compute_less/``: ``compute_less.json`` and PNG maps.

Run from the repository root, on the CPU::

    CUDA_VISIBLE_DEVICES="" python examples/medium/experiment/acceleration/analysis/compute_less.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *

torch.set_num_threads(3)  # other jobs share the CPU

import matplotlib

import exp_less

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from visionsim.medium import apply_medium, trace_shadows
from visionsim.medium.optics import height_fog_optical_depth, henyey_greenstein
from visionsim.medium.render import camera_rays

SAVE = OUT / "analysis" / "compute_less"
FRAMES = (255, 345)
EDGE_RADIUS = 4
"""pixels around an edge counted as the edge, as far as a 1/4 resolution sample reaches"""
DEPTH_JUMP = 0.1
"""change of the log distance between neighbors that marks a depth edge"""
SHADOW_JUMP = {"day": 0.05, "night": 0.02}
"""change between neighbors of the ratio of light with and without shadows that marks a shadow edge: of sunlight
scattered once by day, whose light shafts have sharp edges, and of the lamp's light at night, whose shadow wedge has
soft edges once integrated along rays"""
SKY_SAMPLES_FINE = 32
"""samples of the sky's visibility along rays, instead of 8, to tell its coarse sampling apart from other causes"""
SUN_NODES, SMOOTH_NODES = 512, 48
"""nodes along rays to march the day's lighting, as ``exp_cache.py``"""
UPSAMPLERS = ("bilinear", "guided", "guided_source")


# --------------------------------------------------------------------------------------------------------------------
# Helpers


def luminance(image: np.ndarray) -> np.ndarray:
    return image @ LUMA


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return ndimage.binary_dilation(mask, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))


def local_range(image: np.ndarray) -> np.ndarray:
    """Largest difference between a pixel's 3x3 neighbors."""
    return ndimage.maximum_filter(image, 3) - ndimage.minimum_filter(image, 3)


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return luminance(a) / np.maximum(luminance(b), 1e-12)


def distance_of(transform: dict, depth: np.ndarray) -> np.ndarray:
    """Distance along each camera ray to the surface, or to the fog box for the sky."""
    _, _, scale = camera_rays(transform, transform["transform_matrix"])
    return depth * scale.numpy()


def depth_edges(transform: dict, depth: np.ndarray) -> np.ndarray:
    return dilate(local_range(np.log(distance_of(transform, depth))) > DEPTH_JUMP, EDGE_RADIUS)


def world_normals(frames) -> np.ndarray:
    """World-space normals of both scenes, which share the day scene's geometry and camera."""
    transforms = load_transforms(ROOT / "clear" / "frames", frames)
    normals = load_images(ROOT / "clear" / "normals", frames)
    rotations = [np.asarray(t["transform_matrix"])[:3, :3] for t in transforms]
    return np.stack([n @ (r / np.linalg.norm(r, axis=0)) for n, r in zip(normals, rotations)])


def regions(transform, depth, raw_depth, normals, shadow_ratio, light=None, shadow_jump=SHADOW_JUMP["day"]):
    """Disjoint masks of one frame, by priority: object silhouettes, horizon, shadow edges, the lamp's halo core
    (night, given its light), then sky, flat ground (normal z > 0.95) and object faces.

    - silhouettes: within EDGE_RADIUS px of a jump of the distance by more than 10% between neighbors, next to an
      object; horizon: other such jumps (far ground against the sky, where the distance changes quickly);
    - shadow edges: within EDGE_RADIUS px of a change of more than ``shadow_jump`` between neighbors of
      ``shadow_ratio``, the light with over without shadows: of sunlight scattered once by day (edges of light
      shafts), of the lamp's light at night (edges of the cube's shadow wedge);
    - halo core: the brightest 1% of the fog's light.
    """
    sky = raw_depth >= 1e9
    ground = (normals[..., 2] > 0.95) & ~sky
    objects = ~sky & ~ground
    jump = local_range(np.log(distance_of(transform, depth))) > DEPTH_JUMP
    silhouette = dilate(jump & dilate(objects, 1), EDGE_RADIUS)
    horizon = dilate(jump, EDGE_RADIUS) & ~silhouette
    taken = silhouette | horizon
    shadow_edge = dilate(local_range(shadow_ratio) > shadow_jump, EDGE_RADIUS) & ~taken
    taken = taken | shadow_edge
    masks = {"silhouette": silhouette, "horizon": horizon, "shadow_edge": shadow_edge}
    if light is not None:
        masks["halo_core"] = (light >= np.percentile(light, 99)) & ~taken
        taken = taken | masks["halo_core"]
    masks |= {"sky": sky & ~taken, "ground": ground & ~taken, "object_faces": objects & ~taken}
    return masks


def by_region(ours: np.ndarray, exact: np.ndarray, masks: dict[str, np.ndarray], total: float) -> dict:
    """Per region: its share of pixels, its share of the total error (sum over the region of |ours - exact| over
    ``total``), its signed bias over ``total``, and its own relative error (over the region's exact light)."""
    error, signed = np.abs(ours - exact).sum(-1), (ours - exact).sum(-1)
    light = exact.sum(-1)
    return {
        name: {
            "pixels": float(mask.mean()),
            "error_share": float(error[mask].sum() / total),
            "bias_share": float(signed[mask].sum() / total),
            "local_rel_l1": float(error[mask].sum() / max(light[mask].sum(), 1e-12)),
        }
        for name, mask in masks.items()
    }


def stack_masks(per_frame: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([m[name] for m in per_frame]) for name in per_frame[0]}


def l1(ours: np.ndarray, exact: np.ndarray, total: float, where=1.0) -> float:
    return float((np.abs(ours - exact) * where).sum() / total)


def error_map(ours: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """|ours - exact| summed over channels, over the frame's mean light summed over channels."""
    return np.abs(ours - exact).sum(-1) / exact.sum(-1).mean()


def show(image: np.ndarray) -> np.ndarray:
    """Linear light to display: scaled to its 99.5th percentile, gamma 2.2."""
    return np.clip(image / np.percentile(image, 99.5), 0, 1) ** (1 / 2.2)


REGION_COLORS = {
    "silhouette": "#d62728",
    "horizon": "#ff7f0e",
    "shadow_edge": "#9467bd",
    "halo_core": "#e377c2",
    "sky": "#7fb2e5",
    "ground": "#8c8c5a",
    "object_faces": "#2ca02c",
}


def draw_regions(ax, masks: dict[str, np.ndarray], title: str = "regions") -> None:
    names = list(masks)
    label = np.zeros(next(iter(masks.values())).shape, dtype=int)
    for k, name in enumerate(names):
        label[masks[name]] = k
    cmap = ListedColormap([REGION_COLORS[n] for n in names])
    ax.imshow(label, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[n]) for n in names]
    ax.legend(handles, names, fontsize=7, loc="lower left", framealpha=0.8)


def no_ticks(axes) -> None:
    for ax in np.ravel(axes):
        ax.set_xticks([]), ax.set_yticks([])


def sun_only(scene: Scene) -> tuple[Medium, Lighting]:
    """Medium and lighting with only sunlight scattered once: no sky, no ground light, no multiple scattering."""
    medium = scene.medium.model_copy(update={"multiple_scattering": False})
    lighting = scene.lighting.model_copy(update={"sky": (0.0, 0.0, 0.0), "ground_albedo": (0.0, 0.0, 0.0)})
    return medium, lighting


# --------------------------------------------------------------------------------------------------------------------
# 1. Low resolution


def low_resolution_study() -> dict:
    day, night = day_scene(frames=FRAMES), night_scene(FRAMES)
    raw_depth = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    normals = world_normals(FRAMES)
    results: dict = {}
    for scene in (day, night):
        n = len(FRAMES)
        is_day = scene.name == "day"
        sun_medium, sun_lighting = sun_only(scene)

        def closed_form(i, sun=False, occlusion=scene.occlusion):
            options = {"medium": sun_medium, "lighting": sun_lighting} if sun else {}
            return scene.closed_form(i, normals=None, occlusion=occlusion, **options).inscatter.numpy()

        exact = np.stack([closed_form(i) for i in range(n)])
        unshadowed = np.stack([closed_form(i, occlusion=None) for i in range(n)])
        if is_day:
            exact_sun = np.stack([closed_form(i, sun=True) for i in range(n)])
            unshadowed_sun = np.stack([closed_form(i, sun=True, occlusion=None) for i in range(n)])
            shadow_ratio = ratio(exact_sun, unshadowed_sun)
        else:
            shadow_ratio = ratio(exact, unshadowed)
        # The whole image (surfaces lit through the fog plus the fog's light) by day, the fog's light at night
        whole = np.stack([scene.closed_form(i).radiance.numpy() for i in range(n)]) if is_day else exact
        masks_per_frame = [
            regions(
                scene.transforms[i],
                scene.depth[i],
                raw_depth[i],
                normals[i],
                shadow_ratio[i],
                light=None if is_day else luminance(exact[i]),
                shadow_jump=SHADOW_JUMP[scene.name],
            )
            for i in range(n)
        ]
        masks = stack_masks(masks_per_frame)
        entry: dict = {
            "closed_form_vs_cycles": errors(whole, scene.reference),
            "fog_light_share_of_whole_image": float(exact.sum() / whole.sum()),
            "region_pixels": {name: float(m.mean()) for name, m in masks.items()},
        }
        if is_day:
            entry["sun_single_scattering_share_of_fog_light"] = float(exact_sun.sum() / exact.sum())
        maps = {}
        for s in (2, 4):

            def low(i, medium=scene.medium, lighting=scene.lighting, occlusion=scene.occlusion):
                result = exp_less.low_resolution(
                    scene.clear[i], scene.depth[i], scene.transforms[i], medium, lighting, occlusion, s
                )
                return result["inscatter"]

            full = [low(i) for i in range(n)]
            free = [low(i, occlusion=None) for i in range(n)]
            sun = [low(i, sun_medium, sun_lighting) for i in range(n)] if is_day else None
            entry[f"1/{s}"] = {}
            for name in UPSAMPLERS:
                ours = np.stack([f[name] for f in full])
                ours_free = np.stack([f[name] for f in free])
                v = {
                    "rays": f"{320 // s}x{180 // s}",
                    "vs_closed_form_fog_light": errors(ours, exact),
                    "rel_l1_of_whole_image": l1(ours, exact, whole.sum()),
                    "without_shadows_rel_l1": l1(ours_free, unshadowed, unshadowed.sum()),
                    "regions": by_region(ours, exact, masks, exact.sum()),
                }
                if is_day:
                    ours_sun = np.stack([f[name] for f in sun])
                    v["sun_part_error_share"] = l1(ours_sun, exact_sun, exact.sum())
                    v["rest_part_error_share"] = l1(ours - ours_sun, exact - exact_sun, exact.sum())
                entry[f"1/{s}"][name] = v
                maps[s, name] = error_map(ours[-1], exact[-1])
            print(
                "low resolution",
                scene.name,
                s,
                json.dumps({k: round(v["vs_closed_form_fog_light"]["rel_l1"], 5) for k, v in entry[f"1/{s}"].items()}),
                flush=True,
            )
        if is_day:
            entry["finer_sky_sampling"] = finer_sky_sampling(scene, exact)
        results[scene.name] = entry
        plot_low_resolution(scene.name, exact[-1], shadow_ratio[-1], masks_per_frame[-1], maps)
    return results


def low_resolution_with(scene: Scene, i: int, s: int, **options) -> dict[str, np.ndarray]:
    """As :func:`exp_less.low_resolution`, with options of :func:`apply_medium`, such as more sky samples."""
    camera = scene.transforms[i]
    low_depth = exp_less.subsample(scene.depth[i], s)
    result = apply_medium(
        exp_less.subsample(scene.clear[i], s),
        low_depth,
        exp_less.low_camera(camera, s),
        camera["transform_matrix"],
        scene.medium,
        scene.lighting,
        occlusion=scene.occlusion,
        dtype=DTYPE,
        device=DEVICE,
        **options,
    )
    inscatter, t_low = result.inscatter.numpy(), result.transmittance.numpy()
    t_full = exp_less.transmittance(scene.medium, camera, scene.depth[i])
    mean = np.where(1 - t_low > 1e-6, inscatter / np.maximum(1 - t_low, 1e-6), 0.0)
    depth = scene.depth[i]
    return {
        "bilinear": exp_less.upsample(inscatter, low_depth, depth, s, guided=False),
        "guided": exp_less.upsample(inscatter, low_depth, depth, s, guided=True),
        "guided_source": exp_less.upsample(mean, low_depth, depth, s, guided=True) * (1 - t_full),
    }


def finer_sky_sampling(scene: Scene, exact: np.ndarray) -> dict:
    """Upsampling error when the sky's visibility is sampled 32 times along rays instead of 8, both for the low
    resolution and the full resolution it is compared with: the 8 samples give the fog's light a blocky texture at
    the scale of pixels (where samples snap to the grid of the sky's maps), which no upsampler can recover."""
    fine = {"shadow_samples": SKY_SAMPLES_FINE}
    n = len(scene.frames)
    exact_fine = np.stack([scene.closed_form(i, normals=None, **fine).inscatter.numpy() for i in range(n)])
    entry = {"8_vs_32_samples_full_resolution": errors(exact, exact_fine)}
    for s in (2, 4):
        images = [low_resolution_with(scene, i, s, **fine) for i in range(n)]
        entry[f"1/{s}"] = {name: errors(np.stack([image[name] for image in images]), exact_fine) for name in UPSAMPLERS}
    print("finer sky sampling", json.dumps(entry), flush=True)
    return entry


def plot_low_resolution(name: str, exact, shadow_ratio, masks, maps) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 8.2), constrained_layout=True)
    axes[0, 0].imshow(show(exact))
    axes[0, 0].set_title(f"{name}: closed form fog light, frame 345 (320x180)")
    im = axes[0, 1].imshow(shadow_ratio, cmap="gray", vmin=0, vmax=1)
    what = "sunlight scattered once" if name == "day" else "lamp light"
    axes[0, 1].set_title(f"{what} with / without shadows")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    draw_regions(axes[0, 2], masks)
    vmax = float(np.percentile(maps[4, "bilinear"], 99.5))
    for row, s in enumerate((2, 4), start=1):
        for col, method in enumerate(UPSAMPLERS):
            m = maps[s, method]
            im = axes[row, col].imshow(m, cmap="magma", vmin=0, vmax=vmax)
            axes[row, col].set_title(f"1/{s} ({320 // s}x{180 // s} rays), {method}: rel L1 {m.mean():.2%}")
            fig.colorbar(im, ax=axes[row, col], shrink=0.8, label="|err| / mean fog light")
    no_ticks(axes)
    fig.savefig(SAVE / f"low_resolution_{name}_345.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# 2. Reuse of a key frame


def mean_source(inscatter: np.ndarray, transmittance: np.ndarray) -> np.ndarray:
    return np.where(1 - transmittance > 1e-6, inscatter / np.maximum(1 - transmittance, 1e-6), 0.0)


def sun_phase(scene: Scene, i: int) -> np.ndarray:
    """Henyey-Greenstein phase function of sunlight scattered towards the camera along the rays of frame i,
    of shape (h, w, 1)."""
    t = scene.transforms[i]
    _, directions, _ = camera_rays(t, t["transform_matrix"])
    towards = torch.as_tensor(scene.lighting.suns[0].direction, dtype=torch.float64)
    return henyey_greenstein(directions @ (towards / towards.norm()), scene.medium.anisotropy)[..., None].numpy()


REUSE_VARIANTS = {
    "shadows": {},
    "shadows_32_sky_samples": {"shadow_samples": 32},
    "no_shadows": {"occlusion": None},
}
"""Closed forms reused: as the method does, with 4x as many samples of the sky's visibility, and without shadows"""


def reuse_study() -> dict:
    offsets = (1, 2, 4, 8, 16, 32)
    frames = (255,) + tuple(255 + k for k in offsets)
    day = day_scene(frames=frames)
    raw_depth = load_images(ROOT / "clear" / "depths", frames)[..., 0]
    sun_medium, sun_lighting = sun_only(day)

    def components(i: int, options: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fog light of frame i, its sunlight scattered once, and the transmittance."""
        full = day.closed_form(i, normals=None, **options)
        sun = day.closed_form(i, medium=sun_medium, lighting=sun_lighting, normals=None, **options)
        return full.inscatter.numpy(), sun.inscatter.numpy(), full.transmittance.numpy()

    pose0 = np.asarray(day.transforms[0]["transform_matrix"])
    keys = {}
    for variant, options in REUSE_VARIANTS.items():
        inscatter, sun, t = components(0, options)
        j_sun, j_all = mean_source(sun, t), mean_source(inscatter, t)
        # Reprojected together: the rest (sky, ground, multiple scattering), the sun's part, and the sun's part
        # without the phase function of the key frame's direction
        keys[variant] = {
            "camera": day.transforms[0],
            "depth": day.depth[0],
            "mean": np.concatenate([j_all - j_sun, j_sun, j_sun / sun_phase(day, 0)], axis=-1),
        }
    results: dict = {}
    maps: dict = {}
    for j, k in enumerate(offsets, start=1):
        target = {"camera": day.transforms[j], "depth": day.depth[j]}
        pose = np.asarray(day.transforms[j]["transform_matrix"])
        r0, r1 = (p[:3, :3] / np.linalg.norm(p[:3, :3], axis=0) for p in (pose0, pose))
        entry: dict = {
            "frame": 255 + k,
            "camera_displacement_m": float(np.linalg.norm(pose[:3, 3] - pose0[:3, 3])),
            "camera_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1)))),
        }
        t_target = exp_less.transmittance(day.medium, day.transforms[j], day.depth[j])
        phase = sun_phase(day, j)
        exacts = {variant: components(j, options) for variant, options in REUSE_VARIANTS.items()}
        unshadowed = exacts["no_shadows"][0]
        masks = regions(
            day.transforms[j],
            day.depth[j],
            raw_depth[j],
            world_normals((255 + k,))[0],
            ratio(exacts["shadows"][1], exacts["no_shadows"][1]),
        )
        edges = depth_edges(day.transforms[j], day.depth[j])
        for variant in REUSE_VARIANTS:
            exact, exact_sun, _ = exacts[variant]
            mean, seen = exp_less.reproject(keys[variant], target)
            rest, sun, sun_over_phase = mean[..., :3], mean[..., 3:6], mean[..., 6:]
            naive = (rest + sun) * (1 - t_target)
            corrected = (rest + sun_over_phase * phase) * (1 - t_target)
            # Where the key frame didn't see the point, the closed form without shadows fills in
            used = np.where(seen[..., None], naive, unshadowed)
            total, unseen, seen3 = exact.sum(), ~seen[..., None], seen[..., None]
            # How much the sun's phase function changed between the key frame's ray and the target's
            lit = seen & (sun_over_phase[..., 0] > 1e-12)
            phase_key = sun[lit] / sun_over_phase[lit]
            change = np.abs(phase[lit] / phase_key - 1)[:, 0]
            v = {
                "seen_pixels": float(seen.mean()),
                "rel_l1": l1(used, exact, total),
                "bias": float((used - exact).sum() / total),
                "unseen_error_share": l1(unshadowed, exact, total, unseen),
                "unseen_bias_share": float(((unshadowed - exact) * unseen).sum() / total),
                "seen_error_share": l1(naive, exact, total, seen3),
                "seen_bias_share": float(((naive - exact) * seen3).sum() / total),
                "seen_error_share_sun_part": l1(sun * (1 - t_target), exact_sun, total, seen3),
                "seen_error_share_sun_part_phase_corrected": l1(
                    sun_over_phase * phase * (1 - t_target), exact_sun, total, seen3
                ),
                "seen_error_share_rest_part": l1(rest * (1 - t_target), exact - exact_sun, total, seen3),
                "seen_error_share_phase_corrected": l1(corrected, exact, total, seen3),
                "seen_error_share_near_depth_edges": l1(naive, exact, total, (seen & edges)[..., None]),
                "sun_phase_change_median": float(np.median(change)),
                "sun_phase_change_p95": float(np.percentile(change, 95)),
                "sun_single_scattering_share_of_fog_light": float(exact_sun.sum() / total),
            }
            if variant == "shadows":
                h, w = seen.shape
                border = ~np.pad(np.ones((h - 2 * EDGE_RADIUS, w - 2 * EDGE_RADIUS), dtype=bool), EDGE_RADIUS)
                count = max(int((~seen).sum()), 1)
                v["unseen_pixels_near_depth_edges"] = float((~seen & edges).sum() / count)
                v["unseen_pixels_on_image_border_only"] = float((~seen & border & ~edges).sum() / count)
                v["seen_regions"] = by_region(np.where(seen3, naive, exact), exact, masks, total)
            if k in (8, 32):
                maps[variant, k] = {
                    "naive": error_map(used, exact),
                    "corrected": error_map(np.where(seen3, corrected, unshadowed), exact),
                    "seen": seen,
                    "exact": exact,
                    "signed": ((naive - exact) * seen3).sum(-1) / exact.sum(-1).mean(),
                }
            entry[variant] = v
        results[k] = entry
        print(
            "reuse",
            k,
            json.dumps(
                {var: {a: round(b, 5) for a, b in entry[var].items() if isinstance(b, float)} for var in REUSE_VARIANTS}
            ),
            flush=True,
        )
    plot_reuse(results, maps, masks)
    return results


def plot_reuse(results: dict, maps: dict, masks) -> None:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 4)
    k = 32
    frame = results[k]["frame"]
    main = maps["shadows", k]
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(show(main["exact"]))
    ax.set_title(f"closed form fog light, frame {frame}")
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(~main["seen"], cmap="gray_r")
    ax.set_title(f"not seen by key frame 255 (black): {1 - main['seen'].mean():.1%}")
    ax = fig.add_subplot(grid[0, 2])
    limit = float(np.percentile(np.abs(main["signed"]), 99.5))
    im = ax.imshow(main["signed"], cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title("seen pixels, reused - exact (red: too bright)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    draw_regions(fig.add_subplot(grid[0, 3]), masks, f"regions, frame {frame}")
    vmax = float(np.percentile(main["naive"], 99.5))
    panels = (
        (("shadows", "naive"), "reused as is (the method)"),
        (("shadows", "corrected"), "sun's phase function corrected"),
        (("shadows_32_sky_samples", "naive"), "32 sky visibility samples, as is"),
        (("no_shadows", "naive"), "no shadows at all, as is"),
    )
    for col, ((variant, key), title) in enumerate(panels):
        ax = fig.add_subplot(grid[1, col])
        m = maps[variant, k][key]
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{frame} from 255, {title}: {m.mean():.2%}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="|err| / mean fog light")
    no_ticks([a for a in fig.axes if a.images])
    ax = fig.add_subplot(grid[2, :])
    offsets = sorted(results)
    shift = [results[o]["camera_displacement_m"] for o in offsets]
    series = (
        ("shadows", "rel_l1", "total error (the method)", "k", "-"),
        ("shadows", "unseen_error_share", "unseen pixels, filled without shadows", "tab:red", "-"),
        ("shadows", "seen_error_share", "seen pixels", "tab:blue", "-"),
        ("shadows", "seen_error_share_phase_corrected", "seen pixels, phase corrected", "tab:blue", "--"),
        ("shadows_32_sky_samples", "seen_error_share", "32 sky samples: seen pixels", "tab:purple", "-"),
        ("no_shadows", "seen_error_share", "no shadows: seen pixels", "tab:green", "-"),
        ("no_shadows", "seen_error_share_phase_corrected", "no shadows: seen, phase corrected", "tab:green", "--"),
    )
    for variant, key, label, color, style in series:
        ax.plot(shift, [results[o][variant][key] * 100 for o in offsets], style, color=color, marker="o", label=label)
    for o, x in zip(offsets, shift):
        ax.annotate(f"+{o}", (x, 0), textcoords="offset points", xytext=(0, 3), fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("camera displacement from key frame 255 (m); label = frame offset")
    ax.set_ylabel("rel L1 vs closed form fog light (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(SAVE / "reuse_day.png", dpi=110)
    plt.close(fig)


def reuse_paths_study() -> dict:
    """What reusing the key frame's mean source must get wrong even with perfect reprojection: the exact mean source
    along the ray from the key frame's camera to the point each target pixel sees, against along the target's own ray.

    Both are marched with the closed form's lighting (:class:`DayLight`, which looks up the sky's visibility at every
    node rather than at 8 samples per ray), and only over pixels the key frame saw. Their difference is split into the
    change of direction (the phase function, and the elevation on which light from the sky, the ground and multiple
    scattering depends), by marching the key's ray with the target's direction, and the change of path (where the ray
    passes, through light shafts, places where objects hide the sky, and heights of the fog).
    """
    offsets = (1, 2, 4, 8, 16, 32)
    frames = (255,) + tuple(255 + k for k in offsets)
    day = day_scene(frames=frames)
    light = DayLight(day)
    fog, beta = light.fog, light.beta
    key_origin = day.rays(0)[0]
    blank = {"camera": day.transforms[0], "depth": day.depth[0], "mean": np.zeros((*day.depth[0].shape, 1))}

    def march(origin, directions, distance, look, nodes, fn):
        """Mean source along rays, sampled where the fog's light comes from, looking along ``look``."""
        t, w = day_nodes(fog, beta, origin, directions, distance, nodes)
        points = origin + directions[:, None] * t[..., None]
        values = fn(points.reshape(-1, 3), look[:, None].expand_as(points).reshape(-1, 3))
        return (values.reshape(*t.shape, 3) * w[..., None]).sum(dim=1) / (w * nodes).clamp_min(1e-12)

    def mean_sources(origin, directions, distance, look) -> dict[str, np.ndarray]:
        parts = {}
        for name, nodes, chunk, fn in (("sun", SUN_NODES, 2048, light.sun), ("rest", SMOOTH_NODES, 1024, light.smooth)):
            parts[name] = by_chunks(
                lambda d, r, k: march(origin, d, r, k, nodes, fn), len(directions), chunk, directions, distance, look
            ).numpy()
        parts["all"] = parts["sun"] + parts["rest"]
        return parts

    results: dict = {}
    maps = {}
    for j, k in enumerate(offsets, start=1):
        shape = day.depth[j].shape
        origin, directions, distance = day.rays(j)
        points = origin + directions * distance[:, None]
        towards = points - key_origin
        key_distance = towards.norm(dim=-1)
        key_directions = towards / key_distance[:, None]
        target = mean_sources(origin, directions, distance, directions)
        key = mean_sources(key_origin, key_directions, key_distance, key_directions)
        key_path = mean_sources(key_origin, key_directions, key_distance, directions)  # key's path, target's direction
        _, seen = exp_less.reproject(blank, {"camera": day.transforms[j], "depth": day.depth[j]})
        seen = seen.reshape(-1, 1)
        opacity = 1 - exp_less.transmittance(day.medium, day.transforms[j], day.depth[j]).reshape(-1, 1)
        total = (target["all"] * opacity).sum()

        def share(a, b, signed=False):
            difference = (a - b) * opacity * seen
            return float((difference if signed else np.abs(difference)).sum() / total)

        entry = {"frame": 255 + k, "seen_pixels": float(seen.mean())}
        for part in ("all", "sun", "rest"):
            entry[part] = {
                "path_and_direction": share(key[part], target[part]),
                "direction_only": share(key[part], key_path[part]),
                "path_only": share(key_path[part], target[part]),
                "path_and_direction_bias": share(key[part], target[part], signed=True),
                "path_only_bias": share(key_path[part], target[part], signed=True),
            }
        results[k] = entry
        if k == 32:
            for name, a, b in (
                ("path_and_direction", key, target),
                ("direction_only", key, key_path),
                ("path_only", key_path, target),
            ):
                difference = ((a["all"] - b["all"]) * opacity * seen).sum(-1)
                maps[name] = difference / (target["all"] * opacity).sum(-1).mean()
            maps = {name: m.reshape(shape) for name, m in maps.items()}
        print("reuse paths", k, json.dumps(entry), flush=True)
    plot_reuse_paths(maps)
    return results


def plot_reuse_paths(maps: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), constrained_layout=True)
    limit = float(np.percentile(np.abs(maps["path_and_direction"]), 99.5))
    titles = {
        "path_and_direction": "key's ray vs target's ray (path and direction)",
        "direction_only": "direction only (phase function, elevation)",
        "path_only": "path only (shafts, hidden sky, heights)",
    }
    for ax, (name, title) in zip(axes, titles.items()):
        im = ax.imshow(maps[name], cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"frame 287 from 255, {title}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="(key - target) / mean fog light")
    no_ticks(axes)
    fig.savefig(SAVE / "reuse_paths_day.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# 3. Shadows shared across densities


def light_fractions(fog, beta: float, origin, directions, bounds: torch.Tensor) -> torch.Tensor:
    """Light scattered towards the camera between consecutive bounds along each ray, of shape (n, m), as a fraction
    of the light behind it."""
    depth = height_fog_optical_depth(fog, origin, directions[:, None, :], bounds.to(directions.dtype)) * beta
    return -torch.diff(torch.exp(-depth), dim=1)


def density_study() -> dict:
    day = day_scene(frames=(345,))
    t = day.transforms[0]
    pose = t["transform_matrix"]
    raw_depth = load_images(ROOT / "clear" / "depths", (345,))[0, ..., 0]
    args = (day.clear[0], day.depth[0], t, pose)
    sun_medium, sun_lighting = sun_only(day)

    def trace(medium, samples=8):
        return trace_shadows(
            day.occlusion, day.depth[0], t, pose, medium, day.lighting, shadow_samples=samples, dtype=DTYPE
        )

    shared = trace(day.medium)
    shadow_ratio = ratio(
        *(
            day.closed_form(0, medium=sun_medium, lighting=sun_lighting, normals=None, occlusion=o).inscatter.numpy()
            for o in (day.occlusion, None)
        )
    )
    masks = regions(t, day.depth[0], raw_depth, day.extra["normals"][0], shadow_ratio)
    origin, directions, _ = camera_rays(t, pose)
    directions = directions.reshape(-1, 3)
    fog = day.medium.components[0]
    results: dict = {}
    maps = {}
    for scale in (1.0, 0.5, 4.0):
        medium = day.medium.model_copy(update={"extinction": day.medium.extinction * scale})
        own = trace(medium)

        def render(shadows):
            return apply_medium(*args, medium, day.lighting, shadows=shadows, dtype=DTYPE)

        separate = render(own)
        converged = render(trace(medium, 64)).radiance.numpy()
        ref, ref_in = separate.radiance.numpy(), separate.inscatter.numpy()
        unshadowed = apply_medium(*args, medium, day.lighting, dtype=DTYPE).inscatter.numpy()
        entry: dict = {
            "fog_light_share_of_image": float(ref_in.sum() / ref.sum()),
            "shadows_share_of_fog_light": float((unshadowed - ref_in).sum() / ref_in.sum()),
            "own_8_samples_vs_64": errors(ref[None], converged[None]),
        }
        maps[scale, "own_vs_64"] = error_map(ref, converged)
        if scale != 1.0:
            variants = {
                "shared": render(shared),
                "sun_segments_shared": render(own._replace(suns=shared.suns)),
                "sky_samples_shared": render(shared._replace(suns=own.suns)),
            }
            for name, result in variants.items():
                ours = result.radiance.numpy()
                entry[name] = {
                    "vs_separate_image": errors(ours[None], ref[None]),
                    "rel_l1_of_fog_light": l1(ours, ref, ref_in.sum()),
                    "regions": by_region(ours, ref, masks, ref.sum()),
                }
            ours = variants["shared"].radiance.numpy()
            entry["shared_8_samples_vs_64"] = errors(ours[None], converged[None])
            maps[scale, "shared"] = error_map(ours, ref)
            maps[scale, "shared_vs_64"] = error_map(ours, converged)

            # How the sky's 8 visibility samples spread over this fog's light, when placed for it or for density x1
            spread = {}
            for which, shadows in (("own", own), ("shared", shared)):
                w = light_fractions(fog, medium.extinction, origin, directions, shadows.bounds)
                light = w.sum(dim=1)
                keep = light > 1e-4
                normalized, light = w[keep] / light[keep, None], light[keep]
                spread[which] = {
                    "effective_samples": float(((1 / (normalized**2).sum(dim=1)) * light).sum() / light.sum()),
                    "largest_sample_light_fraction": float((normalized.max(dim=1).values * light).sum() / light.sum()),
                }
            spread["mean_end_of_samples_m"] = {
                which: float(s.bounds[:, -1].mean()) for which, s in (("own", own), ("shared", shared))
            }
            entry["sky_samples"] = spread
        results[str(scale)] = entry
        print(
            "density",
            scale,
            json.dumps({k: v for k, v in entry.items() if k != "regions"}, default=str)[:1500],
            flush=True,
        )
    plot_density(maps, masks)
    return results


def plot_density(maps: dict, masks) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 6.2), constrained_layout=True)
    titles = {
        "shared": "shadows traced for x1 vs for this density",
        "own_vs_64": "traced for this density (8 sky samples) vs 64",
        "shared_vs_64": "traced for x1 (8 sky samples) vs 64",
    }
    for row, scale in enumerate((0.5, 4.0)):
        vmax = float(np.percentile(maps[scale, "shared"], 99.5))
        for col, name in enumerate(titles):
            m = maps[scale, name]
            im = axes[row, col].imshow(m, cmap="magma", vmin=0, vmax=vmax)
            axes[row, col].set_title(f"density x{scale}: {titles[name]}: {m.mean():.3%}", fontsize=9)
            fig.colorbar(im, ax=axes[row, col], shrink=0.8, label="|diff| / mean image")
        draw_regions(axes[row, 3], masks, "regions (frame 345)")
    no_ticks(axes)
    fig.savefig(SAVE / "density_shared_shadows_345.png", dpi=110)
    plt.close(fig)


STUDIES = {
    "density": lambda: density_study(),
    "low_resolution": lambda: low_resolution_study(),
    "reuse": lambda: reuse_study(),
    "reuse_paths": lambda: reuse_paths_study(),
}


def main():
    """Run the studies named on the command line, or all of them, and update their results in the JSON file."""
    SAVE.mkdir(parents=True, exist_ok=True)
    path = SAVE / "compute_less.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results |= {
        "resolution": "320x180 for everything; low resolution 1/s computes the closed form through (320/s)x(180/s) rays",
        "reference": "the closed form computed for every pixel and frame (noise-free), not Cycles",
    }
    for name in sys.argv[1:] or STUDIES:
        results[name] = STUDIES[name]()
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
