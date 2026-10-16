"""Where in the image each method's error of the day scene comes from, and from which light.

Every method's whole image of frames 255 and 345 (only 345 for the world-space caches), all at 320x180, is compared
with Cycles' reference (4096 spp, 16 volume bounces), and its error is split by region of the image:

- a partition, so that the contributions of regions add up to the method's relative L1 error: silhouettes of objects
  (depth discontinuities, dilated by 2 px), sky, and surfaces 0-5 m, 5-15 m and beyond 15 m away;
- overlapping masks, reported alongside: pixels whose light the objects' shadows in the fog change by more than 2%
  (light shafts), faces of objects (not the ground), the band within 2° of the horizon, ground in the shadow of an
  object, and edges of the checker texture and of shadows on surfaces (from the image without fog).

For each region: the contribution to the whole error (sum over the region of |ours - ref| over the sum of the whole
reference), the local error (same over the sum of the reference in the region), both also after a 5x5 box blur of both
images (which leaves out most Monte Carlo noise, i.e. bias only), and the signed bias. Where a method's image can be
split into light of surfaces and light scattered by the fog (and that into single and multiple scattering), the same
per light; errors of lights can cancel in the whole image, so theirs need not add up to it.

Methods that share the closed form's lighting (froxels, caches) are also compared with the closed form itself, which
isolates their own discretization error from the error of the lighting model they inherit. The closed form's light
scattered more than once is split further into its terms (multiple scattering of the sky and sun, light reflected by
the ground, and both together) to see which one is off where.

Outputs in ``runs/fog-comparison/acceleration/analysis/day_regions``: ``regions.json``, ``error_maps_345.png``
(relative error of frame 345 on one color scale), ``signed_maps_345.png`` (blurred signed error: too bright or too
dark), ``region_contributions.png`` (each method's error split by region), ``edge_profile.png`` (error against the
distance to the nearest silhouette) and ``sun_angle_profile.png`` (bias of the fog's light against the angle to the
sun).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *
from common import (
    DAY_FRAMES,
    OUT,
    ROOT,
    DayLight,
    blur,
    by_chunks,
    day_nodes,
    day_scene,
    errors,
    load_images,
)

torch.set_num_threads(3)  # other jobs share the CPU

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import OpenEXR
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from visionsim.medium.render import camera_rays
from visionsim.utils.color import linearrgb_to_srgb

FRAMES = (255, 345)
SAVE = OUT / "analysis" / "day_regions"
CYCLES = ROOT / "ms" / "cycles_ref"
LUMA = np.array([0.2126, 0.7152, 0.0722])

PARTITION = ("edges", "sky", "near", "mid", "far")
OVERLAPS = ("shafts", "objects", "horizon", "ground_shadow", "texture")
REGION_NAMES = {
    "edges": "轮廓边缘 (±2 px)",
    "sky": "天空",
    "near": "表面 0-5 m",
    "mid": "表面 5-15 m",
    "far": "表面 >15 m",
    "shafts": "光柱 (物体阴影改变雾光 >2%)",
    "objects": "物体表面 (非地面)",
    "horizon": "地平线 ±2°",
    "ground_shadow": "物体阴影中的地面",
    "texture": "纹理与阴影边界",
}
EDGE_BINS = ((0, 1.5), (1.5, 3.5), (3.5, 6.5), (6.5, 12.5), (12.5, np.inf))
EDGE_LABELS = ("≤1", "2-3", "4-6", "7-12", ">12")
SUN_BINS = ((0, 40), (40, 55), (55, 70), (70, 85), (85, 180))
SUN_LABELS = ("<40°", "40-55°", "55-70°", "70-85°", ">85°")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REGION_COLORS = dict(zip(PARTITION, SERIES))
# Sequential blue from the surface (no error) to dark (large error); diverging blue (too dark) to red (too bright)
ERROR_MAP = LinearSegmentedColormap.from_list(
    "error", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b", "#06203f"]
)
SIGNED_MAP = LinearSegmentedColormap.from_list(
    "signed", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f6cbc9", "#ec8b89", "#d03b3b", "#7f1d1d"]
)


# --------------------------------------------------------------------------------------------------------------------
# Regions


def sun_change(scene, light: DayLight, i: int, samples: int = 32) -> np.ndarray:
    """Sunlight that the objects' shadows remove from the fog's light along each camera ray, of shape (h, w, 3), from
    the closed form's sunlight integrated with and without its shadow map."""
    fog = scene.medium.components[0]
    origin, directions, distance = scene.rays(i)
    t, weights = day_nodes(fog, light.beta, origin, directions, distance, samples)
    points = (origin + directions[:, None, :] * t[..., None]).reshape(-1, 3)
    towards = directions[:, None, :].expand(-1, samples, -1).reshape(-1, 3)
    shadowed = by_chunks(lambda p, d: light.sun(p, d), len(points), 2**16, points, towards)
    unshadowed = by_chunks(lambda p, d: light.sun(p, d, torch.ones_like(p[:, 0])), len(points), 2**16, points, towards)
    change = (unshadowed - shadowed).reshape(-1, samples, 3).sum(1) * weights
    return change.numpy().reshape(*scene.depth[i].shape, 3)


def regions(scene, light: DayLight) -> dict[str, np.ndarray]:
    """Masks of shape (n, h, w) of each region, for the scene's frames, and of each pixel its distance to the nearest
    silhouette, the angle between its ray and the direction towards the sun, and its ray's elevation, in degrees."""
    raw = load_images(ROOT / "clear" / "depths", scene.frames)[..., 0]
    masks: dict[str, list] = {name: [] for name in PARTITION + OVERLAPS + ("edge_distance", "sun_angle", "elevation")}
    towards_sun = np.asarray(scene.lighting.suns[0].direction, dtype=np.float64)
    towards_sun /= np.linalg.norm(towards_sun)
    for i in range(len(scene.frames)):
        t = scene.transforms[i]
        origin, directions, scale = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
        distance = scene.depth[i] * scale
        sky = raw[i] >= 1e9
        reference = blur(scene.reference[i][None].astype(np.float64))[0] @ LUMA

        # Silhouettes: a jump of more than 15% in depth to a neighbour whose near side is closer than 100 m, which
        # leaves out the far edge of the ground against the sky (180 m against about 200 m)
        def jumps(values: np.ndarray, threshold: float, keep: np.ndarray | None = None) -> np.ndarray:
            found = np.zeros(values.shape, dtype=bool)
            for axis in (0, 1):
                pairs = np.abs(np.diff(values, axis=axis)) > threshold
                if keep is not None:
                    pairs &= np.minimum(np.delete(keep, -1, axis=axis), np.delete(keep, 0, axis=axis)) < 100
                for pad in ((0, 1), (1, 0)):
                    found |= np.pad(pairs, [pad if a == axis else (0, 0) for a in (0, 1)])
            return found

        silhouette = jumps(np.log(scene.depth[i]), np.log(1.15), keep=distance)
        edges = ndimage.binary_dilation(silhouette, iterations=2)
        masks["edge_distance"].append(ndimage.distance_transform_edt(~silhouette))
        masks["edges"].append(edges)
        masks["sky"].append(sky & ~edges)
        surface = ~sky & ~edges
        masks["near"].append(surface & (distance < 5))
        masks["mid"].append(surface & (distance >= 5) & (distance < 15))
        masks["far"].append(surface & (distance >= 15))

        # Light shafts: pixels whose fog's light the objects' shadows change by more than 2% of the reference's light
        masks["shafts"].append(sun_change(scene, light, i) @ LUMA > 0.02 * reference)
        ground = scene.extra["normals"][i][..., 2] > 0.95
        masks["objects"].append(~sky & ~ground)
        elevation = np.degrees(np.arcsin(np.clip(directions[..., 2], -1, 1)))
        masks["horizon"].append(np.abs(elevation) < 2)
        masks["elevation"].append(elevation)
        masks["sun_angle"].append(np.degrees(np.arccos(np.clip(directions @ towards_sun, -1, 1))))
        # Ground whose point the sun's shadow map says is in an object's shadow
        points = torch.as_tensor(origin + directions * distance[..., None], dtype=DTYPE).reshape(-1, 3)
        sunlit = light.sun_visibility(points).numpy().reshape(sky.shape)
        masks["ground_shadow"].append(~sky & ground & (sunlit < 0.5))
        # Edges of the checker and of shadows on surfaces: jumps of 60% in the light of the image without fog
        clear = np.log(scene.clear[i].astype(np.float64) @ LUMA + 1e-4)
        masks["texture"].append(ndimage.binary_dilation(jumps(clear, np.log(1.6)) & ~sky) & ~edges)
    return {name: np.stack(values) for name, values in masks.items()}


# --------------------------------------------------------------------------------------------------------------------
# Methods


@dataclass
class Method:
    key: str
    label: str
    images: dict[int, np.ndarray]
    """whole image of each frame it has, of shape (h, w, 3)"""
    parts: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    """its image split into light of surfaces, light scattered by the fog, and that once and more than once if known"""
    lighting_of_closed_form: bool = False
    """whether it integrates the closed form's lighting, so that its own error shows against the closed form"""


def read_exr(path) -> np.ndarray:
    with OpenEXR.File(str(path)) as f:
        return np.asarray(f.channels()["RGBA"].pixels, dtype=np.float32)[..., :3]


def split(frame: np.ndarray, fog: np.ndarray, single: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Parts of an image given its fog's light, and the part of it scattered once if known."""
    parts = {"surfaces": frame - fog, "fog": fog}
    if single is not None:
        parts |= {"single": single, "multiple": fog - single}
    return parts


def load_reference(scene) -> Method:
    direct, indirect = (load_images(CYCLES / "volume" / name, FRAMES) for name in ("direct", "indirect"))
    return Method(
        "reference",
        "Cycles 4096 spp (参考)",
        {f: scene.reference[k] for k, f in enumerate(FRAMES)},
        {f: split(scene.reference[k], direct[k] + indirect[k], direct[k]) for k, f in enumerate(FRAMES)},
    )


def closed_form(scene) -> tuple[Method, dict[int, dict[str, np.ndarray]], float]:
    """The closed form split into its lights; the terms of its light scattered more than once (multiple scattering of
    sunlight and skylight alone, light reflected by the ground scattered once, and the rest, i.e. both together); and
    the fraction of pixels whose light changes by more than 2% without any shadows of objects (of the sun or the sky)."""
    no_multiple = scene.medium.model_copy(update={"multiple_scattering": False})
    no_ground = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    saved = np.load(OUT / "results" / "closed_form_day.npy")
    images, parts, terms, changed = {}, {}, {}, []
    for k, f in enumerate(FRAMES):
        full = scene.closed_form(k)
        radiance, fog = full.radiance.numpy(), full.inscatter.numpy()
        stored = saved[DAY_FRAMES.index(f)]
        assert np.abs(stored - radiance).sum() / stored.sum() < 1e-3, "the closed form differs from its saved images"
        single = scene.closed_form(k, medium=no_multiple, lighting=no_ground).inscatter.numpy()
        ground = scene.closed_form(k, medium=no_multiple).inscatter.numpy() - single
        multiple = scene.closed_form(k, lighting=no_ground).inscatter.numpy() - single
        terms[f] = {"multiple_scattering": multiple, "ground": ground, "both": fog - single - ground - multiple}
        unshadowed = scene.closed_form(k, occlusion=None).radiance.numpy()
        reference = blur(scene.reference[k][None].astype(np.float64))[0] @ LUMA
        changed.append(np.abs(radiance - unshadowed) @ LUMA > 0.02 * reference)
        images[f] = radiance
        parts[f] = split(radiance, fog, single)
    return Method("closed_form", "闭式解 (当前方法)", images, parts), terms, float(np.mean(changed))


def monte_carlo(name: str, label: str, noisy: bool) -> Method:
    path = OUT / name
    frames = load_images(path / ("noisy" if noisy else "frames"), FRAMES)
    parts = {}
    if noisy:  # the volume passes are of the noisy image; the denoised image can't be split
        direct, indirect = (load_images(path / "volume" / n, FRAMES) for n in ("direct", "indirect"))
        parts = {f: split(frames[k], direct[k] + indirect[k], direct[k]) for k, f in enumerate(FRAMES)}
    return Method(
        f"{name}_{'noisy' if noisy else 'denoised'}", label, {f: frames[k] for k, f in enumerate(FRAMES)}, parts
    )


def eevee(name: str, label: str) -> Method:
    frames, fog = (load_images(OUT / name / sub, FRAMES) for sub in ("frames", "volume/light"))
    return Method(
        name,
        label,
        {f: frames[k] for k, f in enumerate(FRAMES)},
        {f: split(frames[k], fog[k]) for k, f in enumerate(FRAMES)},
    )


def unreal(preset: str, label: str, transmittance: dict[int, np.ndarray]) -> Method:
    """Unreal's frames, whose fog light is estimated as the frame minus the frame without fog times the exact
    transmittance of the fog (as in exp_unreal.py)."""
    images, parts = {}, {}
    for f in FRAMES:
        frame = read_exr(OUT / "unreal" / "renders" / f"Day_{preset}" / f"{f:04d}.exr")
        clear = read_exr(OUT / "unreal" / "renders" / "Day_nofog" / f"{f:04d}.exr")
        images[f] = frame
        parts[f] = split(frame, frame - transmittance[f] * clear)
    return Method(f"unreal_{preset}", label, images, parts)


def with_surfaces(key: str, label: str, fog: dict[int, np.ndarray], surfaces: dict[int, np.ndarray]) -> Method:
    """A method that only computes the fog's light, added to the closed form's light of surfaces."""
    return Method(
        key,
        label,
        {f: surfaces[f] + fog[f] for f in fog},
        {f: split(surfaces[f] + fog[f], fog[f]) for f in fog},
        lighting_of_closed_form=True,
    )


def load_methods(scene, light: DayLight, closed: Method) -> list[Method]:
    surfaces = {f: closed.parts[f]["surfaces"] for f in FRAMES}
    transmittance = {f: scene.transmittance(k) for k, f in enumerate(FRAMES)}
    # Froxels of frame 255 aren't saved: rerun the prototype, and check that it reproduces the saved frame 345
    from exp_froxel import froxels

    froxel = {345: np.load(OUT / "results" / "froxel_images.npz")["day_8_64_1"]}
    froxel[255], _ = froxels(scene, light, FRAMES.index(255), 8, 64, 1)
    again, _ = froxels(scene, light, FRAMES.index(345), 8, 64, 1)
    assert np.abs(again - froxel[345]).sum() / froxel[345].sum() < 1e-3, "froxels differ from their saved images"
    cache = np.load(OUT / "results" / "cache_day_images.npz")
    seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", FRAMES)
    return [
        Method("reference_seed1", "Cycles 4096 spp 另一种子 (噪声下限)", {f: seed1[k] for k, f in enumerate(FRAMES)}),
        closed,
        monte_carlo("day_spp4", "Cycles 4 spp 降噪", noisy=False),
        monte_carlo("day_spp16", "Cycles 16 spp 降噪", noisy=False),
        monte_carlo("day_spp16", "Cycles 16 spp 未降噪", noisy=True),
        monte_carlo("day_spp1024", "Cycles 1024 spp 未降噪", noisy=True),
        eevee("eevee_day_default", "EEVEE 默认"),
        eevee("eevee_day_1sample", "EEVEE 1 采样"),
        unreal("high", "Unreal High", transmittance),
        unreal("epic", "Unreal Epic", transmittance),
        with_surfaces("froxel_8_64", "Froxel 8 px × 64 层", froxel, surfaces),
        with_surfaces("cache_sh_1.0_2", "缓存 SH2 1 m (仅 345 帧)", {345: cache["sh_1.0_2"]}, surfaces),
        with_surfaces("cache_sun_1.0", "缓存 太阳可见度 1 m (仅 345 帧)", {345: cache["sun_1.0"]}, surfaces),
    ]


# --------------------------------------------------------------------------------------------------------------------
# Errors by region


def region_errors(ours: np.ndarray, ref: np.ndarray, masks: dict[str, np.ndarray], total: float) -> dict:
    """Errors of images of shape (n, h, w, 3) in each region, as fractions of the whole reference's light `total`
    (contribution) and of the region's own light (local)."""
    ours, ref = ours.astype(np.float64), ref.astype(np.float64)
    error = np.abs(ours - ref).sum(-1)
    error_blurred = np.abs(blur(ours) - blur(ref)).sum(-1)
    ours_sum, ref_sum = ours.sum(-1), ref.sum(-1)
    result = {}
    for name, mask in masks.items():
        light = ref_sum[mask].sum()
        result[name] = {
            "pixels": float(mask.mean()),
            "light": float(light / total),
            "contribution": float(error[mask].sum() / total),
            "contribution_blurred": float(error_blurred[mask].sum() / total),
            "local": float(error[mask].sum() / light),
            "local_blurred": float(error_blurred[mask].sum() / light),
            "bias": float((ours_sum[mask].sum() - light) / light),
        }
    return result


def part_errors(ours: dict, ref: dict, masks: dict[str, np.ndarray], total: float) -> dict:
    """Per light and region: its error's contribution to the whole error, and its signed bias, as fractions of the
    whole reference's light, and its bias relative to the reference's light of that kind in the region."""
    result = {}
    everywhere = {"all": np.ones_like(masks["sky"])}
    for part in ours:
        o, r = ours[part].astype(np.float64), ref[part].astype(np.float64)
        error, error_blurred = np.abs(o - r).sum(-1), np.abs(blur(o) - blur(r)).sum(-1)
        signed, r_sum = (o - r).sum(-1), r.sum(-1)
        result[part] = {
            name: {
                "contribution": float(error[mask].sum() / total),
                "contribution_blurred": float(error_blurred[mask].sum() / total),
                "bias": float(signed[mask].sum() / total),
                "reference": float(r_sum[mask].sum() / total),
                "relative_bias": float(signed[mask].sum() / r_sum[mask].sum()),
            }
            for name, mask in (everywhere | masks).items()
        }
    return result


def edge_profile(ours, ref, distance: np.ndarray, ours_fog=None, ref_fog=None) -> dict:
    """Local blurred error against the distance in pixels to the nearest silhouette, of the whole image, and of the
    fog's light relative to the reference's fog light when known (pixels near silhouettes are mostly far away and
    foggy, so the whole image's error grows there with the share of fog light even without any artifact)."""
    error = np.abs(blur(ours.astype(np.float64)) - blur(ref.astype(np.float64))).sum(-1)
    ref_sum = ref.astype(np.float64).sum(-1)
    profile: dict = {"whole": {}}
    if ours_fog is not None:
        fog_error = np.abs(blur(ours_fog.astype(np.float64)) - blur(ref_fog.astype(np.float64))).sum(-1)
        fog_signed, fog_sum = (ours_fog.astype(np.float64) - ref_fog).sum(-1), ref_fog.astype(np.float64).sum(-1)
        profile |= {"fog": {}, "fog_bias": {}, "fog_share": {}}
    for label, (low, high) in zip(EDGE_LABELS, EDGE_BINS):
        mask = (distance >= low) & (distance < high)
        profile["whole"][label] = float(error[mask].sum() / ref_sum[mask].sum())
        if ours_fog is not None:
            profile["fog"][label] = float(fog_error[mask].sum() / fog_sum[mask].sum())
            profile["fog_bias"][label] = float(fog_signed[mask].sum() / fog_sum[mask].sum())
            profile["fog_share"][label] = float(fog_sum[mask].sum() / ref_sum[mask].sum())
    return profile


def sun_profile(ours: dict, ref: dict, masks: dict, index: list[int]) -> dict:
    """Relative bias of each kind of the fog's light against the angle between the camera ray and the sun, over pixels
    of the sky, and of surfaces. The closed form's light scattered more than once only depends on the elevation of
    rays, so its bias against this angle shows the anisotropy it misses."""
    angle = masks["sun_angle"][index]
    regions_ = {"sky": masks["sky"][index], "surfaces": ~masks["sky"][index] & ~masks["edges"][index]}
    profile = {}
    for part in ("fog", "single", "multiple"):
        if part not in ours:
            continue
        o, r = ours[part].astype(np.float64).sum(-1), ref[part].astype(np.float64).sum(-1)
        profile[part] = {}
        for where, mask in regions_.items():
            profile[part][where] = {}
            for label, (low, high) in zip(SUN_LABELS, SUN_BINS):
                m = mask & (angle >= low) & (angle < high)
                profile[part][where][label] = float(o[m].sum() / r[m].sum() - 1) if m.sum() > 100 else None
    return profile


def evaluate(method: Method, reference: Method, masks: dict, frames: tuple[int, ...]) -> dict:
    index = [FRAMES.index(f) for f in frames]
    region_masks = {name: masks[name][index] for name in PARTITION + OVERLAPS}
    ref = np.stack([reference.images[f] for f in frames])
    ours = np.stack([method.images[f] for f in frames])
    total = float(ref.astype(np.float64).sum())
    result = {
        "frames": list(frames),
        "whole": errors(ours, ref),
        "regions": region_errors(ours, ref, region_masks, total),
    }
    share = {name: result["regions"][name]["contribution"] / result["whole"]["rel_l1"] for name in PARTITION}
    result["dominant"] = max(share, key=share.get)
    result["dominant_share"] = share[result["dominant"]]
    for suffix in ("", "_blurred"):
        local = {name: result["regions"][name]["local" + suffix] for name in PARTITION}
        result["worst_local" + suffix] = max(local, key=local.get)
    fog = {}
    if method.parts:
        ours_parts = {p: np.stack([method.parts[f][p] for f in frames]) for p in method.parts[frames[0]]}
        ref_parts = {p: np.stack([reference.parts[f][p] for f in frames]) for p in ours_parts}
        result["parts"] = part_errors(ours_parts, ref_parts, region_masks, total)
        fog = {"ours_fog": ours_parts["fog"], "ref_fog": ref_parts["fog"]}
        result["sun_profile"] = sun_profile(ours_parts, ref_parts, masks, index)
    result["edge_profile"] = edge_profile(ours, ref, masks["edge_distance"][index], **fog)
    return result


def against_closed_form(method: Method, closed: Method, reference: Method, masks: dict) -> dict:
    """Error of a method that integrates the closed form's lighting against the closed form itself, as fractions of
    Cycles' light, by region: its own discretization error."""
    frames = tuple(sorted(method.images))
    index = [FRAMES.index(f) for f in frames]
    region_masks = {name: masks[name][index] for name in PARTITION + OVERLAPS}
    ours = np.stack([method.images[f] for f in frames])
    theirs = np.stack([closed.images[f] for f in frames])
    total = float(np.stack([reference.images[f] for f in frames]).astype(np.float64).sum())
    by_region = region_errors(ours, theirs, region_masks, total)
    return {
        "frames": list(frames),
        "rel_l1": sum(by_region[name]["contribution"] for name in PARTITION),
        "rel_l1_blurred": sum(by_region[name]["contribution_blurred"] for name in PARTITION),
        "bias": float((ours.astype(np.float64).sum() - theirs.sum()) / total),
        "regions": by_region,
        "edge_profile": edge_profile(ours, theirs, masks["edge_distance"][index])["whole"],
    }


def azimuth_at_fixed_elevation(closed: Method, reference: Method, masks: dict, scene) -> dict:
    """Relative bias of the closed form's light scattered more than once in the sky, in bands of elevation, against the
    azimuth of rays away from the sun's: at a fixed elevation, its tables give the same light at every azimuth."""
    towards = np.asarray(scene.lighting.suns[0].direction, dtype=np.float64)
    sun_azimuth = np.degrees(np.arctan2(towards[1], towards[0]))
    azimuths = []
    for t in scene.transforms:
        _, directions, _ = (x.numpy() for x in camera_rays(t, t["transform_matrix"]))
        azimuth = np.degrees(np.arctan2(directions[..., 1], directions[..., 0]))
        azimuths.append(np.abs((azimuth - sun_azimuth + 180) % 360 - 180))
    azimuth = np.stack(azimuths)
    ours = np.stack([closed.parts[f]["multiple"] for f in FRAMES]).astype(np.float64).sum(-1)
    theirs = np.stack([reference.parts[f]["multiple"] for f in FRAMES]).astype(np.float64).sum(-1)
    result = {}
    for low, high in ((2, 8), (8, 15), (15, 25)):
        band = masks["sky"] & (masks["elevation"] >= low) & (masks["elevation"] < high)
        entry = {}
        for a, b in ((0, 30), (30, 50), (50, 70), (70, 90), (90, 180)):
            m = band & (azimuth >= a) & (azimuth < b)
            entry[f"{a}-{b}°"] = float(ours[m].sum() / theirs[m].sum() - 1) if m.sum() > 100 else None
        result[f"elevation {low}-{high}°"] = entry
    return result


def closed_form_terms(terms: dict, reference: Method, masks: dict) -> dict:
    """Light of each term of the closed form's light scattered more than once, against Cycles' (volume indirect), by
    region, as fractions of the whole reference's light."""
    stack = lambda x: np.stack([x[f] for f in FRAMES]).astype(np.float64).sum(-1)
    total = float(stack(reference.images).sum())
    cycles = stack({f: reference.parts[f]["multiple"] for f in FRAMES})
    values = {name: stack({f: terms[f][name] for f in FRAMES}) for name in terms[FRAMES[0]]}
    # Ground at the same distance in the sun and in the shadow of an object, to tell shadows from the angle to the sun
    regions_ = {"all": np.ones_like(masks["sky"])} | {name: masks[name] for name in PARTITION + OVERLAPS}
    for band in ("near", "mid"):
        ground = masks[band] & ~masks["objects"]
        regions_[f"{band}_ground_sunlit"] = ground & ~masks["ground_shadow"]
        regions_[f"{band}_ground_shadow"] = ground & masks["ground_shadow"]
    result = {}
    for region, mask in regions_.items():
        entry = {"cycles": float(cycles[mask].sum() / total)}
        entry |= {name: float(v[mask].sum() / total) for name, v in values.items()}
        entry["relative_bias"] = sum(entry[name] for name in values) / entry["cycles"] - 1
        result[region] = entry
    return result


# --------------------------------------------------------------------------------------------------------------------
# Figures


def tonemap(linear: np.ndarray) -> np.ndarray:
    return np.asarray(linearrgb_to_srgb(np.clip(linear, 0, 1)))


def error_map(ours: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|ours - ref| averaged over channels, smoothed 3x3 for legibility (noise stays visible as a level), over the mean
    of the reference."""
    error = np.abs(ours.astype(np.float64) - ref).mean(-1)
    return ndimage.uniform_filter(error, 3, mode="nearest") / ref.mean()


def signed_map(ours: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Signed difference after a 5x5 blur of both (bias, without most noise), averaged over channels, over the mean of
    the reference."""
    difference = blur(ours[None].astype(np.float64))[0] - blur(ref[None].astype(np.float64))[0]
    return difference.mean(-1) / ref.mean()


def hex_rgb(color: str) -> np.ndarray:
    return np.array([int(color[k : k + 2], 16) / 255 for k in (1, 3, 5)])


def region_image(masks: dict, i: int) -> np.ndarray:
    image = np.ones((*masks["sky"].shape[1:], 3))
    for name in PARTITION:
        image[masks[name][i]] = hex_rgb(REGION_COLORS[name])
    shafts = masks["shafts"][i] & ~masks["edges"][i]
    image[shafts] = image[shafts] * 0.6  # darker where the objects' shadows change the fog's light
    return image


def map_grid(methods, reference, masks, results, maps, colormap, scale, signed: bool, path, caption: str):
    frame, i = 345, FRAMES.index(345)
    ref = reference.images[frame].astype(np.float64)
    panels = [
        ("参考 Cycles 4096 spp (第 345 帧, 320×180)", tonemap(ref)),
        ("区域划分 (暗色 = 光柱)", region_image(masks, i)),
    ]
    for m in methods:
        r = results[m.key]["345"]
        worst = r["worst_local_blurred" if signed else "worst_local"]
        local = r["regions"][worst]["local_blurred" if signed else "local"]
        title = (
            f"{m.label}\nL1 {100 * r['whole']['rel_l1']:.1f}% · 模糊后 {100 * r['whole']['rel_l1_blurred']:.1f}%"
            f" · 最差区域: {REGION_NAMES[worst]} {100 * local:.0f}%"
        )
        panels.append((title, maps(m.images[frame], ref)))
    columns = 3
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.4, rows * 3.1),
        facecolor=SURFACE,
        gridspec_kw={"hspace": 0.5, "wspace": 0.06},
    )
    for ax in axes.flat:
        ax.axis("off")
    for k, (ax, (title, data)) in enumerate(zip(axes.flat, panels)):
        if k < 2:
            ax.imshow(data, interpolation="nearest")
        else:
            image = ax.imshow(data, cmap=colormap, vmin=-scale if signed else 0, vmax=scale, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[n]) for n in PARTITION]
    axes.flat[1].legend(
        handles,
        [REGION_NAMES[n] for n in PARTITION],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7.5,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
    )
    bar = fig.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.022, pad=0.02, aspect=60, extend="both" if signed else "max"
    )
    bar.set_label(caption, color=SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+g}%" if signed else f"{100 * v:g}%")
    )
    bar.outline.set_edgecolor(AXIS)
    fig.savefig(path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def style(ax, axis: str = "x"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def percent(axis):
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))


def contributions_figure(methods: list[Method], results: dict):
    """Each method's relative L1 error stacked by region, as is and once blurred (bias only)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(methods) + 1.6), facecolor=SURFACE, sharey=True)
    labels = []
    for row, m in enumerate(methods):
        key = "both" if "both" in results[m.key] else "345"
        by_region = results[m.key][key]["regions"]
        labels.append(m.label)
        for ax, suffix in zip(axes, ("", "_blurred")):
            left = 0.0
            for name in PARTITION:
                width = by_region[name]["contribution" + suffix]
                ax.barh(row, width, left=left, height=0.62, color=REGION_COLORS[name], edgecolor=SURFACE, linewidth=1.5)
                left += width
            ax.text(left + 0.003, row, f"{100 * left:.1f}%", va="center", fontsize=8, color=SECONDARY)
    titles = ("相对 L1 误差按区域拆分 (噪声 + 偏差)", "两图各 5×5 模糊后 (基本只剩偏差)")
    for ax, title in zip(axes, titles):
        style(ax)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        percent(ax.xaxis)
    axes[0].set_yticks(range(len(methods)), labels, fontsize=9, color=INK)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[n]) for n in PARTITION]
    fig.legend(handles, [REGION_NAMES[n] for n in PARTITION], loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.text(
        0.01,
        0.005,
        "条长 = 总误差; 每段 = 该区域 Σ|方法−参考| / 全图 Σ参考。第 255 与 345 帧之和 (缓存仅 345 帧); 全部 320×180",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(SAVE / "region_contributions.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def edge_figure(methods: list[Method], results: dict):
    chosen = ["closed_form", "eevee_day_default", "unreal_high", "froxel_8_64"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE)
    x = range(len(EDGE_LABELS))
    line = {"linewidth": 2, "markersize": 7, "markeredgecolor": SURFACE, "markeredgewidth": 1.5, "marker": "o"}
    for color, key in zip(SERIES, chosen + ["day_spp4_denoised"]):
        label = next(m.label for m in methods if m.key == key)
        profile = results[key]["both"]["edge_profile"]
        axes[0].plot(x, list(profile["whole"].values()), color=color, label=label, **line)
        if "fog" in profile:
            axes[1].plot(x, list(profile["fog_bias"].values()), color=color, label=label, **line)
    own = results["froxel_8_64"]["vs_closed_form"]["edge_profile"]
    axes[0].plot(
        x, list(own.values()), color=SERIES[5], linestyle="--", label="Froxel 相对闭式解 (自身离散误差)", **line
    )
    share = results["closed_form"]["both"]["edge_profile"]["fog_share"]
    titles = ("整幅图的局部相对误差 (5×5 模糊后)", "雾光的相对偏差 (方法雾光 / 参考雾光 − 1)")
    for ax, title in zip(axes, titles):
        style(ax, "y")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(x, [f"{b}\n雾光占 {100 * share[b]:.0f}%" for b in EDGE_LABELS])
        ax.set_xlabel("到最近物体轮廓的距离 (像素)", color=SECONDARY)
        percent(ax.yaxis)
    axes[1].axhline(0, color=AXIS, linewidth=1)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    fig.savefig(SAVE / "edge_profile.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def sun_figure(results: dict):
    series = [
        ("closed_form", "multiple", "闭式解 多次散射 (对 Cycles volume indirect)"),
        ("closed_form", "single", "闭式解 单次散射 (对 Cycles volume direct)"),
        ("eevee_day_default", "fog", "EEVEE 默认 雾光"),
        ("unreal_high", "fog", "Unreal High 雾光"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    line = {"linewidth": 2, "markersize": 7, "markeredgecolor": SURFACE, "markeredgewidth": 1.5, "marker": "o"}
    for ax, where, title in zip(axes, ("sky", "surfaces"), ("天空像素", "表面像素")):
        for color, (key, part, label) in zip(SERIES, series):
            values = list(results[key]["both"]["sun_profile"][part][where].values())
            kept = [k for k, v in enumerate(values) if v is not None]
            ax.plot(kept, [values[k] for k in kept], color=color, label=label, **line)
        style(ax, "y")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(f"{title}: 雾光相对偏差 vs 视线与太阳方向的夹角", fontsize=10, color=INK, loc="left")
        ax.set_xticks(range(len(SUN_LABELS)), SUN_LABELS)
        ax.set_xlabel("视线与太阳方向的夹角 (g = 0.8, 前向散射峰在小角度)", color=SECONDARY)
        percent(ax.yaxis)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(SAVE / "sun_angle_profile.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    scene = day_scene(FRAMES)
    light = DayLight(scene)
    reference = load_reference(scene)
    closed, terms, occluded = closed_form(scene)
    masks = regions(scene, light)
    methods = load_methods(scene, light, closed)

    results: dict = {
        "frames": list(FRAMES),
        "resolution": [320, 180],
        "regions": {name: REGION_NAMES[name] for name in PARTITION + OVERLAPS},
        "partition": list(PARTITION),
        "pixels_changed_by_any_occlusion": occluded,
        "definitions": {
            "contribution": "sum over the region of |ours - ref| / sum of the whole reference (the partition sums to rel_l1)",
            "local": "sum over the region of |ours - ref| / sum of the reference in the region",
            "bias": "(sum of ours - sum of ref) / sum of ref, in the region",
            "_blurred": "same after a 5x5 box blur of both images",
            "dominant": "partition region with the largest contribution; dominant_share = its contribution / rel_l1",
            "worst_local": "partition region with the largest local blurred error",
            "parts": "per light (surfaces, fog = single + multiple), contributions and signed biases as fractions of the"
            " whole reference's light, relative_bias against the reference's light of that kind in the region",
            "edge_profile": "against the distance in px to the nearest silhouette: local blurred error of the whole image,"
            " and of the fog's light, its relative bias and its share of the reference's light",
            "vs_closed_form": "against the closed form instead of Cycles, normalized by Cycles' light: own discretization",
            "closed_form_terms": "closed form's light scattered more than once split into multiple scattering of sun and"
            " sky, ground light scattered once, and both together, against Cycles' volume indirect, by region",
            "sun_profile": "relative bias of the fog's light (and of its single and multiple parts) against the angle"
            " between rays and the sun, over the sky and over surfaces",
            "closed_form_multiple_by_azimuth": "relative bias of the closed form's multiple scattering in the sky, by band"
            " of elevation and azimuth away from the sun's (30°, elevation 30°)",
        },
        "closed_form_terms": closed_form_terms(terms, reference, masks),
        "closed_form_multiple_by_azimuth": azimuth_at_fixed_elevation(closed, reference, masks, scene),
        "methods": {},
    }
    for m in methods:
        entry: dict = {"label": m.label}
        for frames, name in (((255,), "255"), ((345,), "345"), (FRAMES, "both")):
            if all(f in m.images for f in frames):
                entry[name] = evaluate(m, reference, masks, frames)
        if m.lighting_of_closed_form:
            entry["vs_closed_form"] = against_closed_form(m, closed, reference, masks)
        results["methods"][m.key] = entry
        key = "both" if "both" in entry else "345"
        e = entry[key]
        line = "  ".join(
            f"{n} {100 * e['regions'][n]['contribution']:.2f}/{100 * e['regions'][n]['local_blurred']:.1f}"
            for n in PARTITION + OVERLAPS
        )
        print(
            f"{m.key:20s} {key:4s} L1 {100 * e['whole']['rel_l1']:5.2f}% blur {100 * e['whole']['rel_l1_blurred']:5.2f}%"
            f" ratio {e['whole']['ratio']:.3f} dominant {e['dominant']} | {line}",
            flush=True,
        )

    # One color scale for every panel: the 99th percentile of the closed form's map
    scale = float(np.percentile(error_map(closed.images[345], reference.images[345].astype(np.float64)), 99))
    results["map_scale"] = scale
    map_grid(
        methods,
        reference,
        masks,
        results["methods"],
        error_map,
        ERROR_MAP,
        scale,
        False,
        SAVE / "error_maps_345.png",
        "|方法 − 参考| (3×3 平滑, 噪声仍可见) / 参考全图平均亮度 · 第 345 帧 · 所有面板同一色标",
    )
    map_grid(
        methods,
        reference,
        masks,
        results["methods"],
        signed_map,
        SIGNED_MAP,
        scale,
        True,
        SAVE / "signed_maps_345.png",
        "(方法 − 参考) 两图各 5×5 模糊后 / 参考全图平均亮度 · 红 = 偏亮, 蓝 = 偏暗 · 第 345 帧",
    )
    contributions_figure(methods, results["methods"])
    edge_figure(methods, results["methods"])
    sun_figure(results["methods"])
    (SAVE / "regions.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
