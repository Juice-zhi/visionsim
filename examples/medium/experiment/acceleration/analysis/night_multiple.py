"""Night scene: the closed form with light scattered more than once (the lamp's halo), against Cycles with 16 volume
bounces, which the night comparisons of exp_*.py leave out, as they are against light scattered once.

Cycles' volume passes split the fog's light into light scattered once (``direct``) and the rest (``indirect``), which
also holds light that the lamp lit surfaces with and that they reflected into the fog, which the closed form leaves out.
Renders whose surfaces are black (``render_mc.py --black-surfaces``) hold only light scattered more than once by the
fog in ``indirect``, which is what the halo models; the rest of the reference's is light reflected by surfaces. So the
closed form's light scattered once is compared with ``direct``, its halo with the black render's ``indirect``, and their
sum with both, over both frames, and in the shadows of the cube, where no light is scattered once, apart.

Cycles clamps indirect samples (to 10) by default, which darkens the night's light scattered more than once, though not
the day's. So the references are rendered without clamping (``--no-clamp``), and the same comparisons against the
clamped renders are kept under ``clamped``, with how much clamping darkens each part under ``clamp``.

The halo is also split along camera rays by whether it is scattered at points in the lamp's shadow (both cameras
are in the cube's shadow), and Unreal's fog light is estimated from its frames with and without fog, which is exact in
the sky only.

Whole frames are compared too, as the day scene's are: the render without fog (``night_clear``) lit through the fog by
the closed form, with and without the halo, against the reference's frames, with their error split between the fog's
light and surfaces; and EEVEE's and Unreal's frames, which have no fog-free counterpart to split them.

Writes ``analysis/night_multiple/night_multiple.json`` and the images of frame 345 to ``images.npz``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionsim.medium import apply_medium
from visionsim.medium.halos import _interpolate, cached_lamp_halo
from visionsim.medium.occlusion import lamp_map_index, lamp_visibility, maps_to
from visionsim.medium.optics import optical_depth
from visionsim.medium.render import RGB_WAVELENGTHS

from common import DTYPE, NIGHT_FRAMES, OUT, ROOT, day_scene, errors, load_images, night_scene
from exp_unreal import QUALITIES
from exp_unreal import read as read_unreal

HERE = OUT / "analysis" / "night_multiple"
REFERENCES = {
    "unclamped": (OUT / "night_ms_noclamp", OUT / "night_ms_black_noclamp"),
    "clamped": (OUT / "night_ms_ref", OUT / "night_ms_black"),
}


def whole_frames(scene, medium) -> np.ndarray:
    """Frames rendered without fog, lit through the fog by the closed form, as in exp_closed_form.py."""
    clear = load_images(OUT / "night_clear" / "frames", NIGHT_FRAMES)
    normals = day_scene(NIGHT_FRAMES).extra["normals"]
    frames = []
    for i, t in enumerate(scene.transforms):
        result = apply_medium(
            clear[i],
            scene.depth[i],
            t,
            t["transform_matrix"],
            medium,
            scene.lighting,
            occlusion=scene.occlusion,
            normals=normals[i],
            dtype=DTYPE,
        )
        frames.append(result.radiance.cpu().numpy())
    return np.stack(frames)


def halo_by_shadow(scene, medium, nodes: int = 48) -> np.ndarray:
    """The lamp's halo along each camera ray, split into the light scattered towards the camera at points that the
    lamp's shadow map sees lit, and at points in the shadow of objects, of shape (frames, 2, h, w, c). Integrates the
    halo's table along rays as ``halos.halo_inscatter`` does, marking each node with the shadow map."""
    lamp, occlusion = scene.lighting.lamps[0], scene.occlusion
    beta = torch.as_tensor(medium.extinction_at(RGB_WAVELENGTHS), dtype=DTYPE)
    halo = cached_lamp_halo(
        medium, lamp, beta, scene.lighting.ground_height, occlusion.lamp_maps, map_tolerance=occlusion.lamp_tolerance
    )
    maps = maps_to(occlusion.lamp_maps, dtype=torch.float32)
    index = lamp_map_index(maps, halo.center.float(), occlusion.lamp_tolerance)
    x, w = (torch.as_tensor(a, dtype=DTYPE) for a in np.polynomial.legendre.leggauss(nodes))
    frames = []
    for i in range(len(scene.frames)):
        origin, directions, distance = scene.rays(i, DTYPE)
        distance = torch.where(torch.isfinite(distance) & (distance < 1e6), distance, torch.full_like(distance, np.inf))
        parts = []
        for a in range(0, len(directions), 4000):
            d, length = directions[a : a + 4000], distance[a : a + 4000]
            # Nodes spread evenly in the angle at which the lamp sees points along the ray
            offset = halo.center - origin
            along = (d * offset).sum(-1)
            closest = ((offset * offset).sum(-1) - along**2).clamp_min(0).sqrt().clamp_min(halo.extent[0])
            start, end = torch.atan2(-along, closest), torch.atan2(length - along, closest)
            theta = (start + end)[:, None] / 2 + (end - start)[:, None] / 2 * x
            weights = (end - start)[:, None] / 2 * w * closest[:, None] / torch.cos(theta) ** 2
            s = (along[:, None] + closest[:, None] * torch.tan(theta)).clamp_min(0)
            points = origin + s[..., None] * d[:, None, :]
            tau = optical_depth(medium, origin, d[:, None, :], s)[..., None] * beta
            source = _interpolate(halo, points.reshape(-1, 3), (-d[:, None, :]).expand_as(points).reshape(-1, 3))
            light = torch.nan_to_num(torch.exp(-tau) * source.reshape(*s.shape, -1) * weights[..., None], posinf=0.0)
            lit = lamp_visibility(maps, index, points.reshape(-1, 3).float()).reshape(s.shape).to(light)[..., None]
            parts.append(torch.stack([(light * lit).sum(1), (light * (1 - lit)).sum(1)]))
        h, w_ = scene.depth[i].shape
        frames.append(torch.cat(parts, dim=1).reshape(2, h, w_, -1).numpy())
    return np.stack(frames).astype(np.float64)


def unreal_fog_light(scene) -> np.ndarray:
    """Unreal's light scattered by the fog at night, estimated as its frames with fog minus the exact transmittance
    times its frames without fog, which is exact in the sky only."""
    transmittance = np.stack([scene.transmittance(i) for i in range(len(NIGHT_FRAMES))]).astype(np.float64)
    return read_unreal("Night", "epic", NIGHT_FRAMES) - transmittance * read_unreal("Night", "nofog", NIGHT_FRAMES)


def load_reference(name: str) -> dict[str, np.ndarray]:
    path, black = REFERENCES[name]
    parts = {k: load_images(path / "volume" / k, NIGHT_FRAMES) for k in ("direct", "indirect")}
    parts["frames"] = load_images(path / "frames", NIGHT_FRAMES)
    parts["fog"] = parts["direct"] + parts["indirect"]
    parts["fog_only"] = load_images(black / "volume" / "indirect", NIGHT_FRAMES)
    parts["reflected"] = parts["indirect"] - parts["fog_only"]
    parts["surfaces"] = parts["frames"] - parts["fog"]
    return parts


def compare(
    ref: dict,
    once: np.ndarray,
    both: np.ndarray,
    frames_once: np.ndarray,
    frames_both: np.ndarray,
    split: np.ndarray,
    unreal_fog: np.ndarray,
    sky: np.ndarray,
) -> dict:
    """The closed form, EEVEE and Unreal against one reference."""
    halo = both - once
    direct = ref["direct"]
    # Shadows of the cube: pixels whose light scattered once is under 2% of the median of lit pixels
    luma = direct.mean(axis=-1)
    shadow = luma < 0.02 * np.median(luma[luma > 0])
    total = ref["frames"].sum()
    surfaces = frames_both - both
    result = {
        "indirect_over_direct": float(ref["indirect"].sum() / direct.sum()),
        "fog_only_over_direct": float(ref["fog_only"].sum() / direct.sum()),
        "reflected_over_direct": float(ref["reflected"].sum() / direct.sum()),
        "whole": errors(both, ref["fog"]),
        "single_only_vs_whole": errors(once, ref["fog"]),
        "single_vs_direct": errors(once, direct),
        "halo_vs_indirect": errors(halo, ref["indirect"]),
        "halo_vs_fog_only": errors(halo, ref["fog_only"]),
        "halo_vs_fog_only_shadow": errors(halo, ref["fog_only"], shadow),
        "halo_vs_fog_only_lit": errors(halo, ref["fog_only"], ~shadow),
        "fog_only_shadow_share": float(ref["fog_only"][shadow].sum() / ref["fog_only"].sum()),
        "reflected_shadow_share": float(ref["reflected"][shadow].sum() / ref["reflected"].sum()),
        "shadow": {
            "pixels": float(shadow.mean()),
            "reference_share": float(ref["fog"][shadow].sum() / ref["fog"].sum()),
            "whole": errors(both, ref["fog"], shadow),
        },
        "lit": {"whole": errors(both, ref["fog"], ~shadow)},
        "image": {
            "fog_share": float(ref["fog"].sum() / total),
            "single_only": errors(frames_once, ref["frames"]),
            "with_halo": errors(frames_both, ref["frames"]),
            "surfaces": errors(surfaces, ref["surfaces"]),
            # Each part's error over all of the reference's light; they partly cancel within pixels
            "share": {
                "fog": float(np.abs(both - ref["fog"]).sum() / total),
                "surfaces": float(np.abs(surfaces - ref["surfaces"]).sum() / total),
            },
        },
        "eevee": {},
        "unreal": {q: errors(read_unreal("Night", q, NIGHT_FRAMES), ref["frames"]) for q in QUALITIES},
        # Unreal Epic's fog light over Cycles', whose estimate is only exact in the sky
        "unreal_fog_ratio": {
            "all": float(unreal_fog.sum() / ref["fog"].sum()),
            "sky": float(unreal_fog[sky].sum() / ref["fog"][sky].sum()),
            "surfaces": float(unreal_fog[~sky].sum() / ref["fog"][~sky].sum()),
        },
        # The halo split by where along camera rays it is scattered: at points in the lamp's shadow, or lit
        "halo_split": {"replication": float(np.abs(split.sum(axis=1) - halo).sum() / halo.sum())},
    }
    for region, mask in (("all", np.ones_like(shadow)), ("lit_pixels", ~shadow), ("shadow_pixels", shadow)):
        lit, shadowed = split[:, 0][mask].sum(), split[:, 1][mask].sum()
        result["halo_split"][region] = {
            "from_shadowed_points": float(shadowed / (lit + shadowed)),
            "lit_points_over_fog_only": float(lit / ref["fog_only"][mask].sum()),
        }
    for name in ("eevee_night_default", "eevee_night_fine"):
        result["eevee"][name] = {
            "fog_light": errors(load_images(OUT / name / "volume" / "light", NIGHT_FRAMES), ref["fog"]),
            "whole": errors(load_images(OUT / name / "frames", NIGHT_FRAMES), ref["frames"]),
        }
    return result


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    scene = night_scene(NIGHT_FRAMES)
    multiple = scene.medium.model_copy(update={"multiple_scattering": True})
    once, both, seconds = [], [], []
    for i in range(len(NIGHT_FRAMES)):
        once.append(scene.closed_form(i).inscatter.cpu().numpy())
        start = time.perf_counter()
        both.append(scene.closed_form(i, medium=multiple).inscatter.cpu().numpy())
        seconds.append(time.perf_counter() - start)
    once, both = np.stack(once), np.stack(both)
    frames_once, frames_both = whole_frames(scene, scene.medium), whole_frames(scene, multiple)
    split = halo_by_shadow(scene, multiple)
    unreal_fog = unreal_fog_light(scene)
    sky = load_images(ROOT / "clear" / "depths", NIGHT_FRAMES)[..., 0] >= 1e9
    extra = (split, unreal_fog, sky)

    references = {name: load_reference(name) for name in REFERENCES}
    unclamped, clamped = references["unclamped"], references["clamped"]
    results = {
        "frames": NIGHT_FRAMES,
        "seconds_with_halo": min(seconds),
        "single_vs_bounces0": errors(once, scene.reference),
        "black_direct_vs_direct": errors(
            load_images(REFERENCES["unclamped"][1] / "volume" / "direct", NIGHT_FRAMES), unclamped["direct"]
        ),
        # How much brighter each part of the reference is without clamping
        "clamp": {
            part: float(unclamped[part].sum() / clamped[part].sum())
            for part in ("direct", "indirect", "fog_only", "reflected", "fog", "surfaces", "frames")
        },
        **compare(unclamped, once, both, frames_once, frames_both, *extra),
        "clamped": compare(clamped, once, both, frames_once, frames_both, *extra),
    }
    k = NIGHT_FRAMES.index(345)
    np.savez_compressed(
        HERE / "images.npz",
        reference=unclamped["fog"][k],
        direct=unclamped["direct"][k],
        indirect=unclamped["indirect"][k],
        once=once[k],
        both=both[k],
        fog_only=unclamped["fog_only"][k],
        reflected=unclamped["reflected"][k],
    )
    (HERE / "night_multiple.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "clamped"}, indent=1))


if __name__ == "__main__":
    main()
