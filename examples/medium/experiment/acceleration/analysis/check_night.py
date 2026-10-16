"""Independent checks of ``analysis/night.py``'s findings on the night scene (frames 255 and 345, 320x180).

Re-derives each finding's key numbers with separate code (regions, metrics, ray marching, phase truncation, grids,
froxel and model fits) and runs tests that could refute the causes it gives:

- regions: halo (brightest pixels holding 25% of the reference light), wedge (at least 98% of the single-scattered light
  blocked, from the closed form with and without shadows), edge (within 3 px of the wedge's boundary), shadow, rest;
- N0: whether the camera is in the cube's shadow (ray-box test), and the light-weighted distribution of the scattering
  angle over every sample along the rays, not only its per-pixel mean;
- N2: whether the closed form's error at silhouettes comes from the depth pass (one depth per pixel), from the edge of
  the lamp's shadow, or from both, and how the depth pass picks its depth at the cube's silhouette;
- N3: which term of the 1 m 'lit' grid leaks, and at which distances from the lamp the leaked light is scattered;
- N4: the truncated phase function (numpy Legendre series) against Henyey-Greenstein, and per-angle ratios;
- N6: froxel tiles against slices (8 px x 128 and 4 px x 64), and the banding of the error across columns;
- N7: EEVEE's deficit against a model without lamp light near the ground, and an alternative without lamp light near
  the camera;
- N8/N9: Unreal's sky pixels as a mix of the closed form with g = 0.8 and g = 0 (non-negative least squares), the
  lamp's calibration from frames without fog, and the dependence of its fog-light estimate on the ground's checker.

Outputs in ``runs/fog-comparison/acceleration/analysis/check_night/``: ``check_night.json`` and ``rays_<frame>.npz``
(exact marching, per-pixel mean scattering angle and phase truncations). Sections run by name, e.g.
``check_night.py basic footprint rays leak neural froxel eevee unreal misc`` (all by default, in that order); ``leak``,
``neural``, ``eevee`` and ``misc`` read ``rays_<frame>.npz``. Runs on the CPU with 3 threads; reads existing outputs
and recomputes cheap quantities only (about 5 minutes).
"""

from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *

torch.set_num_threads(3)

import OpenEXR
from scipy import ndimage
from scipy.optimize import nnls

import exp_froxel
from exp_cache import LAMP_NODES, NIGHT_BOX

SAVE = OUT / "analysis" / "check_night"
RESULTS = OUT / "results"
FRAMES = NIGHT_FRAMES  # (255, 345)
LAST = 1
CUBE = (np.array([2.5, 16.5, 0.0]), np.array([5.5, 19.5, 3.0]))  # the green cube, 3 m wide, centered at (4, 18, 1.5)
G = 0.8


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# Metrics


def lum(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).sum(-1)


def blur2(x: np.ndarray) -> np.ndarray:
    """5x5 box blur of (n, h, w) images, zero padded as common.blur."""
    return ndimage.uniform_filter(x, size=(1, 5, 5), mode="constant")


def split(ours: np.ndarray, ref: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Overall errors and per-region contribution (raw and blurred), signed bias and local ratio, in % of total ref."""
    o, r = lum(ours), lum(ref)
    total = r.sum()
    d, db = o - r, blur2(o) - blur2(r)
    out = {
        "all": {"ratio": o.sum() / total, "rel_l1": np.abs(d).sum() / total, "rel_l1_blurred": np.abs(db).sum() / total}
    }
    for name, m in masks.items():
        rm = r[m].sum()
        out[name] = {
            "pixels": m.mean(),
            "ref_share": rm / total,
            "err": np.abs(d)[m].sum() / total,
            "err_blurred": np.abs(db)[m].sum() / total,
            "bias": d[m].sum() / total,
            "local_ratio": o[m].sum() / rm if rm > 1e-6 * total else None,
            "local_rel_l1_blurred": np.abs(db)[m].sum() / rm if rm > 1e-6 * total else None,
        }
    return out


def rounded(x, digits=4):
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.{digits}g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# --------------------------------------------------------------------------------------------------------------------
# Regions


def regions(reference: np.ndarray, shadowed: np.ndarray, unshadowed: np.ndarray) -> tuple[dict, np.ndarray]:
    halo, wedge = [], []
    for i in range(len(reference)):
        r = lum(reference[i]).ravel()
        order = np.argsort(-r)
        k = int(np.argmax(np.cumsum(r[order]) >= 0.25 * r.sum())) + 1
        h = np.zeros(r.size, bool)
        h[order[:k]] = True
        halo.append(h.reshape(reference.shape[1:3]))
        wedge.append(lum(shadowed[i]) <= 0.02 * lum(unshadowed[i]))
    halo, wedge = np.stack(halo), np.stack(wedge)
    w = wedge.astype(np.uint8)
    window = (1, 7, 7)
    edge = ndimage.maximum_filter(w, window, mode="nearest") != ndimage.minimum_filter(w, window, mode="nearest")
    edge &= ~halo
    shadow = wedge & ~edge & ~halo
    return {"halo": halo, "edge": edge, "shadow": shadow, "rest": ~(halo | edge | shadow)}, wedge


def silhouette_mask(depth: np.ndarray, jump: float = 1.25) -> np.ndarray:
    """Pixels with a 4-neighbour more than ``jump`` times nearer or farther, dilated by one pixel (3x3)."""
    ld = np.log(np.maximum(depth, 1e-3))
    m = np.zeros(depth.shape, bool)
    dx = np.abs(np.diff(ld, axis=2)) > math.log(jump)
    dy = np.abs(np.diff(ld, axis=1)) > math.log(jump)
    m[:, :, 1:] |= dx
    m[:, :, :-1] |= dx
    m[:, 1:, :] |= dy
    m[:, :-1, :] |= dy
    return ndimage.binary_dilation(m, np.ones((1, 3, 3), bool))


# --------------------------------------------------------------------------------------------------------------------
# Marching along camera rays, with per-sample access


def hg(cos: torch.Tensor, g: float) -> torch.Tensor:
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cos) ** 1.5)


def truncated(cos: np.ndarray, g: float, order: int) -> np.ndarray:
    """Henyey-Greenstein's Legendre series up to an order: sum (2l+1)/(4 pi) g^l P_l(cos)."""
    coefficients = [(2 * l + 1) * g**l / (4 * math.pi) for l in range(order + 1)]
    return np.polynomial.legendre.legval(cos, coefficients)


def samples(scene: Scene, light: NightLight, i: int, n: int = LAMP_NODES, chunk: int = 2048):
    """Chunks of samples along the rays of frame i: start index, points (r, n, 3), directions (r, n, 3), weights
    sigma T dt (r, n), such that the fog light is the weighted sum of the source at the samples."""
    origin, directions, distance = scene.rays(i)
    fog = light.medium.components[0]
    for s in range(0, len(directions), chunk):
        d, l = directions[s : s + chunk], distance[s : s + chunk]
        t, w = lamp_nodes(origin, d, l, light.position, n)
        points = origin + d[:, None, :] * t[..., None]
        dirs = d[:, None, :].expand_as(points)
        sigma = light.beta * fog.density * torch.exp(-(points[..., 2] - fog.base_height) / fog.falloff)
        tau = height_fog_optical_depth(fog, origin, dirs, t) * light.beta
        yield s, points, dirs, sigma * torch.exp(-tau) * w


def march(scene: Scene, light: NightLight, i: int, source) -> np.ndarray:
    """Fog light (h, w) (one channel, the lamp is white) of a source(points (m, 3), dirs (m, 3)) -> (m,)."""
    h, w = scene.depth[i].shape
    out = np.zeros(h * w)
    for s, p, d, wt in samples(scene, light, i):
        v = source(p.reshape(-1, 3), d.reshape(-1, 3)).reshape(wt.shape)
        out[s : s + len(wt)] = (v * wt).sum(-1).double().numpy()
    return out.reshape(h, w)


def exact_source(light: NightLight):
    return lambda p, d: light(p, d)[:, 1]


def cosine(light: NightLight, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    to = light.position - p
    return (d * to).sum(-1) / to.norm(dim=-1).clamp_min(1e-12)


# --------------------------------------------------------------------------------------------------------------------
# The scene's geometry (as in unreal/export_scene.py's description of the Blender scene): exact depth and exact
# visibility of the lamp, to separate the pixel footprint from the lamp's shadow map at the cube's silhouette

BOXES = [((-3.0, 10.0, 1.0), 2.0), ((4.0, 18.0, 1.5), 3.0), ((-6.0, 30.0, 2.0), 4.0), ((7.0, 45.0, 3.0), 6.0)]
POLES = [(10, 15), (-10, 30), (10, 45), (-10, 60), (10, 75), (-10, 90), (10, 105)]  # radius 0.2 m, 8 m tall


def hit_distance(origin: torch.Tensor, d: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Distance along rays (n, 3) from origins (n, 3) or (3,) to the first surface, at most ``limit`` (n,)."""
    o = origin.expand_as(d).double()
    d = d.double()
    best = limit.double().clone()
    with torch.no_grad():
        t = torch.where(d[:, 2] < 0, -o[:, 2] / d[:, 2], torch.full_like(best, math.inf))  # ground z = 0
        best = torch.minimum(best, torch.where(t > 1e-6, t, torch.full_like(t, math.inf)))
        for center, size in BOXES:
            lo = torch.tensor(center, dtype=torch.float64) - torch.tensor([size / 2, size / 2, size / 2])
            hi = lo + size
            inv = 1 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
            t1, t2 = (lo - o) * inv, (hi - o) * inv
            near = torch.minimum(t1, t2).max(-1).values
            far = torch.maximum(t1, t2).min(-1).values
            hit = (far >= near) & (far > 1e-6)
            best = torch.minimum(best, torch.where(hit, near.clamp_min(0), torch.full_like(near, math.inf)))
        for x, y in POLES:
            px, py = o[:, 0] - x, o[:, 1] - y
            a = d[:, 0] ** 2 + d[:, 1] ** 2
            b = 2 * (px * d[:, 0] + py * d[:, 1])
            c = px**2 + py**2 - 0.2**2
            disc = b * b - 4 * a * c
            t = (-b - disc.clamp_min(0).sqrt()) / (2 * a.clamp_min(1e-12))
            z = o[:, 2] + t * d[:, 2]
            hit = (disc >= 0) & (t > 1e-6) & (z >= 0) & (z <= 8)
            best = torch.minimum(best, torch.where(hit, t, torch.full_like(t, math.inf)))
    return best


def geometric_visibility(points: torch.Tensor, lamp: torch.Tensor) -> torch.Tensor:
    """1 where the segment from points (n, 3) to the lamp crosses no object, else 0."""
    to = lamp.double() - points.double()
    r = to.norm(dim=-1)
    blocked = hit_distance(points.double(), to / r[:, None], r - 1e-4) < r - 1e-4
    return (~blocked).to(points.dtype)


def subpixel_rays(camera: dict, ys: np.ndarray, xs: np.ndarray, offsets: np.ndarray):
    """Unit directions (len(ys), k, 3) through pixel (y, x) shifted by offsets (k, 2) in pixels, and the origin."""
    pose = np.asarray(camera["transform_matrix"], dtype=np.float64)
    u = xs[:, None] + 0.5 + offsets[None, :, 0]
    v = ys[:, None] + 0.5 + offsets[None, :, 1]
    local = np.stack([(u - camera["cx"]) / camera["fl_x"], -(v - camera["cy"]) / camera["fl_y"], -np.ones_like(u)], -1)
    d = local @ pose[:3, :3].T
    return torch.tensor(pose[:3, 3]), torch.tensor(d / np.linalg.norm(d, axis=-1, keepdims=True))


def box_exit(origin: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    o = origin.double()
    lo, hi = torch.tensor(BOX_MIN), torch.tensor(BOX_MAX)
    bounds = torch.where(d > 0, hi, lo)
    t = torch.where(d != 0, (bounds - o) / torch.where(d == 0, torch.ones_like(d), d), torch.full_like(d, math.inf))
    return t.min(-1).values


def march_rays(light: NightLight, origin, d, distance, visibility: str, n: int = 128, chunk: int = 2048) -> np.ndarray:
    """Fog light (green channel) along rays (m, 3) up to distance (m,), with the lamp's shadow map or exact geometry."""
    fog = light.medium.components[0]
    out = []
    origin = origin.float()
    for s in range(0, len(d), chunk):
        dd, ll = d[s : s + chunk].float(), distance[s : s + chunk].float()
        t, w = lamp_nodes(origin, dd, ll, light.position, n)
        p = origin + dd[:, None, :] * t[..., None]
        dirs = dd[:, None, :].expand_as(p)
        sigma = light.beta * fog.density * torch.exp(-(p[..., 2] - fog.base_height) / fog.falloff)
        tau = height_fog_optical_depth(fog, origin, dirs, t) * light.beta
        pf, df = p.reshape(-1, 3), dirs.reshape(-1, 3)
        to = light.position - pf
        r = to.norm(dim=-1).clamp_min(1e-6)
        trans = torch.exp(-relative_optical_depth(light.medium, pf, to / r[:, None], r) * light.beta)
        if visibility == "map":
            vis = lamp_visibility(light.maps, light.index, pf)
        else:
            vis = geometric_visibility(pf, light.position).float()
        v = light.falloff(pf, df)[:, 1] * trans * vis
        out.append((v.reshape(t.shape) * sigma * torch.exp(-tau) * w).sum(-1).double().numpy())
    return np.concatenate(out)


def blackman_harris(x: np.ndarray, width: float = 1.5) -> np.ndarray:
    v = 2 * math.pi * (x / width + 0.5)
    return 0.35875 - 0.48829 * np.cos(v) + 0.14128 * np.cos(2 * v) - 0.01168 * np.cos(3 * v)


# --------------------------------------------------------------------------------------------------------------------


def unreal(preset: str, frames=FRAMES) -> np.ndarray:
    images = []
    for f in frames:
        with OpenEXR.File(str(OUT / "unreal" / "renders" / f"Night_{preset}" / f"{f:04d}.exr")) as e:
            images.append(np.asarray(e.channels()["RGBA"].pixels, dtype=np.float64)[..., :3])
    return np.stack(images)


def main(only: set[str]):
    SAVE.mkdir(parents=True, exist_ok=True)
    path = SAVE / "check_night.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    scene = night_scene(FRAMES)
    light = NightLight(scene)
    single = scene.reference.astype(np.float64)
    multiple = (
        load_images(OUT / "night_ms_ref" / "volume" / "direct", FRAMES)
        + load_images(OUT / "night_ms_ref" / "volume" / "indirect", FRAMES)
    ).astype(np.float64)
    second = load_images(OUT / "night_ms_ref" / "volume" / "direct", FRAMES).astype(np.float64)
    cf = np.load(RESULTS / "closed_form_night.npy").astype(np.float64)
    unshadowed = np.stack([scene.closed_form(i, occlusion=None).inscatter.numpy() for i in range(2)]).astype(np.float64)
    masks, wedge = regions(single, cf, unshadowed)
    last = {k: v[[LAST]] for k, v in masks.items()}
    raw_depth = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0]
    sky = raw_depth >= 1e9
    sil = silhouette_mask(scene.depth)
    cache = np.load(RESULTS / "cache_night_images.npz")
    log("loaded")

    if "basic" in only:
        r = {}
        r["regions"] = {
            k: {
                "pixels": [m[i].mean() for i in range(2)],
                "light": [lum(single[i])[m[i]].sum() / lum(single[i]).sum() for i in range(2)],
            }
            for k, m in masks.items()
        }
        # camera inside the cube's shadow: does the segment camera -> lamp cross the cube?
        cams = [np.asarray(t["transform_matrix"])[:3, 3] for t in scene.transforms]
        lamp = light.position.numpy().astype(np.float64)
        crossing = []
        for c in cams:
            s = np.linspace(0, 1, 10001)[:, None]
            seg = c + s * (lamp - c)
            crossing.append(bool((((seg >= CUBE[0]) & (seg <= CUBE[1])).all(-1)).any()))
        r["camera_segment_to_lamp_crosses_cube"] = crossing
        r["camera_to_lamp_m"] = [float(np.linalg.norm(lamp - c)) for c in cams]
        r["second_render_vs_reference"] = split(second, single, masks)
        r["multiple_vs_single"] = split(multiple, single, masks)
        for spp in (64, 1024):
            mc = load_images(OUT / f"night_spp{spp}" / "volume" / "direct", FRAMES).astype(np.float64)
            r[f"mc{spp}_both"] = split(mc, single, masks)
            r[f"mc{spp}_345"] = split(mc[[LAST]], single[[LAST]], last)
        # Closed form: silhouettes, and silhouettes split by whether they are near the wedge's boundary
        r["closed_form_both"] = split(cf, single, masks)
        near_wedge = ndimage.binary_dilation(masks["edge"], np.ones((1, 5, 5), bool))
        sm = {"silhouette": sil, "not_silhouette": ~sil}
        r["closed_form_345_silhouettes"] = split(cf[[LAST]], single[[LAST]], {k: v[[LAST]] for k, v in sm.items()})
        mc1024 = load_images(OUT / "night_spp1024" / "volume" / "direct", FRAMES).astype(np.float64)
        r["mc1024_345_silhouettes"] = split(mc1024[[LAST]], single[[LAST]], {k: v[[LAST]] for k, v in sm.items()})
        fine = {
            "sil_near_wedge": sil & near_wedge,
            "sil_elsewhere": sil & ~near_wedge,
            "edge_sil": masks["edge"] & sil,
            "edge_not_sil": masks["edge"] & ~sil,
            "halo_sil": masks["halo"] & sil,
            "halo_not_sil": masks["halo"] & ~sil,
        }
        for f in range(2):
            r[f"closed_form_{FRAMES[f]}_fine_regions"] = split(
                cf[[f]], single[[f]], {k: v[[f]] for k, v in fine.items()}
            )
            r[f"mc1024_{FRAMES[f]}_fine_regions"] = split(mc1024[[f]], single[[f]], {k: v[[f]] for k, v in fine.items()})
        out["basic"] = r
        log("basic", json.dumps(rounded({k: r[k] for k in ("regions", "camera_segment_to_lamp_crosses_cube")})))

    if "footprint" in only:
        # On the pixels around the cube's silhouette, compare variants of the exact marching against Cycles: pixel
        # center with the depth pass (as the closed form), geometric depth, exact visibility instead of the shadow map,
        # and the pixel's footprint (Blackman-Harris 1.5 px, Cycles' default filter) with geometric depth
        r = {}
        k = 6
        grid = (np.arange(k) + 0.5) / k * 1.5 - 0.75
        oy, ox = np.meshgrid(grid, grid, indexing="ij")
        offsets = np.stack([ox.ravel(), oy.ravel()], -1)
        weights = blackman_harris(offsets[:, 0]) * blackman_harris(offsets[:, 1])
        weights = weights / weights.sum()
        near_wedge = ndimage.binary_dilation(masks["edge"], np.ones((1, 5, 5), bool))
        for f in range(2):
            sel = ndimage.binary_dilation(
                masks["halo"][f] | masks["edge"][f] | (sil[f] & near_wedge[f]), np.ones((3, 3), bool)
            )
            ys, xs = np.nonzero(sel)
            cam = scene.transforms[f]
            origin, center = subpixel_rays(cam, ys, xs, np.zeros((1, 2)))
            center = center[:, 0]
            _, sub = subpixel_rays(cam, ys, xs, offsets)
            sub = sub.reshape(-1, 3)
            o_, dirs_, scale = camera_rays(cam, cam["transform_matrix"])
            pass_distance = torch.tensor(scene.depth[f][ys, xs]) * scale[ys, xs]
            geo_center = hit_distance(origin, center, box_exit(origin, center))
            geo_sub = hit_distance(origin, sub, box_exit(origin, sub))
            variants = {
                "center_pass_map": march_rays(light, origin, center, pass_distance, "map"),
                "center_geo_map": march_rays(light, origin, center, geo_center, "map"),
                "center_pass_geo": march_rays(light, origin, center, pass_distance, "geo"),
                "footprint_geo_map": (march_rays(light, origin, sub, geo_sub, "map").reshape(-1, k * k) * weights).sum(
                    -1
                ),
                "footprint_geo_geo": (march_rays(light, origin, sub, geo_sub, "geo").reshape(-1, k * k) * weights).sum(
                    -1
                ),
            }
            ref = lum(single[f])[ys, xs] / 3
            sec = lum(second[f])[ys, xs] / 3
            cfp = lum(cf[f])[ys, xs] / 3
            groups = {
                "sil_near_wedge": (sil[f] & near_wedge[f])[ys, xs],
                "edge": masks["edge"][f][ys, xs],
                "halo": masks["halo"][f][ys, xs],
                "selected": np.ones(len(ys), bool),
            }
            entry = {
                "pixels": len(ys),
                "depth_pass_vs_geometric_center": {
                    "non_silhouette_median_rel_diff": float(
                        np.median(np.abs(pass_distance.numpy() / geo_center.numpy() - 1)[~sil[f][ys, xs]])
                    ),
                    "silhouette_share_pass_differs_by_5pc": float(
                        (np.abs(pass_distance.numpy() / geo_center.numpy() - 1) > 0.05)[sil[f][ys, xs]].mean()
                    ),
                    "silhouette_share_pass_nearer": float(
                        (pass_distance.numpy() < 0.95 * geo_center.numpy())[sil[f][ys, xs]].mean()
                    ),
                    "silhouette_share_pass_farther": float(
                        (pass_distance.numpy() > 1.05 * geo_center.numpy())[sil[f][ys, xs]].mean()
                    ),
                },
            }
            for g, m in groups.items():
                row = {"ref_share_of_frame": ref[m].sum() * 3 / lum(single[f]).sum()}
                for name, v in {**variants, "closed_form": cfp, "second_render": sec}.items():
                    row[name] = {
                        "local_rel_l1": np.abs(v[m] - ref[m]).sum() / ref[m].sum(),
                        "local_ratio": v[m].sum() / ref[m].sum(),
                        "share_of_frame_error": np.abs(v[m] - ref[m]).sum() * 3 / lum(single[f]).sum(),
                    }
                entry[g] = row
            r[str(FRAMES[f])] = entry
            log(
                "footprint",
                FRAMES[f],
                json.dumps(
                    rounded(
                        {g: {n: v["local_rel_l1"] for n, v in entry[g].items() if isinstance(v, dict)} for g in groups}
                    )
                ),
            )
        out["footprint"] = r

    if "rays" in only:
        r = {}
        bins = np.array([0, 10, 20, 30, 45, 60, 80, 100, 135, 180], float)
        for f in range(2):
            h, w = scene.depth[f].shape
            exact = np.zeros(h * w)
            sh = {o: np.zeros(h * w) for o in (2, 4, 6, 8)}
            sum_cos = np.zeros(h * w)
            hist = np.zeros(len(bins) - 1)
            hist_fine = np.zeros(180)
            for s, p, d, wt in samples(scene, light, f):
                pf, df = p.reshape(-1, 3), d.reshape(-1, 3)
                v = light(pf, df)[:, 1].reshape(wt.shape)
                c = cosine(light, pf, df).reshape(wt.shape)
                contrib = (v * wt).double().numpy()
                cn = c.double().numpy()
                exact[s : s + len(wt)] = contrib.sum(-1)
                sum_cos[s : s + len(wt)] = (contrib * cn).sum(-1)
                ang = np.degrees(np.arccos(np.clip(cn, -1, 1)))
                hist += np.histogram(ang, bins, weights=contrib)[0]
                hist_fine += np.histogram(ang, np.arange(181), weights=contrib)[0]
                ratio_hg = 1 / hg(c.double(), G).numpy()
                for o in sh:
                    sh[o][s : s + len(wt)] = (contrib * truncated(cn, G, o) * ratio_hg).sum(-1)
            mean_angle = np.degrees(np.arccos(np.clip(sum_cos / np.maximum(exact, 1e-30), -1, 1)))
            mean_angle[exact <= 0] = np.nan
            np.savez_compressed(
                SAVE / f"rays_{FRAMES[f]}.npz",
                exact=exact.reshape(h, w),
                mean_angle=mean_angle.reshape(h, w),
                **{f"sh{o}": v.reshape(h, w) for o, v in sh.items()},
            )
            total = hist.sum()
            r[f"sample_angle_share_{FRAMES[f]}"] = {
                f"{bins[k]:g}-{bins[k + 1]:g}": hist[k] / total for k in range(len(bins) - 1)
            }
            r[f"sample_angle_median_{FRAMES[f]}"] = float(np.searchsorted(np.cumsum(hist_fine) / total, 0.5))
            ex = exact.reshape(h, w)
            ma = mean_angle.reshape(h, w)
            share = {}
            for k in range(len(bins) - 1):
                m = (ma >= bins[k]) & (ma < bins[k + 1])
                share[f"{bins[k]:g}-{bins[k + 1]:g}"] = {
                    "light": ex[m].sum() / ex.sum(),
                    **{
                        f"sh{o}_ratio": (sh[o].reshape(h, w)[m].sum() / ex[m].sum()) if ex[m].sum() > 0 else None
                        for o in sh
                    },
                }
            r[f"pixel_mean_angle_bins_{FRAMES[f]}"] = share
            r[f"exact_vs_closed_form_{FRAMES[f]}"] = split(np.repeat(ex[None, ..., None], 3, -1), cf[[f]], {})["all"]
            r[f"exact_vs_saved_cache_exact_{FRAMES[f]}"] = (
                split(np.repeat(ex[None, ..., None], 3, -1), cache["exact"][None] * 1.0, {})["all"]
                if f == LAST
                else None
            )
            for o in sh:
                r[f"sh{o}_vs_exact_{FRAMES[f]}"] = split(sh[o].reshape(1, h, w, 1), ex.reshape(1, h, w, 1), {})["all"]
            if f == LAST:
                r["sh2_regions_vs_exact_345"] = split(
                    sh[2].reshape(1, h, w, 1), ex.reshape(1, h, w, 1), {k: v for k, v in last.items()}
                )
                r["cache_sh2_vs_exact_345"] = split(cache["sh_0.25_2"][None], cache["exact"][None], last)
        angles = np.array([0, 30, 45, 60, 79.5, 90, 133, 180])
        cs = np.cos(np.radians(angles))
        r["phase_ratio_order2"] = {
            f"{a:g}": float(truncated(c, G, 2) / hg(torch.tensor(c), G).item()) for a, c in zip(angles, cs)
        }
        out["rays"] = r
        log("rays", json.dumps(rounded({k: v for k, v in r.items() if "share" in k or "median" in k or "phase" in k})))

    if "leak" in only:
        r = {}
        f = LAST
        h, w = scene.depth[f].shape
        exact = np.load(SAVE / f"rays_{FRAMES[f]}.npz")["exact"]
        low, high = NIGHT_BOX
        for size in (1.0, 0.25):
            axes = [torch.arange(lo, hi + size * 0.5, size, dtype=torch.float32) for lo, hi in zip(low, high)]
            pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            shape = tuple(len(a) for a in axes)
            to = light.position - pts
            rr = to.norm(dim=-1).clamp_min(1e-6)
            tau = (
                torch.cat(
                    [
                        relative_optical_depth(
                            light.medium, pts[k : k + 2**18], (to / rr[:, None])[k : k + 2**18], rr[k : k + 2**18]
                        )
                        for k in range(0, len(pts), 2**18)
                    ]
                )
                * light.beta
            )
            vis = torch.cat(
                [lamp_visibility(light.maps, light.index, pts[k : k + 2**18]) for k in range(0, len(pts), 2**18)]
            )
            grids = {"visibility": vis, "transmittance": torch.exp(-tau), "lit": vis * torch.exp(-tau)}
            kinds = ("lit", "visibility", "transmittance") if size == 1.0 else ("lit",)
            for kind in kinds:
                grid = grids[kind].reshape(*shape, 1)
                dist_bins = np.array([0, 1, 1.5, 2, 3, 4, 6, 10, 100])
                leaked = np.zeros(len(dist_bins) - 1)
                image = np.zeros(h * w)
                for s, p, d, wt in samples(scene, light, f):
                    pf, df = p.reshape(-1, 3), d.reshape(-1, 3)
                    ex = light(pf, df)[:, 1]
                    within = inside(low, high, pf)
                    tq, vq = None, None
                    stored = trilinear(grid, low, size, pf[within])[:, 0]
                    if kind != "lit":
                        to_q = light.position - pf[within]
                        rq = to_q.norm(dim=-1).clamp_min(1e-6)
                        tq = torch.exp(
                            -relative_optical_depth(light.medium, pf[within], to_q / rq[:, None], rq) * light.beta
                        )
                        vq = lamp_visibility(light.maps, light.index, pf[within])
                        stored = stored * (vq if kind == "transmittance" else tq)
                    v = ex.clone()
                    v[within] = light.falloff(pf[within], df[within])[:, 1] * stored
                    contrib = (v.reshape(wt.shape) * wt).double().numpy()
                    image[s : s + len(wt)] = contrib.sum(-1)
                    diff = ((v - ex).reshape(wt.shape) * wt).double().numpy()
                    dist = (light.position - pf).norm(dim=-1).reshape(wt.shape).double().numpy()
                    leaked += np.histogram(dist, dist_bins, weights=diff)[0]
                img = image.reshape(1, h, w, 1)
                key = f"{kind}_{size}"
                r[key] = {
                    "vs_exact": split(img, exact.reshape(1, h, w, 1), last),
                    "leak_by_distance_to_lamp": {
                        f"{dist_bins[k]:g}-{dist_bins[k + 1]:g}": leaked[k] / exact.sum() for k in range(len(leaked))
                    },
                }
                if kind == "lit":
                    saved = cache[f"lit_{size if size == 0.25 else 1.0}_0"][..., 1]
                    r[key]["vs_saved"] = split(img, saved.reshape(1, h, w, 1), {})["all"]
                log("leak", key, json.dumps(rounded(r[key]["vs_exact"]["all"])))
        out["leak"] = r

    if "neural" in only:
        exact = np.load(SAVE / f"rays_{FRAMES[LAST]}.npz")["exact"]
        field = np.load(RESULTS / "learning_field_10945.npy").astype(np.float64)
        r = {
            "vs_single_345": split(field[None], single[[LAST]], last),
            "vs_exact_345": split(field[None][..., 1:2], exact[None, ..., None], last),
        }
        # Is the field's error at the wedge's edge one-sided? Positive and negative parts of the blurred error
        d = blur2(lum(field[None][..., 1:2])) - blur2(exact[None])
        for name, m in last.items():
            r[f"blurred_pos_neg_{name}"] = [d[m][d[m] > 0].sum() / exact.sum(), d[m][d[m] < 0].sum() / exact.sum()]
        lit25 = cache["lit_0.25_0"][..., 1]
        d2 = blur2(lit25[None]) - blur2(exact[None])
        for name, m in last.items():
            r[f"lit025_blurred_pos_neg_{name}"] = [
                d2[m][d2[m] > 0].sum() / exact.sum(),
                d2[m][d2[m] < 0].sum() / exact.sum(),
            ]
        r["lit025_vs_exact_345"] = split(lit25[None, ..., None], exact[None, ..., None], last)
        out["neural"] = r
        log("neural", json.dumps(rounded(r["vs_exact_345"])))

    if "froxel" in only:
        r = {}
        for tile, slices, jitter in ((8, 64, 1), (8, 64, 8), (8, 128, 1), (4, 64, 1), (4, 128, 4), (2, 128, 1)):
            imgs, secs = [], []
            for f in range(2):
                img, s = exp_froxel.froxels(scene, light, f, tile, slices, jitter)
                imgs.append(img)
                secs.append(s)
            imgs = np.stack(imgs).astype(np.float64)
            key = f"{tile}x{slices}_j{jitter}"
            r[key] = {
                "vs_closed_form": split(imgs, cf, masks),
                "vs_single": split(imgs, single, masks)["all"],
                "seconds": secs,
            }
            # Banding: share of the blurred error (rest region, frame 345) explained by per-column means
            d = (blur2(lum(imgs[[LAST]])) - blur2(lum(cf[[LAST]])))[0]
            m = masks["rest"][LAST]
            col = np.array([d[:, x][m[:, x]].mean() if m[:, x].any() else 0 for x in range(d.shape[1])])
            resid = d - col[None, :]
            r[key]["rest_345_column_variance_share"] = 1 - (resid[m] ** 2).sum() / (d[m] ** 2).sum()
            log("froxel", key, json.dumps(rounded(r[key]["vs_closed_form"]["all"])))
        out["froxel"] = r

    if "eevee" in only:
        r = {}
        exact = np.stack([np.load(SAVE / f"rays_{f}.npz")["exact"] for f in FRAMES])[..., None]
        eevee = {
            k: load_images(OUT / f"eevee_night_{k}" / "volume" / "light", FRAMES).astype(np.float64)
            for k in ("default", "fine")
        }
        depth_bins = np.array([0, 4, 6, 8, 10, 15, 30, 1e12])
        depth_value = np.where(sky, 1e9, scene.depth)
        cams = [np.asarray(t["transform_matrix"])[:3, 3] for t in scene.transforms]

        def by_depth(o, ref):
            o, ref = lum(o), lum(ref)
            return {
                f"{depth_bins[k]:g}-{depth_bins[k + 1]:g}": o[
                    (depth_value >= depth_bins[k]) & (depth_value < depth_bins[k + 1])
                ].sum()
                / ref[(depth_value >= depth_bins[k]) & (depth_value < depth_bins[k + 1])].sum()
                for k in range(len(depth_bins) - 1)
            }

        for k, e in eevee.items():
            r[k] = {
                "vs_single": split(e, single, masks),
                "vs_multiple": split(e, multiple, masks)["all"],
                "by_depth_vs_single": by_depth(e, single),
                "sky_vs_exact": lum(e)[sky].sum() / (3 * exact[..., 0][sky].sum()),
            }
        models = {}
        for name, fn in (
            *[
                (f"height>{dl}", (lambda dl: lambda p, d: light(p, d)[:, 1] * (p[:, 2] > dl))(dl))
                for dl in (0.0, 0.15, 0.4)
            ],
        ):
            models[name] = np.stack([march(scene, light, f, fn) for f in range(2)])[..., None]
        for D in (1.0, 2.0, 3.0, 4.0):
            imgs = []
            for f in range(2):
                cam = torch.tensor(cams[f], dtype=torch.float32)
                imgs.append(
                    march(scene, light, f, lambda p, d, cam=cam, D=D: light(p, d)[:, 1] * ((p - cam).norm(dim=-1) > D))
                )
            models[f"camera_distance>{D}"] = np.stack(imgs)[..., None]
        for k, e in eevee.items():
            el = lum(e)[..., None] / 3
            for name, mimg in models.items():
                scale = el[sky].sum() / mimg[sky].sum()
                r[k][f"model_{name}"] = {
                    "scale": scale,
                    "fit": split(el, scale * mimg, {})["all"],
                    "by_depth_model_over_exact": by_depth(scale * mimg, exact),
                }
        out["eevee"] = r
        log(
            "eevee",
            json.dumps(
                rounded(
                    {k: {m: v[m]["fit"]["rel_l1_blurred"] for m in v if m.startswith("model")} for k, v in r.items()}
                )
            ),
        )

    if "unreal" in only:
        r = {}
        T = np.stack([scene.transmittance(i) for i in range(2)]).astype(np.float64)
        nofog = unreal("nofog")
        est = {q: unreal(q) - T * nofog for q in ("high", "epic", "cinematic")}
        u = est["epic"]
        r["sky_pixels"] = sky.mean()
        r["sky_ratio_vs_single"] = {q: lum(e)[sky].sum() / lum(single)[sky].sum() for q, e in est.items()}
        r["sky_ratio_vs_multiple"] = lum(u)[sky].sum() / lum(multiple)[sky].sum()
        r["surface_ratio_vs_multiple"] = lum(u)[~sky].sum() / lum(multiple)[~sky].sum()
        r["all_ratio_vs_multiple"] = lum(u).sum() / lum(multiple).sum()
        cfg = {}
        for g in (-0.2, 0.0, 0.2, 0.8):
            medium = scene.medium.model_copy(update={"anisotropy": g})
            cfg[g] = np.stack([scene.closed_form(i, medium=medium).inscatter.numpy() for i in range(2)]).astype(
                np.float64
            )
            r[f"sky_unreal_over_cf_g{g}"] = lum(u)[sky].sum() / lum(cfg[g])[sky].sum()
            sc = lum(u)[sky].sum() / lum(cfg[g])[sky].sum()
            r[f"sky_fit_scaled_cf_g{g}_rel_l1_blurred"] = (
                np.abs(blur2(lum(u) * sky) - blur2(sc * lum(cfg[g]) * sky)).sum() / lum(u)[sky].sum()
            )
        # Non-negative mix on sky pixels (blurred, both frames): unreal = a cf(0.8) + b cf(0) [+ c (multiple - single)]
        y = blur2(lum(u))[sky]
        A = np.stack([blur2(lum(cfg[0.8]))[sky], blur2(lum(cfg[0.0]))[sky]], -1)
        coef, _ = nnls(A, y)
        r["nnls_sky_a_cf08_b_cf0"] = {
            "a": coef[0],
            "b": coef[1],
            "rel_l1": np.abs(A @ coef - y).sum() / y.sum(),
            "share_from_cf08": (coef[0] * A[:, 0]).sum() / (A @ coef).sum(),
        }
        A3 = np.concatenate([A, blur2(lum(multiple - single))[sky][:, None]], -1)
        coef3, _ = nnls(A3, y)
        r["nnls_sky_a_cf08_b_cf0_c_cycles_multiple_minus_single"] = {
            "coef": coef3.tolist(),
            "rel_l1": np.abs(A3 @ coef3 - y).sum() / y.sum(),
        }
        # Lamp calibration: Unreal without fog against Cycles without fog (night_clear), on surfaces lit by the lamp
        clear = load_images(OUT / "night_clear" / "frames", FRAMES).astype(np.float64)
        lit_surf = (~sky) & (lum(clear) > np.percentile(lum(clear)[~sky], 50))
        r["nofog_unreal_over_cycles_clear_surfaces"] = lum(nofog)[~sky].sum() / lum(clear)[~sky].sum()
        r["nofog_unreal_over_cycles_clear_brighter_half"] = lum(nofog)[lit_surf].sum() / lum(clear)[lit_surf].sum()
        r["nofog_unreal_over_cycles_clear_median_pixel_ratio"] = float(
            np.median(lum(nofog)[lit_surf] / np.maximum(lum(clear)[lit_surf], 1e-12))
        )
        # Checker dependence of the estimate on ground pixels: the clear frame's albedo pattern
        whole_single = load_images(ROOT / "point_shadow" / "bounces0" / "frames", FRAMES).astype(np.float64)
        bias_single = T * clear - (whole_single - single)
        r["cycles_estimator_bias_surfaces_over_single"] = lum(bias_single)[~sky].sum() / lum(single)[~sky].sum()
        # albedo from Unreal nofog / lamp-lit smooth: use ratio of nofog to its 9x9 median as checker indicator
        ground = (~sky) & (scene.depth < 30) & ~masks["shadow"] & ~masks["edge"]
        smooth = ndimage.median_filter(lum(nofog), size=(1, 9, 9))
        bright = ground & (lum(nofog) > 1.3 * smooth)
        dark = ground & (lum(nofog) < 0.77 * smooth)
        r["checker"] = {
            "pixels_bright_dark": [bright.mean(), dark.mean()],
            "unreal_est_over_cycles_single_on_bright_squares": lum(u)[bright].sum() / lum(single)[bright].sum(),
            "unreal_est_over_cycles_single_on_dark_squares": lum(u)[dark].sum() / lum(single)[dark].sum(),
            "cycles_bias_over_single_bright": lum(bias_single)[bright].sum() / lum(single)[bright].sum(),
            "cycles_bias_over_single_dark": lum(bias_single)[dark].sum() / lum(single)[dark].sum(),
            "unreal_negative_estimate_pixels_on_surfaces": float((lum(u)[~sky] < 0).mean()),
        }
        r["shadow_region_unreal_over_single_total"] = lum(u)[masks["shadow"]].sum() / lum(single).sum()
        r["shadow_region_unreal_over_multiple"] = lum(u)[masks["shadow"]].sum() / lum(multiple)[masks["shadow"]].sum()
        out["unreal"] = r
        log("unreal", json.dumps(rounded(r)))

    if "misc" in only:
        r = {}
        # Share of the light by scattering angle per sample, over both frames together
        totals = [np.load(SAVE / f"rays_{f}.npz")["exact"].sum() for f in FRAMES]
        shares = [out["rays"][f"sample_angle_share_{f}"] for f in FRAMES]
        r["sample_angle_share_both"] = {
            k: sum(t * s[k] for t, s in zip(totals, shares)) / sum(totals) for k in shares[0]
        }
        pm = [out["rays"][f"pixel_mean_angle_bins_{f}"] for f in FRAMES]
        r["pixel_mean_angle_share_both"] = {
            k: sum(t * s[k]["light"] for t, s in zip(totals, pm)) / sum(totals) for k in pm[0]
        }
        # Exact light (frame 345) by distance from the sample to the lamp, for the leak's context
        dist_bins = np.array([0, 1, 1.5, 2, 3, 4, 6, 10, 100])
        hist = np.zeros(len(dist_bins) - 1)
        for s, p, d, wt in samples(scene, light, LAST):
            pf, df = p.reshape(-1, 3), d.reshape(-1, 3)
            contrib = (light(pf, df)[:, 1].reshape(wt.shape) * wt).double().numpy()
            hist += np.histogram(
                (light.position - pf).norm(dim=-1).double().numpy(), dist_bins, weights=contrib.ravel()
            )[0]
        r["light_by_distance_to_lamp_345"] = {
            f"{dist_bins[k]:g}-{dist_bins[k + 1]:g}": hist[k] / hist.sum() for k in range(len(hist))
        }
        # How far the halo's rays pass from the lamp
        origin, directions, _ = scene.rays(LAST)
        to = (light.position - origin).double()
        along = directions.double() @ to
        miss = (to.norm() ** 2 - along**2).clamp_min(0).sqrt().numpy().reshape(180, 320)[masks["halo"][LAST]]
        r["halo_ray_distance_from_lamp_p10_median_p90"] = [float(np.percentile(miss, q)) for q in (10, 50, 90)]
        # SH order 2 cache against the angle between the view ray and the direction to the lamp
        view = np.degrees(np.arccos(np.clip((directions.double() @ (to / to.norm())).numpy(), -1, 1))).reshape(180, 320)
        sh2, ex = lum(cache["sh_0.25_2"]), lum(cache["exact"])
        r["cache_sh2_over_exact_by_view_angle"] = {
            f"{a}-{b}": (
                sh2[(view >= a) & (view < b)].sum() / ex[(view >= a) & (view < b)].sum(),
                ex[(view >= a) & (view < b)].sum() / ex.sum(),
            )
            for a, b in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 80))
        }
        # EEVEE: an alternative to the ground cut, a wider shadow of the cube (cube inflated by a margin)
        eevee = {
            k: lum(load_images(OUT / f"eevee_night_{k}" / "volume" / "light", FRAMES))[..., None] / 3
            for k in ("default", "fine")
        }
        global BOXES
        original = list(BOXES)
        for margin in (0.25, 0.5):
            BOXES = [(c, s + 2 * margin) if c == (4.0, 18.0, 1.5) else (c, s) for c, s in original]
            imgs = []
            for f in range(2):
                o, dd, dist = scene.rays(f)
                imgs.append(march_rays(light, o, dd, dist, "geo", n=128).reshape(180, 320))
            model = np.stack(imgs)[..., None]
            for k, e in eevee.items():
                scale = e[sky].sum() / model[sky].sum()
                r[f"eevee_{k}_vs_wider_shadow_{margin}"] = {"scale": scale, "fit": split(e, scale * model, {})["all"]}
            log("wider shadow", margin, json.dumps(rounded({k: v["fit"] for k, v in r.items() if "wider" in k})))
        BOXES = original
        out["misc"] = r
        log("misc", json.dumps(rounded(r)))

    path.write_text(json.dumps(rounded(out, 5), indent=1, default=str))
    log("saved", path)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"basic", "footprint", "rays", "leak", "neural", "froxel", "eevee", "unreal", "misc"})
