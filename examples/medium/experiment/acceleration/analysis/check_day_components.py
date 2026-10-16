"""Independent check of ``day_components.py``: the day scene's error of each method split by part of the light
(surfaces, single scattering, multiple scattering) and by region, against Cycles' 4096 spp reference.

Every number is recomputed here from the saved images and from the closed form, with its own code for the parts,
masks and metrics, and the causes the analysis gives are tested against alternatives:

- the reference's own noise, from its second seed and from Cycles at 64 spp with two seeds;
- the closed form: the error left when one of its parts is replaced by the reference's (what fixing it would gain);
- whether its multiple scattering lacks a forward peak towards the sun, at fixed elevation of rays, away from the
  ground, and with and without the objects' occlusion;
- its light reflected by the ground and scattered once towards the camera, which its "ground" source holds and which
  the table of multiple scattering holds a second time: the table is rebuilt without that light and the closed form
  is rerun with it;
- EEVEE's fog light against single scattering whose skylight is neither attenuated nor hidden (its world probe);
- Unreal's fog light fitted as a sum of the sun's single scattering, the sky's and multiple scattering;
- the overlap of the regions whose shares the analysis adds up.

Saves ``check_day_components.json`` and ``multiple_by_azimuth.png`` under
``runs/fog-comparison/acceleration/analysis/check_day_components``.
"""

import sys

sys.path.insert(0, "examples/medium/experiment/acceleration")
from common import *

torch.set_num_threads(3)

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import OpenEXR
from scipy import ndimage
from visionsim.medium import scattering
from visionsim.medium.optics import sky_quadrature

FRAMES = (255, 345)
SAVE = OUT / "analysis" / "check_day_components"


# --------------------------------------------------------------------------------------------------------------------
# Metrics: sums over pixels, channels and both frames, as shares of the reference's whole image


def share(ours, reference, norm, mask=None, blurred=False):
    """Absolute error, as a share of norm, over a mask of shape (n, h, w)."""
    diff = blur(ours - reference) if blurred else ours - reference
    keep = np.ones(diff.shape[:-1], bool) if mask is None else mask
    return float(np.abs(diff[keep]).sum() / norm)


def bias(ours, reference, mask=None, local=True, norm=None):
    keep = np.ones(ours.shape[:-1], bool) if mask is None else mask
    difference = float((ours - reference)[keep].sum())
    return difference / float(reference[keep].sum()) if local else difference / norm


def percent(x, digits=2):
    return round(100 * float(x), digits)


# --------------------------------------------------------------------------------------------------------------------
# Multiple scattering without the ground's light gathered unscattered


def multiple_scattering_without_ground_echo(
    medium,
    lighting,
    beta,
    channels,
    n_heights=40,
    height_scale=12.0,
    n_directions=48,
    n_steps=16,
    n_elevations=scattering.TABLE_SIZE[0],
    higher_orders=True,
):
    """:func:`visionsim.medium.scattering.multiple_scattering` without the ground's radiance among the light gathered
    at each height: that light, scattered once towards the camera, is already the closed form's "ground" source."""
    S = scattering
    kwargs = {"dtype": beta.dtype, "device": beta.device}
    g, albedo = medium.anisotropy, medium.albedo
    sky = S._per_channel(lighting.sky, channels, "sky", **kwargs)
    ground = S.ground_radiance(medium, lighting, beta, channels)
    depth_ground = S.depth_above_ground(medium, lighting, beta)
    heights = torch.linspace(0, height_scale, n_heights, **kwargs)
    depth_at = depth_ground[None, :] * torch.exp(-heights)[:, None]
    omega, omega_weights = S._gather_nodes(n_directions, **kwargs)
    suns = []
    for sun in lighting.suns:
        towards = np.asarray(sun.direction, dtype=float)
        sun_z = float(towards[2] / np.linalg.norm(towards))
        if sun_z > 0:
            phase = S.azimuthal_phase(omega.double(), omega.new_tensor(sun_z).double(), g).to(beta.dtype) / (2 * math.pi)
            suns.append((sun_z, phase, S._per_channel(sun.irradiance, channels, "sun", **kwargs)))
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_steps))
    t, w = (t + 1) / 2, w / 2
    once = torch.zeros(n_heights, len(omega), channels, **kwargs)
    escape = torch.zeros(n_heights, len(omega), len(beta), **kwargs)
    v = omega[:, None]
    up = v > 0
    for h in range(n_heights):
        c0 = depth_at[h]
        end = torch.where(
            up, torch.exp(-c0 / torch.where(up, v, torch.ones_like(v))), torch.exp(-(depth_ground - c0) / v.abs())
        )
        escape[h] = end
        y = (1 - end)[:, None, :] * t[None, :, None]
        depth = (c0 + v[..., None] * torch.log1p(-y).clamp_min(-80.0)).clamp_min(0)
        source = sky * S.sky_source(omega, depth, g, n_angle=8, n_azimuth=12)
        source = source + ground * S.ground_source(omega, depth, depth_ground, g, n_angle=8, n_azimuth=12)
        for sun_z, phase, irradiance in suns:
            source = source + phase[:, None, None] * irradiance * torch.exp(-depth / sun_z)
        once[h] = albedo * (source * w[None, :, None]).sum(dim=1) * (1 - end)  # no "+ (~up) * end * ground"
    elevations = S.elevation_grid(n_elevations, **kwargs)
    kernel = S.azimuthal_phase(omega[None, :].double(), elevations[:, None].double(), g).to(beta.dtype)
    second = albedo * torch.einsum("ed,hdc,d->ehc", kernel, once, omega_weights)
    again = albedo * ((1 - escape) * omega_weights[None, :, None]).sum(dim=1) / 2
    if higher_orders:
        second = second / (1 - again[None]).clamp_min(1e-3)
    return S.ScatteringTable(second, depth_ground, height_scale, once, omega)


def ground_echo_in_table(scene: Scene) -> dict:
    """The part of the table of multiple scattering that is the ground's light scattered once, against the "ground"
    source at the same heights and elevations."""
    beta = torch.tensor([float(scene.medium.extinction)], dtype=torch.float64)
    full = scattering.multiple_scattering(scene.medium, scene.lighting, beta, 3)
    without = multiple_scattering_without_ground_echo(scene.medium, scene.lighting, beta, 3)
    sources = scattering.elevation_sources(scene.medium, scene.lighting, beta, 3)
    elevations = scattering.elevation_grid(full.values.shape[0], dtype=torch.float64)
    depth_ground = scattering.depth_above_ground(scene.medium, scene.lighting, beta)
    heights = torch.linspace(0, full.height_scale, full.values.shape[1], dtype=torch.float64)
    depth = (depth_ground * torch.exp(-heights))[None, :, None].expand(len(elevations), -1, 1).contiguous()
    ground = sources["ground"](elevations, depth)
    rows = []
    for h in (0, 3, 8):
        for e in (60, 110, 145, 200):
            echo = float(full.values[e, h, 0] - without.values[e, h, 0])
            rows.append(
                {
                    "height_falloffs": round(float(heights[h]), 2),
                    "elevation_z": round(float(elevations[e]), 3),
                    "ground_source": float(ground[e, h, 0]),
                    "table": float(full.values[e, h, 0]),
                    "table_without_echo": float(without.values[e, h, 0]),
                    "echo": echo,
                    "echo_over_ground_source": echo / float(ground[e, h, 0]),
                }
            )
    return {"rows": rows}


# --------------------------------------------------------------------------------------------------------------------
# Parts of each method


def reference_parts():
    folder = ROOT / "ms" / "cycles_ref"
    whole = load_images(folder / "frames", FRAMES).astype(np.float64)
    single = load_images(folder / "volume" / "direct", FRAMES).astype(np.float64)
    multiple = load_images(folder / "volume" / "indirect", FRAMES).astype(np.float64)
    return {
        "whole": whole,
        "single": single,
        "multiple": multiple,
        "surfaces": whole - single - multiple,
        "fog": single + multiple,
    }


def closed_form(scene: Scene, **options) -> dict:
    """Closed form's parts: surfaces, single scattering (no multiple scattering nor ground light), multiple scattering
    (the rest of its in-scattering), and its ground term alone, with options passed to it."""
    once_medium = scene.medium.model_copy(update={"multiple_scattering": False})
    dark_ground = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    out = {k: [] for k in ("whole", "fog", "single", "ground", "transmittance")}
    for i in range(len(FRAMES)):
        full = scene.closed_form(i, **options)
        single = scene.closed_form(i, medium=once_medium, lighting=dark_ground, **options).inscatter.numpy()
        once = scene.closed_form(i, medium=once_medium, **options).inscatter.numpy()
        out["whole"].append(full.radiance.numpy())
        out["fog"].append(full.inscatter.numpy())
        out["single"].append(single)
        out["ground"].append(once - single)
        out["transmittance"].append(full.transmittance.numpy())
    out = {k: np.stack(v).astype(np.float64) for k, v in out.items()}
    out["surfaces"] = out["whole"] - out["fog"]
    out["multiple"] = out["fog"] - out["single"]
    out["tabulated"] = out["multiple"] - out["ground"]
    return out


def single_by_source(scene: Scene) -> dict:
    """Closed form's single scattering of the sun alone and of the sky alone (attenuated, occluded)."""
    once_medium = scene.medium.model_copy(update={"multiple_scattering": False})
    sun_only = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0), "sky": (0.0, 0.0, 0.0)})
    sky_only = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0), "suns": []})
    result = {}
    for name, lighting in (("sun", sun_only), ("sky", sky_only)):
        result[name] = np.stack(
            [scene.closed_form(i, medium=once_medium, lighting=lighting).inscatter.numpy() for i in range(len(FRAMES))]
        ).astype(np.float64)
    return result


def read_unreal(preset):
    images = []
    for frame in FRAMES:
        with OpenEXR.File(str(OUT / "unreal" / "renders" / f"Day_{preset}" / f"{frame:04d}.exr")) as f:
            images.append(np.asarray(f.channels()["RGBA"].pixels, dtype=np.float64)[..., :3])
    return np.stack(images)


# --------------------------------------------------------------------------------------------------------------------
# Geometry and masks


def geometry(scene: Scene) -> dict:
    sky = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0] >= 1e9
    sun = np.asarray(scene.lighting.suns[0].direction, dtype=np.float64)
    sun /= np.linalg.norm(sun)
    light = DayLight(scene)
    n, h, w = sky.shape
    out = {k: [] for k in ("directions", "points", "visible")}
    for i in range(n):
        origin, directions, length = (x.numpy() for x in scene.rays(i, torch.float64))
        points = origin + directions * length[:, None]
        normals = scene.extra["normals"][i].reshape(-1, 3).astype(np.float64)
        visible = np.ones(len(points))
        solid = ~sky[i].reshape(-1)
        lifted = torch.as_tensor(points[solid] + 0.02 * normals[solid], **KW)
        visible[solid] = light.sun_visibility(lifted).numpy()
        out["directions"].append(directions.reshape(h, w, 3))
        out["points"].append(points.reshape(h, w, 3))
        out["visible"].append(visible.reshape(h, w))
    out = {k: np.stack(v) for k, v in out.items()}
    d = out["directions"]
    out["sky"] = sky
    out["elevation"] = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    out["to_sun"] = np.degrees(np.arccos(np.clip(d @ sun, -1, 1)))
    heading = np.degrees(np.arctan2(d[..., 1], d[..., 0]) - np.arctan2(sun[1], sun[0]))
    out["azimuth"] = np.abs((heading + 180) % 360 - 180)
    out["ground"] = ~sky & (scene.extra["normals"][..., 2] > 0.95) & (np.abs(out["points"][..., 2]) < 0.1)
    return out


def band_masks(values, edges):
    return {f"{lo}..{hi}": (values >= lo) & (values < hi) for lo, hi in zip(edges[:-1], edges[1:])}


ANGLES = (0, 40, 55, 65, 75, 85, 95, 180)


# --------------------------------------------------------------------------------------------------------------------


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    scene = day_scene(FRAMES)
    ref = reference_parts()
    norm = float(ref["whole"].sum())
    results: dict = {}

    # D0: the reference's parts and its own noise
    results["D0"] = {"shares": {p: percent(ref[p].sum() / norm) for p in ("surfaces", "single", "multiple")}}
    seed = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", FRAMES).astype(np.float64)
    results["D0"]["two_seeds_over_sqrt2"] = {
        "whole": percent(share(seed, ref["whole"], norm) / math.sqrt(2)),
        "whole_blurred": percent(share(seed, ref["whole"], norm, blurred=True) / math.sqrt(2)),
    }
    a = load_images(OUT / "day_spp64" / "noisy", FRAMES).astype(np.float64)
    b = load_images(OUT / "day_spp64_seed1" / "noisy", FRAMES).astype(np.float64)
    # Two 64 spp renders differ by noise sqrt(2) times theirs, which is sqrt(4096 / 64) = 8 times the reference's
    results["D0"]["spp64_two_seeds_scaled"] = {
        "whole": percent(share(a, b, norm) / math.sqrt(2) / 8),
        "whole_blurred": percent(share(a, b, norm, blurred=True) / math.sqrt(2) / 8),
    }
    print("D0", results["D0"], flush=True)

    # D1: closed form, by part, and what replacing one part by the reference's would leave
    cf = closed_form(scene)
    saved = np.load(OUT / "results" / "closed_form_day.npy")[[DAY_FRAMES.index(f) for f in FRAMES]]
    results["D1"] = {"max_abs_vs_saved": float(np.abs(cf["whole"] - saved).max())}
    for p in ("whole", "surfaces", "single", "multiple"):
        results["D1"][p] = [
            percent(share(cf[p], ref[p], norm)),
            percent(share(cf[p], ref[p], norm, blurred=True)),
            percent(bias(cf[p], ref[p], local=False, norm=norm)),
        ]
    for p in ("surfaces", "single", "multiple"):
        fixed = cf["whole"] - cf[p] + ref[p]
        results["D1"][f"whole_if_{p}_were_exact"] = [
            percent(share(fixed, ref["whole"], norm)),
            percent(share(fixed, ref["whole"], norm, blurred=True)),
        ]
    dimmed = np.stack([scene.transmittance(i) for i in range(2)]).astype(np.float64) * scene.clear.astype(np.float64)
    results["D1"]["dimmed_surfaces"] = [
        percent(share(dimmed, ref["surfaces"], norm)),
        percent(share(dimmed, ref["surfaces"], norm, blurred=True)),
        percent(bias(dimmed, ref["surfaces"], local=False, norm=norm)),
    ]
    print("D1", results["D1"], flush=True)

    geo = geometry(scene)
    sky, ground, elevation, azimuth, to_sun = (geo[k] for k in ("sky", "ground", "elevation", "azimuth", "to_sun"))
    visible = geo["visible"]
    shadowed_ground = ground & (visible < 0.01)
    results["D1"]["surfaces_local_bias_shadowed_ground_below_-8"] = percent(
        bias(cf["surfaces"], ref["surfaces"], shadowed_ground & (elevation < -8))
    )
    results["D1"]["surfaces_local_bias_shadowed_surfaces"] = percent(
        bias(cf["surfaces"], ref["surfaces"], ~sky & (visible < 0.5))
    )
    results["D1"]["surfaces_share_blurred_in_shadow"] = percent(
        share(cf["surfaces"], ref["surfaces"], norm, ~sky & (visible < 0.5), blurred=True)
    )

    # D2: multiple scattering against the angle to the sun, everywhere and away from rays looking down
    unoccluded = closed_form(scene, occlusion=None)
    blurred_bias = lambda ours, theirs, mask: float((blur(ours - theirs)[mask]).sum() / theirs[mask].sum())
    d2 = {"all": {}, "elevation_above_-2": {}, "sky_only": {}, "single_all": {}, "unoccluded_all": {}}
    for name, mask in band_masks(to_sun, ANGLES).items():
        d2["all"][name] = percent(blurred_bias(cf["multiple"], ref["multiple"], mask), 1)
        d2["single_all"][name] = percent(blurred_bias(cf["single"], ref["single"], mask), 1)
        d2["unoccluded_all"][name] = percent(blurred_bias(unoccluded["multiple"], ref["multiple"], mask), 1)
        up = mask & (elevation >= -2)
        if up.sum() > 500:
            d2["elevation_above_-2"][name] = percent(blurred_bias(cf["multiple"], ref["multiple"], up), 1)
        s = mask & sky
        if s.sum() > 500:
            d2["sky_only"][name] = percent(blurred_bias(cf["multiple"], ref["multiple"], s), 1)
    # At fixed elevation, in the sky, the shape against azimuth of each multiple scattering, over its mean there
    azimuths = (0, 20, 35, 50, 65, 80, 100, 180)
    shapes = {}
    for e_lo, e_hi in ((-2, 5), (5, 15)):
        cell = sky & (elevation >= e_lo) & (elevation < e_hi)
        rows = {}
        for name, a_mask in band_masks(azimuth, azimuths).items():
            m = cell & a_mask
            if m.sum() < 150:
                continue
            rows[name] = {"pixels": int(m.sum())} | {
                k: float(v[m].mean())
                for k, v in (
                    ("reference_multiple", ref["multiple"]),
                    ("cf_multiple", cf["multiple"]),
                    ("cf_unoccluded_multiple", unoccluded["multiple"]),
                    ("reference_single", ref["single"]),
                    ("cf_single", cf["single"]),
                )
            }
        shapes[f"sky_{e_lo}..{e_hi}deg"] = rows
    d2["sky_by_azimuth"] = shapes
    toward = to_sun < 40
    d2["multiple_over_single_toward_sun"] = {
        "reference": float(ref["multiple"][toward].sum() / ref["single"][toward].sum()),
        "closed_form": float(cf["multiple"][toward].sum() / cf["single"][toward].sum()),
        "closed_form_unoccluded": float(unoccluded["multiple"][toward].sum() / unoccluded["single"][toward].sum()),
    }
    results["D2"] = d2
    print("D2", json.dumps(d2)[:1500], flush=True)

    # D3: looking down at the ground; and the ground's light counted twice
    results["D3"] = {"ground_echo_in_table": ground_echo_in_table(scene)}
    sunlit_down = ground & (visible > 0.99) & (elevation < -8)
    shadow_down = ground & (visible < 0.01) & (elevation < -8)
    erode = lambda m: np.stack([ndimage.binary_erosion(x, iterations=3) for x in m])
    scattering.multiple_scattering, original = multiple_scattering_without_ground_echo, scattering.multiple_scattering
    scattering._cached_multiple_scattering.cache_clear()
    try:
        fixed = closed_form(scene)
    finally:
        scattering.multiple_scattering = original
        scattering._cached_multiple_scattering.cache_clear()
    d3 = {}
    for name, mask in (
        ("sunlit_ground_below_-8", sunlit_down),
        ("shadowed_ground_below_-8", shadow_down),
        ("rest", ~(sunlit_down | shadow_down)),
        ("sky", sky),
        ("all", np.ones_like(sky)),
    ):
        eroded = erode(mask) if "ground" in name else mask
        T = cf["transmittance"]
        taken = ((1 - T) * ref["surfaces"] / np.maximum(T, 1e-6))[eroded].sum()
        d3[name] = {
            "pixels_share": float(mask.mean()),
            "local_bias": percent(bias(cf["multiple"], ref["multiple"], mask), 1),
            "bias_share": percent(bias(cf["multiple"], ref["multiple"], mask, local=False, norm=norm)),
            "without_echo_local_bias": percent(bias(fixed["multiple"], ref["multiple"], mask), 1),
            "without_echo_bias_share": percent(bias(fixed["multiple"], ref["multiple"], mask, local=False, norm=norm)),
            "glow_reference": float(ref["multiple"][eroded].sum() / taken),
            "glow_cf": float(cf["multiple"][eroded].sum() / taken),
            "glow_cf_without_echo": float(fixed["multiple"][eroded].sum() / taken),
        }
    d3["without_echo_parts"] = {
        p: [
            percent(share(fixed[p], ref[p], norm)),
            percent(share(fixed[p], ref[p], norm, blurred=True)),
            percent(bias(fixed[p], ref[p], local=False, norm=norm)),
        ]
        for p in ("whole", "surfaces", "multiple")
    }
    d3["without_echo_by_angle"] = {
        name: percent(blurred_bias(fixed["multiple"], ref["multiple"], mask), 1)
        for name, mask in band_masks(to_sun, ANGLES).items()
    }
    d3["by_elevation"] = {
        name: [
            percent(bias(cf["multiple"], ref["multiple"], m), 1),
            percent(bias(fixed["multiple"], ref["multiple"], m), 1),
        ]
        for name, m in band_masks(elevation, (-90, -20, -8, -2, 5, 15, 90)).items()
    }
    results["D3"] |= d3
    print("D3", json.dumps(d3)[:2500], flush=True)

    # D4: surfaces near objects, and the overlap of the sun-shadow regions whose shares were added
    shafts = np.abs(cf["single"] - unoccluded["single"]).sum(-1) > 0.1 * ref["whole"].sum(-1)
    union = shafts | shadow_down
    objects = ~sky & ~ground
    results["D4"] = {
        "object_faces_local_bias": percent(bias(cf["multiple"], ref["multiple"], objects), 1),
        "shafts_pixels": float(shafts.mean()),
        "shadowed_ground_below_-8_pixels": float(shadow_down.mean()),
        "overlap_pixels": float((shafts & shadow_down).mean()),
        "shafts_bias_share": percent(bias(cf["multiple"], ref["multiple"], shafts, local=False, norm=norm)),
        "shadowed_ground_bias_share": percent(
            bias(cf["multiple"], ref["multiple"], shadow_down, local=False, norm=norm)
        ),
        "union_bias_share": percent(bias(cf["multiple"], ref["multiple"], union, local=False, norm=norm)),
        "union_local_bias": percent(bias(cf["multiple"], ref["multiple"], union), 1),
        "union_without_echo_local_bias": percent(bias(fixed["multiple"], ref["multiple"], union), 1),
        "unoccluded_multiple_bias_share": percent(bias(unoccluded["multiple"], ref["multiple"], local=False, norm=norm)),
    }
    print("D4", results["D4"], flush=True)

    # D5: Monte Carlo
    d5 = {}
    for spp in (16, 1024):
        folder = OUT / f"day_spp{spp}"
        noisy = load_images(folder / "noisy", FRAMES).astype(np.float64)
        direct = load_images(folder / "volume" / "direct", FRAMES).astype(np.float64)
        indirect = load_images(folder / "volume" / "indirect", FRAMES).astype(np.float64)
        parts = {"whole": noisy, "single": direct, "multiple": indirect, "surfaces": noisy - direct - indirect}
        d5[spp] = {
            p: [
                percent(share(v, ref[p], norm)),
                percent(share(v, ref[p], norm, blurred=True)),
                percent(bias(v, ref[p], local=False, norm=norm)),
            ]
            for p, v in parts.items()
        }
        d5[spp]["local_multiple"] = percent(share(indirect, ref["multiple"], float(ref["multiple"].sum())))
        d5[spp]["local_single"] = percent(share(direct, ref["single"], float(ref["single"].sum())))
    denoised = load_images(OUT / "day_spp16" / "frames", FRAMES).astype(np.float64)
    d5["16_denoised"] = [
        percent(share(denoised, ref["whole"], norm)),
        percent(share(denoised, ref["whole"], norm, blurred=True)),
        percent(bias(denoised, ref["whole"], local=False, norm=norm)),
    ]
    results["D5"] = d5
    print("D5", d5, flush=True)

    # D6: EEVEE, against single scattering whose skylight is neither attenuated nor hidden
    by_source = single_by_source(scene)
    T = cf["transmittance"]
    up_fraction = []
    for i in range(2):
        d = torch.as_tensor(geo["directions"][i].reshape(-1, 3), dtype=torch.float64)
        # The quadrature's weights sum to the fraction of the phase function above the horizon
        fraction = torch.cat([sky_quadrature(c, scene.medium.anisotropy)[1].sum(-1) for c in d.split(4096)])
        up_fraction.append(fraction.reshape(*sky.shape[1:], 1).numpy())
    sky_radiance = np.asarray(scene.lighting.sky, dtype=np.float64)
    sky_open_hg = (1 - T) * sky_radiance * np.stack(up_fraction)
    sky_open_iso = (1 - T) * sky_radiance * 0.5
    d6 = {"reference_fog_over_single": float(ref["fog"].sum() / ref["single"].sum())}
    for name in ("default", "fine"):
        whole = load_images(OUT / f"eevee_day_{name}" / "frames", FRAMES).astype(np.float64)
        fog = load_images(OUT / f"eevee_day_{name}" / "volume" / "light", FRAMES).astype(np.float64)
        d6[name] = {
            "whole": [
                percent(share(whole, ref["whole"], norm)),
                percent(share(whole, ref["whole"], norm, blurred=True)),
                percent(bias(whole, ref["whole"], local=False, norm=norm)),
            ],
            "fog_local_bias": percent(bias(fog, ref["fog"]), 1),
            "surfaces_local_bias": percent(bias(whole - fog, ref["surfaces"]), 1),
            "fog_over_reference_single": float(fog.sum() / ref["single"].sum()),
            "fog_over_open_sky_single_hg": float(fog.sum() / (by_source["sun"] + sky_open_hg).sum()),
            "fog_over_open_sky_single_iso": float(fog.sum() / (by_source["sun"] + sky_open_iso).sum()),
            "fog_vs_open_sky_single_hg_rel_l1_blurred": percent(
                share(fog, by_source["sun"] + sky_open_hg, float((by_source["sun"] + sky_open_hg).sum()), blurred=True)
            ),
        }
        # If EEVEE's surfaces are those of Cycles' clear image times its own transmittance T_e, its fog lets through
        # T_e / T more of them; a fog that thin would also scatter about (1 - T_e) / (1 - T) of the light
        d6[name]["thin_fog_test"] = {}
        clear = scene.clear.astype(np.float64)
        for region, mask in (
            ("ground_below_-8", ground & (elevation < -8)),
            ("ground_-8..-2", ground & (elevation >= -8) & (elevation < -2)),
            ("objects", ~sky & ~ground),
            ("sky", sky),
        ):
            ratio = float((whole - fog)[mask].sum() / (T * clear)[mask].sum())
            one_minus = float((1 - T)[mask].mean())
            d6[name]["thin_fog_test"][region] = {
                "surfaces_over_T_clear": ratio,
                "implied_scattered_fraction_ratio": (1 - ratio * (1 - one_minus)) / one_minus,
                "fog_over_reference_fog": float(fog[mask].sum() / ref["fog"][mask].sum()),
                "mean_one_minus_T": one_minus,
            }
        d6[name]["by_elevation_fog_over_open_sky_hg"] = {
            k: float(fog[m].sum() / (by_source["sun"] + sky_open_hg)[m].sum())
            for k, m in band_masks(elevation, (-90, -20, -8, -2, 5, 15, 90)).items()
        }
    d6["sun_single_share_of_reference_single"] = float(by_source["sun"].sum() / ref["single"].sum())
    d6["open_sky_hg_over_cf_sky_single"] = float(sky_open_hg.sum() / by_source["sky"].sum())
    results["D6"] = d6
    print("D6", json.dumps(d6)[:2000], flush=True)

    # D7: Unreal's fog light as a combination of the sun's and the sky's single scattering and multiple scattering
    unreal = read_unreal("epic")
    clear_unreal = read_unreal("nofog")
    transmittance = np.stack([scene.transmittance(i) for i in range(2)]).astype(np.float64)
    fog_unreal = unreal - transmittance * clear_unreal
    columns = [blur(by_source["sun"]), blur(by_source["sky"]), blur(ref["multiple"])]
    target = blur(fog_unreal)
    A = np.stack([c.reshape(-1) for c in columns], axis=1)
    coefficients, *_ = np.linalg.lstsq(A, target.reshape(-1), rcond=None)
    fit = A @ coefficients
    residual = float(np.abs(fit - target.reshape(-1)).sum() / np.abs(target).sum())
    # Without the sky term, and without the multiple scattering term
    two = {}
    for name, keep in (("sun_multiple", [0, 2]), ("sun_sky", [0, 1])):
        c2, *_ = np.linalg.lstsq(A[:, keep], target.reshape(-1), rcond=None)
        two[name] = {
            "coefficients": c2.tolist(),
            "rel_residual": float(np.abs(A[:, keep] @ c2 - target.reshape(-1)).sum() / np.abs(target).sum()),
        }
    # With the sun's and the sky's single scattering taken as right, the share of multiple scattering that remains
    rest = (target - columns[0] - columns[1]).reshape(-1)
    held = float(rest @ A[:, 2] / (A[:, 2] @ A[:, 2]))
    two["sun_sky_fixed_multiple_fitted"] = {
        "multiple": held,
        "rel_residual": float(np.abs(rest - held * A[:, 2]).sum() / np.abs(target).sum()),
    }
    correlations = np.corrcoef(A.T).tolist()
    d7 = {
        "whole": [
            percent(share(unreal, ref["whole"], norm)),
            percent(share(unreal, ref["whole"], norm, blurred=True)),
            percent(bias(unreal, ref["whole"], local=False, norm=norm)),
        ],
        "fog_local_bias": percent(bias(fog_unreal, ref["fog"]), 1),
        "fit_sun_sky_multiple": coefficients.tolist(),
        "fit_rel_residual": residual,
        "two_term_fits": two,
        "column_correlations": correlations,
        "clear_unreal_over_cycles": float(clear_unreal.sum() / scene.clear.sum()),
    }
    bands = {}
    for name, mask in band_masks(to_sun, ANGLES).items():
        f = ref["fog"][mask].sum()
        bands[name] = {
            "fog_local_bias": percent(bias(fog_unreal, ref["fog"], mask), 1),
            "sun_single_share": percent(by_source["sun"][mask].sum() / f, 1),
            "sky_single_share": percent(by_source["sky"][mask].sum() / f, 1),
            "multiple_share": percent(ref["multiple"][mask].sum() / f, 1),
        }
    d7["by_angle"] = bands
    results["D7"] = d7
    print("D7", json.dumps(d7)[:2500], flush=True)

    # D8: froxels and caches on frame 345, against the closed form's fog light
    last = [1]
    norm_last = float(ref["whole"][last].sum())
    d8 = {}
    with np.load(OUT / "results" / "froxel_images.npz") as data:
        froxel = data["day_8_64_1"][None].astype(np.float64)
    with np.load(OUT / "results" / "cache_day_images.npz") as data:
        caches = {k: data[k][None].astype(np.float64) for k in ("exact", "sh_1.0_2", "sun_1.0")}
    for name, fog in [("froxel_8_64", froxel)] + list(caches.items()):
        own = [
            percent(share(fog, cf["fog"][last], norm_last)),
            percent(share(fog, cf["fog"][last], norm_last, blurred=True)),
            percent(bias(fog, cf["fog"][last], local=False, norm=norm_last)),
        ]
        whole = cf["surfaces"][last] + fog
        vs = [
            percent(share(whole, ref["whole"][last], norm_last)),
            percent(bias(whole, ref["whole"][last], local=False, norm=norm_last)),
        ]
        d8[name] = {"own": own, "vs_cycles": vs}
    d8["sh_by_elevation_local_bias"] = {
        k: percent(bias(caches["sh_1.0_2"], cf["fog"][last], m[last]), 1)
        for k, m in band_masks(elevation, (-90, -20, -8, -2, 5, 15, 90)).items()
    }
    summary = json.loads((OUT / "results" / "cache.json").read_text())["day"]["variants"]
    d8["sh_orders_and_sizes_vs_closed_form_rel_l1"] = {
        f"order{v['order']}_{v['size']}m": round(100 * v["vs_closed_form"]["rel_l1"], 2)
        for v in summary
        if v["kind"] == "sh"
    }
    results["D8"] = d8
    print("D8", d8, flush=True)

    (SAVE / "check_day_components.json").write_text(json.dumps(results, indent=1))

    # Figure: shape against azimuth, in the sky at 5..15 deg, of each multiple scattering and single scattering
    rows = shapes["sky_5..15deg"]
    centers = [(float(k.split("..")[0]) + float(k.split("..")[1])) / 2 for k in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for key, style, label in (
        ("reference_multiple", "-", "Cycles multiple"),
        ("cf_multiple", "--", "closed form multiple"),
        ("cf_unoccluded_multiple", ":", "closed form multiple, no occlusion"),
        ("reference_single", "-", "Cycles single"),
        ("cf_single", "--", "closed form single"),
    ):
        values = np.array([rows[k][key] for k in rows])
        ax.plot(centers, values / values.mean(), style, marker="o", label=label)
    ax.set_xlabel("azimuth of the ray from the sun's (deg)")
    ax.set_ylabel("mean over sky pixels / its mean over azimuths")
    ax.set_title("Sky pixels at 5-15 deg elevation, frames 255 + 345", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(color="#e0e0dc", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(SAVE / "multiple_by_azimuth.png", dpi=130)
    plt.close(fig)
    print("done", flush=True)


if __name__ == "__main__":
    main()
