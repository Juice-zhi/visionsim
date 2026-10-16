"""Night scene: where in the image, and why, each method's fog light differs from Cycles'.

The night fog (extinction 0.08/m at the ground, falloff 2.5 m, Henyey-Greenstein g = 0.8, albedo 1) is lit by a 3000 W
point lamp at (4, 21, 1.5), 1.5 m behind a 3 m cube. The camera stands inside the cube's shadow, so the cube hides
the lamp, a bright halo surrounds the cube's silhouette, and rays towards the cube and part of the ground never leave
the shadow. Every method is compared on the light the fog scatters towards the camera (Cycles' volume direct pass,
single scattering, 4096 spp), at frames 255 and 345, 320x180; Unreal and EEVEE also against Cycles' multiple
scattering (2048 spp, volume direct + indirect).

Regions, per frame, disjoint, taken in this order:

- ``halo``: the brightest pixels of the reference, which hold a quarter of its light, around the hidden lamp;
- ``edge``: pixels within 3 px of the edge of the shadow wedge in the image, i.e. of the pixels whose rays never leave
  the cube's shadow (at least 98% of their single-scattered light blocked, from the closed form with and without
  shadows), which includes the cube's silhouette;
- ``shadow``: the rest of those pixels, where single scattering is zero, so any light there is an error;
- ``rest``: the rest of the fog.

For each method and region: its contribution to the total relative L1 error (sum over the region of |ours - ref|
over the reference's total, so that regions add up to rel_l1), as is and once both are blurred (bias without Monte
Carlo noise), its signed bias (adds up to ratio - 1), and its local relative error.

Diagnostics of the causes, from quantities along each pixel's ray weighted by the light it scatters (exact lighting):
the scattering angle, the optical depth towards the lamp, the height and the distance to the lamp.

- reference noise, from a second Cycles render (2048 spp, ``night_ms_ref``'s volume direct pass);
- pixel footprint at depth discontinuities (silhouettes), shared by every method that uses one depth per pixel;
- spherical harmonics: the phase function truncated to order L, alone (exact lighting) and in the 0.25 m cache,
  against the scattering angle;
- voxel leaking: 1 m grids storing the lamp's visibility only, its transmittance only, or both, as the ``lit`` cache;
- froxels: the prototype rerun with and without jitter, against the closed form, and the size of a froxel at the lamp;
- EEVEE: its ratio to Cycles against the scattering angle, the pixel's depth, the height and the optical depth
  towards the lamp, and a model of it where fog less than some height above the ground gets no light from the lamp;
- Unreal: its ratio to Cycles' single and multiple scattering, on pixels that see the sky, where its fog light is
  exact, the asymmetry g of the phase function that fits it best, and the bias of the estimate
  ``frame - T * frame without fog`` on surfaces.

Outputs, in ``runs/fog-comparison/acceleration/analysis/night/``: ``night.json``, ``error_maps.png`` (signed error
of every method at frame 345), ``regions.png`` (where each method's error lies) and ``causes.png`` (errors against
the scattering angle and the pixel's depth).
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
import OpenEXR
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from scipy import ndimage

import exp_froxel
from exp_cache import LAMP_NODES, NIGHT_BOX

SAVE = OUT / "analysis" / "night"
RESULTS = OUT / "results"
FRAMES = NIGHT_FRAMES  # (255, 345)
LAST = 1  # index of frame 345, the only frame saved by some experiments
HALO_SHARE, BLOCKED, EDGE = 0.25, 0.98, 3
REGIONS = ("halo", "edge", "shadow", "rest")
LAMP = (4.0, 21.0, 1.5)
BINS = {
    "scatter_angle": np.array([0, 20, 30, 45, 60, 80, 100, 135, 180], dtype=float),
    "view_angle": np.array([0, 10, 20, 30, 45, 60, 80, 180], dtype=float),
    "depth": np.array([0, 4, 6, 8, 10, 15, 30, 1e9], dtype=float),
    "tau_lamp": np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 5], dtype=float),
    "height": np.array([0, 0.5, 1, 1.5, 2, 3, 5, 30], dtype=float),
}


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# Regions and metrics


def make_regions(scene: Scene, shadowed: np.ndarray, unshadowed: np.ndarray) -> dict[str, np.ndarray]:
    """Disjoint masks of shape (n, h, w), see the module's docstring."""
    masks = {name: [] for name in REGIONS}
    square = np.ones((2 * EDGE + 1,) * 2, dtype=bool)
    for i in range(len(scene.frames)):
        ref = scene.reference[i].sum(-1).astype(np.float64)
        order = np.argsort(ref.ravel())[::-1]
        cumulative = np.cumsum(ref.ravel()[order]) / ref.sum()
        halo = np.zeros(ref.size, dtype=bool)
        halo[order[: np.searchsorted(cumulative, HALO_SHARE) + 1]] = True
        halo = halo.reshape(ref.shape)
        blocked = 1 - shadowed[i].sum(-1) / np.maximum(unshadowed[i].sum(-1), 1e-12)
        wedge = blocked >= BLOCKED
        # The image's border isn't an edge of the wedge
        edge = ndimage.binary_dilation(wedge, square) & ~ndimage.binary_erosion(wedge, square, border_value=1)
        edge &= ~halo
        shadow = wedge & ~edge & ~halo
        masks["halo"].append(halo)
        masks["edge"].append(edge)
        masks["shadow"].append(shadow)
        masks["rest"].append(~(halo | edge | shadow))
    return {name: np.stack(m) for name, m in masks.items()}


def silhouettes(depth: np.ndarray, jump: float = 1.25) -> np.ndarray:
    """Pixels next to a depth discontinuity (a neighbor more than ``jump`` times farther or nearer), (n, h, w)."""
    log_depth = np.log(np.maximum(depth, 1e-3))
    edge = np.zeros(depth.shape, dtype=bool)
    for axis in (1, 2):
        step = np.abs(np.diff(log_depth, axis=axis)) > math.log(jump)
        pad = [(0, 0)] * 3
        pad[axis] = (0, 1)
        edge |= np.pad(step, pad)
        pad[axis] = (1, 0)
        edge |= np.pad(step, pad)
    return np.stack([ndimage.binary_dilation(e, np.ones((3, 3), dtype=bool)) for e in edge])


def by_region(ours: np.ndarray, ref: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Errors over the whole image, and each region's share of them, of images of shape (n, h, w, 3). Local errors
    are left out where the reference has no light."""
    ours, ref = np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    total = ref.sum()
    diff, diff_blurred = ours - ref, blur(ours) - blur(ref)
    result = {"all": errors(ours, ref)}
    for name, mask in masks.items():
        m = np.broadcast_to(mask[..., None], ours.shape)
        ref_m = ref[m].sum()
        lit = ref_m > 1e-6 * total
        result[name] = {
            "pixels": float(mask.mean()),
            "reference_share": float(ref_m / total),
            "error_share": float(np.abs(diff)[m].sum() / total),
            "error_share_blurred": float(np.abs(diff_blurred)[m].sum() / total),
            "bias": float(diff[m].sum() / total),
            "local_rel_l1": float(np.abs(diff)[m].sum() / ref_m) if lit else None,
            "local_rel_l1_blurred": float(np.abs(diff_blurred)[m].sum() / ref_m) if lit else None,
            "local_ratio": float(ours[m].sum() / ref_m) if lit else None,
        }
    return result


def by_bins(
    ours: np.ndarray, ref: np.ndarray, values: np.ndarray, edges: np.ndarray, keep: np.ndarray | None = None
) -> list[dict]:
    """Ratio, share of the light and share of the error of images (n, h, w, 3) in bins of a per-pixel value (n, h, w);
    pixels whose value is NaN, or not kept, are left out."""
    ours, ref = np.asarray(ours, dtype=np.float64).sum(-1), np.asarray(ref, dtype=np.float64).sum(-1)
    total = ref.sum()
    keep = np.isfinite(values) if keep is None else keep & np.isfinite(values)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        m = keep & (values >= low) & (values < high)
        lit = ref[m].sum() > 1e-6 * total
        rows.append(
            {
                "range": [float(low), float(high)],
                "pixels": float(m.mean()),
                "reference_share": float(ref[m].sum() / total),
                "ratio": float(ours[m].sum() / ref[m].sum()) if lit else None,
                "error_share": float(np.abs(ours - ref)[m].sum() / total),
                "bias": float((ours - ref)[m].sum() / total),
            }
        )
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Lighting along camera rays


def render(scene: Scene, light: NightLight, i: int, source) -> np.ndarray:
    """Light of the lamp scattered along the camera rays of frame i, given its source J at points (3 values), of shape
    (h, w, 3), with the same equiangular samples as the caches' experiment."""
    origin, directions, distance = scene.rays(i)
    image = by_chunks(
        lambda d, l: integrate_lamp(light, origin, d, l, source, n=LAMP_NODES),
        len(directions),
        4096,
        directions,
        distance,
    )
    return image.reshape(*scene.depth[i].shape, 3).numpy()


def scattering_cosine(light: NightLight, points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Cosine of the angle by which light from the lamp scatters towards the camera, at points, (n,)."""
    to_light = light.position - points
    return (directions * to_light).sum(-1) / to_light.norm(dim=-1).clamp_min(1e-12)


def truncated_phase(cos: torch.Tensor, g: float, order: int) -> torch.Tensor:
    """Henyey-Greenstein phase function truncated to spherical harmonics up to an order: sum of (2l+1)/4π g^l P_l."""
    previous, current = torch.ones_like(cos), cos
    total = torch.ones_like(cos) + (3 * g * cos if order >= 1 else 0)
    for l in range(2, order + 1):
        previous, current = current, ((2 * l - 1) * cos * current - (l - 1) * previous) / l
        total = total + (2 * l + 1) * g**l * current
    return total / (4 * math.pi)


def lamp_terms(light: NightLight, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Optical depth of the fog towards the lamp, and the lamp's visibility through its shadow map, at points, (n,)."""
    to_light = light.position - points
    r = to_light.norm(dim=-1).clamp_min(1e-6)
    return relative_optical_depth(light.medium, points, to_light / r[:, None], r) * light.beta, lamp_visibility(
        light.maps, light.index, points
    )


def along_rays(scene: Scene, light: NightLight, i: int) -> dict[str, np.ndarray]:
    """Per-pixel light of the exact lighting, its truncations to spherical harmonics, and quantities averaged along the
    ray weighted by the light scattered (NaN where no light is), of shape (h, w) each."""

    def first(p, d):
        value, cos = light(p, d)[:, 1], scattering_cosine(light, p, d)
        tau, _ = lamp_terms(light, p)
        return torch.stack([value, value * cos, value * tau], -1)

    def second(p, d):
        value = light(p, d)[:, 1]
        return torch.stack([value * p[:, 2], value * (light.position - p).norm(dim=-1), value * 0], -1)

    def truncations(orders):
        def source(p, d):
            value, cos = light(p, d)[:, 1], scattering_cosine(light, p, d)
            hg = henyey_greenstein(cos, light.g)
            return torch.stack([value * truncated_phase(cos, light.g, o) / hg for o in orders], -1)

        return source

    a, b = render(scene, light, i, first), render(scene, light, i, second)
    t1, t2 = render(scene, light, i, truncations((2, 4, 6))), render(scene, light, i, truncations((8, 8, 8)))
    value = a[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = lambda x: np.where(value > 0, x / value, np.nan)
        return {
            "exact": value,
            "scatter_angle": np.degrees(np.arccos(np.clip(mean(a[..., 1]), -1, 1))),
            "tau_lamp": mean(a[..., 2]),
            "height": mean(b[..., 0]),
            "distance_to_lamp": mean(b[..., 1]),
            "sh2": t1[..., 0],
            "sh4": t1[..., 1],
            "sh6": t1[..., 2],
            "sh8": t2[..., 0],
        }


def grid_source(light: NightLight, size: float, cached: str):
    """The ``lit`` cache, with the transmittance, the visibility or both ('lit') stored on a grid at voxel corners and
    interpolated trilinearly, the rest computed exactly."""
    low, high = NIGHT_BOX
    points, shape = grid_points(low, high, size)
    tau, visible = by_chunks(lambda p: torch.stack(lamp_terms(light, p), -1), len(points), 2**18, points).unbind(-1)
    transmittance = torch.exp(-tau)
    stored = {"lit": transmittance * visible, "transmittance": transmittance, "visibility": visible}[cached]
    grid = stored.reshape(*shape, 1)

    def source(p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        result = light(p, d)
        within = inside(low, high, p)
        q = p[within]
        value = trilinear(grid, low, size, q)[:, 0]
        if cached != "lit":
            tau_q, v = lamp_terms(light, q)
            value = value * (v if cached == "transmittance" else torch.exp(-tau_q))
        result[within] = light.falloff(q, d[within]) * value[:, None]
        return result

    return source


# --------------------------------------------------------------------------------------------------------------------
# Inputs


def unreal_frames(scene: Scene, preset: str) -> np.ndarray:
    images = []
    for frame in scene.frames:
        with OpenEXR.File(str(OUT / "unreal" / "renders" / f"Night_{preset}" / f"{frame:04d}.exr")) as f:
            images.append(np.asarray(f.channels()["RGBA"].pixels, dtype=np.float64)[..., :3])
    return np.stack(images)


def froxel_size(scene: Scene, i: int, tile: int, slices: int, end: float = exp_froxel.END) -> dict:
    """Size of the froxel holding the lamp, with slices spread as EEVEE spreads them up to ``end``."""
    camera = scene.transforms[i]
    origin, directions, scale = camera_rays(camera, camera["transform_matrix"], dtype=torch.float64)
    to_lamp = torch.tensor(LAMP, dtype=torch.float64) - origin
    k = int((directions @ (to_lamp / to_lamp.norm())).reshape(-1).argmax())
    depth = float(to_lamp.norm() / scale.reshape(-1)[k])
    start, mix = exp_froxel.START, exp_froxel.DISTRIBUTION
    u = np.linspace(0, 1, 200001)
    slice_depth = (1 - mix) * (start + u * (end - start)) + mix * start * (end / start) ** u
    index = int(np.interp(depth, slice_depth, u) * slices)
    near, far = np.interp([index / slices, (index + 1) / slices], u, slice_depth)
    return {
        "lamp_depth_m": depth,
        "slice_thickness_m": float(far - near),
        "tile_width_m": tile * depth / float(camera["fl_x"]),
    }


# --------------------------------------------------------------------------------------------------------------------
# Figures

BLUE, GRAY, RED, VIOLET, ORANGE, AQUA, INK, MUTED = (
    "#2a78d6",
    "#f0efec",
    "#e34948",
    "#4a3aa7",
    "#eb6834",
    "#1baf7a",
    "#0b0b0b",
    "#52514e",
)
REGION_COLORS = {"halo": BLUE, "edge": ORANGE, "shadow": AQUA, "rest": "#eda100"}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, GRAY, RED])


def outline(ax, masks: dict[str, np.ndarray], i: int) -> None:
    ax.contour(masks["halo"][i], levels=[0.5], colors=INK, linewidths=0.6)
    ax.contour(masks["shadow"][i] | masks["edge"][i], levels=[0.5], colors=MUTED, linewidths=0.4, linestyles="dashed")


def error_maps(panels: list[tuple[str, np.ndarray, np.ndarray]], references, masks, path) -> None:
    """Signed error, blurred, over the reference's mean, of frame 345, with the halo (solid) and wedge (dashed)."""
    columns = 4
    rows = math.ceil((len(panels) + 3) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.8 * rows), constrained_layout=True)
    axes = axes.ravel()
    for ax, (title, image) in zip(
        axes[:2], zip(("Cycles single scattering, log10", "Cycles multiple scattering, log10"), references)
    ):
        im = ax.imshow(np.log10(image.sum(-1) / 3 + 1e-4), cmap="magma", vmin=-3.5, vmax=0.5)
        outline(ax, masks, LAST)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    regions = np.zeros(masks["halo"][LAST].shape + (3,))
    for name, color in REGION_COLORS.items():
        regions[masks[name][LAST]] = matplotlib.colors.to_rgb(color)
        ax.plot([], [], "s", color=color, label=name)
    ax.imshow(regions)
    ax.set_title("regions (halo: 25% of the light; shadow: no single scattering)", fontsize=8)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    norm = SymLogNorm(linthresh=0.05, vmin=-10, vmax=10, base=10)
    im = None
    for ax, (title, ours, ref) in zip(axes[3:], panels):
        signed = (blur(ours[None]) - blur(ref[None]))[0].sum(-1) / ref.sum(-1).mean()
        im = ax.imshow(signed, cmap=DIVERGING, norm=norm)
        outline(ax, masks, LAST)
        e = errors(ours[None], ref[None])
        ax.set_title(
            f"{title}\nrel L1 {e['rel_l1']:.1%} (blurred {e['rel_l1_blurred']:.1%}), ratio {e['ratio']:.2f}", fontsize=8
        )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[3 + len(panels) :]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[3:], shrink=0.45, location="right")
    cbar.set_label("(ours - ref), 5x5 blurred, / mean(ref)\nred: too bright, blue: too dark")
    fig.suptitle(
        "Night fog light, frame 345, 320x180: signed error of each method against Cycles' single scattering "
        "(unless MULTIPLE); halo solid, shadow wedge dashed",
        fontsize=11,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def region_bars(rows: list[tuple[str, dict]], path) -> None:
    """Where each method's error lies: blurred |error| by region (adds up to rel_l1 blurred), and signed bias."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 0.42 * len(rows) + 2.0), sharey=True, constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    limit = 0.5
    for yi, (name, table) in zip(y, rows):
        start = 0.0
        for region in REGIONS:
            width = table[region]["error_share_blurred"]
            if start < limit:
                left.barh(
                    yi,
                    min(width, limit - start),
                    left=start,
                    color=REGION_COLORS[region],
                    edgecolor="white",
                    linewidth=1,
                )
            start += width
        total = table["all"]["rel_l1_blurred"]
        left.text(
            min(total, limit) + 0.005, yi, f"{total:.1%}" + (" (cut)" if total > limit else ""), va="center", fontsize=8
        )
        positive, negative = 0.0, 0.0
        for region in REGIONS:
            bias = table[region]["bias"]
            base = positive if bias >= 0 else negative
            shown = float(np.clip(base + bias, -limit, limit)) - float(np.clip(base, -limit, limit))
            right.barh(
                yi,
                shown,
                left=float(np.clip(base, -limit, limit)),
                color=REGION_COLORS[region],
                edgecolor="white",
                linewidth=1,
            )
            if bias >= 0:
                positive += bias
            else:
                negative += bias
        net = table["all"]["ratio"] - 1
        right.plot([float(np.clip(net, -limit, limit))], [yi], "o", color=INK, markersize=4)
        right.text(
            0.98 * limit if net >= 0 else -0.98 * limit,
            yi,
            f"{net:+.1%}",
            va="center",
            fontsize=8,
            ha="right" if net >= 0 else "left",
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    left.set_yticks(y, [name for name, _ in rows], fontsize=8)
    left.set_xlim(0, limit + 0.08)
    right.set_xlim(-limit, limit)
    right.axvline(0, color=MUTED, linewidth=0.8)
    left.set_xlabel("blurred |ours - ref| in region / total ref  (stacks to rel L1 blurred)")
    right.set_xlabel("signed (ours - ref) in region / total ref  (dot and number: net ratio - 1)")
    handles = [matplotlib.patches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False, title="region")
    for ax in (left, right):
        ax.grid(axis="x", color="#e0dfdb", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Night fog light, frame 345: where each method's error lies (against single scattering unless noted)",
        fontsize=10,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_bins(ax, rows: list[dict], color: str, label: str, min_share: float = 0.005, x_log: bool = False) -> None:
    kept = [r for r in rows if r["reference_share"] >= min_share and r["ratio"] is not None]
    if x_log:
        x = [math.sqrt(max(r["range"][0], 1.0) * min(r["range"][1], 300.0)) for r in kept]
    else:
        x = [np.mean(r["range"]) for r in kept]
    ax.plot(x, [r["ratio"] for r in kept], "o-", color=color, label=label, markersize=5, linewidth=2)


def causes_figure(results: dict, path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
    a, b, c, d = axes
    theta = np.linspace(0, 180, 721)
    cos = torch.as_tensor(np.cos(np.radians(theta)))
    a.plot(theta, henyey_greenstein(cos, 0.8).numpy(), color=INK, linewidth=2.5, label="Henyey-Greenstein g = 0.8")
    for order, color in ((2, RED), (4, VIOLET), (8, BLUE)):
        a.plot(
            theta,
            truncated_phase(cos, 0.8, order).numpy(),
            color=color,
            linewidth=1.5,
            label=f"spherical harmonics, order {order}",
        )
    share = results["sh"]["by_scatter_angle_345"]["exact"]
    for r in share:
        if r["reference_share"] >= 0.005:
            a.axvspan(*r["range"], color="#cfcdc6", alpha=min(1.0, r["reference_share"] * 1.5), linewidth=0)
    a.axhline(0, color=MUTED, linewidth=0.6)
    a.set_yscale("symlog", linthresh=0.01)
    a.set_xlabel("scattering angle (deg); shading: where the scene's light is")
    a.set_ylabel("phase function (1/sr)")
    a.set_title("(a) Phase function and its truncations", fontsize=10)
    a.legend(fontsize=7)

    rows = results["sh"]["by_scatter_angle_345"]
    for key, color, label in (
        ("sh2", RED, "order 2, exact lighting"),
        ("cache_sh2", "#e87ba4", "order 2 cache (0.25 m)"),
        ("sh4", VIOLET, "order 4, exact lighting"),
        ("sh8", BLUE, "order 8, exact lighting"),
    ):
        plot_bins(b, rows[key], color, label)
    for r in rows["exact"]:
        if r["reference_share"] >= 0.005:
            b.text(np.mean(r["range"]), -1.35, f"{r['reference_share']:.0%}", ha="center", fontsize=7, color=MUTED)
    b.axhline(1, color=MUTED, linewidth=0.6)
    b.axhline(0, color=MUTED, linewidth=0.4, linestyle="dotted")
    b.set_ylim(-1.5, 3)
    b.set_xlabel("light-weighted scattering angle along the pixel's ray (deg); % = share of the light")
    b.set_ylabel("ratio to the exact phase function")
    b.set_title("(b) Spherical harmonics, frame 345", fontsize=10)
    b.legend(fontsize=7, loc="upper right")

    methods = results["methods"]
    plot_bins(
        c,
        methods["EEVEE default (8px x 64)"]["both_frames"]["by_scatter_angle_vs_single"],
        ORANGE,
        "EEVEE default / single",
    )
    plot_bins(
        c, methods["EEVEE fine (2px x 256)"]["both_frames"]["by_scatter_angle_vs_single"], AQUA, "EEVEE fine / single"
    )
    plot_bins(c, results["unreal"]["sky_by_scatter_angle_vs_single"], RED, "Unreal epic / single (sky pixels)")
    plot_bins(c, results["unreal"]["sky_by_scatter_angle_vs_multiple"], VIOLET, "Unreal epic / multiple (sky pixels)")
    plot_bins(
        c,
        results["unreal"]["phase_fit"]["g=0.0"]["by_scatter_angle_sky"],
        "#e87ba4",
        "Unreal epic / single with ISOTROPIC phase (sky pixels)",
    )
    plot_bins(c, results["multiple_by_scatter_angle"], INK, "Cycles multiple / single")
    c.axhline(1, color=MUTED, linewidth=0.6)
    c.set_yscale("log")
    c.set_xlabel("light-weighted scattering angle along the pixel's ray (deg)")
    c.set_ylabel("ratio of fog light")
    c.set_title("(c) Renderers against Cycles, frames 255 + 345", fontsize=10)
    c.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    plot_bins(
        d,
        methods["EEVEE default (8px x 64)"]["both_frames"]["by_depth_vs_single"],
        ORANGE,
        "EEVEE default (8 px, 64 slices to 100 m)",
        x_log=True,
    )
    plot_bins(
        d,
        methods["EEVEE fine (2px x 256)"]["both_frames"]["by_depth_vs_single"],
        AQUA,
        "EEVEE fine (2 px, 256 slices to 300 m)",
        x_log=True,
    )
    plot_bins(
        d,
        methods["froxel 8px x 64, jitter 8"]["both_frames"]["by_depth_vs_single"],
        BLUE,
        "froxel prototype 8 px, 64 slices, jitter 8",
        x_log=True,
    )
    cuts = results["eevee_ground_cut"]
    for key, color in (("default", ORANGE), ("fine", AQUA)):
        best = min(
            (k for k in cuts if k.startswith("delta=")),
            key=lambda k: cuts[k][f"eevee_{key}_vs_scaled_cut"]["rel_l1_blurred"],
        )
        scale = cuts["sky_scale"][key]
        rows = [dict(r, ratio=None if r["ratio"] is None else r["ratio"] * scale) for r in cuts[best]["cut_by_depth"]]
        kept = [r for r in rows if r["reference_share"] >= 0.005 and r["ratio"] is not None]
        x = [math.sqrt(max(r["range"][0], 1.0) * min(r["range"][1], 300.0)) for r in kept]
        d.plot(
            x,
            [r["ratio"] for r in kept],
            "--",
            color=color,
            linewidth=1.5,
            label=f"model: {scale:.3f} x exact, no lamp light below {best[6:]} m",
        )
    d.axhline(1, color=MUTED, linewidth=0.6)
    d.set_xscale("log")
    d.set_xlabel("depth of the pixel's surface (m; sky drawn at 300 m)")
    d.set_ylabel("ratio to Cycles single scattering")
    d.set_title("(d) Froxel renderers against the pixel's depth, frames 255 + 345", fontsize=10)
    d.legend(fontsize=7)
    for ax in axes:
        ax.grid(color="#e0dfdb", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    scene = night_scene(FRAMES)
    light = NightLight(scene)
    single = scene.reference.astype(np.float64)
    multiple_whole = load_images(OUT / "night_ms_ref" / "frames", FRAMES).astype(np.float64)
    multiple = (
        load_images(OUT / "night_ms_ref" / "volume" / "direct", FRAMES)
        + load_images(OUT / "night_ms_ref" / "volume" / "indirect", FRAMES)
    ).astype(np.float64)
    second = load_images(OUT / "night_ms_ref" / "volume" / "direct", FRAMES)
    closed_form = np.load(RESULTS / "closed_form_night.npy")
    unshadowed = np.stack([scene.closed_form(i, occlusion=None).inscatter.numpy() for i in range(2)])
    masks = make_regions(scene, closed_form, unshadowed)
    masks_last = {name: m[LAST : LAST + 1] for name, m in masks.items()}
    sky = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0] >= 1e9
    edges = silhouettes(scene.depth)
    sil_masks = {"silhouette": edges, "elsewhere": ~edges}
    sil_masks_last = {k: v[LAST : LAST + 1] for k, v in sil_masks.items()}
    depth_value = np.where(sky, 1e6, scene.depth)
    log("regions", {name: [float(m[i].mean()) for i in range(2)] for name, m in masks.items()})

    rays = [along_rays(scene, light, i) for i in range(2)]
    path = {key: np.stack([r[key] for r in rays]) for key in rays[0]}
    exact = np.repeat(path["exact"][..., None], 3, -1)
    sh = {o: np.repeat(path[f"sh{o}"][..., None], 3, -1) for o in (2, 4, 6, 8)}
    log("rays done")

    # Methods' fog light: {name: {frame index: image}}
    cache = np.load(RESULTS / "cache_night_images.npz")
    saved_froxels = np.load(RESULTS / "froxel_images.npz")
    methods: dict[str, dict[int, np.ndarray]] = {}
    for spp in (64, 1024):
        images = load_images(OUT / f"night_spp{spp}" / "volume" / "direct", FRAMES)
        methods[f"Cycles {spp} spp"] = {0: images[0], 1: images[1]}
    methods["closed form"] = {0: closed_form[0], 1: closed_form[1]}
    for key, name in (
        ("exact", "cache: none (same marching)"),
        ("lit_1.0_0", "cache lit 1 m"),
        ("lit_0.25_0", "cache lit 0.25 m"),
        ("irradiance_0.25_0", "cache irradiance 0.25 m"),
        ("sh_0.25_2", "cache SH order 2, 0.25 m"),
    ):
        methods[name] = {LAST: cache[key]}
    methods["neural field 10945"] = {LAST: np.load(RESULTS / "learning_field_10945.npy")}
    froxel_seconds = {}
    for tile, slices, jitter in ((8, 64, 1), (8, 64, 8), (4, 128, 4), (2, 128, 1)):
        name = f"froxel {tile}px x {slices}, jitter {jitter}"
        runs = [exp_froxel.froxels(scene, light, i, tile, slices, jitter) for i in range(2)]
        methods[name] = {i: runs[i][0] for i in range(2)}
        froxel_seconds[name] = min(r[1] for r in runs)
    log("froxels done")
    for key, name in (("default", "EEVEE default (8px x 64)"), ("fine", "EEVEE fine (2px x 256)")):
        images = load_images(OUT / f"eevee_night_{key}" / "volume" / "light", FRAMES)
        methods[name] = {0: images[0], 1: images[1]}
    transmittance = np.stack([scene.transmittance(i) for i in range(2)])
    unreal_nofog = unreal_frames(scene, "nofog")
    unreal = unreal_frames(scene, "epic") - transmittance * unreal_nofog
    methods["Unreal epic (8px x 128)"] = {0: unreal[0], 1: unreal[1]}

    checks = {
        "froxel_8_64_1_vs_saved": errors(
            methods["froxel 8px x 64, jitter 1"][LAST][None], saved_froxels["night_8_64_1"][None]
        ),
        "froxel_4_128_4_vs_saved": errors(
            methods["froxel 4px x 128, jitter 4"][LAST][None], saved_froxels["night_4_128_4"][None]
        ),
        "unreal_vs_saved": errors(unreal[LAST][None], np.load(RESULTS / "unreal_night_epic.npy")[None]),
        "exact_marching_vs_saved_cache_exact": errors(exact[[LAST]], cache["exact"][None]),
    }
    log("checks", json.dumps(checks))

    def subset(images: dict[int, np.ndarray], indices) -> np.ndarray:
        return np.stack([images[i] for i in indices])

    results: dict = {
        "settings": {
            "halo_share": HALO_SHARE,
            "blocked": BLOCKED,
            "edge_px": EDGE,
            "frames": list(FRAMES),
            "resolution": [320, 180],
            "bins": {k: v.tolist() for k, v in BINS.items()},
        },
        "checks": checks,
        "regions": {
            name: {
                "pixels": [float(m[i].mean()) for i in range(2)],
                "single_share": [float(single[i][m[i]].sum() / single[i].sum()) for i in range(2)],
            }
            for name, m in masks.items()
        },
        "silhouette_pixels": [float(edges[i].mean()) for i in range(2)],
        "reference_noise": {
            "second_render_2048spp_vs_reference_4096spp": by_region(second, single, masks),
            "note": "|a - b| of independent noise; the 4096 spp reference alone is about this / sqrt(3)",
        },
        "multiple_vs_single": by_region(multiple, single, masks),
        "multiple_by_scatter_angle": by_bins(multiple, single, path["scatter_angle"], BINS["scatter_angle"]),
        "light_by_scatter_angle": by_bins(exact, exact, path["scatter_angle"], BINS["scatter_angle"]),
        "methods": {},
    }
    for name, images in methods.items():
        frames = sorted(images)
        entry = {
            "frame_345": {
                "vs_single": by_region(subset(images, [LAST]), single[[LAST]], masks_last),
                "silhouettes": by_region(subset(images, [LAST]), single[[LAST]], sil_masks_last),
            }
        }
        if len(frames) == 2:
            ours = subset(images, frames)
            entry["both_frames"] = {
                "vs_single": by_region(ours, single, masks),
                "silhouettes": by_region(ours, single, sil_masks),
                **{
                    f"by_{key}_vs_single": by_bins(ours, single, depth_value if key == "depth" else path[key], BINS[key])
                    for key in ("scatter_angle", "depth", "tau_lamp", "height")
                },
            }
        if name.startswith(("EEVEE", "Unreal")):
            entry["frame_345"]["vs_multiple"] = by_region(subset(images, [LAST]), multiple[[LAST]], masks_last)
            entry["both_frames"]["vs_multiple"] = by_region(ours, multiple, masks)
            entry["both_frames"]["by_scatter_angle_vs_multiple"] = by_bins(
                ours, multiple, path["scatter_angle"], BINS["scatter_angle"]
            )
        if name.startswith("froxel"):
            entry["both_frames"]["vs_closed_form"] = by_region(ours, closed_form, masks)
            entry["seconds"] = froxel_seconds[name]
        if name.startswith(("cache", "neural")):
            entry["frame_345"]["vs_exact_marching"] = by_region(images[LAST][None], exact[[LAST]], masks_last)
        results["methods"][name] = entry
        log(name, json.dumps(entry["frame_345"]["vs_single"]["all"]))

    # Unreal: where pixels see the sky, its frame without fog is black and the estimate of its fog light is exact. On
    # surfaces, ``frame - T * frame without fog`` also removes the light that surfaces lose because the lamp's light
    # reaches them through the fog, which Cycles' renders measure: T * clear - (whole - fog light)
    clear = load_images(OUT / "night_clear" / "frames", FRAMES).astype(np.float64)
    single_whole = load_images(ROOT / "point_shadow" / "bounces0" / "frames", FRAMES).astype(np.float64)
    estimate_bias = {
        "single": transmittance * clear - (single_whole - single),
        "multiple": transmittance * clear - (multiple_whole - multiple),
    }
    results["unreal"] = {
        "by_pixel_kind": {
            kind: {
                "pixels": float(mask.mean()),
                "ratio_vs_single": float(unreal[mask].sum() / single[mask].sum()),
                "ratio_vs_multiple": float(unreal[mask].sum() / multiple[mask].sum()),
                "nofog_light_over_fog_light": float(
                    (transmittance * unreal_nofog)[mask].sum() / max(unreal[mask].sum(), 1e-30)
                ),
                "estimate_bias_if_matching_cycles_single": float(
                    estimate_bias["single"][mask].sum() / single[mask].sum()
                ),
                "estimate_bias_if_matching_cycles_multiple": float(
                    estimate_bias["multiple"][mask].sum() / multiple[mask].sum()
                ),
                "ratio_vs_multiple_after_removing_that_bias": float(
                    (unreal + estimate_bias["multiple"])[mask].sum() / multiple[mask].sum()
                ),
            }
            for kind, mask in (("sky", sky), ("surfaces", ~sky))
        },
        "sky_by_region_vs_multiple": by_region(
            unreal * sky[..., None], multiple * sky[..., None], {k: v & sky for k, v in masks.items()}
        ),
        "sky_by_scatter_angle_vs_single": by_bins(
            unreal, single, path["scatter_angle"], BINS["scatter_angle"], keep=sky
        ),
        "sky_by_scatter_angle_vs_multiple": by_bins(
            unreal, multiple, path["scatter_angle"], BINS["scatter_angle"], keep=sky
        ),
        "qualities_fog_light_vs_single": {
            q: float((unreal_frames(scene, q) - transmittance * unreal_nofog)[sky].sum() / single[sky].sum())
            for q in ("high", "epic", "cinematic")
        },
    }
    # Which phase function Unreal's fog light follows: the closed form's single scattering with other asymmetries g,
    # on pixels that see the sky
    fits = {}
    for g in (0.0, 0.2, 0.4, 0.6, 0.8):
        medium = scene.medium.model_copy(update={"anisotropy": g})
        phase_g = np.stack([scene.closed_form(i, medium=medium).inscatter.numpy() for i in range(2)]).astype(np.float64)
        fits[f"g={g}"] = {
            "unreal_vs_closed_form_g_sky": errors(unreal * sky[..., None], phase_g * sky[..., None]),
            "closed_form_g_over_g0.8": float(phase_g.sum() / closed_form.sum()),
        }
        if g == 0.0:
            fits["g=0.0"]["by_scatter_angle_sky"] = by_bins(
                unreal, phase_g, path["scatter_angle"], BINS["scatter_angle"], keep=sky
            )
    results["unreal"]["phase_fit"] = fits
    log("unreal done", json.dumps(results["unreal"]["by_pixel_kind"]))

    # EEVEE: compared with the exact lighting where fog less than delta above the ground gets no light from the lamp,
    # scaled to match on pixels that see the sky
    eevee = {
        key: subset(methods[name], (0, 1))
        for key, name in (("default", "EEVEE default (8px x 64)"), ("fine", "EEVEE fine (2px x 256)"))
    }
    results["eevee_ground_cut"] = {"sky_scale": {k: float(v[sky].sum() / exact[sky].sum()) for k, v in eevee.items()}}
    for delta in (0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        cut = np.stack(
            [render(scene, light, i, lambda p, d, delta=delta: light(p, d) * (p[:, 2:3] > delta)) for i in range(2)]
        )
        entry = {
            "cut_over_exact": float(cut.sum() / exact.sum()),
            "cut_by_depth": by_bins(cut, exact, depth_value, BINS["depth"]),
        }
        for key, images in eevee.items():
            entry[f"eevee_{key}_vs_scaled_cut"] = errors(images, results["eevee_ground_cut"]["sky_scale"][key] * cut)
        results["eevee_ground_cut"][f"delta={delta}"] = entry
    log("eevee done")

    # Spherical harmonics: the truncated phase alone (exact lighting) and the cache, against the scattering angle
    angles = np.array([0, 10, 20, 30, 45, 60, 80, 90, 100, 120, 133, 150, 180], dtype=float)
    cos = torch.as_tensor(np.cos(np.radians(angles)))
    hg = henyey_greenstein(cos, 0.8)
    angle_last = path["scatter_angle"][[LAST]]
    results["sh"] = {
        "phase_ratio_truncated_over_hg": {
            str(order): {f"{a:g}": float(r) for a, r in zip(angles, (truncated_phase(cos, 0.8, order) / hg).numpy())}
            for order in (2, 4, 6, 8)
        },
        "truncated_exact_lighting_345": {
            str(order): {
                "vs_exact": errors(sh[order][[LAST]], exact[[LAST]]),
                "vs_single": errors(sh[order][[LAST]], single[[LAST]]),
            }
            for order in (2, 4, 6, 8)
        },
        "by_scatter_angle_345": {
            "exact": by_bins(exact[[LAST]], exact[[LAST]], angle_last, BINS["scatter_angle"]),
            "cache_sh2": by_bins(cache["sh_0.25_2"][None], cache["exact"][None], angle_last, BINS["scatter_angle"]),
            "cache_sh2_vs_single": by_bins(cache["sh_0.25_2"][None], single[[LAST]], angle_last, BINS["scatter_angle"]),
            **{f"sh{o}": by_bins(sh[o][[LAST]], exact[[LAST]], angle_last, BINS["scatter_angle"]) for o in (2, 4, 6, 8)},
        },
        "regions_sh2_exact_lighting_vs_exact_345": by_region(sh[2][[LAST]], exact[[LAST]], masks_last),
    }
    origin, directions, _ = scene.rays(LAST)
    to_lamp = light.position - origin
    view_angle = np.degrees(np.arccos((directions @ (to_lamp / to_lamp.norm())).clamp(-1, 1).numpy())).reshape(
        1, 180, 320
    )
    results["sh"]["by_view_angle_345"] = by_bins(
        cache["sh_0.25_2"][None], cache["exact"][None], view_angle, BINS["view_angle"]
    )
    log("sh done")

    # Voxel leaking: which stored term of the 1 m 'lit' cache causes its error
    leak = {}
    for cached in ("lit", "visibility", "transmittance"):
        image = render(scene, light, LAST, grid_source(light, 1.0, cached))
        leak[cached] = {
            "vs_exact_marching": by_region(image[None], exact[[LAST]], masks_last),
            "vs_single": by_region(image[None], single[[LAST]], masks_last),
        }
        if cached == "lit":
            leak["lit_vs_saved_lit_1.0"] = errors(image[None], cache["lit_1.0_0"][None])
    results["leak_1m"] = leak
    log("leak done")

    # Froxels: their size at the lamp, against how far the rays of the halo pass from the lamp
    miss = float(to_lamp.norm()) * np.sin(np.radians(view_angle[0]))[masks["halo"][LAST]]
    results["froxel_size_at_lamp_345"] = {
        **{f"prototype {t}px x {s}": froxel_size(scene, LAST, t, s) for t, s in ((8, 64), (4, 128), (2, 128))},
        "EEVEE default 8px x 64 to 100 m": froxel_size(scene, LAST, 8, 64, 100.0),
        "EEVEE fine 2px x 256 to 300 m": froxel_size(scene, LAST, 2, 256, 300.0),
        "halo_rays_distance_from_lamp_m": {
            "p10": float(np.percentile(miss, 10)),
            "median": float(np.median(miss)),
            "p90": float(np.percentile(miss, 90)),
        },
        "halo_light_weighted_distance_to_lamp_m": float(
            np.nanmedian(path["distance_to_lamp"][LAST][masks["halo"][LAST]])
        ),
    }

    (SAVE / "night.json").write_text(json.dumps(results, indent=1))
    log("saved json")

    # Figures, frame 345
    panels = [(name, images[LAST], single[LAST]) for name, images in methods.items()]
    panels += [
        ("Unreal epic vs MULTIPLE scattering", unreal[LAST], multiple[LAST]),
        ("EEVEE fine vs MULTIPLE scattering", methods["EEVEE fine (2px x 256)"][LAST], multiple[LAST]),
    ]
    error_maps(panels, (single[LAST], multiple[LAST]), masks, SAVE / "error_maps.png")
    rows = [(name, results["methods"][name]["frame_345"]["vs_single"]) for name in methods]
    rows += [
        ("EEVEE fine vs multiple", results["methods"]["EEVEE fine (2px x 256)"]["frame_345"]["vs_multiple"]),
        ("Unreal epic vs multiple", results["methods"]["Unreal epic (8px x 128)"]["frame_345"]["vs_multiple"]),
    ]
    region_bars(rows, SAVE / "regions.png")
    causes_figure(results, SAVE / "causes.png")
    log("figures saved")


if __name__ == "__main__":
    main()
