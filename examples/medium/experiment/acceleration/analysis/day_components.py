"""Day scene: where each method's error comes from, split by the part of the light it is in and by image region.

Cycles' reference (4096 spp, 16 volume bounces) splits each image into three parts that add up to it:

- surfaces: the light of surfaces seen through the fog, i.e. the combined image minus both volume passes;
- single scattering: light of the sun and the sky scattered once by the fog towards the camera (volume direct);
- multiple scattering: all other light the fog scatters towards the camera, i.e. light scattered more than once, or
  reflected by the ground or objects before the fog scatters it (volume indirect).

Each method is split the same way when it can be:

- closed form: surfaces are its radiance minus its in-scattering, single scattering its in-scattering without multiple
  scattering nor light reflected by the ground, and multiple scattering the rest of its in-scattering, itself split
  into sunlight and skylight reflected by the ground then scattered once, and the rest (tabulated multiple scattering);
- Cycles at 4 to 1024 spp: its noisy image and noisy volume passes, as the reference's (and its denoised image whole);
- EEVEE (default, and fine: 2 px tiles, 256 slices up to 300 m): surfaces are its image minus its volume light pass,
  which holds both kinds of scattering at once;
- Unreal (epic): its fog's light is its image minus the exact transmittance times its image without fog, which are
  its surfaces;
- froxels, and world-space caches of the sun's visibility or of the sky's light as spherical harmonics (frame 345
  only): their fog's light, with the closed form's surfaces.

The error of a part is ``sum |ours - reference|`` over that part, as a share of the reference's whole image (summed
over pixels, channels and frames), so that the shares of a method show which part its error is in; they add up to
more than the error of the whole image when errors of different parts cancel. Its bias is ``sum (ours - reference)``,
as a share of the same. Errors are given as is and once both images are blurred by a 5 x 5 box, which leaves out most
of Monte Carlo noise (of the method and of the reference) and keeps bias.

Errors are also split by image region: sky, distance of the surface seen, ground in the open, ground within 2 m of an
object, faces of objects, surfaces in the sun's shadow, rays whose single scattering objects' shadows change, depth
edges, and the direction of rays relative to the sun (elevation, azimuth from the sun's, angle to the sun). The
closed form's multiple scattering is also computed without the objects' occlusion, and on rays that look down at the
ground it is compared with the ground's light that the fog takes out of the ray, ``(1 - T) L``, which a fog that
scatters mostly forward (g = 0.8) sends on towards the camera. EEVEE's fog stops at 100 m, which is tested by cutting
the closed form's fog there.

Saves ``day_components.json``, a bar chart of each method's error shares and biases, maps of where errors are in
frame 345, the closed form's multiple scattering error against the direction to the sun, and the region masks, under
``runs/fog-comparison/acceleration/analysis/day_components``.
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
from scipy.spatial import cKDTree

FRAMES = (255, 345)
SAVE = OUT / "analysis" / "day_components"
PARTS = ("surfaces", "single", "multiple")
ALL_PARTS = ("whole", "surfaces", "single", "multiple", "fog")
CUT = 100.0  # where EEVEE's volumetrics end by default, in view-space depth (m)
NEAR_OBJECT = 2.0  # horizontal distance from the ground to an object below which the ground is "next to" it (m)
ELEVATIONS = (-90, -20, -8, -2, 5, 15, 90)  # bands of the elevation of rays (degrees)
AZIMUTHS = (0, 35, 50, 65, 80, 180)  # bands of the azimuth of rays from the sun's (degrees)
ANGLES = (0, 40, 55, 65, 75, 85, 95, 180)  # bands of the angle between rays and the direction to the sun (degrees)


# --------------------------------------------------------------------------------------------------------------------
# Images of each part


def complete(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Adds the fog's light (single and multiple scattering) and the whole image, when the parts allow it."""
    parts = dict(parts)
    if "single" in parts and "fog" not in parts:
        parts["fog"] = parts["single"] + parts["multiple"]
    if "whole" not in parts:
        parts["whole"] = parts["surfaces"] + parts["fog"]
    return parts


def reference_parts() -> dict[str, np.ndarray]:
    folder = ROOT / "ms" / "cycles_ref"
    whole = load_images(folder / "frames", FRAMES).astype(np.float64)
    single = load_images(folder / "volume" / "direct", FRAMES).astype(np.float64)
    multiple = load_images(folder / "volume" / "indirect", FRAMES).astype(np.float64)
    return complete({"surfaces": whole - single - multiple, "single": single, "multiple": multiple, "whole": whole})


def closed_form_parts(scene: Scene, occlusion: bool = True) -> dict[str, np.ndarray]:
    """The closed form's parts, and its multiple scattering split into light reflected by the ground then scattered
    once (``ground``) and the rest (``tabulated``), along with its transmittance."""
    single_medium = scene.medium.model_copy(update={"multiple_scattering": False})
    no_ground = scene.lighting.model_copy(update={"ground_albedo": (0.0, 0.0, 0.0)})
    options = {} if occlusion else {"occlusion": None}
    parts = {k: [] for k in ("surfaces", "single", "multiple", "ground", "tabulated", "transmittance")}
    for i in range(len(scene.frames)):
        full = scene.closed_form(i, **options)
        single = scene.closed_form(i, medium=single_medium, lighting=no_ground, **options).inscatter.numpy()
        once = scene.closed_form(i, medium=single_medium, **options).inscatter.numpy()
        radiance, inscatter = full.radiance.numpy(), full.inscatter.numpy()
        parts["surfaces"].append(radiance - inscatter)
        parts["single"].append(single)
        parts["multiple"].append(inscatter - single)
        parts["ground"].append(once - single)
        parts["tabulated"].append(inscatter - once)
        parts["transmittance"].append(full.transmittance.numpy())
    return complete({k: np.stack(v).astype(np.float64) for k, v in parts.items()})


def cycles_parts(name: str) -> dict[str, np.ndarray]:
    folder = OUT / name
    noisy = load_images(folder / "noisy", FRAMES).astype(np.float64)
    single = load_images(folder / "volume" / "direct", FRAMES).astype(np.float64)
    multiple = load_images(folder / "volume" / "indirect", FRAMES).astype(np.float64)
    return complete({"surfaces": noisy - single - multiple, "single": single, "multiple": multiple, "whole": noisy})


def eevee_parts(name: str) -> dict[str, np.ndarray]:
    whole = load_images(OUT / name / "frames", FRAMES).astype(np.float64)
    fog = load_images(OUT / name / "volume" / "light", FRAMES).astype(np.float64)
    return complete({"surfaces": whole - fog, "fog": fog, "whole": whole})


def read_unreal(preset: str) -> np.ndarray:
    images = []
    for frame in FRAMES:
        with OpenEXR.File(str(OUT / "unreal" / "renders" / f"Day_{preset}" / f"{frame:04d}.exr")) as f:
            images.append(np.asarray(f.channels()["RGBA"].pixels, dtype=np.float64)[..., :3])
    return np.stack(images)


def truncated_fog(scene: Scene) -> dict[str, np.ndarray]:
    """What the closed form loses when its fog stops at a view-space depth of CUT, as EEVEE's does: the change of its
    fog's light, and of the light of surfaces (and the sky) behind the fog, which the fog dims less."""
    fog, surfaces = [], []
    for i in range(len(scene.frames)):
        t = scene.transforms[i]
        kwargs = {"occlusion": scene.occlusion, "dtype": DTYPE, "device": DEVICE}
        args = (t, t["transform_matrix"], scene.medium, scene.lighting)
        full = apply_medium(scene.clear[i], scene.depth[i], *args, **kwargs)
        cut = apply_medium(scene.clear[i], np.minimum(scene.depth[i], CUT), *args, **kwargs)
        fog.append(cut.inscatter.numpy() - full.inscatter.numpy())
        surfaces.append((cut.transmittance.numpy() - full.transmittance.numpy()) * scene.clear[i])
    return {"fog": np.stack(fog).astype(np.float64), "surfaces": np.stack(surfaces).astype(np.float64)}


# --------------------------------------------------------------------------------------------------------------------
# Errors


class Difference:
    """Difference between a part of a method's images and the reference's, as is and blurred (blurring is linear)."""

    def __init__(self, ours: np.ndarray, reference: np.ndarray):
        self.reference = reference
        self.diff = ours - reference
        self.blurred = blur(self.diff)

    def stats(self, norm: float, keep: np.ndarray | None = None) -> dict[str, float]:
        """Error and bias as shares of ``norm`` (the reference's whole image), and relative to the reference's part
        within the region kept."""
        mask = (
            np.ones(self.diff.shape, dtype=bool) if keep is None else np.broadcast_to(keep[..., None], self.diff.shape)
        )
        local = max(float(self.reference[mask].sum()), 1e-30)
        error = float(np.abs(self.diff[mask]).sum())
        blurred = float(np.abs(self.blurred[mask]).sum())
        bias = float(self.diff[mask].sum())
        return {
            "share": error / norm,
            "share_blurred": blurred / norm,
            "bias": bias / norm,
            "local": error / local,
            "local_blurred": blurred / local,
            "local_bias": bias / local,
        }


def method_stats(ours: dict, reference: dict, frames: list[int]) -> dict:
    """Errors of each part a method has, on the frames given (indices into FRAMES)."""
    norm = float(reference["whole"][frames].sum())
    return {part: Difference(ours[part], reference[part][frames]).stats(norm) for part in ALL_PARTS if part in ours}


def region_stats(ours: dict, reference: dict, partitions: dict, parts=ALL_PARTS) -> dict:
    norm = float(reference["whole"].sum())
    differences = {p: Difference(ours[p], reference[p]) for p in parts if p in ours}
    result = {}
    for partition, masks in partitions.items():
        result[partition] = {}
        for name, mask in masks.items():
            entry = {
                "pixels": float(mask.mean()),
                "reference_share": {p: float(reference[p][mask].sum()) / norm for p in parts if p in reference},
            }
            entry |= {p: d.stats(norm, mask) for p, d in differences.items()}
            result[partition][name] = entry
    return result


# --------------------------------------------------------------------------------------------------------------------
# Regions


def geometry(scene: Scene) -> dict[str, np.ndarray]:
    """Per pixel of each frame: whether it sees the sky, distance to the surface, its world position and the
    direction of its ray, of shapes (n, h, w) and (n, h, w, 3)."""
    sky = load_images(ROOT / "clear" / "depths", FRAMES)[..., 0] >= 1e9
    n, h, w = sky.shape
    distance, points, directions = [], [], []
    for i in range(n):
        origin, d, length = (x.numpy() for x in scene.rays(i, torch.float64))
        distance.append(length.reshape(h, w))
        points.append((origin + d * length[:, None]).reshape(h, w, 3))
        directions.append(d.reshape(h, w, 3))
    return {"sky": sky, "distance": np.stack(distance), "points": np.stack(points), "directions": np.stack(directions)}


def sun_visible(scene: Scene, geo: dict) -> np.ndarray:
    """Visible fraction of the sun from the surface each pixel sees (1 for the sky), through the shadow maps."""
    light = DayLight(scene)
    normals = scene.extra["normals"].astype(np.float64)
    visible = np.ones(geo["sky"].shape)
    for i in range(len(FRAMES)):
        solid = ~geo["sky"][i]
        points = geo["points"][i][solid] + 0.02 * normals[i][solid]  # just off the surface, out of its own shadow
        visible[i][solid] = light.sun_visibility(torch.as_tensor(points, **KW)).numpy()
    return visible


def angles(geo: dict, scene: Scene) -> dict[str, np.ndarray]:
    """Elevation of each ray, its azimuth from the sun's and its angle to the direction to the sun, in degrees."""
    sun = np.asarray(scene.lighting.suns[0].direction, dtype=np.float64)
    sun /= np.linalg.norm(sun)
    d = geo["directions"]
    azimuth = np.degrees(np.arctan2(d[..., 1], d[..., 0]) - np.arctan2(sun[1], sun[0]))
    return {
        "elevation": np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1))),
        "azimuth": np.abs((azimuth + 180) % 360 - 180),
        "to_sun": np.degrees(np.arccos(np.clip(d @ sun, -1, 1))),
    }


def bands(values: np.ndarray, edges, unit: str = "deg") -> dict[str, np.ndarray]:
    label = lambda lo, hi: f">{lo}{unit}" if hi >= 1e9 else f"{lo}..{hi}{unit}"
    return {label(lo, hi): (values >= lo) & (values < hi) for lo, hi in zip(edges[:-1], edges[1:])}


def regions(
    scene: Scene, geo: dict, visible: np.ndarray, shafts: np.ndarray, angle: dict
) -> dict[str, dict[str, np.ndarray]]:
    """Partitions of the pixels of each frame into regions, as boolean masks of shape (n, h, w)."""
    sky, distance, points = geo["sky"], geo["distance"], geo["points"]
    n, h, w = sky.shape
    normal_z = scene.extra["normals"][..., 2]
    ground = ~sky & (normal_z > 0.95) & (np.abs(points[..., 2] - scene.lighting.ground_height) < 0.1)
    objects = ~sky & ~ground

    # Ground next to objects: within NEAR_OBJECT meters, horizontally, of any visible point of an object
    tree = cKDTree(points[objects][:, :2])
    gap = np.full(sky.shape, np.inf)
    gap[ground] = tree.query(points[ground][:, :2], distance_upper_bound=10.0)[0]
    next_to = ground & (gap < NEAR_OBJECT)

    # Depth edges: where the distance changes by more than 10% from a neighbor, widened by 2 pixels
    edges = np.zeros(sky.shape, dtype=bool)
    for i in range(n):
        d = np.where(sky[i], 1e4, distance[i])
        jump = np.zeros((h, w), dtype=bool)
        vertical = np.abs(np.diff(d, axis=0)) / np.minimum(d[:-1], d[1:]) > 0.1
        horizontal = np.abs(np.diff(d, axis=1)) / np.minimum(d[:, :-1], d[:, 1:]) > 0.1
        jump[:-1] |= vertical
        jump[1:] |= vertical
        jump[:, :-1] |= horizontal
        jump[:, 1:] |= horizontal
        edges[i] = ndimage.binary_dilation(jump, iterations=2)

    solid = ~sky
    return {
        "distance": {"sky": sky} | {k: solid & m for k, m in bands(distance, (0, 5, 15, 50, 1e9), "m").items()},
        "surface": {
            "sky": sky,
            "ground_open": ground & ~next_to,
            "ground_next_to_objects": next_to,
            "object_faces": objects,
        },
        "sun_shadow": {
            "sky": sky,
            "surface_sunlit": solid & (visible >= 0.5),
            "surface_in_sun_shadow": solid & (visible < 0.5),
        },
        "shafts": {"single_scattering_shadowed": shafts, "not": ~shafts},
        "edges": {"depth_edges": edges, "interior": ~edges},
        "elevation": bands(angle["elevation"], ELEVATIONS),
        "angle_to_sun": bands(angle["to_sun"], ANGLES),
        "looking_down": {
            "sunlit_ground_below_-8deg": ground & (visible > 0.99) & (angle["elevation"] < -8),
            "shadowed_ground_below_-8deg": ground & (visible < 0.01) & (angle["elevation"] < -8),
            "rest": ~(ground & ((visible > 0.99) | (visible < 0.01)) & (angle["elevation"] < -8)),
        },
    }


def direction_table(ours: np.ndarray, reference: np.ndarray, norm: float, angle: dict, keep: np.ndarray) -> list[list]:
    """Bias of a part, blurred, in cells of the elevation of rays and their azimuth from the sun's: rows of
    [elevation band, azimuth band, pixels, bias relative to the reference's part there, bias as a share of the
    whole image], for cells of at least 200 pixels."""
    blurred = blur(ours - reference)
    rows = []
    for e_name, e_mask in bands(angle["elevation"], ELEVATIONS).items():
        for a_name, a_mask in bands(angle["azimuth"], AZIMUTHS).items():
            mask = (e_mask & a_mask & keep)[..., None]
            if mask.sum() >= 200:
                local = float((blurred * mask).sum() / (reference * mask).sum())
                rows.append([e_name, a_name, int(mask.sum()), local, float((blurred * mask).sum() / norm)])
    return rows


def ground_glow(closed: dict, reference: dict, masks: dict) -> dict:
    """On rays that look down at the ground, the fog takes ``(1 - T) L`` of the ground's light L out of each ray, and,
    as it scatters mostly forward, sends most of it on towards the camera: the multiple scattering of the reference
    and of the closed form (its ground and tabulated terms) relative to it, eroded by 3 pixels away from shadow edges."""
    transmittance = closed["transmittance"]
    ground_light = reference["surfaces"] / np.maximum(transmittance, 1e-6)
    taken = (1 - transmittance) * ground_light
    result = {}
    for name, mask in masks.items():
        if name == "rest":
            continue
        mask = np.stack([ndimage.binary_erosion(m, iterations=3) for m in mask])[..., None]
        total = float((taken * mask).sum())
        result[name] = {
            "pixels": int(mask.sum()),
            "mean_one_minus_T": float(((1 - transmittance) * mask).sum() / (mask.sum() * transmittance.shape[-1])),
            "reference_multiple": float((reference["multiple"] * mask).sum()) / total,
            "closed_form_multiple": float((closed["multiple"] * mask).sum()) / total,
            "closed_form_ground_term": float((closed["ground"] * mask).sum()) / total,
            "closed_form_tabulated_term": float((closed["tabulated"] * mask).sum()) / total,
        }
    return result


# --------------------------------------------------------------------------------------------------------------------
# Figures

COLORS = {"whole": "#52514e", "surfaces": "#2a78d6", "single": "#eb6834", "multiple": "#1baf7a", "fog": "#eda100"}
LABELS = {
    "whole": "whole image",
    "surfaces": "surfaces",
    "single": "single scattering",
    "multiple": "multiple scattering",
    "fog": "fog light (single + multiple)",
}


def bar_chart(rows: list[tuple[str, dict]], path: Path):
    """Error share (blurred bars, with the unblurred error as a thin line beyond them) and bias of each part."""
    fig, axes = plt.subplots(1, 2, figsize=(14, len(rows) * 1.0 + 1.5), sharey=True)
    height = 0.15
    for r, (label, stats) in enumerate(rows):
        present = [p for p in ALL_PARTS if p in stats]
        for k, part in enumerate(present):
            y = r + (k - (len(present) - 1) / 2) * (height + 0.02)
            s = stats[part]
            hatch = "///" if part == "fog" else None
            axes[0].barh(
                y, s["share_blurred"], height, color=COLORS[part], hatch=hatch, edgecolor="white", linewidth=0.5
            )
            axes[0].plot([s["share_blurred"], s["share"]], [y, y], color=COLORS[part], linewidth=1.5, alpha=0.6)
            axes[0].text(
                s["share"] + 0.002, y, f"{100 * s['share_blurred']:.1f}", va="center", fontsize=7, color="#52514e"
            )
            axes[1].barh(y, s["bias"], height, color=COLORS[part], hatch=hatch, edgecolor="white", linewidth=0.5)
            right = s["bias"] >= 0
            axes[1].text(
                s["bias"] + (0.002 if right else -0.002),
                y,
                f"{100 * s['bias']:+.1f}",
                va="center",
                ha="left" if right else "right",
                fontsize=7,
                color="#52514e",
            )
    axes[0].set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(
        "error, share of the reference's whole image\n(bar: after 5x5 blur, i.e. bias; thin line: unblurred, adds noise)"
    )
    axes[1].set_xlabel("signed bias, share of the reference's whole image")
    axes[1].axvline(0, color="#52514e", linewidth=0.8)
    for ax in axes:
        ax.grid(axis="x", color="#e0e0dc", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05 if ax is axes[1] else 0.02))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles = [
        matplotlib.patches.Patch(color=COLORS[p], label=LABELS[p], hatch="///" if p == "fog" else None)
        for p in ALL_PARTS
    ]
    fig.suptitle(
        "Day scene at 320x180 against Cycles 4096 spp: frames 255 + 345 (rows marked 345: frame 345 only)",
        y=0.995,
        fontsize=11,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_maps(methods: dict, reference: dict, path: Path, frame: int = 1):
    """Signed error of each part, blurred, over the mean of the reference's whole image, on one frame."""
    columns = ("surfaces", "single", "multiple", "fog")
    mean = float(reference["whole"][frame].mean())
    rows = [("reference / mean", None)] + list(methods.items())
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(3.3 * len(columns), 2.05 * len(rows)))
    for r, (name, parts) in enumerate(rows):
        for c, part in enumerate(columns):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(LABELS[part], fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
            if parts is None:
                image = ax.imshow(reference[part][frame].mean(-1) / mean, cmap="Greys_r", vmin=0, vmax=2)
            elif part in parts:
                index = 0 if len(parts[part]) == 1 else frame
                error = blur(parts[part][index : index + 1] - reference[part][frame : frame + 1])[0].mean(-1) / mean
                image = ax.imshow(error, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
            else:
                ax.text(0.5, 0.5, "not separated", ha="center", va="center", transform=ax.transAxes, color="#8a8984")
                ax.set_facecolor("#f4f4f2")
                continue
            if c == len(columns) - 1 and r in (0, 1):
                fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(
        "Frame 345: reference parts (top, / mean of the whole image) and each method's signed error, 5x5 blurred, "
        "/ mean of the whole image (red: too bright)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def direction_figure(table: list[list], by_angle: dict[str, dict], path: Path):
    """Closed form's multiple scattering bias against the elevation of rays and their azimuth from the sun's, and the
    fog light's bias of several methods against the angle between rays and the direction to the sun."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    e_names = list(bands(np.zeros(1), ELEVATIONS))
    a_names = list(bands(np.zeros(1), AZIMUTHS))
    grid = np.full((len(e_names), len(a_names)), np.nan)
    for e, a, _, local, _ in table:
        grid[e_names.index(e), a_names.index(a)] = local
    image = axes[0].imshow(grid[::-1], cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for (r, c), value in np.ndenumerate(grid[::-1]):
        if np.isfinite(value):
            axes[0].text(c, r, f"{100 * value:+.0f}%", ha="center", va="center", fontsize=8)
    axes[0].set_xticks(range(len(a_names)), a_names, fontsize=8)
    axes[0].set_yticks(range(len(e_names)), e_names[::-1], fontsize=8)
    axes[0].set_xlabel("azimuth of the ray from the sun's (sun's azimuth: 30 deg, camera heading ~90 deg)")
    axes[0].set_ylabel("elevation of the ray")
    axes[0].set_title("Closed form: multiple scattering bias / reference's, 5x5 blurred", fontsize=9)
    fig.colorbar(image, ax=axes[0], fraction=0.04)
    styles = {
        "Closed form, multiple": ("#1baf7a", "-"),
        "Closed form, fog light": ("#eda100", "-"),
        "EEVEE default, fog light": ("#4a3aa7", "--"),
        "Unreal epic, fog light": ("#e34948", "--"),
    }
    for name, values in by_angle.items():
        centers = [(lo + min(hi, 120)) / 2 for lo, hi in zip(ANGLES[:-1], ANGLES[1:])]
        color, style = styles[name]
        axes[1].plot(
            centers,
            [100 * values[k] for k in values],
            style,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            label=name,
        )
    axes[1].axhline(0, color="#52514e", linewidth=0.8)
    axes[1].set_xlabel("angle between the ray and the direction to the sun (deg; last point: 95-180)")
    axes[1].set_ylabel("bias / reference's, in that band (%)")
    axes[1].grid(color="#e0e0dc", linewidth=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mask_figure(partitions: dict, path: Path, frame: int = 1):
    shown = {k: partitions[k] for k in ("distance", "surface", "sun_shadow", "shafts", "edges", "looking_down")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
    palette = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for ax, (partition, masks) in zip(axes.flat, shown.items()):
        labels = np.zeros(next(iter(masks.values()))[frame].shape, dtype=int)
        for k, mask in enumerate(masks.values()):
            labels[mask[frame]] = k
        colors = palette[: len(masks)]
        cmap = matplotlib.colors.ListedColormap(colors)
        ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest")
        ax.set_title(partition, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        handles = [matplotlib.patches.Patch(color=c, label=n) for c, n in zip(colors, masks)]
        ax.legend(handles=handles, fontsize=6, loc="lower left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    scene = day_scene(FRAMES)
    reference = reference_parts()
    norm = float(reference["whole"].sum())
    both, last = [0, 1], [1]
    results: dict = {"frames": FRAMES, "reference_share": {p: float(reference[p].sum()) / norm for p in PARTS}}
    print("reference shares", results["reference_share"], flush=True)

    # Methods split into parts
    closed = closed_form_parts(scene)
    saved = np.load(OUT / "results" / "closed_form_day.npy")[[DAY_FRAMES.index(f) for f in FRAMES]]
    results["closed_form_vs_saved"] = float(np.abs(closed["whole"] - saved).sum() / np.abs(saved).sum())
    unoccluded = closed_form_parts(scene, occlusion=False)
    cycles = {spp: cycles_parts(f"day_spp{spp}") for spp in (4, 16, 64, 256, 1024)}
    eevee = {name: eevee_parts(f"eevee_day_{name}") for name in ("default", "fine")}
    transmittance = np.stack([scene.transmittance(i) for i in range(len(FRAMES))]).astype(np.float64)
    unreal_whole, unreal_clear = read_unreal("epic"), read_unreal("nofog")
    unreal = complete(
        {
            "surfaces": transmittance * unreal_clear,
            "fog": unreal_whole - transmittance * unreal_clear,
            "whole": unreal_whole,
        }
    )
    with np.load(OUT / "results" / "froxel_images.npz") as data:
        froxel_fog = data["day_8_64_1"][None].astype(np.float64)
    with np.load(OUT / "results" / "cache_day_images.npz") as data:
        cache_fog = {k: data[k][None].astype(np.float64) for k in ("exact", "sh_1.0_2", "sun_1.0")}
    shared = closed["surfaces"][last]
    last_only = {
        "froxel_8_64 (345)": complete({"surfaces": shared, "fog": froxel_fog}),
        "cache_sh_1.0_2 (345)": complete({"surfaces": shared, "fog": cache_fog["sh_1.0_2"]}),
        "cache_sun_1.0 (345)": complete({"surfaces": shared, "fog": cache_fog["sun_1.0"]}),
    }

    # (a) Errors of each part
    stats = {
        "closed_form": method_stats(closed, reference, both),
        "closed_form_unoccluded": method_stats(unoccluded, reference, both),
        "eevee_default": method_stats(eevee["default"], reference, both),
        "eevee_fine": method_stats(eevee["fine"], reference, both),
        "unreal_epic": method_stats(unreal, reference, both),
        "closed_form (345)": method_stats({k: v[last] for k, v in closed.items()}, reference, last),
    }
    for spp, parts in cycles.items():
        stats[f"cycles_{spp}spp_noisy"] = method_stats(parts, reference, both)
    denoised = load_images(OUT / "day_spp16" / "frames", FRAMES).astype(np.float64)
    stats["cycles_16spp_denoised"] = method_stats({"whole": denoised}, reference, both)
    for name, parts in last_only.items():
        stats[name] = method_stats(parts, reference, last)
    extra = {}
    # EEVEE's fog against single scattering alone, as if it only scattered light once
    for name in eevee:
        extra[f"eevee_{name}_fog_vs_single_only"] = Difference(eevee[name]["fog"], reference["single"]).stats(norm)
    # Froxels' and caches' own error: their fog's light against the closed form's, on frame 345
    norm_last = float(reference["whole"][last].sum())
    for name, parts in last_only.items():
        extra[f"{name}_fog_vs_closed_form"] = Difference(parts["fog"], closed["fog"][last]).stats(norm_last)
    extra["cache_exact_marching (345)_fog_vs_closed_form"] = Difference(cache_fog["exact"], closed["fog"][last]).stats(
        norm_last
    )
    # Surfaces: Unreal's image without fog against Cycles' (renderer), and surfaces only dimmed by the fog, not lit
    # through it (model, which Unreal's surfaces are), against the reference's
    naive = transmittance * scene.clear.astype(np.float64)
    extra["unreal_surfaces_vs_dimmed_cycles_clear"] = Difference(unreal["surfaces"], naive).stats(norm)
    extra["dimmed_cycles_clear_vs_reference_surfaces"] = Difference(naive, reference["surfaces"]).stats(norm)
    # The closed form's multiple scattering in its two terms, as shares of the reference's whole image
    extra["closed_form_multiple_terms"] = {k: float(closed[k].sum()) / norm for k in ("ground", "tabulated")}
    results["parts"], results["extra"] = stats, extra

    # (c) Monte Carlo: if Cycles at N spp is unbiased, its error is the noise of both, which scales as
    # sqrt(1/N + 1/4096), so dividing it by sqrt(1 + 4096/N) gives the reference's own noise (error of a perfect method)
    results["reference_noise_estimate"] = {}
    for spp in cycles:
        scale = math.sqrt(1 + 4096 / spp)
        entry = stats[f"cycles_{spp}spp_noisy"]
        results["reference_noise_estimate"][f"{spp}spp"] = {
            p: {"share": entry[p]["share"] / scale, "share_blurred": entry[p]["share_blurred"] / scale}
            for p in PARTS + ("whole",)
        }

    # (b) Regions
    geo = geometry(scene)
    visible = sun_visible(scene, geo)
    angle = angles(geo, scene)
    shafts = np.abs(closed["single"] - unoccluded["single"]).sum(-1) > 0.1 * reference["whole"].sum(-1)
    task_mask = np.abs(closed["whole"] - unoccluded["whole"]).sum(-1) > 0.02 * reference["whole"].sum(-1)
    results["occlusion_changes_whole_by_2pct_pixels"] = float(task_mask.mean())
    partitions = regions(scene, geo, visible, shafts, angle)
    fog_only = {k: partitions[k] for k in ("distance", "elevation", "angle_to_sun")}
    results["regions"] = {
        "closed_form": region_stats(closed, reference, partitions),
        "closed_form_unoccluded": region_stats(unoccluded, reference, partitions),
        "eevee_default": region_stats(eevee["default"], reference, fog_only),
        "eevee_fine": region_stats(eevee["fine"], reference, fog_only),
        "unreal_epic": region_stats(unreal, reference, fog_only),
        "cycles_16spp_noisy": region_stats(cycles[16], reference, {"distance": partitions["distance"]}),
    }
    results["closed_form_multiple_by_direction"] = {
        "columns": ["elevation", "azimuth_from_sun", "pixels", "local_bias", "bias"],
        "all": direction_table(closed["multiple"], reference["multiple"], norm, angle, np.ones_like(geo["sky"])),
        "sky_only": direction_table(closed["multiple"], reference["multiple"], norm, angle, geo["sky"]),
    }
    results["ground_glow"] = ground_glow(closed, reference, partitions["looking_down"])
    # Where froxels and caches add to the closed form's error: their fog's light against the closed form's on frame 345,
    # by elevation of rays, as shares of the reference's whole image and relative to the closed form's fog light there
    results["own_error_by_elevation_345"] = {}
    for name, parts in last_only.items():
        difference = Difference(parts["fog"], closed["fog"][last])
        results["own_error_by_elevation_345"][name] = {
            band: difference.stats(norm_last, mask[last]) for band, mask in partitions["elevation"].items()
        }
    # EEVEE's fog stops at 100 m by default: the closed form cut there loses this much, as shares of the whole image
    cut = truncated_fog(scene)
    results["cut_at_100m_change"] = {
        region: {p: float(cut[p][mask].sum()) / norm for p in cut} for region, mask in partitions["distance"].items()
    }
    (SAVE / "day_components.json").write_text(json.dumps(results, indent=1))

    # Figures
    pick = lambda s: {k: v for k, v in s.items() if k in COLORS}
    rows = [
        ("Closed form", stats["closed_form"]),
        ("Cycles 16 spp (noisy)", stats["cycles_16spp_noisy"]),
        ("Cycles 16 spp (denoised)", stats["cycles_16spp_denoised"]),
        ("Cycles 1024 spp (noisy)", stats["cycles_1024spp_noisy"]),
        ("EEVEE default (8 px, 64 slices)", pick(stats["eevee_default"])),
        ("EEVEE fine (2 px, 256 slices)", pick(stats["eevee_fine"])),
        ("Unreal epic", pick(stats["unreal_epic"])),
        ("Closed form (345)", stats["closed_form (345)"]),
        ("Froxels 8 px / 64 (345)", pick(stats["froxel_8_64 (345)"])),
        ("Cache SH order 2, 1 m (345)", pick(stats["cache_sh_1.0_2 (345)"])),
        ("Cache sun 1 m (345)", pick(stats["cache_sun_1.0 (345)"])),
    ]
    bar_chart(rows, SAVE / "component_errors.png")
    error_maps(
        {
            "Closed form": closed,
            "Cycles 16 spp noisy": cycles[16],
            "EEVEE default": eevee["default"],
            "Unreal epic": unreal,
            "Froxels 8/64": last_only["froxel_8_64 (345)"],
            "Cache SH 1 m": last_only["cache_sh_1.0_2 (345)"],
        },
        reference,
        SAVE / "error_maps_345.png",
    )
    regions_of = results["regions"]
    by_angle = {
        "Closed form, multiple": {
            k: v["multiple"]["local_bias"] for k, v in regions_of["closed_form"]["angle_to_sun"].items()
        },
        "Closed form, fog light": {
            k: v["fog"]["local_bias"] for k, v in regions_of["closed_form"]["angle_to_sun"].items()
        },
        "EEVEE default, fog light": {
            k: v["fog"]["local_bias"] for k, v in regions_of["eevee_default"]["angle_to_sun"].items()
        },
        "Unreal epic, fog light": {
            k: v["fog"]["local_bias"] for k, v in regions_of["unreal_epic"]["angle_to_sun"].items()
        },
    }
    direction_figure(results["closed_form_multiple_by_direction"]["all"], by_angle, SAVE / "direction_to_sun.png")
    mask_figure(partitions, SAVE / "regions_345.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
