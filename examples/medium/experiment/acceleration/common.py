"""Shared pieces of the acceleration study: scenes, references, error metrics, and the closed form's lighting
evaluated at any point, which world-space caches, froxels and neural caches store and integrate.

Two scenes of ``runs/fog-comparison`` are used, at 320x180:

- day: the ground fog lit by a sun and the sky, with objects casting shadows onto it, and light scattered many times,
  against Cycles with 4096 spp and 16 volume bounces (``ms/cycles_ref``), whole images;
- night: the same fog lit by a point light right behind a cube, which casts a shadow into the fog in front of it,
  against Cycles with 4096 spp and single scattering, light scattered by the fog only (its volume direct pass).

Everything runs on the CPU, with 12 threads as Blender's renders, unless the environment variable ``ACCEL_DEVICE``
names another device, e.g. ``cuda``.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium, camera_rays, load_occlusion
from visionsim.medium.lights import emitters
from visionsim.medium.model import HeightFog
from visionsim.medium.occlusion import (
    Occlusion,
    _arriving,
    _band_weights,
    _blocker_distances,
    _distance_at,
    _ratio,
    band_attenuation,
    band_radiance,
    lamp_map_index,
    lamp_visibility,
    maps_to,
    occlusion_to,
    select_maps,
    visibility,
)
from visionsim.medium.optics import height_fog_optical_depth, henyey_greenstein
from visionsim.medium.optics import optical_depth as relative_optical_depth
from visionsim.medium.scattering import cached_multiple_scattering, elevation_sources

ROOT = Path("runs/fog-comparison")
OUT = ROOT / "acceleration"
MEDIA = Path("examples/medium/media")
BOX_MIN, BOX_MAX = np.array([-200.0, -200.0, -1.0]), np.array([200.0, 200.0, 25.0])
GROUND_ALBEDO = 0.39
MATERIALS = {"surface_roughness": 0.8, "surface_ior": 1.5}
"""The scenes' surfaces: every Principled BSDF of fog_ref.blend and fog_point_shadow.blend has roughness 0.8 and IOR 1.5
(specular level 0.5), given to libraries whose Lighting models a specular layer"""
DAY_FRAMES = (255, 256, 257, 258, 259, 260, 261, 262, 345)
NIGHT_FRAMES = (255, 345)
THREADS = 12
DTYPE = torch.float32
DEVICE = os.environ.get("ACCEL_DEVICE", "cpu")
KW = {"dtype": DTYPE, "device": DEVICE}
torch.set_num_threads(THREADS)


def synchronize() -> None:
    """Wait for the device to finish, before reading the time."""
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------------------------------
# Scenes and references


def frame_indices(path: Path) -> dict[int, int]:
    """Index in a dataset of each frame number, read from the names of its files."""
    dataset = Dataset.from_path(path)
    return {int(re.search(r"(\d+)\.\w+$", str(dataset[i][1]["file_path"]))[1]): i for i in range(len(dataset))}


def load_images(path: Path, frames: Sequence[int]) -> np.ndarray:
    """Frames of a dataset as linear RGB images, of shape (n, h, w, 3)."""
    dataset, index = Dataset.from_path(path), frame_indices(path)
    images = []
    for frame in frames:
        data = np.asarray(dataset[index[frame]][0], dtype=np.float32)
        images.append(np.repeat(data, 3, axis=-1) if data.shape[-1] == 1 else data[..., :3])
    return np.stack(images)


def load_transforms(path: Path, frames: Sequence[int]) -> list[dict]:
    dataset, index = Dataset.from_path(path), frame_indices(path)
    return [dict(dataset[index[frame]][1]) for frame in frames]


def box_depth(depth: np.ndarray, transform: dict) -> np.ndarray:
    """Depth where rays that see the sky stop when leaving the fog volume, as in Cycles."""
    origin, directions, scale = (x.numpy() for x in camera_rays(transform, transform["transform_matrix"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        bounds = np.where(directions > 0, BOX_MAX, BOX_MIN)
        exit_distance = np.where(directions != 0, (bounds - origin) / directions, np.inf).min(axis=-1)
    return np.where(depth >= 1e9, exit_distance / scale, depth).astype(np.float32)


@dataclass
class Scene:
    name: str
    medium: Medium
    lighting: Lighting
    occlusion: Occlusion
    frames: tuple[int, ...]
    clear: np.ndarray
    """frames rendered without fog, of shape (n, h, w, 3)"""
    depth: np.ndarray
    """depth of each pixel, up to the fog box for pixels that see the sky, of shape (n, h, w)"""
    transforms: list[dict]
    reference: np.ndarray
    """Cycles' reference, whole images for the day scene and light scattered by the fog for the night scene"""
    whole: bool
    """whether the reference is the whole image, rather than the light scattered by the fog"""
    extra: dict = field(default_factory=dict)

    def rays(self, i: int, dtype: torch.dtype = DTYPE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Origin, unit directions of shape (h*w, 3) and length of shape (h*w,) of the camera rays of frame i."""
        t = self.transforms[i]
        origin, directions, scale = camera_rays(t, t["transform_matrix"], dtype=dtype, device=DEVICE)
        distance = torch.as_tensor(self.depth[i], dtype=dtype, device=DEVICE) * scale
        return origin, directions.reshape(-1, 3), distance.reshape(-1)

    def closed_form(self, i: int, medium: Medium | None = None, lighting: Lighting | None = None, **kwargs):
        """The closed form of :func:`apply_medium`, with the scene's shadow maps and surfaces lit through the fog
        when their normals are known, on the CPU."""
        t = self.transforms[i]
        options = {"occlusion": self.occlusion, "dtype": DTYPE, "device": DEVICE} | kwargs
        if "normals" in self.extra and "normals" not in options:
            options["normals"] = self.extra["normals"][i]
        medium, lighting = medium or self.medium, lighting or self.lighting
        return apply_medium(self.clear[i], self.depth[i], t, t["transform_matrix"], medium, lighting, **options)

    def surfaces(self, i: int) -> np.ndarray:
        """Light of surfaces seen through the fog, as the closed form models it (lit through the fog and attenuated),
        which every method shares, as they only differ in how they compute the light scattered by the fog."""
        cache = self.extra.setdefault("surfaces", {})
        if i not in cache:
            result = self.closed_form(i)
            illumination = 1.0 if result.illumination is None else result.illumination.cpu().numpy()
            cache[i] = (result.transmittance.cpu().numpy() * illumination * self.clear[i]).astype(np.float32)
        return cache[i]

    def output(self, i: int, inscatter: np.ndarray) -> np.ndarray:
        """Image to compare with the reference, given the light scattered towards the camera."""
        return self.surfaces(i) + inscatter if self.whole else inscatter

    def transmittance(self, i: int) -> np.ndarray:
        origin, directions, distance = self.rays(i, torch.float64)
        fog = self.medium.components[0]
        tau = height_fog_optical_depth(fog, origin, directions, distance) * self.medium.extinction
        return torch.exp(-tau).reshape(*self.depth[i].shape, 1).float().numpy()


def with_materials(lighting: Lighting) -> Lighting:
    """The lighting with the scenes' materials (:data:`MATERIALS`), for libraries whose Lighting has these fields."""
    known = {k: v for k, v in MATERIALS.items() if k in type(lighting).model_fields}
    return lighting.model_copy(update=known) if known else lighting


def day_scene(frames: Sequence[int] = DAY_FRAMES, multiple: bool = True, medium_file: str = "ground_fog.json") -> Scene:
    lighting = with_materials(Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text()))
    medium = Medium.model_validate_json((MEDIA / medium_file).read_text())
    if multiple:
        medium = medium.model_copy(update={"multiple_scattering": True})
        lighting = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    transforms = load_transforms(ROOT / "clear" / "frames", frames)
    depths = load_images(ROOT / "clear" / "depths", frames)[..., 0]
    # These normals were saved rotated by the camera's rotation rather than its inverse, which this undoes
    normals = load_images(ROOT / "clear" / "normals", frames)
    rotations = [np.asarray(t["transform_matrix"])[:3, :3] for t in transforms]
    normals = np.stack([n @ (r / np.linalg.norm(r, axis=0)) for n, r in zip(normals, rotations)]).astype(np.float32)
    return Scene(
        name="day",
        medium=medium,
        lighting=lighting,
        occlusion=load_occlusion(ROOT / "occlusion" / "occlusion.npz", dtype=DTYPE),
        frames=tuple(frames),
        clear=load_images(ROOT / "clear" / "frames", frames),
        depth=np.stack([box_depth(d, t) for d, t in zip(depths, transforms)]),
        transforms=transforms,
        reference=load_images(ROOT / "ms" / "cycles_ref" / "frames", frames),
        whole=True,
        extra={"normals": normals},
    )


def night_scene(frames: Sequence[int] = NIGHT_FRAMES) -> Scene:
    lighting = with_materials(Lighting.model_validate_json((ROOT / "point_shadow" / "lighting.json").read_text()))
    medium = Medium.model_validate_json((MEDIA / "ground_fog.json").read_text())
    transforms = load_transforms(ROOT / "clear" / "frames", frames)
    depths = load_images(ROOT / "clear" / "depths", frames)[..., 0]
    return Scene(
        name="night",
        medium=medium,
        lighting=lighting,
        occlusion=load_occlusion(ROOT / "point_shadow" / "occlusion.npz", dtype=DTYPE),
        frames=tuple(frames),
        clear=np.zeros((len(frames), *depths.shape[1:], 3), dtype=np.float32),
        depth=np.stack([box_depth(d, t) for d, t in zip(depths, transforms)]),
        transforms=transforms,
        reference=load_images(ROOT / "point_shadow" / "bounces0" / "volume" / "direct", frames),
        whole=False,
    )


# --------------------------------------------------------------------------------------------------------------------
# Metrics


def blur(images: np.ndarray, size: int = 5) -> np.ndarray:
    """Box blur of a stack of images, of shape (n, h, w, c), which leaves out most of Monte Carlo noise."""
    kernel = np.ones(size) / size
    images = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, images)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 2, images)


def errors(ours: np.ndarray, theirs: np.ndarray, keep: np.ndarray | None = None) -> dict[str, float]:
    """Ratio of the total light of both, and relative L1 errors, as is and once blurred (bias, without noise)."""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    mask = np.ones(ours.shape, dtype=bool) if keep is None else np.broadcast_to(keep[..., None], ours.shape)
    total = theirs[mask].sum()
    return {
        "ratio": float(ours[mask].sum() / total),
        "rel_l1": float(np.abs(ours - theirs)[mask].sum() / total),
        "rel_l1_blurred": float(np.abs(blur(ours) - blur(theirs))[mask].sum() / total),
    }


LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def log_luma(frames: np.ndarray) -> np.ndarray:
    """Log intensity as seen by an event camera (v2e), which is linear below 20 digital numbers."""
    luma = np.clip(frames, 0, None) @ LUMA * 255.0
    return np.where(luma <= 20, luma * np.log(20) / 20, np.log(np.maximum(luma, 20)))


def flicker(frames: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Errors of the log intensity of consecutive frames against the reference's: as is (static) and of its change
    from one frame to the next (temporal), which drives false events in event cameras."""
    error = log_luma(frames) - log_luma(reference)
    return {"static": float(np.abs(error).mean()), "temporal": float(np.abs(np.diff(error, axis=0)).mean())}


# --------------------------------------------------------------------------------------------------------------------
# Lighting at any point, as the closed form models it


def _unit(v: Sequence[float]) -> torch.Tensor:
    v = torch.as_tensor(v, dtype=torch.float64)
    return v / v.norm()


class DayLight:
    """Light that the day scene's fog scatters towards a direction, per unit of optical depth, at any point.

    The sun's light is attenuated by the fog above the point, as in the closed form, and blocked according to its
    shadow map. Light from the sky, the ground and light scattered more than once only depends on the direction's
    elevation and on the optical depth above the point, and is tabulated against both; it is weighted by the visible
    fraction of the cells of the sky, as in :func:`shade <visionsim.medium.occlusion.shade>`.
    """

    def __init__(self, scene: Scene, elevations: int = 257, depths: int = 96):
        medium, lighting = scene.medium, scene.lighting
        self.fog = medium.components[0]
        assert isinstance(self.fog, HeightFog)
        self.beta = float(medium.extinction)
        self.g, self.albedo = medium.anisotropy, medium.albedo
        beta = torch.tensor([self.beta], dtype=torch.float64)
        self.suns = [
            (_unit(s.direction).to(**KW), medium.albedo * torch.as_tensor(s.irradiance, **KW)) for s in lighting.suns
        ]
        self.depth_ground = self.beta * self.fog.density * self.fog.falloff
        # Sources against the elevation of directions and the logarithm of the optical depth above points
        self.mu = torch.linspace(-1, 1, elevations, dtype=torch.float64)
        self.log_c = torch.linspace(math.log(1e-7), math.log(self.depth_ground * 1.05), depths, dtype=torch.float64)
        self.tables = {}
        for name, source in elevation_sources(medium, lighting, beta, 3).items():
            c = torch.exp(self.log_c)[None, :, None].expand(elevations, depths, 1).contiguous()
            self.tables[name] = source(self.mu, c).to(**KW)  # (elevations, depths, 3)

        occlusion = occlusion_to(scene.occlusion, device=DEVICE, dtype=DTYPE)
        self.occlusion = occlusion
        self.sun_maps = []
        for towards, _ in self.suns:
            alignment = occlusion.sun_directions @ towards
            self.sun_maps.append(select_maps(occlusion.sun_maps, int(alignment.argmax())) if len(alignment) else None)
        self.edges, self.counts = occlusion.sky_edges, occlusion.sky_counts
        self.towards = occlusion.sky_maps.axes[:, 2]
        self.across = self.towards[:, :2].norm(dim=-1)
        self.ground_height = lighting.ground_height
        self.scattering = None
        if "scattered" in self.tables:
            self.scattering = cached_multiple_scattering(medium, lighting, beta, 3)
            self.band_light = band_radiance(self.scattering, self.edges).sum(dim=-1).to(**KW)  # (heights, 2, b)

    def depth_above(self, points: torch.Tensor) -> torch.Tensor:
        """Optical depth of the fog above points, of shape (n,)."""
        fog = self.fog
        return self.beta * fog.density * fog.falloff * torch.exp(-(points[:, 2] - fog.base_height) / fog.falloff)

    def source(self, name: str, mu: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Tabulated source at elevations and optical depths above, both of shape (n,), of shape (n, 3)."""
        table = self.tables[name]
        fi = ((mu.clamp(-1, 1) + 1) / 2 * (len(self.mu) - 1)).clamp(0, len(self.mu) - 1 - 1e-4)
        log_c = torch.log(c.clamp_min(1e-30))
        fj = ((log_c - float(self.log_c[0])) / float(self.log_c[1] - self.log_c[0])).clamp(0, len(self.log_c) - 1 - 1e-4)
        i, j = fi.long(), fj.long()
        wi, wj = (fi - i)[:, None], (fj - j)[:, None]
        near = torch.lerp(table[i, j], table[i, j + 1], wj)
        far = torch.lerp(table[i + 1, j], table[i + 1, j + 1], wj)
        return torch.lerp(near, far, wi)

    def sun_visibility(self, points: torch.Tensor, index: int = 0) -> torch.Tensor:
        """Visible fraction of a sun from points, of shape (n,), as filtered by its shadow map."""
        maps = self.sun_maps[index]
        if maps is None:
            return torch.ones_like(points[:, 0])
        origin = torch.zeros(3, dtype=points.dtype, device=points.device)
        return visibility(maps, origin, points, torch.ones_like(points[:, :1]))[:, 0, 0]

    def sky_visibility(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Whether each cell of the sky is visible from points, above and mirrored below the horizon, (n, k) each."""
        blocker = _blocker_distances(self.occlusion.sky_maps, points, 0.5)
        reach = (points[:, 2:3] - self.ground_height).clamp_min(0) * self.across / self.towards[:, 2]
        return torch.isinf(blocker), blocker * self.across >= reach

    def sun(self, points: torch.Tensor, directions: torch.Tensor, visible: torch.Tensor | None = None) -> torch.Tensor:
        """Sunlight scattered towards directions at points, of shape (n, 3), given its visibility if known."""
        c = self.depth_above(points)
        total = torch.zeros(len(points), 3, dtype=points.dtype, device=points.device)
        for index, (towards, color) in enumerate(self.suns):
            if float(towards[2]) <= 0:
                continue
            seen = self.sun_visibility(points, index) if visible is None else visible
            phase = henyey_greenstein(directions @ towards, self.g)
            total = total + (phase * torch.exp(-c / float(towards[2])) * seen)[:, None] * color
        return total

    def ratios(
        self, points: torch.Tensor, directions: torch.Tensor, lit: torch.Tensor, lit_below: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Visible fraction of the light of each source, of shape (n, 1), as in the closed form's shading."""
        c = self.depth_above(points)
        weights = _band_weights(
            self.edges,
            self.counts,
            self.g,
            directions,
            lit[:, None].to(points.dtype),
            lit_below[:, None].to(points.dtype),
        )
        visible, total, visible_below, total_below = weights
        ratios = {}
        if "sky" in self.tables:
            attenuation = band_attenuation(self.edges, c[:, None])  # (n, 1, b)
            ratios["sky"] = _ratio((attenuation * visible).sum(-1), (attenuation[:, 0] * total).sum(-1, keepdim=True))
        if "ground" in self.tables:
            ratios["ground"] = _ratio(visible_below.sum(-1), total_below.sum(-1, keepdim=True))
        if self.scattering is not None:
            n_heights = len(self.band_light)
            heights = torch.log(self.depth_ground / c.clamp_min(1e-30))[:, None]
            fh = heights.clamp(0, self.scattering.height_scale) / self.scattering.height_scale * (n_heights - 1)
            h = fh.floor().clamp(0, n_heights - 2).long()
            (low_visible, low_total), (high_visible, high_total) = (
                _arriving(self.band_light, index, weights) for index in (h, h + 1)
            )
            weight = fh - h
            ratios["scattered"] = _ratio(
                torch.lerp(low_visible, high_visible, weight), torch.lerp(low_total, high_total, weight)
            )
        return ratios

    def smooth(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        lit: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Light from the sky, the ground and multiple scattering scattered towards directions at points, (n, 3)."""
        c = self.depth_above(points)
        lit = self.sky_visibility(points) if lit is None else lit
        ratios = self.ratios(points, directions, *lit)
        total = torch.zeros(len(points), 3, dtype=points.dtype, device=points.device)
        for name in self.tables:
            total = total + self.source(name, directions[:, 2], c) * ratios[name]
        return total

    def __call__(self, points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.sun(points, directions) + self.smooth(points, directions)


class NightLight:
    """Light of the night scene's point lamp that the fog scatters towards a direction, per unit of optical depth.

    It is the lamp's intensity times the phase function, attenuated by the fog towards the lamp and blocked according
    to its shadow map, over the squared distance, which is clamped to the lamp's radius.
    """

    def __init__(self, scene: Scene):
        medium = scene.medium
        self.medium, self.beta, self.g, self.albedo = medium, float(medium.extinction), medium.anisotropy, medium.albedo
        (self.emitter,) = [e for lamp in scene.lighting.lamps for e in emitters(lamp, 3, 1, **KW)]
        self.position = self.emitter.position
        self.maps = maps_to(scene.occlusion.lamp_maps, device=DEVICE, dtype=DTYPE)
        self.index = lamp_map_index(self.maps, self.position, scene.occlusion.lamp_tolerance)

    def lit(self, points: torch.Tensor) -> torch.Tensor:
        """Transmittance towards the lamp times its visibility, of shape (n,), which doesn't fall off with distance."""
        to_light = self.position - points
        r = to_light.norm(dim=-1).clamp_min(1e-6)
        depth = relative_optical_depth(self.medium, points, to_light / r[:, None], r) * self.beta
        return torch.exp(-depth) * lamp_visibility(self.maps, self.index, points)

    def falloff(self, points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """Phase function over the squared distance to the lamp, times its intensity, of shape (n, 3)."""
        to_light = self.position - points
        r = to_light.norm(dim=-1).clamp_min(max(self.emitter.radius, 1e-4))
        phase = henyey_greenstein((directions * to_light).sum(dim=-1) / to_light.norm(dim=-1).clamp_min(1e-12), self.g)
        return self.albedo * (phase / (r * r))[:, None] * self.emitter.intensity

    def __call__(self, points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.falloff(points, directions) * self.lit(points)[:, None]


# --------------------------------------------------------------------------------------------------------------------
# Integration along camera rays


def day_nodes(
    fog: HeightFog, beta: float, origin: torch.Tensor, directions: torch.Tensor, distance: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distances along rays of n samples evenly spread in the fraction of light scattered or absorbed so far, of shape
    (r, n), and the weight of each, of shape (r, 1): the integral of σ T J is the weighted sum of J at the samples."""
    tau = height_fog_optical_depth(fog, origin, directions, distance) * beta
    y_end = -torch.expm1(-tau)
    y = (torch.arange(n, dtype=directions.dtype, device=directions.device) + 0.5) / n * y_end[:, None]
    t = _distance_at(fog, float(origin[2]), directions[:, 2:3], -torch.log1p(-y) / beta)
    return torch.minimum(t, distance[:, None]), (y_end / n)[:, None]


def lamp_nodes(
    origin: torch.Tensor, directions: torch.Tensor, distance: torch.Tensor, center: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distances along rays of n samples evenly spread in the angle seen from a lamp (equiangular), of shape (r, n),
    and their weights, such that the integral of f along each ray is the weighted sum of f at the samples."""
    offset = center - origin
    t_c = directions @ offset
    h = (offset.dot(offset) - t_c**2).clamp_min(1e-8).sqrt()
    a, b = torch.atan(-t_c / h), torch.atan((distance - t_c) / h)
    steps = torch.arange(n, dtype=directions.dtype, device=directions.device)
    theta = a[:, None] + (b - a)[:, None] * (steps + 0.5) / n
    t = t_c[:, None] + h[:, None] * torch.tan(theta)
    weights = ((b - a) / n)[:, None] * h[:, None] / torch.cos(theta) ** 2
    return t, weights


def integrate_lamp(
    light: NightLight, origin, directions, distance, source: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], n=128
) -> torch.Tensor:
    """Light of the lamp scattered along rays, given its source J at points, with equiangular samples, (r, 3)."""
    t, w = lamp_nodes(origin, directions, distance, light.position, n)
    points = origin + directions[:, None, :] * t[..., None]
    fog = light.medium.components[0]
    sigma = light.beta * fog.density * torch.exp(-(points[..., 2] - fog.base_height) / fog.falloff)
    tau = height_fog_optical_depth(fog, origin, directions[:, None, :].expand_as(points), t) * light.beta
    weight = (sigma * torch.exp(-tau) * w).reshape(-1)
    values = source(points.reshape(-1, 3), directions[:, None, :].expand_as(points).reshape(-1, 3))
    return (values * weight[:, None]).reshape(*t.shape, 3).sum(dim=1)


def by_chunks(fn: Callable[..., torch.Tensor], count: int, size: int, *arrays: torch.Tensor) -> torch.Tensor:
    """Apply fn to consecutive chunks of rays, to bound memory."""
    return torch.cat([fn(*(a[s : s + size] for a in arrays)) for s in range(0, count, size)])


# --------------------------------------------------------------------------------------------------------------------
# Spherical harmonics


def sh_basis(directions: torch.Tensor, order: int) -> torch.Tensor:
    """Real spherical harmonics up to an order, at unit directions of shape (n, 3), of shape (n, (order + 1)²),
    ordered by degree l, then by m from -l to l."""
    x, y, z = directions.unbind(-1)
    phi = torch.atan2(y, x)
    s = (1 - z * z).clamp_min(0).sqrt()
    legendre = {}  # P_l^m(z) without the Condon-Shortley phase
    for m in range(order + 1):
        p_mm = torch.full_like(z, float(np.prod(np.arange(1, 2 * m, 2)) if m else 1.0)) * s**m
        legendre[m, m] = p_mm
        if m < order:
            legendre[m + 1, m] = z * (2 * m + 1) * p_mm
        for l in range(m + 2, order + 1):
            legendre[l, m] = ((2 * l - 1) * z * legendre[l - 1, m] - (l + m - 1) * legendre[l - 2, m]) / (l - m)
    columns = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            k = abs(m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - k) / math.factorial(l + k))
            if m == 0:
                columns.append(norm * legendre[l, 0])
            elif m > 0:
                columns.append(math.sqrt(2) * norm * torch.cos(k * phi) * legendre[l, k])
            else:
                columns.append(math.sqrt(2) * norm * torch.sin(k * phi) * legendre[l, k])
    return torch.stack(columns, dim=-1)


def sphere_quadrature(n_theta: int, n_phi: int, dtype: torch.dtype = DTYPE) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes in cos θ times uniform azimuths: unit directions (M, 3) and weights (M,) summing to 4π."""
    t, w = np.polynomial.legendre.leggauss(n_theta)
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    z, p = np.meshgrid(t, phi, indexing="ij")
    s = np.sqrt(1 - z**2)
    directions = np.stack([s * np.cos(p), s * np.sin(p), z], axis=-1).reshape(-1, 3)
    weights = (w[:, None] * np.full(n_phi, 2 * np.pi / n_phi)[None, :]).reshape(-1)
    return torch.as_tensor(directions, dtype=dtype, device=DEVICE), torch.as_tensor(weights, dtype=dtype, device=DEVICE)


def hg_sh_factors(g: float, order: int) -> torch.Tensor:
    """Factor by which the Henyey-Greenstein phase function scales each spherical harmonic when convolved: g^l."""
    return torch.tensor([g**l for l in range(order + 1) for _ in range(2 * l + 1)], **KW)


# --------------------------------------------------------------------------------------------------------------------
# Grids


def trilinear(grid: torch.Tensor, low: Sequence[float], size: float, points: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a grid of values at voxel corners, of shape (nx, ny, nz, c), at points (n, 3),
    clamped to the grid, of shape (n, c)."""
    kwargs = {"dtype": points.dtype, "device": points.device}
    shape = torch.tensor(grid.shape[:3], **kwargs)
    f = ((points - torch.as_tensor(low, **kwargs)) / size).clamp(torch.zeros(3, **kwargs), shape - 1 - 1e-4)
    i = f.floor().long()
    w = f - i
    flat = grid.reshape(-1, grid.shape[-1])
    nx, ny, nz = grid.shape[:3]
    result = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                index = ((i[:, 0] + dx) * ny + (i[:, 1] + dy)) * nz + (i[:, 2] + dz)
                weight = (
                    (w[:, 0] if dx else 1 - w[:, 0])
                    * (w[:, 1] if dy else 1 - w[:, 1])
                    * (w[:, 2] if dz else 1 - w[:, 2])
                )
                result = result + flat[index].to(points.dtype) * weight[:, None]
    return result


def inside(low: Sequence[float], high: Sequence[float], points: torch.Tensor) -> torch.Tensor:
    lo, hi = (torch.as_tensor(b, dtype=points.dtype, device=points.device) for b in (low, high))
    return ((points >= lo) & (points <= hi)).all(dim=-1)


def grid_points(low: Sequence[float], high: Sequence[float], size: float) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Corners of a grid of voxels of a given size covering a box, as points of shape (nx*ny*nz, 3), and its shape."""
    axes = [torch.arange(lo, hi + size * 0.5, size, **KW) for lo, hi in zip(low, high)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3), tuple(len(a) for a in axes)  # type: ignore[return-value]
