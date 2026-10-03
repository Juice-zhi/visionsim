"""Where the closed form with multiple scattering and shadows differs from Cycles, on frames 255 and 345.

Cycles renders of these frames with up to 32 volume bounces and their volume passes (``bounces.py``) give the light
that surfaces send through the fog, i.e. the combined image minus the volume passes, divided by the transmittance along
camera rays, and the light that the fog scatters towards the camera, split into light scattered once from the sun and
the sky (volume direct) and the rest (volume indirect). The closed form's are compared with them by distance to the
surfaces seen by pixels.

The light of surfaces is also compared with that of a render without fog whose light is split between the sun and the
sky (``render_passes.py --light-groups``), attenuated by the fog towards the sun and over the sky seen by surfaces,
which shows how much the fog dims surfaces before its own glow lights them again. Results are saved to
``results_ms/error_sources.json``.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from run_experiment import GROUND_ALBEDO, OCCLUSION, ROOT

from visionsim.dataset import Dataset
from visionsim.medium import Lighting, Medium, apply_medium, load_occlusion
from visionsim.medium.optics import height_fog_optical_depth
from visionsim.medium.render import camera_rays

FRAMES = {255: 250, 345: 340}
"""Frames of the animation rendered with their passes, and their index in the dataset"""
REGIONS = {"surfaces_near": (0, 5), "surfaces_mid": (5, 15), "surfaces_far": (15, math.inf)}
"""Distances, in meters, of the surfaces seen by pixels in each region"""


def load(path: Path, index: int) -> torch.Tensor:
    return torch.as_tensor(np.asarray(Dataset.from_path(path)[index][0], dtype=float)[..., :3])


def sky_attenuation(depth_above: torch.Tensor, normal_z: torch.Tensor) -> torch.Tensor:
    """Attenuation of the light of a uniform sky by an exponential height fog, of optical depth ``depth_above`` above
    surfaces whose normals have a vertical component ``normal_z``, weighted by the cosine over the sky they see."""
    mu = (torch.arange(64, dtype=torch.float64) + 0.5) / 64  # vertical components of directions above the horizon
    a = normal_z[..., None] * mu
    b = (1 - normal_z[..., None] ** 2).clamp_min(0).sqrt() * (1 - mu**2).sqrt()
    phi = torch.arccos((-a / b.clamp_min(1e-12)).clamp(-1, 1))
    cosine = torch.where(b > 1e-9, 2 * (a * phi + b * torch.sin(phi)), 2 * math.pi * a.clamp_min(0))
    return (cosine * torch.exp(-depth_above[..., None] / mu)).sum(dim=-1) / cosine.sum(dim=-1).clamp_min(1e-12)


def main():
    lighting = Lighting.model_validate_json((ROOT / "clear" / "lighting.json").read_text())
    lighting = lighting.model_copy(update={"ground_albedo": (GROUND_ALBEDO,) * 3})
    medium = Medium.model_validate_json(Path("examples/medium/media/ground_fog.json").read_text())
    medium = medium.model_copy(update={"multiple_scattering": True})
    single = medium.model_copy(update={"multiple_scattering": False})
    unlit_ground = lighting.model_copy(update={"ground_albedo": (0.0,) * 3})
    fog = medium.components[0]
    towards_sun = np.asarray(lighting.suns[0].direction, dtype=float)
    sun_z = float(towards_sun[2] / np.linalg.norm(towards_sun))
    occlusion = load_occlusion(OCCLUSION, device="cuda", dtype=torch.float32)
    kwargs = {"device": "cuda", "dtype": torch.float32}
    results: dict = {}
    for k, (frame, index) in enumerate(FRAMES.items()):
        passes = ROOT / "bounces" / "32"
        total = load(passes / "frames", k)
        direct = load(passes / "volume" / "direct", k)
        volume = direct + load(passes / "volume" / "indirect", k)
        lights = ROOT / "lightgroups"
        clear = load(lights / "frames", k)
        sun, sky = load(lights / "lights" / "sun", k), load(lights / "lights" / "sky", k)
        normal_z = load(lights / "normals", k)[..., 2].clamp(-1, 1)

        image, transform = Dataset.from_path(ROOT / "clear" / "frames")[index]
        raw = np.asarray(Dataset.from_path(ROOT / "clear" / "depths")[index][0], dtype=float)[..., 0]
        origin, directions, scale = camera_rays(transform, transform["transform_matrix"], dtype=torch.float64)
        surface = torch.as_tensor(raw < 1e9)
        distance = torch.where(surface, torch.as_tensor(raw) * scale, torch.zeros_like(scale))
        depth = height_fog_optical_depth(fog, origin, directions.reshape(-1, 3), distance.reshape(-1))
        transmittance = torch.exp(-depth * medium.extinction).reshape(distance.shape)[..., None]
        seen = (total - volume) / transmittance  # light of surfaces through the fog, before the camera ray
        points = origin + directions * distance[..., None]
        above = (
            medium.extinction * fog.density * fog.falloff * torch.exp(-(points[..., 2] - fog.base_height) / fog.falloff)
        )
        attenuated = sun * torch.exp(-above / sun_z)[..., None] + sky * sky_attenuation(above, normal_z)[..., None]
        attenuated = attenuated + clear - sun - sky

        args = (
            np.asarray(image, dtype=np.float32)[..., :3],
            raw.astype(np.float32),
            transform,
            transform["transform_matrix"],
        )
        ours = apply_medium(*args, medium, lighting, occlusion=occlusion, **kwargs)
        ours_direct = apply_medium(*args, single, unlit_ground, occlusion=occlusion, **kwargs).inscatter.cpu().double()
        ours_volume, ours_total = ours.inscatter.cpu().double(), ours.radiance.cpu().double()

        def ratio(a: torch.Tensor, b: torch.Tensor, region: torch.Tensor) -> float:
            return float(a.sum(dim=-1)[region].sum() / b.sum(dim=-1)[region].sum())

        regions = {name: surface & (distance >= low) & (distance < high) for name, (low, high) in REGIONS.items()}
        regions["sky"] = ~surface
        results[str(frame)] = {}
        for name, region in regions.items():
            entry = {
                "pixels": float(region.double().mean()),
                "fog_light": ratio(ours_volume, volume, region),
                "fog_direct": ratio(ours_direct, direct, region),
                "fog_indirect": ratio(ours_volume - ours_direct, volume - direct, region),
                "indirect_share": ratio(volume - direct, volume, region),
                "frame": ratio(ours_total, total, region),
            }
            if name != "sky":
                entry |= {
                    "surface_clear": ratio(clear, seen, region),
                    "surface_attenuated": ratio(attenuated, seen, region),
                }
            results[str(frame)][name] = entry
        print(frame, json.dumps(results[str(frame)], indent=1), flush=True)
    (ROOT / "results_ms" / "error_sources.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
