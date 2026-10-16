"""Describe the fog scenes of the acceleration study for Unreal Engine: ``runs/.../acceleration/unreal/scene.json``.

Blender's right-handed, Z-up frame in meters becomes Unreal's left-handed, Z-up frame in centimeters by negating Y,
which keeps images unmirrored. The fog's extinction ``0.08 exp(-z / 2.5 m)`` per meter becomes Unreal's exponential
height fog, whose volumetric fog has an extinction ``0.5 * FogDensity / 1000 * 2^(-FogHeightFalloff / 1000 * z)`` per
centimeter (``VolumetricFog.usf`` and ``SceneCore.cpp``), hence ``FogDensity = 1.6`` and
``FogHeightFalloff = 1000 / (250 ln 2)``. Lights keep Blender's numbers in photometric units, i.e. the sun's 4 W/m²
become 4 lux and the lamp's 3000 W a point light of 3000 / 4π candela, as Unreal's outputs are calibrated against
Cycles afterwards. The sky's uniform radiance is saved as an equirectangular HDR image for Unreal's sky light.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, ROOT, load_transforms

UNREAL = OUT / "unreal"
# Frames rendered: from well before the ones compared, so that the volumetric fog's temporal accumulation has a
# history as in a game
FRAMES = list(range(230, 346))


def ue(v) -> list[float]:
    """A Blender vector in Unreal's frame, without scaling."""
    return [float(v[0]), -float(v[1]), float(v[2])]


def write_hdr(path: Path, color, width: int = 64, height: int = 32) -> None:
    """Uncompressed Radiance HDR image of a uniform color."""
    color = np.asarray(color, dtype=np.float64)
    mantissa, exponent = np.frexp(color.max())
    rgbe = np.concatenate([np.round(color / 2.0 ** (exponent - 8)).clip(0, 255), [exponent + 128]]).astype(np.uint8)
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n-Y {height} +X {width}\n".encode()
    path.write_bytes(header + np.tile(rgbe, width * height).tobytes())


def main():
    UNREAL.mkdir(parents=True, exist_ok=True)
    cameras = {}
    for frame, t in zip(FRAMES, load_transforms(ROOT / "clear" / "frames", FRAMES)):
        m = np.asarray(t["transform_matrix"], dtype=np.float64)
        cameras[frame] = {
            "location": [100 * c for c in ue(m[:3, 3])],
            "forward": ue(-m[:3, 2] / np.linalg.norm(m[:3, 2])),
            "up": ue(m[:3, 1] / np.linalg.norm(m[:3, 1])),
        }
    camera = load_transforms(ROOT / "clear" / "frames", [255])[0]
    lighting = json.loads((ROOT / "clear" / "lighting.json").read_text())
    point = json.loads((ROOT / "point_shadow" / "lighting.json").read_text())["points"][0]
    medium = json.loads(Path("examples/medium/media/ground_fog.json").read_text())
    fog = medium["components"][0]
    sun = np.asarray(lighting["suns"][0]["direction"], dtype=np.float64)
    objects = [
        {"mesh": "Cube", "location": (-3.0, 10.0, 1.0), "size": (2.0, 2.0, 2.0), "color": (0.8, 0.2, 0.2)},
        {"mesh": "Cube", "location": (4.0, 18.0, 1.5), "size": (3.0, 3.0, 3.0), "color": (0.2, 0.7, 0.3)},
        {"mesh": "Cube", "location": (-6.0, 30.0, 2.0), "size": (4.0, 4.0, 4.0), "color": (0.2, 0.3, 0.8)},
        {"mesh": "Cube", "location": (7.0, 45.0, 3.0), "size": (6.0, 6.0, 6.0), "color": (0.9, 0.8, 0.3)},
    ]
    for k, (x, y) in enumerate([(10, 15), (-10, 30), (10, 45), (-10, 60), (10, 75), (-10, 90), (10, 105)]):
        objects.append({"mesh": "Cylinder", "location": (x, y, 4.0), "size": (0.4, 0.4, 8.0), "color": (0.9, 0.9, 0.9)})
    for o in objects:
        o["location"] = [100 * c for c in ue(o["location"])]
        o["scale"] = list(o.pop("size"))  # Unreal's basic shapes are 1 m wide
    scene = {
        "frames": cameras,
        "compare": [255, 256, 257, 258, 259, 260, 261, 262, 345],
        "resolution": [int(camera["w"]), int(camera["h"])],
        "focal_mm": float(camera["lens"]),
        "sensor_mm": [36.0, 36.0 * int(camera["h"]) / int(camera["w"])],
        "sun": {
            "towards": ue(sun / np.linalg.norm(sun)),
            "lux": float(lighting["suns"][0]["irradiance"][0]),
            "angle_deg": math.degrees(0.009180432185530663),
        },
        "sky": {"color": lighting["sky"]},
        "point": {
            "location": [100 * c for c in ue(point["position"])],
            "candela": point["power"][0] / (4 * math.pi),
            "radius_cm": 100 * point["radius"],
        },
        "fog": {
            "density": medium["extinction"] / 100 * 1000 * 2 * fog["density"],
            "falloff": 1000 / (100 * fog["falloff"] * math.log(2)),
            "g": medium["anisotropy"],
            "distance_cm": 25000.0,
        },
        "objects": objects,
        "ground": {"scale": 400.0, "colors": [0.6, 0.18]},
        "sky_sphere_cm": 19000.0,
    }
    (UNREAL / "scene.json").write_text(json.dumps(scene, indent=1))
    write_hdr(UNREAL / "sky.hdr", lighting["sky"])
    # The ground's checker as a 2x2 texture of linear values, tiled once per 2 m
    a, b = (round(255 * c) for c in scene["ground"]["colors"])
    Image.fromarray(np.array([[b, a], [a, b]], dtype=np.uint8)).convert("RGB").save(UNREAL / "checker.png")
    print(json.dumps({k: v for k, v in scene.items() if k not in ("frames", "objects")}, indent=1))


if __name__ == "__main__":
    main()
