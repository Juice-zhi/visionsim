"""Check that the intrinsics reported by ``BlenderService.camera_info`` agree with Blender's own projection.

Random points in front of the camera are projected both with Blender's ``world_to_camera_view`` and with
the pinhole model defined by ``camera_info``/``camera_extrinsics``, for a variety of resolutions, resolution
percentages, pixel aspect ratios, sensor fits and lens shifts. The script exits with an error if any
projection is off by more than a thousandth of a pixel.

This needs to run from within Blender, for instance::

    blender -b --factory-startup --python-exit-code 1 --python camera_intrinsics.py -- path/to/scene.blend
"""

from __future__ import annotations

import sys
import tempfile

import bpy  # type: ignore
import numpy as np
from bpy_extras.object_utils import world_to_camera_view  # type: ignore
from mathutils import Vector  # type: ignore

from visionsim.simulate.blender import BlenderService

# (width, height), resolution percentage, (pixel_aspect_x, pixel_aspect_y), sensor fit, (shift_x, shift_y)
# Note: `world_to_camera_view` normalizes by the unscaled resolution, so percentages are chosen such that
#   the scaled resolution is not truncated, otherwise the reference itself is off by a fraction of a pixel.
CONFIGS = [
    ((640, 640), 100, (1, 1), "AUTO", (0.0, 0.0)),
    ((640, 480), 100, (1, 1), "AUTO", (0.0, 0.0)),
    ((480, 640), 100, (1, 1), "AUTO", (0.0, 0.0)),
    ((640, 480), 50, (1, 1), "AUTO", (0.0, 0.0)),
    ((640, 360), 75, (1, 1), "AUTO", (0.0, 0.0)),
    ((641, 359), 100, (1, 1), "AUTO", (0.0, 0.0)),
    ((640, 480), 100, (1, 1), "AUTO", (0.1, -0.05)),
    ((480, 640), 100, (1, 1), "AUTO", (-0.08, 0.12)),
    ((640, 360), 100, (1, 1), "HORIZONTAL", (0.05, 0.1)),
    ((360, 640), 100, (1, 1), "HORIZONTAL", (0.05, 0.1)),
    ((640, 360), 100, (1, 1), "VERTICAL", (-0.1, 0.05)),
    ((360, 640), 80, (1, 1), "VERTICAL", (0.1, -0.1)),
    ((640, 480), 100, (1, 2), "AUTO", (0.05, 0.05)),
    ((640, 480), 100, (3, 2), "AUTO", (0.0, 0.0)),
]
TOLERANCE_PX = 1e-3


def main(blend_file: str) -> None:
    service = BlenderService()
    rng = np.random.default_rng(1234)
    failures = []

    with tempfile.TemporaryDirectory() as root:
        service.exposed_initialize(blend_file, root)
        scene = bpy.context.scene
        camera = service.camera
        camera.data.lens = 35
        camera.data.sensor_width = 36
        camera.data.sensor_height = 24

        for (w, h), percentage, (xasp, yasp), fit, (sx, sy) in CONFIGS:
            scene.render.resolution_x, scene.render.resolution_y = w, h
            scene.render.resolution_percentage = percentage
            scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = xasp, yasp
            camera.data.sensor_fit = fit
            camera.data.shift_x, camera.data.shift_y = sx, sy
            bpy.context.view_layer.update()

            info = service.exposed_camera_info()
            pose = np.array(service.exposed_camera_extrinsics())

            # Sample points in camera space and keep the ones that land inside the frame
            local = np.stack([rng.uniform(-4, 4, 2000), rng.uniform(-4, 4, 2000), rng.uniform(-12, -3, 2000)], axis=-1)
            world = [camera.matrix_world @ Vector(p) for p in local]
            ndc = np.array([tuple(world_to_camera_view(scene, camera, p)) for p in world])
            inside = np.all((ndc[:, :2] > 0) & (ndc[:, :2] < 1), axis=-1)
            world_pts, ndc = np.array([tuple(p) for p in world])[inside], ndc[inside]

            # Blender's normalized frame coordinates have their origin at the bottom-left corner
            expected = np.stack([ndc[:, 0] * info["w"], (1 - ndc[:, 1]) * info["h"]], axis=-1)

            # Pinhole projection in Blender/OpenGL camera coordinates (+X right, +Y up, -Z forward)
            homogeneous = np.concatenate([world_pts, np.ones((len(world_pts), 1))], axis=-1)
            x, y, z, _ = (np.linalg.inv(pose) @ homogeneous.T).astype(float)
            actual = np.stack([info["cx"] + info["fl_x"] * x / -z, info["cy"] - info["fl_y"] * y / -z], axis=-1)

            error = np.abs(actual - expected).max()
            print(
                f"{w}x{h} @ {percentage}%, aspect={xasp}:{yasp}, fit={fit}, shift=({sx}, {sy}): "
                f"{len(ndc)} points, max error {error:.2e}px",
                flush=True,
            )
            if error > TOLERANCE_PX:
                failures.append((w, h, percentage, xasp, yasp, fit, sx, sy, error))

    if failures:
        raise AssertionError(f"Intrinsics disagree with Blender's projection for {len(failures)} configurations.")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1])
