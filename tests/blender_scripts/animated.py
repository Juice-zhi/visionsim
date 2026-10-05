"""Check that ``BlenderService.save_lighting`` and ``save_occlusion`` follow lights that change over frames.

This needs to run from within Blender, for instance::

    blender -b --factory-startup --python-exit-code 1 --python animated.py -- path/to/scene.blend path/to/output

The scene's point light rises and brightens over frames 1 to 5. Its lighting and shadow maps are saved to the given
directory, so they can be validated outside of Blender.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService


def main(blend_file: str, output: str) -> None:
    service = BlenderService()
    root = Path(output)

    with tempfile.TemporaryDirectory() as tmp:
        service.exposed_initialize(blend_file, tmp)
        scene = bpy.context.scene

        # The point light rises by 2 m from frame 1 to 5, and its power quadruples from frame 1 to 3, linearly
        (point,) = [o for o in scene.objects if o.type == "LIGHT"]
        point.keyframe_insert("location", frame=1)
        point.location.z += 2.0
        point.keyframe_insert("location", frame=5)
        point.data.keyframe_insert("energy", frame=1)
        point.data.energy *= 4
        point.data.keyframe_insert("energy", frame=3)
        actions = [point.animation_data.action, point.data.animation_data.action]
        for fcurve in service.exposed_iter_fcurves(actions):
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"

        # The current frame is left as it was
        scene.frame_set(2)
        service.exposed_save_lighting(root / "lighting.json", frames=range(1, 6))
        assert scene.frame_current == 2

        # Only the cube casts shadows, the scene's huge axes and grids would otherwise make the maps span them. The lamp
        # gets a map wherever it moved further than 0.75 m from where its maps were rendered
        exclude = [obj.name for obj in scene.objects if obj.name != "Cube"]
        service.exposed_save_occlusion(
            root / "occlusion.npz",
            exclude=exclude,
            sky_elevations=(0, 30, 90),
            sky_cells=(4, 2),
            sun_resolution=64,
            sky_resolution=16,
            lamp_resolution=64,
            frames=range(1, 6),
            lamp_spacing=0.75,
        )
        assert scene.frame_current == 2


if __name__ == "__main__":
    main(*sys.argv[sys.argv.index("--") + 1 :][:2])
