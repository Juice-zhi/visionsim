"""Check that ``BlenderService.lighting_info`` exports lights and the world background correctly.

This needs to run from within Blender, for instance::

    blender -b --factory-startup --python-exit-code 1 --python lighting.py -- path/to/scene.blend path/to/lighting.json

The lighting of the modified scene is saved to the given JSON file, so it can be validated outside of Blender.
"""

from __future__ import annotations

import math
import sys
import tempfile

import bpy  # type: ignore
import numpy as np

from visionsim.simulate.blender import BlenderService


def main(blend_file: str, output: str) -> None:
    service = BlenderService()

    with tempfile.TemporaryDirectory() as root:
        service.exposed_initialize(blend_file, root)
        scene = bpy.context.scene

        # The test scene has a single point light
        (point,) = [o for o in scene.objects if o.type == "LIGHT"]
        info = service.exposed_lighting_info()
        assert not info["suns"] and len(info["points"]) == 1
        assert np.allclose(info["points"][0]["position"], point.matrix_world.translation)
        assert np.allclose(info["points"][0]["power"], np.array(point.data.color) * point.data.energy)

        # Add a sun whose light travels along -Z rotated by 30 degrees around X, i.e. it points towards (0, -0.5, cos 30)
        sun = bpy.data.lights.new("Sun", type="SUN")
        sun.energy, sun.color, sun.exposure, sun.volume_factor = 3.0, (1.0, 0.9, 0.8), 1.0, 0.5
        obj = bpy.data.objects.new("Sun", sun)
        obj.rotation_euler = (math.radians(30), 0.0, 0.0)
        scene.collection.objects.link(obj)
        bpy.context.view_layer.update()

        (exported,) = service.exposed_lighting_info()["suns"]
        assert np.allclose(exported["direction"], (0.0, -0.5, math.cos(math.radians(30))))
        assert np.allclose(exported["irradiance"], np.array((1.0, 0.9, 0.8)) * 3.0 * 2**1.0 * 0.5)

        # Constant world color, scaled by its strength
        background = next(n for n in scene.world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
        background.inputs["Color"].default_value = (0.2, 0.3, 0.4, 1.0)
        background.inputs["Strength"].default_value = 2.0
        assert np.allclose(service.exposed_lighting_info()["ambient"], (0.4, 0.6, 0.8))

        # Environment map, bright in the upper hemisphere only, whose solid angle is half of the sphere
        image = bpy.data.images.new("Sky", width=64, height=32, float_buffer=True)
        pixels = np.zeros((32, 64, 4), dtype=np.float32)
        pixels[16:] = (1.0, 0.5, 0.25, 1.0)
        image.pixels.foreach_set(pixels.ravel())
        environment = scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = image
        scene.world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
        assert np.allclose(service.exposed_lighting_info()["ambient"], np.array((1.0, 0.5, 0.25)) * 2.0 / 2)

        service.exposed_save_lighting(output)


if __name__ == "__main__":
    main(*sys.argv[sys.argv.index("--") + 1 :][:2])
