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

from visionsim.simulate.blender import BlenderService, _blackbody


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

        # Spot light whose color is set by a Blackbody node, and its strength by the linear output of a Light Falloff
        spot = bpy.data.lights.new("Spot", type="SPOT")
        spot.energy, spot.spot_size, spot.spot_blend, spot.shadow_soft_size = 100.0, math.radians(60), 0.2, 0.05
        spot.use_nodes = True
        tree = spot.node_tree
        emission = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeEmission")
        blackbody, falloff = tree.nodes.new("ShaderNodeBlackbody"), tree.nodes.new("ShaderNodeLightFalloff")
        blackbody.inputs["Temperature"].default_value = 3000.0
        falloff.inputs["Strength"].default_value, falloff.inputs["Smooth"].default_value = 5.0, 0.5
        tree.links.new(blackbody.outputs["Color"], emission.inputs["Color"])
        tree.links.new(falloff.outputs["Linear"], emission.inputs["Strength"])
        obj = bpy.data.objects.new("Spot", spot)
        obj.location, obj.rotation_euler = (1.0, 2.0, 3.0), (math.radians(30), 0.0, 0.0)
        scene.collection.objects.link(obj)

        # Rectangular area light stretched by its object's scale, and a disk light without normalization
        rectangle = bpy.data.lights.new("Rectangle", type="AREA")
        rectangle.shape, rectangle.size, rectangle.size_y, rectangle.energy = "RECTANGLE", 2.0, 1.0, 50.0
        rectangle.spread = math.radians(150)
        obj = bpy.data.objects.new("Rectangle", rectangle)
        obj.location, obj.rotation_euler, obj.scale = (0.0, 0.0, 4.0), (0.0, 0.0, math.radians(90)), (1.5, 2.0, 1.0)
        scene.collection.objects.link(obj)
        disk = bpy.data.lights.new("Disk", type="AREA")
        disk.shape, disk.size, disk.energy, disk.normalize = "DISK", 0.5, 20.0, False
        scene.collection.objects.link(bpy.data.objects.new("Disk", disk))
        bpy.context.view_layer.update()

        info = service.exposed_lighting_info()
        (exported,) = info["spots"]
        assert np.allclose(exported["direction"], (0.0, math.sin(math.radians(30)), -math.cos(math.radians(30))))
        assert np.allclose([exported["angle"], exported["blend"]], [math.radians(60), 0.2])
        assert exported["falloff"] == "linear" and exported["smooth"] == 0.5
        assert np.allclose(exported["power"], np.array(_blackbody(3000.0)) * 5.0 * 100.0)
        rectangle, disk = sorted(info["areas"], key=lambda area: area["shape"], reverse=True)
        assert np.allclose([rectangle["direction"], rectangle["axis_u"]], [(0, 0, -1), (0, 1, 0)], atol=1e-6)
        assert np.allclose(rectangle["size"], (3.0, 2.0)) and rectangle["shape"] == "rectangle"
        assert np.allclose([*rectangle["power"], rectangle["spread"]], [50.0] * 3 + [math.radians(150)])
        assert disk["shape"] == "ellipse" and np.allclose(disk["size"], (0.5, 0.5))
        assert np.allclose(disk["power"], 20.0 * math.pi / 4 * 0.5**2)
        for name in ("Spot", "Rectangle", "Disk"):
            bpy.data.objects.remove(bpy.data.objects[name])

        # Constant world color, scaled by its strength
        background = next(n for n in scene.world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
        background.inputs["Color"].default_value = (0.2, 0.3, 0.4, 1.0)
        background.inputs["Strength"].default_value = 2.0
        assert np.allclose(service.exposed_lighting_info()["sky"], (0.4, 0.6, 0.8))

        # Environment map, whose upper hemisphere is bright and lower one is dark. Only the sky above the horizon is
        # exported, and its brightest band, near the zenith, covers a small solid angle so it weighs less
        image = bpy.data.images.new("Sky", width=64, height=32, float_buffer=True)
        pixels = np.zeros((32, 64, 4), dtype=np.float32)
        pixels[16:] = (1.0, 0.5, 0.25, 1.0)
        pixels[-1] = (101.0, 100.5, 100.25, 1.0)
        image.pixels.foreach_set(pixels.ravel())
        environment = scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = image
        scene.world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
        latitudes = (np.arange(16) + 0.5) / 32 * np.pi
        zenith_weight = np.cos(latitudes[-1]) / np.cos(latitudes).sum()
        expected = (np.array((1.0, 0.5, 0.25)) + 100 * zenith_weight) * 2.0
        assert np.allclose(service.exposed_lighting_info()["sky"], expected)

        service.exposed_save_lighting(output)


if __name__ == "__main__":
    main(*sys.argv[sys.argv.index("--") + 1 :][:2])
