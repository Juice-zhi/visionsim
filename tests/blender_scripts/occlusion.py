"""Check that ``BlenderService.save_occlusion`` renders shadow maps without changing the scene's render settings.

This needs to run from within Blender, for instance::

    blender -b --factory-startup --python-exit-code 1 --python occlusion.py -- path/to/scene.blend path/to/occlusion.npz

A sun and a large ground plane, which is excluded from the shadow maps, are added to the scene. The shadow maps are
saved to the given file, so they can be validated outside of Blender.
"""

from __future__ import annotations

import math
import sys
import tempfile

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService


def main(blend_file: str, output: str) -> None:
    service = BlenderService()

    with tempfile.TemporaryDirectory() as root:
        service.exposed_initialize(blend_file, root)
        scene = bpy.context.scene
        service.exposed_cycles_settings(device_type="cpu", use_cpu=True, max_samples=7, use_denoising=True)
        service.exposed_include_depths(preview=False)

        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type="SUN"))
        sun.rotation_euler = (math.radians(30), 0.0, 0.0)
        scene.collection.objects.link(sun)
        bpy.ops.mesh.primitive_plane_add(size=1000, location=(0, 0, -5))
        bpy.context.active_object.name = "Ground"
        # A glowing ball above the cube, which casts shadows from its surface, through itself
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(0, 0, 4))
        ball = bpy.context.active_object
        ball.name = "Ball"
        material = bpy.data.materials.new("Glow")
        material.use_nodes = True
        tree = material.node_tree
        emission = tree.nodes.new("ShaderNodeEmission")
        # As bright as the scene's point light, rather than left out next to it
        emission.inputs["Strength"].default_value = 1000.0
        surface = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial").inputs["Surface"]
        tree.links.new(emission.outputs[0], surface)
        ball.data.materials.append(material)

        before = {
            "camera": scene.camera,
            "resolution": (scene.render.resolution_x, scene.render.resolution_y),
            "samples": scene.cycles.samples,
            "denoising": scene.cycles.use_denoising,
            "format": scene.render.image_settings.file_format,
            "filepath": scene.render.filepath,
            "objects": len(bpy.data.objects),
            "outputs": [(n.name, n.mute) for n in service.tree.nodes if n.bl_idname == "CompositorNodeOutputFile"],
            "links": sorted((link.from_node.name, link.to_node.name) for link in service.tree.links),
            "material": sorted((link.from_node.name, link.to_node.name) for link in tree.links) + [len(tree.nodes)],
        }
        # Only the cube and the ball cast shadows, the scene's huge axes and grids would otherwise make the maps span them
        exclude = [obj.name for obj in scene.objects if obj.name not in ("Cube", "Ball")]
        service.exposed_save_occlusion(
            output,
            exclude=exclude,
            sky_elevations=(0, 30, 90),
            sky_cells=(4, 2),
            sun_resolution=128,
            sky_resolution=32,
            lamp_resolution=256,
        )
        after = {
            "camera": scene.camera,
            "resolution": (scene.render.resolution_x, scene.render.resolution_y),
            "samples": scene.cycles.samples,
            "denoising": scene.cycles.use_denoising,
            "format": scene.render.image_settings.file_format,
            "filepath": scene.render.filepath,
            "objects": len(bpy.data.objects),
            "outputs": [(n.name, n.mute) for n in service.tree.nodes if n.bl_idname == "CompositorNodeOutputFile"],
            "links": sorted((link.from_node.name, link.to_node.name) for link in service.tree.links),
            "material": sorted((link.from_node.name, link.to_node.name) for link in tree.links) + [len(tree.nodes)],
        }
        assert before == after, (before, after)
        assert not any(bpy.data.objects[name].hide_render for name in exclude)


if __name__ == "__main__":
    main(*sys.argv[sys.argv.index("--") + 1 :][:2])
