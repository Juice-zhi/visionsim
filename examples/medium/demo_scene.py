"""Create a small animated scene to try out every sensor, with and without participating media.

The camera moves through a textured ground plane with objects at various depths, lit by a sun and a bluish
sky, so that fog affects near and far objects differently. The animation lasts 100 frames at 25 fps. This needs
to run from within Blender, e.g.::

    blender -b --factory-startup --python demo_scene.py -- demo.blend [width height]
"""

import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

SUN_ELEVATION, SUN_AZIMUTH, SUN_IRRADIANCE = 30.0, 60.0, 4.0
SKY_RADIANCE = (0.25, 0.35, 0.55)


def material(name, color, checker=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Base Color"].default_value = (*color, 1)

    if checker:
        texture = nodes.new("ShaderNodeTexChecker")
        texture.inputs["Scale"].default_value = checker
        texture.inputs["Color1"].default_value = (*color, 1)
        texture.inputs["Color2"].default_value = (*(c * 0.3 for c in color), 1)
        links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add(operator, mat, **kwargs):
    operator(**kwargs)
    obj = bpy.context.active_object
    obj.data.materials.append(mat)
    return obj


def build(path, width=320, height=180):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.fps = 25
    scene.frame_start, scene.frame_end = 1, 100

    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*SKY_RADIANCE, 1)
    scene.world = world

    add(bpy.ops.mesh.primitive_plane_add, material("Ground", (0.6, 0.6, 0.6), checker=400), size=400)
    for i, (x, y, size, color) in enumerate(
        [
            (-3, 10, 2, (0.8, 0.2, 0.2)),
            (4, 18, 3, (0.2, 0.7, 0.3)),
            (-6, 30, 4, (0.2, 0.3, 0.8)),
            (7, 45, 6, (0.9, 0.8, 0.3)),
        ]
    ):
        add(bpy.ops.mesh.primitive_cube_add, material(f"Cube{i}", color), size=size, location=(x, y, size / 2))
    for i, y in enumerate(range(15, 120, 15)):
        add(
            bpy.ops.mesh.primitive_cylinder_add,
            material(f"Pole{i}", (0.9, 0.9, 0.9)),
            radius=0.2,
            depth=8,
            location=(-10 if i % 2 else 10, y, 4),
        )

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.angle = math.radians(70)
    camera.rotation_euler = (math.radians(86), 0, 0)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for frame, (y, yaw) in ((1, (-4.0, 8.0)), (100, (20.0, -8.0))):
        camera.location = (0, y, 1.6)
        camera.rotation_euler.z = math.radians(yaw)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type="SUN"))
    sun.data.energy = SUN_IRRADIANCE
    elevation, azimuth = math.radians(SUN_ELEVATION), math.radians(SUN_AZIMUTH)
    towards = Vector(
        (math.sin(azimuth) * math.cos(elevation), math.cos(azimuth) * math.cos(elevation), math.sin(elevation))
    )
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = towards.to_track_quat("Z", "Y")
    scene.collection.objects.link(sun)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()))


if __name__ == "__main__":
    path, *size = sys.argv[sys.argv.index("--") + 1 :]
    build(path, *map(int, size))
