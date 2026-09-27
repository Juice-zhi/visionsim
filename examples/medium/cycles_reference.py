"""Render a test scene without a medium through visionsim, and with the equivalent volume in Cycles.

The volume is an exponential height fog lit by the sun, with single scattering only, as this is what
:mod:`visionsim.medium` models in closed form. Shadows cast onto the fog are disabled, and the world is black,
so that the only difference between the two is how light transport is computed. Use ``compare_cycles.py``
to compare the results. This needs to run from within Blender, e.g.::

    blender -b --factory-startup --python cycles_reference.py -- output/ [optix|cuda|cpu]
"""

import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

from visionsim.simulate.blender import BlenderService

# Fog: extinction at ground level (1/m), height falloff (m) and Henyey-Greenstein anisotropy
EXTINCTION, FALLOFF, ANISOTROPY = 0.04, 6.0, 0.6
# Sun elevation and azimuth (degrees, azimuth from +Y towards +X) and irradiance (W/m²)
SUN_ELEVATION, SUN_AZIMUTH, SUN_IRRADIANCE = 35.0, 90.0, 3.0
# Extent of the fog volume in Cycles, which rays leave through its sides
BOX_HALF_WIDTH, BOX_BOTTOM, BOX_TOP = 300.0, -1.0, 99.0


def diffuse(name, albedo):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (albedo, albedo, albedo, 1)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    return mat


def emissive(name, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = strength
    mat.node_tree.links.new(emission.outputs[0], nodes["Material Output"].inputs["Surface"])
    # The panel is only meant to be seen through the fog, not to light it (point and area lights aren't modeled yet)
    mat.cycles.emission_sampling = "NONE"
    return mat


def fog_material():
    # Volume scatter with density `EXTINCTION * exp(-z / FALLOFF)`, and no absorption (albedo of one)
    mat = bpy.data.materials.new("Fog")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.remove(nodes["Principled BSDF"])
    geometry, xyz = nodes.new("ShaderNodeNewGeometry"), nodes.new("ShaderNodeSeparateXYZ")
    scale, exp, density = (nodes.new("ShaderNodeMath") for _ in range(3))
    scale.operation, scale.inputs[1].default_value = "MULTIPLY", -1 / FALLOFF
    exp.operation = "EXPONENT"
    density.operation, density.inputs[1].default_value = "MULTIPLY", EXTINCTION
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (1, 1, 1, 1)  # defaults to gray, which would scale the density
    scatter.inputs["Anisotropy"].default_value = ANISOTROPY
    links.new(geometry.outputs["Position"], xyz.inputs[0])
    links.new(xyz.outputs["Z"], scale.inputs[0])
    links.new(scale.outputs[0], exp.inputs[0])
    links.new(exp.outputs[0], density.inputs[0])
    links.new(density.outputs[0], scatter.inputs["Density"])
    links.new(scatter.outputs[0], nodes["Material Output"].inputs["Volume"])
    return mat


def add(operator, material, **kwargs):
    operator(**kwargs)
    obj = bpy.context.active_object
    obj.data.materials.append(material)
    return obj


def build_scene(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = 320, 180
    scene.frame_start = scene.frame_end = 1

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world

    add(bpy.ops.mesh.primitive_plane_add, diffuse("Ground", 0.5), size=2 * BOX_HALF_WIDTH)
    objects = [
        add(bpy.ops.mesh.primitive_cube_add, diffuse("CubeA", 0.8), size=2, location=(-3, 15, 1)),
        add(bpy.ops.mesh.primitive_cube_add, diffuse("CubeB", 0.8), size=4, location=(6, 30, 2)),
        add(
            bpy.ops.mesh.primitive_plane_add,
            emissive("Panel", 5.0),
            size=4,
            location=(-2, 45, 3),
            rotation=(math.pi / 2, 0, 0),
        ),
    ]
    for obj in objects:
        obj.visible_shadow = False  # shadows cast onto the medium are not modeled yet
    objects[-1].visible_volume_scatter = objects[-1].visible_diffuse = False

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.angle = math.radians(60)
    camera.location, camera.rotation_euler = (0, -2, 1.6), (math.radians(88), 0, 0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type="SUN"))
    sun.data.energy = SUN_IRRADIANCE
    elevation, azimuth = math.radians(SUN_ELEVATION), math.radians(SUN_AZIMUTH)
    towards_sun = Vector(
        (math.sin(azimuth) * math.cos(elevation), math.cos(azimuth) * math.cos(elevation), math.sin(elevation))
    )
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = towards_sun.to_track_quat("Z", "Y")
    scene.collection.objects.link(sun)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def render(blend_file, output, device, samples, volume_passes=False):
    service = BlenderService()
    service.exposed_initialize(blend_file, output)
    service.exposed_cycles_settings(
        device_type=device, use_cpu=device == "cpu", max_samples=samples, use_denoising=False
    )
    service.scene.cycles.use_adaptive_sampling = False
    service.exposed_include_frames(file_format="OPEN_EXR", bit_depth=32, exr_codec="ZIP")

    if volume_passes:
        # Single scattering only, without Cycles' step limit which otherwise truncates long rays
        service.scene.cycles.volume_bounces = 0
        service.scene.cycles.volume_biased = False
        service.scene.cycles.volume_max_steps = 100_000
        service.view_layer.cycles.use_pass_volume_direct = True
        service.view_layer.cycles.use_pass_volume_indirect = True
        for name, subpath in (("Volume Direct", "volume/direct"), ("Volume Indirect", "volume/indirect")):
            socket = service.render_layers.outputs[name]
            service._include_output(subpath, socket, file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)
    else:
        service.exposed_include_depths(preview=False, exr_codec="ZIP")
        service.exposed_save_lighting()
    service.exposed_render_frame(1)


if __name__ == "__main__":
    root, *device = sys.argv[sys.argv.index("--") + 1 :]
    root, device = Path(root).resolve(), (device or ["optix"])[0]
    root.mkdir(parents=True, exist_ok=True)

    build_scene(root / "scene.blend")
    render(root / "scene.blend", root / "clear", device, samples=256)

    bpy.ops.wm.open_mainfile(filepath=str(root / "scene.blend"))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, (BOX_BOTTOM + BOX_TOP) / 2))
    box = bpy.context.active_object
    box.scale = (2 * BOX_HALF_WIDTH, 2 * BOX_HALF_WIDTH, BOX_TOP - BOX_BOTTOM)
    box.data.materials.append(fog_material())
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "fog.blend"))
    render(root / "fog.blend", root / "reference", device, samples=2048, volume_passes=True)
