"""Save a night variant of a fog scene, lit only by a lamp, to validate how the medium scatters its light.

The sun and the sky are turned off, and a point, spot or area light is added, whose strength can optionally be set by
a Light Falloff node. This needs to run from within Blender, e.g.::

    blender -b fog.blend --python lamp_scene.py -- output.blend --position -1.5 22 3 --power 3000
    blender -b fog.blend --python lamp_scene.py -- spot.blend --type SPOT --position 1.5 22 4 --direction -0.35 -0.3 -0.9
"""

import argparse
import sys
from pathlib import Path

import bpy  # type: ignore
import mathutils  # type: ignore


def main(args):
    scene = bpy.context.scene
    next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND").inputs["Strength"].default_value = 0.0
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True

    light = bpy.data.lights.new("Lamp", type=args.type)
    light.energy = args.power
    if args.type == "AREA":
        light.shape, light.size, light.size_y, light.spread = "RECTANGLE", *args.size, args.spread
    else:
        light.shadow_soft_size = args.radius
    if args.type == "SPOT":
        light.spot_size, light.spot_blend = args.angle, args.blend
    if args.smooth is not None:
        # Light Falloff node, whose quadratic output keeps the physical falloff and smooths the light near the lamp
        light.use_nodes = True
        tree = light.node_tree
        emission = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeEmission")
        falloff = tree.nodes.new("ShaderNodeLightFalloff")
        falloff.inputs["Strength"].default_value, falloff.inputs["Smooth"].default_value = 1.0, args.smooth
        tree.links.new(falloff.outputs["Quadratic"], emission.inputs["Strength"])

    lamp = bpy.data.objects.new("Lamp", light)
    lamp.location = args.position
    # Lamps shine along their local -Z axis
    lamp.rotation_euler = (-mathutils.Vector(args.direction)).to_track_quat("Z", "Y").to_euler()
    scene.collection.objects.link(lamp)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.output).resolve()), copy=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="blend file to save the variant to")
    parser.add_argument("--type", choices=("POINT", "SPOT", "AREA"), default="POINT", help="type of lamp")
    parser.add_argument("--position", type=float, nargs=3, default=(-1.5, 22.0, 3.0), help="light position (m)")
    parser.add_argument("--direction", type=float, nargs=3, default=(0.0, 0.0, -1.0), help="direction of spot/area")
    parser.add_argument("--power", type=float, default=3000.0, help="light power (W)")
    parser.add_argument("--radius", type=float, default=0.1, help="radius of point/spot lights (m)")
    parser.add_argument("--angle", type=float, default=0.9, help="angle of the cone of spot lights (rad)")
    parser.add_argument("--blend", type=float, default=0.15, help="softness of the edge of spot lights")
    parser.add_argument("--size", type=float, nargs=2, default=(2.0, 1.0), help="size of area lights (m)")
    parser.add_argument("--spread", type=float, default=3.14159265, help="spread of area lights (rad)")
    parser.add_argument("--smooth", type=float, help="smoothing of a Light Falloff node setting the strength")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
