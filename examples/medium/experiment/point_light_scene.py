"""Save a night variant of a fog scene, lit only by a point light, to validate how the medium scatters its light.

The sun and the sky are turned off, and a point light is added. This needs to run from within Blender, e.g.::

    blender -b fog.blend --python point_light_scene.py -- output.blend --position -1.5 22 3 --power 3000
"""

import argparse
import sys
from pathlib import Path

import bpy  # type: ignore


def main(args):
    scene = bpy.context.scene
    next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND").inputs["Strength"].default_value = 0.0
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True

    light = bpy.data.lights.new("Lamp", type="POINT")
    light.energy = args.power
    light.shadow_soft_size = args.radius
    lamp = bpy.data.objects.new("Lamp", light)
    lamp.location = args.position
    scene.collection.objects.link(lamp)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.output).resolve()), copy=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="blend file to save the variant to")
    parser.add_argument("--position", type=float, nargs=3, default=(-1.5, 22.0, 3.0), help="light position (m)")
    parser.add_argument("--power", type=float, default=3000.0, help="light power (W)")
    parser.add_argument("--radius", type=float, default=0.1, help="light radius (m)")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
