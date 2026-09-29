"""Save variants of a fog scene that isolate each source of light, to break down where the closed form differs.

Variants, each saved next to the output prefix: ``sun`` (black world), ``sun_noshadow`` (black world, objects cast
no shadows), ``sky`` (no sun) and ``sky_noobjects`` (no sun, only the ground and the fog remain). This needs to run
from within Blender, e.g.::

    blender -b fog.blend --python breakdown_scenes.py -- output/fog
"""

import sys

import bpy  # type: ignore

prefix = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
background = next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND")
sun = next(o for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN")
objects = [o for o in scene.objects if o.type == "MESH" and o.name not in ("Fog", "Plane")]


def save(name):
    bpy.ops.wm.save_as_mainfile(filepath=f"{prefix}_{name}.blend", copy=True)


background.inputs["Strength"].default_value = 0.0
save("sun")
for obj in objects:
    obj.visible_shadow = False
save("sun_noshadow")
for obj in objects:
    obj.visible_shadow = True

background.inputs["Strength"].default_value = 1.0
sun.hide_render = True
save("sky")
for obj in objects:
    bpy.data.objects.remove(obj, do_unlink=True)
save("sky_noobjects")
