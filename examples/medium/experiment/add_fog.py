"""Add a Cycles fog volume, equivalent to a visionsim medium, to an existing scene.

The volume is a box filled with the density of the medium's homogeneous and height fog components, scattering light
according to the medium's albedo and Henyey-Greenstein anisotropy. It is used to render reference images with
Cycles' volumetric path tracing, against which the closed-form medium can be compared. This needs to run from
within Blender, e.g.::

    blender -b scene.blend --python add_fog.py -- medium.json output.blend --half-width 200 --top 25
"""

import argparse
import json
import sys
from pathlib import Path

import bpy  # type: ignore


def density_socket(nodes, links, medium):
    """Build nodes evaluating the medium's extinction at each shading point, and return their output socket."""
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(nodes.new("ShaderNodeNewGeometry").outputs["Position"], xyz.inputs[0])
    total = None

    for component in medium["components"]:
        if component["type"] == "homogeneous":
            value = nodes.new("ShaderNodeValue")
            value.outputs[0].default_value = component["density"]
            term = value.outputs[0]
        elif component["type"] == "height":
            # density * exp(-(z - base_height) / falloff)
            shift, scale, exp, weight = (nodes.new("ShaderNodeMath") for _ in range(4))
            shift.operation, shift.inputs[1].default_value = "SUBTRACT", component["base_height"]
            scale.operation, scale.inputs[1].default_value = "MULTIPLY", -1 / component["falloff"]
            exp.operation = "EXPONENT"
            weight.operation, weight.inputs[1].default_value = "MULTIPLY", component["density"]
            links.new(xyz.outputs["Z"], shift.inputs[0])
            links.new(shift.outputs[0], scale.inputs[0])
            links.new(scale.outputs[0], exp.inputs[0])
            links.new(exp.outputs[0], weight.inputs[0])
            term = weight.outputs[0]
        else:
            raise NotImplementedError(f"Component '{component['type']}' is not supported by this script.")

        if total is None:
            total = term
        else:
            add = nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            links.new(total, add.inputs[0])
            links.new(term, add.inputs[1])
            total = add.outputs[0]

    extinction = nodes.new("ShaderNodeMath")
    extinction.operation, extinction.inputs[1].default_value = "MULTIPLY", medium["extinction"]
    links.new(total, extinction.inputs[0])
    return extinction.outputs[0]


def fog_material(medium):
    if medium.get("angstrom", 0.0) != 0.0:
        raise NotImplementedError("Wavelength dependent extinction is not supported by this script.")

    mat = bpy.data.materials.new("Fog")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    # Look nodes up by type, as their names are translated when Blender's interface isn't in English
    for node in [n for n in nodes if n.type != "OUTPUT_MATERIAL"]:
        nodes.remove(node)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    extinction = density_socket(nodes, links, medium)

    # Scattering and absorption coefficients are the extinction weighted by the albedo and its complement
    scatter, absorption = nodes.new("ShaderNodeVolumeScatter"), nodes.new("ShaderNodeVolumeAbsorption")
    scatter.inputs["Color"].default_value = absorption.inputs["Color"].default_value = (1, 1, 1, 1)
    scatter.inputs["Anisotropy"].default_value = medium["anisotropy"]
    for node, weight in ((scatter, medium["albedo"]), (absorption, 1 - medium["albedo"])):
        scale = nodes.new("ShaderNodeMath")
        scale.operation, scale.inputs[1].default_value = "MULTIPLY", weight
        links.new(extinction, scale.inputs[0])
        links.new(scale.outputs[0], node.inputs["Density"])

    mix = nodes.new("ShaderNodeAddShader")
    links.new(scatter.outputs[0], mix.inputs[0])
    links.new(absorption.outputs[0], mix.inputs[1])
    links.new(mix.outputs[0], output.inputs["Volume"])
    return mat


def main(args):
    medium = json.loads(Path(args.medium).read_text())
    scene = bpy.context.scene

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, (args.bottom + args.top) / 2))
    box = bpy.context.active_object
    box.name = "Fog"
    box.scale = (2 * args.half_width, 2 * args.half_width, args.top - args.bottom)
    box.data.materials.append(fog_material(medium))

    scene.cycles.volume_bounces = args.volume_bounces
    scene.cycles.volume_biased = False
    scene.cycles.volume_max_steps = 100_000
    if args.no_adaptive:
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = args.seed
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.output).resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("medium", help="medium JSON file")
    parser.add_argument("output", help="blend file to save the scene with fog to")
    parser.add_argument("--half-width", type=float, default=200.0, help="half extent of the fog box in x and y (m)")
    parser.add_argument("--bottom", type=float, default=-1.0, help="height of the bottom of the fog box (m)")
    parser.add_argument("--top", type=float, default=25.0, help="height of the top of the fog box (m)")
    parser.add_argument("--volume-bounces", type=int, default=0, help="maximum number of volume scattering events")
    parser.add_argument("--no-adaptive", action="store_true", help="disable adaptive sampling, e.g. for references")
    parser.add_argument("--seed", type=int, default=0, help="Cycles sampling seed")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
