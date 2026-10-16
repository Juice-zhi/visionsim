"""Render a fog scene with Cycles' Monte Carlo path tracing at a given sampling budget, for the acceleration study.

Saves the combined image and the volume direct/indirect passes, and, with ``--denoise``, the image denoised by
OpenImageDenoise alongside the noisy one, and with ``--normals`` the normals of surfaces. Path guiding (CPU only), the
light tree and the number of volume bounces can be set, and the time taken by each frame is printed. Surfaces can be
made black, so that the fog's light scattered more than once leaves out light reflected by surfaces, and Cycles'
clamping of samples can be turned off. This needs to run from within Blender, e.g.::

    blender -b fog_ref.blend --python render_mc.py -- output/ --samples 64 --frame 255 345 --volume-bounces 16 \\
        --device cpu --threads 12 --denoise --guiding
"""

import argparse
import sys
import time

import bpy  # type: ignore
from visionsim.simulate.blender import BlenderService


def black_surfaces():
    """Make every surface a black diffuse one, which still casts shadows but reflects no light, so that the fog is
    only lit by lights, directly or through itself. Materials with a volume, i.e. the fog's, are kept."""
    black = bpy.data.materials.new("Black")
    black.use_nodes = True
    nodes = black.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    black.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    def has_volume(material) -> bool:
        return (
            material is not None
            and material.use_nodes
            and any(
                node.bl_idname == "ShaderNodeOutputMaterial" and node.inputs["Volume"].is_linked
                for node in material.node_tree.nodes
            )
        )

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            for slot in obj.material_slots:
                if not has_volume(slot.material):
                    slot.material = black


def main(args):
    service = BlenderService()
    service.exposed_initialize(bpy.data.filepath, args.output)
    service.exposed_cycles_settings(
        device_type=args.device, use_cpu=args.device == "cpu", max_samples=args.samples, use_denoising=args.denoise
    )
    scene = service.scene
    scene.render.threads_mode, scene.render.threads = "FIXED", args.threads
    scene.cycles.use_adaptive_sampling = args.adaptive is not None
    if args.adaptive is not None:
        scene.cycles.adaptive_threshold = args.adaptive
    scene.cycles.seed = args.seed
    scene.cycles.denoising_use_gpu = args.gpu_denoise
    scene.cycles.use_light_tree = not args.no_light_tree
    scene.cycles.use_guiding = args.guiding
    if args.guiding:
        scene.cycles.use_surface_guiding = scene.cycles.use_volume_guiding = True
        scene.cycles.guiding_training_samples = args.training_samples
    if args.volume_bounces is not None:
        scene.cycles.volume_bounces = args.volume_bounces
        scene.cycles.max_bounces = max(scene.cycles.max_bounces, args.volume_bounces + 12)
    if args.width or args.height:
        service.exposed_set_resolution(height=args.height, width=args.width)
    if args.no_clamp:
        scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0.0
    if args.black_surfaces:
        black_surfaces()

    service.exposed_include_frames(file_format="OPEN_EXR", bit_depth=32, exr_codec="ZIP")
    service.view_layer.cycles.use_pass_volume_direct = True
    service.view_layer.cycles.use_pass_volume_indirect = True
    outputs = {"Volume Direct": "volume/direct", "Volume Indirect": "volume/indirect"}
    if args.denoise:
        # The combined image is denoised, and the noisy one is kept alongside it
        service.view_layer.cycles.denoising_store_passes = True
        outputs["Noisy Image"] = "noisy"
    for name, subpath in outputs.items():
        socket = service.render_layers.outputs[name]
        service._include_output(subpath, socket, file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)

    if args.normals:
        # Saved as VisionSIM saves them, see common.day_scene for how they are read
        service.exposed_include_normals(preview=False, exr_codec="ZIP")

    service.exposed_move_keyframes(scale=args.keyframe_multiplier)
    for frame in args.frame:
        start = time.perf_counter()
        service.exposed_render_frame(frame)
        print(f"FRAME_TIME frame={frame} seconds={time.perf_counter() - start:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="directory in which to save the passes")
    parser.add_argument(
        "--frame", type=int, nargs="+", required=True, help="frames to render, after rescaling keyframes"
    )
    parser.add_argument("--keyframe-multiplier", type=float, default=5.0, help="same as the render CLI's option")
    parser.add_argument("--samples", type=int, default=64, help="samples per pixel")
    parser.add_argument("--adaptive", type=float, help="noise threshold of adaptive sampling, off by default")
    parser.add_argument("--seed", type=int, default=0, help="Cycles sampling seed")
    parser.add_argument("--device", default="cpu", help="Cycles device type")
    parser.add_argument("--threads", type=int, default=12, help="number of CPU threads")
    parser.add_argument("--volume-bounces", type=int, help="maximum number of volume scattering events")
    parser.add_argument("--denoise", action="store_true", help="denoise with OpenImageDenoise, keeping the noisy image")
    parser.add_argument("--gpu-denoise", action="store_true", help="denoise on the GPU rather than the CPU")
    parser.add_argument("--guiding", action="store_true", help="path guiding of surfaces and volumes (CPU only)")
    parser.add_argument("--training-samples", type=int, default=128, help="samples over which path guiding trains")
    parser.add_argument("--no-light-tree", action="store_true", help="sample lights without the light tree")
    parser.add_argument("--black-surfaces", action="store_true", help="surfaces reflect no light, see black_surfaces")
    parser.add_argument("--no-clamp", action="store_true", help="turn off Cycles' clamping of samples")
    parser.add_argument("--normals", action="store_true", help="also save the normals of surfaces")
    parser.add_argument("--width", type=int, help="frame width in pixels, defaults to the scene's")
    parser.add_argument("--height", type=int, help="frame height in pixels, defaults to the scene's")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
