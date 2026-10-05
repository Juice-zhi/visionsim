"""Render a scene with fog along with the passes needed to break down where light comes from.

Saves the combined image and the volume direct/indirect passes, i.e. the light scattered by the fog towards the
camera, without denoising or adaptive sampling so that the result is an unbiased reference. This needs to run from
within Blender, e.g.::

    blender -b fog.blend --python render_passes.py -- output/ --samples 4096 [--frame 255]

With ``--light-groups``, the light of suns and of the sky (the world) are also saved separately, along with world-space
normals and the diffuse albedo, e.g. to model how surfaces are lit through the fog from a render without fog.
"""

import argparse
import sys
import time

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService


def main(args):
    service = BlenderService()
    service.exposed_initialize(bpy.data.filepath, args.output)
    service.exposed_cycles_settings(
        device_type=args.device, use_cpu=args.device == "cpu", max_samples=args.samples, use_denoising=False
    )
    if args.threads:
        service.scene.render.threads_mode, service.scene.render.threads = "FIXED", args.threads
    if not args.clamp:
        # Clamping bright samples biases light scattered more than once near lamps, by default that of indirect light
        service.scene.cycles.sample_clamp_direct = service.scene.cycles.sample_clamp_indirect = 0.0
    service.scene.cycles.use_adaptive_sampling = False
    service.scene.cycles.seed = args.seed
    if args.width or args.height:
        service.exposed_set_resolution(height=args.height, width=args.width)
    if args.volume_bounces is not None:
        # The total number of bounces also limits volume bounces, so leave room for surface bounces too
        service.scene.cycles.volume_bounces = args.volume_bounces
        service.scene.cycles.max_bounces = max(service.scene.cycles.max_bounces, args.volume_bounces + 12)

    service.exposed_include_frames(file_format="OPEN_EXR", bit_depth=32, exr_codec="ZIP")
    if args.light_groups:
        # Light from each sun and from the world (the sky) in separate passes, and world-space normals
        groups = {"sky": [service.scene.world]}
        for obj in service.scene.objects:
            if obj.type == "LIGHT":
                groups.setdefault("sun" if obj.data.type == "SUN" else "lamps", []).append(obj)
        for name, members in groups.items():
            if name not in service.view_layer.lightgroups:
                service.view_layer.lightgroups.add(name=name)
            for member in members:
                member.lightgroup = name
        service.view_layer.use_pass_normal = True
        service.view_layer.use_pass_diffuse_color = True
        for name in groups:
            socket = service.render_layers.outputs[f"Combined_{name}"]
            service._include_output(f"lights/{name}", socket, file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)
        for name, subpath in (("Normal", "normals"), ("Diffuse Color", "albedo")):
            socket = service.render_layers.outputs[name]
            service._include_output(subpath, socket, file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)
    if args.volume_passes:
        service.view_layer.cycles.use_pass_volume_direct = True
        service.view_layer.cycles.use_pass_volume_indirect = True
        for name, subpath in (("Volume Direct", "volume/direct"), ("Volume Indirect", "volume/indirect")):
            socket = service.render_layers.outputs[name]
            service._include_output(subpath, socket, file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)

    service.exposed_move_keyframes(scale=args.keyframe_multiplier)
    start = time.perf_counter()
    if args.frame is None:
        service.exposed_render_animation()
        count = len(service.exposed_animation_range())
    else:
        for frame in args.frame:
            service.exposed_render_frame(frame)
        count = len(args.frame)
    elapsed = time.perf_counter() - start
    print(f"RENDER_TIME frames={count} total={elapsed:.2f}s per_frame={elapsed / count:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="directory in which to save the passes")
    parser.add_argument("--frame", type=int, nargs="+", help="frames to render, after rescaling keyframes, or all")
    parser.add_argument("--keyframe-multiplier", type=float, default=5.0, help="same as the render CLI's option")
    parser.add_argument("--samples", type=int, default=4096, help="samples per pixel")
    parser.add_argument("--width", type=int, help="frame width in pixels, defaults to the scene's")
    parser.add_argument("--height", type=int, help="frame height in pixels, defaults to the scene's")
    parser.add_argument("--seed", type=int, default=0, help="Cycles sampling seed")
    parser.add_argument("--device", default="optix", help="Cycles device type, e.g. optix, cuda or cpu")
    parser.add_argument("--threads", type=int, help="number of CPU threads, defaults to all")
    parser.add_argument(
        "--volume-bounces",
        type=int,
        help="maximum number of volume scattering events, i.e. 0 for single scattering, defaults to the scene's",
    )
    parser.add_argument("--no-volume-passes", dest="volume_passes", action="store_false", help="only save frames")
    parser.add_argument(
        "--no-clamp", dest="clamp", action="store_false", help="don't clamp bright samples, which biases indirect light"
    )
    parser.add_argument(
        "--light-groups", action="store_true", help="also save the light of suns and of the sky, normals and albedo"
    )
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
