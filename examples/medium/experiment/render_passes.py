"""Render a scene with fog along with the passes needed to break down where light comes from.

Saves the combined image and the volume direct/indirect passes, i.e. the light scattered by the fog towards the
camera, without denoising or adaptive sampling so that the result is an unbiased reference. This needs to run from
within Blender, e.g.::

    blender -b fog.blend --python render_passes.py -- output/ --samples 4096 [--frame 255]
"""

import argparse
import sys
import time

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService


def main(args):
    service = BlenderService()
    service.exposed_initialize(bpy.data.filepath, args.output)
    service.exposed_cycles_settings(device_type="optix", use_cpu=False, max_samples=args.samples, use_denoising=False)
    service.scene.cycles.use_adaptive_sampling = False
    service.scene.cycles.seed = args.seed

    service.exposed_include_frames(file_format="OPEN_EXR", bit_depth=32, exr_codec="ZIP")
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
        service.exposed_render_frame(args.frame)
        count = 1
    elapsed = time.perf_counter() - start
    print(f"RENDER_TIME frames={count} total={elapsed:.2f}s per_frame={elapsed / count:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="directory in which to save the passes")
    parser.add_argument("--frame", type=int, help="frame to render, after rescaling keyframes, defaults to all")
    parser.add_argument("--keyframe-multiplier", type=float, default=5.0, help="same as the render CLI's option")
    parser.add_argument("--samples", type=int, default=4096, help="samples per pixel")
    parser.add_argument("--seed", type=int, default=0, help="Cycles sampling seed")
    parser.add_argument("--no-volume-passes", dest="volume_passes", action="store_false", help="only save frames")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
