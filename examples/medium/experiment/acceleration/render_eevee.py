"""Render a fog scene with EEVEE's volumetrics, i.e. an off-the-shelf real-time renderer, for the acceleration study.

EEVEE integrates volumes in a grid of froxels aligned with the camera (``--tile`` pixels wide, ``--slices`` deep,
spread between ``--start`` and ``--end`` meters), lit by suns and lamps through their shadow maps when volumetric
shadows are enabled, and by the world through its light probe. Saves the combined image and the light scattered by
volumes (EEVEE's volume light pass), and prints the time taken by each frame. This needs to run from within Blender,
e.g. on the integrated GPU through Vulkan, hiding NVIDIA's driver from Vulkan's loader::

    VK_LOADER_DRIVERS_DISABLE='*nv-vk*' blender -b --gpu-backend vulkan fog_ref.blend --python render_eevee.py -- \\
        output/ --frame 255 345 --tile 8 --slices 64 --end 100
"""

import argparse
import sys
import time

import bpy  # type: ignore
from visionsim.simulate.blender import BlenderService


def main(args):
    service = BlenderService()
    service.exposed_initialize(bpy.data.filepath, args.output)
    scene = service.scene
    scene.render.engine = "BLENDER_EEVEE"
    eevee = scene.eevee
    eevee.volumetric_tile_size = str(args.tile)
    eevee.volumetric_samples = args.slices
    eevee.volumetric_sample_distribution = args.distribution
    eevee.volumetric_start, eevee.volumetric_end = args.start, args.end
    eevee.use_volumetric_shadows = True
    eevee.volumetric_shadow_samples = args.shadow_samples
    eevee.taa_render_samples = args.taa
    gpu = __import__("gpu")
    gpu.init()
    print(f"GPU_DEVICE {gpu.platform.renderer_get()}", flush=True)

    service.exposed_include_frames(file_format="OPEN_EXR", bit_depth=32, exr_codec="ZIP")
    service.view_layer.eevee.use_pass_volume_direct = True
    socket = next(s for s in service.render_layers.outputs if s.name.startswith("Volume") and s.enabled)
    service._include_output("volume/light", socket, file_format="OPEN_EXR", exr_codec="ZIP", bit_depth=32)

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
    parser.add_argument("--tile", type=int, default=8, choices=(1, 2, 4, 8, 16), help="froxel width in pixels")
    parser.add_argument("--slices", type=int, default=64, help="number of froxels along depth")
    parser.add_argument("--distribution", type=float, default=0.8, help="how exponentially slices are spread")
    parser.add_argument("--start", type=float, default=0.1, help="distance where volumetrics start (m)")
    parser.add_argument("--end", type=float, default=100.0, help="distance where volumetrics end (m)")
    parser.add_argument("--shadow-samples", type=int, default=16, help="samples of volumetric shadows")
    parser.add_argument("--taa", type=int, default=64, help="number of jittered samples accumulated per frame")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
