"""Time Cycles rendering a frame with visionsim's default settings, as other programs may use the GPU in bursts.

The frame is rendered several times in the same Blender session, each time once the GPU is otherwise idle, without
saving it, and the fastest render is kept. This needs to run from within Blender, e.g.::

    blender -b fog.blend --python time_cycles.py -- --frame 255 --width 800 --height 800 --repeats 12
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService


def wait_for_idle_gpu(threshold: int = 10, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if int(output.split()[0]) < threshold:
            return
        time.sleep(0.2)


def main(args):
    with tempfile.TemporaryDirectory() as root:
        service = BlenderService()
        service.exposed_initialize(bpy.data.filepath, root)
        service.exposed_cycles_settings(
            device_type="optix", use_cpu=False, adaptive_threshold=0.05, max_samples=256, use_denoising=True
        )
        service.exposed_set_resolution(height=args.height, width=args.width)
        service.exposed_move_keyframes(scale=args.keyframe_multiplier)
        bpy.context.scene.frame_set(args.frame)
        bpy.ops.render.render()  # compiles kernels and builds the scene, which is only done once
        times = []
        for _ in range(args.repeats):
            wait_for_idle_gpu()
            start = time.perf_counter()
            bpy.ops.render.render()
            times.append(time.perf_counter() - start)
        print(f"CYCLES_TIME min={min(times):.3f}s median={statistics.median(times):.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=int, default=255, help="frame to render, after rescaling keyframes")
    parser.add_argument("--keyframe-multiplier", type=float, default=5.0, help="same as the render CLI's option")
    parser.add_argument("--width", type=int, default=800, help="frame width in pixels")
    parser.add_argument("--height", type=int, default=800, help="frame height in pixels")
    parser.add_argument("--repeats", type=int, default=12, help="number of timed renders")
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1 :]))
