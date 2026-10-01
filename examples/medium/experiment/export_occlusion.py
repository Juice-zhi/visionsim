"""Render the shadow maps of a scene, as ``--config.include-occlusion`` does when rendering.

This needs to run from within Blender, e.g.::

    blender -b scene.blend --python export_occlusion.py -- occlusion.npz --exclude Plane
"""

import argparse
import sys
import tempfile
import time

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="path of the .npz file")
    parser.add_argument("--exclude", nargs="*", default=[], help="objects that do not cast shadows, e.g. the ground")
    parser.add_argument("--device", default="cpu", help="Cycles device, e.g. cpu or optix")
    parser.add_argument("--sun-resolution", type=int, default=2048, help="texels along the longest side of sun maps")
    parser.add_argument("--sky-resolution", type=int, default=512, help="texels along the longest side of sky maps")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    service = BlenderService()
    with tempfile.TemporaryDirectory() as root:
        service.exposed_initialize(bpy.data.filepath, root)
        service.exposed_cycles_settings(device_type=args.device, use_cpu=args.device.lower() == "cpu")
        start = time.perf_counter()
        service.exposed_save_occlusion(
            args.output,
            exclude=args.exclude,
            sun_resolution=args.sun_resolution,
            sky_resolution=args.sky_resolution,
        )
        print(f"OCCLUSION_TIME {time.perf_counter() - start:.2f}s", flush=True)
