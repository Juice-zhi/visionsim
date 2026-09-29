"""Export the lighting of a scene to a JSON file, as ``--config.include-lighting`` does when rendering.

This needs to run from within Blender, e.g.::

    blender -b scene.blend --python export_lighting.py -- lighting.json
"""

import sys
import tempfile

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService

if __name__ == "__main__":
    service = BlenderService()
    with tempfile.TemporaryDirectory() as root:
        service.exposed_initialize(bpy.data.filepath, root)
        service.exposed_save_lighting(sys.argv[sys.argv.index("--") + 1])
