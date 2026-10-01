"""Participating media, such as fog, haze or smoke, added to rendered frames in closed form.

Instead of volumetric path tracing, which is slow and noisy, the medium is added after rendering, using the
depth and linear radiance of the scene. The medium is described by a small set of physical parameters
(:class:`Medium`), and lit by the same lights as the scene (:class:`Lighting`). Its effect on each pixel is
computed in closed form, which yields a unique, noise-free solution that all sensor emulators can share, including
active sensors whose time-resolved measurements are given by :func:`flash_transient`. Objects can cast shadows onto
the medium, through shadow maps of the scene loaded with :func:`load_occlusion`, and the shadows traced along the
rays of a frame can be reused for several media (:func:`trace_shadows`). The same model can also be integrated by
ray marching (:func:`ray_march_medium`), which serves as a baseline.
"""

from visionsim.medium.model import Blob, HeightFog, Homogeneous, Lighting, Medium, PointLight, Sun, kim_exponent
from visionsim.medium.occlusion import Occlusion, Shadows, load_occlusion
from visionsim.medium.raymarch import ray_march_medium
from visionsim.medium.render import RGB_WAVELENGTHS, MediumResult, apply_medium, camera_rays, trace_shadows
from visionsim.medium.transient import Flash, Transient, capture_histogram, estimate_distance, flash_transient

__all__ = [
    "RGB_WAVELENGTHS",
    "Blob",
    "Flash",
    "HeightFog",
    "Homogeneous",
    "Lighting",
    "Medium",
    "MediumResult",
    "Occlusion",
    "PointLight",
    "Shadows",
    "Sun",
    "Transient",
    "apply_medium",
    "camera_rays",
    "capture_histogram",
    "estimate_distance",
    "flash_transient",
    "kim_exponent",
    "load_occlusion",
    "ray_march_medium",
    "trace_shadows",
]
