"""Participating media, such as fog, haze or smoke, added to rendered frames in closed form.

Instead of volumetric path tracing, which is slow and noisy, the medium is added after rendering, using the
depth and linear radiance of the scene. The medium is described by a small set of physical parameters
(:class:`Medium`), and lit by the same lights as the scene (:class:`Lighting`). Its effect on each pixel is
computed in closed form, which yields a unique, noise-free solution that all sensor emulators can share.
"""

from visionsim.medium.model import Blob, HeightFog, Homogeneous, Lighting, Medium, PointLight, Sun, kim_exponent
from visionsim.medium.render import RGB_WAVELENGTHS, MediumResult, apply_medium, camera_rays

__all__ = [
    "RGB_WAVELENGTHS",
    "Blob",
    "HeightFog",
    "Homogeneous",
    "Lighting",
    "Medium",
    "MediumResult",
    "PointLight",
    "Sun",
    "apply_medium",
    "camera_rays",
    "kim_exponent",
]
