from __future__ import annotations

import functools
import inspect
import math
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable


import numpy as np
import numpy.typing as npt
import pint
import torch
from pint import Quantity, UnitRegistry, set_application_registry
from ruamel.yaml import YAML
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
from scipy.constants import c  # Top-level export for unit tests
import torch.nn.functional as F

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
Q_ = ureg.Quantity
set_application_registry(ureg)

radiance = ureg.watts / ureg.steradian / ureg.meter**2
irradiance = ureg.watts / ureg.meter**2
radiance_photons = ureg.count / ureg.steradian / ureg.meter**2
irradiance_photons = ureg.count / ureg.meter**2

# Blender writes a depth of 1e10 m for rays that miss every object. Depths (in meters) at or beyond this cutoff,
# like non-finite and non-positive depths, are treated as no laser return instead of a surface to bin.
DEPTH_CUTOFF = 1e9

# Relative tolerance when checking max_depth, or the histogram's window, against the unambiguous range
# c / (2 * pulse_repetition), so that a value equal to that range is not rejected because of floating-point rounding.
MAX_DEPTH_RTOL = 1e-6

# Relative tolerance when checking the intrinsics of a dataset item against those of the sensor's field of view (see
# preproc_frames). A focal length off by 1e-3 changes a field of view of 60 deg by 0.05 deg.
INTRINSICS_RTOL = 1e-3

# Accepted values of preproc_frames' depth_type. "planar_z" is the distance along the optical axis, which is what
# Blender's Z pass (a rendered depths/ directory) holds, and "range" is the distance along each pixel's ray.
DEPTH_TYPES = ("planar_z", "range")

# Accepted values of preproc_normals' normal_type, the frame a normal map is expressed in. "rendered" is what VisionSim's
# normal pass (a rendered normals/ directory) holds, "camera" is the camera frame and "world" the world frame.
NORMAL_TYPES = ("rendered", "camera", "world")


def validate_units(enum_aware=True, ignore_kwargs=None):
    ignore_kwargs = ignore_kwargs or []

    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            signature = inspect.signature(func)
            params = signature.parameters.items()

            params = filter(lambda it: it[0] in kwargs and it[0] not in ignore_kwargs, params)
            params = filter(lambda it: it[1].kind != inspect.Parameter.VAR_POSITIONAL, params)
            params = filter(lambda it: it[1].default != inspect.Parameter.empty, params)

            for param_name, param_value in params:
                value = kwargs.get(param_name)

                if enum_aware and isinstance(value, Enum):
                    if isinstance(param_value.default, Enum):
                        continue
                    value = value.value

                if isinstance(param_value.default, Quantity):
                    if not isinstance(value, Quantity):
                        raise pint.PintTypeError(
                            f"Expected a quantity for keyword-argument '{param_name}' with "
                            f"dimensionality {param_value.default.dimensionality}, got {type(value)}."
                        )
                    if not value.check(param_value.default.dimensionality):
                        raise pint.DimensionalityError(value, param_value.default)

            return func(*args, **kwargs)

        return inner

    return wrapper


@ureg.wraps(ureg.meter, ureg.second)
def tof2depth(t):
    return c * t / 2


@ureg.wraps(ureg.count, (ureg.watt, ureg.second, ureg.meter))
def watts2photons(watts, t, wavelength):
    from scipy.constants import h

    energy = watts * t
    photon_energy = h * c / wavelength
    return energy / photon_energy


def histogram_bin_width(n_bins: int, pulse_repetition: Quantity, bin_width: Quantity | None = None) -> Quantity:
    """Returns the width of a histogram bin as a one-way distance (c * t / 2), in meters.

    It is ``bin_width`` when given, and otherwise one laser period, the unambiguous range
    ``c / (2 * pulse_repetition)``, divided into ``n_bins`` bins. The histogram's window, ``n_bins * bin_width``, may be
    shorter than one laser period, in which case returns that arrive after it are not recorded, but not longer, since
    it would then also record the returns of the next pulse.

    Raises:
        ValueError: if ``bin_width`` is not positive, or if the window exceeds one laser period.
    """
    period = tof2depth(1 / pulse_repetition).to(ureg.meter)
    if bin_width is None:
        return period / n_bins

    bin_width_m = bin_width.to(ureg.meter)
    if not bin_width_m.magnitude > 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}.")

    window = n_bins * bin_width_m
    if window > period * (1 + MAX_DEPTH_RTOL):
        raise ValueError(
            f"The histogram window, n_bins x bin_width = {n_bins} x {bin_width_m:~P} = {window:.6g~P}, exceeds one "
            f"laser period, c / (2 * pulse_repetition) = {period:.6g~P} at {pulse_repetition:.6g~P}. Use fewer or "
            "narrower bins, or a lower pulse_repetition."
        )
    return bin_width_m


def is_whole_period(window: float | Quantity, period: float | Quantity) -> bool:
    """Whether a histogram window is one whole laser period, up to floating-point rounding (see MAX_DEPTH_RTOL).

    ``window`` and ``period`` are both floats in the same units, or both quantities. A window that is shorter by more
    than the rounding is not the whole period, however little it is short by: the returns after it are not recorded.
    """
    return bool(abs(window - period) <= MAX_DEPTH_RTOL * period)


@ureg.wraps(ureg.radian, (ureg.meter, ureg.meter))
def fov_from_focal_length(fl, d):
    return 2 * np.arctan2(d, 2 * fl)


@ureg.wraps(ureg.meter, (ureg.radian, ureg.meter))
def focal_length_from_fov(fov, d):
    return d / (2 * np.tan(fov / 2))


@ureg.wraps(ureg.steradian, (ureg.radian, ureg.radian))
def pyramid_solid_angle(a, b):
    return 4 * np.arcsin(np.sin(a / 2) * np.sin(b / 2))


def get_irradiance_with_fov(irradiance, sensor_fov, pixel_fov, omega, w, h):
    """Rescales the irradiance of a zone's pixels by the mean solid angle of a pixel in a zone of the same angular
    size centered on the optical axis, relative to omega, the sensor's mean pixel solid angle.

    The factor depends on the zone's size but not on its position. It stands in for the solid angles of the pixels
    when the sensor has no ``falloff`` (``"none"``); with a falloff, each pixel's irradiance already carries its own
    solid angle and the detector does not rescale zones.
    """
    fov_x, fov_y = sensor_fov[0], sensor_fov[1]
    w_ratio, h_ratio = pixel_fov[3] - pixel_fov[2], pixel_fov[1] - pixel_fov[0]
    w_new, h_new = w * w_ratio, h * h_ratio
    fov_x, fov_y = fov_x * w_ratio, fov_y * h_ratio
    omega_new = pyramid_solid_angle(fov_x, fov_y) / (w_new * h_new)
    scale = (omega_new / omega).to(ureg.dimensionless).magnitude
    return irradiance * scale


def planar_depth_to_range(
    depth: npt.ArrayLike | torch.Tensor | Quantity, fl_x: float, fl_y: float, cx: float, cy: float
) -> np.ndarray | torch.Tensor | Quantity:
    """Converts planar depth, the distance along the optical axis, to range, the distance along each pixel's ray.

    The time of flight and the falloff of a laser return depend on the range, while a rendered ``depths/`` directory
    holds Blender's Z pass, i.e. planar depth. Each pixel is scaled by the length of its ray
    ``K^-1 [u + 0.5, v + 0.5, 1]`` through the pixel center, so ``range = Z / cos(theta)`` with ``theta`` the pixel's
    angle to the optical axis. The intrinsics are in pixels, with the origin at the corner of the image as in a
    dataset's transforms.

    Args:
        depth: planar depths of shape ``(..., H, W)``, as an array, a tensor (gradients are kept) or a Pint quantity.
        fl_x: focal length along the image width, in pixels.
        fl_y: focal length along the image height, in pixels.
        cx: principal point along the image width, in pixels.
        cy: principal point along the image height, in pixels.

    Returns:
        The ranges, as the same type (and units) with the same shape. Values that mean no return (not finite, not
        positive, or at least DEPTH_CUTOFF) still do.
    """
    if isinstance(depth, Quantity):
        return ureg.Quantity(planar_depth_to_range(depth.magnitude, fl_x, fl_y, cx, cy), depth.units)

    if not isinstance(depth, torch.Tensor):
        depth = np.asarray(depth)

    h, w = depth.shape[-2:]
    x = (np.arange(w) + 0.5 - cx) / fl_x
    y = (np.arange(h) + 0.5 - cy) / fl_y
    ray_lengths = np.sqrt(1.0 + x[np.newaxis, :] ** 2 + y[:, np.newaxis] ** 2)

    if isinstance(depth, torch.Tensor):
        dtype = depth.dtype if depth.is_floating_point() else torch.float64
        return depth * torch.as_tensor(ray_lengths, dtype=dtype, device=depth.device)
    return depth * ray_lengths


def camera_ray_directions(h: int, w: int, fl_x: float, fl_y: float, cx: float, cy: float) -> np.ndarray:
    """Unit directions of the rays through the pixel centers of an ``(h, w)`` image, as an array of shape ``(h, w, 3)``.

    They are in the camera frame of VisionSim's camera poses (``transform_matrix``) and normal maps: +X points right,
    +Y up, and the camera looks along -Z (OpenGL / Blender convention). The intrinsics are in pixels, with the origin at
    the corner of the image as in a dataset's transforms, so ``cos(theta) = -z`` with ``theta`` the angle of a ray to
    the optical axis.
    """
    x = (np.arange(w) + 0.5 - cx) / fl_x
    y = (np.arange(h) + 0.5 - cy) / fl_y
    rays = np.stack(np.broadcast_arrays(x[np.newaxis, :], -y[:, np.newaxis], -1.0), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def incidence_cosine(
    normals: npt.ArrayLike | torch.Tensor | Quantity, fl_x: float, fl_y: float, cx: float, cy: float
) -> np.ndarray | torch.Tensor:
    """Returns ``|cos|`` of the angle between each pixel's surface normal and its ray, which scales the light a
    Lambertian surface receives from a source next to the camera, such as an active sensor's laser.

    Args:
        normals: surface normals of shape ``(..., H, W, 3)`` in the camera frame of :func:`camera_ray_directions`, as
            an array, a tensor (gradients are kept) or a dimensionless Pint quantity. They need not have unit length,
            and either side of the surface may face the camera.
        fl_x: focal length along the image width, in pixels.
        fl_y: focal length along the image height, in pixels.
        cx: principal point along the image width, in pixels.
        cy: principal point along the image height, in pixels.

    Returns:
        The cosines, of shape ``(..., H, W)``, as a float64 array or a floating-point tensor. A pixel whose normal is
        zero or not finite, as where a ray misses every object, gets 0.
    """
    if isinstance(normals, Quantity):
        normals = normals.to(ureg.dimensionless).magnitude

    is_tensor = isinstance(normals, torch.Tensor)
    values = normals if isinstance(normals, torch.Tensor) else torch.tensor(np.asarray(normals, dtype=np.float64))
    dtype = values.dtype if values.is_floating_point() else torch.float64
    h, w = values.shape[-3:-1]
    rays = torch.as_tensor(camera_ray_directions(h, w, fl_x, fl_y, cx, cy), dtype=dtype, device=values.device)

    # Masked before the division, so that pixels without a normal do not produce NaN gradients
    values = torch.where(torch.isfinite(values).all(dim=-1, keepdim=True), values.to(dtype), 0.0)
    length_sq = (values * values).sum(dim=-1)
    valid = length_sq > 0
    cosine = (values * rays).sum(dim=-1).abs() / torch.sqrt(torch.where(valid, length_sq, 1.0))
    cosine = torch.where(valid, cosine, 0.0).clamp(max=1.0)
    return cosine if is_tensor else cosine.numpy()


def _normals_to_camera_frame(normals: np.ndarray, transform: Any, normal_type: str) -> np.ndarray:
    """Expresses normals of shape ``(..., 3)`` in the camera frame, see :func:`preproc_normals`."""
    if normal_type == "camera":
        return normals

    pose = transform.get("transform_matrix") if isinstance(transform, dict) else None
    if pose is None:
        raise ValueError(
            f"normal_type {normal_type!r} needs the camera pose (transform_matrix) of the normal frame, which only a "
            "dataset item has. Pass normals that are in the camera frame with normal_type='camera'."
        )
    rotation = np.asarray(pose, dtype=np.float64)[:3, :3]

    # Row vectors: n @ R is R^T n, the world-to-camera rotation of a world-frame normal
    if normal_type == "world":
        return normals @ rotation
    return normals @ rotation @ rotation


def _item_transform(frame: Any) -> Any:
    """The transform of a ``(data, transform)`` item of a ``Dataset``, or None for an array or tensor."""
    return frame[1] if isinstance(frame, (tuple, list)) and len(frame) == 2 else None


def _item_intrinsics(
    transform: Any, shape: tuple[int, ...], sensor_config: dict, name: str = "depth"
) -> tuple[float, float, float, float] | None:
    """Returns the pinhole intrinsics ``(fl_x, fl_y, cx, cy)``, in pixels, of a frame of shape ``(H, W)`` from its
    transform, or None if the transform has none (an array, or a dataset item without intrinsics).

    The emulator resizes every frame to the sensor, and weights each pixel by its off-axis falloff and incidence cosine
    along the rays of the sensor's field of view (:meth:`~visionsim.emulate.aspc.sensors.SensorBase.frame_intrinsics`),
    so a frame must span that field of view: its intrinsics must be the sensor's, scaled to the frame, to within a
    relative INTRINSICS_RTOL.

    Raises:
        ValueError: if the transform's size is not the frame's, or its intrinsics are not those of the sensor.
    """
    if not (isinstance(transform, dict) and all(transform.get(k) is not None for k in ("fl_x", "fl_y", "cx", "cy"))):
        return None
    h, w = shape
    if transform.get("h") is not None and transform.get("w") is not None:
        if (int(transform["h"]), int(transform["w"])) != (h, w):
            raise ValueError(
                f"{name.capitalize()} frame of shape {(h, w)} does not match the size (h={transform['h']}, "
                f"w={transform['w']}) of its transform, so the transform's intrinsics do not apply to it."
            )
    intrinsics = float(transform["fl_x"]), float(transform["fl_y"]), float(transform["cx"]), float(transform["cy"])

    # Imported here since the sensors module imports this one
    from visionsim.emulate.aspc.sensors import SPADSensor

    sensor = SPADSensor(**sensor_config)
    expected = sensor.frame_intrinsics((h, w))
    if not np.allclose(intrinsics, expected, rtol=INTRINSICS_RTOL, atol=0):
        fov_x = math.degrees(2 * math.atan(w / 2 / intrinsics[0]))
        fov_y = math.degrees(2 * math.atan(h / 2 / intrinsics[1]))
        raise ValueError(
            f"The intrinsics (fl_x, fl_y, cx, cy) = ({', '.join(f'{v:.6g}' for v in intrinsics)}) of the {h}x{w} "
            f"{name} frame, a field of view of {fov_x:.4g} x {fov_y:.4g} deg, are not those of the sensor's field of "
            f"view of {sensor.fov_x.magnitude:.4g} x {sensor.fov_y.magnitude:.4g} deg at that size, "
            f"({', '.join(f'{v:.6g}' for v in expected)}). The emulator gives each pixel the falloff and incidence "
            "cosine of the sensor's rays, so a frame must span the sensor's field of view, centered on its optical "
            "axis: set sensor.fov to the field of view of the render."
        )
    return intrinsics


def _frame_intrinsics(transform: Any, shape: tuple[int, ...], sensor_config: dict) -> tuple[float, float, float, float]:
    """Returns the pinhole intrinsics ``(fl_x, fl_y, cx, cy)``, in pixels, of a depth frame of shape ``(H, W)``.

    They are taken from the frame's transform when it has them (a dataset item, checked against the sensor by
    :func:`_item_intrinsics`), and otherwise from the emulated sensor, whose field of view the frame is resized to.
    """
    intrinsics = _item_intrinsics(transform, shape, sensor_config)
    if intrinsics is not None:
        return intrinsics

    # Imported here since the sensors module imports this one
    from visionsim.emulate.aspc.sensors import SPADSensor

    h, w = shape
    return SPADSensor(**sensor_config).frame_intrinsics((h, w))


def preproc_frames(
    depth_frame: np.ndarray | torch.Tensor,
    albedo_frame: np.ndarray | torch.Tensor,
    ambient_frame: np.ndarray | torch.Tensor,
    config: dict,
    device: str | torch.device = "cpu",
    requires_grad: bool = False,
    depth_type: str = "planar_z",
) -> tuple[Quantity, Quantity, Quantity]:
    """Preprocesses a single frame triplet (depth, albedo, ambient) into PyTorch Tensors

    wrapped in Pint quantities. Each frame is an array or tensor, or a ``(data, transform)`` item of a ``Dataset``.

    The emulator bins and attenuates each pixel's laser return by its range, so the depth frame is returned as range.
    ``depth_type`` says what the depth frame holds:

    - ``"planar_z"`` (default): planar depth, the distance along the optical axis, which is what a rendered
      ``depths/`` directory holds. It is converted to range with :func:`planar_depth_to_range`, using the intrinsics
      ``fl_x, fl_y, cx, cy`` of the item's transform, or, for a frame without them, those of the sensor in
      ``config``, scaled to the frame's resolution (the frame then spans the sensor's field of view).
    - ``"range"``: range, used as is.

    Every frame is resized to the sensor, and the emulator gives each pixel the off-axis falloff and incidence cosine
    of the sensor's field of view, so the frames must span that field of view. An item whose transform has intrinsics
    (depth, albedo or ambient) must therefore have the sensor's, scaled to its resolution, to within a relative
    INTRINSICS_RTOL: a render with another field of view, or with its principal point off the center, raises a
    ValueError, as does a transform whose size is not that of its frame.

    Depths that are not finite, not positive, or at least DEPTH_CUTOFF (rays that miss every object) are set to 0,
    which the histogrammer treats as no laser return.

    ``histogrammer.max_depth`` in ``config``, the largest depth expected in the scene, is optional. When set, it must
    not exceed the unambiguous range ``c / (2 * pulse_repetition)``, and it is the depth of the value 255 of an 8-bit
    depth image, which needs it.
    """
    if depth_type not in DEPTH_TYPES:
        raise ValueError(f"Unknown depth_type {depth_type!r}, expected one of {DEPTH_TYPES}.")

    Nr, Nc = config["sensor"]["size"]

    # 1. Validate max depth against laser pulse repetition rate
    pulse_rep = config["active_source"]["pulsed_laser"]["pulse_repetition"]
    if not hasattr(pulse_rep, "to"):
        pulse_rep = pulse_rep * ureg.hertz

    tmax = (1.0 / pulse_rep).to(ureg.second)
    max_resolvable_depth = (tmax * (c * ureg.meter / ureg.second) / 2).to(ureg.meter)

    max_depth = None
    scale_depth = config.get("histogrammer", {}).get("max_depth")
    if scale_depth is not None:
        if not hasattr(scale_depth, "to"):
            scale_depth = scale_depth * ureg.meter
        else:
            scale_depth = scale_depth.to(ureg.meter)

        if scale_depth > max_resolvable_depth * (1 + MAX_DEPTH_RTOL):
            raise ValueError(
                f"histogrammer.max_depth ({scale_depth}) exceeds max resolvable depth ({max_resolvable_depth})."
            )
        max_depth = scale_depth.magnitude

    def _to_numpy(frame):
        # Dataset.__getitem__ returns (data, transform); image array is at index 0
        if isinstance(frame, (tuple, list)) and len(frame) == 2:
            frame = frame[0]

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()

        return np.asarray(frame)

    # 2. Process Depth Frame
    raw_depth = _to_numpy(depth_frame)
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[:, :, 0]

    if np.issubdtype(raw_depth.dtype, np.integer):
        if max_depth is None:
            raise ValueError("An 8-bit depth image needs histogrammer.max_depth, the depth of its value 255.")
        depth_img = raw_depth.astype(np.float32) / 255.0 * float(max_depth)
    else:
        depth_img = raw_depth.astype(np.float32)

    # Missed rays and invalid depths become 0 (no return), see DEPTH_CUTOFF
    depth_img = np.where(np.isfinite(depth_img) & (depth_img > 0) & (depth_img < DEPTH_CUTOFF), depth_img, 0.0)

    # Planar depth to range, with the intrinsics of the frame at its own resolution (before resizing). A dataset item's
    # intrinsics must be the sensor's whatever the depth type, as the falloff and incidence cosine use the sensor's
    transform = _item_transform(depth_frame)
    if depth_type == "planar_z":
        depth_img = planar_depth_to_range(depth_img, *_frame_intrinsics(transform, depth_img.shape, config["sensor"]))
    else:
        _item_intrinsics(transform, depth_img.shape, config["sensor"])

    # 3. Process Albedo Frame (Diffuse Color Pass)
    raw_albedo = _to_numpy(albedo_frame)
    if raw_albedo.ndim == 3:
        raw_albedo = raw_albedo[:, :, 0]
    _item_intrinsics(_item_transform(albedo_frame), raw_albedo.shape, config["sensor"], "albedo")

    if np.issubdtype(raw_albedo.dtype, np.integer):
        albedo_img = raw_albedo.astype(np.float32) / 255.0
    else:
        albedo_img = raw_albedo.astype(np.float32)

    # 4. Process Ambient Frame (Full RGB Render Pass)
    raw_ambient = _to_numpy(ambient_frame)
    if raw_ambient.ndim == 3:
        raw_ambient = raw_ambient[:, :, 0]
    _item_intrinsics(_item_transform(ambient_frame), raw_ambient.shape, config["sensor"], "ambient")

    if np.issubdtype(raw_ambient.dtype, np.integer):
        ambient_img = raw_ambient.astype(np.float32) / 255.0
    else:
        ambient_img = raw_ambient.astype(np.float32)

    # 5. Convert to PyTorch Tensors and Resize with F.interpolate
    # Shape requirements: F.interpolate expects (Batch, Channel, Height, Width) -> (1, 1, H, W)
    depth_tensor = torch.tensor(depth_img, device=device, dtype=torch.float64, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)
    albedo_tensor = torch.tensor(albedo_img, device=device, dtype=torch.float64, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)
    ambient_tensor = torch.tensor(ambient_img, device=device, dtype=torch.float64, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)

    # Apply interpolation: 'nearest' for depth, 'bilinear' for smooth color/ambient maps
    depth_tensor = F.interpolate(depth_tensor, size=(Nr, Nc), mode="nearest").squeeze(0).squeeze(0)
    albedo_tensor = F.interpolate(albedo_tensor, size=(Nr, Nc), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    ambient_tensor = F.interpolate(ambient_tensor, size=(Nr, Nc), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    depth_qty = ureg.Quantity(depth_tensor, "meter")
    albedo_qty = ureg.Quantity(albedo_tensor, "dimensionless")
    ambient_qty = ureg.Quantity(ambient_tensor, "dimensionless")

    return depth_qty, albedo_qty, ambient_qty


def preproc_normals(
    normal_frame: np.ndarray | torch.Tensor,
    config: dict,
    device: str | torch.device = "cpu",
    requires_grad: bool = False,
    normal_type: str = "rendered",
) -> Quantity:
    """Preprocesses a normal frame into unit normals in the camera frame, resized to the sensor like the depth frame
    of :func:`preproc_frames`, for ``ASPCEmulator.process_frame``.

    The frame is an ``(H, W, 3)`` array or tensor, or a ``(data, transform)`` item of a ``Dataset``. The camera frame
    is that of VisionSim's camera poses: +X points right, +Y up, and the camera looks along -Z. With ``R`` the
    camera-to-world rotation of the item's ``transform_matrix``, ``normal_type`` says what the frame holds:

    - ``"rendered"`` (default): a normal map from VisionSim's Blender normal pass (``include_normals``, a rendered
      ``normals/`` directory). Its compositor drivers index ``camera.matrix_world`` column-major, so the pass holds
      ``R @ n_world`` instead of the camera-frame normal ``R.T @ n_world`` that the documentation describes. It is
      converted with ``R.T @ R.T``.
    - ``"camera"``: normals in the camera frame, used as is.
    - ``"world"``: normals in the world frame, converted with ``R.T``.

    Normals are scaled to unit length, and normals that are zero or not finite (rays that miss every object) are set
    to 0, for which the emulator gives no laser return.

    The emulator takes each pixel's incidence cosine along the rays of the sensor's field of view, so an item whose
    transform has intrinsics must have the sensor's, as in :func:`preproc_frames`, or a ValueError is raised.

    Returns:
        A dimensionless quantity holding a float64 tensor of shape ``(H, W, 3)`` at the sensor's size.
    """
    if normal_type not in NORMAL_TYPES:
        raise ValueError(f"Unknown normal_type {normal_type!r}, expected one of {NORMAL_TYPES}.")

    Nr, Nc = config["sensor"]["size"]

    transform = None
    if isinstance(normal_frame, (tuple, list)) and len(normal_frame) == 2:
        normal_frame, transform = normal_frame
    if hasattr(normal_frame, "detach"):
        normal_frame = normal_frame.detach().cpu().numpy()

    raw_normals = np.asarray(normal_frame, dtype=np.float64)
    if raw_normals.ndim == 3 and raw_normals.shape[-1] == 1:
        # Dataset.load_data collapses an EXR whose channels are all equal (e.g. a frame without any hit) to one channel
        raw_normals = np.repeat(raw_normals, 3, axis=-1)
    if raw_normals.ndim != 3 or raw_normals.shape[-1] < 3:
        raise ValueError(f"Expected a normal frame of shape (H, W, 3), got shape {raw_normals.shape}.")
    # The incidence cosine is taken along the sensor's rays, see preproc_frames
    _item_intrinsics(transform, raw_normals.shape[:2], config["sensor"], "normal")

    normals = _normals_to_camera_frame(raw_normals[..., :3], transform, normal_type)

    normals = np.where(np.isfinite(normals).all(axis=-1, keepdims=True), normals, 0.0)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(length > 0, normals / np.where(length > 0, length, 1.0), 0.0)

    # Resized 'nearest', like the depth frame, so that each pixel keeps the normal of the surface its depth comes from
    normal_tensor = torch.tensor(normals, device=device, dtype=torch.float64).permute(2, 0, 1).unsqueeze(0)
    normal_tensor = F.interpolate(normal_tensor, size=(Nr, Nc), mode="nearest").squeeze(0).permute(1, 2, 0)
    normal_tensor = normal_tensor.contiguous().requires_grad_(requires_grad)

    return ureg.Quantity(normal_tensor, "dimensionless")


def _validate_config(config: dict) -> None:
    errors = []

    expected_dimensionality = {
        ("histogrammer", "min_depth"): ureg.meter.dimensionality,
        ("histogrammer", "max_depth"): ureg.meter.dimensionality,
        ("histogrammer", "bin_width"): ureg.meter.dimensionality,
        ("histogrammer", "time_offset"): ureg.second.dimensionality,
        ("histogrammer", "dead_time"): ureg.second.dimensionality,
        ("active_source", "pulsed_laser", "wavelength"): ureg.meter.dimensionality,
        ("active_source", "pulsed_laser", "pulse_repetition"): ureg.hertz.dimensionality,
        ("active_source", "pulsed_laser", "pulse_width"): ureg.second.dimensionality,
        ("active_source", "pulsed_laser", "avg_watts"): ureg.watt.dimensionality,
        ("ambient_source", "sun", "temperature"): ureg.kelvin.dimensionality,
        ("ambient_source", "sun", "lambda_pass"): ureg.meter.dimensionality,
        ("ambient_source", "sun", "delta_lambda"): ureg.meter.dimensionality,
        ("ambient_source", "sun", "intensity"): (ureg.watt / ureg.meter**2).dimensionality,
        ("sensor", "pixel_pitch"): ureg.meter.dimensionality,
        ("sensor", "fov"): ureg.degree.dimensionality,
    }

    for key_path, expected_dim in expected_dimensionality.items():
        found, val = _get_nested_val(config, key_path)
        if not found or val is None:
            continue

        param_name = ".".join(key_path)

        if isinstance(val, (tuple, list)):
            for idx, item in enumerate(val):
                if not isinstance(item, Quantity):
                    errors.append(f"'{param_name}[{idx}]' must be a Pint Quantity, got {type(item).__name__}")
                elif not item.check(expected_dim):
                    errors.append(f"'{param_name}[{idx}]' has incorrect dimensionality.")
        elif isinstance(val, Quantity):
            if not val.check(expected_dim):
                errors.append(f"'{param_name}' has incorrect unit dimensionality.")
        else:
            errors.append(f"'{param_name}' must be a Pint Quantity, got {type(val).__name__}")

    if errors:
        raise ValueError("Unit Validation Errors:\n - " + "\n - ".join(errors))

    hist = config.get("histogrammer", {})
    active = config.get("active_source", {}).get("pulsed_laser", {})

    min_depth = hist.get("min_depth")
    max_depth = hist.get("max_depth")
    n_bins = hist.get("n_bins")
    bin_width = hist.get("bin_width")
    pulse_rep = active.get("pulse_repetition")

    if min_depth is not None and min_depth < 0 * ureg.meters:
        errors.append(f"min_depth must be >= 0m, got {min_depth}")

    if max_depth is not None and min_depth is not None and max_depth <= min_depth:
        errors.append(f"max_depth ({max_depth}) must be > min_depth ({min_depth})")

    if pulse_rep is not None and pulse_rep > 0 * ureg.megahertz:
        max_resolvable_depth = tof2depth(1 / pulse_rep)
        if max_depth is not None and max_depth > max_resolvable_depth * (1 + MAX_DEPTH_RTOL):
            errors.append(f"Max depth ({max_depth}) exceeds max resolvable depth ({max_resolvable_depth}).")

        if n_bins is not None:
            try:
                histogram_bin_width(n_bins, pulse_rep, bin_width)
            except ValueError as err:
                errors.append(str(err))

    if errors:
        raise ValueError("Invalid configuration:\n - " + "\n - ".join(errors))


def yaml_constructor(cls_or_func, *extra_args):
    """Builds a YAML tag constructor that applies ``cls_or_func`` to the tagged node.

    A scalar is passed as is (``!Quantity 10 ns``), each item of a sequence is converted on its own
    (``!Quantity [90 degree, 60 degree]`` gives a list of quantities), and a mapping is passed as keyword arguments.
    """

    def constructor(loader, node):
        if isinstance(node, ScalarNode):
            return cls_or_func(loader.construct_scalar(node), *extra_args)
        elif isinstance(node, SequenceNode):
            return [cls_or_func(item, *extra_args) for item in loader.construct_sequence(node, deep=True)]
        elif isinstance(node, MappingNode):
            return cls_or_func(*extra_args, **loader.construct_mapping(node, deep=True))
        return cls_or_func(node.value, *extra_args)

    return constructor


def _load_config(config_path: Path | str | dict | None) -> dict:
    if config_path is None:
        return {}

    if isinstance(config_path, dict):
        return deepcopy(config_path)

    yaml = YAML(typ="safe")
    safe_builtins = {
        "__builtins__": {"list": list, "dict": dict, "tuple": tuple},
        "np": np,
        "math": math,
    }

    def file_tag_constructor(loader, node):
        val = loader.construct_scalar(node)
        file_p = Path(val)
        if config_path and not file_p.is_absolute():
            file_p = Path(config_path).parent / file_p
        return str(file_p)

    yaml.Constructor.add_constructor(tag="!Quantity", constructor=yaml_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=yaml_constructor(eval, safe_builtins))
    yaml.Constructor.add_constructor(tag="!file", constructor=file_tag_constructor)

    with open(config_path, "r") as f:
        return yaml.load(f) or {}


def _to_quantity(val: Any, unit_str: str) -> Any:
    if val is None:
        return None

    if isinstance(val, ureg.Quantity):
        if hasattr(val.magnitude, "__len__") and not isinstance(val.magnitude, (str, bytes)):
            return [ureg.Quantity(x, val.units) for x in val.magnitude]
        return val

    if isinstance(val, (tuple, list)):
        return [x if isinstance(x, ureg.Quantity) else ureg.Quantity(x, unit_str) for x in val]

    if isinstance(val, (int, float)):
        return ureg.Quantity(val, unit_str)

    if isinstance(val, str):
        try:
            q = ureg.Quantity(val)
            if q.dimensionless:
                return ureg.Quantity(q.magnitude, unit_str)
            return q
        except Exception:
            return ureg.Quantity(float(val), unit_str)

    return val


def _get_nested_val(d: dict, path: tuple[str, ...]) -> tuple[bool, Any]:
    curr = d
    for step in path:
        if isinstance(curr, dict) and step in curr:
            curr = curr[step]
        else:
            return False, None
    return True, curr


def _build_config(
    DEFAULT_CONFIG: dict,
    user_args: dict,
    config_file: Path | str | dict | None = None,
) -> dict:
    yml_config = _load_config(config_file)

    cli_mapping = {
        "min_depth": ("histogrammer", "min_depth"),
        "max_depth": ("histogrammer", "max_depth"),
        "n_bins": ("histogrammer", "n_bins"),
        "bin_width": ("histogrammer", "bin_width"),
        "time_offset": ("histogrammer", "time_offset"),
        "pixel_fov_list": ("histogrammer", "pixel_fov_list"),
        "vignette": ("histogrammer", "vignette"),
        "n_pulses": ("histogrammer", "n_pulses"),
        "dead_time": ("histogrammer", "dead_time"),
        "free_running": ("histogrammer", "free_running"),
        "max_detections_per_cycle": ("histogrammer", "max_detections_per_cycle"),
        "fast_sim": ("histogrammer", "fast_sim"),
        "fast_sim_noise": ("histogrammer", "fast_sim_noise"),
        "active_enabled": ("active_source", "pulsed_laser", "enabled"),
        "active_wavelength": ("active_source", "pulsed_laser", "wavelength"),
        "active_pulse_repetition": ("active_source", "pulsed_laser", "pulse_repetition"),
        "active_pulse_width": ("active_source", "pulsed_laser", "pulse_width"),
        "active_avg_watts": ("active_source", "pulsed_laser", "avg_watts"),
        "active_pulse_shape": ("active_source", "pulsed_laser", "pulse_shape"),
        "active_pulse_shape_custom": ("active_source", "pulsed_laser", "pulse_shape_custom"),
        "ambient_enabled": ("ambient_source", "sun", "enabled"),
        "ambient_temperature": ("ambient_source", "sun", "temperature"),
        "ambient_light_conditions": ("ambient_source", "sun", "light_conditions"),
        "ambient_lambda_pass": ("ambient_source", "sun", "lambda_pass"),
        "ambient_delta_lambda": ("ambient_source", "sun", "delta_lambda"),
        "ambient_intensity": ("ambient_source", "sun", "intensity"),
        "sensor_size": ("sensor", "size"),
        "sensor_pixel_pitch": ("sensor", "pixel_pitch"),
        "sensor_f_number": ("sensor", "f_number"),
        "sensor_fov": ("sensor", "fov"),
        "sensor_falloff": ("sensor", "falloff"),
    }

    cli_param_units = {
        "min_depth": "meters",
        "max_depth": "meters",
        "bin_width": "meters",
        "time_offset": "nanoseconds",
        "dead_time": "nanoseconds",
        "active_wavelength": "nanometers",
        "active_pulse_repetition": "megahertz",
        "active_pulse_width": "nanoseconds",
        "active_avg_watts": "watts",
        "ambient_temperature": "kelvin",
        "ambient_lambda_pass": "nanometers",
        "ambient_delta_lambda": "nanometers",
        "ambient_intensity": "watts / meter**2",
        "sensor_pixel_pitch": "micrometer",
        "sensor_fov": "degree",
    }

    # In a config file, null is a value for these keys (no cap), not a missing entry
    nullable_params = {"max_detections_per_cycle"}

    config: dict = {}

    for param_name, key_path in cli_mapping.items():
        val = user_args.get(param_name)

        if val is None:
            found, yml_val = _get_nested_val(yml_config, key_path)
            if found and (yml_val is not None or param_name in nullable_params):
                val = yml_val
            else:
                _, val = _get_nested_val(DEFAULT_CONFIG, key_path)

        if param_name in cli_param_units:
            val = _to_quantity(val, cli_param_units[param_name])

        d = config
        for step in key_path[:-1]:
            d = d.setdefault(step, {})
        d[key_path[-1]] = val

    # A path given as the pulse shape (e.g. ``--active-pulse-shape irf.txt``) selects a custom pulse with its taps
    pulsed_laser = config["active_source"]["pulsed_laser"]
    if isinstance(pulsed_laser["pulse_shape"], Path):
        pulsed_laser["pulse_shape_custom"] = pulsed_laser["pulse_shape"]
        pulsed_laser["pulse_shape"] = "custom"

    for key in ("input_dir", "output_dir", "force", "pattern"):
        if key in user_args and user_args[key] is not None:
            config[key] = user_args[key]

    return config


def prepare_config(user_args: dict, config_file: Path | str | None = None) -> dict:
    DEFAULT_CONFIG = {
        "histogrammer": {
            "min_depth": 0 * ureg.meters,
            # Unset: no depth limit to check, and the histogram spans one laser period in n_bins bins
            "max_depth": None,
            "n_bins": 672,
            "bin_width": None,
            "time_offset": 0 * ureg.nanoseconds,
            "pixel_fov_list": [[0, 1.0, 0, 1.0]],
            "vignette": False,
            "n_pulses": 10000,
            "dead_time": 10 * ureg.nanoseconds,
            "free_running": True,
            "max_detections_per_cycle": 1,
            "fast_sim": False,
            "fast_sim_noise": False,
        },
        "active_source": {
            "pulsed_laser": {
                "enabled": True,
                "wavelength": 940 * ureg.nanometers,
                "pulse_repetition": 10 * ureg.megahertz,
                "pulse_width": 6 * ureg.nanoseconds,
                "avg_watts": 0.000000007 * ureg.watts,
                "pulse_shape": "gaussian",
                "pulse_shape_custom": None,
            }
        },
        "ambient_source": {
            "sun": {
                "enabled": True,
                "temperature": 5778 * ureg.kelvin,
                "light_conditions": "BRIGHT_SUNLIGHT",
                "lambda_pass": 550 * ureg.nanometers,
                "delta_lambda": 10 * ureg.nanometers,
                "intensity": 3.828e26 * ureg("watts / meter**2"),
            }
        },
        "sensor": {
            "size": (1080, 1920),
            "pixel_pitch": 10 * ureg.micrometer,
            "f_number": 1.4 * ureg.dimensionless,
            "fov": [90.5 * ureg.degree, 59.14 * ureg.degree],
            "falloff": "cos4",
        },
    }
    config = _build_config(DEFAULT_CONFIG, user_args, config_file=config_file)
    _validate_config(config)
    return config
