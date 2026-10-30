"""Radiometry of the laser return: the incidence cosine from a normal frame, each pixel's off-axis falloff, and the
weight of each zone.

A Lambertian surface lit by a laser next to the sensor returns light in proportion to ``|cos|`` of the angle between
its normal and the pixel's ray. By the camera equation, a pixel at an angle theta to the optical axis collects
``cos^4(theta)`` times as much light as a pixel on the axis: ``cos^3(theta)`` for its solid angle and ``cos(theta)``
for the foreshortened aperture. A zone is the mean of its pixels.
"""

import logging

import numpy as np
import pytest
import torch
from scipy.constants import c as C_LIGHT
from scipy.spatial.transform import Rotation

from visionsim.dataset.models import Metadata
from visionsim.emulate.aspc.ambient import LightConditions, Sun
from visionsim.emulate.aspc.detector import Detector, _log_without_normals
from visionsim.emulate.aspc.emitter import PulsedLaser
from visionsim.emulate.aspc.emulator import ASPCEmulator
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import FALLOFF_TYPES, SPADSensor
from visionsim.emulate.aspc.utils import (
    camera_ray_directions,
    get_irradiance_with_fov,
    incidence_cosine,
    planar_depth_to_range,
    prepare_config,
    preproc_frames,
    preproc_normals,
    ureg,
)

RES, FOV_DEG = 33, 60.0  # an odd size, so that one pixel sits on the optical axis
CENTER = (RES // 2, RES // 2)
N_BINS, BIN_M = 64, 0.02
F_REP = C_LIGHT / (2 * N_BINS * BIN_M)  # one laser period is 1.28 m of range
RHO, D0 = 0.5, 0.5
POWER = {"cos4": 4, "cos3": 3, "none": 0}

# A camera pose whose rotation is far from symmetric, so that R, R.T and R @ R all differ
POSE = np.eye(4)
POSE[:3, :3] = Rotation.from_euler("zyx", [35.0, -20.0, 55.0], degrees=True).as_matrix()
POSE[:3, 3] = [0.4, -1.2, 2.0]
ROTATION = POSE[:3, :3]


def _config(size=(RES, RES), fov=(FOV_DEG, FOV_DEG), falloff=None, zones=None):
    """Emulator config with 2 cm bins, a short pulse and no ambient light."""
    sensor = {
        "size": tuple(size),
        "pixel_pitch": 10 * ureg.micrometer,
        "f_number": 1.4,
        "fov": [f * ureg.degree for f in fov],
    }
    if falloff is not None:
        sensor["falloff"] = falloff
    return {
        "histogrammer": {
            "min_depth": 0 * ureg.meter,
            "max_depth": N_BINS * BIN_M * ureg.meter,
            "n_bins": N_BINS,
            "bin_width": BIN_M * ureg.meter,
            "n_pulses": 1000,
            "dead_time": (1 / F_REP) * ureg.second,
            "free_running": False,
            "fast_sim": True,
            "vignette": False,
            "pixel_fov_list": zones if zones is not None else [[0.0, 1.0, 0.0, 1.0]],
        },
        "active_source": {
            "pulsed_laser": {
                "enabled": True,
                "wavelength": 940 * ureg.nanometer,
                "pulse_repetition": F_REP * ureg.hertz,
                "pulse_width": 100 * ureg.picosecond,
                "avg_watts": 1e-3 * ureg.watt,
                "pulse_shape": "gaussian",
            }
        },
        "ambient_source": {"sun": {"enabled": False}},
        "sensor": sensor,
    }


def _pixel_zones(h, w):
    """One zone per pixel, row by row."""
    return [[r / h, (r + 1) / h, c / w, (c + 1) / w] for r in range(h) for c in range(w)]


def _block_zone(r1, r2, c1, c2, h=RES, w=RES):
    """The zone of pixel rows r1:r2 and columns c1:c2."""
    return [r1 / h, r2 / h, c1 / w, c2 / w]


def _signal(emulator, rng, albedo=RHO, normals=None):
    """Photons per laser cycle of each zone: its signal transient summed over the bins."""
    transients, _ = emulator.detector.get_transients(
        depth_frame=torch.as_tensor(rng, dtype=torch.float64) * ureg.meter,
        albedo_frame=torch.tensor(np.broadcast_to(albedo, np.shape(rng)), dtype=torch.float64),
        emitter=emulator.emitter,
        scene=None,
        normal_frame=normals,
    )
    return transients.sum(-1).double().numpy()


def _pixel_signal(rng, falloff=None, albedo=RHO, normals=None):
    """Photons per laser cycle of each pixel of an (h, w) frame, from one zone per pixel."""
    h, w = rng.shape
    emulator = ASPCEmulator(_config(size=(h, w), falloff=falloff, zones=_pixel_zones(h, w)))
    return _signal(emulator, rng, albedo, normals).reshape(h, w)


def _cos_theta(h, w, fl_x, fl_y, cx, cy):
    """Cosine of the angle between each pixel center's ray and the optical axis."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return 1.0 / np.sqrt(1.0 + ((u - cx) / fl_x) ** 2 + ((v - cy) / fl_y) ** 2)


def _sensor_cos_theta(sensor, shape=None):
    """_cos_theta with the sensor's intrinsics, scaled to a frame of the given shape."""
    h, w = shape or (sensor.h, sensor.w)
    return _cos_theta(h, w, sensor.f_x_px * w / sensor.w, sensor.f_y_px * h / sensor.h, w / 2, h / 2)


def _on_axis_photons(emulator, rng=D0, albedo=RHO):
    """The camera equation for a pixel on the axis: N_ph * albedo / (pi d^2 Omega) * pi / (4 N^2) * pitch^2."""
    sensor, laser = emulator.detector.sensor, emulator.emitter
    n_photons = laser.num_photons_per_cycle.magnitude
    field = (sensor.omega * sensor.num_pixels).to(ureg.steradian).magnitude
    pitch = sensor.pixel_pitch.to(ureg.meter).magnitude
    return n_photons * albedo / (np.pi * rng**2 * field) * np.pi / (4 * sensor.f_number.magnitude**2) * pitch**2


def _camera_rays(h, w, fl_x, fl_y, cx, cy):
    """Unnormalized pixel rays in the OpenGL camera frame (+X right, +Y up, looking along -Z), with z = -1."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - cx) / fl_x, -(v - cy) / fl_y, -np.ones_like(u)], axis=-1)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _random_unit_vectors(shape, seed):
    return _unit(np.random.default_rng(seed).normal(size=(*shape, 3)))


def _sun():
    return Sun(
        temperature=5778 * ureg.kelvin,
        light_conditions=LightConditions.BRIGHT_SUNLIGHT,
        lambda_pass=550 * ureg.nanometer,
        delta_lambda=10 * ureg.nanometer,
    )


# =========================================================================== #
# camera_ray_directions                                                        #
# =========================================================================== #
class TestCameraRayDirections:
    def test_rays_are_in_the_opengl_camera_frame(self):
        # 3 x 5 image with its principal point on the center of pixel (1, 2)
        rays = camera_ray_directions(3, 5, 2.0, 2.0, 2.5, 1.5)
        assert rays.shape == (3, 5, 3)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(rays[1, 2], [0.0, 0.0, -1.0], atol=1e-15)
        assert (rays[..., 2] < 0).all()
        assert rays[1, 4, 0] > 0 and rays[1, 0, 0] < 0  # columns to the right point to +X
        assert rays[0, 2, 1] > 0 and rays[2, 2, 1] < 0  # the top row points up (+Y), the bottom row down

    def test_cos_theta_is_one_over_the_ray_length(self):
        rays = camera_ray_directions(6, 8, 5.0, 4.0, 3.0, 3.5)
        lengths = planar_depth_to_range(np.ones((6, 8)), 5.0, 4.0, 3.0, 3.5)
        np.testing.assert_allclose(-rays[..., 2], 1.0 / lengths, rtol=1e-12)

    def test_matches_the_rays_through_the_pixel_centers(self):
        rays = camera_ray_directions(4, 7, 3.0, 2.0, 3.1, 1.7)
        np.testing.assert_allclose(rays, _unit(_camera_rays(4, 7, 3.0, 2.0, 3.1, 1.7)), rtol=1e-12)


# =========================================================================== #
# incidence_cosine                                                             #
# =========================================================================== #
class TestIncidenceCosine:
    def test_fronto_parallel_plane_gives_cos_theta(self):
        normals = np.zeros((9, 12, 3))
        normals[..., 2] = 1.0
        cosine = incidence_cosine(normals, 8.0, 8.0, 6.0, 4.5)
        np.testing.assert_allclose(cosine, _cos_theta(9, 12, 8.0, 8.0, 6.0, 4.5), rtol=1e-12)

    def test_normal_tilted_up_faces_the_rays_below_the_axis(self):
        # Rows 0, 1, 2 have rays at -a, 0, +a below the axis. A normal tilted up by a, towards the camera, is hit
        # head-on by the ray below the axis and at 2a by the ray above it. A y-down camera frame would swap them.
        a = np.radians(20.0)
        normals = np.broadcast_to([0.0, np.sin(a), np.cos(a)], (3, 1, 3))
        cosine = incidence_cosine(normals, 1.0, 1.0 / np.tan(a), 0.5, 1.5)[:, 0]
        np.testing.assert_allclose(cosine, [abs(np.cos(2 * a)), np.cos(a), 1.0], rtol=1e-12)

    def test_tilted_plane_matches_its_geometry(self):
        h, w, fl = 9, 12, 10.0
        normal = _unit(np.array([0.3, -0.5, 0.8]))
        rays = _unit(_camera_rays(h, w, fl, fl, w / 2, h / 2))
        cosine = incidence_cosine(np.broadcast_to(normal, (h, w, 3)), fl, fl, w / 2, h / 2)
        np.testing.assert_allclose(cosine, np.abs(rays @ normal), atol=1e-12)
        assert np.ptp(cosine) > 0.3

    def test_either_side_and_any_length(self):
        normals = _random_unit_vectors((5, 6), seed=0)
        cosine = incidence_cosine(normals, 4.0, 4.0, 3.0, 2.5)
        np.testing.assert_allclose(incidence_cosine(-2.5 * normals, 4.0, 4.0, 3.0, 2.5), cosine, rtol=1e-12)
        np.testing.assert_allclose(incidence_cosine(1e-3 * normals, 4.0, 4.0, 3.0, 2.5), cosine, rtol=1e-12)
        assert ((cosine >= 0) & (cosine <= 1)).all()

    def test_missing_normals_give_zero(self):
        normals = np.zeros((1, 5, 3))
        normals[0, 1] = [np.nan, 0.0, 1.0]
        normals[0, 2] = [0.0, np.inf, 1.0]
        normals[0, 3] = [0.0, 0.0, 1.0]
        normals[0, 4] = [0.0, 0.0, -np.inf]
        cosine = incidence_cosine(normals, 2.0, 2.0, 2.5, 0.5)
        assert cosine[0, 3] == pytest.approx(1 / np.sqrt(1.25), rel=1e-12)  # (3.5 - 2.5) / 2 = tan(theta)
        assert cosine[0, [0, 1, 2, 4]].tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_leading_dimensions(self):
        normals = _random_unit_vectors((2, 3, 4, 5), seed=1)
        cosine = incidence_cosine(normals, 4.0, 3.0, 2.5, 2.0)
        assert cosine.shape == (2, 3, 4, 5)
        for i in range(2):
            for j in range(3):
                np.testing.assert_array_equal(cosine[i, j], incidence_cosine(normals[i, j], 4.0, 3.0, 2.5, 2.0))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tensor_keeps_dtype_and_gradient(self, dtype):
        values = _random_unit_vectors((4, 5), seed=2)
        values[0, 0] = 0.0  # a pixel without a surface must not give NaN gradients
        normals = torch.tensor(values, dtype=dtype, requires_grad=True)
        cosine = incidence_cosine(normals, 4.0, 4.0, 2.5, 2.0)
        assert isinstance(cosine, torch.Tensor)
        assert cosine.dtype == dtype
        atol = 1e-6 if dtype == torch.float32 else 1e-12
        np.testing.assert_allclose(
            cosine.detach().double().numpy(), incidence_cosine(values, 4.0, 4.0, 2.5, 2.0), atol=atol
        )

        cosine.sum().backward()
        assert torch.isfinite(normals.grad).all()
        assert (normals.grad[0, 0] == 0).all()
        assert normals.grad.abs().sum() > 0

    def test_integer_tensor_gives_float64(self):
        normals = torch.zeros((2, 2, 3), dtype=torch.int64)
        normals[..., 2] = 3
        cosine = incidence_cosine(normals, 1.0, 1.0, 1.0, 1.0)
        assert cosine.dtype == torch.float64
        np.testing.assert_allclose(cosine.numpy(), np.full((2, 2), 1 / np.sqrt(1.5)), rtol=1e-12)

    def test_quantity_is_read_as_its_magnitude(self):
        normals = _random_unit_vectors((3, 4), seed=3)
        cosine = incidence_cosine(ureg.Quantity(normals, "dimensionless"), 2.0, 2.0, 2.0, 1.5)
        np.testing.assert_array_equal(cosine, incidence_cosine(normals, 2.0, 2.0, 2.0, 1.5))


# =========================================================================== #
# preproc_normals                                                              #
# =========================================================================== #
class TestPreprocNormals:
    H, W = 9, 12

    def _normals(self, frame, config=None, **kwargs):
        config = config or _config(size=(self.H, self.W))
        qty = preproc_normals(frame, config, **kwargs)
        return qty.magnitude.numpy()

    def test_rendered_pass_is_rotated_to_the_camera_frame(self):
        # VisionSim's normal pass holds R @ n_world; the camera-frame normal is R.T @ n_world
        world = _random_unit_vectors((self.H, self.W), seed=4)
        rendered = world @ ROTATION.T
        out = self._normals((rendered.astype(np.float32), {"transform_matrix": POSE}))
        np.testing.assert_allclose(out, world @ ROTATION, atol=1e-6)

    def test_rendered_is_the_default(self):
        rendered = _random_unit_vectors((self.H, self.W), seed=5)
        item = (rendered, {"transform_matrix": POSE})
        np.testing.assert_array_equal(self._normals(item), self._normals(item, normal_type="rendered"))

    def test_world_normals_are_rotated_to_the_camera_frame(self):
        world = _random_unit_vectors((self.H, self.W), seed=6)
        out = self._normals((world, {"transform_matrix": POSE.tolist()}), normal_type="world")
        np.testing.assert_allclose(out, world @ ROTATION, atol=1e-12)

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda n: n, id="array"),
            pytest.param(torch.from_numpy, id="tensor"),
            pytest.param(lambda n: (n, {"transform_matrix": POSE}), id="item"),
        ],
    )
    def test_camera_normals_are_used_as_is(self, wrap):
        camera = _random_unit_vectors((self.H, self.W), seed=7)
        np.testing.assert_allclose(self._normals(wrap(camera), normal_type="camera"), camera, atol=1e-12)

    @pytest.mark.parametrize("normal_type", ["rendered", "world", "camera"])
    def test_every_type_gives_the_incidence_of_a_world_plane(self, normal_type):
        # A tilted plane in the world, seen by the rotated camera: the angle of incidence is the angle between the
        # plane's world normal and each pixel's ray in the world, R @ d_camera
        h, w, fl = self.H, self.W, 9.0
        world_normal = _unit(np.array([0.2, 0.9, -0.4]))
        world_rays = _unit(_camera_rays(h, w, fl, fl, w / 2, h / 2)) @ ROTATION.T
        expected = np.abs(world_rays @ world_normal)

        frame = {
            "rendered": ROTATION @ world_normal,
            "world": world_normal,
            "camera": ROTATION.T @ world_normal,
        }[normal_type]
        out = self._normals((np.broadcast_to(frame, (h, w, 3)), {"transform_matrix": POSE}), normal_type=normal_type)
        np.testing.assert_allclose(incidence_cosine(out, fl, fl, w / 2, h / 2), expected, atol=1e-12)

    def test_the_rendered_pass_is_not_in_the_camera_frame(self):
        # Reading the rendered pass as camera-frame normals, as the documentation describes it, gives other angles
        h, w, fl = self.H, self.W, 9.0
        rendered = np.broadcast_to(ROTATION @ ROTATION @ _unit(np.array([0.1, -0.2, 1.0])), (h, w, 3))
        as_rendered = incidence_cosine(self._normals((rendered, {"transform_matrix": POSE})), fl, fl, w / 2, h / 2)
        as_camera = incidence_cosine(self._normals(rendered, normal_type="camera"), fl, fl, w / 2, h / 2)
        assert np.abs(as_rendered - as_camera).max() > 0.3

    @pytest.mark.parametrize("normal_type", ["rendered", "world"])
    @pytest.mark.parametrize(
        "frame",
        [
            pytest.param(np.ones((9, 12, 3)), id="array"),
            pytest.param((np.ones((9, 12, 3)), {"file_path": "normals/0000.exr"}), id="item-without-pose"),
        ],
    )
    def test_a_rotation_needs_the_camera_pose(self, frame, normal_type):
        with pytest.raises(ValueError, match="transform_matrix"):
            self._normals(frame, normal_type=normal_type)

    @pytest.mark.parametrize("scale", [1.0, 2.0], ids=["sensor", "half-the-field-of-view"])
    def test_item_must_span_the_sensors_field_of_view(self, scale):
        """The incidence cosine follows the rays of the sensor's field of view; a normal frame of another field of
        view was accepted, and got the cosines of the sensor's."""
        config = _config(size=(self.H, self.W))
        fl_x, fl_y, cx, cy = SPADSensor(**config["sensor"]).frame_intrinsics()
        k = {"fl_x": scale * fl_x, "fl_y": scale * fl_y, "cx": cx, "cy": cy, "h": self.H, "w": self.W}
        camera = _random_unit_vectors((self.H, self.W), seed=13)
        item = (camera, {"transform_matrix": POSE, **k})
        if scale == 1.0:
            np.testing.assert_allclose(self._normals(item, config, normal_type="camera"), camera, atol=1e-12)
        else:
            with pytest.raises(ValueError, match="normal frame, .* not those of the sensor's field of view"):
                self._normals(item, config, normal_type="camera")

    @pytest.mark.parametrize("normal_type", ["Camera", "opengl", "world_space", "", None])
    def test_unknown_normal_type_raises(self, normal_type):
        with pytest.raises(ValueError, match="normal_type"):
            self._normals(np.ones((9, 12, 3)), normal_type=normal_type)

    @pytest.mark.parametrize("shape", [(9, 12), (9, 12, 2), (1, 9, 12, 3)])
    def test_frame_of_another_shape_raises(self, shape):
        with pytest.raises(ValueError, match="normal frame"):
            self._normals(np.ones(shape), normal_type="camera")

    def test_normals_get_unit_length_and_missing_ones_are_zero(self):
        normals = np.zeros((self.H, self.W, 3))
        normals[..., 2] = 2.0
        normals[0, 0] = [3.0, 0.0, 4.0]
        normals[0, 1] = 0.0
        normals[0, 2] = [np.nan, 0.0, 1.0]
        normals[0, 3] = [0.0, np.inf, 1.0]
        out = self._normals(normals, normal_type="camera")
        np.testing.assert_allclose(out[0, 0], [0.6, 0.0, 0.8], rtol=1e-12)
        assert (out[0, 1:4] == 0).all()
        np.testing.assert_array_equal(out[1:, :, 2], 1.0)

    def test_extra_channels_are_dropped(self):
        normals = _random_unit_vectors((self.H, self.W), seed=8)
        rgba = np.concatenate([normals, np.ones((self.H, self.W, 1))], axis=-1)
        np.testing.assert_allclose(self._normals(rgba, normal_type="camera"), normals, atol=1e-12)

    def test_exr_collapsed_to_one_channel_is_expanded(self):
        # Dataset.load_data collapses an EXR whose three channels are equal to one channel
        assert (self._normals(np.zeros((self.H, self.W, 1)), normal_type="camera") == 0).all()
        out = self._normals(np.full((self.H, self.W, 1), 2.0), normal_type="camera")
        np.testing.assert_allclose(out, np.full((self.H, self.W, 3), 1 / np.sqrt(3)), rtol=1e-12)

    def test_returns_a_float64_tensor_at_the_sensor_size(self):
        qty = preproc_normals(np.ones((18, 24, 3)), _config(size=(self.H, self.W)), normal_type="camera")
        assert str(qty.units) == "dimensionless"
        assert isinstance(qty.magnitude, torch.Tensor)
        assert qty.magnitude.dtype == torch.float64
        assert qty.magnitude.shape == (self.H, self.W, 3)
        assert not qty.magnitude.requires_grad
        grad = preproc_normals(
            np.ones((18, 24, 3)), _config(size=(self.H, self.W)), normal_type="camera", requires_grad=True
        )
        assert grad.magnitude.requires_grad

    @pytest.mark.parametrize("frame_shape", [(18, 24), (13, 17), (5, 7)])
    def test_resized_like_the_depth_frame(self, frame_shape):
        # Each pixel keeps the normal of the frame pixel its depth comes from: both carry the frame pixel's index
        h, w = frame_shape
        config = _config(size=(self.H, self.W))
        index = np.arange(h * w, dtype=np.float64).reshape(h, w)
        normals = np.stack([index, np.ones((h, w)), np.zeros((h, w))], axis=-1)
        depth = 0.1 + 1e-4 * index

        depth_qty, _, _ = preproc_frames(depth, np.ones((h, w)), np.zeros((h, w)), config, depth_type="range")
        out = self._normals(normals, config, normal_type="camera")
        from_depth = np.rint((depth_qty.magnitude.numpy() - 0.1) / 1e-4)
        np.testing.assert_array_equal(np.rint(out[..., 0] / out[..., 1]), from_depth)


# =========================================================================== #
# Off-axis falloff                                                             #
# =========================================================================== #
class TestFalloff:
    def test_default_is_the_camera_equation(self):
        assert SPADSensor().falloff == "cos4"
        assert SPADSensor(size=(4, 4), fov=60 * ureg.degree).falloff == "cos4"
        assert FALLOFF_TYPES == ("cos4", "cos3", "none")

    @pytest.mark.parametrize("falloff", ["cos2", "COS4", "cos^4", "", None, 4])
    def test_unknown_falloff_raises(self, falloff):
        with pytest.raises(ValueError, match="falloff"):
            SPADSensor(size=(4, 4), fov=60 * ureg.degree, falloff=falloff)

    @pytest.mark.parametrize("falloff", FALLOFF_TYPES)
    @pytest.mark.parametrize("shape", [None, (14, 22), (3, 5)])
    def test_pixel_falloff_is_a_power_of_cos_theta(self, falloff, shape):
        sensor = SPADSensor(size=(7, 11), fov=(70 * ureg.degree, 50 * ureg.degree), falloff=falloff)
        expected = _sensor_cos_theta(sensor, shape) ** POWER[falloff]
        np.testing.assert_allclose(sensor.pixel_falloff(shape), expected, rtol=1e-12)

    def test_sensor_params_include_the_falloff(self):
        a = SPADSensor(size=(4, 4), fov=60 * ureg.degree, falloff="cos3")
        assert a.params["falloff"] == "cos3"
        assert a == SPADSensor(size=(4, 4), fov=60 * ureg.degree, falloff="cos3")
        assert a != SPADSensor(size=(4, 4), fov=60 * ureg.degree, falloff="cos4")

    @pytest.mark.parametrize("falloff", FALLOFF_TYPES)
    def test_pixels_at_one_range_follow_the_falloff(self, falloff):
        signal = _pixel_signal(np.full((RES, RES), D0), falloff)
        cos_theta = _sensor_cos_theta(SPADSensor(**_config()["sensor"]))
        np.testing.assert_allclose(signal / signal[CENTER], cos_theta ** POWER[falloff], rtol=1e-5)

    @pytest.mark.parametrize("falloff", ["cos4", "cos3"])
    def test_pixel_on_the_axis_follows_the_camera_equation(self, falloff):
        emulator = ASPCEmulator(_config(falloff=falloff, zones=[_block_zone(16, 17, 16, 17)]))
        assert _signal(emulator, np.full((RES, RES), D0))[0] == pytest.approx(_on_axis_photons(emulator), rel=1e-5)

    def test_cos3_collects_the_lambertian_return_through_the_aperture(self):
        # The pixels' solid angles add up to the field of view, so the whole frame of a sphere at range d around the
        # sensor collects N_ph * albedo * A / (pi d^2), with A the aperture area: pi p^2 / (4 N^2) = A * omega_0
        emulator = ASPCEmulator(_config(falloff="cos3"))
        sensor = emulator.detector.sensor
        omega_0 = 1.0 / (sensor.f_x_px * sensor.f_y_px)
        field = (sensor.omega * sensor.num_pixels).to(ureg.steradian).magnitude
        assert sensor.pixel_falloff().sum() * omega_0 == pytest.approx(field, rel=1e-3)

        pitch = sensor.pixel_pitch.to(ureg.meter).magnitude
        aperture = np.pi * pitch**2 / (4 * sensor.f_number.magnitude**2 * omega_0)
        n_photons = emulator.emitter.num_photons_per_cycle.magnitude
        total = _signal(emulator, np.full((RES, RES), D0))[0] * sensor.num_pixels
        assert total == pytest.approx(n_photons * RHO * aperture / (np.pi * D0**2), rel=1e-3)

    @pytest.mark.parametrize("falloff", FALLOFF_TYPES)
    def test_fronto_parallel_plane_with_normals(self, falloff):
        # Range Z / cos, incidence cos: cos * cos^2 * cos^k
        sensor = SPADSensor(**_config()["sensor"])
        cos_theta = _sensor_cos_theta(sensor)
        normals = torch.zeros((RES, RES, 3), dtype=torch.float64)
        normals[..., 2] = 1.0
        signal = _pixel_signal(D0 / cos_theta, falloff, normals=normals)
        np.testing.assert_allclose(signal / signal[CENTER], cos_theta ** (3 + POWER[falloff]), rtol=1e-5)

    @pytest.mark.parametrize("falloff", FALLOFF_TYPES)
    @pytest.mark.parametrize("source", ["ambient_frame", "sun"])
    def test_ambient_light_gets_the_falloff(self, falloff, source):
        sensor = SPADSensor(size=(9, 13), fov=(70 * ureg.degree, 50 * ureg.degree), falloff=falloff)
        detector = Detector(sensor=sensor, histogrammer=Histogrammer(HistConfig()))
        laser = PulsedLaser(pulse_repetition=10e6 * ureg.hertz)
        uniform = ureg.Quantity(torch.full((9, 13), 0.3, dtype=torch.float64), "dimensionless")
        offsets = detector._get_ambient_offset(
            albedo_frame=uniform,
            emitter=laser,
            scene=_sun(),
            ambient_frame=uniform if source == "ambient_frame" else None,
        ).double()
        expected = _sensor_cos_theta(sensor) ** POWER[falloff]
        np.testing.assert_allclose((offsets / offsets[4, 6]).numpy(), expected, rtol=1e-5)

    @pytest.mark.parametrize("falloff", ["cos4", "cos3"])
    def test_frame_at_another_resolution_spans_the_field_of_view(self, falloff):
        # A 22 x 22 frame given to a 11 x 11 sensor: its pixels are weighted by their own angles to the axis
        h = w = 22
        config = _config(size=(11, 11), falloff=falloff, zones=_pixel_zones(h, w))
        signal = _signal(ASPCEmulator(config), np.full((h, w), D0)).reshape(h, w)
        expected = _sensor_cos_theta(SPADSensor(**config["sensor"]), (h, w)) ** POWER[falloff]
        np.testing.assert_allclose(signal / signal.max(), expected / expected.max(), rtol=1e-5)


# =========================================================================== #
# Zones of any size are the mean of their pixels                              #
# =========================================================================== #
ZONES = {
    "center-pixel": _block_zone(16, 17, 16, 17),
    "center-3x3": _block_zone(15, 18, 15, 18),
    "corner-11x11": _block_zone(0, 11, 0, 11),
    "off-center-5x7": _block_zone(3, 8, 20, 27),
    "full-frame": _block_zone(0, RES, 0, RES),
}


class TestZoneWeights:
    """A zone sums up to 1089 pixels in float32, hence a tolerance of 1e-4; the size factor it rules out is 8 %."""

    @pytest.mark.parametrize("falloff", ["cos4", "cos3"])
    @pytest.mark.parametrize("scene", ["sphere", "plane"])
    def test_zone_is_the_mean_of_its_pixels_whatever_its_size(self, falloff, scene):
        cos_theta = _sensor_cos_theta(SPADSensor(**_config()["sensor"]))
        rng = np.full((RES, RES), D0) if scene == "sphere" else D0 / cos_theta
        pixels = _pixel_signal(rng, falloff)

        emulator = ASPCEmulator(_config(falloff=falloff, zones=list(ZONES.values())))
        zones = _signal(emulator, rng)
        masks = emulator.detector.get_perpixel_fov_masks(torch.zeros(RES, RES), vignette=False).numpy() > 0
        expected = [pixels[mask].mean() for mask in masks]
        np.testing.assert_allclose(zones, expected, rtol=1e-4)

    def test_zones_have_no_factor_for_their_size(self):
        # At one range, a zone is the camera equation on the axis times the mean of cos^4 over its pixels
        emulator = ASPCEmulator(_config(zones=list(ZONES.values())))
        zones = _signal(emulator, np.full((RES, RES), D0))
        cos4 = _sensor_cos_theta(emulator.detector.sensor) ** 4
        masks = emulator.detector.get_perpixel_fov_masks(torch.zeros(RES, RES), vignette=False).numpy() > 0
        expected = _on_axis_photons(emulator) * np.array([cos4[mask].mean() for mask in masks])
        np.testing.assert_allclose(zones, expected, rtol=1e-4)

    def test_without_falloff_zones_keep_their_size_rescale(self):
        # falloff "none" is the former model: every pixel collects as much as the one on the axis, and each zone is
        # rescaled by get_irradiance_with_fov, which depends on its size only
        emulator = ASPCEmulator(_config(falloff="none", zones=list(ZONES.values())))
        sensor = emulator.detector.sensor
        zones = _signal(emulator, np.full((RES, RES), D0))
        scales = [get_irradiance_with_fov(1.0, sensor.fov, zone, sensor.omega, RES, RES) for zone in ZONES.values()]
        np.testing.assert_allclose(zones, _on_axis_photons(emulator) * np.array(scales), rtol=1e-4)
        assert scales[0] == pytest.approx(1.0849, abs=1e-4)
        assert scales[-1] == pytest.approx(1.0, rel=1e-12)


# =========================================================================== #
# The normal frame in the emulator                                             #
# =========================================================================== #
class TestIncidenceInTheEmulator:
    def test_scene_radiance_scales_with_the_incidence_cosine(self):
        laser = PulsedLaser()
        facing = laser.get_scene_radiance(0.5, 2.0 * ureg.meter, 100, 1e-4 * ureg.steradian)
        oblique = laser.get_scene_radiance(0.5, 2.0 * ureg.meter, 100, 1e-4 * ureg.steradian, cos_incidence=0.25)
        assert oblique.magnitude == pytest.approx(0.25 * facing.magnitude, rel=1e-12)

    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, 85.0])
    @pytest.mark.parametrize("side", [1.0, -1.0])
    def test_surface_on_the_axis_scales_with_the_cosine_of_its_tilt(self, angle_deg, side):
        a = np.radians(angle_deg)
        emulator = ASPCEmulator(_config(size=(1, 1)))
        tilted = torch.tensor([[[0.0, np.sin(a), np.cos(a)]]], dtype=torch.float64) * side
        facing = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
        ratio = _signal(emulator, np.full((1, 1), D0), normals=tilted) / _signal(
            emulator, np.full((1, 1), D0), normals=facing
        )
        assert ratio[0] == pytest.approx(np.cos(a), rel=1e-6)

    def test_normals_scale_the_return_like_an_albedo(self):
        rng = np.random.default_rng(9).uniform(0.3, 1.0, (RES, RES))
        normals = _random_unit_vectors((RES, RES), seed=10)
        fl = (RES / 2) / np.tan(np.radians(FOV_DEG / 2))
        cosine = np.abs((_unit(_camera_rays(RES, RES, fl, fl, RES / 2, RES / 2)) * normals).sum(axis=-1))
        with_normals = _pixel_signal(rng, normals=torch.from_numpy(normals))
        np.testing.assert_allclose(with_normals, _pixel_signal(rng, albedo=RHO * cosine), rtol=1e-5)

    def test_frames_in_a_batch(self):
        emulator = ASPCEmulator(_config(size=(9, 12), zones=_pixel_zones(9, 12)))
        rng = np.random.default_rng(11).uniform(0.3, 1.0, (2, 9, 12))
        normals = torch.from_numpy(_random_unit_vectors((2, 9, 12), seed=12))
        batch = _signal(emulator, rng, normals=normals)
        single = np.concatenate([_signal(emulator, rng[i], normals=normals[i]) for i in range(2)])
        np.testing.assert_allclose(batch, single, rtol=1e-6)

    @pytest.mark.parametrize("shape", [(9, 12), (9, 12, 2), (10, 12, 3), (9, 11, 3)])
    def test_normal_frame_of_another_shape_raises(self, shape):
        emulator = ASPCEmulator(_config(size=(9, 12)))
        with pytest.raises(ValueError, match="normal frame"):
            _signal(emulator, np.full((9, 12), D0), normals=torch.ones(shape))

    def test_without_normals_it_is_logged_once(self, caplog):
        _log_without_normals.cache_clear()
        with caplog.at_level(logging.WARNING, logger="visionsim.emulate.aspc.detector"):
            emulator = ASPCEmulator(_config(size=(4, 4)))
            _signal(emulator, np.full((4, 4), D0))
            _signal(emulator, np.full((4, 4), D0))
            _signal(ASPCEmulator(_config(size=(4, 4))), np.full((4, 4), D0))
        records = [r for r in caplog.records if r.name == "visionsim.emulate.aspc.detector"]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert "incidence cosine" in records[0].getMessage()

    def test_with_normals_nothing_is_logged(self, caplog):
        _log_without_normals.cache_clear()
        emulator = ASPCEmulator(_config(size=(4, 4)))
        normals = torch.zeros((4, 4, 3))
        normals[..., 2] = 1.0
        with caplog.at_level(logging.WARNING, logger="visionsim.emulate.aspc.detector"):
            _signal(emulator, np.full((4, 4), D0), normals=normals)
        assert not [r for r in caplog.records if r.name == "visionsim.emulate.aspc.detector"]

    def test_process_frame_uses_the_normal_frame(self):
        a = np.radians(50.0)
        emulator = ASPCEmulator(_config(size=(5, 5)))
        depth = torch.full((5, 5), D0, dtype=torch.float64) * ureg.meter
        ambient = ureg.Quantity(torch.zeros((5, 5), dtype=torch.float64), "dimensionless")
        normals = torch.zeros((5, 5, 3), dtype=torch.float64)
        normals[..., 0], normals[..., 2] = np.sin(a), np.cos(a)
        fl = 2.5 / np.tan(np.radians(FOV_DEG / 2))
        cosine = np.abs(_unit(_camera_rays(5, 5, fl, fl, 2.5, 2.5)) @ normals[0, 0].numpy())

        hist = emulator.process_frame(depth, torch.full((5, 5), RHO, dtype=torch.float64), ambient, normals)
        expected = emulator.process_frame(depth, torch.from_numpy(RHO * cosine), ambient)
        np.testing.assert_allclose(hist, expected, rtol=1e-5)
        assert not np.allclose(hist, emulator.process_frame(depth, torch.full((5, 5), RHO), ambient), rtol=1e-2)


# =========================================================================== #
# CLI: a rendered normals/ directory                                           #
# =========================================================================== #
class TestCliNormals:
    N = 17
    ARGS = {
        "sensor_size": (N, N),
        "sensor_fov": (FOV_DEG, FOV_DEG),
        "n_bins": N_BINS,
        "bin_width": BIN_M,
        "max_depth": N_BINS * BIN_M,
        "active_pulse_repetition": F_REP / 1e6,
        "active_pulse_width": 0.1,
        "active_avg_watts": 1e-4,
        "ambient_enabled": False,
        "free_running": False,
        "fast_sim": True,
    }

    @staticmethod
    def _write_dataset(root, frames, intrinsics):
        """(H, W, C) frames, stored as a stack in a .npy file, seen from POSE, with their transforms.json."""
        root.mkdir(parents=True)
        np.save(root / "frames.npy", np.stack(frames))
        transforms = [
            {"file_path": "frames.npy", "offset": i, "transform_matrix": POSE.tolist()} for i in range(len(frames))
        ]
        Metadata.from_frames(transforms, camera={**intrinsics, "c": frames[0].shape[-1]}).save(root / "transforms.json")

    def _write_scene(self, root, n_normals=1):
        """A plane tilted in the camera frame. Returns its planar depth, intrinsics and per-pixel incidence cosine."""
        n = self.N
        fl = (n / 2) / np.tan(np.radians(FOV_DEG / 2))
        intrinsics = {"fl_x": fl, "fl_y": fl, "cx": n / 2, "cy": n / 2, "h": n, "w": n}

        # Points s * (x, -y, -1) on the plane normal . P = -h0, with s the planar depth
        normal = _unit(np.array([0.3, -0.4, 1.0]))
        rays = _camera_rays(n, n, fl, fl, n / 2, n / 2)
        z = (-0.5 * normal[2] / (rays @ normal)).astype(np.float32)
        cosine = np.abs(_unit(rays) @ normal)

        # VisionSim's normal pass holds R @ n_world, with n_world = R @ n_camera
        rendered = np.broadcast_to(ROTATION @ ROTATION @ normal, (n, n, 3)).astype(np.float32)

        self._write_dataset(root / "depths", [z[..., np.newaxis]], intrinsics)
        self._write_dataset(root / "diffuse" / "color", [np.ones((n, n, 3), np.float32)], intrinsics)
        self._write_dataset(root / "frames", [np.zeros((n, n, 3), np.float32)], intrinsics)
        self._write_dataset(root / "normals", [rendered] * n_normals, intrinsics)
        return z, intrinsics, cosine

    def test_cli_uses_the_rendered_normals(self, tmp_path):
        from visionsim.cli.emulate import aspc as aspc_cli

        n = self.N
        z, k, cosine = self._write_scene(tmp_path / "in")
        aspc_cli(input_dir=tmp_path / "in", output_dir=tmp_path / "out", **self.ARGS)
        hist = np.load(tmp_path / "out" / "frames" / "histogram_0000.npy")

        emulator = ASPCEmulator(prepare_config(dict(self.ARGS)))
        rng = planar_depth_to_range(z, k["fl_x"], k["fl_y"], k["cx"], k["cy"])
        depth = ureg.Quantity(torch.from_numpy(rng), "meter")
        ambient = ureg.Quantity(torch.zeros(n, n, dtype=torch.float64), "dimensionless")
        expected = emulator.process_frame(depth, torch.from_numpy(cosine), ambient)
        np.testing.assert_allclose(hist, expected, rtol=1e-5)

        without_normals = emulator.process_frame(depth, torch.ones(n, n, dtype=torch.float64), ambient)
        assert not np.allclose(hist, without_normals, rtol=1e-2)

    def test_cli_raises_when_the_normal_maps_do_not_match_the_depth_maps(self, tmp_path):
        from visionsim.cli.emulate import aspc as aspc_cli

        self._write_scene(tmp_path / "in", n_normals=2)
        with pytest.raises(ValueError, match="normal maps"):
            aspc_cli(input_dir=tmp_path / "in", output_dir=tmp_path / "out", **self.ARGS)
