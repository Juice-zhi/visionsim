"""Planar depth vs range: the emulator bins and attenuates each pixel's laser return by its range.

A rendered ``depths/`` directory holds Blender's Z pass, the distance along the optical axis. ``preproc_frames``
converts it to range, the distance along each pixel's ray, with the camera intrinsics.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.constants import c as C_LIGHT

from visionsim.dataset.models import Metadata
from visionsim.emulate.aspc.emulator import ASPCEmulator
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.utils import (
    DEPTH_CUTOFF,
    INTRINSICS_RTOL,
    planar_depth_to_range,
    prepare_config,
    preproc_frames,
    ureg,
)

N_BINS, BIN_M = 1024, 1.25e-3
F_REP = C_LIGHT / (2 * N_BINS * BIN_M)  # one laser period is 1.28 m of range, so a bin is 1.25 mm
Z0 = 0.5


def _intrinsics(h, w, fov_deg):
    """Intrinsics of an (h, w) pinhole render with square pixels and a horizontal field of view of fov_deg."""
    fl = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    return {"fl_x": fl, "fl_y": fl, "cx": w / 2, "cy": h / 2, "h": h, "w": w}


# Intrinsics of 9 x 12 renders that do not span the field of view of a 60 deg sensor of square pixels
MISMATCHED = {
    "narrower": _intrinsics(9, 12, 30.0),
    "wider": _intrinsics(9, 12, 61.0),
    "principal-point": {**_intrinsics(9, 12, 60.0), "cx": 3.5, "cy": 2.5},
    "fl_y": {**_intrinsics(9, 12, 60.0), "fl_y": _intrinsics(9, 12, 60.0)["fl_y"] * (1 + 2 * INTRINSICS_RTOL)},
}


def _depth_item(z, intrinsics):
    """A depths/ item as a Dataset returns it: the (H, W, 1) float32 EXR data and its transform."""
    return z.astype(np.float32)[..., np.newaxis], {"file_path": "depths/0000.exr", **intrinsics}


def _true_range(z, intrinsics):
    """Length of the vector to the 3D point that each pixel center sees at planar depth z."""
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - intrinsics["cx"]) / intrinsics["fl_x"] * z
    y = (v - intrinsics["cy"]) / intrinsics["fl_y"] * z
    return np.linalg.norm(np.stack([x, y, z], axis=-1), axis=-1)


def _config(size, fov=60.0):
    """Emulator config with 1.25 mm bins, a short pulse and no ambient light.

    A scalar fov is the horizontal field of view of square pixels, as in _intrinsics; a tuple sets both.
    """
    fov = [f * ureg.degree for f in fov] if isinstance(fov, tuple) else fov * ureg.degree
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
            "pixel_fov_list": [[0.0, 1.0, 0.0, 1.0]],
        },
        "active_source": {
            "pulsed_laser": {
                "enabled": True,
                "wavelength": 940 * ureg.nanometer,
                "pulse_repetition": F_REP * ureg.hertz,
                "pulse_width": 50 * ureg.picosecond,
                "avg_watts": 1e-3 * ureg.watt,
                "pulse_shape": "gaussian",
            }
        },
        "ambient_source": {"sun": {"enabled": False}},
        "sensor": {"size": tuple(size), "pixel_pitch": 10 * ureg.micrometer, "f_number": 1.4, "fov": fov},
    }


def _transient(depth_qty, albedo_qty, config):
    """Signal transient of the whole frame (one zone), before the pulse shape and noise."""
    emulator = ASPCEmulator(config)
    transients, _ = emulator.detector.get_transients(
        depth_frame=depth_qty, albedo_frame=albedo_qty, emitter=emulator.emitter, scene=None, ambient_frame=None
    )
    return transients[0].double().numpy()


# =========================================================================== #
# planar_depth_to_range                                                        #
# =========================================================================== #
class TestPlanarDepthToRange:
    def test_rays_at_known_angles(self):
        # Columns 0, 1, 2 have tan = 0, 1, 2 (fl_x = 1, cx = 0.5), rows 0, 1, 2 have tan = 0, 0.5, 1 (fl_y = 2)
        r = planar_depth_to_range(np.full((3, 3), 2.0), fl_x=1.0, fl_y=2.0, cx=0.5, cy=0.5)
        expected = 2.0 * np.sqrt(1.0 + np.add.outer([0.0, 0.25, 1.0], [0.0, 1.0, 4.0]))
        np.testing.assert_allclose(r, expected, rtol=1e-12)

    def test_pixel_on_the_principal_point_keeps_its_depth(self):
        r = planar_depth_to_range(np.full((5, 7), 3.0), 10.0, 10.0, 3.5, 2.5)
        assert r[2, 3] == 3.0
        assert np.count_nonzero(r == 3.0) == 1
        assert (r >= 3.0).all()

    def test_pixel_centers_are_symmetric_about_a_centered_principal_point(self):
        # Taking pixel u at u instead of u + 0.5 would shift every ray by half a pixel and break the symmetry
        r = planar_depth_to_range(np.ones((6, 8)), 5.0, 5.0, 4.0, 3.0)
        np.testing.assert_allclose(r, r[::-1, ::-1], rtol=1e-12)
        np.testing.assert_allclose(r, r[:, ::-1], rtol=1e-12)

    def test_matches_rays_cast_to_a_tilted_plane(self):
        h, w = 9, 12
        k = _intrinsics(h, w, 60.0)
        normal, offset = np.array([0.3, -0.2, 1.0]), 0.8
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([(u - k["cx"]) / k["fl_x"], (v - k["cy"]) / k["fl_y"], np.ones_like(u)], axis=-1)
        points = rays * (offset / (rays @ normal))[..., np.newaxis]

        r = planar_depth_to_range(points[..., 2], k["fl_x"], k["fl_y"], k["cx"], k["cy"])
        np.testing.assert_allclose(r, np.linalg.norm(points, axis=-1), rtol=1e-12)

    def test_corner_of_a_fronto_parallel_plane(self):
        # 90 deg square render of 2 x 2 pixels: every pixel center is at tan = 0.5 on both axes
        k = _intrinsics(2, 2, 90.0)
        r = planar_depth_to_range(np.full((2, 2), Z0), k["fl_x"], k["fl_y"], k["cx"], k["cy"])
        np.testing.assert_allclose(r, np.full((2, 2), Z0 * np.sqrt(1.5)), rtol=1e-12)

    def test_leading_dimensions_broadcast(self):
        z = np.random.default_rng(0).uniform(0.5, 2.0, (2, 3, 4, 5))
        r = planar_depth_to_range(z, 4.0, 3.0, 2.5, 2.0)
        assert r.shape == z.shape
        for i in range(2):
            for j in range(3):
                np.testing.assert_array_equal(r[i, j], planar_depth_to_range(z[i, j], 4.0, 3.0, 2.5, 2.0))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tensor_keeps_dtype_device_and_gradient(self, dtype):
        z = torch.full((3, 3), 2.0, dtype=dtype, requires_grad=True)
        r = planar_depth_to_range(z, 1.0, 2.0, 0.5, 0.5)
        assert isinstance(r, torch.Tensor)
        assert r.dtype == dtype
        assert r.device == z.device

        r.sum().backward()
        ray_lengths = planar_depth_to_range(np.ones((3, 3)), 1.0, 2.0, 0.5, 0.5)
        np.testing.assert_allclose(z.grad.double().numpy(), ray_lengths, rtol=1e-6)

    def test_integer_tensor_gives_float64(self):
        r = planar_depth_to_range(torch.full((2, 2), 3, dtype=torch.int64), 1.0, 1.0, 1.0, 1.0)
        assert r.dtype == torch.float64
        np.testing.assert_allclose(r.numpy(), np.full((2, 2), 3.0 * np.sqrt(1.5)), rtol=1e-12)

    def test_quantity_keeps_units(self):
        r = planar_depth_to_range(np.full((3, 3), 2000.0) * ureg.millimeter, 1.0, 2.0, 0.5, 0.5)
        assert r.units == ureg.millimeter
        expected = planar_depth_to_range(np.full((3, 3), 2.0), 1.0, 2.0, 0.5, 0.5)
        np.testing.assert_allclose(r.to(ureg.meter).magnitude, expected, rtol=1e-12)

    def test_values_without_a_return_keep_meaning_no_return(self):
        z = np.array([[0.0, 1e10, DEPTH_CUTOFF, np.nan, np.inf, -1.0]])
        r = planar_depth_to_range(z, 1.0, 1.0, 0.0, 0.5)
        assert r[0, 0] == 0.0
        assert r[0, 1] >= DEPTH_CUTOFF and r[0, 2] >= DEPTH_CUTOFF
        assert np.isnan(r[0, 3])
        assert r[0, 4] == np.inf
        assert r[0, 5] < 0.0


# =========================================================================== #
# preproc_frames(depth_type=...)                                               #
# =========================================================================== #
class TestPreprocFramesDepthType:
    @staticmethod
    def _depth(depth_frame, config, **kwargs):
        h, w = config["sensor"]["size"]
        depth_qty, _, _ = preproc_frames(depth_frame, np.ones((h, w)), np.zeros((h, w)), config, **kwargs)
        return depth_qty

    def test_default_is_planar_z(self):
        k = _intrinsics(9, 12, 60.0)
        item = _depth_item(np.full((9, 12), Z0), k)
        default = self._depth(item, _config((9, 12)))
        planar = self._depth(item, _config((9, 12)), depth_type="planar_z")
        assert torch.equal(default.magnitude, planar.magnitude)

    def test_depths_item_is_converted_with_its_intrinsics(self):
        h, w = 9, 12
        k = _intrinsics(h, w, 60.0)
        z = np.random.default_rng(1).uniform(0.2, 1.2, (h, w)).astype(np.float32)
        depth_qty = self._depth(_depth_item(z, k), _config((h, w)))

        assert str(depth_qty.units) == "meter"
        assert depth_qty.magnitude.dtype == torch.float64
        np.testing.assert_allclose(depth_qty.magnitude.numpy(), _true_range(z.astype(np.float64), k), rtol=1e-12)

    def test_transform_intrinsics_are_used_over_the_sensor(self):
        # Intrinsics within INTRINSICS_RTOL of the sensor's still convert the depth exactly as it was rendered
        h, w = 9, 12
        k = _intrinsics(h, w, 60.0)
        k = {**k, "fl_x": k["fl_x"] * (1 + INTRINSICS_RTOL / 2), "cy": k["cy"] * (1 - INTRINSICS_RTOL / 2)}
        z = np.full((h, w), Z0)
        r = self._depth(_depth_item(z, k), _config((h, w))).magnitude.numpy()

        np.testing.assert_allclose(r, _true_range(z, k), rtol=1e-12)
        assert not np.allclose(r, _true_range(z, _intrinsics(h, w, 60.0)), rtol=1e-9, atol=0)

    def test_transform_without_size_is_used(self):
        h, w = 9, 12
        k = {key: v for key, v in _intrinsics(h, w, 50.0).items() if key not in ("h", "w")}
        z = np.full((h, w), Z0)
        r = self._depth(_depth_item(z, k), _config((h, w), fov=50.0)).magnitude.numpy()
        np.testing.assert_allclose(r, _true_range(z, k), rtol=1e-12)

    def test_transform_of_another_size_raises(self):
        k = _intrinsics(9, 12, 60.0)
        with pytest.raises(ValueError, match="does not match"):
            self._depth(_depth_item(np.full((8, 12), Z0), k), _config((8, 12)))

    @pytest.mark.parametrize("depth_type", ["planar_z", "range"])
    @pytest.mark.parametrize("name", list(MISMATCHED))
    def test_transform_of_another_field_of_view_raises(self, name, depth_type):
        """The falloff and the incidence cosine follow the rays of the sensor's field of view. A 30 deg render on a
        60 deg sensor was accepted, and its pixels got the falloff and incidence cosines of the 60 deg rays."""
        with pytest.raises(ValueError, match="not those of the sensor's field of view"):
            self._depth(_depth_item(np.full((9, 12), Z0), MISMATCHED[name]), _config((9, 12)), depth_type=depth_type)

    @pytest.mark.parametrize("frame", ["albedo", "ambient"])
    def test_albedo_or_ambient_of_another_field_of_view_raises(self, frame):
        h, w = 9, 12
        frames = {"albedo": np.ones((h, w)), "ambient": np.zeros((h, w))}
        frames[frame] = (frames[frame][..., None], {"file_path": f"{frame}/0000.exr", **MISMATCHED["narrower"]})
        with pytest.raises(ValueError, match=f"{frame} frame"):
            preproc_frames(np.full((h, w), Z0), frames["albedo"], frames["ambient"], _config((h, w)))

    def test_transform_of_the_same_field_of_view_at_another_resolution_is_used(self):
        """Guard: a render at twice the sensor's resolution spans the sensor's field of view."""
        k = _intrinsics(18, 24, 60.0)
        z = np.random.default_rng(5).uniform(0.2, 1.2, (18, 24))
        r = self._depth(_depth_item(z, k), _config((9, 12))).magnitude

        full = torch.from_numpy(_true_range(z.astype(np.float32).astype(np.float64), k))
        expected = F.interpolate(full[None, None], size=(9, 12), mode="nearest")[0, 0]
        np.testing.assert_allclose(r.numpy(), expected.numpy(), rtol=1e-12)

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda z: z, id="array"),
            pytest.param(torch.from_numpy, id="tensor"),
            pytest.param(lambda z: (z[..., np.newaxis], {"file_path": "depths/0000.exr"}), id="item-without-intrinsics"),
        ],
    )
    def test_frame_without_intrinsics_uses_the_sensor(self, wrap):
        h, w = 8, 12
        config = _config((h, w), fov=(70.0, 50.0))
        z = np.random.default_rng(2).uniform(0.2, 1.2, (h, w)).astype(np.float32)
        r = self._depth(wrap(z), config).magnitude.numpy()

        k = {"fl_x": 6 / np.tan(np.radians(35.0)), "fl_y": 4 / np.tan(np.radians(25.0)), "cx": 6.0, "cy": 4.0}
        np.testing.assert_allclose(r, _true_range(z.astype(np.float64), k), rtol=1e-12)

        sensor = SPADSensor(**config["sensor"])
        assert (sensor.f_x_px, sensor.f_y_px, sensor.c_x, sensor.c_y) == pytest.approx(tuple(k.values()), rel=1e-12)

    def test_scalar_sensor_fov_gives_square_pixels(self):
        h, w = 8, 12
        z = np.full((h, w), Z0)
        r = self._depth(z, _config((h, w), fov=60.0)).magnitude.numpy()
        fl = 6 / np.tan(np.radians(30.0))
        np.testing.assert_allclose(r, _true_range(z, {"fl_x": fl, "fl_y": fl, "cx": 6.0, "cy": 4.0}), rtol=1e-12)

    def test_sensor_intrinsics_are_scaled_to_the_frame(self):
        # A frame at twice the sensor's resolution still spans the sensor's field of view
        config = _config((4, 6), fov=(60.0, 45.0))
        z = np.random.default_rng(3).uniform(0.2, 1.2, (8, 12))
        r = self._depth(z, config).magnitude

        k = {"fl_x": 6 / np.tan(np.radians(30.0)), "fl_y": 4 / np.tan(np.radians(22.5)), "cx": 6.0, "cy": 4.0}
        full = torch.from_numpy(_true_range(z.astype(np.float32).astype(np.float64), k))
        expected = F.interpolate(full[None, None], size=(4, 6), mode="nearest")[0, 0]
        assert r.shape == (4, 6)
        np.testing.assert_allclose(r.numpy(), expected.numpy(), rtol=1e-12)

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda z: z, id="array"),
            pytest.param(lambda z: _depth_item(z, _intrinsics(*z.shape, 60.0)), id="depths-item"),
        ],
    )
    def test_range_is_used_as_is(self, wrap):
        z = np.random.default_rng(4).uniform(0.2, 1.2, (9, 12)).astype(np.float32)
        r = self._depth(wrap(z), _config((9, 12)), depth_type="range").magnitude.numpy()
        np.testing.assert_array_equal(r, z.astype(np.float64))

    @pytest.mark.parametrize("depth_type", ["z", "Range", "planar", "depth", "", None])
    def test_unknown_depth_type_raises(self, depth_type):
        with pytest.raises(ValueError, match="depth_type"):
            self._depth(np.full((2, 2), Z0), _config((2, 2)), depth_type=depth_type)

    def test_missed_and_invalid_depths_stay_no_return(self):
        z = np.full((3, 4), Z0)
        z[0, :] = [1e10, np.nan, np.inf, -1.0]
        z[1, 0] = 0.0
        k = _intrinsics(3, 4, 60.0)
        r = self._depth(_depth_item(z, k), _config((3, 4))).magnitude.numpy()

        assert r[0].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert r[1, 0] == 0.0
        valid = np.ones((3, 4), dtype=bool)
        valid[0], valid[1, 0] = False, False
        np.testing.assert_allclose(r[valid], _true_range(np.full((3, 4), Z0), k)[valid], rtol=1e-12)

    def test_integer_depth_image_is_converted(self):
        # An 8-bit depth image is scaled to [0, max_depth] and then converted like any planar depth
        k = _intrinsics(3, 4, 60.0)
        raw = np.full((3, 4, 1), 128, dtype=np.uint8)
        r = self._depth((raw, {"file_path": "depths/0000.png", **k}), _config((3, 4))).magnitude.numpy()
        z = np.full((3, 4), np.float32(128) / np.float32(255.0) * np.float32(N_BINS * BIN_M), dtype=np.float64)
        np.testing.assert_allclose(r, _true_range(z, k), rtol=1e-6)


# =========================================================================== #
# Both the time bin and the 1/d^2 falloff use the range                        #
# =========================================================================== #
class TestRangeInTheEmulator:
    @pytest.mark.parametrize("theta_deg", [15.0, 30.0])
    def test_off_axis_pixel_bins_and_falls_off_with_its_range(self, theta_deg):
        # Two pixels whose centers are theta_deg left and right of the optical axis, at planar depth Z0: a 1 x 2
        # sensor whose field of view is 2 * atan(2 * tan(theta)) wide and 2 * theta high
        theta = np.radians(theta_deg)
        fl = 0.5 / np.tan(theta)
        k = {"fl_x": fl, "fl_y": fl, "cx": 1.0, "cy": 0.5, "h": 1, "w": 2}
        config = _config((1, 2), fov=(np.degrees(2 * np.arctan(1 / fl)), 2 * theta_deg))
        ones = np.ones((1, 2))
        depth_qty, albedo_qty, _ = preproc_frames(_depth_item(np.full((1, 2), Z0), k), ones, 0 * ones, config)
        t = _transient(depth_qty, albedo_qty, config)

        rng = Z0 / np.cos(theta)  # 0.5176 m at 15 deg (bin 414), 0.5774 m at 30 deg (bin 461); Z0 is bin 400
        assert np.flatnonzero(t).tolist() == [int(np.floor(rng / BIN_M))]

        # The planar depth taken as range would be brighter by 1 / cos^2(theta)
        z_qty, z_albedo, _ = preproc_frames(np.full((1, 2), Z0), ones, 0 * ones, config, depth_type="range")
        t_z = _transient(z_qty, z_albedo, config)
        assert np.flatnonzero(t_z).tolist() == [int(np.floor(Z0 / BIN_M))]
        assert t.sum() / t_z.sum() == pytest.approx(np.cos(theta) ** 2, rel=1e-6)

        # And the same as the true range given as range
        r_qty, r_albedo, _ = preproc_frames(np.full((1, 2), rng), ones, 0 * ones, config, depth_type="range")
        np.testing.assert_allclose(t, _transient(r_qty, r_albedo, config), rtol=1e-6)

    def test_plane_covers_the_bins_of_its_ranges(self):
        # A fronto-parallel plane at Z0 seen by a 60 deg sensor: one depth, but ranges from Z0 to Z0 / cos(39.2 deg)
        n = 65
        k = _intrinsics(n, n, 60.0)
        config = _config((n, n))
        z = np.full((n, n), Z0)
        ones = np.ones((n, n))
        depth_qty, albedo_qty, _ = preproc_frames(_depth_item(z, k), ones, 0 * ones, config)
        t = _transient(depth_qty, albedo_qty, config)

        rng = _true_range(z, k)
        bins = np.unique(np.floor(rng / BIN_M).astype(int))
        assert np.flatnonzero(t).tolist() == bins.tolist()
        assert bins.size > 100

        r_qty, r_albedo, _ = preproc_frames(rng, ones, 0 * ones, config, depth_type="range")
        np.testing.assert_allclose(t, _transient(r_qty, r_albedo, config), rtol=1e-6)


# =========================================================================== #
# CLI: a rendered depths/ directory                                            #
# =========================================================================== #
class TestCliDepths:
    N = 17
    ARGS = {
        "sensor_size": (N, N),
        "sensor_fov": (60.0, 60.0),
        "n_bins": N_BINS,
        "bin_width": BIN_M,
        "max_depth": N_BINS * BIN_M,
        "active_pulse_repetition": F_REP / 1e6,
        "active_pulse_width": 0.05,
        "active_avg_watts": 1e-4,
        "ambient_enabled": False,
        "free_running": False,
        "fast_sim": True,
    }

    @staticmethod
    def _write_dataset(root, data, intrinsics):
        """One (H, W, C) frame, stored as a stack of frames in a .npy file, with its transforms.json."""
        root.mkdir(parents=True)
        np.save(root / "frames.npy", data[np.newaxis])
        frame = {"file_path": "frames.npy", "offset": 0, "transform_matrix": np.eye(4).tolist()}
        Metadata.from_frames([frame], camera={**intrinsics, "c": data.shape[-1]}).save(root / "transforms.json")

    def test_cli_converts_rendered_planar_depth_to_range(self, tmp_path):
        from visionsim.cli.emulate import aspc as aspc_cli

        n = self.N
        k = _intrinsics(n, n, 60.0)
        z = np.full((n, n), Z0)
        self._write_dataset(tmp_path / "in" / "depths", z.astype(np.float32)[..., np.newaxis], k)
        self._write_dataset(tmp_path / "in" / "diffuse" / "color", np.ones((n, n, 3), np.float32), k)
        self._write_dataset(tmp_path / "in" / "frames", np.zeros((n, n, 3), np.float32), k)

        aspc_cli(input_dir=tmp_path / "in", output_dir=tmp_path / "out", **self.ARGS)
        hist = np.load(tmp_path / "out" / "frames" / "histogram_0000.npy")

        emulator = ASPCEmulator(prepare_config(dict(self.ARGS)))
        albedo = ureg.Quantity(torch.ones(n, n, dtype=torch.float64), "dimensionless")
        ambient = ureg.Quantity(torch.zeros(n, n, dtype=torch.float64), "dimensionless")
        expected = emulator.process_frame(ureg.Quantity(torch.from_numpy(_true_range(z, k)), "meter"), albedo, ambient)
        np.testing.assert_allclose(hist, expected, rtol=1e-5)

        as_range = emulator.process_frame(ureg.Quantity(torch.from_numpy(z), "meter"), albedo, ambient)
        assert not np.allclose(hist, as_range, rtol=1e-2)

    def test_cli_raises_when_the_render_has_another_field_of_view(self, tmp_path):
        """A 30 deg render emulated by a 60 deg sensor got the falloff and incidence cosines of 60 deg."""
        from visionsim.cli.emulate import aspc as aspc_cli

        n = self.N
        k = _intrinsics(n, n, 30.0)
        self._write_dataset(tmp_path / "in" / "depths", np.full((n, n, 1), Z0, np.float32), k)
        self._write_dataset(tmp_path / "in" / "diffuse" / "color", np.ones((n, n, 3), np.float32), k)
        self._write_dataset(tmp_path / "in" / "frames", np.zeros((n, n, 3), np.float32), k)
        with pytest.raises(ValueError, match="set sensor.fov to the field of view of the render"):
            aspc_cli(input_dir=tmp_path / "in", output_dir=tmp_path / "out", **self.ARGS)
