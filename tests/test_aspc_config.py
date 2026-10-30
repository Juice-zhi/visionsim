"""Configuration: YAML loading with the ``!Quantity`` / ``!expr`` / ``!file`` tags, the max_depth check, and custom
pulse shapes from a config file or the CLI."""

from pathlib import Path

import numpy as np
import pytest
from pint import Quantity

import visionsim.emulate.aspc as aspc
from visionsim.emulate.aspc.emitter import PulsedLaser
from visionsim.emulate.aspc.utils import _load_config, c, prepare_config, preproc_frames, ureg

ASPC_DIR = Path(aspc.__file__).parent
SHIPPED_CONFIGS = [ASPC_DIR / "config.yaml", *sorted((ASPC_DIR / "examples").glob("*.yaml"))]


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


class TestShippedConfigs:
    def test_all_shipped_configs_are_found(self):
        assert len(SHIPPED_CONFIGS) == 4
        assert all(p.exists() for p in SHIPPED_CONFIGS)

    @pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    def test_shipped_config_loads(self, path):
        config = _load_config(path)
        sun = config["ambient_source"]["sun"]
        assert sun["temperature"] == 5778.0 * ureg.kelvin
        assert sun["intensity"].check(ureg.watt / ureg.meter**2)

    @pytest.mark.parametrize("path", SHIPPED_CONFIGS[1:], ids=lambda p: p.name)
    def test_shipped_fov_is_a_list_of_angles(self, path):
        fov = _load_config(path)["sensor"]["fov"]
        assert isinstance(fov, list) and len(fov) == 2
        assert all(isinstance(f, Quantity) and f.check(ureg.degree.dimensionality) for f in fov)


class TestTags:
    def test_quantity_scalar(self, tmp_path):
        config = _load_config(_write(tmp_path, "dead_time: !Quantity 10 ns\n"))
        assert config["dead_time"] == 10 * ureg.nanosecond

    def test_quantity_without_units_is_dimensionless(self, tmp_path):
        config = _load_config(_write(tmp_path, "f_number: !Quantity 1.4\n"))
        assert config["f_number"] == 1.4 * ureg.dimensionless

    def test_quantity_sequence_converts_each_item(self, tmp_path):
        config = _load_config(_write(tmp_path, "fov: !Quantity [30 degree, 20 degree]\n"))
        assert config["fov"] == [30 * ureg.degree, 20 * ureg.degree]

    def test_quantity_mapping_is_passed_as_keywords(self, tmp_path):
        config = _load_config(_write(tmp_path, "max_depth: !Quantity {value: 2.5, units: meter}\n"))
        assert config["max_depth"] == 2.5 * ureg.meter

    def test_expr_scalar(self, tmp_path):
        config = _load_config(_write(tmp_path, "n_pulses: !expr 10 * 1000\nangle: !expr math.pi / 2\n"))
        assert config["n_pulses"] == 10000
        assert config["angle"] == pytest.approx(1.5707963267948966)

    def test_expr_has_no_builtins(self, tmp_path):
        with pytest.raises(NameError):
            _load_config(_write(tmp_path, "x: !expr open('config.yaml')\n"))

    def test_file_is_relative_to_config(self, tmp_path):
        config = _load_config(_write(tmp_path, "pixel_fov_list: !file zones.txt\n"))
        assert Path(config["pixel_fov_list"]) == tmp_path / "zones.txt"


class TestMaxDepthCheck:
    """max_depth may equal the unambiguous range c / (2 * pulse_repetition) up to rounding, but not exceed it."""

    @staticmethod
    def _preproc(max_depth, pulse_rep_hz):
        config = {
            "sensor": {"size": (2, 2)},
            "active_source": {"pulsed_laser": {"pulse_repetition": pulse_rep_hz * ureg.hertz}},
            "histogrammer": {"max_depth": max_depth * ureg.meter},
        }
        frame = np.ones((2, 2), dtype=np.float32)
        return preproc_frames(frame, frame, frame, config)

    @staticmethod
    def _prepare(max_depth, pulse_rep_hz):
        user_args = {
            "max_depth": max_depth,
            "active_pulse_repetition": pulse_rep_hz / 1e6,
            "n_bins": 64,
            "bin_width": max_depth / 64,
        }
        return prepare_config(user_args)

    @pytest.mark.parametrize("max_depth", [1.28, 7.5, 3.0, 0.6])
    def test_unambiguous_range_is_accepted(self, max_depth):
        pulse_rep_hz = c / (2 * max_depth)
        self._preproc(max_depth, pulse_rep_hz)
        config = self._prepare(max_depth, pulse_rep_hz)
        assert config["histogrammer"]["max_depth"] == max_depth * ureg.meter

    @pytest.mark.parametrize("excess", [1.0001, 1.01])
    def test_beyond_unambiguous_range_is_rejected(self, excess):
        max_depth = 1.28
        pulse_rep_hz = c / (2 * max_depth)
        with pytest.raises(ValueError, match="exceeds max resolvable depth"):
            self._preproc(max_depth * excess, pulse_rep_hz)
        with pytest.raises(ValueError, match="exceeds max resolvable depth"):
            self._prepare(max_depth * excess, pulse_rep_hz)


CUSTOM_TAPS = [0, 1, 4, 2, 1]


def _custom_pulse_yaml(value, shape="custom"):
    return f"active_source:\n  pulsed_laser:\n    pulse_shape: {shape}\n    pulse_shape_custom: {value}\n"


def _kernel(config):
    laser = {k: v for k, v in config["active_source"]["pulsed_laser"].items() if k != "enabled"}
    return PulsedLaser(**laser).get_kernel(1 * ureg.centimeter, normalize=None)[1]


class TestCustomPulseConfig:
    """pulse_shape_custom from a config file, and a path given as the pulse shape."""

    def test_default_is_gaussian(self):
        laser = prepare_config({})["active_source"]["pulsed_laser"]
        assert laser["pulse_shape"] == "gaussian"
        assert laser["pulse_shape_custom"] is None

    def test_config_file_keeps_taps(self, tmp_path):
        config = prepare_config({}, _write(tmp_path, _custom_pulse_yaml(CUSTOM_TAPS)))
        assert config["active_source"]["pulsed_laser"]["pulse_shape_custom"] == CUSTOM_TAPS
        np.testing.assert_array_equal(_kernel(config), CUSTOM_TAPS)

    def test_config_file_keeps_file_tag(self, tmp_path):
        np.savetxt(tmp_path / "irf.txt", CUSTOM_TAPS)
        config = prepare_config({}, _write(tmp_path, _custom_pulse_yaml("!file irf.txt")))
        assert Path(config["active_source"]["pulsed_laser"]["pulse_shape_custom"]) == tmp_path / "irf.txt"
        np.testing.assert_array_equal(_kernel(config), CUSTOM_TAPS)

    def test_config_file_keeps_expr_callable(self, tmp_path):
        config = prepare_config({}, _write(tmp_path, _custom_pulse_yaml("!expr 'lambda u: np.exp(-(u**2))'")))
        assert callable(config["active_source"]["pulsed_laser"]["pulse_shape_custom"])
        kernel = _kernel(config)
        assert kernel.size > 1 and int(np.argmax(kernel)) == kernel.size // 2

    def test_path_as_pulse_shape_selects_custom_taps(self, tmp_path):
        path = tmp_path / "irf.txt"
        np.savetxt(path, CUSTOM_TAPS)
        config = prepare_config({"active_pulse_shape": path})
        laser = config["active_source"]["pulsed_laser"]
        assert laser["pulse_shape"] == "custom"
        assert laser["pulse_shape_custom"] == path
        np.testing.assert_array_equal(_kernel(config), CUSTOM_TAPS)


class TestCliPulseShape:
    """``--active-pulse-shape``: a name, or the path to a file of custom taps."""

    @staticmethod
    def _cli_laser(monkeypatch, *args):
        """Parses the arguments of ``emulate.aspc`` and returns the laser config, without emulating anything."""
        import tyro

        import visionsim.emulate.aspc.utils as utils
        from visionsim.cli.emulate import aspc as aspc_cli

        prepare, prepared = utils.prepare_config, []

        class Prepared(Exception):
            pass

        def prepare_and_stop(user_args, config_file=None):
            prepared.append(prepare(user_args, config_file))
            raise Prepared

        monkeypatch.setattr(utils, "prepare_config", prepare_and_stop)
        with pytest.raises(Prepared):
            tyro.cli(aspc_cli, args=["--input-dir", "in", "--output-dir", "out", *args])
        return prepared[0]["active_source"]["pulsed_laser"]

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_named_shape(self, monkeypatch, shape):
        laser = self._cli_laser(monkeypatch, "--active-pulse-shape", shape)
        assert laser["pulse_shape"] == shape

    def test_path_is_a_custom_pulse(self, monkeypatch, tmp_path):
        path = tmp_path / "irf.txt"
        np.savetxt(path, CUSTOM_TAPS)
        laser = self._cli_laser(monkeypatch, "--active-pulse-shape", str(path))
        assert laser["pulse_shape"] == "custom"
        assert laser["pulse_shape_custom"] == path

    def test_custom_takes_taps_from_config_file(self, monkeypatch, tmp_path):
        config_file = _write(tmp_path, _custom_pulse_yaml(CUSTOM_TAPS, shape="gaussian"))
        laser = self._cli_laser(monkeypatch, "--config-file", str(config_file), "--active-pulse-shape", "custom")
        assert laser["pulse_shape"] == "custom"
        assert laser["pulse_shape_custom"] == CUSTOM_TAPS

    @pytest.mark.parametrize("missing_file", [False, True], ids=["custom-without-taps", "missing-file"])
    def test_invalid_pulse_fails_before_the_output_is_touched(self, tmp_path, missing_file):
        from visionsim.cli.emulate import aspc as aspc_cli

        shape = tmp_path / "missing_irf.txt" if missing_file else "custom"
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "previous.txt").write_text("previous run")
        with pytest.raises(FileNotFoundError if missing_file else ValueError, match="pulse_shape_custom"):
            aspc_cli(input_dir=tmp_path / "in", output_dir=output_dir, active_pulse_shape=shape, force=True)
        assert (output_dir / "previous.txt").exists()
