"""YAML configuration loading: ``_load_config`` and the ``!Quantity`` / ``!expr`` / ``!file`` tags."""

from pathlib import Path

import pytest
from pint import Quantity

import visionsim.emulate.aspc as aspc
from visionsim.emulate.aspc.utils import _load_config, ureg

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
