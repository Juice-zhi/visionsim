"""Zone definitions: how ``Detector`` reads ``pixel_fov_list``."""

import numpy as np
import pytest

from visionsim.emulate.aspc.detector import Detector
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.utils import ureg

FULL_FRAME = [[0.0, 1.0, 0.0, 1.0]]


def _detector(pixel_fov_list):
    sensor = SPADSensor(size=(8, 8), fov=66 * ureg.degree)
    return Detector(sensor=sensor, histogrammer=Histogrammer(HistConfig()), pixel_fov_list=pixel_fov_list)


class TestPixelFovList:
    def test_none_is_the_full_frame(self):
        assert _detector(None).pixel_fov_list == FULL_FRAME

    def test_list_is_used_as_given(self):
        zones = [[0.0, 0.5, 0.0, 0.5], [0.5, 1.0, 0.5, 1.0]]
        assert _detector(zones).pixel_fov_list == zones

    def test_file_with_several_zones(self, tmp_path):
        zones = [[0.0, 0.5, 0.0, 0.5], [0.5, 1.0, 0.25, 0.75]]
        path = tmp_path / "zones.txt"
        np.savetxt(path, np.array(zones))
        assert _detector(path).pixel_fov_list == zones

    def test_file_with_one_zone(self, tmp_path):
        path = tmp_path / "zones.txt"
        np.savetxt(path, np.array([0.0, 0.5, 0.25, 0.75]))
        assert _detector(str(path)).pixel_fov_list == [[0.0, 0.5, 0.25, 0.75]]

    @pytest.mark.parametrize("as_str", [False, True])
    def test_missing_file_raises(self, tmp_path, as_str):
        path = tmp_path / "missing_zones.txt"
        with pytest.raises(FileNotFoundError, match="missing_zones.txt"):
            _detector(str(path) if as_str else path)
