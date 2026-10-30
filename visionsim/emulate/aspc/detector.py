from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pint import Quantity

from visionsim.emulate.aspc.ambient import Sun
from visionsim.emulate.aspc.emitter import PulsedLaser
from visionsim.emulate.aspc.histogrammers import HistogrammerBase
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.utils import (
    histogram_bin_width,
    incidence_cosine,
    irradiance_photons,
    tof2depth,
    ureg,
    radiance_photons,
)

logger = logging.getLogger(__name__)


@functools.cache
def _log_without_normals() -> None:
    """Logs, once per process, that frames without a normal frame have no incidence cosine."""
    logger.warning(
        "No normal frame given: the laser return has no incidence cosine, as if every surface faced the sensor."
    )


class Detector:
    """
    Unified Detector module combining physical camera optics (SPADSensor),
    spatial pixel FOV masking, and readout processing (Histogrammer).

    Frames passed to the detector span the sensor's field of view, and each pixel is weighted by the sensor's
    ``falloff`` (see :class:`~visionsim.emulate.aspc.sensors.SensorBase`). A zone's transient is the mean of its
    pixels', i.e. photons per pixel of the zone.

    The histogram has the histogrammer's ``n_bins`` bins of ``bin_width`` (see :meth:`get_bin_width`), starting
    ``time_offset`` before the laser pulse (see :class:`~visionsim.emulate.aspc.histogrammers.HistConfig`).
    """

    def __init__(
        self,
        sensor: SPADSensor,
        histogrammer: HistogrammerBase,
        device: str | torch.device = "cpu",
        pixel_fov_list: list[list[float]] | str | Path | None = None,
    ):
        self.sensor = sensor
        self.histogrammer: Any = histogrammer
        self.device = torch.device(device) if isinstance(device, str) else device

        # Load or set registered FOV coordinates
        parsed_fov_list: list[list[float]]
        if isinstance(pixel_fov_list, (str, Path)):
            path = Path(pixel_fov_list)
            if not path.exists():
                # A mistyped path must not silently replace the zones with the full frame
                raise FileNotFoundError(f"pixel_fov_list file '{path}' does not exist.")
            raw_data = np.loadtxt(path, dtype=np.float32).tolist()
            if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], (float, int)):
                parsed_fov_list = [raw_data]
            else:
                parsed_fov_list = raw_data
        elif pixel_fov_list is None:
            parsed_fov_list = [[0.0, 1.0, 0.0, 1.0]]
        else:
            parsed_fov_list = pixel_fov_list

        self.pixel_fov_list: list[list[float]] = parsed_fov_list

        # Cached simulation state
        self.transients: torch.Tensor | None = None
        self.arrival_rates: torch.Tensor | None = None
        self.ambient_offsets: list[torch.Tensor] | torch.Tensor | None = None
        # Arguments of the last calculate_transients call but the FOV masks, to spread the same returns by the pulse
        self._transient_args: dict[str, Any] | None = None

    def build_pixel_fov_list(
        self,
        per_pixel_fov: Quantity | tuple[Quantity, Quantity],
        output_path: str | None = None,
    ) -> list:
        """Builds normalized FOV bounds [r1, r2, c1, c2] for every pixel in sensor array."""
        if isinstance(per_pixel_fov, Quantity):
            fov_h = per_pixel_fov
            fov_w = per_pixel_fov
        else:
            fov_h, fov_w = per_pixel_fov[0], per_pixel_fov[1]

        fov_w_ratio = (fov_w / self.sensor.fov_x).to(ureg.dimensionless).magnitude
        fov_h_ratio = (fov_h / self.sensor.fov_y).to(ureg.dimensionless).magnitude

        pixel_fov_list = []
        for row in range(self.sensor.h):
            cy = (row + 0.5) / self.sensor.h
            for col in range(self.sensor.w):
                cx = (col + 0.5) / self.sensor.w
                r1 = max(0.0, cy - fov_h_ratio / 2.0)
                r2 = min(1.0, cy + fov_h_ratio / 2.0)
                c1 = max(0.0, cx - fov_w_ratio / 2.0)
                c2 = min(1.0, cx + fov_w_ratio / 2.0)
                pixel_fov_list.append([r1, r2, c1, c2])

        if output_path:
            np.savetxt(output_path, np.array(pixel_fov_list, dtype=np.float32), fmt="%.6f", delimiter=" ")

        self.pixel_fov_list = pixel_fov_list
        return pixel_fov_list

    def get_pixel_fov_mask(
        self,
        empty_mask: torch.Tensor,
        row1: float,
        row2: float,
        col1: float,
        col2: float,
        vignette: bool = True,
    ) -> torch.Tensor:
        """Generates a 2D spatial FOV mask with optional smooth vignetting."""
        img_rows, img_cols = empty_mask.shape

        r1 = max(0, min(img_rows, round(row1 * img_rows)))
        r2 = max(0, min(img_rows, round(row2 * img_rows)))
        c1 = max(0, min(img_cols, round(col1 * img_cols)))
        c2 = max(0, min(img_cols, round(col2 * img_cols)))

        mask = torch.zeros((img_rows, img_cols), dtype=torch.float32, device=self.device)

        if r1 == r2 or c1 == c2:
            return mask

        if vignette:
            rr = torch.arange(r1, r2, device=self.device)
            cc = torch.arange(c1, c2, device=self.device)
            yy, xx = torch.meshgrid(rr, cc, indexing="ij")

            cy = (r1 + r2 - 1) / 2.0
            cx = (c1 + c2 - 1) / 2.0
            half_h = max((r2 - r1) / 2.0, 1e-6)
            half_w = max((c2 - c1) / 2.0, 1e-6)

            dy = (yy - cy) / half_h
            dx = (xx - cx) / half_w
            radial = torch.sqrt(dx * dx + dy * dy)

            vignette_weights = torch.clamp(1.0 - torch.pow(radial, 2.0), 0.0, 1.0)
            mask[r1:r2, c1:c2] = vignette_weights
        else:
            mask[r1:r2, c1:c2] = 1.0

        return mask

    def get_perpixel_fov_masks(self, empty_mask: torch.Tensor, vignette: bool = True) -> torch.Tensor:
        """Generates stacked FOV mask tensors for all registered pixel FOVs."""
        mask_list = []
        for r1, r2, c1, c2 in self.pixel_fov_list:
            mask = self.get_pixel_fov_mask(empty_mask, r1, r2, c1, c2, vignette=vignette)
            mask_list.append(mask.unsqueeze(0))

        return torch.concat(mask_list)

    def get_bin_width(self, emitter: PulsedLaser) -> Quantity:
        """Returns the histogram's bin width as a one-way distance (c * t / 2), in meters.

        It is the histogrammer's ``bin_width`` if set, and otherwise one laser period divided into ``n_bins`` bins, see
        :func:`~visionsim.emulate.aspc.utils.histogram_bin_width`. It sets the bins of the returns, the bins the pulse
        is integrated over and the dead time in bins; the photons per laser cycle follow from the repetition rate.

        Raises:
            ValueError: if the histogram's window, ``n_bins * bin_width``, exceeds one laser period, or if a
                free-running SPAD with dead time has a window shorter than the period. Such a SPAD also detects
                photons between the end of the window and the next pulse, which is not simulated.
        """
        bin_width = histogram_bin_width(
            self.histogrammer.n_bins, emitter.pulse_repetition, getattr(self.histogrammer, "bin_width", None)
        )

        if getattr(self.histogrammer, "free_running", False) and self._get_dead_time_bins(bin_width) > 0:
            period = emitter.max_resolvable_depth.to(ureg.meter)
            # Like the dead time, the period is simulated in whole bins
            if round((period / bin_width).to(ureg.dimensionless).magnitude) > self.histogrammer.n_bins:
                raise ValueError(
                    f"A free-running SPAD with dead time needs a histogram window of one laser period, got "
                    f"n_bins x bin_width = {self.histogrammer.n_bins * bin_width:.6g~P} for a period of "
                    f"{period:.6g~P}. Set free_running to False (a SPAD re-armed at every laser cycle), or leave "
                    "bin_width unset."
                )
        return bin_width

    def _get_time_offset(self) -> Quantity:
        """The histogrammer's ``time_offset`` as a one-way distance (c * t / 2), in meters."""
        time_offset = getattr(self.histogrammer, "time_offset", None)
        return 0.0 * ureg.meter if time_offset is None else tof2depth(time_offset).to(ureg.meter)

    def _get_dead_time_bins(self, bin_width: Quantity) -> int:
        """The histogrammer's ``dead_time`` in whole bins of ``bin_width``."""
        return int(np.round((tof2depth(self.histogrammer.dead_time) / bin_width).to(ureg.dimensionless).magnitude))

    def _get_falloff(self, shape: tuple[int, ...]) -> torch.Tensor:
        """The sensor's off-axis falloff for frames of shape ``(..., H, W)``."""
        h, w = shape[-2:]
        return torch.as_tensor(self.sensor.pixel_falloff((h, w)), dtype=torch.float32, device=self.device)

    def _get_incidence_cosine(self, normal_frame: torch.Tensor | Quantity, shape: tuple[int, ...]) -> torch.Tensor:
        """|cos| of the angle between each pixel's camera-frame normal and its ray, for frames of shape (..., H, W)."""
        normals = normal_frame.magnitude if hasattr(normal_frame, "magnitude") else normal_frame
        normals = torch.as_tensor(normals)
        h, w = shape[-2:]
        if normals.ndim < 3 or normals.shape[-1] != 3 or tuple(normals.shape[-3:-1]) != (h, w):
            raise ValueError(f"Expected a normal frame of shape (..., {h}, {w}, 3), got shape {tuple(normals.shape)}.")
        return torch.as_tensor(incidence_cosine(normals, *self.sensor.frame_intrinsics((h, w))))

    def _get_signal(
        self,
        depth_frame: torch.Tensor,
        albedo_frame: torch.Tensor,
        emitter: PulsedLaser,
        normal_frame: torch.Tensor | Quantity | None = None,
    ) -> torch.Tensor:
        """Converts active laser radiance to focal plane irradiance (#photons).

        With a ``normal_frame``, the radiance of each surface scales with its incidence cosine; without one, every
        surface is taken to face the sensor. The camera equation then applies the sensor's off-axis falloff.
        """
        depth_ref = depth_frame.magnitude if hasattr(depth_frame, "magnitude") else depth_frame
        cos_incidence: torch.Tensor | float = 1.0
        if normal_frame is not None:
            cos_incidence = self._get_incidence_cosine(normal_frame, depth_ref.shape)

        num_pixels = self.sensor.w * self.sensor.h
        radiance = emitter.get_scene_radiance(albedo_frame, depth_frame, num_pixels, self.sensor.omega, cos_incidence)
        irradiance = (radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(irradiance_photons) * (
            self.sensor.pixel_pitch.to(ureg.meter)
        ) ** 2

        irradiance = torch.as_tensor(irradiance.magnitude, dtype=torch.float32, device=self.device)
        return irradiance * self._get_falloff(irradiance.shape)

    def _get_ambient_offset(
            self,
            albedo_frame: torch.Tensor | Quantity,
            emitter: PulsedLaser,
            scene: Sun | None = None,
            ambient_frame: torch.Tensor | Quantity | None = None,
        ) -> torch.Tensor:
            """Converts ambient background radiance to focal plane offset irradiance (#photons).

            Like the laser return, it is weighted by the sensor's off-axis falloff.
            """
            # 1. Primary path: Use rendered ambient spatial frame scaled by physical radiance
            if ambient_frame is not None:
                if scene is not None:
                    base_radiance = scene.get_scene_radiance(1.0, emitter.pulse_repetition)
                else:
                    base_radiance = 1.0 * radiance_photons

                spatial_radiance = ambient_frame * base_radiance
                ambient_irradiance = (
                    spatial_radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2
                ).to(irradiance_photons) * (self.sensor.pixel_pitch.to(ureg.meter)) ** 2

                offsets = torch.as_tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)
                return offsets * self._get_falloff(offsets.shape)

            # 2. Secondary path: Fall back to analytical Sun model
            elif scene is not None:
                ambient_radiance = scene.get_scene_radiance(albedo_frame, emitter.pulse_repetition)
                ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(
                    irradiance_photons
                ) * (self.sensor.pixel_pitch.to(ureg.meter)) ** 2

                offsets = torch.as_tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)
                return offsets * self._get_falloff(offsets.shape)

            # 3. Default path: No ambient source
            else:
                raw_ref = albedo_frame.magnitude if hasattr(albedo_frame, "magnitude") else albedo_frame
                return torch.zeros(raw_ref.shape, dtype=torch.float32, device=self.device)
    def get_transients(
        self,
        depth_frame: torch.Tensor | Quantity,
        albedo_frame: torch.Tensor | Quantity,
        emitter: PulsedLaser,
        scene: Sun | None = None,
        ambient_frame: torch.Tensor | Quantity | None = None,
        normal_frame: torch.Tensor | Quantity | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calculates spatial-temporal transients over laser ToF bins.

        ``depth_frame`` is each pixel's range (the distance along its ray, not planar depth), which sets both its bin
        and its 1 / range^2 falloff. See :func:`~visionsim.emulate.aspc.utils.planar_depth_to_range`. A return at
        range r arrives ``2 * r / c + time_offset`` after the start of the histogram's window, modulo one laser period,
        and lands in one of its ``n_bins`` bins of :meth:`get_bin_width`, or, if it arrives after the window, is not
        recorded (see :meth:`~visionsim.emulate.aspc.histogrammers.HistogrammerBase.calculate_transients`). The
        transients hold no pulse, as an ideal time-to-digital converter records the returns; :meth:`get_arrival_rates`
        adds the pulse.

        ``normal_frame`` holds each pixel's surface normal in the camera frame (+X right, +Y up, looking along -Z),
        with shape ``(..., H, W, 3)``, as :func:`~visionsim.emulate.aspc.utils.preproc_normals` returns it. The laser
        return then scales with ``|cos|`` of the angle between the normal and the pixel's ray. Without a normal
        frame there is no incidence cosine, i.e. every surface is taken to face the sensor, which overestimates
        oblique surfaces; this is logged once.

        Every pixel is weighted by the sensor's off-axis falloff. With a falloff, the zones are not rescaled, since
        each pixel already carries its own solid angle; with ``falloff="none"`` each zone is rescaled by
        :func:`~visionsim.emulate.aspc.utils.get_irradiance_with_fov`, as before.
        """
        bin_width = self.get_bin_width(emitter)
        if normal_frame is None:
            _log_without_normals()

        irradiance = self._get_signal(depth_frame, albedo_frame, emitter, normal_frame)
        offsets = self._get_ambient_offset(
            albedo_frame=albedo_frame,
            emitter=emitter,
            scene=scene,
            ambient_frame=ambient_frame,
        )

        # Without a falloff every pixel has the same weight, and each zone is rescaled by a solid angle of its size
        rescale_zones = self.sensor.falloff == "none"
        transient_args = dict(
            irradiance_frames=irradiance,
            depth_frames=depth_frame,
            offsets=offsets,
            gt_ntime_bins=self.histogrammer.n_bins,
            max_depth=emitter.max_resolvable_depth.to(ureg.meter).magnitude,
            sensor_fov=self.sensor.fov if rescale_zones else None,
            pixel_fov_list=self.pixel_fov_list,
            w=self.sensor.w,
            h=self.sensor.h,
            omega=self.sensor.omega if rescale_zones else None,
            bin_width=bin_width.magnitude,
            time_offset=self._get_time_offset().magnitude,
        )
        transients, ambient_offsets = self.histogrammer.calculate_transients(
            fov_masks=self._get_fov_masks(depth_frame), **transient_args
        )

        self.transients = transients
        self.ambient_offsets = ambient_offsets
        self._transient_args = transient_args
        return transients, ambient_offsets

    def _get_fov_masks(self, depth_frame: torch.Tensor | Quantity) -> torch.Tensor:
        """The zones' FOV masks for frames of the shape of ``depth_frame``, ``(H, W)`` or ``(N, H, W)``."""
        empty_ref = depth_frame.magnitude if hasattr(depth_frame, "magnitude") else depth_frame
        if empty_ref.dim() == 3:
            empty_ref = empty_ref[0]

        vignette_flag = getattr(self.histogrammer, "vignette", True)
        return self.get_perpixel_fov_masks(empty_ref, vignette=vignette_flag)

    def get_arrival_rates(self, emitter: PulsedLaser) -> torch.Tensor:
        """Spreads the returns of the last :meth:`get_transients` over the bins by the pulse, and adds the ambient.

        Each return is spread from its exact arrival time, not from the start of its bin: bin j gets the fraction of
        the pulse that arrives during bin j (see :meth:`~visionsim.emulate.aspc.emitter.DynamicSource.get_pulse_cdf`
        and :meth:`~visionsim.emulate.aspc.histogrammers.HistogrammerBase.calculate_transients`), so the arrival rates
        keep where in its bin each return arrived, which the binned transients do not. With a pulse of zero width the
        arrival rates are the transients plus the ambient offsets. Only the returns that arrive in the histogram's
        window are spread, and their pulse is cut at both ends of the window.
        """
        if self.transients is None or self._transient_args is None:
            raise RuntimeError("Transients not computed. Call get_transients() first.")

        pulse_transients, _ = self.histogrammer.calculate_transients(
            fov_masks=self._get_fov_masks(self._transient_args["depth_frames"]),
            pulse_cdf=emitter.get_pulse_cdf(self.get_bin_width(emitter)),
            **self._transient_args,
        )
        arrival_rates = self.histogrammer.calculate_arrival_rates(
            None, pulse_transients, self.ambient_offsets, self.histogrammer.n_bins
        )

        self.arrival_rates = arrival_rates
        return arrival_rates

    def get_ewh(self, emitter: PulsedLaser) -> list[torch.Tensor]:
        """Simulates SPAD Equi-Width Histogram (EWH) output. The dead time is rounded to whole bins of
        :meth:`get_bin_width`."""
        if self.arrival_rates is None:
            raise RuntimeError("Arrival rates not computed. Call get_arrival_rates() first.")

        dead_time_bins = self._get_dead_time_bins(self.get_bin_width(emitter))

        ewh_list = self.histogrammer.simulate_ewh(
            arrival_rates=self.arrival_rates,
            n_pulses=self.histogrammer.n_pulses,
            n_bins=self.histogrammer.n_bins,
            free_running=self.histogrammer.free_running,
            dead_time_bins=dead_time_bins,
            fast_sim=self.histogrammer.fast_sim,
            fast_sim_noise=self.histogrammer.fast_sim_noise,
            max_detections_per_cycle=self.histogrammer.max_detections_per_cycle,
        )
        return ewh_list
