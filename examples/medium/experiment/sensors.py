"""Sensor models and losses used to compare ways of rendering participating media.

Each sensor has an expected (noise-free) measurement, on which losses are computed so that they reflect differences
in the rendered scene rather than sensor noise, and a noisy measurement, emulated with visionsim, for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from visionsim.emulate.dvs import EventEmulator
from visionsim.emulate.rgb import emulate_rgb_from_sequence
from visionsim.utils.color import linearrgb_to_srgb

# --------------------------------------------------------------------------------------------------------------------
# RGB camera: exposures made of consecutive frames, with shot and read noise, clipping and quantization


@dataclass(frozen=True)
class RGBCamera:
    exposure: int = 5
    """number of 125 fps frames per exposure, i.e. a 25 fps camera with 40 ms exposures"""
    factor: float = 100.0
    """photons per frame for a unit linear radiance"""
    fwc: float = 500.0
    """full well capacity, in photons"""
    readout_std: float = 5.0
    bitdepth: int = 12

    def expected(self, frames: np.ndarray) -> np.ndarray:
        """Noise-free output, as 8-bit sRGB values in [0, 1], replicating `emulate_rgb_from_sequence`."""
        burst = len(frames)
        patch = np.sum(frames, axis=0) * self.factor
        patch = np.clip(patch / self.fwc, 0, 1.0)
        patch = np.round(patch * (2**self.bitdepth - 1)) / (2**self.bitdepth - 1)
        patch *= self.fwc / (burst * self.factor)
        patch = np.asarray(linearrgb_to_srgb(patch))
        return np.clip(np.round(patch * 255) / 255, 0, 1)

    def noisy(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = emulate_rgb_from_sequence(
            frames, readout_std=self.readout_std, fwc=self.fwc, bitdepth=self.bitdepth, factor=self.factor, rng=rng
        )
        return np.clip(out, 0, 1)


def psnr(x: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((x - reference) ** 2))
    return 10 * np.log10(1 / max(mse, 1e-12))


def ssim(x: np.ndarray, reference: np.ndarray, device: str = "cuda") -> float:
    """Mean SSIM over color channels, with the usual 11x11 Gaussian window (sigma 1.5), for images in [0, 1]."""
    a = torch.as_tensor(x, dtype=torch.float64, device=device).permute(2, 0, 1)[:, None]
    b = torch.as_tensor(reference, dtype=torch.float64, device=device).permute(2, 0, 1)[:, None]
    coords = torch.arange(11, dtype=torch.float64, device=device) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    window = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]

    def blur(v):
        return torch.nn.functional.conv2d(v, window)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(value.mean())


# --------------------------------------------------------------------------------------------------------------------
# Passive single photon camera: one binary frame per 125 fps frame


@dataclass(frozen=True)
class SPAD:
    factor: float = 1.0

    def probability(self, frame: np.ndarray) -> np.ndarray:
        """Probability of detecting at least one photon, per pixel and color channel."""
        return -np.expm1(-self.factor * np.clip(frame, 0, None))

    def noisy(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return (rng.random(frame.shape) < self.probability(frame)).astype(np.uint8)


def bernoulli_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Mean KL divergence, in bits, from Bernoulli(q) to Bernoulli(p), i.e. the information lost per pixel and
    frame when binary frames drawn from q are used in place of those drawn from p."""
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    return float(np.mean(p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))))


# --------------------------------------------------------------------------------------------------------------------
# Event camera: v2e, with thresholds but without noise, so that events only depend on the input frames


@dataclass
class EventCamera:
    fps: float = 125.0
    window: int = 5
    """number of frames over which events are accumulated for losses and previews (40 ms)"""
    threshold: float = 0.2
    emulator: EventEmulator = field(init=False)

    def __post_init__(self):
        self.emulator = EventEmulator(
            pos_thres=self.threshold,
            neg_thres=self.threshold,
            sigma_thres=0.0,
            cutoff_hz=200.0,
            leak_rate_hz=0.0,
            shot_noise_rate_hz=0.0,
            seed=42087,
        )

    def counts(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Number of events per window, pixel and polarity (ON, OFF), of shape (windows, h, w, 2)."""
        self.emulator.reset()
        h, w = frames[0].shape[:2]
        counts = np.zeros((len(frames) // self.window, h, w, 2), dtype=np.int32)
        for i, frame in enumerate(frames):
            luma = (np.clip(frame, 0, None)[..., :3] @ np.array([0.2126, 0.7152, 0.0722])) * 255.0
            events = self.emulator.generate_events(luma.astype(np.float64), i / self.fps)
            if events is None or len(events) == 0 or i // self.window >= len(counts):
                continue
            _, x, y, p = events.T
            np.add.at(counts[i // self.window], (y.astype(int), x.astype(int), (p < 0).astype(int)), 1)
        return counts


def event_f1(counts: np.ndarray, reference: np.ndarray) -> float:
    """Agreement of event counts per window, pixel and polarity, as an F1 score: 1 when identical, 0 when disjoint."""
    matched = np.minimum(counts, reference).sum()
    total = counts.sum() + reference.sum()
    return float(2 * matched / total) if total else 1.0


def event_preview(counts: np.ndarray) -> np.ndarray:
    """White background, ON events in blue and OFF events in red, as in visionsim's previews."""
    image = np.full((*counts.shape[:2], 3), 255, dtype=np.uint8)
    image[counts[..., 1] > 0] = (255, 0, 0)
    image[counts[..., 0] > 0] = (0, 0, 255)
    return image


def relative_l1(x: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(x - reference).sum() / max(np.abs(reference).sum(), 1e-12))
