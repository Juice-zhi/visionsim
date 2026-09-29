import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import exp1

from visionsim.medium import HeightFog, Homogeneous, Medium, camera_rays
from visionsim.medium.optics import henyey_greenstein
from visionsim.medium.transient import Flash, capture_histogram, estimate_distance, flash_transient

# A single pixel looking horizontally along +Y, from 1.5m above the ground
CAMERA = {"w": 1, "h": 1, "fl_x": 1.0, "fl_y": 1.0, "cx": 0.5, "cy": 0.5}
POSE = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1.5], [0, 0, 0, 1]])
# Pulses much shorter than bins, so that the transient is the per-bin impulse response
SHORT = Flash(bins=200, bin_width=1e-9, pulse_width=1e-12, background=0.0)


def transient(medium, distance, flash=SHORT, **kwargs):
    ones = np.ones((1, 1))
    return flash_transient(ones * distance, ones * 0.5, ones, CAMERA, POSE, medium, flash, **kwargs)


def test_homogeneous_backscatter_matches_exponential_integral():
    medium = Medium(extinction=0.05, anisotropy=0.7, components=[Homogeneous()])
    result = transient(medium, 1e10)
    edges = np.maximum(np.arange(SHORT.bins + 1) * SHORT.bin_range, SHORT.min_range)

    # ∫ e^(-2σr) / r² dr = [-e^(-2σr) / r + 2σ E1(2σr)]
    a = 2 * medium.extinction
    antiderivative = -np.exp(-a * edges) / edges + a * exp1(a * edges)
    phase = henyey_greenstein(torch.tensor(-1.0, dtype=torch.float64), medium.anisotropy).item()
    scale = SHORT.signal * math.pi * SHORT.reference_distance**2 * medium.extinction * phase
    expected = scale * np.diff(antiderivative)
    assert np.allclose(result.backscatter[0, 0].numpy(), expected, rtol=1e-7, atol=1e-15)
    assert (result.surface == 0).all() and (result.background == 0).all()


def test_height_fog_backscatter_matches_quadrature():
    medium = Medium(extinction=0.08, anisotropy=0.8, components=[HeightFog(falloff=2.5)])
    distance = 17.3
    result = transient(medium, distance)
    origin, directions, _ = (x.numpy() for x in camera_rays(CAMERA, POSE))
    v, phase = directions[0, 0], henyey_greenstein(torch.tensor(-1.0, dtype=torch.float64), 0.8).item()

    def sigma(r):
        return medium.extinction * math.exp(-(origin[2] + r * v[2]) / 2.5)

    def integrand(r):
        tau, _ = quad(sigma, 0, r, epsabs=0, epsrel=1e-12)
        return sigma(r) * phase * math.exp(-2 * tau) / r**2

    scale = SHORT.signal * math.pi * SHORT.reference_distance**2
    for k in (0, 3, 50, 114):
        lo, hi = max(k * SHORT.bin_range, SHORT.min_range), min((k + 1) * SHORT.bin_range, distance)
        expected = scale * quad(integrand, lo, hi, epsabs=0, epsrel=1e-10)[0] if hi > lo else 0.0
        assert result.backscatter[0, 0, k].item() == pytest.approx(expected, rel=1e-8, abs=1e-15)
    # Nothing is scattered back from beyond the surface
    assert (result.backscatter[0, 0, math.ceil(distance / SHORT.bin_range) :] == 0).all()


def test_ray_marched_backscatter_converges():
    medium = Medium(extinction=0.08, anisotropy=0.8, components=[HeightFog(falloff=2.5)])
    exact = transient(medium, 25.0)
    errors = []
    for steps in (100, 1000, 10000):
        marched = transient(medium, 25.0, steps=steps)
        errors.append((marched.photons - exact.photons).abs().sum().item() / exact.photons.sum().item())
    # Coarse steps leave most bins empty, which averages out as steps get finer, although bins can still receive
    # slightly different numbers of steps when their sizes aren't multiples of each other
    assert errors[0] > 0.3 and max(errors[1:]) < 5e-3


def test_surface_return_and_pulse():
    flash = Flash(bins=1000, bin_width=0.2e-9, pulse_width=1e-9, background=0.0)
    medium = Medium(extinction=0.02, components=[Homogeneous()])
    result = transient(medium, 12.34, flash=flash)
    surface = result.surface[0, 0].numpy()

    # The pulse spreads the return over several bins, but keeps its energy, attenuated twice by the medium
    expected = flash.signal * 0.5 * (flash.reference_distance / 12.34) ** 2 * math.exp(-2 * 0.02 * 12.34)
    assert surface.sum() == pytest.approx(expected, rel=1e-6)
    assert (surface > 1e-3 * surface.max()).sum() > 5
    centroid = (surface * (np.arange(flash.bins) + 0.5)).sum() / surface.sum() * flash.bin_range
    assert centroid == pytest.approx(12.34, abs=1e-3)


def test_background_and_depth_estimation():
    flash = Flash(bins=500, bin_width=0.2e-9, signal=2.0, background=1.0)
    ones = np.ones((4, 4))
    camera = CAMERA | {"w": 4, "h": 4, "cx": 2.0, "cy": 2.0, "fl_x": 40.0, "fl_y": 40.0}
    result = flash_transient(ones * 9.0, ones, ones, camera, POSE, None, flash, ambient_radiance=ones * 0.3)
    assert result.background.sum(dim=-1).numpy() == pytest.approx(0.3 * np.ones((4, 4)))
    assert (result.backscatter == 0).all()

    # Pile-up hides returns behind strong background, which Coates' estimator undoes before finding the peak
    histogram = capture_histogram(result.photons, cycles=2000, generator=torch.Generator().manual_seed(0))
    distance = estimate_distance(histogram, 2000, flash)
    _, _, scale = camera_rays(camera, POSE)
    assert torch.allclose(distance, 9.0 * scale, atol=1.5 * flash.bin_range)
