"""How well low orders of spherical harmonics represent the light that the day scene's fog scatters towards each
direction, for each source separately: the sun (sharp, through the phase function's forward peak), the sky, the
ground, and light scattered more than once.

At random points of the cache's box of ``exp_cache.py``, the light scattered towards a fine set of directions is
projected onto spherical harmonics up to each order, reconstructed, and compared with the original over the directions
camera rays take (elevations within ±30°), as a relative L1 error. Results are saved to ``results/sh.json``.
"""

from __future__ import annotations

import json

import torch

from common import DTYPE, OUT, DayLight, day_scene, sh_basis, sphere_quadrature
from exp_cache import DAY_BOX


def main():
    scene = day_scene(frames=(345,))
    light = DayLight(scene)
    generator = torch.Generator().manual_seed(0)
    low, high = (torch.as_tensor(b, dtype=DTYPE) for b in DAY_BOX)
    # Points near the camera's path and the objects, where the fog is dense
    points = low + (high - low) * torch.rand(400, 3, generator=generator)
    points[:, 2] = points[:, 2] * 0.5
    directions, weights = sphere_quadrature(32, 64)
    near_horizon = directions[:, 2].abs() < 0.5
    m = len(directions)
    parts: dict[str, list[torch.Tensor]] = {}
    for chunk in points.split(25):
        n = len(chunk)
        lit = light.sky_visibility(chunk)
        c = light.depth_above(chunk)
        flat_points = chunk.repeat_interleave(m, dim=0)
        flat_dirs = directions.repeat(n, 1)
        parts.setdefault("sun", []).append(light.sun(flat_points, flat_dirs).reshape(n, m, 3))
        ratios = light.ratios(
            flat_points, flat_dirs, lit[0].repeat_interleave(m, dim=0), lit[1].repeat_interleave(m, dim=0)
        )
        for name in light.tables:
            value = light.source(name, flat_dirs[:, 2], c.repeat_interleave(m)) * ratios[name]
            parts.setdefault(name, []).append(value.reshape(n, m, 3))
    sources = {name: torch.cat(values) for name, values in parts.items()}
    results = {}
    for order in (1, 2, 3, 4, 6, 8, 12):
        basis = sh_basis(directions, order)  # (m, K)
        projection = basis * weights[:, None]
        entry = {}
        for name, values in sources.items():
            coefficients = torch.einsum("nmc,mk->nkc", values, projection)
            rebuilt = torch.einsum("nkc,mk->nmc", coefficients, basis)
            error = (rebuilt - values).abs()[:, near_horizon].sum() / values[:, near_horizon].abs().sum()
            entry[name] = float(error)
        results[order] = entry
        print(order, json.dumps(entry), flush=True)
    (OUT / "results" / "sh.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
