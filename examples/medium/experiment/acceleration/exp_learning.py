"""Direction 6: learning, trained on the CPU.

- ``image``: a small U-Net mapping the frame without fog and its depth to the frame with fog, trained on Cycles'
  references of frames 5 to 404 of the day scene, and tested on frames 405 to 504 (unseen views of the same scene and
  fog), on the same scene with fog twice as dense, and on the night scene lit by the lamp (out of distribution).
- ``residual``: the same U-Net given the closed form's frame too, learning a correction to it, i.e. what the closed form
  misses (mostly light scattered many times near objects).
- ``field``: a neural field (an MLP of Fourier features) storing the lamp's transmittance times visibility in the night
  scene's cache box, as a neural radiance cache would, against voxel grids of similar size (``exp_cache.py``).

Results are saved to ``results/learning.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from visionsim.medium import Medium, apply_medium

from common import (
    DTYPE,
    NIGHT_FRAMES,
    OUT,
    ROOT,
    NightLight,
    by_chunks,
    day_scene,
    errors,
    integrate_lamp,
    load_images,
    night_scene,
)
from exp_cache import NIGHT_BOX

TRAIN, TEST = range(400), range(400, 500)
torch.manual_seed(0)


class UNet(nn.Module):
    def __init__(self, inputs: int, width: int = 24):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())

        self.down1, self.down2, self.down3 = block(inputs, width), block(width, 2 * width), block(2 * width, 4 * width)
        self.bottom = block(4 * width, 4 * width)
        self.up3, self.up2, self.up1 = block(8 * width, 2 * width), block(4 * width, width), block(2 * width, width)
        self.out = nn.Conv2d(width, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        pool, up = nn.functional.avg_pool2d, nn.functional.interpolate
        a = self.down1(x)
        b = self.down2(pool(a, 2))
        c = self.down3(pool(b, 2))
        d = self.bottom(pool(c, 2))
        c = self.up3(torch.cat([up(d, size=c.shape[-2:], mode="bilinear"), c], 1))
        b = self.up2(torch.cat([up(c, size=b.shape[-2:], mode="bilinear"), b], 1))
        a = self.up1(torch.cat([up(b, size=a.shape[-2:], mode="bilinear"), a], 1))
        return self.out(a)


def features(clear: np.ndarray, depth: np.ndarray, closed_form: np.ndarray | None) -> torch.Tensor:
    """Inputs of the networks, of shape (n, c, h, w): log radiance of the frame without fog, log depth, and the log
    radiance of the closed form's frame."""
    parts = [np.log1p(clear * 10), np.log(np.clip(depth, 0.1, 1e4))[..., None] / 5]
    if closed_form is not None:
        parts.append(np.log1p(closed_form * 10))
    return torch.as_tensor(np.concatenate(parts, axis=-1), dtype=DTYPE).permute(0, 3, 1, 2).contiguous()


def predict(net: nn.Module, x: torch.Tensor, closed_form: np.ndarray | None) -> np.ndarray:
    with torch.no_grad():
        out = torch.cat([net(x[k : k + 8]) for k in range(0, len(x), 8)]).permute(0, 2, 3, 1).numpy()
    if closed_form is None:
        return np.expm1(out) / 10
    # The residual network scales the closed form's log radiance
    return np.expm1(np.log1p(closed_form * 10) + out) / 10


def train(
    x: torch.Tensor, target: np.ndarray, closed_form: np.ndarray | None, steps: int = 1500
) -> tuple[nn.Module, float]:
    net = UNet(x.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    y = torch.as_tensor(np.log1p(target * 10), dtype=DTYPE).permute(0, 3, 1, 2)
    base = None if closed_form is None else torch.as_tensor(np.log1p(closed_form * 10), dtype=DTYPE).permute(0, 3, 1, 2)
    generator = torch.Generator().manual_seed(0)
    start = time.perf_counter()
    for step in range(steps):
        index = torch.randint(len(x), (6,), generator=generator)
        top = int(torch.randint(x.shape[2] - 128 + 1, (1,), generator=generator))
        left = int(torch.randint(x.shape[3] - 128 + 1, (1,), generator=generator))
        crop = (index, slice(None), slice(top, top + 128), slice(left, left + 128))
        out = net(x[crop])
        if base is not None:
            out = out + base[crop]
        loss = (out - y[crop]).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    return net, time.perf_counter() - start


def images_experiment() -> dict:
    frames = tuple(range(5, 505))
    day = day_scene(frames=frames, multiple=True)
    closed_form = np.load(ROOT / "results" / "cache" / "closed_form_ms_occ_500.npy")
    sets = {}
    sets["day_test"] = (day.clear[list(TEST)], day.depth[list(TEST)], closed_form[list(TEST)], day.reference[list(TEST)])

    # Denser fog (out of distribution): the closed form at that density, and Cycles' render
    dense = Medium.model_validate_json(Path(__file__).with_name("ground_fog_dense.json").read_text())
    dense = dense.model_copy(update={"multiple_scattering": True})
    pair = [frames.index(f) for f in (255, 345)]
    dense_cf = np.stack(
        [
            apply_medium(
                day.clear[i],
                day.depth[i],
                day.transforms[i],
                day.transforms[i]["transform_matrix"],
                dense,
                day.lighting,
                occlusion=day.occlusion,
                dtype=DTYPE,
            ).radiance.numpy()
            for i in pair
        ]
    )
    sets["dense"] = (day.clear[pair], day.depth[pair], dense_cf, load_images(OUT / "dense_ref" / "frames", (255, 345)))

    # Night scene (out of distribution): lit by the lamp only, with light scattered many times (halos)
    night = night_scene(NIGHT_FRAMES)
    night_clear = load_images(OUT / "night_clear" / "frames", NIGHT_FRAMES)
    night_medium = night.medium.model_copy(update={"multiple_scattering": True})
    night_cf = np.stack(
        [
            apply_medium(
                night_clear[i],
                night.depth[i],
                night.transforms[i],
                night.transforms[i]["transform_matrix"],
                night_medium,
                night.lighting,
                occlusion=night.occlusion,
                dtype=DTYPE,
            ).radiance.numpy()
            for i in range(2)
        ]
    )
    sets["night"] = (night_clear, night.depth, night_cf, load_images(OUT / "night_ms_ref" / "frames", NIGHT_FRAMES))

    results: dict = {"closed_form": {name: errors(cf, ref) for name, (_, _, cf, ref) in sets.items()}}
    train_cf = closed_form[list(TRAIN)]
    for kind in ("image", "residual"):
        use_cf = kind == "residual"
        x = features(day.clear[list(TRAIN)], day.depth[list(TRAIN)], train_cf if use_cf else None)
        net, seconds = train(x, day.reference[list(TRAIN)], train_cf if use_cf else None)
        entry: dict = {
            "train_seconds": seconds,
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "train_frames": len(TRAIN),
        }
        train_pred = predict(net, x[:8], train_cf[:8] if use_cf else None)
        entry["train"] = errors(train_pred, day.reference[:8])
        for name, (clear, depth, cf, ref) in sets.items():
            inputs = features(clear, depth, cf if use_cf else None)
            start = time.perf_counter()
            pred = predict(net, inputs, cf if use_cf else None)
            entry[f"{name}_seconds_per_frame"] = (time.perf_counter() - start) / len(inputs)
            entry[name] = errors(pred, ref)
            if name in ("dense", "night"):
                np.save(OUT / "results" / f"learning_{kind}_{name}.npy", pred[-1])
        results[kind] = entry
        print(kind, json.dumps(entry), flush=True)
    return results


class Field(nn.Module):
    """MLP of Fourier features of positions in a box, with values in [0, 1]."""

    def __init__(self, frequencies: int, width: int, depth: int):
        super().__init__()
        self.register_buffer("scales", 2.0 ** torch.arange(frequencies, dtype=DTYPE) * math.pi)
        layers, inputs = [], 3 + 6 * frequencies
        for _ in range(depth):
            layers += [nn.Linear(inputs, width), nn.GELU()]
            inputs = width
        self.mlp = nn.Sequential(*layers, nn.Linear(inputs, 1))

    def forward(self, x):
        angles = x[..., None] * self.scales
        encoded = torch.cat([x, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)
        return torch.sigmoid(self.mlp(encoded))


def field_experiment() -> dict:
    scene = night_scene(NIGHT_FRAMES)
    light = NightLight(scene)
    low, high = (torch.as_tensor(b, dtype=DTYPE) for b in NIGHT_BOX)
    cf = np.load(OUT / "results" / "closed_form_night.npy")
    rays = [scene.rays(i) for i in range(2)]

    def render(source) -> tuple[np.ndarray, float]:
        out, seconds = [], []
        for (origin, directions, distance), depth in zip(rays, scene.depth):
            start = time.perf_counter()
            image = by_chunks(
                lambda d, l: integrate_lamp(light, origin, d, l, source, n=256),
                len(directions),
                4096,
                directions,
                distance,
            )
            seconds.append(time.perf_counter() - start)
            out.append(image.reshape(*depth.shape, 3).numpy())
        return np.stack(out), min(seconds)

    results = []
    generator = torch.Generator().manual_seed(0)
    for frequencies, width, depth, steps in ((6, 64, 3, 3000), (8, 128, 4, 3000)):
        net = Field(frequencies, width, depth)
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        start = time.perf_counter()
        for step in range(steps):
            points = low + (high - low) * torch.rand(16384, 3, generator=generator)
            target = light.lit(points)[:, None]
            loss = (net((points - low) / (high - low) * 2 - 1) - target).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        train_seconds = time.perf_counter() - start

        def source(points, directions, net=net):
            result = light(points, directions)
            within = ((points >= low) & (points <= high)).all(dim=-1)
            with torch.no_grad():
                stored = net((points[within] - low) / (high - low) * 2 - 1)
            result[within] = light.falloff(points[within], directions[within]) * stored
            return result

        image, seconds = render(source)
        entry = {
            "kind": "field",
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "bytes_fp32": int(sum(p.numel() for p in net.parameters()) * 4),
            "train_seconds": train_seconds,
            "seconds": seconds,
            "vs_closed_form": errors(image, cf),
            "vs_cycles": errors(image, scene.reference),
        }
        results.append(entry)
        print("field", json.dumps(entry), flush=True)
        np.save(OUT / "results" / f"learning_field_{entry['parameters']}.npy", image[1])
    return results


def main():
    results = {"field": field_experiment(), "images": images_experiment()}
    (OUT / "results" / "learning.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
