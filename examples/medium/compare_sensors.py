"""Compare what each emulated sensor sees with and without a participating medium.

Expects a directory containing ``fog`` (the output of ``medium.apply``) and emulated outputs named
``fog-<sensor>``, where sensor is one of ``rgb``, ``spad`` or ``dvs``, and a directory without the medium
containing ``src/frames`` (the linear frames the medium was applied to) and ``clear-<sensor>``. These can be the
same directory. Missing sensors are skipped. A side by side comparison is saved to ``comparison.png``::

    python compare_sensors.py runs/fog-demo --frame 250 --rgb-chunk 5 --dvs-step 5
    python compare_sensors.py runs/ground-fog --clear runs/fog-demo --frame 250
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from natsort import natsorted

from visionsim.dataset import Dataset
from visionsim.emulate.spc import spc_avg_to_rgb
from visionsim.utils.color import linearrgb_to_srgb


def rgb(img):
    img = np.asarray(img, dtype=float)
    return np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img[..., :3]


def tonemap(linear, exposure):
    return (linearrgb_to_srgb(np.clip(rgb(linear) * exposure, 0, 1)) * 255).astype(np.uint8)


def contrast(img):
    # RMS contrast of the luminance, i.e. its standard deviation relative to its mean
    luminance = rgb(img) @ [0.2126, 0.7152, 0.0722]
    return luminance.std() / max(luminance.mean(), 1e-12)


def nth_image(directory, index):
    paths = natsorted(p for p in Path(directory).glob("**/*.png"))
    return iio.imread(paths[min(index, len(paths) - 1)])[..., :3] if paths else None


def main(root, clear_root, frame, rgb_chunk, dvs_step, spad_frames):
    clear, _ = Dataset.from_path(clear_root / "src" / "frames")[frame]
    fog, _ = Dataset.from_path(root / "fog" / "frames")[frame]
    transmittance, _ = Dataset.from_path(root / "fog" / "transmittance")[frame]
    inscatter, _ = Dataset.from_path(root / "fog" / "inscatter")[frame]
    exposure = 0.8 / np.percentile(rgb(clear), 99)

    print(f"Frame {frame}")
    print(f"  ground truth: mean radiance {rgb(clear).mean():.4f} -> {rgb(fog).mean():.4f}, ", end="")
    print(f"contrast {contrast(clear):.3f} -> {contrast(fog):.3f}")
    print(f"  medium: mean transmittance {transmittance.mean():.3f}, ", end="")
    print(f"in-scattered share of radiance {rgb(inscatter).mean() / rgb(fog).mean():.1%}")

    rows = [[tonemap(clear, exposure)], [tonemap(fog, exposure)]]
    for name in ("rgb", "spad", "dvs"):
        images = []
        for condition in ("clear", "fog"):
            path = (clear_root if condition == "clear" else root) / f"{condition}-{name}"
            if not path.exists():
                break

            if name == "rgb":
                image = nth_image(path, frame // rgb_chunk)
                stats = f"contrast {contrast(image / 255):.3f}"
            elif name == "spad":
                binary = Dataset.from_path(path)
                start = max(0, min(frame - spad_frames // 2, len(binary) - spad_frames))
                mean = np.mean([binary[i][0] for i in range(start, start + spad_frames)], axis=0)
                image = tonemap(spc_avg_to_rgb(mean), exposure)

                # Photons are detected with probability 1 - exp(-intensity), for the frames the SPAD was emulated from
                source = Dataset.from_path(
                    clear_root / "src" / "frames" if condition == "clear" else root / "fog" / "frames"
                )
                expected = np.mean([-np.expm1(-np.maximum(source[i][0], 0)) for i in range(start, start + spad_frames)])
                stats = f"detection probability {mean.mean():.4f} (expected from ground truth: {expected:.4f})"
            else:
                image = nth_image(path / "preview", frame // dvs_step)
                with open(path / "events.txt") as f:
                    stats = f"{sum(1 for _ in f):,} events in total"
            print(f"  {name} ({condition}): {stats}")
            images.append(image)

        if len(images) == 2 and all(i is not None for i in images):
            for row, image in zip(rows, images):
                row.append(image)

    comparison = np.concatenate([np.concatenate(row, axis=1) for row in rows], axis=0)
    iio.imwrite(root / "comparison.png", comparison)
    print(f"Saved {root / 'comparison.png'}: ground truth, then emulated sensors; top row clear, bottom row with medium")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory containing the medium and the sensors emulated with it")
    parser.add_argument("--clear", type=Path, help="directory without the medium, defaults to root")
    parser.add_argument("--frame", type=int, default=0, help="index of the ground truth frame to compare")
    parser.add_argument("--rgb-chunk", type=int, default=5, help="number of frames per RGB exposure")
    parser.add_argument("--dvs-step", type=int, default=5, help="number of frames per DVS preview")
    parser.add_argument("--spad-frames", type=int, default=64, help="number of binary frames to average")
    args = parser.parse_args()
    main(args.root, args.clear or args.root, args.frame, args.rgb_chunk, args.dvs_step, args.spad_frames)
