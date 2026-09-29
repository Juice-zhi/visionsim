"""Compare how each method's error changes over time, which is what event cameras are sensitive to.

An event camera fires when the log intensity of a pixel changes by more than a threshold, so an error that stays the
same from one frame to the next produces few false events, while one that changes from frame to frame (Monte Carlo
noise, or a denoiser working on each frame independently) does. For every method, this compares the error of each
frame's log intensity against the reference (static), with the error of its change between consecutive frames
(temporal), both as mean absolute errors, alongside how much the log intensity changes between frames on average
(change), which drives the number of events. Results are saved to ``results/temporal.json``.
"""

import json

import numpy as np
from run_experiment import ROOT, load_sequence

WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
RENDERED = {
    "clear": "clear",
    "cycles_default": "cycles_default",
    "cycles_nodenoise": "cycles_nodenoise",
    "reference_seed1": "cycles_ref_seed1",
}
COMPUTED = ("raymarch_16", "raymarch_64s16", "closed_form_m1", "closed_form")


def log_luma(frames: np.ndarray) -> np.ndarray:
    """Log intensity as seen by v2e, which is linear below 20 digital numbers."""
    luma = np.clip(frames, 0, None) @ WEIGHTS * 255.0
    return np.where(luma <= 20, luma * np.log(20) / 20, np.log(np.maximum(luma, 20)))


def main():
    reference = log_luma(load_sequence(ROOT / "cycles_ref" / "frames")[0])
    results = {"reference": {"change": float(np.abs(np.diff(reference, axis=0)).mean())}}
    for name in [*RENDERED, *COMPUTED]:
        if name in RENDERED:
            frames = log_luma(load_sequence(ROOT / RENDERED[name] / "frames")[0])
        else:
            frames = log_luma(np.load(ROOT / "results" / "cache" / f"{name}_{len(reference)}.npy"))
        error = frames - reference
        results[name] = {
            "static": float(np.abs(error).mean()),
            "temporal": float(np.abs(np.diff(error, axis=0)).mean()),
            "change": float(np.abs(np.diff(frames, axis=0)).mean()),
        }
        print(name, results[name], flush=True)
    (ROOT / "results" / "temporal.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
