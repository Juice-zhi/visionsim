"""Split the error of each computed method against the multiple scattering reference by region and sign.

Pixels are split between those that see the sky and those that see surfaces, by distance, on every tenth frame. The
signed error shows whether a method is too bright or too dark where, which separates the errors of a lighting model
from those of the effects of objects, which are only present on surfaces. Results are saved to
``results_ms/regions.json``.
"""

import json

import numpy as np
from run_experiment import ROOT, load_sequence

from visionsim.dataset import Dataset

METHODS = ("closed_form", "closed_form_ms", "closed_form_m1", "closed_form_occ", "closed_form_ms_occ")


def main():
    indices = list(range(0, 500, 10))
    reference = load_sequence(ROOT / "ms" / "cycles_ref" / "frames")[0][indices]
    depths = Dataset.from_path(ROOT / "clear" / "depths")
    depth = np.stack([np.asarray(depths[i][0])[..., 0] for i in indices])
    sky = depth >= 1e9
    regions = {
        "all": np.ones_like(sky),
        "sky": sky,
        "surfaces_near": ~sky & (depth < 5),
        "surfaces_mid": ~sky & (depth >= 5) & (depth < 15),
        "surfaces_far": ~sky & (depth >= 15),
    }
    results = {}
    for name in METHODS:
        ours = np.load(ROOT / "results" / "cache" / f"{name}_500.npy")[indices]
        results[name] = {
            region: {
                "signed": float((ours[mask] - reference[mask]).sum() / reference[mask].sum()),
                "rel_l1": float(np.abs(ours[mask] - reference[mask]).sum() / reference[mask].sum()),
                "pixels": float(mask.mean()),
            }
            for region, mask in regions.items()
        }
    (ROOT / "results_ms" / "regions.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
