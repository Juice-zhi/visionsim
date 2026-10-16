"""Summary of the final comparison: how much time the closed form saves against Cycles, and at what error.

Reads the JSON of ``final_benchmark.py`` (``results/final_gpu.json`` and ``results/final_cpu.json`` by default) and
prints, for each hardware, case and resolution, the pipelines (Cycles renders the frame without fog at N samples per
pixel, denoised, and the closed form adds the fog at 1/factor of the resolution) with their time per frame and error,
the time Cycles needs for the same error (relative L1 and, without the reference's noise, relative RMSE) or the same
flicker, and the ratio of both times. Run from the repository root::

    python examples/medium/experiment/acceleration/analysis/final_report.py [results/final_gpu.json ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS = Path("runs/fog-comparison/acceleration/results")


def ratio(cycles: float | None, ours: float) -> str:
    return "-" if cycles is None else f"{cycles / ours:.2f}x"


def ms(seconds: float | None) -> str:
    return "-" if seconds is None else f"{1e3 * seconds:.0f}"


def curve(data: dict, row: dict, metric: str, flicker: bool = False) -> list[tuple[float, float, str]]:
    """Cycles' (error, seconds, spp) points that the row's Cycles time was interpolated on."""
    entry = data[row["case"]]
    points = []
    if flicker:
        flickers = data.get("day_flicker") or {}
        for spp, v in entry["cycles"].items():
            if f"cycles_spp{spp}" in flickers and v.get("seconds"):
                points.append((flickers[f"cycles_spp{spp}"]["temporal"], v["seconds"], spp))
        return points
    hd = row["resolution"] == "1920x1080"
    measured = entry["1080p"].get("cycles", {})
    for spp, v in entry["cycles"].items():
        seconds = v.get("seconds_1080p") if hd else v.get("seconds")
        if not seconds:
            continue
        if hd and row.get("errors_measured_at") == "1920x1080":
            found = measured.get(spp) or measured.get(str(spp))
            if found:
                points.append((found[metric], seconds, spp))
        elif "denoised" in v:
            points.append((v["denoised"][metric], seconds, spp))
    return points


def versus(data: dict, row: dict, key: str, metric: str, value: float | None, flicker: bool = False) -> tuple[str, str]:
    """Cycles' time at the row's error (ms) and the saving, or where the error falls outside Cycles' curve."""
    cycles = row.get(key)
    if cycles is not None:
        return ms(cycles), ratio(cycles, row["seconds"])
    points = curve(data, row, metric, flicker)
    if value is None or math.isnan(value) or not points:
        return "-", "-"
    worst = max(points, key=lambda p: p[0])
    best = min(points, key=lambda p: p[0])
    if value >= worst[0]:
        return f"<{ms(worst[1])} ({worst[2]} spp already closer)", f"<{worst[1] / row['seconds']:.2f}x"
    return f">{ms(best[1])} (beyond {best[2]} spp)", f">{best[1] / row['seconds']:.2f}x"


def report(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"\n## {path.name}: closed form on {data['device']}, Cycles renders in {data.get('renders')}\n")
    print(
        "| case | resolution | factor | frames without fog | pipeline ms | rel L1 | Cycles ms at same L1 | saving |"
        " RMSE (no ref. noise) | Cycles ms at same RMSE | saving | flicker | Cycles ms at same flicker | saving |"
    )
    print("|" + "---|" * 14)
    for row in data["comparison"]:
        l1 = versus(data, row, "cycles_seconds_at_equal_error", "rel_l1", row["rel_l1"])
        rmse = versus(data, row, "cycles_seconds_at_equal_rmse", "rel_rmse_corrected", row.get("rel_rmse_corrected"))
        flick = versus(data, row, "cycles_seconds_at_equal_flicker", "", row.get("temporal_flicker"), flicker=True)
        print(
            f"| {row['case']} | {row['resolution']} | {row['factor']} | {row['clear_spp']} spp | {ms(row['seconds'])} |"
            f" {100 * row['rel_l1']:.2f}% | {l1[0]} | {l1[1]} |"
            f" {100 * row.get('rel_rmse_corrected', float('nan')):.2f}% | {rmse[0]} | {rmse[1]} |"
            f" {row.get('temporal_flicker', float('nan')):.4f} | {flick[0]} | {flick[1]} |"
        )
    for case in ("day", "night"):
        if case not in data:
            continue
        entry = data[case]
        print(f"\nCycles + OIDN, {case} (320x180 rel L1 / RMSE; 1080p rel L1 / RMSE; times ms at 320x180 / 1080p):")
        hd = entry["1080p"].get("cycles", {})
        for spp, v in entry["cycles"].items():
            if "denoised" not in v:
                continue
            d, h = v["denoised"], hd.get(spp) or hd.get(str(spp))
            hd_text = f"{100 * h['rel_l1']:.2f}% / {100 * h['rel_rmse_corrected']:.2f}%" if h else "-"
            print(
                f"  {spp:>5} spp: {100 * d['rel_l1']:.2f}% / {100 * d['rel_rmse_corrected']:.2f}%; {hd_text};"
                f" {ms(v.get('seconds'))} / {ms(v.get('seconds_1080p'))}"
            )
        cf = entry["closed_form"]
        for factor, v in cf.items():
            print(
                f"  closed form factor {factor}: {ms(v['seconds'])} ms (first call {v['first_seconds']:.2f} s),"
                f" clean frames without fog {100 * v['whole']['rel_l1']:.2f}% / {100 * v['whole']['rel_rmse_corrected']:.2f}%"
            )
        for factor, v in entry["1080p"]["closed_form"].items():
            whole = v.get("whole")
            text = f", {100 * whole['rel_l1']:.2f}% / {100 * whole['rel_rmse_corrected']:.2f}%" if whole else ""
            print(f"  closed form 1080p factor {factor}: {ms(v['seconds'])} ms{text}")
        if "reference_noise" in entry["1080p"]:
            print(
                f"  1080p reference noise: rel L1 {100 * entry['1080p']['reference_noise']:.2f}%,"
                f" RMSE {100 * entry['1080p']['reference_noise_rmse']:.2f}%"
            )


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [RESULTS / "final_gpu.json", RESULTS / "final_cpu.json"]
    for path in paths:
        if path.exists():
            report(path)
        else:
            print(f"\n({path} not found)")
