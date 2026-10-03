"""Build the HTML report of the fog comparison from ``results/metrics.json``, ``convergence.json``, ``breakdown.json``,
``temporal.json`` and ``bounces.json``, and ``results_ms/metrics.json`` for the comparison against multiple
scattering, along with the render times. The report is written to ``results/index.html`` and references the images
and videos in ``results/images`` and ``results/videos``."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("runs/fog-comparison")
RESULTS = ROOT / "results"

LABELS = {
    "clear": "无雾 baseline",
    "cycles_default": "Cycles 默认设置（单次散射）",
    "cycles_nodenoise": "Cycles 不降噪（单次散射）",
    "raymarch_16": "Ray marching 16 步",
    "raymarch_64s16": "Ray marching 64 步 + 朝太阳 16 步",
    "closed_form_m1": "闭式解（修正天光前）",
    "closed_form": "闭式解",
    "closed_form_occ": "闭式解 + 阴影",
    "reference_seed1": "参考本身的噪声（换一个种子）",
    "reference": "参考：Cycles 4096 spp，单次散射",
}
# Against multiple scattering, Cycles' renders with multiple scattering are compared alongside its single scattering one
LABELS_MS = {
    "clear": LABELS["clear"],
    "cycles_default": "Cycles 默认设置（多次散射）",
    "cycles_nodenoise": "Cycles 不降噪（多次散射）",
    "cycles_default_ss": "Cycles 默认设置（单次散射）",
    "raymarch_16": LABELS["raymarch_16"],
    "raymarch_64s16": LABELS["raymarch_64s16"],
    "closed_form_m1": LABELS["closed_form_m1"],
    "closed_form": "闭式解（单次散射）",
    "closed_form_occ": "闭式解（单次散射）+ 阴影",
    "closed_form_ms": "闭式解（多次散射）",
    "closed_form_ms_occ": "闭式解（多次散射）+ 阴影",
    "reference_seed1": LABELS["reference_seed1"],
    "reference": "参考：Cycles 4096 spp，多次散射",
}
KINDS = {
    "clear": "对照",
    "cycles_default": "体积渲染",
    "cycles_nodenoise": "体积渲染",
    "cycles_default_ss": "体积渲染",
    "raymarch_16": "Ray marching",
    "raymarch_64s16": "Ray marching",
    "closed_form_m1": "闭式解",
    "closed_form": "闭式解",
    "closed_form_occ": "闭式解",
    "closed_form_ms": "闭式解",
    "closed_form_ms_occ": "闭式解",
    "reference_seed1": "噪声底",
}


def render_seconds() -> dict[str, float]:
    """Seconds per frame of each Blender render, including Blender's startup, from the render logs."""
    times = {}
    for line in (ROOT / "render_times.txt").read_text(encoding="utf-8-sig").splitlines():
        if match := re.match(r"(\w+) exit=0 ([\d,.]+)s", line):
            times[match[1]] = float(match[2].replace(",", "")) / 500
    for name, log in (
        ("cycles_ref", "cycles_ref.log"),
        ("cycles_ref_seed1", "cycles_ref_seed1.log"),
        ("ms_cycles_ref", "ms/cycles_ref.log"),
        ("ms_cycles_ref_seed1", "ms/cycles_ref_seed1.log"),
    ):
        if (ROOT / log).exists():
            text = (ROOT / log).read_text(encoding="utf-8", errors="replace")
            if match := re.search(r"RENDER_TIME frames=\d+ total=[\d.]+s per_frame=([\d.]+)s", text):
                times[name] = float(match[1])
    return times


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(value: float) -> str:
    """Scientific notation for prose, such as 1.2×10⁻³."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(SUPERSCRIPT)}"


def fmt(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "pct2":
        return f"{100 * value:.2f}%"
    if kind == "db":
        return f"{value:.1f}"
    if kind == "milli":
        return f"{1000 * value:.2f}"
    if kind == "f3":
        return f"{value:.3f}"
    if kind == "f4":
        return f"{value:.4f}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "sec":
        return f"{value:.3f} s" if value >= 0.1 else f"{1000 * value:.0f} ms"
    return str(value)


def table(
    rows: list[str],
    columns: list[tuple[str, str, str, bool | None]],
    data: dict[str, dict],
    note: str = "",
    labels: dict[str, str] = LABELS,
) -> str:
    """Rows are methods, columns are (key, header, format, higher_is_better), best values among methods are marked."""
    candidates = [r for r in rows if r not in ("reference_seed1", "reference")]
    best = {}
    for key, _, _, higher in columns:
        values = {r: data[r][key] for r in candidates if data.get(r, {}).get(key) is not None}
        if higher is not None and values:
            best[key] = (max if higher else min)(values.values())
    head = "".join(f'<th scope="col">{h}</th>' for _, h, _, _ in columns)
    body = []
    for r in rows:
        cells = []
        for key, _, kind, _ in columns:
            value = data.get(r, {}).get(key)
            if value is None:
                cells.append('<td class="na">—</td>')
            else:
                mark = ' class="best"' if best.get(key) == value and r in candidates else ""
                cells.append(f"<td{mark}>{fmt(value, kind)}</td>")
        kind = KINDS.get(r, "")
        row_class = ' class="floor"' if r in ("reference_seed1", "reference") else ""
        body.append(
            f'<tr{row_class}><th scope="row"><span class="kind">{kind}</span>{html.escape(labels[r])}</th>{"".join(cells)}</tr>'
        )
    caption = f'<p class="table-note">{note}</p>' if note else ""
    return f'<div class="table-wrap"><table><thead><tr><th scope="col">方法</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{caption}'


def figure(src: str, caption: str, alt: str) -> str:
    return (
        f'<figure><a href="{src}" target="_blank" rel="noopener"><img src="{src}" alt="{html.escape(alt)}" loading="lazy"></a>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


STYLE = """
:root {
  --bg: #eef1f2; --surface: #ffffff; --ink: #162026; --muted: #55636c; --rule: #d2d9dd;
  --accent: #0d6a86; --accent-soft: #dbecf1; --good: #1c7449; --bad: #a8432a; --code: #e4e9eb;
  --serif: "Noto Serif SC", "Songti SC", "SimSun", serif;
  --sans: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1316; --surface: #151c20; --ink: #e1e8eb; --muted: #93a2ab; --rule: #27323a;
    --accent: #5cb6d4; --accent-soft: #15313b; --good: #5cbf8b; --bad: #e3866a; --code: #1b242a;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1316; --surface: #151c20; --ink: #e1e8eb; --muted: #93a2ab; --rule: #27323a;
  --accent: #5cb6d4; --accent-soft: #15313b; --good: #5cbf8b; --bad: #e3866a; --code: #1b242a;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15.5px; line-height: 1.7;
  padding-inline: 20px; padding-block: 28px 64px; }
main { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 44px; }
section { display: flex; flex-direction: column; gap: 16px; }
h1, h2, h3 { font-family: var(--serif); font-weight: 700; line-height: 1.3; text-wrap: balance; margin: 0; }
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); }
h2 { font-size: 1.5rem; padding-top: 18px; border-top: 1px solid var(--rule); }
h3 { font-size: 1.12rem; }
p, li { max-width: 72ch; margin: 0; }
ul, ol { margin: 0; padding-left: 1.3em; display: flex; flex-direction: column; gap: 6px; }
.eyebrow { font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.06em; color: var(--muted); text-transform: uppercase; }
.lede { font-size: 1.08rem; color: var(--ink); }
.meta { display: flex; flex-wrap: wrap; gap: 8px 22px; color: var(--muted); font-size: 0.88rem; }
.meta code { font-size: 0.85rem; }
code, .num { font-family: var(--mono); }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
pre { background: var(--code); padding: 14px 16px; border-radius: 6px; overflow-x: auto; font-size: 0.84rem; line-height: 1.55; margin: 0; }
pre code { background: none; padding: 0; }
.findings { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 20px 22px; }
.findings li { max-width: none; }
.findings strong { color: var(--accent); }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
thead th { font-weight: 500; color: var(--muted); font-size: 0.82rem; background: var(--bg); position: sticky; top: 0; }
tbody th { text-align: left; font-weight: 500; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: none; }
td { font-family: var(--mono); font-size: 0.86rem; }
td.best { color: var(--good); font-weight: 600; }
td.na { color: var(--muted); }
tr.floor th, tr.floor td { color: var(--muted); background: var(--bg); }
.kind { display: inline-block; min-width: 6.2em; margin-right: 8px; font-size: 0.74rem; color: var(--muted);
  font-weight: 400; letter-spacing: 0.02em; }
.table-note { font-size: 0.85rem; color: var(--muted); }
figure { margin: 0; display: flex; flex-direction: column; gap: 8px; }
figure img, video { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--rule); background: #fff; display: block; }
figcaption { font-size: 0.86rem; color: var(--muted); max-width: 90ch; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; align-items: start; }
.callout { border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; border-radius: 0; }
.callout p { max-width: 80ch; }
dl.setup { display: grid; grid-template-columns: max-content 1fr; gap: 8px 18px; margin: 0; }
dl.setup dt { color: var(--muted); font-size: 0.9rem; }
dl.setup dd { margin: 0; }
@media (max-width: 640px) { dl.setup { grid-template-columns: 1fr; gap: 2px; } dl.setup dd { margin-bottom: 8px; } }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""


def hd_timings() -> dict | None:
    """Seconds per frame at 1920x1080, from ``timing_1080p.ps1`` and ``timing_1080p.py``, if they were run."""
    path = ROOT / "timing_1080p.txt"
    if not path.exists():
        return None
    timings: dict = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if match := re.match(r"(\w+)_1080p exit=0 ([\d,.]+)s for (\d+) frames", line):
            timings[match[1]] = float(match[2].replace(",", "")) / int(match[3])
        elif line.startswith("{"):
            timings |= json.loads(line)["seconds_per_frame"]
    return timings if {"clear", "cycles_default", "closed_form"} <= timings.keys() else None


def build_prose(
    rows: dict,
    rows_ms: dict,
    convergence: dict,
    bd: dict,
    renders: dict,
    hd: dict,
    bounces: dict,
    timings: dict,
    validation: dict,
    scale: dict,
    regions: dict,
    occlusion: dict,
    maps: dict,
    reuse: dict,
    cycles_ms_seconds: dict[str, float],
) -> dict[str, str]:
    """Text of the report, which quotes the measured numbers."""
    r = rows
    cf, cd, nd, cl = (r[k] for k in ("closed_form", "cycles_default", "cycles_nodenoise", "clear"))
    cf_occ = r["closed_form_occ"]
    cf_ms, ms_ms, cd_ms, nd_ms, ss_ms, cl_ms = (
        rows_ms[k]
        for k in ("closed_form", "closed_form_ms", "cycles_default", "cycles_nodenoise", "cycles_default_ss", "clear")
    )
    occ_ms = rows_ms["closed_form_ms_occ"]
    # Shadows are traced once per frame, then reused to add each fog
    traced = {k: v["trace"] for k, v in reuse["seconds"].items()}
    shadowed_seconds = {k: v["closed_form_ms_occ"] for k, v in reuse["seconds"].items()}
    shadowed_speedups = "、".join(
        f"{cycles_ms_seconds[k] / v:.1f} 倍（{k.replace('x', '×')}，每帧 {fmt(v, 'sec')}）"
        for k, v in shadowed_seconds.items()
    )
    traced_speedups = "、".join(f"{cycles_ms_seconds[k] / (traced[k] + v):.1f}" for k, v in shadowed_seconds.items())
    traced_text = "、".join(fmt(v, "sec") for v in traced.values())
    reuse_all, reuse_sun, reuse_sky = (reuse["errors"][k] for k in ("all", "sun", "sky"))
    of_inscatter = [e["of_inscatter"] for e in reuse_all.values()]
    of_shadowed = [e["of_shadowed_light"] for e in reuse_all.values()]
    densest = reuse_sky[max(reuse_sky, key=float)]
    sun_without, sun_with, sky_without, sky_with = (
        occlusion[k] for k in ("sun_without", "sun_with", "sky_without", "sky_with")
    )
    floor_ms = rows_ms.get("reference_seed1")
    pct = lambda v: f"{100 * v:.1f}%"
    signed = lambda region: f"{100 * region['signed']:+.1f}%"
    steps = {c["steps"]: c for c in convergence["ray_marching"] if c["shadow_steps"] == 0}
    shadow = {c["steps"]: c for c in convergence["ray_marching"] if c["shadow_steps"]}
    converged = [c["error_vs_reference"] for c in convergence["ray_marching"] if c["steps"] >= 8]
    cf_seconds = convergence["closed_form_seconds"]
    extra_events = nd["dvs_events"] / r["reference"]["dvs_events"] - 1
    by_bounces = {b["volume_bounces"]: b for b in bounces["bounces"]}
    single, most = by_bounces[0], by_bounces[max(by_bounces)]
    hd_ms = timings["1920x1080"]["closed_form_ms"]
    speedups = {
        "320×180": cycles_ms_seconds["320x180"] / ms_ms["update"],
        "800×800": cycles_ms_seconds["800x800"] / scale["closed_form_ms"]["update"],
        "1920×1080": cycles_ms_seconds["1920x1080"] / hd_ms,
    }
    speedup_text = "、".join(f"{v:.0f} 倍（{k}）" for k, v in speedups.items())
    floor_text = f"，参考自身的噪声底为 {pct(floor_ms['frames_rel_l1'])}" if floor_ms else ""
    free = validation["multiple_vs_32_bounces"]
    elevations = {k: v for k, v in free.items() if k != "all"}
    point = validation["point_vs_0_bounces"]
    temporal_text = (
        f"DVS 只对亮度随时间的变化敏感：闭式解在 v2e 看到的对数亮度上，单帧误差是 Cycles 默认设置的 {cf['log_static'] / cd['log_static']:.1f} 倍"
        f"（{cf['log_static']:.3f} 对 {cd['log_static']:.3f}），但相邻两帧之间误差的变化反而更小（{cf['log_temporal']:.4f} 对 {cd['log_temporal']:.4f}），"
        "因为它的误差是随场景缓慢变化的平滑偏差。"
        if "log_temporal" in cf
        else ""
    )

    findings = f"""<ul>
<li><strong>加上光柱和天空遮挡之后，以物理上完整的多次散射为真值，闭式解的渲染帧误差从 {pct(ms_ms["frames_rel_l1"])} 降到 {pct(occ_ms["frames_rel_l1"])}；只算单次散射时是 {pct(cf_ms["frames_rel_l1"])}，Cycles 默认设置（打开多次散射）是 {pct(cd_ms["frames_rel_l1"])}{floor_text}。</strong>四种传感器上：RGB PSNR {occ_ms["rgb_psnr"]:.1f} dB（不加阴影 {ms_ms["rgb_psnr"]:.1f} dB，Cycles {cd_ms["rgb_psnr"]:.1f} dB），SPAD 检测概率 MAE {1000 * occ_ms["spad_mae"]:.0f}×10⁻³（不加阴影 {1000 * ms_ms["spad_mae"]:.0f}，Cycles {1000 * cd_ms["spad_mae"]:.1f}），DVS 事件 F1 {occ_ms["dvs_f1_pooled"]:.3f}（不加阴影 {ms_ms["dvs_f1_pooled"]:.3f}，Cycles {cd_ms["dvs_f1_pooled"]:.3f}）。</li>
<li><strong>阴影本身和 Cycles 吻合：</strong>第 255 帧只开太阳时，雾散射进相机的光与 Cycles 之比在表面像素上从 {sun_without["surfaces"]:.3f} 变为 {sun_with["surfaces"]:.3f}，天空像素上从 {sun_without["sky"]:.3f} 变为 {sun_with["sky"]:.3f}；只开天光时，表面像素上从 {sky_without["surfaces"]:.3f} 变为 {sky_with["surfaces"]:.3f}，天空像素上从 {sky_without["sky"]:.3f} 变为 {sky_with["sky"]:.3f}。阴影图每个场景只需渲染一次（{maps["count"]} 张正交深度图，{maps["seconds"]:.1f} s），与相机、分辨率和雾参数都无关。</li>
<li><strong>时间：阴影与雾参数无关，每帧只需沿视线追踪一次，之后每组雾参数都复用。只改雾参数时，闭式解（多次散射）+ 阴影比打开多次散射的 Cycles 快 {shadowed_speedups}。</strong>追踪阴影要查阴影图，开销随像素数增加，三种分辨率下每帧分别要 {traced_text}，算上它也分别快 {traced_speedups} 倍；复用的误差只占雾散射光的 {fmt(min(of_inscatter), "pct2")}～{fmt(max(of_inscatter), "pct2")}。不加阴影时快 {speedup_text}，每帧只要 {fmt(ms_ms["update"], "sec")}、{fmt(scale["closed_form_ms"]["update"], "sec")} 和 {fmt(hd_ms, "sec")}。天光、地面反射和多次散射都是每帧一张查找表，多次散射表与相机无关，每组雾参数只算一次（{fmt(timings["multiple_scattering_table"], "sec")}）。</li>
<li><strong>多次散射近似本身在去掉物体的场景里和 Cycles 吻合：雾散射进相机的光整体是 Cycles 32 次弹射的 {free["all"]:.3f} 倍，各仰角在 {min(elevations.values()):.2f}～{max(elevations.values()):.2f} 倍之间。</strong>加上阴影后，完整场景里剩下的误差主要来自表面透过雾被照亮（表面在 Cycles 里暗约 {pct(1 - bd["surface_dimming"]["median"])}），以及多次散射近似本身。</li>
<li><strong>Blender 默认的 0 次体积弹射只算单次散射：</strong>用 Cycles 默认设置渲染这团雾，误差是 {pct(ss_ms["frames_rel_l1"])}，比完全不加雾（{pct(cl_ms["frames_rel_l1"])}）还差。雾散射进相机的光里有 {pct(most["indirect_share"])} 散射过不止一次，画面整体亮 {pct(most["brightness"] - 1)}。在 Cycles 里打开多次散射（8 次弹射已收敛），每帧只从 {renders["cycles_default"]:.2f} s 增加到 {renders["ms_cycles_default"]:.2f} s，用 Cycles 渲染雾时应该把体积弹射次数设到 8 以上。</li>
<li><strong>点光源：</strong>单次散射和 Cycles 吻合（雾光比值 {point["ratio"]:.3f}），但点光源的多次散射还没有建模，在这团雾里要少约 {pct(1 - validation["point_vs_32_bounces"]["ratio"])} 的光晕。</li>
<li><strong>以单次散射为真值时，闭式解的实现是对的：</strong>加阴影后误差从 {pct(cf["frames_rel_l1"])} 降到 {pct(cf_occ["frames_rel_l1"])}，剩下的主要是表面透过雾被照亮；ray marching 与（不加阴影的）闭式解只差 {sci(steps[16]["error_vs_closed_form"])}，耗时是它的 {steps[16]["seconds"] / cf_seconds:.1f} 倍。{temporal_text}ToF 只能由闭式解或 ray marching 生成（Cycles 不能渲染瞬态），激光在雾中的多次散射这次无法评估。</li>
</ul>"""
    ms_intro = (
        "Cycles 的“体积弹射次数”决定雾里的光最多被散射几次。Blender 默认为 0，也就是单次散射：雾中每一点只接收直接来自太阳和天空的光。"
        "真实的雾里，光会在雾滴之间反复散射，所以这里把弹射次数设到 16（8 次时已收敛），渲染出物理上完整的多次散射参考。"
        "ray marching 是沿视线的数值积分方法，它的光源项和闭式解一样，所以和闭式解积分的是同一个模型。"
        "“闭式解（多次散射）”加上了地面反射和多次散射的近似：在一组高度上收集散射过一次的光（来自雾或地面），用相函数再散射一次，更高阶按几何级数累加"
        f"（Hillaire，EGSR 2020）。地面按 Lambert 平面处理，反照率取棋盘格地面两种颜色的平均值 {validation['ground_albedo']:.2f}。"
        "“+ 阴影”的两行再让物体在雾里投下阴影（光柱）并遮挡天空，见下文“光柱和天空遮挡”。"
        "下表以多次散射为真值；Cycles 的三行里，“单次散射”即 Blender 的默认设置。"
    )
    ms_text = (
        f"左图（单次散射真值）里，闭式解只改雾参数时比 Cycles 默认设置快 {cd['update'] / cf['update']:.1f} 倍，误差 {pct(cf['frames_rel_l1'])} 对 {pct(cd['frames_rel_l1'])}；"
        f"加阴影后误差降到 {pct(cf_occ['frames_rel_l1'])}，复用每帧的阴影时每帧 {fmt(cf_occ['update'], 'sec')}。"
        f"右图（多次散射真值）里，只算单次散射的方法都落在虚线上方，比不加雾还差；加上多次散射项后，闭式解降到 {pct(ms_ms['frames_rel_l1'])}，每帧耗时只从 {fmt(cf_ms['update'], 'sec')} 增加到 {fmt(ms_ms['update'], 'sec')}。"
        f"不加阴影时，它的误差有方向性：整体偏亮 {signed(regions['closed_form_ms']['all'])}，其中看天空的像素只差 {signed(regions['closed_form_ms']['sky'])}，"
        f"偏差集中在物体表面（5 m 以内 {signed(regions['closed_form_ms']['surfaces_near'])}、5–15 m {signed(regions['closed_form_ms']['surfaces_mid'])}），正是光柱和物体遮挡天空的位置。"
        f"加上阴影后，误差降到 {pct(occ_ms['frames_rel_l1'])}，整体偏差变为 {signed(regions['closed_form_ms_occ']['all'])}：天空像素 {signed(regions['closed_form_ms_occ']['sky'])}，"
        f"表面像素 5 m 以内 {signed(regions['closed_form_ms_occ']['surfaces_near'])}、5–15 m {signed(regions['closed_form_ms_occ']['surfaces_mid'])}、15 m 以外 {signed(regions['closed_form_ms_occ']['surfaces_far'])}；"
        f"每帧先追踪一次阴影（{fmt(traced['320x180'], 'sec')}），之后每组雾参数只要 {fmt(occ_ms['update'], 'sec')}。"
        f"修正天光前的 M1 模型总误差相近（{pct(rows_ms['closed_form_m1']['frames_rel_l1'])}），但原因不同：它在天空上偏暗 {signed(regions['closed_form_m1']['sky'])}、"
        f"远处表面偏暗 {signed(regions['closed_form_m1']['surfaces_far'])}，只在近处表面碰巧抵消，没有物理依据。"
        f"多次散射也放大了 Cycles 的噪声：不降噪时渲染帧误差为 {pct(nd_ms['frames_rel_l1'])}，DVS 事件 F1 为 {nd_ms['dvs_f1_pooled']:.3f}，"
        f"都比单次散射时（{pct(nd['frames_rel_l1'])}、{nd['dvs_f1_pooled']:.3f}）更差，所以降噪器在这里必不可少。"
    )
    validation_intro = (
        f"为了把多次散射近似本身的误差和物体造成的误差分开，把去掉方块和柱子、只剩地面和雾的场景的第 {validation['frames'][0]} 和 {validation['frames'][1]} 帧，"
        "用 Cycles 分别以 0 次和 32 次体积弹射渲染（1024 spp），比较雾散射进相机的光（Cycles 的 Volume Direct 和 Indirect pass）。比值为 1 表示完全一致："
    )
    validation_text = (
        f"单次散射部分在所有仰角都和 Cycles 吻合到 ±1%（整体 {validation['single_vs_0_bounces']['all']:.3f}），这同时验证了天光查找表。"
        f"加上地面反射和多次散射后整体是 {free['all']:.3f}：地平线附近少 {pct(1 - min(elevations.values()))}，俯视地面时多 {pct(max(elevations.values()) - 1)}，"
        "说明各向同性的级数近似在水平方向略弱，而地面的 Lambert 近似略强。"
        f"点光源（夜景，关掉太阳和天空，在相机前方放一盏 3000 W 的灯）的单次散射和 Cycles 的比值是 {point['ratio']:.3f}，逐像素误差 {pct(point['rel_l1'])}（含渲染噪声），"
        "证实了辐射强度按功率除以 4π 换算；多次散射下 Cycles 的灯光光晕更亮，差的部分就是尚未建模的点光源多次散射。"
    )
    sky_texels = "～".join(f"{100 * t:.0f}" for t in maps["texels"]["sky"])
    occlusion_intro = (
        "光柱和天空遮挡都用阴影图（shadow map）计算：在 Blender 里沿太阳方向，以及把天空分成 107 个方向单元格后沿每个单元格的中心方向，"
        "各渲染一张只含深度的正交投影图，记录每个方向上离光源最近的物体表面。雾中一点如果落在某张图记录的表面后面，就看不到这个方向的光。"
        f"太阳阴影图的纹素边长为 {100 * maps['texels']['sun'][0]:.1f} cm，天空的为 {sky_texels} cm。"
        "沿每条视线，在物体可能投下阴影的区间里采样可见性，相邻两个采样点之间仍用闭式解积分，再乘以这一段的可见性，"
        "所以结果依然确定、没有噪声，没有遮挡的地方和原来完全一样。太阳的阴影边缘锐利：把视线切成 8 个纹素长的小段，"
        "整段都在周围所有表面之上（或之下）的小段直接判为照亮（或遮挡），只有阴影边缘可能穿过的小段才每个纹素采样一次。"
        "天光来自许多单元格，变化平缓，每条视线只取 8 个采样点（按每段对相机的贡献均匀分布），每个单元格按相函数在格内的积分乘以天光在这个仰角范围的衰减加权；"
        "相邻视线的采样点在空间里挨得很近，所以先把它们对齐到比天空阴影图纹素更细的网格上，每个网格单元只查一次。"
        "多次散射的光也来自各个方向，用同样的方式遮挡，单元格改按到达这个仰角的单次散射光加权，地平线以下的部分不遮挡。"
        f"整套阴影图每个场景只需渲染一次（{maps['count']} 张，Cycles 每像素 1 个样本，CPU 上共 {maps['seconds']:.1f} s），与相机、分辨率和雾参数都无关。"
        "下表把第 255 帧 Cycles 的 Volume Direct pass（单次散射）分别和不加阴影、加阴影的闭式解对比，比值为 1 表示完全一致："
    )
    occlusion_text = (
        f"加阴影后，太阳项在表面像素上的比值从 {sun_without['surfaces']:.3f} 降到 {sun_with['surfaces']:.3f}，天光项从 {sky_without['surfaces']:.3f} 降到 {sky_with['surfaces']:.3f}，"
        f"都和 Cycles 吻合到约 1%；逐像素误差分别从 {pct(sun_without['rel_l1'])} 降到 {pct(sun_with['rel_l1'])}、从 {pct(sky_without['rel_l1'])} 降到 {pct(sky_with['rel_l1'])}。"
        "上图里方块投在雾中的阴影（斜向的暗楔）在加阴影后完整出现；剩下的天光误差主要是 Cycles 自身的渲染噪声（地面附近雾光弱，噪声相对更大），"
        "以及物体边缘的抗锯齿差异。天空单元格宽 15°～120°，遮挡物的边缘正好落在视线的前向散射波瓣上时（例如沿墙面方向看），"
        "整格只按中心方向判定为全遮挡或全可见，会多算或少算一部分遮挡；这在合成测试里最多差约 25%，在本场景里整体只差约 1%。"
    )
    reuse_intro = (
        "沿视线追踪阴影，也就是找出每条视线上太阳被挡住的区间、在采样点上查每个天空单元格是否可见，只和场景几何与相机有关，"
        "所以每帧只需追踪一次，之后每组雾参数都复用这些可见性，只重新做闭式积分；雾的各向异性变了也能复用，单元格的相函数权重会按保存的可见性重新计算。"
        "不过天光的采样点是按追踪时的雾选的（按每段对相机的贡献均匀分布），视线上的采样范围也截到这团雾光学厚度为 12 的位置，所以换一组雾后，结果和为它重新追踪略有不同。"
        "下表每隔 10 帧（共 50 帧）用这团雾追踪一次阴影，再用于密度乘以不同倍数的雾，和为每组雾各自追踪的结果对比："
    )
    reuse_text = (
        f"复用的误差只相当于阴影带来的修正的 {pct(min(of_shadowed))}～{pct(max(of_shadowed))}，占雾散射光的 {fmt(min(of_inscatter), 'pct2')}～{fmt(max(of_inscatter), 'pct2')}。"
        f"误差几乎全部来自天光：太阳被挡住的区间与雾无关，复用后最多差 {fmt(max(e['of_shadowed_light'] for e in reuse_sun.values()), 'pct2')}；"
        "天光每条视线只有 8 个采样点，雾一变，它们就不再按新的贡献均匀分布。雾变浓时，贡献集中到离相机更近的地方，那里的采样点偏少，"
        f"密度变为 {float(max(reuse_sky, key=float)):g} 倍时，天光的遮挡差了 {pct(densest['of_shadowed_light'])}，"
        f"不过这时被物体挡住的天光只占雾所散射天光的 {pct(densest['of_inscatter'] / densest['of_shadowed_light'])}。雾参数的范围很大时，可以按几档密度各追踪一次。"
        "下表是每帧的开销（RTX 5080，GPU 空闲时取最快的一次）："
    )
    reused = [s[k] for s in reuse["seconds"].values() for k in ("closed_form_occ", "closed_form_ms_occ")]
    reused_speedups = [cycles_ms_seconds[k] / v for k, v in shadowed_seconds.items()]
    reuse_time_text = (
        f"复用阴影后加一组雾只要 {fmt(min(reused), 'sec')}～{fmt(max(reused), 'sec')}，多次散射 + 阴影仍比打开多次散射的 Cycles 快 {min(reused_speedups):.0f}～{max(reused_speedups):.0f} 倍。"
        "它比不加阴影慢，是因为每条视线都要按保存的可见性逐段重新积分，开销随像素数增加，而不加阴影时的开销主要是与分辨率无关的查找表。"
        f"每帧的阴影在 800×800 时占 {size(reuse['bytes']['800x800'])} 显存，存盘并不划算，所以生成多组雾参数时应在同一遍里处理完每帧的所有雾参数。"
    )
    clear_800 = scale["clear"]
    # Time to trace shadows before tracing skipped repeated lookups, if it was kept when rerunning shadow_reuse.py
    before = RESULTS / "shadow_reuse_before_speedup.json"
    traced_before = (
        f"（改进前 {fmt(json.loads(before.read_text())['seconds']['800x800']['trace'], 'sec')}）"
        if before.exists()
        else ""
    )
    shadowed_first = scale["closed_form_ms_occ"]["first"]
    first_ratio = shadowed_first / scale["cycles_default_ms"]["first"]
    cycles_first = hours(VISIONSIM50_FRAMES * scale["cycles_default_ms"]["first"])
    if first_ratio < 0.9:
        first_versus = (
            f"比打开多次散射的 Cycles（{cycles_first}）少 {pct(1 - first_ratio)}，主要的好处仍在之后的每组雾参数"
        )
    elif first_ratio <= 1.1:
        first_versus = f"和打开多次散射的 Cycles（{cycles_first}）相当，好处在于之后的每组雾参数"
    else:
        first_versus = f"是打开多次散射的 Cycles（{cycles_first}）的 {first_ratio:.1f} 倍，好处在于之后的每组雾参数"
    scale_intro = (
        "VisionSIM-50 有 50 个室内场景，每个场景 12 秒，以 100 fps、800×800 渲染，共 59,950 帧，只包含真值（RGB、深度、法线、光流、分割），约 1.0 TB。"
        "下表在同样的 800×800 分辨率下，实测本实验场景每帧的耗时（RTX 5080；Cycles 用 visionsim 的默认设置把第 255 帧反复渲染 12 次取最快的一次，不含 Blender 启动和保存文件），"
        "再乘以帧数；误差对照同分辨率第 255–264 帧的多次散射参考（4096 spp）。参考级渲染每帧要几十秒以上，不在表中。"
    )
    scale_text = (
        f"闭式解第一份需要先渲染无雾画面（每帧 {clear_800:.1f} s，另外还要保存线性 EXR 和深度）和阴影图（每个场景约 {maps['seconds']:.0f} s），"
        f"加阴影时每帧还要追踪一次阴影（{fmt(scale['trace'], 'sec')}，共 {hours(VISIONSIM50_FRAMES * scale['trace'])}）；"
        f"之后每组雾参数复用这些阴影，加阴影要 {hours(VISIONSIM50_FRAMES * scale['closed_form_ms_occ']['update'])}，不加阴影只要 {hours(VISIONSIM50_FRAMES * scale['closed_form_ms']['update'])}；"
        f"打开多次散射的 Cycles 每组都要 {hours(VISIONSIM50_FRAMES * scale['cycles_default_ms']['update'])}。"
        f"所以加阴影时，第一份（{hours(VISIONSIM50_FRAMES * shadowed_first)}）{first_versus}；"
        f"如果数据集本来就要渲染无雾画面，加一组带阴影的雾只需追踪阴影再加雾，共 {hours(VISIONSIM50_FRAMES * (scale['trace'] + scale['closed_form_ms_occ']['update']))}。"
        "复用要求在同一遍里为每帧加上所有雾参数（见上文“多组雾参数共用阴影”）。"
        "不加阴影时每帧只要几十毫秒，也可以像传感器仿真一样在读取数据时现场加雾，不必把每组雾参数的结果都存下来；加阴影时每次读取都要先追踪阴影，更适合一次生成多组雾参数。"
        "注意三点：本场景比 VisionSIM-50 的室内场景简单，Cycles 的耗时会随场景复杂度和弹射次数增加，而闭式解只和像素数有关；"
        "VisionSIM-50 发布的是色调映射后的 8 位 PNG，要用闭式解需要按线性 EXR 重新渲染一遍；室内场景主要靠灯光照明，目前点光源只算单次散射，聚光灯和面光源还不支持。"
        "插帧和传感器仿真的耗时对两种方法相同，不在表中。"
    )
    bounces_intro = (
        f"把第 {bounces['frames'][0]} 和 {bounces['frames'][1]} 帧（后者是闭式解误差最大的一帧）用 Cycles 以 0 到 {max(by_bounces)} 次体积弹射各渲染一遍"
        "（1024 spp，同一个随机种子）。0 次就是单次散射，也是 Blender 的默认值："
    )
    bounces_text = (
        f"多散射一次就让画面亮了 {pct(by_bounces[1]['brightness'] - 1)}，8 次后收敛：画面总亮度是单次散射的 {most['brightness']:.2f} 倍，物体表面像素上是 {most['brightness_surfaces']:.2f} 倍。"
        "这团雾完全不吸收，垂直方向的光学厚度只有 0.2，但沿地面方向 200 m 的光学厚度约为 10，光在水平方向上反复散射后大量进入视线。"
        f"闭式解相对它的误差随弹射次数从 {pct(single['closed_form_error'])} 升到 {pct(most['closed_form_error'])}；而打开多次散射只让 Cycles 每帧多花 {pct(most['seconds_per_frame'] / single['seconds_per_frame'] - 1)} 的时间。"
    )
    overview_ms_caption = (
        "第 250 帧，以多次散射为真值。多次散射让远处的地面、方块和天空都更亮、更白（最右列）。只算单次散射的两列（Cycles 单次散射、闭式解（单次散射））"
        "在远处的地面和背景上误差都超过 50%；闭式解（多次散射）把背景和地面的误差降了下来，剩下的集中在方块和它周围的雾里，也就是光柱和遮挡所在的位置；"
        "加上阴影后，方块周围的误差明显减小，剩下的主要在方块表面（表面透过雾被照亮）。打开多次散射的 Cycles 默认设置几乎看不出误差。"
    )

    return {
        "findings": findings,
        "ms_intro": ms_intro,
        "ms_text": ms_text,
        "bounces_intro": bounces_intro,
        "bounces_text": bounces_text,
        "validation_intro": validation_intro,
        "validation_text": validation_text,
        "occlusion_intro": occlusion_intro,
        "occlusion_text": occlusion_text,
        "reuse_intro": reuse_intro,
        "reuse_text": reuse_text,
        "reuse_time_text": reuse_time_text,
        "scale_intro": scale_intro,
        "scale_text": scale_text,
        "overview_ms_caption": overview_ms_caption,
        "tradeoff_caption": "每个点是一种方法。横轴是只改雾参数时生成一帧带雾画面的耗时（Cycles 要整帧重新渲染，闭式解和 ray marching 只需在无雾渲染上加雾，加阴影时复用每帧追踪好的阴影），"
        "纵轴是渲染帧相对参考的 L1 误差，两轴都是对数坐标。左图以单次散射为真值，右图以多次散射为真值。虚线是完全不加雾的误差，点线是参考自身的噪声底。",
        "overview_caption": "第一行：各方法生成的带雾画面；第二行：相对参考的误差。闭式解和 ray marching 的误差集中在方块表面和方块周围的雾里，对应“差距从哪来”中的三种可见性效应；"
        "加阴影后方块周围雾里的误差基本消失，剩下的集中在方块表面。"
        "不降噪的 Cycles 满屏都是噪声，这些噪声会原样进入每一种传感器。最后一行 DVS 里，不降噪的 Cycles 在绿色方块表面和天空中出现了大量散落的假事件；"
        "三列 Cycles 渲染（包括参考）在右上角的天空里都有少量噪声事件，闭式解和 ray marching 没有。",
        "curves_caption": "逐帧 loss。t≈2.7 s（第 340 帧）时，相机紧贴着绿色方块经过，方块占了画面右侧的一大片，三种可见性效应都集中在它身上："
        "闭式解和 ray marching 的误差达到最大，方块表面的误差超过 50%，RGB 的 PSNR 从约 34 dB 降到约 25 dB。"
        "Cycles 默认设置的误差很平稳；不降噪版本则被噪声压在固定的水平。DVS 曲线首尾的骤降是因为相机缓入缓出，这时事件很少。ToF 子图中 ray marching 的瞬态分别用了 128 步和 1024 步。",
        "rgb_text": f"RGB 的 loss 是在期望输出（不含光子噪声和读出噪声）上计算的。不降噪的 Cycles 虽然平均误差（{pct(nd['rgb_rel_l1'])}）比闭式解（{pct(cf['rgb_rel_l1'])}）还低，"
        f"但 SSIM 只有 {nd['rgb_ssim']:.3f}：噪声破坏了局部结构，而闭式解的误差是平滑的低频偏差，SSIM 为 {cf['rgb_ssim']:.3f}。",
        "spad_text": f"SPAD 的检测概率 p = 1 − e<sup>−λ</sup> 直接由画面决定，所以它的 loss 排序和渲染帧基本一致。KL 散度表示：用某种方法的二值帧代替参考的二值帧时，每个像素每帧损失的信息量。"
        f"闭式解是 {1000 * cf['spad_kl_bits']:.1f}×10⁻³ bit，完全不加雾是 {1000 * cl['spad_kl_bits']:.0f}×10⁻³ bit。",
        "dvs_text": "事件 F1 的计算方式：按 40 ms 窗口、像素和极性统计事件数，把所有窗口汇总后与参考对比；每窗口 F1 的平均值见第二列。v2e 关闭了所有噪声源，所以事件只由输入画面决定。"
        "后两列是 v2e 看到的对数亮度相对参考的平均绝对误差：单帧误差衡量画面整体的偏差，帧间误差衡量这个偏差在相邻两帧之间变了多少。事件由亮度变化触发，所以只有后者会产生假事件。"
        "Cycles 每一帧都用同一个随机种子（Blender 的默认设置），噪声图案大致固定在画面上，所以它和参考自身噪声底的帧间误差都远小于单帧误差。"
        f"闭式解的事件数（{cf['dvs_events']:,}）比参考少 {pct(1 - cf['dvs_events'] / r['reference']['dvs_events'])}：它的对数亮度平均每帧变化 {cf['log_change']:.4f}，"
        f"比参考的 {r['reference']['log_change']:.4f} 小 {pct(1 - cf['log_change'] / r['reference']['log_change'])}，因为它在表面上多算了雾的散射光（见“差距从哪来”），对比度偏低，一部分亮度变化没有越过阈值。"
        f"不降噪的 Cycles 帧间误差是闭式解的 {nd['log_temporal'] / cf['log_temporal']:.1f} 倍，噪声直接变成假事件：事件数多出 {pct(extra_events)}，"
        f"F1 降到 {nd['dvs_f1_pooled']:.3f}，和完全不加雾（{cl['dvs_f1_pooled']:.3f}）差不多。",
        "tof_text": "Cycles 不能渲染瞬态，所以 ToF 的参考是同一个瞬态模型的高精度解：每个 bin 用 16 点 Gauss–Legendre 积分，背景光取 Cycles 参考的被动辐射度。"
        "因此这里的激光回波 loss 只衡量数值误差，不包括模型本身（单次散射、HG 相函数）和真实物理之间的差距。"
        "“激光回波”只比较表面回波和雾的后向散射，这部分与被动画面无关。“含背景光”则把各方法自己算出的被动辐射度也作为背景光子算进去，所以会同时反映被动模型的误差。"
        "测距取期望光通量的峰值（无噪声），或者取 500 个周期的首光子直方图经 Coates 校正后的峰值（含光子噪声）。"
        f"闭式解在含光子噪声时的 {pct(cf['tof_disagree'])} 不一致，来自背景光水平的差异改变了随机采样结果；无噪声时两者完全一致。",
        "convergence_text": f"Ray marching 在这种平滑的高度雾里收敛得很快：误差大约按步数的平方下降，16 步时和闭式解只差 {sci(steps[16]['error_vs_closed_form'])}，256 步时降到 {sci(steps[256]['error_vs_closed_form'])}，接近 float32 的精度下限。"
        "但它的每一步都要重新计算到达这一点的光：天空要对 1152 个方向求和，所以耗时和步数成正比；闭式解则把天光按仰角制成查找表，每帧只算一次。"
        f"如果朝太阳方向也用 ray marching 来算衰减，误差会被这部分主导，停在 {sci(shadow[64]['error_vs_closed_form'])} 左右。"
        f"8 步以上时，和参考之间的误差都在 {pct(min(converged))}～{pct(max(converged))} 之间，闭式解是 {pct(convergence['closed_form_error_vs_reference'])}：剩下的是模型误差，不是数值误差。",
        "breakdown_intro": "闭式解对它自己的模型是精确的：太阳和环境光项与逐点暴力积分吻合到 6 位有效数字，天光项的方向求积误差低于 0.1%。为了找出它和单次散射真值差在哪里，在第 255 帧（相机正从红色方块旁经过）用 Cycles 渲染了几种只保留部分光源、或去掉部分物体的变体，"
        "再把 Cycles 的 Volume Direct pass（雾对相机射线的单次散射）和闭式解的散射光逐项对比。比值为 1 表示完全一致。",
        "breakdown_text": f"只要物体不参与光的可见性计算，太阳项（{bd['sun_noshadow']['surfaces']:.3f}）和天光项（{bd['sky_noobjects']['surfaces']:.3f}）都和 Cycles 吻合到 1% 以内；天空像素的透射率也一致（比值 {bd['sky_transmittance']:.3f}）。"
        "剩下的差距全部来自三种和几何有关的可见性效应，它们都是真实存在的物理现象："
        f"① 物体在雾里投下的阴影（光柱）：物体投影后，表面像素上太阳项的比值从 {bd['sun_noshadow']['surfaces']:.3f} 升到 {bd['sun']['surfaces']:.3f}；"
        f"② 附近的物体挡住了雾中各点能看到的一部分天空：放回方块和柱子后，天光项的比值从 {bd['sky_noobjects']['surfaces']:.3f} 升到 {bd['sky']['surfaces']:.3f}；"
        f"③ 表面本身是透过雾被照亮的：阳光和天光到达表面之前都被雾衰减，雾的散射又补回一部分，净效果是表面在 Cycles 里暗了 {pct(1 - bd['surface_dimming']['median'])}（中位数，p10–p90 为 {pct(1 - bd['surface_dimming']['p90'])}–{pct(1 - bd['surface_dimming']['p10'])}）。"
        f"①和②现在已经用阴影图建模，加阴影后两项的比值分别回到 {sun_with['surfaces']:.3f} 和 {sky_with['surfaces']:.3f}（见上文“光柱和天空遮挡”），③还没有建模。"
        f"修正前的模型让下半球也向雾里照射天光，并且天光不衰减，结果表面像素上的散射光是 Cycles 的 {bd['full_before_fix']['surfaces']:.1f} 倍。",
        "next_steps": f"""<ul>
<li><strong>表面透过雾被照亮（现在最大的误差来源）</strong>：用 Blender 的 Light Groups 把表面上的太阳贡献和天光贡献分开，分别乘以表面点处的阳光透射率和天光透射率（高度雾都有闭式解，阴影图也能直接用）；雾光对表面的照明可以用多次散射表里已经算好的、散射过一次的光场来近似。</li>
<li><strong>阴影的速度</strong>：多组雾参数共用每帧的阴影，追踪阴影本身也已经去掉了大部分重复查询，800×800 时每帧 {fmt(scale["trace"], "sec")}{traced_before}，是无雾渲染的 {pct(scale["trace"] / clear_800)}。天空在每个网格单元上的可见性与相机和雾都无关，还可以跨帧缓存；查阴影图的部分用自定义 GPU 核函数合并后还能再快几倍。</li>
<li><strong>阴影的精度</strong>：天空单元格较粗，遮挡物边缘正好落在前向散射波瓣上时会整格误判，可以每个单元格渲染几张子方向的阴影图，沿视线交替使用；运动的物体需要逐帧渲染阴影图；地面反射到雾里的光和点光源也还没有阴影。</li>
<li><strong>点光源的多次散射、聚光灯和面光源</strong>：点光源的多次散射可以用大气点扩散函数（Narasimhan 和 Nayar 2003）一类的解析近似；聚光灯只是带角度衰减的点光源，面光源可以用若干个点光源近似。</li>
<li><strong>多次散射近似的改进</strong>：地平线附近少约 7%，俯视地面时多约 12%。可以用 Cycles 的无物体参考标定级数因子，或者把收集过程迭代两三次代替几何级数；地面也可以换成更接近 Principled BSDF 的反射模型。</li>
<li><strong>ToF</strong>：HG 相函数会低估雾滴在 180° 附近的后向散射（真实雾有 glory 峰），可以改用 lidar ratio 参数化；激光在雾中的多次散射可以用支持瞬态渲染的 mitransient（Mitsuba 3）生成参考。</li>
<li><strong>其他</strong>：visionsim 保存的相机坐标系法线方向是反的，实验里已经换算回世界坐标。这个问题已经单独开了一个修复任务。</li>
</ul>""",
    }


def method_rows(summary: dict, seconds: dict, labels: dict[str, str], temporal: dict | None = None) -> dict[str, dict]:
    """Metrics of each method against the reference, and the time it takes to produce a frame with fog."""
    rows = {}
    for name in [*(m for m in labels if m in summary["rgb"]), "reference"]:
        row = {"first": seconds[name][0], "update": seconds[name][1]}
        if name in summary["frames"]:
            row |= {"frames_rel_l1": summary["frames"][name]["rel_l1"], "frames_psnr": summary["frames"][name]["psnr"]}
        if name in summary["rgb"]:
            row |= {f"rgb_{k}": v for k, v in summary["rgb"][name].items()}
            row |= {f"spad_{k}": v for k, v in summary["spad"][name].items()}
            row |= {"dvs_f1": summary["dvs"][name]["f1"], "dvs_f1_pooled": summary["dvs"][name]["f1_pooled"]}
        row["dvs_events"] = summary["dvs"].get(name, {}).get("events")
        if name in summary["tof"]:
            row |= {f"tof_{k}": v for k, v in summary["tof"][name].items()}
        row |= {f"log_{k}": v for k, v in (temporal or {}).get(name, {}).items()}
        rows[name] = row
    return rows


# Points of the trade-off chart: label, color, marker and where the label goes relative to the point
POINTS = {
    "closed_form": ("闭式解", "#b0304a", "D", (-8, 3, "right")),
    "closed_form_occ": ("闭式解 + 阴影", "#d9667f", "D", (0, -14, "center")),
    "closed_form_ms": ("闭式解（多次散射）", "#6b1d2c", "D", (7, -3, "left")),
    "closed_form_ms_occ": ("闭式解（多次散射）+ 阴影", "#3d0f19", "D", (0, -15, "center")),
    "closed_form_m1": ("闭式解（修正前）", "#b9a3c9", "D", (0, -14, "center")),
    "raymarch_16": ("RM 16 步", "#d9822b", "s", (0, 8, "center")),
    "raymarch_64s16": ("RM 64+16 步", "#e8b27a", "s", (7, 3, "left")),
}
CYCLES_MS = {
    "cycles_default": ("Cycles 默认（多次散射）", "#1f5fa8", "o", (7, 3, "left")),
    "cycles_nodenoise": ("Cycles 不降噪（多次散射）", "#6d9fd6", "o", (7, 3, "left")),
    "cycles_default_ss": ("Cycles 默认（单次散射）", "#7b8794", "o", (7, 4, "left")),
}
CYCLES_SS = {
    "cycles_default": ("Cycles 默认（单次散射）", "#7b8794", "o", (7, 3, "left")),
    "cycles_nodenoise": ("Cycles 不降噪（单次散射）", "#aab4bd", "o", (7, -12, "left")),
}


def tradeoff_chart(panels: list[tuple[str, dict, dict]], path: Path) -> None:
    """Error of each method against the time it takes to produce a frame with fog when only the fog changes, i.e.
    re-rendering for Cycles, and adding the fog to the clear render for the closed form and ray marching."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (title, rows, points) in zip(axes[0], panels):
        for name, (label, color, marker, (dx, dy, align)) in points.items():
            if name not in rows or rows[name].get("update") is None:
                continue
            x, y = rows[name]["update"], rows[name]["frames_rel_l1"]
            ax.scatter(x, y, s=46, color=color, marker=marker, zorder=3)
            ax.annotate(label, (x, y), fontsize=8.5, xytext=(dx, dy), textcoords="offset points", ha=align)
        ax.axhline(rows["clear"]["frames_rel_l1"], ls="--", lw=1, color="#888", label="不加雾")
        if "reference_seed1" in rows:
            ax.axhline(rows["reference_seed1"]["frames_rel_l1"], ls=":", lw=1, color="#888", label="参考自身的噪声底")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.5e-3, 1.5)
        ax.set_xlabel("只改雾参数时，生成一帧带雾画面的耗时 (s)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("渲染帧相对参考的 L1 误差")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def bounces_table(bounces: dict) -> str:
    """Brightness and closed form error as the number of volume bounces grows, from ``bounces.py``."""
    entries = bounces["bounces"]
    head = "".join(f'<th scope="col">{e["volume_bounces"]}</th>' for e in entries)
    lines = [
        ("画面总亮度（相对单次散射）", "brightness", "{:.2f}"),
        ("其中天空像素", "brightness_sky", "{:.2f}"),
        ("其中物体表面像素", "brightness_surfaces", "{:.2f}"),
        ("雾散射进相机的光中，散射过不止一次的占比", "indirect_share", "pct"),
        ("闭式解相对它的误差（渲染帧相对 L1）", "closed_form_error", "pct"),
        ("Cycles 每帧耗时（1024 spp）", "seconds_per_frame", "{:.2f} s"),
    ]
    body = "".join(
        f'<tr><th scope="row">{title}</th>'
        + "".join(f"<td>{fmt(e[key], 'pct') if style == 'pct' else style.format(e[key])}</td>" for e in entries)
        + "</tr>"
        for title, key, style in lines
    )
    return (
        f'<div class="table-wrap"><table><thead><tr><th scope="col">最大体积弹射次数</th>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def validation_table(validation: dict) -> str:
    """Ratio of the closed form's in-scattering to Cycles' in the scene without objects, by elevation of the rays."""
    ranges = [k for k in validation["multiple_vs_32_bounces"] if k != "all"]
    head = "".join(f'<th scope="col">{k.replace("..", " ～ ")}°</th>' for k in ranges)
    rows = [
        ("单次散射 ÷ Cycles 0 次弹射", validation["single_vs_0_bounces"]),
        ("加上地面反射和多次散射 ÷ Cycles 32 次弹射", validation["multiple_vs_32_bounces"]),
    ]
    body = "".join(
        f'<tr><th scope="row">{label}</th><td>{r["all"]:.3f}</td>'
        + "".join(f"<td>{r[k]:.2f}</td>" for k in ranges)
        + "</tr>"
        for label, r in rows
    )
    return (
        f'<div class="table-wrap"><table><thead><tr><th scope="col">雾散射进相机的光</th><th scope="col">全部</th>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def occlusion_table(occlusion: dict) -> str:
    """Light scattered by the fog in the closed form, with and without shadows, against Cycles, on frame 255."""
    body = "".join(
        f'<tr><th scope="row">{label}</th>'
        + "".join(
            f"<td>{occlusion[f'{variant}_{kind}']['sky']:.3f}</td><td>{occlusion[f'{variant}_{kind}']['surfaces']:.3f}</td>"
            f"<td>{fmt(occlusion[f'{variant}_{kind}']['rel_l1'], 'pct')}</td>"
            for kind in ("without", "with")
        )
        + "</tr>"
        for variant, label in (("sun", "只有太阳（光柱）"), ("sky", "只有天光（遮挡）"))
    )
    head = "".join(f'<th scope="col">{h}</th>' for h in ("天空像素", "表面像素", "逐像素误差") * 2)
    return (
        '<div class="table-wrap"><table><thead>'
        '<tr><th scope="col" rowspan="2">第 255 帧，雾的散射光：闭式解 / Cycles</th>'
        '<th scope="col" colspan="3">无阴影</th><th scope="col" colspan="3">加阴影</th></tr>'
        f"<tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def shadow_maps() -> dict:
    """Number of shadow maps of the scene, and the time it took to render them, from ``export_occlusion.py``."""
    with np.load(ROOT / "occlusion" / "occlusion.npz") as data:
        count = len(data["sun_texels"]) + len(data["sky_texels"])
        texels = {
            kind: (float(data[f"{kind}_texels"].min()), float(data[f"{kind}_texels"].max())) for kind in ("sun", "sky")
        }
    log = (ROOT / "occlusion" / "export.log").read_text(encoding="utf-8", errors="replace")
    return {"count": count, "seconds": float(re.findall(r"OCCLUSION_TIME ([\d.]+)s", log)[-1]), "texels": texels}


def reuse_tables(reuse: dict, timings: dict, cycles: dict[str, float]) -> tuple[str, str]:
    """Error of reusing the shadows of a frame for fogs of other densities, and time to trace shadows and to add a fog
    with them, against adding it without shadows and rendering it with Cycles, from ``shadow_reuse.py``."""
    errors = reuse["errors"]
    lines = [
        ("全部光源（多次散射）：相对阴影带来的修正", errors["all"], "of_shadowed_light", "pct"),
        ("全部光源（多次散射）：相对雾的散射光", errors["all"], "of_inscatter", "pct2"),
        ("只有太阳：相对阴影带来的修正", errors["sun"], "of_shadowed_light", "pct"),
        ("只有天光：相对阴影带来的修正", errors["sky"], "of_shadowed_light", "pct"),
    ]
    head = "".join(f'<th scope="col">{float(k):g} 倍</th>' for k in errors["all"])
    body = "".join(
        f'<tr><th scope="row">{label}</th>' + "".join(f"<td>{fmt(e[key], kind)}</td>" for e in values.values()) + "</tr>"
        for label, values, key, kind in lines
    )
    error_table = (
        f'<div class="table-wrap"><table><thead><tr><th scope="col">复用的误差（相对 L1），雾的密度</th>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
    seconds = reuse["seconds"]
    lines = [
        ("追踪阴影（每帧一次）", [s["trace"] for s in seconds.values()]),
        ("复用阴影，加一组雾（单次散射）", [s["closed_form_occ"] for s in seconds.values()]),
        ("复用阴影，加一组雾（多次散射）", [s["closed_form_ms_occ"] for s in seconds.values()]),
        ("对照：不加阴影（多次散射）", [timings[k]["closed_form_ms"] for k in seconds]),
        ("对照：Cycles 默认设置（多次散射）", [cycles[k] for k in seconds]),
    ]
    head = "".join(f'<th scope="col">{k.replace("x", "×")}</th>' for k in seconds)
    body = "".join(
        f'<tr><th scope="row">{label}</th>' + "".join(f"<td>{fmt(v, 'sec')}</td>" for v in values) + "</tr>"
        for label, values in lines
    )
    body += (
        '<tr><th scope="row">每帧阴影占用的显存</th>'
        + "".join(f"<td>{size(reuse['bytes'][k])}</td>" for k in seconds)
        + "</tr>"
    )
    time_table = (
        f'<div class="table-wrap"><table><thead><tr><th scope="col">每帧的开销</th>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
    return error_table, time_table


VISIONSIM50_FRAMES = 59_950
"""Frames of VisionSIM-50: 50 scenes animated for 12 s, rendered at 100 fps and 800x800"""


def cycles_seconds(scene: str, resolution: str) -> float:
    """Fastest of repeated Cycles renders of frame 255 with visionsim's default settings, from ``time_cycles.py``,
    where ``scene`` is ``demo`` (without fog), ``fog_default`` or ``fog_default_ms``."""
    text = (ROOT / "cycles_timing" / f"{scene}_{resolution}.log").read_text(encoding="utf-8", errors="replace")
    return float(re.findall(r"CYCLES_TIME min=([\d.]+)s", text)[-1])


def size(count: int) -> str:
    """Number of bytes in MB or GB."""
    return f"{count / 1e6:.0f} MB" if count < 1e9 else f"{count / 1e9:.1f} GB"


def hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分钟"
    if seconds < 10 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 3600:,.0f} h" if seconds < 72 * 3600 else f"{seconds / 86400:,.1f} 天"


def convergence_chart(convergence: dict, seconds: dict, path: Path) -> None:
    """Error of ray marching against the closed form, against its time per frame, as the number of steps grows."""
    times = {(c["steps"], c["shadow_steps"]): c["seconds"] for c in seconds["ray_marching"]}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for shadow, marker in ((0, "o"), (16, "s")):
        points = [c for c in convergence["ray_marching"] if c["shadow_steps"] == shadow]
        x = [times[c["steps"], shadow] for c in points]
        y = [c["error_vs_closed_form"] for c in points]
        ax.loglog(x, y, marker=marker, label=f"Ray marching（{'光照衰减用闭式' if shadow == 0 else '朝太阳 16 步'}）")
        for c, xi, yi in zip(points, x, y):
            ax.annotate(str(c["steps"]), (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axvline(seconds["closed_form"], color="k", ls="--", lw=1, label="闭式解耗时")
    ax.set_xlabel("每帧耗时 (s)")
    ax.set_ylabel("与闭式解的相对 L1 误差")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def scale_table(timings: dict, quality: dict, shadows: dict, variants: int = 10) -> tuple[str, dict]:
    """Time to produce fog for as many frames as VisionSIM-50 at its resolution, from 10 frames of the experiment's
    scene rendered at 800x800 (timing_resolution.ps1) and the time taken by the closed form and ray marching, where
    shadows are traced once per frame and reused for every fog (shadow_reuse.py)."""
    clear = cycles_seconds("demo", "800x800")
    rows = [
        ("Cycles 默认设置（多次散射）", cycles_seconds("fog_default_ms", "800x800"), None, "cycles_default_ms"),
        ("Cycles 默认设置（单次散射）", cycles_seconds("fog_default", "800x800"), None, "cycles_default"),
        ("闭式解（多次散射）+ 阴影", shadows["closed_form_ms_occ"], clear + shadows["trace"], "closed_form_ms_occ"),
        ("闭式解（多次散射）", timings["closed_form_ms"], clear, "closed_form_ms"),
        ("闭式解（单次散射）", timings["closed_form"], clear, "closed_form"),
        ("Ray marching 16 步", timings["raymarch_16"], clear, "raymarch_16"),
    ]
    body: list[str] = []
    numbers: dict[str, float | dict] = {"clear": clear, "trace": shadows["trace"]}
    for label, update, base, key in rows:
        error = quality.get(key, {}).get("rel_l1")
        first = (base or 0) + update
        cells = [
            fmt(update, "sec"),
            fmt(error, "pct") if error is not None else "—",
            hours(VISIONSIM50_FRAMES * first),
            hours(VISIONSIM50_FRAMES * update),
            hours(VISIONSIM50_FRAMES * ((base or 0) + variants * update)),
        ]
        body.append(f'<tr><th scope="row">{html.escape(label)}</th>' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        numbers[key] = {"update": update, "first": first, "error": error}
    head = "".join(
        f'<th scope="col">{h}</th>'
        for h in (
            "只改雾参数时每帧",
            "渲染帧误差（多次散射真值）",
            "第一份（含无雾渲染和追踪阴影）",
            "每多一组雾参数",
            f"{variants} 组雾参数",
        )
    )
    table = (
        f'<div class="table-wrap"><table><thead><tr><th scope="col">方法（800×800）</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
    return table, numbers


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    summary = metrics["summary"]
    convergence = json.loads((RESULTS / "convergence.json").read_text(encoding="utf-8"))
    breakdown = json.loads((RESULTS / "breakdown.json").read_text(encoding="utf-8"))
    temporal = json.loads((RESULTS / "temporal.json").read_text()) if (RESULTS / "temporal.json").exists() else {}
    bounces = json.loads((RESULTS / "bounces.json").read_text())
    summary_ms = json.loads((ROOT / "results_ms" / "metrics.json").read_text(encoding="utf-8"))["summary"]
    renders = render_seconds()
    hd = hd_timings()
    passive = [m for m in LABELS if m in summary["rgb"]]
    passive_ms = [m for m in LABELS_MS if m in summary_ms["rgb"]]
    # The closed form and ray marching are timed separately with the GPU otherwise idle, see time_methods.py
    timings = json.loads((RESULTS / "timings.json").read_text())
    medium = timings["320x180"]
    times = {(c["steps"], c["shadow_steps"]): c["seconds"] for c in timings["convergence"]["ray_marching"]}
    convergence["closed_form_seconds"] = timings["convergence"]["closed_form"]
    for entry in convergence["ray_marching"]:
        entry["seconds"] = times[entry["steps"], entry["shadow_steps"]]
    convergence_chart(convergence, timings["convergence"], RESULTS / "images" / "convergence.png")
    validation = json.loads((RESULTS / "validation.json").read_text())
    regions = json.loads((ROOT / "results_ms" / "regions.json").read_text())
    quality_800 = json.loads((ROOT / "res800x800" / "results.json").read_text())["vs_multiple"]
    # Shadows are traced once per frame, then reused for every fog, see shadow_reuse.py
    reuse = json.loads((RESULTS / "shadow_reuse.json").read_text())
    scale_html, scale = scale_table(timings["800x800"], quality_800, reuse["seconds"]["800x800"])
    occlusion = json.loads((RESULTS / "occlusion.json").read_text())
    maps = shadow_maps()

    # Time to produce a frame with fog, the first time, and after changing only the fog's parameters
    clear_render = renders["clear"]
    seconds = {"clear": (clear_render, None), "reference_seed1": (None, None)}  # not a method, the reference's noise
    for name in ("raymarch_16", "raymarch_64s16", "closed_form_m1", "closed_form", "closed_form_ms"):
        seconds[name] = (clear_render + medium[name], medium[name])
    shadowed = reuse["seconds"]["320x180"]
    for name in ("closed_form_occ", "closed_form_ms_occ"):
        seconds[name] = (clear_render + shadowed["trace"] + shadowed[name], shadowed[name])
    cycles_ms = {
        "320x180": renders["ms_cycles_default"],
        "800x800": cycles_seconds("fog_default_ms", "800x800"),
        "1920x1080": cycles_seconds("fog_default_ms", "1920x1080"),
    }
    reuse_error_table, reuse_time_table = reuse_tables(reuse, timings, cycles_ms)
    seconds_ss = seconds | {
        "cycles_default": (renders["cycles_default"],) * 2,
        "cycles_nodenoise": (renders["cycles_nodenoise"],) * 2,
        "reference": (renders["cycles_ref"],) * 2,
    }
    seconds_ms = seconds | {
        "cycles_default": (renders["ms_cycles_default"],) * 2,
        "cycles_nodenoise": (renders["ms_cycles_nodenoise"],) * 2,
        "cycles_default_ss": (renders["cycles_default"],) * 2,
        "reference": (renders["ms_cycles_ref"],) * 2,
    }
    rows = method_rows(summary, seconds_ss, LABELS, temporal)
    rows_ms = method_rows(summary_ms, seconds_ms, LABELS_MS)
    tradeoff_chart(
        [
            ("真值：单次散射（Blender 默认）", rows, POINTS | CYCLES_SS),
            ("真值：多次散射（物理上完整）", rows_ms, POINTS | CYCLES_MS),
        ],
        RESULTS / "images" / "tradeoff.png",
    )
    # The report only references files next to it
    shutil.copyfile(ROOT / "results_ms" / "images" / "overview.png", RESULTS / "images" / "overview_ms.png")
    shutil.copyfile(ROOT / "results_ms" / "videos" / "sensors.mp4", RESULTS / "videos" / "sensors_ms.mp4")

    body = TEMPLATE.format(
        style=STYLE,
        ms_summary_table=table(
            [*passive_ms, "reference"],
            [
                ("frames_rel_l1", "渲染帧 相对 L1 ↓", "pct", False),
                ("rgb_psnr", "RGB PSNR (dB) ↑", "db", True),
                ("spad_mae", "SPAD 检测概率 MAE ×10⁻³ ↓", "milli", False),
                ("dvs_f1_pooled", "DVS 事件 F1 ↑", "f3", True),
                ("first", "生成一帧", "sec", False),
                ("update", "只改雾参数时", "sec", False),
            ],
            rows_ms,
            "所有 loss 都相对多次散射参考（Cycles 4096 spp，最多 16 次体积弹射）计算。绿色为除参考外的最佳值，灰色为参考自身的噪声底。"
            "“只改雾参数时”：Cycles 要整帧重新渲染，闭式解和 ray marching 只需在无雾渲染上加雾；加阴影的行复用每帧追踪好的阴影，“生成一帧”包含追踪阴影的时间。"
            "耗时包含 Blender 启动时间，在 RTX 5080 上测得。"
            "ToF 不在表中：Cycles 不能渲染瞬态，瞬态的参考只能用同一个单次散射模型，无法评估多次散射对激光回波的影响。",
            labels=LABELS_MS,
        ),
        bounces_table=bounces_table(bounces),
        validation_table=validation_table(validation),
        occlusion_table=occlusion_table(occlusion),
        reuse_error_table=reuse_error_table,
        reuse_time_table=reuse_time_table,
        scale_table=scale_html,
        summary_table=table(
            [*passive, "reference"],
            [
                ("frames_rel_l1", "渲染帧 相对 L1 ↓", "pct", False),
                ("rgb_psnr", "RGB PSNR (dB) ↑", "db", True),
                ("spad_mae", "SPAD 检测概率 MAE ×10⁻³ ↓", "milli", False),
                ("dvs_f1_pooled", "DVS 事件 F1 ↑", "f3", True),
                ("tof_disagree_expected", "ToF 测距不一致 ↓", "pct", False),
                ("first", "生成一帧", "sec", False),
                ("update", "只改雾参数时", "sec", False),
            ],
            rows,
            "所有 loss 都是相对单次散射参考（Cycles 4096 spp，0 次体积弹射）计算的。绿色为除参考外的最佳值。灰色的“参考本身的噪声”一行：换一个随机种子把参考重新渲染一遍，它和参考之间的差异就是参考的蒙特卡洛噪声底，"
            "任何方法的 loss 都不会明显低于它。— 表示不适用：Cycles 不能渲染时间分辨的瞬态，所以没有 ToF 结果。ToF 一列里，两行 ray marching 的瞬态分别用了 128 步和 1024 步。"
            "加阴影的行“只改雾参数时”复用每帧追踪好的阴影，“生成一帧”包含追踪阴影的时间。耗时包含 Blender 启动时间，在 RTX 5080 上测得。",
        ),
        rgb_table=table(
            passive,
            [
                ("rgb_psnr", "PSNR (dB) ↑", "db", True),
                ("rgb_ssim", "SSIM ↑", "f4", True),
                ("rgb_rel_l1", "曝光图 相对 L1 ↓", "pct2", False),
            ],
            rows,
        ),
        spad_table=table(
            passive,
            [
                ("spad_mae", "检测概率 MAE ×10⁻³ ↓", "milli", False),
                ("spad_rel_l1", "检测概率 相对 L1 ↓", "pct2", False),
                ("spad_kl_bits", "KL 散度 ×10⁻³ bit/像素/帧 ↓", "milli", False),
            ],
            rows,
        ),
        dvs_table=table(
            [*passive, "reference"],
            [
                ("dvs_f1_pooled", "事件 F1（全部窗口汇总）↑", "f3", True),
                ("dvs_f1", "每窗口 F1 的平均 ↑", "f3", True),
                ("dvs_events", "事件总数", "int", None),
                ("log_static", "单帧 log 亮度误差 ↓", "f4", False),
                ("log_temporal", "帧间 log 变化误差 ↓", "f4", False),
            ],
            rows,
        ),
        tof_table=table(
            [m for m in [*passive, "reference"] if m in summary["tof"]],
            [
                ("tof_rel_l1_laser", "激光回波 相对 L1 ↓", "pct2", False),
                ("tof_rel_l1", "含背景光 相对 L1 ↓", "pct2", False),
                ("tof_disagree_expected", "测距不一致（无噪声）↓", "pct", False),
                ("tof_disagree", "测距不一致（含光子噪声）↓", "pct", False),
                ("tof_valid", "测距正确率 ↑", "pct", True),
            ],
            rows,
            "ToF 对步长比被动画面敏感得多，所以两行 ray marching 的瞬态用了更多步数：“16 步”一行用 128 步，“64 步 + 朝太阳 16 步”一行用 1024 步；背景光分别取自对应的被动结果。",
        ),
        convergence_rows="".join(
            f'<tr><th scope="row">{c["steps"]} 步{"（+ 朝太阳 " + str(c["shadow_steps"]) + " 步）" if c["shadow_steps"] else ""}</th>'
            f"<td>{fmt(c['seconds'], 'sec')}</td><td>{c['error_vs_closed_form']:.1e}</td><td>{fmt(c['error_vs_reference'], 'pct2')}</td></tr>"
            for c in convergence["ray_marching"]
        ),
        closed_form_seconds=fmt(convergence["closed_form_seconds"], "sec"),
        closed_form_error=fmt(convergence["closed_form_error_vs_reference"], "pct2"),
        bd=breakdown,
        occ=occlusion,
        r=renders,
        **build_prose(
            rows,
            rows_ms,
            convergence,
            breakdown,
            renders,
            hd,
            bounces,
            timings,
            validation,
            scale,
            regions,
            occlusion,
            maps,
            reuse,
            cycles_ms,
        ),
    )
    (RESULTS / "index.html").write_text(body, encoding="utf-8")
    print("wrote", RESULTS / "index.html")


TEMPLATE = (
    (Path(__file__).parent / "report_template.html").read_text(encoding="utf-8")
    if (Path(__file__).parent / "report_template.html").exists()
    else ""
)

if __name__ == "__main__":
    main()
