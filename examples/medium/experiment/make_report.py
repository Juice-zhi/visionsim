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
    "closed_form": LABELS["closed_form"],
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
    rows: dict, rows_ms: dict, convergence: dict, bd: dict, renders: dict, hd: dict, bounces: dict
) -> dict[str, str]:
    """Text of the report, which quotes the measured numbers."""
    r = rows
    cf, rm, cd, nd, cl, m1 = (
        r[k] for k in ("closed_form", "raymarch_16", "cycles_default", "cycles_nodenoise", "clear", "closed_form_m1")
    )
    cf_ms, rm_ms, cd_ms, nd_ms, ss_ms, cl_ms, m1_ms = (
        rows_ms[k]
        for k in (
            "closed_form",
            "raymarch_16",
            "cycles_default",
            "cycles_nodenoise",
            "cycles_default_ss",
            "clear",
            "closed_form_m1",
        )
    )
    floor_ms = rows_ms.get("reference_seed1")
    pct = lambda v: f"{100 * v:.1f}%"
    steps = {c["steps"]: c for c in convergence["ray_marching"] if c["shadow_steps"] == 0}
    shadow = {c["steps"]: c for c in convergence["ray_marching"] if c["shadow_steps"]}
    cf_seconds = convergence["closed_form_seconds"]
    extra_events = nd["dvs_events"] / r["reference"]["dvs_events"] - 1
    # The model before the sky was fixed is the same closed form, without integrating skylight over directions
    sky_share = 1 - m1["update"] / cf["update"]
    by_bounces = {b["volume_bounces"]: b for b in bounces["bounces"]}
    single, most = by_bounces[0], by_bounces[max(by_bounces)]
    speedup, speedup_hd = cd_ms["update"] / cf["update"], hd["cycles_default_ms"] / hd["closed_form"]
    floor_text = f"（参考自身的噪声底为 {pct(floor_ms['frames_rel_l1'])}）" if floor_ms else ""
    temporal_text = (
        f"DVS 只对亮度随时间的变化敏感：闭式解在 v2e 看到的对数亮度上，单帧误差是 Cycles 默认设置的 {cf['log_static'] / cd['log_static']:.1f} 倍"
        f"（{cf['log_static']:.3f} 对 {cd['log_static']:.3f}），但相邻两帧之间误差的变化反而更小（{cf['log_temporal']:.4f} 对 {cd['log_temporal']:.4f}），"
        "因为它的误差是随场景缓慢变化的平滑偏差。"
        if "log_temporal" in cf
        else ""
    )

    findings = f"""<ul>
<li><strong>以物理上完整的多次散射为真值：闭式解只改雾参数时比 Cycles 快 {speedup:.1f} 倍（1080p 下 {speedup_hd:.1f} 倍），但渲染帧误差是 {pct(cf_ms["frames_rel_l1"])}，Cycles 默认设置只有 {pct(cd_ms["frames_rel_l1"])}{floor_text}。</strong>四种传感器上的差距同样大：RGB PSNR {cf_ms["rgb_psnr"]:.1f} 对 {cd_ms["rgb_psnr"]:.1f} dB，SPAD 检测概率 MAE {1000 * cf_ms["spad_mae"]:.0f} 对 {1000 * cd_ms["spad_mae"]:.1f}（×10⁻³），DVS 事件 F1 {cf_ms["dvs_f1_pooled"]:.3f} 对 {cd_ms["dvs_f1_pooled"]:.3f}。当前的闭式解在这团雾里甚至比完全不加雾（{pct(cl_ms["frames_rel_l1"])}）还差：以现在的模型，这个时间优势换来的精度损失不可接受。</li>
<li><strong>原因是多次散射，而闭式解和 ray marching 都只算单次散射。</strong>这团雾完全不吸收（反照率为 1），沿地面方向的光学厚度约为 10：雾散射进相机的光里有 {pct(most["indirect_share"])} 散射过不止一次，画面整体比单次散射亮 {pct(most["brightness"] - 1)}。ray marching 只是沿视线的数值积分，每一步的光源项和闭式解一样只含直接光，所以两者对多次散射真值的误差几乎一样（{pct(rm_ms["frames_rel_l1"])} 和 {pct(cf_ms["frames_rel_l1"])}）。多次散射没有一般的闭式解，因为雾中每一点的光源项本身取决于整团雾里的光场。</li>
<li><strong>Blender 默认的 0 次体积弹射也只是单次散射：</strong>用 Cycles 默认设置渲染这团雾，误差是 {pct(ss_ms["frames_rel_l1"])}。在 Cycles 里打开多次散射（8 次弹射就已收敛），每帧只从 {renders["cycles_default"]:.2f} s 增加到 {renders["ms_cycles_default"]:.2f} s（1080p 下从 {hd["cycles_default"]:.2f} s 增加到 {hd["cycles_default_ms"]:.2f} s）。在 visionsim 里用 Cycles 渲染雾时，应该把体积弹射次数设到 8 以上。</li>
<li><strong>加一个近似的多次散射项，就能收回大部分差距。</strong>修正天光前的 M1 模型把整个球面的天光都当成不衰减的入射光，多算的光恰好和多次散射补上的量级相近，所以它对多次散射真值的误差只有 {pct(m1_ms["frames_rel_l1"])}。这是巧合，不是正确的物理，但它说明给闭式解加上一个随高度变化的多次散射光源项，是最值得做的下一步（见“局限与下一步”）。</li>
<li><strong>和同样只算单次散射的真值比，闭式解的实现是对的：</strong>误差 {pct(cf["frames_rel_l1"])}，剩下的来自光柱、物体遮挡天空、表面透过雾被照亮这三种几何效应；ray marching 与它只差 {sci(steps[16]["error_vs_closed_form"])}，但耗时是它的 {steps[16]["seconds"] / cf_seconds:.1f} 倍。这时闭式解只改雾参数时比 Cycles 默认设置快 {cd["update"] / cf["update"]:.1f} 倍，误差 {pct(cf["frames_rel_l1"])} 对 {pct(cd["frames_rel_l1"])}。{temporal_text}细节见后面各节。</li>
<li><strong>ToF 只能由闭式解或 ray marching 生成</strong>，因为 Cycles 不能渲染时间分辨的瞬态。瞬态模型同样只算单次散射，参考也只能用这个模型自己，所以雾中激光的多次散射（它会拉长回波的拖尾）这次无法评估。在单次散射模型内，闭式解和每 bin 16 点积分的收敛解在 float32 精度内一致；ray marching 用 128 步时，{pct(rm["tof_disagree_expected"])} 的像素测距与参考不一致，要大约每个 bin 一步才准。</li>
</ul>"""
    ms_intro = (
        "Cycles 的“体积弹射次数”决定雾里的光最多被散射几次。Blender 默认为 0，也就是单次散射：雾中每一点只接收直接来自太阳和天空的光。"
        "真实的雾里，光会在雾滴之间反复散射，所以这里把弹射次数设到 16（8 次时已收敛），渲染出物理上完整的多次散射参考。"
        "闭式解和 ray marching 都只算单次散射：ray marching 是沿视线的数值积分方法，它的光源项和闭式解一样只包含直接光，所以只是闭式解的数值版本，而不是更完整的物理模型。"
        "下表以多次散射为真值；Cycles 的三行里，“单次散射”即 Blender 的默认设置。"
    )
    ms_text = (
        f"左图里，闭式解只改雾参数时比 Cycles 默认设置快 {cd['update'] / cf['update']:.1f} 倍，误差 {pct(cf['frames_rel_l1'])} 对 {pct(cd['frames_rel_l1'])}，还算一个可以讨论的折中。"
        "换成物理上完整的真值（右图）后，只算单次散射的方法（闭式解、ray marching、Cycles 的默认设置）都落到了虚线上方，比不加雾还差；只有打开多次散射的 Cycles 贴近噪声底。"
        f"1080p 下，闭式解每帧 {hd['closed_form']:.2f} s，打开多次散射的 Cycles 每帧 {hd['cycles_default_ms']:.2f} s（含 Blender 启动时间，10 帧平均），闭式解的优势只有 {speedup_hd:.1f} 倍；"
        f"它 {100 * sky_share:.0f}% 的时间花在天光的方向求积上，改成查找表后才能拉开差距。"
        f"多次散射也放大了 Cycles 的噪声：不降噪时渲染帧误差为 {pct(nd_ms['frames_rel_l1'])}，DVS 事件 F1 为 {nd_ms['dvs_f1_pooled']:.3f}，"
        f"都比单次散射时（{pct(nd['frames_rel_l1'])}、{nd['dvs_f1_pooled']:.3f}）更差，所以降噪器在这里必不可少。"
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
        "第 250 帧，以多次散射为真值。多次散射让远处的地面、方块和天空都更亮、更白（最右列）。只算单次散射的三列（Cycles 单次散射、ray marching、闭式解）"
        "在远处的地面和背景上误差都超过 50%；打开多次散射的 Cycles 默认设置几乎看不出误差。多次散射也让 Cycles 的蒙特卡洛噪声更大："
        "不降噪时 DVS 满屏都是假事件，Cycles 默认设置和参考在绿色方块表面也都有一些噪声事件。"
    )

    return {
        "findings": findings,
        "ms_intro": ms_intro,
        "ms_text": ms_text,
        "bounces_intro": bounces_intro,
        "bounces_text": bounces_text,
        "overview_ms_caption": overview_ms_caption,
        "tradeoff_caption": "每个点是一种方法。横轴是只改雾参数时生成一帧带雾画面的耗时（Cycles 要整帧重新渲染，闭式解和 ray marching 只需在无雾渲染上加雾），"
        "纵轴是渲染帧相对参考的 L1 误差，两轴都是对数坐标。左图以单次散射为真值，右图以多次散射为真值。虚线是完全不加雾的误差，点线是参考自身的噪声底。",
        "overview_caption": "第一行：各方法生成的带雾画面；第二行：相对参考的误差。闭式解和 ray marching 的误差集中在方块表面和方块周围的雾里，对应“差距从哪来”中的三种可见性效应。"
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
        "但它的每一步都要重新计算到达这一点的光：天空要对 1152 个方向求和，所以耗时和步数成正比。闭式解每个像素只需要对这些方向各算一次。"
        f"如果朝太阳方向也用 ray marching 来算衰减，误差会被这部分主导，停在 {sci(shadow[64]['error_vs_closed_form'])} 左右。"
        f"不管哪种设置，和参考之间的误差都在 {pct(convergence['closed_form_error_vs_reference'])} 左右：剩下的是模型误差，不是数值误差。",
        "breakdown_intro": "闭式解对它自己的模型是精确的：太阳和环境光项与逐点暴力积分吻合到 6 位有效数字，天光项的方向求积误差低于 0.1%。为了找出它和单次散射真值差在哪里，在第 255 帧（相机正从红色方块旁经过）用 Cycles 渲染了几种只保留部分光源、或去掉部分物体的变体，"
        "再把 Cycles 的 Volume Direct pass（雾对相机射线的单次散射）和闭式解的散射光逐项对比。比值为 1 表示完全一致。",
        "breakdown_text": f"只要物体不参与光的可见性计算，太阳项（{bd['sun_noshadow']['surfaces']:.3f}）和天光项（{bd['sky_noobjects']['surfaces']:.3f}）都和 Cycles 吻合到 1% 以内；天空像素的透射率也一致（比值 {bd['sky_transmittance']:.3f}）。"
        "剩下的差距全部来自三种和几何有关的可见性效应。它们都是真实存在的物理现象，当前模型都没有包含："
        f"① 物体在雾里投下的阴影（光柱）：物体投影后，表面像素上太阳项的比值从 {bd['sun_noshadow']['surfaces']:.3f} 升到 {bd['sun']['surfaces']:.3f}；"
        f"② 附近的物体挡住了雾中各点能看到的一部分天空：放回方块和柱子后，天光项的比值从 {bd['sky_noobjects']['surfaces']:.3f} 升到 {bd['sky']['surfaces']:.3f}；"
        f"③ 表面本身是透过雾被照亮的：阳光和天光到达表面之前都被雾衰减，雾的散射又补回一部分，净效果是表面在 Cycles 里暗了 {pct(1 - bd['surface_dimming']['median'])}（中位数，p10–p90 为 {pct(1 - bd['surface_dimming']['p90'])}–{pct(1 - bd['surface_dimming']['p10'])}）。"
        f"修正前的模型让下半球也向雾里照射天光，并且天光不衰减，结果表面像素上的散射光是 Cycles 的 {bd['full_before_fix']['surfaces']:.1f} 倍。",
        "next_steps": f"""<ul>
<li><strong>多次散射项（最优先）</strong>：给闭式解加一个多次散射光源。一种做法沿用 Hillaire（2020）在大气渲染中的思路：假设多次散射光近似各向同性，由二阶散射加几何级数求和得到每个高度上的多次散射光源，预计算成以高度和太阳高度角为变量的查找表；再把它随高度的变化拟合成几个指数项，沿视线的积分就仍有闭式解，结果依然确定、没有噪声，对所有传感器一致。另一种做法是用 Cycles 为每组雾参数渲染空场景，标定一个随视线仰角和距离变化的补偿系数，更准，但每换一组雾参数都要重新标定。这次的多次散射参考可以直接用来验证。</li>
<li><strong>性能</strong>：闭式解 {100 * sky_share:.0f}% 的时间花在天光的 1152 个求积方向上（没有这一项的修正前模型每帧只要 {fmt(m1["update"], "sec")}），而且这部分和像素数成正比。同一帧的所有射线起点相同，天光项只取决于射线方向的 v<sub>z</sub> 和射线长度，所以可以每帧预计算一张 (v<sub>z</sub>, 距离) 的二维查找表，这样开销就和分辨率基本无关。</li>
<li><strong>光柱和天光遮挡</strong>：从太阳方向渲染一张 shadow map，再从 Blender 输出一张天空可见性（环境光遮蔽）pass，用来调制源项。每条射线上被遮挡的区间，仍然可以用同一个闭式积分分段计算，结果依然确定、没有噪声。</li>
<li><strong>表面透过雾被照亮</strong>：用 Blender 的 Light Groups 把表面上的太阳贡献和天光贡献分开，分别乘以表面点处的阳光透射率和天光透射率（高度雾都有闭式解）。雾对表面的回照，可以用天光那套方向求积来计算。</li>
<li><strong>ToF 的后向散射</strong>：HG 相函数会低估雾滴在 180° 附近的后向散射（真实雾有 glory 峰），主动传感器可以改用 lidar ratio 来参数化。</li>
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
    "closed_form_m1": ("闭式解（修正前）", "#b9a3c9", "D", (7, 3, "left")),
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
    medium = summary["seconds"]["medium"]

    # Time to produce a frame with fog, the first time, and after changing only the fog's parameters. The closed form
    # and ray marching are timed in the first run, as they don't depend on the reference
    clear_render = renders["clear"]
    seconds = {"clear": (clear_render, None), "reference_seed1": (None, None)}  # not a method, the reference's noise
    for name in ("raymarch_16", "raymarch_64s16", "closed_form_m1", "closed_form"):
        seconds[name] = (clear_render + medium[name], medium[name])
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
            "“只改雾参数时”：Cycles 要整帧重新渲染，闭式解和 ray marching 只需在无雾渲染上加雾。耗时包含 Blender 启动时间，在 RTX 5080 上测得。"
            "ToF 不在表中：Cycles 不能渲染瞬态，瞬态的参考只能用同一个单次散射模型，无法评估多次散射对激光回波的影响。",
            labels=LABELS_MS,
        ),
        bounces_table=bounces_table(bounces),
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
            "耗时包含 Blender 启动时间，在 RTX 5080 上测得。",
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
        r=renders,
        **build_prose(rows, rows_ms, convergence, breakdown, renders, hd, bounces),
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
