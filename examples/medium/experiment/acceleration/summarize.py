"""Gather the results of the acceleration study into a report: ``runs/fog-comparison/acceleration/report.html``.

The report compares six directions of rendering fog faster, all at 320x180 against Cycles' references. It opens with
how the methods differ (which lighting model each uses, where it evaluates it, and how it turns that into pixels),
then their pros and cons, then tables of time, storage and error of the day scene, of the night scene's light
scattered once, and of all of its fog's light, then each method's error and its causes, as measured by the scripts in
``analysis/``, then the problems found in the closed form. Timings at 1920x1080 are kept apart, in an appendix.

Reads ``results/*.json`` written by the ``exp_*.py`` scripts, ``analysis/*/*.json`` written by the analysis scripts,
and the frame times in the logs of the renders on the CPU (``logs/``) and the GPU (``gpu/logs/``). The page is in
Chinese, as are the other reports of the fog experiments.
"""

from __future__ import annotations

import base64
import io
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from visionsim.utils.color import linearrgb_to_srgb

from common import DAY_FRAMES, NIGHT_FRAMES, OUT, ROOT, blur, day_scene, errors, flicker, load_images

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RESULTS = OUT / "results"
ANALYSIS = OUT / "analysis"


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_analysis(path: str) -> dict:
    return json.loads((ANALYSIS / path).read_text())


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def sec(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.1:
        ms = 1000 * value
        return f"{ms:.3f} ms" if ms < 1 else (f"{ms:.2f} ms" if ms < 10 else f"{ms:.0f} ms")
    return f"{value:.2f} s" if value < 10 else f"{value:.1f} s"


def size(value: float | None) -> str:
    """Bytes in decimal units."""
    if value is None:
        return "—"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value:.0f} B"


def frame_times(path) -> float | None:
    """Mean frame time printed by render_mc.py or render_eevee.py, leaving out the first frame, which warms up."""
    if not path.exists():
        return None
    times = [
        float(t) for _, t in re.findall(r"FRAME_TIME frame=(\d+) seconds=([\d.]+)", path.read_text(errors="replace"))
    ]
    return float(np.mean(times[1:])) if len(times) > 1 else None


def gpu_seconds(name: str) -> float | None:
    """Frame time of a Blender render on the RTX 5080, from gpu_rerun.ps1's logs."""
    return frame_times(OUT / "gpu" / "logs" / f"{name}.log")


def cpu_seconds(name: str) -> float | None:
    """Frame time of a Blender render on the CPU, from render_all.ps1's logs."""
    return frame_times(OUT / "logs" / f"{name}.log")


def variant(entries: list[dict], **match) -> dict:
    return next(e for e in entries if all(e.get(k) == v for k, v in match.items()))


# --------------------------------------------------------------------------------------------------------------------
# Data


class Study:
    """Every result the report needs."""

    def __init__(self):
        names = (
            "mc",
            "closed_form",
            "less",
            "cache",
            "froxel",
            "learning",
            "unreal_igpu",
            "unreal_rtx5080",
            "gpu",
            "sh",
        )
        self.mc, self.cf, self.less, self.cache, self.froxel, self.learning, self.ue, self.ue_gpu, self.gpu, self.sh = (
            load(n) for n in names
        )
        self.floor = reference_floor()
        # A reference of 2048 samples against the one of 4096: their difference has about sqrt(3) times the noise of the
        # latter
        self.night_floor = load_analysis("night/night.json")["reference_noise"][
            "second_render_2048spp_vs_reference_4096spp"
        ]["all"]
        self.night_ms = load_analysis("night_multiple/night_multiple.json")
        components = load_analysis("day_components/day_components.json")
        self.parts = components["parts"]
        self.parts_share = components["reference_share"]["single"] + components["reference_share"]["multiple"]
        self.azimuth = load_analysis("check_day_components/check_day_components.json")["D2"]["sky_by_azimuth"]
        self.eevee_ms = self.night_ms["eevee"]
        self.day_clamp = day_clamp()

    def froxel_entry(self, scene: str, tile: int, slices: int, jitter: int) -> dict:
        return variant(self.froxel["froxel"][scene], tile=tile, slices=slices, jitter=jitter)

    def ue_ms(self, level: str, res: str, quality: str, key: str = "GPUTime") -> float | None:
        value = self.ue_gpu.get("timings", {}).get(f"{level}_{res}_{quality}", {}).get(key)
        return None if value is None else value / 1000

    def ue_igpu(self, level: str, quality: str, key: str = "GPUTime") -> float | None:
        value = self.ue.get("timings", {}).get(f"{level}_320_{quality}", {}).get(key)
        return None if value is None else value / 1000


def reference_floor() -> dict:
    """Noise of the day reference: two references of different seeds differ by about sqrt(2) times the noise of
    each, i.e. its error against the converged image."""
    frames = tuple(range(255, 263))
    reference = load_images(ROOT / "ms" / "cycles_ref" / "frames", frames)
    seed1 = load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", frames)
    pair = (255, 345)
    still = errors(
        load_images(ROOT / "ms" / "cycles_ref_seed1" / "frames", pair),
        load_images(ROOT / "ms" / "cycles_ref" / "frames", pair),
    )
    return {
        "rel_l1": still["rel_l1"] / np.sqrt(2),
        "rel_l1_blurred": still["rel_l1_blurred"] / np.sqrt(2),
        "flicker": flicker(seed1, reference)["temporal"] / np.sqrt(2),
    }


def day_clamp() -> dict:
    """How much brighter the day reference is without Cycles' clamping of samples, from a render without it (and with
    32 volume bounces), as the ratio of their light."""
    pair = (255, 345)
    unclamped, reference = ROOT / "noclamp" / "ref32", ROOT / "ms" / "cycles_ref"
    return {
        part: float(load_images(unclamped / part, pair).sum() / load_images(reference / part, pair).sum())
        for part in ("frames", "volume/indirect")
    }


# --------------------------------------------------------------------------------------------------------------------
# Images and figures


def tonemap(linear: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    return (np.asarray(linearrgb_to_srgb(np.clip(linear * exposure, 0, 1))) * 255).astype(np.uint8)


def strip(images: list[np.ndarray], labels: list[str], columns: int, scale: int = 2) -> str:
    """Images side by side, scaled up, each with a label above it, as a PNG data URI."""
    h, w = images[0].shape[:2]
    h, w = h * scale, w * scale
    top, gap = 22, 6
    rows = -(-len(images) // columns)
    canvas = Image.new("RGB", (columns * (w + gap) - gap, rows * (h + top + gap) - gap), SURFACE)
    draw = ImageDraw.Draw(canvas)
    for k, (image, label) in enumerate(zip(images, labels)):
        x, y = (k % columns) * (w + gap), (k // columns) * (h + top + gap)
        canvas.paste(Image.fromarray(image).resize((w, h), Image.NEAREST), (x, y + top))
        draw.text((x + 2, y + 3), label, fill=SECONDARY, font=FONT)
    return png(canvas)


def png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def file_uri(path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def figure_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=144, facecolor=SURFACE)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def axes(ax, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=SECONDARY, fontsize=10)


def plot_series(ax, xs, ys, label, color, labels=None, line=True, offsets=None, marker="o"):
    """A series of 2px lines and ringed markers, with selective direct labels, each offset by (dx, dy) points."""
    style = {"color": color, "linewidth": 2 if line else 0, "solid_capstyle": "round", "solid_joinstyle": "round"}
    ax.plot(xs, ys, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, **style)
    for x, y, text in zip(xs, ys, labels or []):
        if text:
            dx, dy = (offsets or {}).get(text, (6, 4))
            ax.annotate(
                text,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                color=SECONDARY,
                ha="right" if dx < 0 else "left",
                va="top" if dy < 0 else "bottom",
            )


def percent_axis(axis, ticks):
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))


def time_axis(axis, ticks):
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{1000 * v:g} ms" if v < 0.1 else f"{v:g} s"))


def fig_time_error(S: Study) -> str:
    """Time on the RTX 5080 against error, of the methods that output the whole frame, by day and at night."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    xlabel = "每帧时间（RTX 5080，320×180，对数刻度）"
    for ax, scene in ((left, "day"), (right, "night")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        time_axis(ax.xaxis, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
        ax.set_xlim(0.0009, 2.2)
    axes(left, xlabel, "整幅图像与 Cycles 参考的相对 L1 误差")
    axes(right, xlabel, "雾光与 Cycles 单次散射参考的相对 L1 误差")
    percent_axis(left.yaxis, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    percent_axis(right.yaxis, [0.01, 0.02, 0.05, 0.1, 0.2])

    spp = (4, 16, 64, 256, 1024)
    times = [gpu_seconds(f"day_spp{s}") for s in spp]
    mcd = S.mc["day"]
    plot_series(
        left,
        times,
        [mcd[f"day_spp{s}"]["noisy"]["still"]["rel_l1"] for s in spp],
        "路径追踪（不降噪）",
        SERIES[0],
        ["4", "16", "64", "256", "1024 spp"],
        offsets={"4": (-6, 2), "16": (-6, 2)},
    )
    plot_series(
        left,
        times,
        [mcd[f"day_spp{s}"]["denoised"]["still"]["rel_l1"] for s in spp],
        "路径追踪 + OIDN 降噪",
        SERIES[1],
        ["4", "16", "64", "", ""],
        offsets={"4": (-6, -4), "16": (2, -9), "64": (2, -9)},
    )
    e = S.froxel["eevee"]
    points = [
        ("闭式解", S.gpu["cf_day_320_shadows_surfaces"], S.cf["day"]["still"]["rel_l1"], (6, 4)),
        ("EEVEE 默认", gpu_seconds("eevee_day_default"), e["eevee_day_default"]["whole"]["rel_l1"], (-8, -6)),
        ("EEVEE 1 样本", gpu_seconds("eevee_day_1sample"), e["eevee_day_1sample"]["whole"]["rel_l1"], (-6, 4)),
        ("Unreal Epic", S.ue_ms("Day", "320", "epic"), S.ue["day"]["epic"]["whole"]["rel_l1"], (6, -4)),
    ]
    for k, (text, x, y, offset) in enumerate(points):
        plot_series(left, [x], [y], None, SERIES[2 + k], [text], line=False, offsets={text: offset})
    left.axhline(S.floor["rel_l1"], color=MUTED, linewidth=1, linestyle="--")
    left.annotate(
        "参考自身噪声", (0.0011, S.floor["rel_l1"]), textcoords="offset points", xytext=(0, 3), fontsize=8, color=MUTED
    )
    left.set_ylim(0.01, 0.4)
    left.set_title("白天：整幅图像（多次散射）", color=INK, fontsize=11, loc="left")
    left.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="upper right")

    spp = (16, 64, 256, 1024)
    mcn = S.mc["night"]
    plot_series(
        right,
        [gpu_seconds(f"night_spp{s}") for s in spp],
        [mcn[f"night_spp{s}"]["fog_light"]["rel_l1"] for s in spp],
        "路径追踪（不降噪）",
        SERIES[0],
        ["16", "64", "256", "1024 spp"],
    )
    night = S.cache["night"]
    points = [
        ("闭式解", S.gpu["cf_night_320_shadows"], S.cf["night"]["fog_light"]["rel_l1"], (6, -4)),
        (
            "froxel 8 px×64 层",
            S.gpu["froxel_night_320_8_64_1"],
            S.froxel_entry("night", 8, 64, 1)["vs_cycles"]["rel_l1"],
            (-6, 4),
        ),
        (
            "灯网格 0.25 m",
            S.gpu["cache_night_lit_0.25_320"],
            variant(night["variants"], kind="lit", size=0.25)["vs_cycles"]["rel_l1"],
            (6, 4),
        ),
        ("神经场", S.gpu["field_night_320"], S.learning["field"][0]["vs_cycles"]["rel_l1"], (6, -2)),
        ("EEVEE 默认", gpu_seconds("eevee_night_default"), e["eevee_night_default"]["fog_light"]["rel_l1"], (6, 4)),
    ]
    for k, (text, x, y, offset) in enumerate(points):
        plot_series(right, [x], [y], None, SERIES[2 + k], [text], line=False, offsets={text: offset})
    right.axhline(S.night_floor["rel_l1"] / np.sqrt(3), color=MUTED, linewidth=1, linestyle="--")
    right.set_ylim(0.01, 0.4)
    right.set_title("夜间：雾散射的灯光（只比单次散射）", color=INK, fontsize=11, loc="left")
    right.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="upper right")
    fig.tight_layout()
    return figure_uri(fig)


def fig_own_error(S: Study) -> str:
    """Own errors of the methods that keep the closed form's lighting, against the closed form's own error."""
    day = S.cache["day"]["variants"]
    small = S.less["small"]["day"]["4"]
    reuse = S.less["reuse"]
    bars = [
        ("低分辨率 1/4 + 上采样平均源 J", small["guided_source"]["vs_closed_form"]["rel_l1"], SERIES[0]),
        ("低分辨率 1/4 + 深度引导上采样", small["guided"]["vs_closed_form"]["rel_l1"], SERIES[0]),
        ("低分辨率 1/4 + 双线性上采样", small["bilinear"]["vs_closed_form"]["rel_l1"], SERIES[0]),
        ("跨帧重投影，隔 4 帧", reuse["4"]["vs_closed_form"]["rel_l1"], SERIES[1]),
        ("跨帧重投影，隔 16 帧", reuse["16"]["vs_closed_form"]["rel_l1"], SERIES[1]),
        ("跨帧重投影，隔 32 帧", reuse["32"]["vs_closed_form"]["rel_l1"], SERIES[1]),
        ("网格缓存：太阳可见度 0.5 m", variant(day, kind="sun", size=0.5)["vs_closed_form"]["rel_l1"], SERIES[2]),
        ("网格缓存：太阳可见度 1 m", variant(day, kind="sun", size=1.0)["vs_closed_form"]["rel_l1"], SERIES[2]),
        (
            "网格缓存：天光 4 阶球谐 1 m",
            variant(day, kind="sh", size=1.0, order=4)["vs_closed_form"]["rel_l1"],
            SERIES[2],
        ),
        (
            "网格缓存：天光 2 阶球谐 1 m",
            variant(day, kind="sh", size=1.0, order=2)["vs_closed_form"]["rel_l1"],
            SERIES[2],
        ),
        ("froxel 原型 8 px × 64 层", S.froxel_entry("day", 8, 64, 1)["vs_closed_form"]["rel_l1"], SERIES[3]),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    axes(ax, "雾光的相对 L1 误差（分母：闭式解的雾光）", "")
    ys = np.arange(len(bars))[::-1]
    ax.barh(ys, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
    for y, (_, value, _) in zip(ys, bars):
        ax.annotate(
            pct(value, 2),
            (value, y),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    ax.set_yticks(ys, [b[0] for b in bars], fontsize=9, color=SECONDARY)
    model = S.cf["day"]["fog_light"]["rel_l1"]
    ax.axvline(model, color=SERIES[7], linewidth=1.5, linestyle="--")
    ax.annotate(
        f"闭式解自己的雾光\n与 Cycles 相差 {pct(model)}",
        (model, ys[-1]),
        textcoords="offset points",
        xytext=(-6, 0),
        ha="right",
        va="center",
        fontsize=8,
        color=SERIES[7],
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.set_xlim(0, model * 1.08)
    ax.set_title(
        "白天：沿用闭式解光照的方法，各自新增的误差（帧 255 + 345；重投影为单帧）", color=INK, fontsize=10.5, loc="left"
    )
    fig.tight_layout()
    return figure_uri(fig)


def fig_cf_parts(S: Study) -> str:
    """Error of the closed form by day, by the part of the light it comes from."""
    parts = S.parts["closed_form"]
    names = [
        ("surfaces", "表面（经雾衰减）"),
        ("single", "单次散射（太阳+天空）"),
        ("multiple", "多次散射 + 地面反射光"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 2.9), facecolor=SURFACE)
    axes(ax, "占整幅图像误差（分母：参考整图的光）", "")
    ys = np.arange(len(names))[::-1]
    raw = [parts[k]["share"] for k, _ in names]
    smooth = [parts[k]["share_blurred"] for k, _ in names]
    ax.barh(ys + 0.18, raw, height=0.34, color=SERIES[0], label="误差")
    ax.barh(ys - 0.18, smooth, height=0.34, color=SERIES[1], label="模糊后（≈ 偏差）")
    for y, (key, _) in zip(ys, names):
        p = parts[key]
        ax.annotate(
            f"{pct(p['share'], 2)} · 相对这部分光 {pct(p['local'])}，偏{'亮' if p['local_bias'] > 0 else '暗'} {pct(abs(p['local_bias']))}",
            (p["share"], y + 0.18),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    ax.axvline(S.floor["rel_l1_blurred"], color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(ys, [n for _, n in names], fontsize=9, color=SECONDARY)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.set_xlim(0, 0.2)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="lower right")
    ax.set_title(
        f"闭式解白天的误差按光的来源拆开（整图 {pct(parts['whole']['share'], 2)}；虚线 = 参考自身噪声，模糊后）",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    return figure_uri(fig)


def fig_azimuth(S: Study) -> str:
    """Ratio of the closed form's light to Cycles' in the sky, against the azimuth from the sun."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    axes(ax, "视线与太阳的方位角差（°）", "闭式解 ÷ Cycles")
    centers = {"20..35": 27.5, "35..50": 42.5, "50..65": 57.5, "65..80": 72.5, "80..100": 90}
    for (band, kind, label), color in zip(
        (
            ("sky_5..15deg", "multiple", "多次散射，仰角 5–15° 的天空"),
            ("sky_-2..5deg", "multiple", "多次散射，仰角 −2–5° 的天空"),
            ("sky_5..15deg", "single", "单次散射，仰角 5–15° 的天空"),
        ),
        (SERIES[1], SERIES[3], SERIES[0]),
    ):
        bins = S.azimuth[band]
        xs = [centers[b] for b in bins]
        ys = [bins[b][f"cf_{kind}"] / bins[b][f"reference_{kind}"] for b in bins]
        plot_series(ax, xs, ys, label, color)
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set_ylim(0.5, 1.4)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="lower right")
    ax.set_title(
        "同一仰角下，Cycles 的多次散射朝太阳一侧更亮；闭式解的表格与方位角无关", color=INK, fontsize=10.5, loc="left"
    )
    fig.tight_layout()
    return figure_uri(fig)


def luma(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64).mean(axis=-1)


def fig_night_single_maps(S: Study) -> str:
    """Where each method's light scattered once at night differs from Cycles', relative to the local light."""
    k = NIGHT_FRAMES.index(345)
    reference = load_images(ROOT / "point_shadow" / "bounces0" / "volume" / "direct", (345,))[0]
    cache, froxel = np.load(RESULTS / "cache_night_images.npz"), np.load(RESULTS / "froxel_images.npz")
    night = S.cache["night"]["variants"]
    methods = [
        (
            "路径追踪 64 spp",
            load_images(OUT / "night_spp64" / "volume" / "direct", (345,))[0],
            S.mc["night"]["night_spp64"]["fog_light"],
        ),
        ("闭式解", np.load(RESULTS / "closed_form_night.npy")[k], S.cf["night"]["fog_light"]),
        ("灯网格 1 m", cache["lit_1.0_0"], variant(night, kind="lit", size=1.0)["vs_cycles"]),
        ("灯网格 0.25 m", cache["lit_0.25_0"], variant(night, kind="lit", size=0.25)["vs_cycles"]),
        ("灯光 2 阶球谐 0.25 m", cache["sh_0.25_2"], variant(night, kind="sh", size=0.25, order=2)["vs_cycles"]),
        ("神经场 1.1 万参数", np.load(RESULTS / "learning_field_10945.npy"), S.learning["field"][0]["vs_cycles"]),
        ("froxel 8 px × 64 层", froxel["night_8_64_1"], S.froxel_entry("night", 8, 64, 1)["vs_cycles"]),
        ("EEVEE 默认", froxel["eevee_night"], S.froxel["eevee"]["eevee_night_default"]["fog_light"]),
    ]
    ref = blur(luma(reference)[None, ..., None])[0, ..., 0]
    floor = 0.005 * np.percentile(ref, 99)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d8d2")
    fig, grid = plt.subplots(2, 4, figsize=(13, 5.8), facecolor=SURFACE, layout="constrained")
    for ax, (name, image, e) in zip(grid.flat, methods):
        ours = blur(luma(image)[None, ..., None])[0, ..., 0]
        local = np.where(ref > floor, (ours - ref) / np.maximum(ref, floor), np.nan)
        im = ax.imshow(np.ma.masked_invalid(local), cmap=cmap, vmin=-0.5, vmax=0.5)
        ax.set_title(
            f"{name}\n误差 {pct(e['rel_l1'])} · 模糊后 {pct(e['rel_l1_blurred'])}", fontsize=9, color=INK, loc="left"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(im, ax=grid, orientation="horizontal", fraction=0.05, pad=0.04, aspect=60)
    bar.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}%"))
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.set_label(
        "（方法 − 参考）÷ 参考，各自 5×5 模糊后；红 = 偏亮，蓝 = 偏暗，灰 = 参考的单次散射光不到最亮处的 0.5%（立方体的影子、离灯很远处）",
        fontsize=9,
        color=SECONDARY,
    )
    return figure_uri(fig)


def fig_night_multiple(S: Study) -> str:
    """All of Cycles' fog light at night, split by where it comes from, against the closed form's."""
    d = np.load(ANALYSIS / "night_multiple" / "images.npz")
    direct, fog_only, reflected = luma(d["direct"]), luma(d["fog_only"]), np.clip(luma(d["reflected"]), 0, None)
    once, both, reference = luma(d["once"]), luma(d["both"]), luma(d["reference"])
    n = S.night_ms
    total = 1 + n["fog_only_over_direct"] + n["reflected_over_direct"]
    panels = [
        (f"Cycles 单次散射（{pct(1 / total, 0)}）", direct),
        (f"Cycles 雾内多次散射（{pct(n['fog_only_over_direct'] / total, 0)}）", fog_only),
        (f"Cycles 地面/物体反射进雾（{pct(n['reflected_over_direct'] / total, 0)}）", reflected),
        ("闭式解 单次散射", once),
        (f"闭式解 灯晕（= Cycles 雾内多次的 {n['halo_vs_fog_only']['ratio']:.1f} 倍）", both - once),
    ]
    vmax = 0.5 * np.percentile(reference, 99.5)
    fig, grid = plt.subplots(2, 3, figsize=(13, 5.4), facecolor=SURFACE, layout="constrained")
    for ax, (title, image) in zip(grid.flat, panels):
        ax.imshow(image, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax = grid.flat[-1]
    im = ax.imshow(
        np.log2((blur(both[None, ..., None]) + 1e-3) / (blur(reference[None, ..., None]) + 1e-3))[0, ..., 0],
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )
    ax.set_title(f"闭式解全部 ÷ Cycles 全部（误差 {pct(n['whole']['rel_l1'])}）", fontsize=9.5, color=INK, loc="left")
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    bar.set_ticks([-2, -1, 0, 1, 2], labels=["¼", "½", "1", "2", "4"])
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    for ax in grid.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    return figure_uri(fig)


def fig_cache(S: Study) -> str:
    """Own error of the night's lamp caches against their storage."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    axes(ax, "存储（float32，对数刻度）", "与闭式解的相对 L1 误差（对数刻度）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    percent_axis(ax.yaxis, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2**15, 2**18, 2**21, 2**24, 2**27]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: size(v)))
    variants = S.cache["night"]["variants"]
    for kind, label, color in (
        ("lit", "网格存 透射率×可见度（方向、1/r² 解析）", SERIES[0]),
        ("irradiance", "网格存 到达的光（含 1/r²）", SERIES[1]),
    ):
        entries = [e for e in variants if e["kind"] == kind]
        plot_series(
            ax,
            [e["bytes_fp32"] for e in entries],
            [e["vs_closed_form"]["rel_l1"] for e in entries],
            label,
            color,
            [f"{e['size']} m" for e in entries] if kind == "lit" else None,
        )
    entries = [e for e in variants if e["kind"] == "sh"]
    plot_series(
        ax,
        [e["bytes_fp32"] for e in entries],
        [e["vs_closed_form"]["rel_l1"] for e in entries],
        "网格存 球谐（0.25 m，各阶）",
        SERIES[2],
        [f"{e['order']} 阶" for e in entries],
    )
    fields = S.learning["field"]
    plot_series(
        ax,
        [e["bytes_fp32"] for e in fields],
        [e["vs_closed_form"]["rel_l1"] for e in fields],
        "神经场（MLP）",
        SERIES[3],
        [f"{e['parameters']:,} 参数" for e in fields],
        line=False,
        marker="D",
        offsets={f"{e['parameters']:,} 参数": (0, -9) for e in fields},
    )
    ax.set_title("夜间：灯光缓存的自身误差与存储", color=INK, fontsize=11, loc="left")
    fig.legend(
        *ax.get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=SECONDARY, loc="lower center", ncol=2
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return figure_uri(fig)


def strips(S: Study) -> dict[str, str]:
    """Side-by-side images of frame 345."""
    result = {}
    day = day_scene(frames=(345,))
    surfaces = day.surfaces(0)
    froxel, cache = np.load(RESULTS / "froxel_images.npz"), np.load(RESULTS / "cache_day_images.npz")
    images = [
        (day.reference[0], "参考：Cycles 4096 spp"),
        (load_images(OUT / "day_spp4" / "noisy", (345,))[0], "路径追踪 4 spp"),
        (load_images(OUT / "day_spp4" / "frames", (345,))[0], "路径追踪 4 spp + 降噪"),
        (np.load(RESULTS / "closed_form_day.npy")[DAY_FRAMES.index(345)], "闭式解"),
        (surfaces + cache["sun_1.0"], "网格缓存：太阳可见度 1 m"),
        (surfaces + cache["sh_1.0_2"], "网格缓存：天光 2 阶球谐 1 m"),
        (surfaces + froxel["day_8_64_1"], "froxel 原型 8 px × 64 层"),
        (froxel["eevee_day"], "EEVEE 默认"),
        (np.load(RESULTS / "unreal_day_epic.npy"), "Unreal 5.8 Epic"),
    ]
    result["day"] = strip([tonemap(x) for x, _ in images], [t for _, t in images], 3)

    k = NIGHT_FRAMES.index(345)
    single = load_images(ROOT / "point_shadow" / "bounces0" / "volume" / "direct", (345,))[0]
    exposure = 0.5 / np.percentile(single, 99)
    cache = np.load(RESULTS / "cache_night_images.npz")
    images = [
        (single, "参考：Cycles 单次散射"),
        (load_images(OUT / "night_spp16" / "volume" / "direct", (345,))[0], "路径追踪 16 spp"),
        (np.load(RESULTS / "closed_form_night.npy")[k], "闭式解"),
        (cache["lit_0.25_0"], "灯网格 0.25 m"),
        (cache["lit_1.0_0"], "灯网格 1 m"),
        (cache["sh_0.25_2"], "灯光 2 阶球谐 0.25 m"),
        (np.load(RESULTS / "learning_field_10945.npy"), "神经场 1.1 万参数"),
        (froxel["night_8_64_1"], "froxel 8 px × 64 层"),
    ]
    result["night"] = strip([tonemap(x, exposure) for x, _ in images], [t for _, t in images], 4)
    d = np.load(ANALYSIS / "night_multiple" / "images.npz")
    images = [
        (d["reference"], "参考：Cycles 全部雾光（16 次反弹，不截断）"),
        (d["both"], "闭式解 + 灯晕"),
        (froxel["eevee_night"], "EEVEE 默认"),
        (np.load(RESULTS / "unreal_night_epic.npy"), "Unreal 5.8 Epic"),
    ]
    result["night_all"] = strip([tonemap(x, exposure) for x, _ in images], [t for _, t in images], 4)

    saved = np.load(RESULTS / "less_hd_images.npz")
    crops = [saved["full"], saved["bilinear"], saved["guided_source"]]
    result["hd"] = strip(
        [tonemap(x, 2.0) for x in crops], ["全分辨率", "1/4 分辨率 + 双线性", "1/4 分辨率 + 上采样平均源 J"], 3, 1
    )
    ood = []
    for scene, reference in (("dense", OUT / "dense_ref" / "frames"), ("night", OUT / "night_ms_ref" / "frames")):
        ref = load_images(reference, (345,))[0]
        exposure = 1.0 if scene == "dense" else 0.5 / np.percentile(ref, 99)
        ood += [
            tonemap(x, exposure)
            for x in (
                ref,
                np.load(RESULTS / f"learning_image_{scene}.npy"),
                np.load(RESULTS / f"learning_residual_{scene}.npy"),
            )
        ]
    labels = []
    for scene in ("浓雾 ×2", "夜间点灯"):
        labels += [f"{scene}：Cycles", f"{scene}：图像网络", f"{scene}：残差网络"]
    result["ood"] = strip(ood, labels, 3)
    return result


# --------------------------------------------------------------------------------------------------------------------
# Tables


def table(headers: list[str], rows: list, text_columns: tuple[int, ...] = (), cls: str = "") -> str:
    """A table whose rows are lists of cells, or strings for a heading spanning all columns."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        if isinstance(row, str):
            body += f"<tr class='group'><td colspan='{len(headers)}'>{row}</td></tr>"
            continue
        cells = "".join(
            f"<td class='text'>{c}</td>" if i in text_columns else f"<td>{c}</td>" for i, c in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    return f"<div class='wrap'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def err(e: dict | None) -> list[str]:
    if not e:
        return ["—", "—", "—"]
    return [pct(e["rel_l1"]), pct(e["rel_l1_blurred"]), f"{e['ratio']:.3f}"]


def own(e: dict | None, digits: int = 2) -> str:
    return "—" if not e else pct(e["rel_l1"], digits)


def day_table(S: Study) -> str:
    mcd, cf, gpu = S.mc["day"], S.cf["day"], S.gpu
    rows = ["参考"]
    rows.append(
        [
            "Cycles 4096 spp 自身的噪声",
            "—",
            "—",
            "—",
            pct(S.floor["rel_l1"], 2),
            pct(S.floor["rel_l1_blurred"], 2),
            "1.000",
            "—",
            f"{S.floor['flicker']:.4f}",
            "低于这一行的差别测不出来",
        ]
    )

    def mc_row(name, label, kind, cause):
        e = mcd[name]
        return [
            label,
            sec(gpu_seconds(name)),
            sec(e["seconds"]),
            "—",
            *err(e[kind]["still"]),
            "—",
            f"{e[kind]['flicker']['temporal']:.4f}",
            cause,
        ]

    rows.append("① 路径追踪（Cycles）：与参考同一套物理，只是每像素路径少")
    rows += [
        mc_row("day_spp4", "4 spp + OIDN 降噪", "denoised", "降噪留下的低频斑块；整体亮 0.6%"),
        mc_row("day_spp16", "16 spp + OIDN 降噪", "denoised", "降噪留下的低频斑块（约 70% 随种子变）"),
        mc_row("day_spp64", "64 spp + OIDN 降噪", "denoised", "同上"),
        mc_row("day_spp1024", "1024 spp，不降噪", "noisy", "采样噪声（多次散射占方差的 3/4）"),
        mc_row("day_default", "VisionSIM 默认：自适应 ≤256 spp + 降噪", "denoised", "自适应采样整体暗 0.44%"),
    ]
    e = mcd["day_guided64"]
    rows.append(
        [
            "64 spp + 路径引导 + 降噪",
            "—（只能 CPU）",
            sec(e["seconds"]),
            "—",
            *err(e["denoised"]["still"]),
            "—",
            f"{e['denoised']['flicker']['temporal']:.4f}",
            "噪声帧间不再相关，不降噪时闪烁 2.8 倍",
        ]
    )

    rows.append("② 逐像素解析积分（闭式解，现方案）：分项光照模型，沿每条视线积分，无噪声")
    rows.append(
        [
            "完整（阴影 + 表面经雾照明）",
            sec(gpu["cf_day_320_shadows_surfaces"]),
            sec(cf["seconds"]["shadows_surfaces"]),
            f"阴影图 {size(cf['storage']['file_bytes'])}（内存 {size(cf['storage']['memory_bytes'])}）",
            *err(cf["still"]),
            "—",
            f"{cf['flicker']['temporal']:.4f}",
            "多次散射模型（§5.2）",
        ]
    )

    day = S.cache["day"]
    exact = day["exact"]

    def cache_row(label, e, gpu_time, cpu_exact, cause):
        return [
            label,
            gpu_time,
            f"{sec(e['seconds'])}（直接 {sec(cpu_exact)}）",
            f"{size(e['bytes'])}，预计算 {sec(e['build_seconds'])}",
            *err(e["vs_cycles"]),
            own(e["vs_closed_form"]),
            "—",
            cause,
        ]

    rows.append("③ 世界网格缓存：光照模型同 ②，把一部分光预先存在 3D 网格上，沿视线插值；时间只算被缓存的那部分")
    rows += [
        cache_row(
            "太阳可见度，0.5 m 网格",
            variant(day["variants"], kind="sun", size=0.5),
            f"{sec(gpu['cache_day_sun0.5_320'])}（直接 {sec(gpu['exact_day_sun_320'])}）",
            exact["sun_seconds"],
            "插值抹开阴影边界",
        ),
        cache_row(
            "太阳可见度，1 m 网格",
            variant(day["variants"], kind="sun", size=1.0),
            "—",
            exact["sun_seconds"],
            "同上；偏差符号随网格对齐变",
        ),
        cache_row(
            "天光+地面+多次散射，4 阶球谐，1 m",
            variant(day["variants"], kind="sh", size=1.0, order=4),
            f"{sec(gpu['cache_day_sh4_1m_320'])}（直接 {sec(gpu['exact_day_smooth_320'])}）",
            exact["smooth_seconds"],
            "球谐截断：地平线、物体遮挡边缘",
        ),
        cache_row(
            "同上，2 阶球谐",
            variant(day["variants"], kind="sh", size=1.0, order=2),
            "—",
            exact["smooth_seconds"],
            "同上，物体附近雾光高估 13–20%",
        ),
        cache_row(
            "天空各格可见度（每格 214 位），1 m",
            variant(day["variants"], kind="cells", size=1.0),
            "—",
            exact["smooth_seconds"],
            "最近邻取格",
        ),
    ]

    small = S.less["small"]["day"]["4"]
    low_seconds = sum(small["seconds"].values())
    rows.append("④ 少算：光照模型同 ②，少算一些像素、帧或阴影")
    for key, label, cause in (
        ("bilinear", "1/4 分辨率 + 双线性上采样", "轮廓两侧的雾光被平均"),
        ("guided", "1/4 分辨率 + 深度引导上采样", "地平线、地面上的横向台阶"),
        ("guided_source", "1/4 分辨率 + 上采样平均源 J", "天空遮挡项（8 个天空样本）"),
    ):
        rows.append(
            [
                label,
                "—",
                sec(low_seconds),
                "—",
                *err(small[key]["vs_cycles"]),
                own(small[key]["vs_closed_form"]),
                "—",
                cause,
            ]
        )
    for gap in ("4", "16"):
        e = S.less["reuse"][gap]
        rows.append(
            [
                f"跨帧重投影，隔 {gap} 帧 †",
                "—",
                sec(e["seconds"] + e["fill_seconds"]),
                "关键帧 J（0.7 MB）",
                *err(e["vs_cycles"]),
                own(e["vs_closed_form"]),
                "—",
                f"J 随视线方向和路径变；新露出 {pct(1 - e['seen'])} 像素（同帧闭式解 {pct(e['closed_form_vs_cycles']['rel_l1'])}）",
            ]
        )
    dens = S.less["density"]
    rows.append(
        [
            "共享阴影扫 4 种浓度（×0.5–×4）‡",
            "—",
            (f"4 种共 {sec(dens['trace_seconds'] + dens['shade_seconds'])}（分别算 {sec(dens['separate_seconds'])}）"),
            "—",
            "—",
            "—",
            "—",
            f"≤ {pct(max(e['rel_l1'] for e in dens['errors']), 2)}（整图）",
            "—",
            "8 个天空样本的位置随浓度变",
        ]
    )

    rows.append("⑤ 实时体积雾：视锥体素（froxel）上算光照、沿深度累积")
    f = S.froxel_entry("day", 8, 64, 1)
    rows.append(
        [
            "froxel 原型 8 px × 64 层（光照同 ②）",
            sec(gpu["froxel_day_320_8_64_1"]),
            sec(f["seconds"]),
            f"{f['froxels']:,} 体素（每帧临时）",
            *err(f["vs_cycles"]),
            own(f["vs_closed_form"]),
            "—",
            "深度层厚度；对 Cycles 同 ②",
        ]
    )
    e = S.froxel["eevee"]
    rows.append(
        [
            "EEVEE 默认（8 px × 64 层，16 样本）",
            sec(gpu_seconds("eevee_day_default")),
            f"集显 {sec(e['eevee_day_default']['seconds'])}",
            "—",
            *err(e["eevee_day_default"]["whole"]),
            "—",
            f"{e['eevee_day_default']['flicker']['temporal']:.4f}",
            "自己的光照：近地面雾太薄（雾光 −17%），表面 +8%",
        ]
    )
    rows.append(
        [
            "EEVEE 每帧 1 样本",
            sec(gpu_seconds("eevee_day_1sample")),
            f"集显 {sec(e['eevee_day_1sample']['seconds'])}",
            "—",
            *err(e["eevee_day_1sample"]["whole"]),
            "—",
            f"{e['eevee_day_1sample']['flicker']['temporal']:.4f}",
            "同上 + 混叠与抖动",
        ]
    )
    for quality, label in (("epic", "Unreal 5.8 Epic（8 px × 128 层）"), ("high", "Unreal 5.8 High（16 px × 64 层）")):
        u = S.ue["day"][quality]
        rows.append(
            [
                label,
                f"整帧 {sec(S.ue_ms('Day', '320', quality))}（雾 {sec(S.ue_ms('Day', '320', quality, 'GPU/VolumetricFog'))}）",
                f"集显整帧 {sec(S.ue_igpu('Day', quality))}",
                "—",
                *err(u["whole"]),
                "—",
                f"{u['flicker']['temporal']:.4f}",
                "自己的光照：雾光整体少 31%，离太阳越远缺得越多，推测缺体积多次散射",
            ]
        )

    images = S.learning["images"]
    rows.append("⑥ 学习（† 测试帧为 405–504，不是 255 + 345）")
    for kind, label, cause in (
        (
            "residual",
            "残差网络：U-Net 修正闭式解",
            f"同组帧上旧版闭式解 {pct(images['closed_form']['day_test']['rel_l1'])}",
        ),
        (
            "image",
            "图像网络：U-Net 从无雾图+深度直接出图",
            f"训练集上就有 {pct(images['image']['train']['rel_l1'])}（欠拟合）",
        ),
    ):
        e = images[kind]
        rows.append(
            [
                label + " †",
                f"{sec(gpu['unet_320'])} + 闭式解" if kind == "residual" else sec(gpu["unet_320"]),
                sec(e["day_test_seconds_per_frame"]),
                f"{e['parameters'] / 1e4:.0f} 万参数（{size(4 * e['parameters'])}），训练 {sec(e['train_seconds'])}",
                *err(e["day_test"]),
                "—",
                "—",
                cause,
            ]
        )
    return table(
        [
            "方法（设置）",
            "GPU 每帧",
            "CPU 每帧",
            "额外存储 / 预计算",
            "对 Cycles 误差",
            "模糊后",
            "亮度比",
            "自身误差（对闭式解）",
            "闪烁",
            "误差主要来自",
        ],
        rows,
        text_columns=(0, 3, 9),
        cls="main",
    )


def night_table(S: Study) -> str:
    mcn, cf, gpu, night = S.mc["night"], S.cf["night"], S.gpu, S.cache["night"]
    rows = ["参考"]
    rows.append(
        [
            "Cycles 4096 spp 自身的噪声",
            "—",
            "—",
            "—",
            pct(S.night_floor["rel_l1"] / np.sqrt(3), 2),
            pct(S.night_floor["rel_l1_blurred"] / np.sqrt(3), 2),
            "1.000",
            "—",
            "低于这一行的差别测不出来",
        ]
    )
    rows.append("① 路径追踪（雾光不降噪）")
    for spp in (16, 64, 256, 1024):
        e = mcn[f"night_spp{spp}"]
        rows.append(
            [
                f"{spp} spp",
                sec(gpu_seconds(f"night_spp{spp}")),
                sec(e["seconds"]),
                "—",
                *err(e["fog_light"]),
                "—",
                "采样噪声，集中在灯附近",
            ]
        )
    rows.append("② 逐像素解析积分（闭式解）")
    rows.append(
        [
            "灯：去奇点的数值求积 + 灯阴影图",
            sec(gpu["cf_night_320_shadows"]),
            sec(cf["seconds"]["shadows"]),
            f"阴影图 {size(cf['storage']['file_bytes'])}（内存 {size(cf['storage']['memory_bytes'])}）",
            *err(cf["fog_light"]),
            "—",
            "轮廓处像素滤波不同；灯周围的亮区亮 1.6%",
        ]
    )
    rows.append("③ 世界网格缓存（光照同 ②；对照：不缓存、同样沿视线 256 个节点求积）")
    e = night["exact"]
    rows.append(
        [
            "不缓存的逐节点求积（对照）",
            sec(gpu["exact_night_320"]),
            sec(e["seconds"]),
            "—",
            *err(e["vs_cycles"]),
            own(e["vs_closed_form"]),
            "求积节点",
        ]
    )
    for kind, size_, order, label, cause in (
        ("lit", 1.0, 0, "灯网格 1 m：存 透射率×可见度", "网格角点落在立方体阴影面上，可见度被插值成 0.5"),
        ("lit", 0.25, 0, "灯网格 0.25 m", "阴影面附近的插值"),
        ("lit", 0.125, 0, "灯网格 0.125 m", "同上"),
        ("irradiance", 0.25, 0, "灯网格 0.25 m：存 到达的光（含 1/r²）", "同上"),
        ("sh", 0.25, 2, "灯光 2 阶球谐 0.25 m", "相函数 g=0.8 太尖，球谐表示不了"),
        ("sh", 0.25, 8, "灯光 8 阶球谐 0.25 m", "同上"),
    ):
        v = variant(night["variants"], kind=kind, size=size_, order=order)
        gpu_time = (
            f"{sec(gpu['cache_night_lit_0.25_320'])}（直接 {sec(gpu['exact_night_320'])}）"
            if (kind, size_) == ("lit", 0.25)
            else "—"
        )
        rows.append(
            [
                label,
                gpu_time,
                sec(v["seconds"]),
                f"{size(v['bytes_fp32'])}，预计算 {sec(v['build_seconds'])}",
                *err(v["vs_cycles"]),
                own(v["vs_closed_form"]),
                cause,
            ]
        )
    field = S.learning["field"][0]
    rows.append(
        [
            f"神经场：{field['parameters']:,} 参数的 MLP 代替灯网格",
            f"{sec(gpu['field_night_320'])} + 训练 {sec(gpu['field_train_3000_steps'])}",
            f"{sec(field['seconds'])} + 训练 {sec(field['train_seconds'])}",
            size(field["bytes_fp32"]),
            *err(field["vs_cycles"]),
            own(field["vs_closed_form"]),
            "阴影边缘的拟合误差",
        ]
    )
    rows.append("④ 少算：1/4 分辨率（光照同 ②）")
    small = S.less["small"]["night"]["4"]
    for key, label, cause in (
        ("bilinear", "+ 双线性上采样", "灯周围的亮光被抹到立方体上"),
        ("guided", "+ 深度引导上采样", "立方体边缘"),
        ("guided_source", "+ 上采样平均源 J", "J 在夜间不比雾光平滑"),
    ):
        rows.append(
            [
                label,
                "—",
                sec(sum(small["seconds"].values())),
                "—",
                *err(small[key]["vs_cycles"]),
                own(small[key]["vs_closed_form"]),
                cause,
            ]
        )
    rows.append("⑤ 实时体积雾")
    for tile, slices, jitter, cause in (
        (8, 64, 1, "深度层太厚（灯附近 0.38 m）"),
        (8, 64, 8, "同上，抖动 8 帧平均"),
        (4, 128, 4, "同上"),
    ):
        f = S.froxel_entry("night", tile, slices, jitter)
        key = f"froxel_night_320_{tile}_{slices}_{jitter}"
        rows.append(
            [
                f"froxel 原型 {tile} px × {slices} 层" + (f"，抖动 {jitter} 帧" if jitter > 1 else ""),
                sec(gpu.get(key)),
                sec(f["seconds"]),
                f"{f['froxels']:,} 体素",
                *err(f["vs_cycles"]),
                own(f["vs_closed_form"]),
                cause,
            ]
        )
    e = S.froxel["eevee"]
    rows.append(
        [
            "EEVEE 默认（8 px × 64 层）",
            sec(gpu_seconds("eevee_night_default")),
            f"集显 {sec(e['eevee_night_default']['seconds'])}",
            "—",
            *err(e["eevee_night_default"]["fog_light"]),
            "—",
            "离地 0.4 m 内的雾收不到灯光",
        ]
    )
    rows.append(
        [
            "EEVEE 精细（2 px × 256 层到 300 m）",
            "—",
            f"集显 {sec(e['eevee_night_fine']['seconds'])}",
            "—",
            *err(e["eevee_night_fine"]["fog_light"]),
            "—",
            "同上，降到离地 0.15 m",
        ]
    )
    return table(
        [
            "方法（设置）",
            "GPU 每帧",
            "CPU 每帧",
            "额外存储 / 预计算",
            "对 Cycles 单次散射",
            "模糊后",
            "亮度比",
            "自身误差（对闭式解）",
            "误差主要来自",
        ],
        rows,
        text_columns=(0, 3, 8),
        cls="main",
    )


def night_all_table(S: Study) -> str:
    n, image, learning = S.night_ms, S.night_ms["image"], S.learning["images"]
    rows = [
        [
            "闭式解，只有单次散射",
            "单次",
            *err(image["single_only"]),
            *err(n["single_only_vs_whole"]),
            "缺雾内多次散射和表面反射光",
        ],
        [
            "闭式解 + 灯晕（多次散射开）",
            "单次 + 雾内多次（灯晕）",
            *err(image["with_halo"]),
            *err(n["whole"]),
            (
                f"灯晕总量是 Cycles 雾内多次散射的 {n['halo_vs_fog_only']['ratio']:.1f} 倍，但 {pct(n['halo_split']['all']['from_shadowed_points'], 0)} 落在立方体的灯影里；"
                "表面反射进雾的光不建模"
            ),
        ],
    ]
    for name, label, note in (
        ("eevee_night_default", "EEVEE 默认", "无多次散射；近地面雾收不到灯光"),
        ("eevee_night_fine", "EEVEE 精细", "单次散射更准，整图反而略差"),
    ):
        e = S.eevee_ms[name]
        rows.append([label, "单次（近似）", *err(e["whole"]), *err(e["fog_light"]), note])
    for quality in ("epic", "high", "cinematic"):
        rows.append(
            [
                f"Unreal 5.8 {quality.capitalize()}（Lumen 开）",
                "单次 + Lumen 反射（推测，没关 Lumen 验证）",
                *err(n["unreal"][quality]),
                "—",
                "—",
                "—",
                (
                    "雾光分不干净：表面上的雾光要用“有雾帧 − 透射率 × 无雾帧”估计，这个估计本身有偏"
                    f"（对截断参考估计为 {pct(S.ue['night']['epic']['fog_light_vs_multiple']['rel_l1'])}）；"
                    f"天空像素上是 Cycles 的 {n['unreal_fog_ratio']['sky']:.1f} 倍（对截断参考 {n['clamped']['unreal_fog_ratio']['sky']:.1f} 倍）"
                )
                if quality == "epic"
                else "同上",
            ]
        )
    for kind, label in (("residual", "残差网络 †"), ("image", "图像网络 †")):
        rows.append(
            [
                label,
                "（学到的）",
                *err(learning[kind]["night"]),
                "—",
                "—",
                "—",
                "† 对截断的参考（预测只存了第 345 帧，没法换参考重算）；只用白天训练",
            ]
        )
    return table(
        ["方法", "包含的光", "整幅图误差", "模糊后", "亮度比", "雾光误差", "模糊后", "亮度比", "说明"],
        rows,
        text_columns=(0, 1, 8),
        cls="main",
    )


def scope_table(S: Study) -> str:
    """The closed form's errors by day and at night, measured the same ways, against references with all the light."""
    parts, n, image, clamp = S.parts["closed_form"], S.night_ms, S.night_ms["image"], S.night_ms["clamp"]
    old, old_image = n["clamped"], n["clamped"]["image"]
    day_surfaces_ratio = 1 + parts["surfaces"]["local_bias"]

    def cell(e: dict) -> str:
        return f"<b>{pct(e['rel_l1'])}</b>（模糊后 {pct(e['rel_l1_blurred'])}），亮度比 {e['ratio']:.2f}"

    rows = [
        [
            "参考",
            "Cycles 4096 spp，16 次体积反弹（默认截断，对白天几乎没有影响，见下）",
            "Cycles 2048 spp，16 次体积反弹，不截断",
        ],
        ["雾光占整幅图", pct(S.parts_share, 0), pct(image["fog_share"], 0)],
        ["<b>整幅图</b>", cell(S.cf["day"]["still"]), cell(image["with_halo"])],
        ["<b>只看雾光</b>", cell(S.cf["day"]["fog_light"]), cell(n["whole"])],
        [
            "只看表面",
            f"{pct(parts['surfaces']['local'])}（模糊后 {pct(parts['surfaces']['local_blurred'])}），亮度比 {day_surfaces_ratio:.2f}",
            f"{pct(image['surfaces']['rel_l1'])}（模糊后 {pct(image['surfaces']['rel_l1_blurred'])}），亮度比 {image['surfaces']['ratio']:.2f}",
        ],
        [
            "雾光 / 表面各自的误差（÷ 整幅图参考）",
            f"{100 * parts['fog']['share']:.1f} / {100 * parts['surfaces']['share']:.1f} 个百分点",
            f"{100 * image['share']['fog']:.1f} / {100 * image['share']['surfaces']:.1f} 个百分点",
        ],
        [
            "误差的性质",
            "太阳、天空、地面反射、多次散射都有对应项，但近似不准（多次散射表的三个问题）；物体反射进雾的光没建模，白天占多少没测",
            (
                f"整类光没建模：表面反射进雾的灯光占雾光 {pct(n['reflected_over_direct'] / (1 + n['fog_only_over_direct'] + n['reflected_over_direct']), 0)}；"
                f"灯晕总量是 Cycles 雾内多次散射的 {n['halo_vs_fog_only']['ratio']:.1f} 倍，但 {pct(n['halo_split']['all']['from_shadowed_points'], 0)} "
                "落在立方体的灯影里；总量碰巧相抵，分布完全不对"
            ),
        ],
        [
            "参考的截断（clamp = 10）",
            (
                f"测过：不截断的白天渲染（32 次反弹）整图只亮 {pct(S.day_clamp['frames'] - 1, 2)}，"
                f"间接光亮 {pct(S.day_clamp['volume/indirect'] - 1, 2)}，可以忽略"
            ),
            (
                f"影响大：不截断时间接光亮 {pct(clamp['indirect'] - 1, 0)}（雾内多次散射亮 {pct(clamp['fog_only'] - 1, 0)}，"
                f"表面反射进雾的光亮 {pct(clamp['reflected'] - 1, 0)}），全部雾光亮 {pct(clamp['fog'] - 1, 0)}，表面亮 {pct(clamp['surfaces'] - 1, 0)}，"
                f"整幅图亮 {pct(clamp['frames'] - 1, 0)}。"
                f"所以夜间用不截断的参考；对截断的参考，闭式解整幅图 {pct(old_image['with_halo']['rel_l1'])}，雾光 {pct(old['whole']['rel_l1'])}"
            ),
        ],
        [
            "单次散射子问题（参考也只有单次散射）",
            "—",
            (
                f"雾光 {pct(S.cf['night']['fog_light']['rel_l1'])}（模糊后 {pct(S.cf['night']['fog_light']['rel_l1_blurred'])}）；"
                f"整幅图 {pct(S.cf['night']['whole']['rel_l1'])}（同一子问题里路径追踪 16 spp + 降噪 "
                f"{pct(S.mc['night']['night_spp16']['whole_denoised']['rel_l1'])}）。只说明单次散射的积分算对了"
            ),
        ],
    ]
    return table(["口径", "白天", "夜间"], rows, text_columns=(0, 1, 2), cls="main")


def overview_table(S: Study) -> str:
    """Pros and cons of each direction, at its most useful setting."""
    gpu, cf, mcd = S.gpu, S.cf, S.mc["day"]
    small_day, small_night = S.less["small"]["day"]["4"], S.less["small"]["night"]["4"]
    day = S.cache["day"]["variants"]
    night = S.cache["night"]["variants"]
    f_day, f_night = S.froxel_entry("day", 8, 64, 1), S.froxel_entry("night", 8, 64, 1)
    f_jitter = S.froxel_entry("night", 8, 64, 8)
    e = S.froxel["eevee"]
    n, image = S.night_ms, S.night_ms["image"]
    field = S.learning["field"][0]
    reuse = S.less["reuse"]

    def both(entry: dict, digits: int = 2) -> str:
        """Error against Cycles, and own error against the closed form in brackets."""
        return f"{pct(entry['vs_cycles']['rel_l1'])}（自身 {pct(entry['vs_closed_form']['rel_l1'], digits)}）"

    # The methods that were only tested with the closed form's light scattered once
    untested = (
        f"未测（只实现了单次散射；估计 ≈ 闭式解只单次 {pct(image['single_only']['rel_l1'])} / "
        f"{pct(n['single_only_vs_whole']['rel_l1'])}，加上闭式解的灯晕约 {pct(image['with_halo']['rel_l1'])} / "
        f"{pct(n['whole']['rel_l1'])}）"
    )
    sun, sh4 = variant(day, kind="sun", size=0.5), variant(day, kind="sh", size=1.0, order=4)
    lit, lit1 = variant(night, kind="lit", size=0.25), variant(night, kind="lit", size=1.0)
    rows = [
        [
            "① 路径追踪",
            "16 spp + OIDN",
            sec(gpu_seconds("day_spp16")),
            "无",
            pct(mcd["day_spp16"]["denoised"]["still"]["rel_l1"]),
            "—（与参考同一物理，误差只有噪声；没渲低样本数的完整版）",
            f"{pct(S.mc['night']['night_spp64']['fog_light']['rel_l1'])}（64 spp，雾光不降噪）",
            "物理完整：多次散射、表面反射全都有；误差可用样本数控制；没有要维护的模型",
            "有噪声，要降噪；降噪留下低频斑块（模糊后仍 1.4%）；夜间灯附近要很多样本；320×180 时 GPU 时间大半是 Blender 每帧开销",
        ],
        [
            "② 逐像素解析（闭式解）",
            "完整",
            f"{sec(gpu['cf_day_320_shadows_surfaces'])}（不含表面模型 {sec(gpu['cf_day_320_shadows'])}）",
            f"阴影图 {size(cf['day']['storage']['file_bytes'])}",
            pct(cf["day"]["still"]["rel_l1"]),
            f"<b>{pct(image['with_halo']['rel_l1'])} / {pct(n['whole']['rel_l1'])}</b>（开灯晕）",
            pct(cf["night"]["fog_light"]["rel_l1"]),
            "无噪声、确定性、闪烁最小；夜间单次散射的雾光算得准",
            "精度由光照模型决定，加样本也改不了：白天多次散射这一项有问题（含一个 bug），夜间灯晕大半落在立方体灯影里、表面反射光没建模（§6）",
        ],
        [
            "③ 世界网格缓存",
            "太阳 0.5 m / 天光 4 阶球谐 1 m / 灯 0.25 m",
            f"天光部分 {sec(gpu['exact_day_smooth_320'])} → {sec(gpu['cache_day_sh4_1m_320'])}；太阳、灯反而变慢",
            f"{size(sun['bytes'])} / {size(sh4['bytes'])} / {size(lit['bytes_fp32'])}",
            f"{both(sun)} / {both(sh4)}",
            untested,
            f"{both(lit)}；1 m 网格 {both(lit1, 1)}",
            "与相机无关，所有帧、所有相机共用；昂贵的天光部分快 6 倍",
            "插值抹开阴影边界，误差符号随网格对齐变；球谐表示不了尖的相函数；存储随分辨率立方增长",
        ],
        [
            "③ 神经场",
            f"{field['parameters']:,} 参数",
            f"夜间灯光查询 {sec(gpu['field_night_320'])}（直接算 {sec(gpu['exact_night_320'])}）+ 训练 {sec(gpu['field_train_3000_steps'])}；白天未测",
            size(field["bytes_fp32"]),
            "—（白天没测）",
            untested,
            both(field),
            "存储是同精度网格的 1/50",
            f"训练和查询都比直接算慢（查询慢 {gpu['field_night_320'] / gpu['exact_night_320']:.1f} 倍）",
        ],
        [
            "④ 低分辨率 + 上采样",
            "1/4 分辨率 + 上采样平均源 J",
            f"320×180 未测 GPU；1080p {sec(gpu['cf_day_1080_shadows'])} → {sec(gpu['cf_day_1080_low4'])}",
            "无",
            f"{pct(small_day['guided_source']['vs_cycles']['rel_l1'])}（自身 {pct(small_day['guided_source']['vs_closed_form']['rel_l1'], 2)}）",
            untested,
            f"{pct(small_night['guided_source']['vs_cycles']['rel_l1'])}（自身 {pct(small_night['guided_source']['vs_closed_form']['rel_l1'], 2)}）",
            "几乎无损、实现简单，可叠加在任何逐像素方法上",
            "320×180 下白天只快 2 倍（阴影追踪等固定开销不随像素减少）",
        ],
        [
            "④ 跨帧重投影",
            "隔 4 帧",
            "未测 GPU（CPU 补洞占大头）",
            "关键帧 J 图",
            (
                f"{pct(reuse['4']['vs_cycles']['rel_l1'])}（自身 {pct(reuse['4']['vs_closed_form']['rel_l1'], 2)}）†；"
                f"隔 16 帧 {pct(reuse['16']['vs_cycles']['rel_l1'])}（自身 {pct(reuse['16']['vs_closed_form']['rel_l1'], 2)}）†"
            ),
            "未测",
            "未测",
            "复用整帧的雾",
            "误差随相机位移增长（J 依赖整条视线，不只是表面点）；新露出的像素要补",
        ],
        [
            "④ 共享阴影扫浓度",
            "4 种浓度",
            "CPU 5.8 s → 3.0 s",
            "无",
            f"自身 ≤ {pct(max(x['rel_l1'] for x in S.less['density']['errors']), 2)}（整幅图为分母，只有第 345 帧）",
            "未测",
            "未测",
            "多浓度渲染几乎白送；精度不降",
            "只对“同一帧多种雾”有用",
        ],
        [
            "⑤ froxel 原型",
            "8 px × 64 层",
            sec(gpu["froxel_day_320_8_64_1"]),
            f"{f_day['froxels']:,} 体素",
            both(f_day),
            untested,
            f"{both(f_night)}；抖动 8 帧 {both(f_jitter)}",
            "与引擎同构，开销与图像复杂度无关",
            "原型没比闭式解快（每个体素仍查 107 个天空格）；灯附近要细的深度层或抖动",
        ],
        [
            "⑤ EEVEE",
            "默认",
            sec(gpu_seconds("eevee_day_default")),
            "无",
            pct(e["eevee_day_default"]["whole"]["rel_l1"]),
            f"{pct(S.eevee_ms['eevee_night_default']['whole']['rel_l1'])} / {pct(S.eevee_ms['eevee_night_default']['fog_light']['rel_l1'])}",
            pct(e["eevee_night_default"]["fog_light"]["rel_l1"]),
            "现成，Blender 里直接用",
            "近地面的雾变薄、收不到灯光；没有多次散射；每帧 1 样本时闪烁是闭式解的 3.5 倍",
        ],
        [
            "⑤ Unreal 5.8",
            "Epic",
            f"整帧 {sec(S.ue_ms('Day', '320', 'epic'))}（雾 {sec(S.ue_ms('Day', '320', 'epic', 'GPU/VolumetricFog'))}）",
            "无",
            pct(S.ue["day"]["epic"]["whole"]["rel_l1"]),
            f"{pct(n['unreal']['epic']['rel_l1'])} / —（雾光分不干净）",
            "不可比（Unreal 不能只渲单次散射；Lumen 开着，可能含反射光，没验证）",
            "快三个数量级；白天画质档位几乎不影响误差",
            "白天雾光整体少 31%，离太阳越远缺得越多（55° 以外 −36% 到 −42%），推测缺体积多次散射；夜间灯光被近乎各向同性散射；参数要从源码换算",
        ],
        [
            "⑥ 学习",
            "残差 / 图像 U-Net",
            f"推理 {sec(gpu['unet_320'])}",
            f"{S.learning['images']['image']['parameters'] / 1e4:.0f} 万参数",
            f"{pct(S.learning['images']['residual']['day_test']['rel_l1'])} / {pct(S.learning['images']['image']['day_test']['rel_l1'])}（其他帧）",
            (
                f"残差 {pct(S.learning['images']['residual']['night']['rel_l1'])} / 图像 "
                f"{pct(S.learning['images']['image']['night']['rel_l1'], 0)}（都是整幅图，对截断参考；雾光没测）"
            ),
            "—",
            "推理最快",
            "离开训练分布就失败（夜间、浓雾）；需要大量参考图",
        ],
    ]
    return table(
        [
            "方向",
            "代表设置",
            "GPU 每帧（白天）",
            "存储",
            "白天整幅图对 Cycles（括号内：自身误差）",
            "夜间对完整参考（不截断）：整幅图 / 雾光",
            "夜间单次散射子问题对 Cycles，只比雾光（括号内：自身误差）",
            "优点",
            "缺点",
        ],
        rows,
        text_columns=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        cls="overview",
    )


def appendix_tables(S: Study) -> str:
    gpu = S.gpu
    parts = [
        "<h3>A1. 1920×1080 的时间（RTX 5080）</h3>",
        "<p class='note'>只测了时间。1080p 没有 Cycles 参考，所以没有对 Cycles 的误差。</p>",
    ]
    rows = [
        [
            "路径追踪 16 / 64 / 256 spp + 降噪",
            " / ".join(sec(gpu_seconds(f"hd_spp{s}")) for s in (16, 64, 256)),
            " / ".join(sec(cpu_seconds(f"hd_spp{s}")) for s in (16, 64, 256)),
        ],
        ["路径追踪 VisionSIM 默认", sec(gpu_seconds("hd_default")), "—"],
        [
            "闭式解 无阴影 / 带阴影（不含表面模型）",
            f"{sec(gpu['cf_day_1080_no_shadows'])} / {sec(gpu['cf_day_1080_shadows'])}",
            f"{sec(S.cf['day_1080p']['seconds']['no_shadows'])} / {sec(S.cf['day_1080p']['seconds']['shadows'])}",
        ],
        [
            "闭式解 1/2、1/4、1/8 分辨率 + 上采样",
            " / ".join(sec(gpu[f"cf_day_1080_low{s}"]) for s in (2, 4, 8)),
            " / ".join(sec(sum(S.less["hd"][str(s)]["seconds"].values())) for s in (2, 4, 8)),
        ],
        ["froxel 原型 8 px × 64 层", sec(gpu["froxel_day_1080_8_64_1"]), "—"],
        ["EEVEE 每帧 1 样本", sec(gpu_seconds("eevee_hd_1sample")), "—"],
        [
            "Unreal 5.8 Epic 整帧（其中体积雾）",
            f"{sec(S.ue_ms('Day', '1920', 'epic'))}（{sec(S.ue_ms('Day', '1920', 'epic', 'GPU/VolumetricFog'))}）",
            "—",
        ],
        ["U-Net 推理", sec(gpu["unet_1920"]), "—"],
    ]
    parts.append(table(["方法", "GPU", "CPU 12 线程"], rows, text_columns=(0,)))
    rows = []
    for factor, e in S.less["hd"].items():
        rows.append(
            [f"1/{factor}", *(pct(e[m]["inscatter"]["rel_l1"], 2) for m in ("bilinear", "guided", "guided_source"))]
        )
    parts.append("<h3>A2. 1080p 下降分辨率的自身误差（只有第 255 帧，只和全分辨率闭式解比雾光）</h3>")
    parts.append(
        "<p class='note'>同样 1/4，1080p 的误差约为 320×180 的 1/3：一个 4×4 像素块在 1080p 覆盖的场景小 6 倍，跨轮廓的块少得多。"
        "所以这些数字不能和正文 320×180 的表放在一起比。</p>"
    )
    parts.append(table(["分辨率", "双线性", "深度引导", "上采样平均源 J"], rows))
    for letter, tag, data in (("a", "RTX 5080", S.ue_gpu), ("b", "AMD 集成显卡", S.ue)):
        timings = data.get("timings", {})
        rows = []
        for level in ("Day", "Night"):
            for res in ("320", "1920"):
                base = timings.get(f"{level}_{res}_nofog", {})
                for quality in ("high", "epic", "cinematic"):
                    t = timings.get(f"{level}_{res}_{quality}", {})
                    if not t:
                        continue
                    # The profiler sometimes reports no GPU time
                    measured = t["GPUTime"] > 0 and base.get("GPUTime", 0) > 0
                    rows.append(
                        [
                            f"{'白天' if level == 'Day' else '夜间'} {res} 宽 {quality}",
                            f"{t['FrameTime']:.2f} ms",
                            f"{t['GPUTime']:.2f} ms" if t["GPUTime"] > 0 else "—",
                            f"{t['GPUTime'] - base['GPUTime']:+.2f} ms" if measured else "—",
                            f"{t.get('GPU/VolumetricFog', np.nan):.3f} ms",
                        ]
                    )
        parts.append(f"<h3>A3{letter}. Unreal 实时帧时间（{tag}，游戏模式，CSV 统计，600 帧中位数）</h3>")
        if letter == "b":
            parts.append(
                "<p class='note'>集显的整帧时间波动比体积雾本身还大，“比无雾多”可能为负；以体积雾 GPU 时间为准。</p>"
            )
        parts.append(table(["设置", "整帧时间", "GPU 时间", "比无雾多", "体积雾 GPU 时间"], rows))
    rows = []
    for quality in ("high", "epic", "cinematic"):
        d, n = S.ue["day"][quality], S.ue["night"][quality]
        rows.append(
            [
                quality,
                pct(d["whole"]["rel_l1"]),
                f"{d['fog_light_vs_multiple']['ratio']:.2f}",
                f"{d['fog_light_vs_single']['ratio']:.2f}",
                f"{d['flicker']['temporal']:.4f}",
                pct(S.night_ms["unreal"][quality]["rel_l1"]),
                pct(n["whole_vs_multiple"]["rel_l1"]),
            ]
        )
    parts.append("<h3>A4. Unreal 三档画质的误差</h3>")
    parts.append(
        table(
            [
                "画质",
                "白天整图误差",
                "白天雾光 ÷ Cycles 全部雾光",
                "÷ Cycles 单次散射",
                "闪烁",
                "夜间整图（对完整参考，不截断）",
                "（对截断的参考）",
            ],
            rows,
        )
    )
    rows = []
    for name, label in (
        [(f"day_spp{s}", f"{s} spp") for s in (4, 16, 64, 256, 1024)]
        + [(f"day_guided{s}", f"{s} spp + 路径引导") for s in (16, 64, 256)]
        + [("day_default", "VisionSIM 默认")]
    ):
        e = S.mc["day"][name]
        rows.append(
            [
                label,
                sec(gpu_seconds(name)),
                sec(e["seconds"]),
                pct(e["noisy"]["still"]["rel_l1"]),
                pct(e["noisy"]["still"]["rel_l1_blurred"]),
                pct(e["denoised"]["still"]["rel_l1"]),
                pct(e["denoised"]["still"]["rel_l1_blurred"]),
                f"{e['noisy']['flicker']['temporal']:.4f}",
                f"{e['denoised']['flicker']['temporal']:.4f}",
                pct(e["hybrid"]["still"]["rel_l1"]),
            ]
        )
    parts.append("<h3>A5. 路径追踪全部设置（白天，320×180）</h3>")
    parts.append(
        table(
            [
                "设置",
                "GPU",
                "CPU",
                "不降噪",
                "模糊后",
                "降噪",
                "模糊后",
                "闪烁（不降噪）",
                "闪烁（降噪）",
                "混合：闭式解单次 + MC 其余",
            ],
            rows,
        )
    )
    rows = []
    for scene in ("day", "night"):
        for f in S.froxel["froxel"][scene]:
            rows.append(
                [
                    f"{'白天' if scene == 'day' else '夜间'} {f['tile']} px × {f['slices']} 层，抖动 {f['jitter']}",
                    f"{f['froxels']:,}",
                    sec(S.gpu.get(f"froxel_{scene}_320_{f['tile']}_{f['slices']}_{f['jitter']}")),
                    sec(f["seconds"]),
                    own(f["vs_closed_form"]),
                    pct(f["vs_cycles"]["rel_l1"]),
                ]
            )
    parts.append("<h3>A6. froxel 原型全部设置</h3>")
    parts.append(
        table(
            ["设置", "体素数", "GPU", "CPU", "自身误差", "对 Cycles（白天：整图对完整参考；夜间：雾光对单次散射参考）"],
            rows,
        )
    )
    rows = []
    for order, e in S.sh.items():
        rows.append([f"{order} 阶", *(pct(e[k]) for k in ("sun", "sky", "ground", "scattered"))])
    parts.append("<h3>A7. 白天各光源单独存成球谐的截断误差（与精确值的相对 L1）</h3>")
    parts.append(table(["阶数", "太阳", "天空", "地面", "多次散射"], rows))
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------------------------------
# Page


STYLE = """
<style>
:root { --surface:#fcfcfb; --page:#f6f6f3; --ink:#0b0b0b; --secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
        --accent:#2a78d6; --warn:#b4471f; --band:#eeeee9; }
body { background:var(--page); color:var(--ink); margin:0; padding-block:28px;
       font:14px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
main { max-width:1240px; margin:0 auto; padding-inline:16px; }
h1 { font-size:27px; margin:0 0 6px; text-wrap:balance; }
h2 { font-size:20px; margin:40px 0 10px; padding-top:14px; border-top:1px solid var(--grid); text-wrap:balance; }
h3 { font-size:15.5px; margin:24px 0 6px; }
p, li, dd { max-width:82ch; }
.note { color:var(--secondary); }
.lead { font-size:15px; }
.lead li { margin-bottom:6px; }
nav { columns:2 260px; font-size:13.5px; margin:12px 0 4px; }
nav a { color:var(--accent); text-decoration:none; }
.wrap { overflow-x:auto; margin:8px 0 18px; }
table { border-collapse:collapse; font-size:12.8px; font-variant-numeric:tabular-nums; background:var(--surface); }
th, td { padding:5px 9px; border-bottom:1px solid var(--grid); text-align:right; white-space:nowrap; vertical-align:top; }
th { color:var(--secondary); font-weight:600; position:sticky; top:0; background:var(--surface); }
td:first-child, th:first-child { text-align:left; }
td.text { text-align:left; white-space:normal; min-width:14ch; max-width:34ch; }
table.overview td.text { max-width:30ch; }
tr.shared td { background:var(--band); }
tr.group td { background:var(--band); color:var(--secondary); font-weight:600; text-align:left; white-space:normal; }
img { max-width:100%; display:block; margin:10px 0 4px; }
figcaption { color:var(--secondary); font-size:13px; max-width:90ch; margin-bottom:14px; }
.method { border-left:3px solid var(--grid); padding-left:14px; margin:18px 0 26px; }
.method dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 14px; margin:6px 0; }
.method dt { color:var(--secondary); font-weight:600; }
.method dd { margin:0; }
.method ul { margin:4px 0; padding-left:20px; }
.facts { font-variant-numeric:tabular-nums; color:var(--secondary); }
.bad { color:var(--warn); font-weight:600; }
code { font-size:12.5px; background:var(--band); padding:0 4px; border-radius:3px; }
@media (max-width:640px) { .method dl { grid-template-columns:1fr; } }
</style>"""


def figure(uri: str, caption: str, width: int | None = None) -> str:
    style = f" style='max-width:{width}px'" if width else ""
    return f"<figure style='margin:0'><img src='{uri}' alt='{caption[:40]}'{style}><figcaption>{caption}</figcaption></figure>"


def method(title: str, anchor: str, items: list[tuple[str, str]]) -> str:
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items)
    return f"<section class='method' id='{anchor}'><h3>{title}</h3><dl>{body}</dl></section>"


def ul(*items: str) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def render(S: Study, figs: dict[str, str], images: dict[str, str]) -> str:
    gpu, cf, mcd, n = S.gpu, S.cf, S.mc["day"], S.night_ms
    total = 1 + n["fog_only_over_direct"] + n["reflected_over_direct"]
    share = {
        "single": 1 / total,
        "fog": n["fog_only_over_direct"] / total,
        "reflected": n["reflected_over_direct"] / total,
    }
    image, clamp, old = n["image"], n["clamp"], n["clamped"]
    day16 = mcd["day_spp16"]["denoised"]["still"]
    small_day = S.less["small"]["day"]["4"]
    representative_day = [
        variant(S.cache["day"]["variants"], kind="sun", size=0.5),
        variant(S.cache["day"]["variants"], kind="sh", size=1.0, order=4),
        S.froxel_entry("day", 8, 64, 1),
        small_day["guided_source"],
        S.less["reuse"]["4"],
    ]
    representative_night = [
        variant(S.cache["night"]["variants"], kind="lit", size=0.25),
        S.learning["field"][0],
        S.less["small"]["night"]["4"]["guided_source"],
        S.froxel_entry("night", 8, 64, 1),
    ]

    def span(values, digits: int = 1) -> str:
        return f"{100 * min(values):.{digits}f}–{100 * max(values):.{digits}f}%"

    unreal_night = [n["unreal"][q]["rel_l1"] for q in ("high", "epic", "cinematic")]
    unreal_day = [S.ue["day"][q]["whole"]["rel_l1"] for q in ("high", "epic", "cinematic")]
    out = [f"<title>雾渲染加速对比</title>{STYLE}<main>", "<h1>雾渲染加速：六个方向的对比</h1>"]
    out.append(
        "<p class='note'>全部误差都在 320×180 上、与 Cycles 参考比较；1080p 只在附录里列时间。"
        "GPU 时间为 RTX 5080，CPU 为 12 线程。</p>"
    )
    out.append(
        "<nav>"
        + "".join(
            f"<div><a href='#{a}'>{t}</a></div>"
            for a, t in (
                ("tldr", "结论"),
                ("how", "1 怎么读"),
                ("lineage", "2 各方法差在哪"),
                ("pros", "3 优缺点总表"),
                ("tables", "4 明细表（320×180）"),
                ("methods", "5 逐个方法：做法与误差来源"),
                ("issues", "6 闭式解的问题清单"),
                ("untested", "7 没测的"),
                ("appendix", "附录：1080p、计时、全部设置"),
            )
        )
        + "</nav>"
    )

    # Conclusions
    out.append("<h2 id='tldr'>结论</h2><ul class='lead'>")
    out.append(
        f"<li><b>白天（整幅图，含多次散射）</b>：路径追踪 16 spp + OIDN 降噪在 GPU 上 {sec(gpu_seconds('day_spp16'))}/帧、误差 {pct(day16['rel_l1'])}，"
        f"比现方案闭式解（{sec(gpu['cf_day_320_shadows_surfaces'])}、{pct(cf['day']['still']['rel_l1'])}）又快又准。"
        "闭式解的误差几乎全在多次散射这一项（换成 Cycles 的多次散射后整图 1.42%）。已找到三个问题，其中一个是地面光重复计数的 bug（§6），"
        "但各占多少、一起修好后还剩多少没测；物体反射进雾的光也没建模，白天占多少没测。</li>"
    )
    out.append(
        f"<li><b>夜间点灯</b>：按和白天同样的口径（整幅图对完整物理），闭式解误差 <span class='bad'>{pct(image['with_halo']['rel_l1'])}</span>"
        f"（白天 {pct(cf['day']['still']['rel_l1'])}）；只看雾光是 <span class='bad'>{pct(n['whole']['rel_l1'])}</span>（白天 {pct(cf['day']['fog_light']['rel_l1'])}）。"
        f"夜间的雾光只有 {pct(share['single'], 0)} 是单次散射，{pct(share['reflected'], 0)} 是地面和物体被灯照亮后反射进雾的光，"
        f"{pct(share['fog'], 0)} 是雾内多次散射；闭式解没有表面反射这一项，灯晕总量又是 Cycles 雾内多次散射的 {n['halo_vs_fog_only']['ratio']:.1f} 倍，"
        f"但其中 {pct(n['halo_split']['all']['from_shadowed_points'], 0)} 落在立方体的灯影里（两个相机都在灯影里）。"
        f"夜间参考关掉了 Cycles 默认的样本截断：截断会压暗夜间的间接光（不截断时雾光亮 {pct(clamp['fog'] - 1, 0)}、"
        f"表面亮 {pct(clamp['surfaces'] - 1, 0)}，整幅图亮 {pct(clamp['frames'] - 1, 0)}），"
        f"对截断参考闭式解是 {pct(old['image']['with_halo']['rel_l1'])} / {pct(old['whole']['rel_l1'])}；白天参考不受截断影响。"
        f"之前的“夜间 {pct(cf['night']['fog_light']['rel_l1'])}”是单次散射子问题里只比雾光的数字（参考也只渲单次散射），只说明闭式解把单次散射的积分算对了。"
        f"在这个口径下它比同精度、不降噪的 Cycles 1024 spp（{sec(gpu_seconds('night_spp1024'))}）快约 30 倍；但按白天的口径（整幅图、路径追踪降噪），"
        f"同一子问题里路径追踪 16 spp + 降噪（{pct(S.mc['night']['night_spp16']['whole_denoised']['rel_l1'])}）反而比闭式解"
        f"（{pct(cf['night']['whole']['rel_l1'])}）准。</li>"
    )
    out.append(
        "<li><b>只换求值方式的方法</b>：白天，网格缓存、降分辨率、重投影、共享阴影、froxel 原型都沿用闭式解的光照模型，"
        f"自身新增的误差只有 {span([e['vs_closed_form']['rel_l1'] for e in representative_day])}（雾光为分母；共享阴影 ≤ "
        f"{pct(max(x['rel_l1'] for x in S.less['density']['errors']), 2)}，以整幅图为分母），而光照模型本身的雾光误差是 "
        f"{pct(cf['day']['fog_light']['rel_l1'])}，所以它们对 Cycles 的整图误差（代表设置）都在 "
        f"{span([e['vs_cycles']['rel_l1'] for e in representative_day])}，彼此看不出区别。"
        "夜间只有网格缓存、降分辨率、froxel 原型和神经场在单次散射子问题上测过（重投影、共享阴影夜间没测），"
        f"自身 {span([e['vs_closed_form']['rel_l1'] for e in representative_night])}，"
        f"而光照模型对完整参考的雾光误差是 {pct(n['whole']['rel_l1'])}。瓶颈在光照模型，不在求值方式。</li>"
    )
    out.append(
        f"<li><b>实时引擎</b>：体积雾本身极便宜（Unreal 在 GPU 上整帧 {sec(S.ue_ms('Day', '320', 'epic'))}，其中雾 "
        f"{sec(S.ue_ms('Day', '320', 'epic', 'GPU/VolumetricFog'))}），但都换成了自己的光照模型："
        f"白天 EEVEE {pct(S.froxel['eevee']['eevee_day_default']['whole']['rel_l1'])}（近地面的雾变薄）、"
        f"Unreal {pct(S.ue['day']['epic']['whole']['rel_l1'])}（雾光整体少 31%，离太阳 55° 以外少 36–42%，推测缺体积多次散射）；"
        f"夜间对完整参考的整幅图 EEVEE {pct(S.eevee_ms['eevee_night_default']['whole']['rel_l1'])}、Unreal {pct(n['unreal']['epic']['rel_l1'])}"
        f"（闭式解 {pct(image['with_halo']['rel_l1'])}）。白天画质档位不改变误差（{span(unreal_day)}），夜间在 {span(unreal_night)} 之间。</li>"
    )
    out.append(
        "<li><b>学习</b>：推理最快，但只在训练分布里有效。残差网络换到夜间，比它修正的闭式解（不带表面模型）还差："
        f"整幅图 {pct(S.learning['images']['closed_form']['night']['rel_l1'])} → {pct(S.learning['images']['residual']['night']['rel_l1'])}"
        f"（对截断参考；带表面模型的现版闭式解对同一参考是 {pct(old['image']['with_halo']['rel_l1'])}）。"
        f"图像网络夜间 {pct(S.learning['images']['image']['night']['rel_l1'], 0)}。</li>"
    )
    out.append(
        "<li><b>怎么选</b>：要和物理一致的白天图像，用 GPU 上的路径追踪 16–64 spp + 降噪。要无噪声、确定性的输出（各传感器仿真的公共输入），"
        f"继续用闭式解，但先修多次散射和夜间灯晕，并补上表面反射进雾的光（§6），否则白天整图有 8% 的系统偏差，夜间 {pct(image['with_halo']['rel_l1'], 0)}。"
        f"降成本的手段里，1/4 分辨率 + 上采样平均源 J（白天自身 {pct(small_day['guided_source']['vs_closed_form']['rel_l1'], 2)}）"
        "和共享阴影扫浓度几乎无损，可以直接用；网格缓存只值得用在天光部分。</li>"
    )
    out.append("</ul>")

    # How to read
    out.append("<h2 id='how'>1 怎么读</h2>")
    out.append(
        ul(
            "<b>分辨率</b>：正文所有误差都在 320×180 上。1080p 没有 Cycles 参考，它的数字（只有时间，以及降分辨率对全分辨率闭式解的误差）都放在附录。",
            "<b>帧</b>：第 255 和 345 两帧合计（两帧光线差别大：第 345 帧被挡住的太阳光是第 255 帧的 3.4 倍）。例外用 † 标出。",
            "<b>参考</b>：白天是 Cycles 4096 spp、16 次体积反弹（完整物理），比整幅图像；Cycles 默认的样本截断（clamp = 10）对白天参考没有可测的影响。"
            f"夜间有两种：完整物理（2048 spp、16 次反弹、关掉截断——截断会把夜间的间接光压暗 {pct(1 - 1 / clamp['indirect'], 0)}），"
            "给整幅图和雾光两个数；以及<b>单次散射子问题</b>（4096 spp、0 次反弹，只比雾光），"
            "大部分夜间方法只在这个子问题上测过，它只能说明单次散射算得准不准。"
            f"参考自身有噪声：白天约 {pct(S.floor['rel_l1'])}（模糊后 {pct(S.floor['rel_l1_blurred'], 2)}），比这还小的差别测不出来。",
            "<b>口径</b>：白天的主数字是整幅图（表面 + 雾光）；夜间要和它对比，看夜间对完整参考的整幅图。"
            f"只看雾光的误差要大得多：雾光约占整幅图的一半，而闭式解的表面误差小得多（白天 {pct(S.parts['closed_form']['surfaces']['local'])}、"
            f"夜间 {pct(image['surfaces']['rel_l1'])}，雾光是 {pct(cf['day']['fog_light']['rel_l1'])}、{pct(n['whole']['rel_l1'])}）。4.1 节把两种口径并排列出。",
            "<b>误差</b> = Σ|方法 − 参考| ÷ Σ参考（所有像素、通道、两帧）。<b>模糊后</b>：两边先做 5×5 均值模糊再算，去掉大部分随机噪声，约等于偏差。"
            "<b>亮度比</b> = Σ方法 ÷ Σ参考，大于 1 偏亮。<b>闪烁</b>：第 255–262 帧上，方法与参考的对数亮度之差逐帧变化量的平均。",
            "<b>两层误差</b>：沿用闭式解光照模型的方法，新增的误差叫<b>自身误差</b>：与闭式解比较雾光（分母为闭式解的雾光）。"
            "它不能直接加到对 Cycles 的误差上：分母不同（对 Cycles 是整幅图），而且两者常部分相抵——白天这些方法的代表设置对 Cycles 都在 "
            f"{span([e['vs_cycles']['rel_l1'] for e in representative_day])}（天光 4 阶球谐还比闭式解低；其余设置见 4.2，最高是重投影隔 16 帧的 "
            f"{pct(S.less['reuse']['16']['vs_cycles']['rel_l1'])}，单帧）。比较这些方法要看自身误差。",
            "<b>时间</b>：每帧时间，GPU 为 RTX 5080，CPU 为 12 线程（测时还有别的程序在跑）。Cycles 和 EEVEE 在 Blender 里渲染，"
            "每帧另有约 50 ms 的同步和存盘开销；PyTorch 原型取预热后 3 次中最快的一次；Unreal 为游戏模式下 CSV 统计的整帧 GPU 时间。",
        )
    )

    # Lineage
    out.append("<h2 id='lineage'>2 各方法差在哪</h2>")
    out.append(
        "<p>每个方法都回答三个问题：光照用什么模型（光从哪来）？在哪里求光照？怎么变成像素？"
        "下表按这三点对比。<b>灰底的方法</b>与闭式解用同一个光照模型，只改了后两点。</p>"
    )
    lineage = [
        [
            "① 路径追踪",
            "物理（与参考相同）",
            "每条随机路径的顶点",
            "每像素平均若干路径，可再降噪",
            "采样噪声；降噪的平滑",
        ],
        [
            "② 闭式解（现方案）",
            "<b>分项模型</b>：太阳（解析 + 阴影图）；天空、地面、多次散射（按仰角制表 × 可见天空格比例）；灯（数值求积 + 灯阴影图）；表面经雾照明",
            "每个像素的视线上",
            "沿视线积分",
            "光照模型本身",
        ],
        ["③ 世界网格缓存", "同 ②", "预先在 3D 网格角点上算好、存起来", "沿视线插值网格再积分", "插值；球谐截断"],
        ["③ 神经场", "同 ②", "用小 MLP 拟合网格的内容", "同上", "拟合误差"],
        ["④ 低分辨率 + 上采样", "同 ②", "只在 1/2 或 1/4 的像素上", "上采样到全分辨率", "跨轮廓插值"],
        ["④ 跨帧重投影", "同 ②", "只在关键帧上", "按深度投影到后续帧", "视线方向和路径变了"],
        ["④ 共享阴影扫浓度", "同 ②", "阴影每帧只追一次", "每种浓度只重新着色", "几乎没有"],
        ["⑤ froxel 原型", "同 ②", "视锥体素（8 px × 8 px × 64 层）中心", "沿深度逐层累积", "体素尺寸，尤其深度层"],
        [
            "⑤ EEVEE",
            "EEVEE 自己的（太阳 + 阴影、世界光近似；无多次散射）",
            "视锥体素",
            "沿深度累积 + 多帧抖动平均",
            "光照模型 + 体素",
        ],
        [
            "⑤ Unreal 5.8",
            "Unreal 自己的（太阳、天光、Lumen；推测无体积多次散射）",
            "视锥体素",
            "沿深度累积 + 历史帧",
            "光照模型 + 体素",
        ],
        ["⑥ 残差网络", "同 ② + 学到的修正", "—", "U-Net 修正闭式解的图像", "泛化"],
        ["⑥ 图像网络", "无（全靠数据）", "—", "U-Net 从无雾图 + 深度直接出图", "泛化"],
    ]
    rows_html = ""
    for row in lineage:
        shared = row[1].startswith("同 ②")
        cells = "".join(f"<td class='text'>{c}</td>" for c in row)
        rows_html += f"<tr{' class=shared' if shared else ''}>{cells}</tr>"
    out.append(
        "<div class='wrap'><table><thead><tr><th>方法</th><th>光照模型</th><th>在哪里求光照</th><th>怎么变成像素</th>"
        f"<th>新增误差的来源</th></tr></thead><tbody>{rows_html}</tbody></table></div>"
    )
    out.append(
        figure(
            figs["time_error"],
            "图 1　只看输出整幅图（或全部雾光）的方法：GPU 时间与对 Cycles 误差。左：白天，路径追踪降噪后在 0.08 s 就比闭式解准；"
            "右：夜间单次散射子问题（参考也只有单次散射；只比雾光，路径追踪不降噪），闭式解和它的衍生方法比同精度的路径追踪快一个数量级；"
            f"按整幅图、降噪后比较，16 spp 已比闭式解准（{pct(S.mc['night']['night_spp16']['whole_denoised']['rel_l1'])} 对 {pct(cf['night']['whole']['rel_l1'])}）。"
            "对完整物理的夜间误差见 4.1、4.4。虚线为参考自身噪声。",
        )
    )
    out.append(
        figure(
            figs["own_error"],
            "图 2　沿用闭式解光照的方法，各自新增的误差都远小于光照模型本身的误差（红虚线）。"
            "所以它们对 Cycles 的误差都在闭式解的 8% 上下，要比较它们只能看这张图。",
            900,
        )
    )

    # Pros and cons
    out.append("<h2 id='pros'>3 优缺点总表</h2>")
    out.append(
        "<p class='note'>每个方向取最有代表性的设置；320×180。括号里的“自身误差”是相对闭式解新增的误差，以闭式解的雾光为分母；"
        "它不能直接加到对 Cycles 的误差上（分母不同，而且常部分相抵）。† 单帧。夜间的完整参考关掉了 Cycles 的样本截断。</p>"
    )
    out.append(overview_table(S))

    # Tables
    out.append("<h2 id='tables'>4 明细表（全部 320×180）</h2>")
    out.append("<h3>4.1 白天和夜间放在同一口径下（闭式解）</h3>")
    out.append(
        f"<p class='note'>白天的 {pct(cf['day']['still']['rel_l1'])} 是整幅图，夜间的 {pct(n['whole']['rel_l1'])} 只是雾光，两者不等价；同口径的对照在同一行。"
        "雾光和表面的误差在同一像素上会部分相抵，所以“各自的误差”相加大于整幅图误差。</p>"
    )
    out.append(scope_table(S))
    out.append("<h3>4.2 白天：整幅图像（含多次散射）</h3>")
    out.append(
        "<p class='note'>† 单帧：重投影为第 255 帧投到第 259 / 271 帧；学习方法为测试帧 405–504。‡ 只有第 345 帧，误差以整图为分母。"
        "缓存的时间只是被缓存的那部分光沿视线积分的时间，括号里是同一积分不用缓存的时间。</p>"
    )
    out.append(day_table(S))
    out.append("<h3>4.3 夜间单次散射子问题：雾光，参考也只有单次散射</h3>")
    out.append(
        "<p class='note'>参考只渲了单次散射（0 次体积反弹），所以这张表只说明各方法把单次散射算得准不准，不代表夜间整体精度（见 4.1、4.4）。</p>"
    )
    out.append(night_table(S))
    out.append("<h3>4.4 夜间：对完整参考（Cycles 2048 spp，16 次反弹，不截断）</h3>")
    out.append(
        f"<p class='note'>Cycles 的全部雾光 = 单次散射 {pct(share['single'], 0)} + 雾内多次散射 {pct(share['fog'], 0)} + "
        f"地面/物体反射进雾 {pct(share['reflected'], 0)}（后两者用一张“表面全黑”的参考拆开，两张都不截断）。"
        f"Cycles 默认的样本截断（clamp = 10）会压暗多次散射：不截断时间接光亮 {pct(clamp['indirect'] - 1, 0)}，"
        f"其中雾内多次散射亮 {pct(clamp['fog_only'] - 1, 0)}、表面反射进雾的光亮 {pct(clamp['reflected'] - 1, 0)}。"
        f"截断参考上的拆分是 {pct(1 / (1 + old['fog_only_over_direct'] + old['reflected_over_direct']), 0)} / {pct(old['fog_only_over_direct'] / (1 + old['fog_only_over_direct'] + old['reflected_over_direct']), 0)} / "
        f"{pct(old['reflected_over_direct'] / (1 + old['fog_only_over_direct'] + old['reflected_over_direct']), 0)}，闭式解整幅图 {pct(old['image']['with_halo']['rel_l1'])}、"
        f"雾光 {pct(old['whole']['rel_l1'])}，灯晕是雾内多次散射的 {old['halo_vs_fog_only']['ratio']:.1f} 倍。"
        "无雾的夜间渲染（闭式解的表面输入）不受截断影响。</p>"
    )
    out.append(night_all_table(S))

    # Methods
    out.append("<h2 id='methods'>5 逐个方法：做法与误差来源</h2>")
    out.append(
        figure(
            file_uri(ANALYSIS / "day_regions" / "signed_maps_345.png"),
            "图 3　白天第 345 帧（单帧，所以数字与两帧合计略有不同）各方法的偏差图：5×5 模糊后（方法 − 参考）÷ 参考平均亮度，红偏亮、蓝偏暗。"
            "闭式解、froxel、两种缓存的图几乎一样——它们共享同一个光照模型。",
            1100,
        )
    )
    out.append(methods_html(S, figs, images))

    # Issues
    out.append("<h2 id='issues'>6 闭式解的问题清单（按影响排序）</h2>")
    out.append("<ol>")
    out.append(
        f"<li><b>夜间灯晕放错了地方</b>（<code>visionsim/medium/halos.py</code>）：总量是 Cycles 雾内多次散射的 "
        f"{n['halo_vs_fog_only']['ratio']:.1f} 倍，但其中 {pct(n['halo_split']['all']['from_shadowed_points'], 0)} 来自视线上落在立方体灯影里的点"
        "（两帧的相机都在立方体的灯影里，每条视线都从影子里出发）。原因：灯晕表在格点上收集二次散射光时，收集光线只截到地面（<code>_to_ground</code>），"
        "不被物体挡住，所以影子里的格点照样收到立方体另一侧被照亮的雾的光，存的值和同距离受光处相近（比值约 0.8–1.2）。"
        f"所以全在影子里的像素是 {n['halo_vs_fog_only_shadow']['ratio']:.0f} 倍；受光像素的 {n['halo_vs_fog_only_lit']['ratio']:.1f} 倍主要也是这个原因"
        f"（其中 {pct(n['halo_split']['lit_pixels']['from_shadowed_points'], 0)} 的灯晕来自视线上的影子段）。只算受光的点，灯晕是 Cycles 雾内多次散射的 "
        f"{n['halo_split']['all']['lit_points_over_fog_only']:.2f} 倍（Cycles 这一项也含影子里的光，所以受光部分是对是偏暗，单从这个数分不出来）。"
        "修法是让收集光线在物体处截断（例如查灯的阴影图）；只修这一条，夜间雾光总量会明显变少，要和第 2 条（表面反射光）一起改。</li>"
    )
    out.append(
        f"<li><b>不建模表面反射进雾的光</b>：夜间这部分是单次散射的 {n['reflected_over_direct']:.2f} 倍、全部雾光的 {pct(share['reflected'], 0)}，"
        "集中在灯下方的地面附近（图 6）。<code>halos.py</code> 的文档写明地面只挡光、不反光。白天的地面反射另有一项，问题见下一条。</li>"
    )
    out.append(
        "<li><b>地面光重复计数</b>（<code>visionsim/medium/scattering.py</code> 约第 433 行）：“地面反射光被雾散射一次”既作为独立的地面项，"
        "又进了多次散射表的第一阶，并被 1/(1 − f) 放大 1.27–1.38 倍。白天向下看近处地面时受光处亮 26%、阴影处亮 70%。</li>"
    )
    out.append(
        "<li><b>多次散射表与方位角无关</b>：Cycles 同一仰角下朝太阳一侧是背对太阳的 1.7–2 倍，表格是平的，朝太阳 40° 内暗 31–41%（图 5）。</li>"
    )
    out.append(
        "<li><b>多次散射和地面项不看太阳阴影</b>（只看可见天空格）：阴影地面上的多次散射项比受光地面偏亮得多"
        "（按距离和太阳角分档，阴影处亮 33–144%，受光处 17–46%），但其中大部分是第 3 条的重复计数；向下 8° 以下的地面去掉重复计数后，"
        "阴影处 −9%、受光处 −31%，相差约 22 个百分点，这大致是不看阴影本身的影响。"
        "表面的辉光项也没乘太阳可见度（<code>visionsim/medium/surfaces.py</code> 约第 316 行），阴影里的表面亮 17%。</li>"
    )
    out.append(
        "<li><b>只用像素中心深度</b>：Cycles 用 1.5 px 的像素滤波，夜间立方体轮廓上的差异，第 345 帧有 40–45% 来自这里（第 255 帧约 27%）。</li>"
    )
    out.append(
        f"<li><b>参考本身</b>：Cycles 默认的样本截断（clamp = 10）让夜间参考的间接光暗 {pct(1 - 1 / clamp['indirect'], 0)}"
        f"（不截断时亮 {pct(clamp['indirect'] - 1, 0)}），全部雾光暗 {pct(1 - 1 / clamp['fog'], 0)}，表面暗 {pct(1 - 1 / clamp['surfaces'], 0)}；"
        f"白天参考几乎不受影响（{pct(max(S.day_clamp.values()) - 1, 2)} 以内）。夜间的完整参考因此改用不截断的渲染。</li>"
    )
    out.append("</ol>")
    out.append(
        "<p><b>注意第 3 条和多次散射整体不足互相抵消</b>：去掉重复计数后，白天多次散射这一项整体暗约 29%，整图暗 11.0%，误差反而从 8.2% 升到 11.6%。"
        "只修 bug 不修表格会更差；第 3–5 条要一起改。</p>"
    )

    # Untested
    out.append("<h2 id='untested'>7 没测的</h2>")
    out.append(
        ul(
            "帧太少：只有两帧，有些方法只有一帧；两帧之间差别很大（太阳网格误差差 3.4 倍），结论能否推广到其他帧没测。网格对齐只在缓存的复核里测过。",
            "1080p 没有参考图；与分辨率有关的成因（像素滤波、以像素计的 froxel 瓦片、上采样、EEVEE）只在 320×180 上看过。",
            "闪烁只测了路径追踪、闭式解、EEVEE 和 Unreal；缓存、froxel 抖动、重投影都没测。",
            "完整闭式解（含表面模型）在浓雾下没测；不带表面模型的版本浓雾（×2）整图 11.7%，没有分解。",
            "EEVEE、Unreal 内部的机制没有在源码里核实；Unreal 的雾密度换算和它自己的透射率没有独立核实，它的无雾帧与 Cycles 差 4.5%；没有关掉 Lumen 的渲染。",
            "学习方法：白天测试帧上用的是 10 月 2 日的旧版闭式解，浓雾和夜间用的是不带表面模型的闭式解；只在 CPU 上训练了约 2 分钟；夜间只对截断的参考算过。",
            f"灯晕在受光的点上是否偏暗：只算受光的点，灯晕是 Cycles 雾内多次散射的 {n['halo_split']['all']['lit_points_over_fog_only']:.2f} 倍，"
            "但 Cycles 这一项也含影子里的光，分不出来；没有物体、地面全黑的点灯场景里，灯晕加单次散射与 Cycles（不截断）相差 1.4%（提交 cd4fcd8 的验证）。"
            "夜间闭式解带表面模型的 GPU 时间没测（29 ms 只含雾光）。",
            "修复第 6 节的问题后效果如何，还没有测。",
        )
    )

    # Appendix
    out.append("<h2 id='appendix'>附录</h2>")
    out.append(appendix_tables(S))
    out.append(
        figure(
            images["hd"],
            "图 A1　1080p 局部的雾光（提亮 2 倍）：1/4 分辨率双线性上采样在轮廓上有锯齿，上采样平均源 J 几乎看不出来。",
            1100,
        )
    )
    out.append("</main>")
    return "\n".join(out)


def methods_html(S: Study, figs: dict[str, str], images_: dict[str, str]) -> str:
    gpu, cf, mcd, mcn, n = S.gpu, S.cf, S.mc["day"], S.mc["night"], S.night_ms
    day = S.cache["day"]["variants"]
    night = S.cache["night"]["variants"]
    small_day, small_night = S.less["small"]["day"]["4"], S.less["small"]["night"]["4"]
    reuse = S.less["reuse"]
    e = S.froxel["eevee"]
    images = S.learning["images"]
    field = S.learning["field"][0]
    parts = []

    parts.append(
        method(
            "5.1 路径追踪（Cycles）",
            "m-mc",
            [
                (
                    "做法",
                    (
                        "与参考完全相同的场景和物理，只是每像素只追 4–1024 条随机路径；可加 OIDN 降噪、路径引导、自适应采样。"
                        "它和参考的差别只有“样本不够”，与光照模型无关。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>16 spp + 降噪：GPU {sec(gpu_seconds('day_spp16'))}，误差 {pct(mcd['day_spp16']['denoised']['still']['rel_l1'])}"
                        f"（模糊后 {pct(mcd['day_spp16']['denoised']['still']['rel_l1_blurred'])}）；1024 spp 不降噪：GPU {sec(gpu_seconds('day_spp1024'))}，"
                        f"{pct(mcd['day_spp1024']['noisy']['still']['rel_l1'])}。夜间单次散射子问题（只比雾光、不降噪）："
                        f"64 spp {pct(mcn['night_spp64']['fog_light']['rel_l1'])}，1024 spp {pct(mcn['night_spp1024']['fog_light']['rel_l1'])}；"
                        f"同一子问题的整幅图降噪后 16 spp {pct(mcn['night_spp16']['whole_denoised']['rel_l1'])}、"
                        f"64 spp {pct(mcn['night_spp64']['whole_denoised']['rel_l1'])}（闭式解 {pct(cf['night']['whole']['rel_l1'])}）。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        "不降噪：误差就是随机噪声。64 spp 起偏差 ≤ 0.04%，误差按 1/√spp 下降，模糊后只剩约 1/5。多次散射只占 34% 的光，却贡献 72–79% 的方差。",
                        "降噪后：16 spp 从 17.8% 降到 2.6%，但模糊后还有 1.4%。这不是平滑的偏差：约 70% 随种子变化（64 spp 两个种子的误差图相关只有 0.27），"
                        "是降噪留下的低频斑块；与种子无关的部分约 0.44%，略集中在棋盘格和深度边缘。脚本固定种子，噪声逐帧相关（不降噪 0.92），降噪后的斑块也基本不动（相邻帧误差相关 0.79–0.87），所以闪烁不大。",
                        "真实的小偏差：4 spp 降噪后整体亮约 0.6%（不降噪亮 1.4%，其中约 0.7% 均匀分布）；默认自适应采样整体暗 0.44%，几乎都在多次散射。",
                        "路径引导：64–256 spp 时误差降 10–13%，但耗时 1.45–1.54 倍、只能在 CPU 上用，而且噪声帧间不再相关，不降噪的闪烁变成 2.8 倍。",
                        "“闭式解算单次散射 + 路径追踪算其余”反而更差：同一条路径里各部分的噪声是负相关的，单独换掉一部分就失去了这种抵消。",
                        "夜间单次散射子问题：雾光无偏，误差全是噪声，集中在灯附近（误差最大的 1% 像素占全部误差的 34%）。",
                    ),
                ),
                ("优点", "物理完整，误差可用样本数控制，没有要维护的模型；GPU 上 16 spp 已经比闭式解快。"),
                (
                    "缺点",
                    "有噪声，降噪后仍有低频斑块；夜间灯光需要很多样本；小分辨率下 GPU 时间大半是 Blender 每帧的固定开销。",
                ),
            ],
        )
    )

    parts.append(
        figure(
            figs["cf_parts"], "图 4　闭式解白天的误差几乎全在多次散射这一项；单次散射和表面只有参考噪声底的几倍。", 900
        )
    )
    parts.append(
        figure(
            figs["azimuth"],
            "图 5　天空像素上闭式解 ÷ Cycles：单次散射在各方位都准；多次散射朝太阳一侧只有 0.6–0.7 倍，背对太阳 1.2 倍。",
            720,
        )
    )
    parts.append(
        method(
            "5.2 逐像素解析积分（闭式解，现方案）",
            "m-cf",
            [
                (
                    "做法",
                    (
                        "不追踪路径，对每个像素沿视线做积分。光照来自一个分项模型：太阳（解析积分，阴影图决定光柱）；天空、地面反射、多次散射"
                        "（按仰角预先制表，再乘该点能看到的天空格比例，共 107 格）；灯（去奇点的数值求积 + 灯的阴影图）；表面（经雾照明后再衰减）。没有噪声。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>白天 GPU {sec(gpu['cf_day_320_shadows_surfaces'])}（其中表面模型约 80%；只算雾 {sec(gpu['cf_day_320_shadows'])}），"
                        f"误差 {pct(cf['day']['still']['rel_l1'])}，模糊后 {pct(cf['day']['still']['rel_l1_blurred'])}，亮 {pct(cf['day']['still']['ratio'] - 1)}；"
                        f"夜间对完整参考：整幅图 {pct(n['image']['with_halo']['rel_l1'])}（只看雾光 {pct(n['whole']['rel_l1'])}）；"
                        f"单次散射子问题（参考也只有单次散射）：雾光 {pct(cf['night']['fog_light']['rel_l1'])}（GPU {sec(gpu['cf_night_320_shadows'])}，只算雾光），"
                        f"整幅图 {pct(cf['night']['whole']['rel_l1'])}。</span>"
                    ),
                ),
                (
                    "白天误差来源",
                    ul(
                        "按光拆开（图 4），各项误差以整幅图为分母：多次散射 7.85 个百分点（相对这部分光 22.9%），单次散射 0.44，表面 1.42；"
                        "它们在同一像素上部分相抵，所以相加大于整图的 8.2%。"
                        "把闭式解的多次散射换成 Cycles 的，整图误差降到 1.42%（模糊后 0.98%）。",
                        "① 地面光重复计数（bug）：地面反射光的一次散射同时进了地面项和多次散射表，并被放大 1.27–1.38 倍。向下看近处地面时最明显：受光处亮 26%，阴影处亮 70%。",
                        "② 多次散射表与方位角无关（图 5）：Cycles 的多次散射朝太阳一侧更亮（前向散射的光再散射一次仍偏向前方），表格是平的，朝太阳 40° 内暗 31–41%。",
                        "③ 多次散射、地面项和表面辉光都不看太阳阴影：阴影地面上的多次散射项按距离和太阳角分档亮 33–144%（受光处 17–46%），"
                        "其中大部分是 ① 的重复计数；向下 8° 以下的地面去掉重复计数后阴影处 −9%、受光处 −31%，相差约 22 个百分点，大致是不看阴影本身的影响。"
                        "阴影里的表面亮 17%（辉光项没乘太阳可见度）。",
                        "① 的过亮和多次散射整体约 29% 的不足互相抵消：去掉重复项后整图暗 11.0%（其中多次散射项贡献 9.8 个百分点），误差升到 11.6%。"
                        "8.2% 是两个更大误差相减的结果。",
                        "单次散射（模糊后 0.30%）和表面（1.00%）是参考噪声底的 6–7 倍，是真实的小偏差。误差不集中在轮廓上。",
                    ),
                ),
                (
                    "夜间误差来源",
                    ul(
                        f"单次散射子问题几乎精确（{pct(cf['night']['fog_light']['rel_l1'])}，模糊后 {pct(cf['night']['fog_light']['rel_l1_blurred'])}；参考也只有单次散射）。"
                        "多出噪声底的部分集中在立方体轮廓约 1% 的像素上，第 345 帧其中 40–45%（第 255 帧约 27%）是因为闭式解只用像素中心的深度，"
                        "而 Cycles 用 1.5 px 的像素滤波平均；灯周围的亮区亮 1.6%。",
                        f"对完整参考（图 6）：打开灯晕后整幅图误差 {pct(n['image']['with_halo']['rel_l1'])}（白天 {pct(cf['day']['still']['rel_l1'])}），"
                        f"只看雾光 {pct(n['whole']['rel_l1'])}，亮度 {n['whole']['ratio']:.2f}。"
                        f"灯晕总量是 Cycles 雾内多次散射的 {n['halo_vs_fog_only']['ratio']:.1f} 倍，但 {pct(n['halo_split']['all']['from_shadowed_points'], 0)} "
                        "落在立方体的灯影里（§6）；而占全部雾光一半的“表面反射进雾的光”没有建模，集中在灯下方的地面附近。"
                        f"多出的灯晕几乎都在影子里，缺的反射光在灯下方，雾光总量只是碰巧相抵（亮度比 {n['whole']['ratio']:.2f}），分布完全不对。"
                        f"表面也偏暗 {pct(1 - n['image']['surfaces']['ratio'], 0)}（表面误差 {pct(n['image']['surfaces']['rel_l1'])}）："
                        f"截断让参考的表面暗 {pct(1 - 1 / n['clamp']['surfaces'], 0)}（无雾输入不受截断影响，差 0.005%），"
                        f"对截断参考闭式解的表面亮度比 {n['clamped']['image']['surfaces']['ratio']:.2f}；不截断时多出的是经雾散射到表面的灯光，"
                        "闭式解只用相似关系近似它，灯晕也不照亮表面。",
                    ),
                ),
                ("优点", "无噪声、确定性，闪烁最小；夜间单次散射的雾光算得准；所有衍生方法（③④⑤ 原型）的基础。"),
                (
                    "缺点",
                    (
                        f"精度上限由光照模型决定，加算力也改不了；整幅图对完整参考，白天差 {pct(cf['day']['still']['rel_l1'])}，"
                        f"夜间差 {pct(n['image']['with_halo']['rel_l1'])}。"
                    ),
                ),
            ],
        )
    )
    parts.append(
        figure(
            figs["night_multiple"],
            "图 6　夜间第 345 帧的全部雾光按来源拆开（同一亮度刻度）。Cycles 的“其余”光主要是灯下方地面反射进雾的光；"
            "闭式解的灯晕是一个绕着灯的光团，大半压在立方体（右侧）的灯影里。最后一张：闭式解全部 ÷ Cycles 全部。",
            1100,
        )
    )

    parts.append(
        figure(
            figs["night_single_maps"],
            "图 7　夜间第 345 帧，单次散射雾光的局部相对偏差（5.3–5.9 都会用到）。"
            "灯网格 1 m 在立方体的两个阴影面上整片偏亮；球谐把灯周围的亮光变成环；EEVEE 在近处地面附近整片偏暗。",
            1100,
        )
    )
    parts.append(
        method(
            "5.3 世界网格缓存",
            "m-cache",
            [
                (
                    "做法",
                    (
                        "把光照模型里只取决于空间位置的部分，预先算在世界坐标 3D 网格的角点上；渲染时沿视线三线性插值，而不是逐节点查阴影图或逐格查天空可见度。"
                        "存三种东西：太阳可见度（每格 1 字节）；灯的“透射率 × 可见度”（方向和 1/r² 仍解析计算）；天空 + 地面 + 多次散射的入射光，按球谐展开。"
                        "网格与相机无关，所有帧、所有相机共用。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>白天太阳 0.5 m：{size(variant(day, kind='sun', size=0.5)['bytes'])}，自身 {own(variant(day, kind='sun', size=0.5)['vs_closed_form'])}；"
                        f"天光 4 阶球谐 1 m：{size(variant(day, kind='sh', size=1.0, order=4)['bytes'])}，自身 {own(variant(day, kind='sh', size=1.0, order=4)['vs_closed_form'])}；"
                        f"夜间（单次散射子问题）灯 0.25 m：{size(variant(night, kind='lit', size=0.25)['bytes_fp32'])}，自身 {own(variant(night, kind='lit', size=0.25)['vs_closed_form'])}；"
                        f"灯 1 m：{size(variant(night, kind='lit', size=1.0)['bytes_fp32'])}，{own(variant(night, kind='lit', size=1.0)['vs_closed_form'])}。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        "太阳可见度：三线性插值把阴影边界抹开，86–97% 的误差在离阴影边一个体素以内。按太阳光算，1 / 0.5 / 0.25 / 0.125 m 网格分别是 4.0 / 2.2 / 0.90 / 0.34%。"
                        "偏差的符号取决于网格和阴影边界的相对位置：1 m 网格默认对齐暗 2.8%，平移后亮 0.9% 或 2.6%。",
                        "天光球谐：误差主要来自阶数截断，不是插值（只插值 1.4%，4 阶球谐 4.2%）。入射光在两处突变：开阔地的地平线（地面和天空亮度跳变），"
                        "以及物体挡住天空的边缘，低阶球谐表示不了，两者大约各占一半；2 阶在物体附近把雾光高估 13–20%。分光源看（附录 A7）：多次散射 3 阶就够，"
                        "天空要 6 阶，太阳 12 阶仍差 27%，所以太阳不能放进球谐。4 阶对 Cycles（7.5%）比闭式解（8.2%）还好，是碰巧：它让地平线变亮，补上了闭式解在那里的偏暗。",
                        "夜间灯网格：误差来自对灯可见度的插值，网格跨过立方体的两个 45° 阴影面（离灯 2–6 m）；只存透射率的网格误差只有 −0.3%。"
                        "1 m 网格默认对齐时角点正好落在阴影面上（可见度 0.5），整体亮 25%；平移网格后误差在 +34.5% 到 −19.7% 之间——大小由插值决定，符号由对齐决定。",
                        "夜间灯光球谐：雾的相函数 g = 0.8 很尖，需要远多于 8 阶（g⁸ = 0.17）；2 阶在 45–60° 方向上是真值的 3.5–3.8 倍，79.5–133° 方向为负。",
                    ),
                ),
                (
                    "速度",
                    (
                        f"GPU 上只有天光部分值得缓存（{sec(gpu['exact_day_smooth_320'])} → {sec(gpu['cache_day_sh4_1m_320'])}）。"
                        f"太阳（{sec(gpu['exact_day_sun_320'])} → {sec(gpu['cache_day_sun0.5_320'])}）和灯（{sec(gpu['exact_night_320'])} → "
                        f"{sec(gpu['cache_night_lit_0.25_320'])}）用了缓存反而更慢：三线性插值 8 个角点比查一次阴影图还贵。"
                    ),
                ),
                ("优点", "与视角无关，可跨帧、跨相机复用；昂贵的天光部分快 6 倍，误差可控。"),
                ("缺点", "阴影边界被抹开、误差符号随对齐变；球谐表示不了尖的相函数；存储随分辨率立方增长。"),
            ],
        )
    )
    parts.append(
        figure(
            figs["cache"],
            "图 8　夜间灯光缓存：存“透射率 × 可见度”时，误差随网格变细稳定下降；存球谐到 8 阶（178 MB）仍有 19%；"
            "神经场 44 KB 就达到 0.25 m 网格（2.2 MB）的精度。",
            720,
        )
    )

    parts.append(
        method(
            "5.4 神经场",
            "m-field",
            [
                (
                    "做法",
                    f"用一个 {field['parameters']:,} 参数的 MLP（输入为位置的傅里叶特征）代替夜间灯的“透射率 × 可见度”网格，其余同 5.3。",
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>{size(field['bytes_fp32'])}；自身 {own(field['vs_closed_form'])}；对 Cycles 单次散射子问题 {pct(field['vs_cycles']['rel_l1'])}"
                        "（只比雾光；白天和夜间完整参考都没测）；"
                        f"GPU 训练 3000 步 {sec(gpu['field_train_3000_steps'])}，查询 {sec(gpu['field_night_320'])}/帧（直接算 {sec(gpu['exact_night_320'])}）。</span>"
                    ),
                ),
                (
                    "误差来源",
                    (
                        "总量几乎无偏（0.999）；90% 的误差在离阴影边 0.125 m 以内（离灯 2–4 m），大多正负相抵；楔形阴影边上有一小段单侧漏光，"
                        "局部亮 1.19 倍（0.25 m 网格是 1.55 倍）。"
                    ),
                ),
                ("优点", "存储是同精度网格的 1/50。"),
                ("缺点", "训练和查询都比直接算慢；场景或灯一变就要重训。"),
            ],
        )
    )

    parts.append(
        method(
            "5.5 低分辨率 + 上采样",
            "m-low",
            [
                (
                    "做法",
                    (
                        "只在 1/2 或 1/4 的像素上（160×90 或 80×45 条视线）算闭式解，再上采样到 320×180。三种上采样：双线性；按深度加权（联合双边）；"
                        "先上采样“平均源”J = 雾光 ÷ (1 − 透射率)，再乘每个像素精确的 1 − 透射率（透射率很便宜，可以全分辨率算）。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>1/4，白天自身误差：双线性 {own(small_day['bilinear']['vs_closed_form'])}，深度引导 {own(small_day['guided']['vs_closed_form'])}，"
                        f"上采样 J {own(small_day['guided_source']['vs_closed_form'])}；夜间（单次散射子问题）{own(small_night['bilinear']['vs_closed_form'])} / "
                        f"{own(small_night['guided']['vs_closed_form'])} / {own(small_night['guided_source']['vs_closed_form'])}。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        "双线性：近处物体和远处背景的雾光在轮廓两侧被平均，1/4 时 78% 的误差在轮廓 4 px 以内。",
                        "深度引导：轮廓误差 0.99 → 0.29，但在地平线、地面、天空上新增约 0.17——高斯深度权重的宽度和相邻采样行之间的深度变化差不多大，插值变成横向台阶。",
                        "上采样 J：雾光主要随距离变，而 1 − 透射率已经精确处理了距离，J 在轮廓两侧几乎连续，几何误差几乎消失（不算阴影时 0.05%）。"
                        "剩下的来自天空、地面、多次散射项的遮挡（太阳只占 0.08）；其中 0.11–0.14 来自闭式解只用 8 个天空可见度样本，换成 32 个降到 0.30%。",
                        "夜间（单次散射子问题）：双线性的误差 87% 在灯后立方体的边上——灯周围的亮光被抹到立方体上，亮区本身变暗；J 在夜间不比雾光平滑多少，所以帮助有限。",
                    ),
                ),
                (
                    "速度",
                    (
                        f"320×180 下白天 1/4 只快 {cf['day']['seconds']['shadows'] / sum(small_day['seconds'].values()):.1f} 倍"
                        f"（CPU {sec(cf['day']['seconds']['shadows'])} → {sec(sum(small_day['seconds'].values()))}，"
                        f"阴影追踪等固定开销不随像素减少）；夜间只算单次散射时快 {cf['night']['seconds']['shadows'] / sum(small_night['seconds'].values()):.0f} 倍"
                        f"（CPU {sec(cf['night']['seconds']['shadows'])} → {sec(sum(small_night['seconds'].values()))}；开灯晕的版本没测）。"
                        f"1080p 上 GPU {sec(gpu['cf_day_1080_shadows'])} → {sec(gpu['cf_day_1080_low4'])}（附录 A1）。"
                    ),
                ),
                ("优点", "几乎无损，实现简单，可以叠加在任何逐像素方法上（也包括路径追踪）。"),
                ("缺点", "小分辨率下提速有限；深度不连续多的场景要用上采样 J。"),
            ],
        )
    )

    parts.append(
        method(
            "5.6 跨帧重投影",
            "m-reuse",
            [
                (
                    "做法",
                    "在关键帧上算 J，后续帧按深度把关键帧的 J 投影过来，再乘本帧精确的 1 − 透射率；关键帧看不到的像素用不带阴影的闭式解补。只测了白天。",
                ),
                (
                    "数字",
                    "<span class='facts'>自身误差：隔 "
                    + "、".join(f"{g} 帧 {own(reuse[g]['vs_closed_form'])}" for g in ("1", "4", "16", "32"))
                    + "。</span>",
                ),
                (
                    "误差来源",
                    ul(
                        "隔 32 帧时，新露出的 3.4% 像素贡献 1.13 个百分点：它们用不带阴影的解补，全部偏亮，其中 28% 是新露出的天空。",
                        "看到的像素贡献 2.81 个百分点，这是方法固有的：J 取决于整条视线，不只取决于表面点。视线方向变了（太阳相函数在新旧两条视线上相差中位数 6%、"
                        "第 95 百分位 43%）贡献 2.76%；视线路径变了（关键帧的视线从后方 2.3 m 出发，路过的雾里被物体挡住的天空少）贡献 2.68%；两者方向相反，部分抵消。",
                    ),
                ),
                ("优点", "复用整帧的雾，投影本身只要 0.01–0.04 s（CPU）。"),
                ("缺点", "误差随相机位移增长；补洞仍要算一遍无阴影的闭式解，占了大部分时间。"),
            ],
        )
    )

    dens = S.less["density"]
    parts.append(
        method(
            "5.7 共享阴影扫浓度",
            "m-density",
            [
                (
                    "做法",
                    "同一帧要渲多种雾浓度时，阴影（太阳阴影段 + 8 个天空可见度样本）只追一次，每种浓度只重新着色。",
                ),
                (
                    "数字",
                    f"<span class='facts'>4 种浓度（×0.5–×4）：CPU {sec(dens['separate_seconds'])} → {sec(dens['trace_seconds'] + dens['shade_seconds'])}；"
                    "与分别计算相差 "
                    + " / ".join(pct(x["rel_l1"], 2) for s, x in zip(dens["scales"], dens["errors"]) if s != 1.0)
                    + "（整图）。</span>",
                ),
                (
                    "误差来源",
                    (
                        "全部来自 8 个天空可见度样本：样本沿视线的位置随浓度变，共享时用的是浓度 ×1 的位置。太阳阴影段与浓度无关（差为 0）。"
                        "以 64 个样本为真值，共享和重新追踪一样准（0.18 / 0.48% 对 0.18 / 0.50%），所以共享不损失精度。"
                    ),
                ),
                ("优点", "多浓度渲染几乎白送一半时间。"),
                ("缺点", "只对“同一帧多种雾”有用。"),
            ],
        )
    )

    fd, fn, fj = S.froxel_entry("day", 8, 64, 1), S.froxel_entry("night", 8, 64, 1), S.froxel_entry("night", 8, 64, 8)
    parts.append(
        method(
            "5.8 froxel 原型（自写，光照同闭式解）",
            "m-froxel",
            [
                (
                    "做法",
                    (
                        "Unreal / EEVEE 的结构，但光照用闭式解的模型：把视锥分成 8 px × 8 px × 64 层深度（层按 EEVEE 的方式分布在 0.1–100 m）的体素，"
                        "在每个体素中心算一次光照（太阳 + 阴影图、107 个天空格），再沿深度逐层累积透射率和散射光；可以每帧抖动体素中心、多帧平均。"
                        "它把“闭式解 vs 引擎”的差别拆成两半：结构的误差在这里测，光照模型的误差看 EEVEE / Unreal。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>白天 GPU {sec(gpu['froxel_day_320_8_64_1'])}，自身 {own(fd['vs_closed_form'])}；夜间（单次散射子问题）自身 {own(fn['vs_closed_form'])}，"
                        f"抖动 8 帧 {own(fj['vs_closed_form'])}。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        "白天：自身误差与瓦片大小无关（8 / 4 / 2 px：1.05 / 0.90 / 1.01%），白天的光照在屏幕上很平滑。",
                        "夜间（单次散射子问题）：灯附近 1.41（偏亮 0.9），其余 1.79。主因是深度层太厚（灯附近 0.38 m），不是瓦片（0.24 m）：深度层加倍降到 1.44%，瓦片减半只降到 2.58%；"
                        "抖动 8 帧平均降到 1.96%。",
                    ),
                ),
                ("优点", "开销固定、与图像复杂度无关，与引擎同构。"),
                (
                    "缺点",
                    "这个原型没比闭式解快：每个体素仍要查 107 个天空格。引擎能快三个数量级，是因为它们把光照模型大幅简化了——误差也就来自那里（5.9、5.10）。",
                ),
            ],
        )
    )

    parts.append(
        method(
            "5.9 EEVEE",
            "m-eevee",
            [
                (
                    "做法",
                    (
                        "Blender 的实时渲染器。froxel 结构（默认 8 px × 64 层到 100 m，每帧 16 个抖动样本），光照用 EEVEE 自己的：太阳 + 阴影、世界光的近似，"
                        "没有多次散射。在 RTX 5080 上用 Blender 渲染，时间里含每帧约 50 ms 的同步和存盘。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>白天 GPU {sec(gpu_seconds('eevee_day_default'))}，误差 {pct(e['eevee_day_default']['whole']['rel_l1'])}；每帧 1 样本 "
                        f"{sec(gpu_seconds('eevee_day_1sample'))}，{pct(e['eevee_day_1sample']['whole']['rel_l1'])}，闪烁 {e['eevee_day_1sample']['flicker']['temporal']:.4f}"
                        f"（闭式解 {cf['day']['flicker']['temporal']:.4f}）。夜间对完整参考：整幅图 {pct(S.eevee_ms['eevee_night_default']['whole']['rel_l1'])}"
                        f"（精细 {pct(S.eevee_ms['eevee_night_fine']['whole']['rel_l1'])}），雾光 {pct(S.eevee_ms['eevee_night_default']['fog_light']['rel_l1'])}"
                        f"（精细 {pct(S.eevee_ms['eevee_night_fine']['fog_light']['rel_l1'])}）；单次散射子问题（只比雾光）{pct(e['eevee_night_default']['fog_light']['rel_l1'])}，"
                        f"精细 {pct(e['eevee_night_fine']['fog_light']['rel_l1'])}。精细设置主要改善单次散射，整幅图反而略差。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        "白天 9.8% 是两个偏差相抵：雾光暗 17%（占整图 −9.4%），表面亮 8%（+3.7%），都在向下看近处地面时最严重。"
                        "从表面亮度反推，EEVEE 近地面的雾只衰减了精确值的 0.68–0.71 倍，即最浓的那层近地雾在 EEVEE 里变薄了；精细体素（2 px × 256 层）改善到 0.86–0.91，"
                        "所以这与体素大小有关。精细设置仍差约 12%，原因不明。",
                        "每帧 1 样本：多出的误差是高频的，在棋盘格和阴影边上（混叠与抖动）；闪烁是闭式解的 3.5 倍。",
                        "夜间（单次散射子问题）：亏损由深度决定，与散射角无关——4 m 以内地面上的雾光只有单次散射参考的 0.15 倍；"
                        "最符合的解释是离地 0.4 m（精细 0.15 m）以内的雾收不到灯光。"
                        f"对完整参考：没有多次散射，雾光只有 Cycles 全部雾光的 {S.eevee_ms['eevee_night_default']['fog_light']['ratio']:.2f} 倍"
                        f"（精细 {S.eevee_ms['eevee_night_fine']['fog_light']['ratio']:.2f} 倍）。",
                    ),
                ),
                ("优点", "现成，直接在 Blender 里用。"),
                ("缺点", "近地面雾变薄、收不到灯光；没有多次散射；内部机制没有在源码里核实。"),
            ],
        )
    )

    u = S.ue["day"]["epic"]
    parts.append(
        method(
            "5.10 Unreal Engine 5.8",
            "m-unreal",
            [
                (
                    "做法",
                    (
                        "指数高度雾 + 体积雾（froxel；Epic = 8 px × 128 层），参数按引擎源码换算成与 Cycles 相同的雾；太阳、天光、点灯，Lumen 全局光照开着。"
                        "图像用 Movie Render Queue 每帧 1 个样本、带历史帧累积，与实时相同；时间用游戏模式的 CSV 统计。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>GPU 整帧 {sec(S.ue_ms('Day', '320', 'epic'))}，其中体积雾 {sec(S.ue_ms('Day', '320', 'epic', 'GPU/VolumetricFog'))}；"
                        f"白天误差 {pct(u['whole']['rel_l1'])}，亮度 {u['whole']['ratio']:.2f}；High / Cinematic 与 Epic 相差不到 0.3 个百分点。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        f"白天几乎全是雾光偏暗 31%，表面只差 −0.5%。离太阳越远缺得越多（40° 以内 +0.9%，55° 以外 −36% 到 −42%），离相机越远也越多（0–5 m −21%，50 m 外 −48%）。"
                        f"它的雾光是 Cycles 单次散射的 {u['fog_light_vs_single']['ratio']:.2f} 倍、全部雾光的 {u['fog_light_vs_multiple']['ratio']:.2f} 倍："
                        "离太阳 40° 内雾光总量对得上，但那里 Cycles 的雾光有 39% 是多次散射，所以不能据此说太阳的前向散射是对的。"
                        "按 Unreal 雾光 ≈ a·太阳单次 + b·天空单次 + c·Cycles 多次散射（含地面反射）拟合，a = 1.69、b = 0.29、c = 0.39，残差 19%，"
                        "三列相关 0.40–0.51：朝太阳方向的单次散射和多次散射分不开，c 那一列混着雾内多次散射和地面反射，也不能据此说 Lumen 提供了地面反射光。"
                        "推测主要缺体积多次散射（离太阳 55° 以外，Cycles 的雾光 64–85% 是多次散射）。没有关掉 Lumen 的渲染，没法验证；"
                        "天空的单次散射是否也缺，两次分析结论不一。",
                        f"白天三档画质误差相同，说明误差来自光照模型，不是体素精度。夜间对完整参考的整幅图：Epic {pct(n['unreal']['epic']['rel_l1'])}、"
                        f"High {pct(n['unreal']['high']['rel_l1'])}、Cinematic {pct(n['unreal']['cinematic']['rel_l1'])}（闭式解 {pct(n['image']['with_halo']['rel_l1'])}）。",
                        f"夜间：天空像素上雾光是 Cycles 单次散射的 3.8 倍、全部雾光的 {n['unreal_fog_ratio']['sky']:.1f} 倍；按相函数拟合，g = 0.8 分量的权重为 0，"
                        "接近各向同性——灯光在 Unreal 的体积雾里基本被均匀散射（白天离太阳 40° 内雾光总量却对得上，原因未查）。"
                        f"雾光总量 {n['unreal_fog_ratio']['all']:.2f} 看着不算差，其实是天空 {n['unreal_fog_ratio']['sky']:.1f} 倍、"
                        f"表面 {n['unreal_fog_ratio']['surfaces']:.2f} 倍相抵（对截断参考 {n['clamped']['unreal_fog_ratio']['all']:.2f} / "
                        f"{n['clamped']['unreal_fog_ratio']['sky']:.1f} / {n['clamped']['unreal_fog_ratio']['surfaces']:.2f}）。"
                        "表面上的雾光要用“有雾帧 − 透射率 × 无雾帧”估计，这个估计本身有偏，只有天空像素和整幅图的比较是干净的。",
                    ),
                ),
                ("优点", "最快（整帧约 1 ms），白天画质档位几乎不影响误差。"),
                (
                    "缺点",
                    "白天雾光整体少 31%，离太阳越远缺得越多，推测缺体积多次散射；夜间灯光散射近乎各向同性；参数要从源码换算，透射率没有独立核实。",
                ),
            ],
        )
    )

    parts.append(
        method(
            "5.11 学习",
            "m-learning",
            [
                (
                    "做法",
                    (
                        "两个 U-Net（约 47 万参数），用白天第 5–404 帧的 Cycles 参考训练（CPU 上约 2 分钟），在第 405–504 帧、浓雾 ×2 和夜间场景上测。"
                        "图像网络从无雾图 + 深度直接出有雾图；残差网络只学修正闭式解的图像。"
                    ),
                ),
                (
                    "数字",
                    (
                        f"<span class='facts'>GPU 推理 320×180 {sec(gpu['unet_320'])}，1080p {sec(gpu['unet_1920'])}。"
                        f"图像网络：训练集 {pct(images['image']['train']['rel_l1'])}，没见过的视角 {pct(images['image']['day_test']['rel_l1'])}，"
                        f"浓雾 {pct(images['image']['dense']['rel_l1'])}，夜间 {pct(images['image']['night']['rel_l1'])}（整幅图，对截断参考）。"
                        f"残差网络：没见过的视角 {pct(images['closed_form']['day_test']['rel_l1'])} → {pct(images['residual']['day_test']['rel_l1'])}，"
                        f"浓雾 {pct(images['closed_form']['dense']['rel_l1'])} → {pct(images['residual']['dense']['rel_l1'])}，"
                        f"夜间 {pct(images['closed_form']['night']['rel_l1'])} → {pct(images['residual']['night']['rel_l1'])}。</span>"
                    ),
                ),
                (
                    "误差来源",
                    ul(
                        "图像网络在训练集上就有 12.7%，是欠拟合（训练太短）；换场景后失败的成因没有分析。",
                        "残差网络到夜间把 87% 的像素压暗，最暗的十分之一压到 0.31 倍，而那里 Cycles 实际比闭式解亮 1.78 倍——它学到的修正只适用于白天。",
                        "白天测试帧上的闭式解是 10 月 2 日的旧版，测试帧为 405–504，不能和正文的 8.2% 直接比；浓雾和夜间用的是现版代码、但不带表面模型的闭式解，"
                        f"帧为 255 + 345，夜间对截断参考（带表面模型的现版对同一参考是 {pct(n['clamped']['image']['with_halo']['rel_l1'])}）。",
                    ),
                ),
                ("优点", "推理最快。"),
                ("缺点", "离开训练分布就失败；需要大量参考图；结果无法解释。"),
            ],
        )
    )
    parts.append(figure(images_["ood"], "图 9　学习方法在训练分布以外：浓雾和夜间点灯。", 980))
    parts.append("<h3>各方法的图像（第 345 帧）</h3>")
    parts.append(figure(images_["day"], "图 10　白天，整幅图像。", 980))
    parts.append(figure(images_["night"], "图 11　夜间，只有单次散射的雾光（提亮显示）。", 1100))
    parts.append(
        figure(
            images_["night_all"],
            "图 12　夜间，全部雾光（与图 11 同一亮度）。闭式解的灯晕大半在立方体的灯影里，Cycles 的光集中在灯下方的地面附近。Unreal 的雾光是“有雾帧 − 精确透射率 × 无雾帧”，能看到表面的残影：它对表面的衰减与精确值不同，所以夜间 Unreal 以整图和天空像素为准。",
            1100,
        )
    )
    return "\n".join(parts)


def main():
    S = Study()
    figs = {
        "time_error": fig_time_error(S),
        "own_error": fig_own_error(S),
        "cf_parts": fig_cf_parts(S),
        "azimuth": fig_azimuth(S),
        "night_single_maps": fig_night_single_maps(S),
        "night_multiple": fig_night_multiple(S),
        "cache": fig_cache(S),
    }
    (OUT / "report.html").write_text(render(S, figs, strips(S)), encoding="utf-8")
    print("report written", OUT / "report.html")


if __name__ == "__main__":
    main()
