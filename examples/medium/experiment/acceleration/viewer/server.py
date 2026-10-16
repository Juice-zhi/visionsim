"""Control page of the closed-form fog of :mod:`visionsim.medium`: switch the terms of its model on and off, and see its
error against Cycles, the time it takes and example images, all computed live by the library.

Run it from the root of the main checkout, where the renders are (``runs/fog-comparison``)::

    .conda/python.exe examples/medium/experiment/acceleration/viewer/server.py --port 8765 [--device auto] [--threads 8]

and open http://127.0.0.1:8765. The library is that given by ``--library``, else the one on ``PYTHONPATH`` if set, so
that a version under development, e.g. in a git worktree, is used by putting it first there, else that of the checkout
this server sits in (see ``library_path``). The switches are read from the JSON schema of its :class:`Effects
<visionsim.medium.model.Effects>` (names, descriptions, groups and prerequisites), and whether each one is effective
is found by its own ``Effects.missing``, so that new switches appear by themselves. A library without ``Effects`` is
served too, with its inputs only.

Times are those of warm calls, once what is kept from one frame to the next is computed, the fastest of ``--repeats``
(3 on the GPU, 1 on the CPU), and of the first call of each configuration, which may include such one-off work.

The study's scripts and renders are untracked, and live in the main checkout: they are found there (or given by
``--root``) when this folder sits in a git worktree.

Inputs, references and metrics are those of ``analysis/eval_closed_form.py`` and ``analysis/final_benchmark.py``, whose
modules (and ``common.py``) are imported rather than copied, at 320x180:

- ``day``: frames 255 to 262 and 345 of the day scene, with multiple scattering, shadow maps and surfaces lit through
  the fog, against Cycles 4096 spp with 16 volume bounces (``ms/cycles_ref``);
- ``night``: frames 255 and 345 of the night scene, a lamp behind a cube, with the ground's mean albedo and the same
  inputs, against Cycles 65536 spp with 16 volume bounces without clamping (``gpu/night_full_spp65536``).

Errors are those of ``common.errors`` (relative L1 over all pixels and channels, as is and once both images are blurred,
and the ratio of light) of whole frames, of the fog's light against Cycles' volume passes, and of surfaces (frames
minus volume passes), over frames 255 and 345 as in ``eval_closed_form.py``, over the frame shown, and over regions of
the image. The closed form runs through ``apply_medium_upsampled`` at the chosen factor, given frames without fog that
are either clean or rendered by Cycles at a few samples per pixel and denoised (``gpu/clear_<scene>_spp<N>``). The
page toggles the inputs of ``INPUTS``: ``Medium.multiple_scattering``, shadow maps and normals, and, for a library whose
``apply_medium`` takes it, the albedo of each pixel (Blender's diffuse color pass, ``clear/diffuse/color``, off by
default as in ``eval_closed_form.py``), which are also given to ``Effects``' helpers.

The closed form runs in one worker thread, from a queue in which the page's requests come before the warm-up of the
default configuration of both scenes, which starts in the background with the server: the night scene's first call
computes the halo of its lamp and the light that surfaces reflect, about 1.5-2.5 min on 4 CPU threads. A request of the
page replaces those it made before and no longer waits for, queued or running (between frames). Frames are kept in
memory, the least recently used going first, by their whole configuration, in which switches are resolved by
``Effects.resolve``: a switch whose prerequisites aren't met is computed exactly as a switch that is off, so that both
share their frames.

Images of the closed form are named by a hash of their configuration salted with a token drawn at start (``instance``),
so that a browser never shows an image that a server started earlier, maybe with another library, computed for the
same configuration; the page reloads itself when the token changes. ``fingerprint`` is a hash of the library's source,
the same for identical copies in different checkouts.

API, in JSON unless said otherwise:

- ``GET /api/schema``: switches, groups, conditions on the inputs, presets, scenes, devices, the library used, its
  fingerprint and the server's instance;
- ``GET /api/status``: what is loading or computing and for how long, the queue, the warm-up and the instance;
- ``POST /api/prerequisites`` (a configuration): what each switch misses, from ``Effects.missing``;
- ``POST /api/compute`` (a configuration): queues it unless its frames are known, and returns its state, with its
  result once done; the page polls it again until then;
- ``GET /api/image?id=<frame>&name=radiance|inscatter|surfaces``: an image of the closed form, as float16 (h, w, 3);
- ``GET /api/static?scene=&frame=&name=reference|reference_fog|reference_surfaces|input|albedo|cycles|cycles_noisy
  |cycles_fog&spp=&v=<instance>``: an image of the scene, as float16;
- ``GET /api/ablation?scene=&factor=&device=``: what each switch costs and the error it removes, from the ablations
  (``results/*ablation*.json``) of the same model as this library's first, see ``Lab.ablation``;
- ``GET /api/benchmark``: Cycles' curve of error against time, from the latest benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STUDY = Path("examples/medium/experiment/acceleration")
"""Folder of the study's scripts, relative to the root of the checkout"""
EVAL_FRAMES = (255, 345)
"""Frames over which errors are measured, as in ``eval_closed_form.py``"""
SEQUENCE = tuple(range(255, 263))
"""Consecutive frames of the day scene, over which flicker is measured, as in ``final_benchmark.py``"""
FACTORS = (1, 2, 4)
WITH_FOG = {"day": "day_spp{}", "night": "night_full_spp{}"}
"""Folders of Cycles' renders with fog at a few samples per pixel, in ``acceleration/gpu``, by scene"""
CLEAR = {"day": "clear_day_spp{}", "night": "clear_night_spp{}"}
"""Folders of Cycles' renders without fog at a few samples per pixel, given to the closed form, by scene"""
INPUTS: dict[str, dict[str, Any]] = {
    "multiple_scattering": {
        "label": "Multiple scattering",
        "description": "Medium.multiple_scattering: the medium scatters light more than once, which gives its glow, "
        "the halos of lamps, and the light of the ground scattered on its way up.",
        "default": True,
        "conditions": ["multiple_scattering"],
    },
    "shadow_maps": {
        "label": "Shadow maps",
        "description": "Shadow maps of the scene are given (the inputs 'shadows' and 'occlusion' of Effects), through "
        "which objects cast shadows onto the medium and onto surfaces, and hide part of the sky from them.",
        "default": True,
        "conditions": ["shadows", "occlusion"],
    },
    "normals": {
        "label": "Surfaces lit through the fog",
        "description": "The normals of surfaces are given (the input 'normals' of Effects), through which surfaces are "
        "lit through the medium, which dims their light while its glow lights them. Without them, surfaces keep the "
        "light they were rendered with, attenuated.",
        "default": True,
        "conditions": ["normals"],
    },
}
"""Inputs of the closed form that the page toggles, which describe the scene rather than choose terms of the model:
their label and description, whether they are given by default, and the conditions of ``EFFECT_INPUTS`` they satisfy,
by which the page tells which input to turn on for a switch that misses one"""
ALBEDO = Path("clear/diffuse/color")
"""Blender's diffuse color pass of the frames without fog, relative to ``common.ROOT``: the albedo of the surface that
each pixel sees, which both scenes share as they share their geometry and camera"""
ALBEDO_INPUT: dict[str, Any] = {
    "label": "Albedo of each pixel",
    "description": "The albedo of the surface each pixel sees is given (apply_medium's albedo, the input 'albedo' of "
    "Effects), from Blender's diffuse color pass (clear/diffuse/color): the irradiance that surfaces received is then "
    "measured from it rather than estimated from the mean albedos of the ground and objects. It is used with the "
    "normals, where surfaces are lit through the fog. Off by default, as in eval_closed_form.py.",
    "default": False,
    "conditions": ["albedo"],
}
"""The optional input of a library whose ``apply_medium`` takes the albedo of each pixel"""
NIGHT_HINT = (
    "the first call of a configuration may compute the halo of the lamp and the light that surfaces reflect, "
    "about 1.5-2.5 min on 4 CPU threads, 6-15 s on the GPU"
)
"""Shown while the first frame of a configuration of the night scene is computed: the library keeps halos by the
switches that change them, so that toggling one of those rebuilds the halo once"""


class BadRequest(Exception):
    """Configuration or query that can't be served, whose message is shown on the page."""


class NotFound(Exception):
    """Image or render that doesn't exist."""


def flag(value: Any, what: str) -> bool:
    """A JSON boolean of a request, which is never guessed from another type (``bool("false")`` is true)."""
    if isinstance(value, bool):
        return value
    raise BadRequest(f"{what} is true or false, not {value!r}.")


def integer(value: Any, what: str) -> int:
    """A whole number of a request, given as a JSON number or a string of digits."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    raise BadRequest(f"{what} {value!r} isn't a whole number.")


def accepts(fn: Callable, name: str) -> bool:
    """Whether a function takes an argument of this name, rather than through ``**kwargs``."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def fingerprint(library: str | Path) -> str:
    """Hash of the source of a visionsim library, its Python modules by their path in the package, the same for
    identical copies in different checkouts whatever their line endings."""
    digest = hashlib.sha1()
    package = Path(library) / "visionsim"
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


def clean(value: Any) -> Any:
    """Value made of JSON types only, with NaN and infinities as None."""
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def masked_errors(
    ours: np.ndarray, theirs: np.ndarray, ours_blurred: np.ndarray, theirs_blurred: np.ndarray, mask: np.ndarray
) -> dict[str, float] | None:
    """``common.errors`` over the pixels of a mask of shape (n, h, w), given both images of shape (n, h, w, 3) and both
    already blurred, which saves blurring them again for every region; None where the reference has no light."""
    keep = np.broadcast_to(mask[..., None], ours.shape)
    total = float(theirs[keep].sum())
    if not keep.any() or total <= 0:
        return None
    return {
        "ratio": float(ours[keep].sum() / total),
        "rel_l1": float(np.abs(ours - theirs)[keep].sum() / total),
        "rel_l1_blurred": float(np.abs(ours_blurred - theirs_blurred)[keep].sum() / total),
    }


def modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def render_device(log: Path) -> str | None:
    """Device that Blender rendered on, as its log names it (``Activated device NVIDIA GeForce RTX 5080, OPTIX``), or
    None if there is no such log or line."""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    devices = [line.split("Activated device", 1)[1].strip() for line in text.splitlines() if "Activated device" in line]
    return " + ".join(dict.fromkeys(devices)) or None


SAME_MODEL = 1e-5
"""Largest difference of relative L1 errors, as fractions, between an ablation's every switch on and this library's
default configuration for both to be the same model: devices differ by about 2e-7, versions of the model by far more"""


def same_model(case: Mapping, baseline: Mapping | None) -> str | None:
    """Whether an ablation of a scene is of the same model as this library's: "same" if its every switch on at factor 1
    gives the errors of the default configuration (``baseline``), whole frames, fog and surfaces, "other" if not, and
    None if either is unknown."""
    on = (((case.get("factors") or {}).get("1") or {}).get("variants") or {}).get("all on")
    if baseline is None or not isinstance(on, Mapping):
        return None
    try:
        close = all(abs(on[k]["rel_l1"] - baseline[k]["rel_l1"]) <= SAME_MODEL for k in ("whole", "fog", "surfaces"))
    except (KeyError, TypeError):
        return None
    return "same" if close else "other"


def main_checkout(start: Path) -> Path | None:
    """Root of the main checkout of the git repository that holds a folder, also from a linked worktree, whose ``.git``
    file points into the main checkout's ``.git``."""
    for folder in [start, *start.parents]:
        git = folder / ".git"
        if git.is_dir():
            return folder
        if git.is_file():
            text = git.read_text().strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.removeprefix("gitdir:").strip())
                common = gitdir / "commondir"
                if common.is_file():
                    return (gitdir / common.read_text().strip()).resolve().parent
            return folder
    return None


def library_path(explicit: str | None) -> Path | None:
    """Folder of the visionsim library to put first on the path: given, else none if ``PYTHONPATH`` is set (which
    then decides), else that of the checkout this folder sits in, so that a server run from a git worktree uses the
    worktree's library, and one run from the main checkout its own, as installed."""
    if explicit:
        if not (Path(explicit) / "visionsim" / "__init__.py").is_file():
            raise SystemExit(f"No visionsim package in {explicit}.")
        return Path(explicit).resolve()
    own = HERE.parents[4]
    if os.environ.get("PYTHONPATH") or not (own / "visionsim" / "__init__.py").is_file():
        return None
    return own


def study_root(explicit: str | None) -> Path:
    """Root of the checkout that holds the study's scripts and renders: given, this one, or the main checkout of the
    worktree this folder sits in."""
    candidates = [Path(explicit)] if explicit else [HERE.parents[4], main_checkout(HERE)]
    for candidate in candidates:
        if (
            candidate
            and (candidate / "runs" / "fog-comparison").is_dir()
            and (candidate / STUDY / "common.py").is_file()
        ):
            return candidate.resolve()
    raise SystemExit(f"No checkout with runs/fog-comparison and {STUDY}/common.py among {candidates}; pass --root.")


# ----------------------------------------------------------------------------------------------------------------------
# Scenes


class SceneData:
    """Inputs of the closed form for the frames of a scene on the CPU, as a ``final_benchmark.Case``, Cycles'
    references, which of its renders at a few samples per pixel exist, and masks of regions of its frames."""

    def __init__(self, lab: Lab, name: str, case: Any, fog: np.ndarray, reference: str, with_fog: str, clear: str):
        self.name, self.case, self.fog = name, case, fog
        self.frames: tuple[int, ...] = tuple(case.frames)
        self.eval_index = [self.frames.index(f) for f in EVAL_FRAMES]
        self.reference_label = reference
        self.with_fog, self.clear_pattern = with_fog, clear
        gpu = lab.fb.GPU
        spp = sorted(int(p.name.rsplit("spp", 1)[1]) for p in gpu.glob(with_fog.format("*")) if p.is_dir())
        self.cycles_spp = [n for n in spp if n < 4096]  # the night's reference isn't one of the variants
        self.clear_spp = sorted(int(p.name.rsplit("spp", 1)[1]) for p in gpu.glob(clear.format("*")) if p.is_dir())
        self.blurred_reference = lab.common.blur(np.asarray(case.reference, dtype=np.float64))
        self.blurred_fog = lab.common.blur(np.asarray(fog, dtype=np.float64))
        self.regions, self.region_info = regions(lab, self)
        self.albedo: np.ndarray | None = None
        """albedo of each pixel of each frame, of shape (n, h, w, 3), if the library takes it and it was rendered"""
        self.unavailable: dict[str, str] = {}
        """inputs of the library that this scene can't give, with the reason"""
        if "albedo" in lab.inputs:
            path = lab.common.ROOT / ALBEDO
            try:
                self.albedo = lab.common.load_images(path, self.frames)
            except Exception as error:
                self.unavailable["albedo"] = f"no diffuse color pass of frames {list(self.frames)} in {path.as_posix()}"
                print(f"The {name} scene has no albedo of each pixel: {error!r}", flush=True)

    def describe(self) -> dict:
        return {
            "frames": list(self.frames),
            "eval_frames": list(EVAL_FRAMES),
            "sequence": [f for f in SEQUENCE if f in self.frames] if self.name == "day" else [],
            "reference": self.reference_label,
            "cycles_spp": self.cycles_spp,
            "clear_spp": self.clear_spp,
            "regions": self.region_info,
            "lamps": len(self.case.lighting.lamps),
            "suns": len(self.case.lighting.suns),
            "size": list(self.case.reference.shape[1:3]),
            "unavailable_inputs": self.unavailable,
        }


def regions(lab: Lab, data: SceneData) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    """Masks of regions of each frame of a scene, of shape (n, h, w), and their labels.

    Sky: pixels that see no surface. Ground: pixels whose normal points up, split by day into the ground that the sun
    lights and the ground in the shadow of an object, as the sun's shadow map tells (as ``analysis/day_regions.py``).
    Objects: the other surfaces. At night, the pixels whose fog lies in the shadow that the cube casts from the lamp, as
    ``eval_closed_form.py`` finds them from the reference's direct volume pass, over frames 255 and 345, and the rest.
    """
    common, case = lab.common, data.case
    raw = common.load_images(common.ROOT / "clear" / "depths", data.frames)[..., 0]
    sky = raw >= 1e9
    ground = (case.normals.cpu().numpy()[..., 2] > 0.95) & ~sky
    masks: dict[str, np.ndarray] = {"sky": sky}
    info = [{"name": "sky", "label": "Sky", "description": "pixels that see no surface"}]
    whole_ground = {"name": "ground", "label": "Ground", "description": "pixels whose normal points up"}
    if data.name == "day":
        try:
            visible = np.stack([sun_visibility(lab, case.scene, i) for i in range(len(data.frames))])
            masks["lit_ground"] = ground & (visible >= 0.5)
            masks["shadowed_ground"] = ground & (visible < 0.5)
            info += [
                {"name": "lit_ground", "label": "Sunlit ground", "description": "ground that the sun's shadow map sees"},
                {
                    "name": "shadowed_ground",
                    "label": "Shadowed ground",
                    "description": "ground in the shadow of an object, as the sun's shadow map tells",
                },
            ]
        except Exception as error:
            print(f"The ground isn't split by the sun's shadows: {error!r}", flush=True)
            masks["ground"] = ground
            info.append(whole_ground)
    else:
        masks["ground"] = ground
        info.append(whole_ground)
    masks["objects"] = ~sky & ~ground
    info.append({"name": "objects", "label": "Objects", "description": "surfaces other than the ground"})
    if data.name == "night":
        direct = common.load_images(lab.fb.NIGHT_REFERENCE / "volume" / "direct", data.frames)
        luma = direct.mean(axis=-1)
        shadow = luma < 0.02 * np.median(luma[luma > 0])
        masks["lamp_shadow"], masks["lamp_lit"] = shadow, ~shadow
        info += [
            {
                "name": "lamp_shadow",
                "label": "Cube's lamp shadow",
                "description": "pixels whose fog lies in the shadow that the cube casts from the lamp, where Cycles' "
                "direct volume pass is under 2% of its median, as eval_closed_form.py",
            },
            {"name": "lamp_lit", "label": "Outside it", "description": "the other pixels"},
        ]
    return masks, info


def sun_visibility(lab: Lab, scene: Any, i: int) -> np.ndarray:
    """Visible fraction of the sun from the point that each pixel of frame i sees, of shape (h, w), from the map of
    the sun that faces it best, as ``common.DayLight.sun_visibility``."""
    torch, common = lab.torch, lab.common
    occlusion = common.occlusion_to(scene.occlusion, device="cpu", dtype=torch.float32)
    towards = torch.as_tensor(scene.lighting.suns[0].direction, dtype=torch.float32)
    towards = towards / towards.norm()
    maps = common.select_maps(occlusion.sun_maps, int((occlusion.sun_directions @ towards).argmax()))
    t = scene.transforms[i]
    origin, directions, scale = lab.camera_rays(t, t["transform_matrix"], dtype=torch.float32)
    distance = torch.as_tensor(scene.depth[i], dtype=torch.float32) * scale
    points = origin + directions.reshape(-1, 3) * distance.reshape(-1, 1)
    seen = common.visibility(maps, torch.zeros(3), points, torch.ones_like(points[:, :1]))[:, 0, 0]
    return seen.numpy().reshape(scene.depth[i].shape)


# ----------------------------------------------------------------------------------------------------------------------
# Jobs


class FrameEntry:
    """A frame of the closed form for a configuration, and how long it took."""

    counter = itertools.count()

    def __init__(self, radiance: np.ndarray, inscatter: np.ndarray, seconds: float, warm: float | None):
        self.serial = next(FrameEntry.counter)
        """number of this computation of the frame, which tells a frame computed again once forgotten, whose time
        differs, from the one a result was made of"""
        self.radiance, self.inscatter = radiance, inscatter
        self.seconds = seconds
        """time of the call that computed it"""
        self.warm = warm
        """time of the fastest call once what is kept from one frame to the next was computed, if measured"""
        self.timed = 0 if warm is None else 1
        """number of such calls"""


class Job:
    """Frames of a configuration to compute, the selected frame last, so that its time is that of a warm call."""

    counter = itertools.count()

    def __init__(self, identity: tuple, config: dict, frames: list[int], origin: str):
        self.identity, self.config, self.frames, self.origin = identity, config, frames, origin
        self.priority = 0 if origin == "page" else 1
        self.order = next(Job.counter)
        self.state, self.stage = "queued", ""
        self.created = time.time()
        self.started: float | None = None
        self.cancelled = False
        """set when the page no longer waits for it: it stops before its next frame, keeping those computed"""

    def describe(self) -> dict:
        c = self.config
        return {
            "scene": c["scene"],
            "frame": c["frame"],
            "frames": self.frames,
            "origin": self.origin,
            "state": self.state,
            "stage": self.stage,
            "device": c["device"],
            "factor": c["factor"],
            "waited": time.time() - self.created,
            "elapsed": None if self.started is None else time.time() - self.started,
        }


class LRU(OrderedDict):
    """Mapping that keeps its ``size`` most recently used items."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


# ----------------------------------------------------------------------------------------------------------------------
# The library, the scenes, the queue and the caches


class Lab:
    def __init__(self, args: argparse.Namespace, root: Path):
        self.args, self.root = args, root
        self.threads = args.threads
        import final_benchmark
        import torch
        import visionsim
        from visionsim.medium import apply_medium, camera_rays
        from visionsim.medium.upsampling import apply_medium_upsampled

        import common

        # common.py sets its own number of threads when imported
        torch.set_num_threads(self.threads)
        self.torch, self.common, self.fb = torch, common, final_benchmark
        self.camera_rays, self.apply = camera_rays, apply_medium_upsampled
        self.library = str(Path(visionsim.__file__).resolve().parent.parent)
        self.fingerprint = fingerprint(self.library)
        self.instance = uuid.uuid4().hex[:12]
        """token of this run of the server, which salts the names of images, see the module's docstring"""
        self.effects_api = self.load_effects()
        self.inputs = dict(INPUTS)
        """inputs that the page toggles, with the albedo of each pixel if this library takes it, which is offered once
        both ``apply_medium`` and ``apply_medium_upsampled`` name it (the latter subsamples it with the radiance)"""
        if accepts(apply_medium, "albedo") and accepts(apply_medium_upsampled, "albedo"):
            self.inputs["albedo"] = ALBEDO_INPUT
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        if args.device == "auto":
            self.device = "cuda" if "cuda" in self.devices else "cpu"
        elif args.device not in self.devices:
            raise SystemExit(f"Device {args.device} isn't available here, only {self.devices}.")
        else:
            self.device = args.device
        self.device_names = {"cpu": f"CPU, {self.threads} threads"}
        if "cuda" in self.devices:
            self.device_names["cuda"] = torch.cuda.get_device_name(0)

        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.scenes: dict[str, SceneData] = {}
        self.scene_errors: dict[str, str] = {}
        self.loading = "starting"
        self.cases: dict[tuple[str, str], Any] = {}
        self.clear_frames: dict[tuple[str, str, int], Any] = {}
        self.albedos: dict[tuple[str, str], Any] = {}
        self.frames = LRU(args.cache_frames)
        self.ids: dict[str, str] = {}
        """frame keys by the short id by which the page asks for images"""
        self.first_calls: dict[str, tuple[int, float]] = {}
        """frame and time of the first call of each configuration, which may include work kept for the next frames"""
        self.results = LRU(64)
        self.baselines: dict[str, dict] = {}
        self.static = LRU(160)
        self.queue: list[Job] = []
        self.jobs: dict[tuple, Job] = {}
        self.running: Job | None = None
        self.failed: dict[tuple, str] = {}
        self.warmup: dict[str, dict] = {}
        self.started = time.time()

    # ------------------------------------------------------------------------------------------------------------------
    # The library's switches

    def load_effects(self) -> dict:
        """Switches of the library's ``Effects``, read from its JSON schema, and its vocabulary of groups and conditions,
        or none for a library that has no switches."""
        empty: dict[str, Any] = {
            "cls": None,
            "switches": [],
            "names": set(),
            "groups": {},
            "conditions": {},
            "given": [],
            "presets": {},
        }
        try:
            import visionsim.medium as medium_module
            from visionsim.medium import Effects
        except ImportError:
            return empty
        schema = Effects.model_json_schema()
        switches = []
        for name, field in schema.get("properties", {}).items():
            if field.get("type") != "boolean":
                continue  # only switches, should the model gain other options
            switches.append(
                {
                    "name": name,
                    "title": field.get("title", name),
                    "description": field.get("description", ""),
                    "group": field.get("group", "other"),
                    "requires": list(field.get("requires") or []),
                    "requires_inputs": list(field.get("requires_inputs") or []),
                    "default": bool(field.get("default", True)),
                }
            )
        model_module = sys.modules.get(Effects.__module__)

        def vocabulary(name: str, default: Any) -> Any:
            return getattr(medium_module, name, None) or getattr(model_module, name, None) or default

        groups = dict(vocabulary("EFFECT_GROUPS", {}))
        for switch in switches:
            groups.setdefault(switch["group"], "")
        names = {s["name"] for s in switches}
        return {
            "cls": Effects,
            "switches": switches,
            "names": names,
            "groups": groups,
            "conditions": dict(vocabulary("EFFECT_INPUTS", {})),
            "given": list(vocabulary("EFFECT_GIVEN", ())),
            "presets": presets(Effects, names, (medium_module, model_module, Effects)),
            "helpers": all(hasattr(Effects, f) for f in ("missing", "effective", "resolve")),
        }

    def given(self, config: dict) -> dict[str, bool]:
        """What is given with frames, by the names of ``EFFECT_GIVEN``, leaving out those the library doesn't know (a
        library without ``EFFECT_GIVEN`` is given the inputs it had before it, shadow maps and normals)."""
        inputs = config["inputs"]
        maps = inputs["shadow_maps"]
        values = {
            "shadows": maps,
            "occlusion": maps,
            "normals": inputs["normals"],
            "albedo": inputs.get("albedo", False),
        }
        known = self.effects_api["given"] or ("shadows", "occlusion", "normals")
        return {k: v for k, v in values.items() if k in known}

    def medium(self, config: dict) -> Any:
        case = self.scenes[config["scene"]].case
        return case.medium.model_copy(update={"multiple_scattering": config["inputs"]["multiple_scattering"]})

    def prerequisites(self, config: dict) -> dict:
        """What each switch misses for a configuration, which makes it ineffective, from the library's helper."""
        api = self.effects_api
        if api["cls"] is None or not api.get("helpers"):
            return {"available": False, "missing": {}, "effective": dict(config["effects"]), "conditions": {}}
        effects = api["cls"](**config["effects"])
        medium, lighting = self.medium(config), self.scenes[config["scene"]].case.lighting
        given = self.given(config)
        try:
            conditions = api["cls"].conditions(medium, lighting, **given) if hasattr(api["cls"], "conditions") else {}
            missing = effects.missing(medium, lighting, **given)
            effective = effects.effective(medium, lighting, **given)
        except TypeError as error:  # helpers whose signature this page doesn't know: nothing is greyed out
            reason = f"Effects' helpers can't be called as expected: {error}"
            return {"available": False, "reason": reason, "missing": {}, "effective": dict(config["effects"])}
        return {"available": True, "missing": missing, "effective": effective, "conditions": conditions}

    def resolved(self, config: dict) -> dict[str, bool]:
        """Switches as the library computes them, those whose prerequisites aren't met off."""
        api = self.effects_api
        if api["cls"] is None:
            return {}
        if not api.get("helpers"):
            return dict(config["effects"])
        effects = api["cls"](**config["effects"])
        lighting = self.scenes[config["scene"]].case.lighting
        try:
            resolved = effects.resolve(self.medium(config), lighting, **self.given(config))
        except TypeError:  # see prerequisites: switches are then keyed as they are
            return dict(config["effects"])
        return {name: bool(getattr(resolved, name)) for name in config["effects"]}

    # ------------------------------------------------------------------------------------------------------------------
    # Configurations

    def normalize(self, raw: Any) -> dict:
        """Configuration with every field checked and filled with its default: every switch on, every input as
        ``INPUTS`` says (all on but the albedo of each pixel), factor 1, clean frames without fog, and the server's
        device."""
        if not isinstance(raw, Mapping):
            raise BadRequest("A configuration is a JSON object.")
        scene = raw.get("scene", "day")
        if not isinstance(scene, str) or scene not in self.scenes:
            known = isinstance(scene, str) and scene in self.scene_errors
            reason = self.scene_errors[scene] if known else "it isn't loaded yet" if self.loading else "no such scene"
            raise BadRequest(f"Scene {scene!r} isn't available: {reason}.")
        data = self.scenes[scene]
        frame = integer(raw.get("frame", EVAL_FRAMES[0]), "Frame")
        if frame not in data.frames:
            raise BadRequest(f"Frame {frame} isn't among the frames of the {scene} scene, {list(data.frames)}.")
        # Left out or null: the defaults; anything but an object of booleans is refused rather than guessed at
        effects_in = {} if raw.get("effects") is None else raw["effects"]
        inputs_in = {} if raw.get("inputs") is None else raw["inputs"]
        if not isinstance(effects_in, Mapping) or not isinstance(inputs_in, Mapping):
            raise BadRequest("Effects and inputs are JSON objects of switches, by name.")
        # Names this library doesn't know, e.g. from a page made for another library, are left out
        effects = {
            s["name"]: flag(effects_in.get(s["name"], s["default"]), f"Switch {s['name']}")
            for s in self.effects_api["switches"]
        }
        inputs = {
            name: flag(inputs_in.get(name, about["default"]), f"Input {name}") for name, about in self.inputs.items()
        }
        for name, reason in data.unavailable.items():
            if inputs.get(name):
                raise BadRequest(f"Input {name} isn't available in the {scene} scene: {reason}.")
        factor = integer(raw.get("factor", 1), "Factor")
        if factor not in FACTORS:
            raise BadRequest(f"The factor of the upsampling is one of {list(FACTORS)}, not {factor}.")
        clear = "clean" if raw.get("clear") is None else raw["clear"]
        if clear != "clean":
            try:
                clear = integer(clear, "Samples")
            except BadRequest:
                raise BadRequest(f"Frames without fog are 'clean' or a number of samples, not {clear!r}.") from None
            if clear not in data.clear_spp:
                raise BadRequest(f"Cycles' frames without fog at {clear} spp weren't rendered, only {data.clear_spp}.")
        device = self.device if raw.get("device") is None else raw["device"]
        if not isinstance(device, str) or device not in self.devices:
            raise BadRequest(f"Device {device!r} isn't available here, only {self.devices}.")
        return {
            "scene": scene,
            "frame": frame,
            "effects": effects,
            "inputs": inputs,
            "factor": factor,
            "clear": clear,
            "device": device,
        }

    def config_key(self, config: dict) -> str:
        """Key of the frames of a configuration, the frame aside, with its switches as the library computes them."""
        parts = [config["scene"], self.resolved(config), config["inputs"], config["factor"], config["clear"]]
        return json.dumps([*parts, config["device"]], sort_keys=True)

    @staticmethod
    def frame_key(config_key: str, frame: int) -> str:
        return f"{config_key}#{frame}"

    def needed(self, config: dict, flicker: bool) -> list[int]:
        """Frames to compute, the selected one last, so that it is timed once the configuration is warm."""
        data = self.scenes[config["scene"]]
        frames = list(EVAL_FRAMES) + (list(SEQUENCE) if flicker and config["scene"] == "day" else [])
        frames = [f for f in dict.fromkeys(frames) if f in data.frames and f != config["frame"]]
        return frames + [config["frame"]]

    def repeats(self, device: str) -> int:
        """Warm calls of which the fastest is the time of a frame: a few on the GPU, which cost little and whose times
        vary from call to call, and one on the CPU, unless ``--repeats`` says otherwise."""
        return self.args.repeats or (3 if device.startswith("cuda") else 1)

    def missing_work(self, config: dict, key: str, frames: list[int]) -> bool:
        if any(self.frames.get(self.frame_key(key, f)) is None for f in frames):
            return True
        return self.frames.get(self.frame_key(key, config["frame"])).timed < self.repeats(config["device"])

    # ------------------------------------------------------------------------------------------------------------------
    # Requests

    def request(self, raw: Any, origin: str = "page", supersede: bool = False, retry: bool = False) -> dict:
        """State of a configuration, which is queued if its frames aren't known, with its result once they are.

        With ``supersede``, the page no longer waits for any other configuration it asked for: those still queued are
        dropped, and one that is running stops before its next frame, whether this one is computed, queued, or already
        known (e.g. when the page goes back to a configuration it saw)."""
        if self.loading:
            return {"state": "loading", "stage": self.loading}
        flicker = flag(raw.get("flicker", False), "flicker") if isinstance(raw, Mapping) else False
        config = self.normalize(raw)
        prerequisites = self.prerequisites(config)
        key = self.config_key(config)
        frames = self.needed(config, flicker)
        identity = (key, config["frame"], flicker)
        with self.lock:
            if supersede and origin == "page":
                self.supersede(identity)
            if not self.missing_work(config, key, frames):
                entries = {f: self.frames.get(self.frame_key(key, f)) for f in frames}
                first = self.first_calls.get(key)
            else:
                if identity in self.failed and not retry:
                    error = self.failed[identity]
                    return {"state": "error", "error": error, "config": config, "prerequisites": prerequisites}
                self.failed.pop(identity, None)
                job = self.jobs.get(identity)
                if job is not None and job.cancelled:  # replaced a moment ago, and asked for again since
                    job.cancelled = False
                if job is None:
                    job = Job(identity, config, frames, origin)
                    self.queue.append(job)
                    self.jobs[identity] = job
                    self.wake.notify()
                elif origin == "page":
                    job.priority = 0  # e.g. the warm-up of the configuration that the page now waits for
                ahead = [j for j in self.queue if (j.priority, j.order) < (job.priority, job.order)]
                return {
                    "state": job.state,
                    "job": job.describe(),
                    "ahead": len(ahead) + (1 if self.running is not None and self.running is not job else 0),
                    "config": config,
                    "prerequisites": prerequisites,
                }
        result = self.result(config, key, flicker, entries, first)
        return {"state": "done", "config": config, "prerequisites": prerequisites, "result": result}

    def supersede(self, identity: tuple) -> None:
        """Drop what the page asked for before, other than a configuration it now waits for, with the lock held: its
        queued jobs go, the running one stops before its next frame, and warm-ups it had put first go back in line."""
        for job in list(self.queue):
            if job.identity == identity:
                continue
            if job.origin == "page":
                self.queue.remove(job)
                self.jobs.pop(job.identity, None)
            else:
                job.priority = 1
        running = self.running
        if running is not None and running.origin == "page" and running.identity != identity:
            running.cancelled = True

    def result(self, config: dict, key: str, flicker: bool, entries: dict[int, FrameEntry], first) -> dict:
        """Errors, times and images of a configuration, given its frames."""
        # Results are kept by the very frames they were made of, and the number of warm calls timed: frames forgotten
        # and computed again give the same errors, but times of their own
        selected = entries[config["frame"]]
        made_of = tuple(sorted((f, e.serial) for f, e in entries.items()))
        identity = (key, config["frame"], flicker, made_of, selected.timed, first)
        cached = self.results.get(identity)
        if cached is not None:
            return cached | {"baseline": self.baseline(config)}
        data = self.scenes[config["scene"]]
        evaluation = [entries[f] for f in EVAL_FRAMES]
        index = [data.frames.index(config["frame"])]
        metrics = {
            "eval": self.measure(
                data, data.eval_index, [e.radiance for e in evaluation], [e.inscatter for e in evaluation]
            ),
            "frame": self.measure(data, index, [selected.radiance], [selected.inscatter]),
        }
        metrics["eval"]["frames"], metrics["frame"]["frames"] = list(EVAL_FRAMES), [config["frame"]]
        timing: dict[str, Any] = {
            "device": config["device"],
            "device_name": self.device_names[config["device"]],
            "threads": self.threads if config["device"] == "cpu" else None,
            "factor": config["factor"],
            "frame": config["frame"],
            "warm": selected.warm,
            "repeats": selected.timed,
            "first": None if first is None else {"frame": first[0], "seconds": first[1]},
            # The first call of a configuration may compute what is kept for the next frames (tables, halos, the
            # light that surfaces reflect, and on the GPU its kernels), which shows as a much longer call
            "one_off": first is not None and selected.warm is not None and first[1] > 1.3 * selected.warm + 0.02,
        }
        result: dict[str, Any] = {
            "id": self.frame_id(self.frame_key(key, config["frame"])),
            "frame": config["frame"],
            "metrics": metrics,
            "timing": timing,
            "resolved": self.resolved(config),
            "cycles": self.cycles_section(config, metrics["eval"], selected.warm),
        }
        if flicker and config["scene"] == "day":
            sequence = [f for f in SEQUENCE if f in data.frames]
            images = np.stack([entries[f].radiance for f in sequence])
            reference = data.case.reference[[data.frames.index(f) for f in sequence]]
            temporal = self.common.flicker(images, reference)
            result["flicker"] = {
                "frames": sequence,
                "closed_form": temporal,
                **self.cycles_flicker(temporal["temporal"]),
            }
        result = clean(result)
        self.results.put(identity, result)
        return result | {"baseline": self.baseline(config)}

    def measure(self, data: SceneData, index: list[int], radiance: list, inscatter: list) -> dict:
        """Errors of whole frames, of the fog and of surfaces, overall as ``eval_closed_form.py``, and per region."""
        errors = self.common.errors
        frames, fog = np.stack(radiance), np.stack(inscatter)
        reference, reference_fog = data.case.reference[index], data.fog[index]
        result: dict[str, Any] = {
            "whole": errors(frames, reference),
            "fog": errors(fog, reference_fog),
            "surfaces": errors(frames - fog, reference - reference_fog),
        }
        ours = {"whole": frames, "fog": fog, "surfaces": frames - fog}
        theirs = {"whole": reference, "fog": reference_fog, "surfaces": reference - reference_fog}
        theirs_blurred = {
            "whole": data.blurred_reference[index],
            "fog": data.blurred_fog[index],
            "surfaces": data.blurred_reference[index] - data.blurred_fog[index],
        }
        ours64 = {kind: np.asarray(images, dtype=np.float64) for kind, images in ours.items()}
        theirs64 = {kind: np.asarray(images, dtype=np.float64) for kind, images in theirs.items()}
        blurred = {kind: self.common.blur(images) for kind, images in ours64.items()}
        regions: dict[str, Any] = {}
        for name, mask in data.regions.items():
            masks = mask[index]
            regions[name] = {"pixels": float(masks.mean())}
            for kind in ours:
                regions[name][kind] = masked_errors(
                    ours64[kind], theirs64[kind], blurred[kind], theirs_blurred[kind], masks
                )
        result["regions"] = regions
        return result

    def baseline(self, config: dict) -> dict | None:
        """Errors of the default configuration of the scene (every switch and input on, factor 1, clean frames), on
        any device, if known, against which the page shows differences."""
        data = self.scenes[config["scene"]]
        if config["scene"] in self.baselines:
            return self.baselines[config["scene"]]
        for device in [config["device"]] + [d for d in self.devices if d != config["device"]]:
            key = self.config_key(self.normalize({"scene": config["scene"], "device": device}))
            with self.lock:
                entries = [self.frames.get(self.frame_key(key, f)) for f in EVAL_FRAMES]
            if all(e is not None for e in entries):
                frames = np.stack([e.radiance for e in entries])
                fog = np.stack([e.inscatter for e in entries])
                reference, reference_fog = data.case.reference[data.eval_index], data.fog[data.eval_index]
                errors = self.common.errors
                self.baselines[config["scene"]] = clean(
                    {
                        "device": device,
                        "whole": errors(frames, reference),
                        "fog": errors(fog, reference_fog),
                        "surfaces": errors(frames - fog, reference - reference_fog),
                    }
                )
                return self.baselines[config["scene"]]
        return None

    def frame_id(self, frame_key: str) -> str:
        """Short name of a frame of a configuration, by which the page asks for its images, unique to this run of the
        server: the browser keeps images by their address, which a server started later, maybe with another library,
        must not share for the same configuration."""
        short = hashlib.sha1(f"{self.instance}#{frame_key}".encode()).hexdigest()[:16]
        with self.lock:
            self.ids[short] = frame_key
        return short

    # ------------------------------------------------------------------------------------------------------------------
    # Worker

    def case(self, scene: str, device: str) -> Any:
        """``final_benchmark.Case`` of a scene, with its inputs on a device."""
        key = (scene, device)
        if key not in self.cases:
            c = self.scenes[scene].case
            if device == "cpu":
                self.cases[key] = c
            else:
                clear, normals = c.clear.cpu().numpy(), c.normals.cpu().numpy()
                args = (c.name, c.frames, c.scene, c.lighting, c.medium, clear, normals, c.reference, device)
                self.cases[key] = self.fb.Case(*args, c.noise_mse)
        return self.cases[key]

    def clear(self, config: dict) -> Any:
        """Frames without fog of a configuration, on its device."""
        case = self.case(config["scene"], config["device"])
        if config["clear"] == "clean":
            return case.clear
        key = (config["scene"], config["device"], config["clear"])
        if key not in self.clear_frames:
            self.clear_frames[key] = case.clear_frames(config["clear"])
        return self.clear_frames[key]

    def albedo(self, scene: str, device: str) -> Any:
        """Albedo of each pixel of the frames of a scene, on a device."""
        key = (scene, device)
        if key not in self.albedos:
            self.albedos[key] = self.case(scene, device).on_device(self.scenes[scene].albedo)
        return self.albedos[key]

    def run_frame(self, config: dict, frame: int) -> tuple[np.ndarray, np.ndarray, float]:
        """The closed form of a frame, its radiance and in-scattering of shape (h, w, 3), and the time of the call, with
        the inputs already on the device, as ``final_benchmark.Case.run``."""
        torch, device = self.torch, config["device"]
        case = self.case(config["scene"], device)
        clear = self.clear(config)
        i = case.frames.index(frame)
        t = case.scene.transforms[i]
        options: dict[str, Any] = {
            "factor": config["factor"],
            "occlusion": case.scene.occlusion if config["inputs"]["shadow_maps"] else None,
            "normals": case.normals[i] if config["inputs"]["normals"] else None,
            "dtype": self.common.DTYPE,
            "device": device,
        }
        if self.effects_api["cls"] is not None:
            options["effects"] = self.effects_api["cls"](**config["effects"])
        if config["inputs"].get("albedo"):
            options["albedo"] = self.albedo(config["scene"], device)[i]
        medium = self.medium(config)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.apply(clear[i], case.depth[i], t, t["transform_matrix"], medium, case.lighting, **options)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return result.radiance.cpu().numpy(), result.inscatter.cpu().numpy(), seconds

    def work(self) -> None:
        """Run the queued jobs, those of the page first, one at a time."""
        self.torch.set_num_threads(self.threads)
        while True:
            with self.lock:
                while not self.queue:
                    self.wake.wait()
                job = min(self.queue, key=lambda j: (j.priority, j.order))
                self.queue.remove(job)
                job.state, job.started = "running", time.time()
                self.running = job
            warmup = self.warmup.get(job.config["scene"]) if job.origin == "warmup" else None
            try:
                if warmup is not None:
                    warmup.update(state="running", started=time.time())
                if not self.execute(job):
                    print(f"Stopped a request for the {job.config['scene']} scene that the page replaced", flush=True)
            except Exception as error:
                traceback.print_exc()
                with self.lock:
                    self.failed[job.identity] = f"{type(error).__name__}: {error}"
                    if warmup is not None:
                        warmup.update(state="error", error=f"{type(error).__name__}: {error}")
            else:
                if warmup is not None:
                    warmup.update(state="done", seconds=time.time() - (job.started or job.created))
            finally:
                with self.lock:
                    job.state = "done"
                    self.running = None
                    self.jobs.pop(job.identity, None)

    def execute(self, job: Job) -> bool:
        """Compute the frames of a job that aren't known, then time its selected frame once warm; False if the page
        replaced it before it was done, see ``supersede``."""
        config, key = job.config, job.identity[0]
        locked = bool(config["device"].startswith("cuda") and self.args.gpu_lock)
        if locked:
            job.stage = "waiting for the GPU lock"
            while True:
                if job.cancelled:
                    return False
                try:
                    os.mkdir(self.args.gpu_lock)
                    break
                except FileExistsError:
                    time.sleep(1)
        try:
            for n, frame in enumerate(job.frames):
                frame_key = self.frame_key(key, frame)
                with self.lock:
                    if job.cancelled:
                        return False
                    if self.frames.get(frame_key) is not None:
                        continue
                    first = key not in self.first_calls
                hint = f"; {NIGHT_HINT}" if first and config["scene"] == "night" else ""
                job.stage = f"frame {frame} ({n + 1}/{len(job.frames)}){hint}"
                radiance, inscatter, seconds = self.run_frame(config, frame)
                with self.lock:
                    self.frames.put(frame_key, FrameEntry(radiance, inscatter, seconds, None if first else seconds))
                    if first:
                        self.first_calls[key] = (frame, seconds)
            with self.lock:
                if job.cancelled:
                    return False
                entry = self.frames.get(self.frame_key(key, config["frame"]))
            repeats = self.repeats(config["device"])
            if entry is not None and entry.timed < repeats:
                job.stage = f"timing frame {config['frame']} once warm, fastest of {repeats} calls"
                for _ in range(repeats - entry.timed):
                    seconds = self.run_frame(config, config["frame"])[2]
                    entry.warm = seconds if entry.warm is None else min(entry.warm, seconds)
                entry.timed = repeats
            return True
        finally:
            if locked:
                os.rmdir(self.args.gpu_lock)

    # ------------------------------------------------------------------------------------------------------------------
    # Start

    def load(self) -> None:
        """Load both scenes, then queue the warm-up of their default configuration."""
        common, fb = self.common, self.fb
        loaders: dict[str, Callable[[], SceneData]] = {
            "day": lambda: SceneData(
                self,
                "day",
                fb.day_case(common.DAY_FRAMES, "cpu"),
                self.reference_fog(common.ROOT / "ms" / "cycles_ref", common.DAY_FRAMES),
                "Cycles 4096 spp, 16 volume bounces (ms/cycles_ref)",
                WITH_FOG["day"],
                CLEAR["day"],
            ),
            "night": lambda: SceneData(
                self,
                "night",
                fb.night_case("cpu"),
                self.reference_fog(fb.NIGHT_REFERENCE, common.NIGHT_FRAMES),
                "Cycles 65536 spp, 16 volume bounces, unclamped (gpu/night_full_spp65536)",
                WITH_FOG["night"],
                CLEAR["night"],
            ),
        }
        for name, loader in loaders.items():
            self.loading = f"loading the {name} scene"
            print(f"Loading the {name} scene", flush=True)
            try:
                data = loader()
            except Exception as error:
                traceback.print_exc()
                self.scene_errors[name] = f"{type(error).__name__}: {error}"
                continue
            with self.lock:
                self.scenes[name] = data
        self.loading = ""
        print(f"Scenes loaded: {sorted(self.scenes)}", flush=True)
        if self.args.no_warmup:
            return
        for name in self.scenes:
            self.warmup[name] = {"state": "queued", "device": self.device}
            self.request({"scene": name, "device": self.device}, origin="warmup")

    def reference_fog(self, path: Path, frames) -> np.ndarray:
        """Cycles' light scattered by the fog: the sum of its volume passes."""
        load = self.common.load_images
        return load(path / "volume" / "direct", frames) + load(path / "volume" / "indirect", frames)

    # ------------------------------------------------------------------------------------------------------------------
    # Images

    def image(self, frame_id: str, name: str) -> np.ndarray:
        """An image of the closed form for a frame of a configuration: its radiance, in-scattering or surfaces."""
        with self.lock:
            frame_key = self.ids.get(frame_id)
            entry = self.frames.get(frame_key) if frame_key else None
        if entry is None:
            raise NotFound(f"Frame {frame_id} isn't in memory anymore; compute its configuration again.")
        if name == "radiance":
            return entry.radiance
        if name == "inscatter":
            return entry.inscatter
        if name == "surfaces":
            return entry.radiance - entry.inscatter
        raise BadRequest(f"Unknown image {name!r}, one of radiance, inscatter and surfaces.")

    def static_image(self, scene: str, frame: int, name: str, spp: str | None) -> np.ndarray:
        """An image of a scene that doesn't depend on the closed form: Cycles' reference and renders, and the frames
        without fog given to the closed form."""
        if scene not in self.scenes:
            raise NotFound(f"Scene {scene!r} isn't available.")
        data = self.scenes[scene]
        if frame not in data.frames:
            raise NotFound(f"Frame {frame} isn't among the frames of the {scene} scene.")
        i = data.frames.index(frame)
        key = (scene, frame, name, spp)
        cached = self.static.get(key)
        if cached is not None:
            return cached
        gpu = self.fb.GPU

        def count() -> int:
            try:
                return int(spp or "")
            except ValueError:
                raise BadRequest(f"Samples per pixel {spp!r} aren't a number.") from None

        if name == "reference":
            image = data.case.reference[i]
        elif name == "reference_fog":
            image = data.fog[i]
        elif name == "reference_surfaces":
            image = data.case.reference[i] - data.fog[i]
        elif name == "albedo":
            if data.albedo is None:
                reason = data.unavailable.get("albedo", "this library doesn't take the albedo of each pixel")
                raise NotFound(f"No albedo of each pixel in the {scene} scene: {reason}.")
            image = data.albedo[i]
        elif name == "input":
            if spp in (None, "", "clean"):
                image = data.case.clear[i].cpu().numpy()
            else:
                image = self.render(gpu / data.clear_pattern.format(count()) / "frames", frame)
        elif name in ("cycles", "cycles_noisy", "cycles_fog"):
            path = gpu / data.with_fog.format(count())
            if name == "cycles":
                image = self.render(path / "frames", frame)
            elif name == "cycles_noisy":
                image = self.render(path / "noisy", frame)
            else:
                image = self.render(path / "volume" / "direct", frame) + self.render(path / "volume" / "indirect", frame)
        else:
            raise BadRequest(f"Unknown image {name!r}.")
        image = np.asarray(image, dtype=np.float32)
        self.static.put(key, image)
        return image

    def render(self, path: Path, frame: int) -> np.ndarray:
        if not path.exists():
            raise NotFound(f"{path.as_posix()} wasn't rendered.")
        try:
            return self.common.load_images(path, [frame])[0]
        except KeyError:
            raise NotFound(f"Frame {frame} wasn't rendered in {path.as_posix()}.") from None

    # ------------------------------------------------------------------------------------------------------------------
    # Cycles: the benchmark's curve and the ablations

    def benchmark(self) -> dict | None:
        """Cycles' curve of error against time from the latest benchmark: ``final_gpu.json`` if present, else the
        newest ``final_benchmark*.json`` made with Cycles' GPU renders."""
        results = self.common.OUT / "results"
        preferred = [results / "final_gpu.json"] if (results / "final_gpu.json").exists() else []
        others = sorted(results.glob("final_benchmark*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = preferred + others
        stamp = ("benchmark", tuple((str(p), p.stat().st_mtime) for p in candidates))
        if stamp in self.static:
            return self.static.get(stamp)
        found = None
        for path in candidates:
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            renders = str(data.get("renders") or "gpu")
            usable = any(
                isinstance(data.get(case), Mapping)
                and any("denoised" in v and v.get("seconds") for v in (data[case].get("cycles") or {}).values())
                for case in ("day", "night")
            )
            if usable and renders.startswith("gpu"):
                found = (path, data, renders)
                break
        result: dict[str, Any] | None = None
        if found is not None:
            path, data, renders = found
            curves: dict[str, list[dict[str, Any]]] = {}
            for case in ("day", "night"):
                rows = []
                for spp, v in ((data.get(case) or {}).get("cycles") or {}).items():
                    row = {"spp": int(spp), "seconds": v.get("seconds"), "clear_seconds": v.get("clear_seconds")}
                    if "denoised" in v:
                        row["denoised"], row["noisy"] = v["denoised"], v.get("noisy")
                    rows.append(row)
                curves[case] = sorted(rows, key=lambda r: r["spp"])
            # The benchmark's own "device" is where it ran the closed form; Cycles' times come from the logs of its
            # renders, which name the device Blender rendered on
            logs = self.common.OUT / renders / "logs"
            hardware = {
                render_device(logs / f"{WITH_FOG[case].format(row['spp'])}.log")
                for case, rows in curves.items()
                for row in rows
                if row.get("seconds")
            }
            known = sorted(h for h in hardware if h)
            result = {
                "provenance": {
                    "file": path.name,
                    "cycles_hardware": " + ".join(known) if known else None,
                    "closed_form_device": data.get("device"),
                    "renders": f"acceleration/{renders}",
                    "date": modified(path),
                },
                "curves": curves,
                "flicker": data.get("day_flicker") or (data.get("day") or {}).get("flicker"),
            }
        self.static.put(stamp, result)
        return result

    def cycles_section(self, config: dict, errors: dict, warm: float | None) -> dict:
        """What Cycles with fog needs to reach the closed form's error, along the benchmark's curve, and the time of the
        whole pipeline: Cycles renders the frame without fog at a few samples per pixel and the closed form adds the
        fog."""
        bench = self.benchmark()
        if bench is None or not bench["curves"].get(config["scene"]):
            reason = "no benchmark of Cycles in results/ (final_gpu.json or final_benchmark*.json)"
            return {"available": False, "reason": reason}
        rows = bench["curves"][config["scene"]]
        curve = [r for r in rows if "denoised" in r and r["seconds"]]
        interpolate = self.fb.interpolate
        section: dict[str, Any] = {"available": True, "provenance": bench["provenance"], "curve": rows}
        for metric in ("rel_l1", "rel_l1_blurred"):
            error = errors["whole"][metric]
            entry: dict[str, Any] = {
                "error": error,
                "seconds": interpolate([(r["denoised"][metric], r["seconds"]) for r in curve], error),
                "spp": interpolate([(r["denoised"][metric], r["spp"]) for r in curve], error),
            }
            if entry["seconds"] is None and curve:
                below = error < min(r["denoised"][metric] for r in curve)
                entry["beyond"] = "below" if below else "above"
                errors_of = [(r["denoised"][metric], r) for r in curve]
                bound = (min if below else max)(errors_of, key=lambda pair: pair[0])[1]
                entry["bound"] = {"spp": bound["spp"], "seconds": bound["seconds"], "error": bound["denoised"][metric]}
            section[metric] = entry
        clear_seconds = None
        if config["clear"] != "clean":
            clear_seconds = next((r.get("clear_seconds") for r in rows if r["spp"] == config["clear"]), None)
        section["pipeline"] = {
            "closed_form": warm,
            "clear": config["clear"],
            "clear_seconds": clear_seconds,
            "seconds": None if warm is None or clear_seconds is None else warm + clear_seconds,
        }
        return section

    def cycles_flicker(self, temporal: float) -> dict:
        """Flicker of Cycles at each number of samples, from the benchmark, and what it needs to flicker as little as
        the closed form."""
        bench = self.benchmark()
        if bench is None or not bench.get("flicker"):
            return {"cycles": None}
        curve = {r["spp"]: r for r in bench["curves"].get("day", [])}
        rows = []
        for key, value in bench["flicker"].items():
            if key.startswith("cycles_spp"):
                spp = int(key.removeprefix("cycles_spp"))
                rows.append({"spp": spp, "temporal": value["temporal"], "seconds": curve.get(spp, {}).get("seconds")})
        rows.sort(key=lambda r: r["spp"])
        return {
            "cycles": rows,
            "provenance": bench["provenance"],
            "seconds_at_equal": self.fb.interpolate(
                [(r["temporal"], r["seconds"]) for r in rows if r["seconds"]], temporal
            ),
            "spp_at_equal": self.fb.interpolate([(r["temporal"], r["spp"]) for r in rows], temporal),
        }

    def ablation(self, scene: str, factor: int, device: str | None) -> dict:
        """What each switch of this library costs per frame and the error it removes in a scene, from the ablations
        (``results/*ablation*.json``, each switch off alone against every switch on).

        Each switch is taken from the first ablation that measured it: those run on the same kind of device first, then
        those of the same model as this library's, the one whose every switch on took the least time first (its errors
        are the same, and work that ran beside it can only have made it slower), then the others, the newest first. An
        ablation is of the same model when its every switch on gives this library's errors of the scene's default
        configuration at factor 1, whole frames, fog and surfaces, within float rounding (which differs between
        devices): ablations of a library that has changed since, or of another one, are told apart that way, since the
        folder an ablation names may hold another version now. Until the default configuration is computed, that is
        unknown (``checked`` is false), and the page asks again.

        A switch that this library's own ``Effects.missing`` finds ineffective in the scene's default configuration is
        listed as such, whatever the ablations say. Variants that are neither switches nor inputs of this library are
        left out."""
        results = self.common.OUT / "results"
        files = sorted(results.glob("*ablation*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            default_config = self.normalize({"scene": scene})
        except BadRequest as error:
            return {"rows": [], "files": [], "checked": False, "error": str(error)}
        baseline = self.baseline(default_config)
        loaded = []
        for path in files:
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            case = (data.get("cases") or {}).get(scene) if isinstance(data, Mapping) else None
            if not isinstance(case, Mapping):
                continue
            same_device = device is not None and str(data.get("device", "cpu")).startswith(device)
            model = same_model(case, baseline)
            factors = case.get("factors") or {}
            used = str(factor) if str(factor) in factors else ("1" if "1" in factors else next(iter(factors), None))
            all_on = ((factors.get(used) or {}).get("variants") or {}).get("all on") or {}
            fastest = float(all_on.get("seconds") or math.inf) if model == "same" else 0.0
            loaded.append((not same_device, {"same": 0, None: 1, "other": 2}[model], fastest, path, data, model, used))
        # Sorting is stable: the newest first where the time of every switch on isn't compared
        loaded.sort(key=lambda entry: entry[:3])
        names = self.effects_api["names"] | set(self.inputs)
        rows: dict[str, dict] = {}
        for *_, path, data, model, used in loaded:
            case = data["cases"][scene]
            factors = case.get("factors") or {}
            on_1 = ((factors.get("1") or {}).get("variants") or {}).get("all on") or {}
            source = {
                "file": path.name,
                "device": data.get("device"),
                "threads": data.get("threads"),
                "date": modified(path),
                "factor": int(used) if used else None,
                "model": model,
                "all_on_whole": (on_1.get("whole") or {}).get("rel_l1"),
            }
            variants: dict[str, Any] = factors[used]["variants"] if used is not None else {}
            on: dict[str, Any] = variants.get("all on") or {}
            for variant, v in variants.items():
                name = variant.removesuffix(" off")
                if not on or variant == "all on" or name == variant or name in rows or name not in names:
                    continue
                rows[name] = {
                    "switch": name,
                    "saved_ms": (on["seconds"] - v["seconds"]) * 1000,
                    "added": {k: (v[k]["rel_l1"] - on[k]["rel_l1"]) * 100 for k in ("whole", "fog", "surfaces")},
                    "fog_ratio": v["fog"]["ratio"],
                    "all_on": {"seconds": on["seconds"], "whole": on["whole"]["rel_l1"]},
                    "source": source,
                }
        # Whether a switch is effective is this library's to tell, also for switches newer than the ablations
        default = self.prerequisites(default_config)
        if default["available"]:
            helper = {"helper": True, "file": "Effects.missing", "device": None, "date": "", "factor": None}
            for name, missing in default["missing"].items():
                if missing:
                    rows[name] = {"switch": name, "ineffective": missing, "source": helper}
        return {
            "rows": list(rows.values()),
            "files": [entry[3].name for entry in loaded],
            "checked": baseline is not None,
            "baseline": None if baseline is None else {"whole": baseline["whole"]["rel_l1"]},
        }

    # ------------------------------------------------------------------------------------------------------------------
    # Description

    def schema(self) -> dict:
        api = self.effects_api
        return clean(
            {
                "switches": api["switches"],
                "groups": api["groups"],
                "conditions": api["conditions"],
                "given": api["given"],
                "presets": api["presets"],
                "helpers": bool(api.get("helpers")),
                "has_effects": api["cls"] is not None,
                "inputs": self.inputs,
                "factors": list(FACTORS),
                "devices": [{"name": d, "label": self.device_names[d]} for d in self.devices],
                "device": self.device,
                "threads": self.threads,
                "library": self.library,
                "fingerprint": self.fingerprint,
                "instance": self.instance,
                "root": str(self.root),
                "scenes": {name: data.describe() for name, data in self.scenes.items()},
                "scene_errors": self.scene_errors,
            }
        )

    def status(self) -> dict:
        with self.lock:
            queue = sorted(self.queue, key=lambda j: (j.priority, j.order))
            return clean(
                {
                    "loading": self.loading,
                    "instance": self.instance,
                    "uptime": time.time() - self.started,
                    "device": self.device,
                    "running": self.running.describe() if self.running else None,
                    "queue": [j.describe() for j in queue],
                    "warmup": {
                        name: {**w, "elapsed": time.time() - w["started"] if w.get("started") else None}
                        for name, w in self.warmup.items()
                    },
                    "cached_frames": len(self.frames),
                    "scene_errors": self.scene_errors,
                }
            )


def presets(effects_cls: Any, names: set[str], owners: tuple) -> dict[str, dict]:
    """Presets of switches that the library defines, if any, e.g. ``EFFECT_PRESETS`` or ``Effects.presets()``, as
    {name: {"values": {switch: bool}, "description": str}}. A preset can be an ``Effects`` or a mapping of switches,
    alone, in a pair with its description, or as {"effects": ..., "description": ...}."""
    found: dict[str, dict] = {}
    for owner in owners:
        for attribute in ("EFFECT_PRESETS", "PRESETS", "presets"):
            value = getattr(owner, attribute, None) if owner is not None else None
            if callable(value) and not isinstance(value, Mapping):
                try:
                    value = value()
                except Exception:  # noqa: S112 - a preset that can't be read is left out
                    continue
            if not isinstance(value, Mapping):
                continue
            for name, preset in value.items():
                description = ""
                if isinstance(preset, tuple) and len(preset) == 2 and isinstance(preset[1], str):
                    preset, description = preset
                if isinstance(preset, Mapping) and "effects" in preset:
                    description = str(preset.get("description") or description)
                    preset = preset["effects"]
                description = description or str(getattr(preset, "description", "") or "")
                try:
                    if isinstance(preset, effects_cls):
                        chosen = preset
                    elif isinstance(preset, Mapping):
                        chosen = effects_cls(**{k: bool(v) for k, v in preset.items() if k in names})
                    else:
                        continue
                except Exception:  # noqa: S112 - a preset that isn't valid is left out
                    continue
                values = {n: bool(getattr(chosen, n)) for n in names}
                found.setdefault(str(name), {"values": values, "description": description})
    return found


# ----------------------------------------------------------------------------------------------------------------------
# HTTP


class Handler(BaseHTTPRequestHandler):
    lab: Lab
    server_version = "FogViewer/1.0"
    timeout = 30
    """seconds a connection may stay silent, e.g. one that announces more of a body than it sends, before its thread
    is freed"""

    def log_message(self, format: str, *args: Any) -> None:
        if self.lab.args.verbose:
            super().log_message(format, *args)

    def send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(clean(value)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_image(self, image: np.ndarray, cache: str) -> None:
        """Linear float image as little-endian float16, of shape (h, w, c), which the page tone-maps itself."""
        image = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
        body = np.clip(image, -65504.0, 65504.0).astype("<f2").tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Height", str(image.shape[0]))
        self.send_header("X-Width", str(image.shape[1]))
        self.send_header("X-Channels", str(image.shape[2] if image.ndim == 3 else 1))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, name: str) -> None:
        # Only the files of the static folder, by their own name
        available = {p.name: p for p in STATIC.iterdir() if p.is_file()}
        if name not in available:
            raise NotFound(f"No such file {name!r}.")
        body = available[name].read_bytes()
        types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
        self.send_response(200)
        self.send_header("Content-Type", types.get(available[name].suffix, "text/plain") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("Content-Length isn't a number.") from None
        # A negative length would read until the client closes the connection, holding this thread
        if length < 0:
            raise BadRequest("Content-Length is negative.")
        if length > 1_000_000:
            raise BadRequest("The request is too large.")
        try:
            raw = self.rfile.read(length)
        except TimeoutError:
            raise BadRequest(f"The request's body didn't arrive within {self.timeout} s.") from None
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            raise BadRequest("The request isn't valid JSON.") from None

    def handle_errors(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except BadRequest as error:
            self.send_json({"error": str(error)}, 400)
        except NotFound as error:
            self.send_json({"error": str(error)}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as error:
            traceback.print_exc()
            try:
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)
            except OSError:
                pass

    def do_GET(self) -> None:
        self.handle_errors(self.get)

    def do_POST(self) -> None:
        self.handle_errors(self.post)

    def get(self) -> None:
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        lab, path = self.lab, url.path
        if path in ("/", "/index.html"):
            return self.send_file("index.html")
        if path.startswith("/static/") or path in ("/app.js", "/style.css", "/favicon.svg"):
            return self.send_file(path.rsplit("/", 1)[-1])
        if path == "/api/status":
            return self.send_json(lab.status())
        if lab.loading and path in ("/api/schema", "/api/static", "/api/ablation"):
            return self.send_json({"loading": lab.loading}, 503)
        if path == "/api/schema":
            return self.send_json(lab.schema())
        if path == "/api/image":
            return self.send_image(lab.image(query.get("id", ""), query.get("name", "")), "private, max-age=3600")
        if path == "/api/static":
            try:
                frame = int(query.get("frame", ""))
            except ValueError:
                raise BadRequest("A frame is a number.") from None
            image = lab.static_image(query.get("scene", ""), frame, query.get("name", ""), query.get("spp"))
            # Renders are read once per run of the server: an address naming this run may be kept by the browser
            cache = "private, max-age=3600" if query.get("v") == lab.instance else "no-cache"
            return self.send_image(image, cache)
        if path == "/api/ablation":
            try:
                factor = int(query.get("factor", 1))
            except ValueError:
                raise BadRequest("A factor is a number.") from None
            return self.send_json(lab.ablation(query.get("scene", "day"), factor, query.get("device")))
        if path == "/api/benchmark":
            bench = lab.benchmark()
            return self.send_json({"available": bench is not None} | (bench or {}))
        raise NotFound(f"No such path {path}.")

    def post(self) -> None:
        url = urlparse(self.path)
        lab, body = self.lab, self.body()
        if not isinstance(body, Mapping):
            raise BadRequest("A configuration is a JSON object.")
        if url.path == "/api/compute":
            supersede = flag(body.get("supersede", False), "supersede")
            response = lab.request(body, supersede=supersede, retry=flag(body.get("retry", False), "retry"))
            return self.send_json(response, 503 if response["state"] == "loading" else 200)
        if url.path == "/api/prerequisites":
            if lab.loading:
                return self.send_json({"loading": lab.loading}, 503)
            config = lab.normalize(body)
            return self.send_json({"config": config, **lab.prerequisites(config)})
        raise NotFound(f"No such path {url.path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on, this machine only by default")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="auto: CUDA if available")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads of the closed form")
    parser.add_argument(
        "--library",
        help="folder of the visionsim library to use, put first on the path; by default the one on PYTHONPATH if set, "
        "else that of the checkout this server sits in",
    )
    parser.add_argument("--root", help="checkout with runs/fog-comparison and the study's scripts, found by default")
    parser.add_argument("--repeats", type=int, help="warm calls timed, the fastest kept: 3 on the GPU, 1 on the CPU")
    parser.add_argument("--cache-frames", type=int, default=192, help="frames of the closed form kept in memory")
    parser.add_argument("--gpu-lock", help="folder created while computing on the GPU, waited for while it exists")
    parser.add_argument("--no-warmup", action="store_true", help="don't compute the default configurations at start")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args()

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    root = study_root(args.root)
    # The study's paths are relative to the root of the checkout
    os.chdir(root)
    if library := library_path(args.library):
        sys.path.insert(0, str(library))
    sys.path[1:1] = [str(root / STUDY), str(root / STUDY / "analysis")]

    lab = Lab(args, root)
    Handler.lab = lab
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    address = f"http://{args.host}:{server.server_address[1]}"
    print(f"Fog viewer on {address}, library {lab.library}, {lab.device_names[lab.device]}", flush=True)
    if lab.effects_api["cls"] is None:
        print(
            f"The library in {lab.library} has no Effects: the page shows its inputs only. Pass --library <a checkout "
            "whose library has them> to switch its effects.",
            flush=True,
        )
    threading.Thread(target=lab.work, name="closed-form", daemon=True).start()
    threading.Thread(target=lab.load, name="load", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
