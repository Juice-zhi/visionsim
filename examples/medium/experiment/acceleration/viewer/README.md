# Fog control panel

A page to switch the terms of VisionSIM's closed-form fog (`visionsim.medium`) on and off, and see what each one does:
the error against Cycles, the time per frame, what Cycles needs for the same error, and the images, all computed live
by the library.

## Running it

From the root of the main checkout, where the renders are (`runs/fog-comparison`):

```sh
.conda/python.exe examples/medium/experiment/acceleration/viewer/server.py --port 8765 --device auto --threads 8
```

then open <http://127.0.0.1:8765>. In Claude Code, the launch configuration `fog-viewer` of
`E:/visionsim/.claude/launch.json` runs the same command, with the server's absolute path
(`E:/visionsim/examples/medium/experiment/acceleration/viewer/server.py`), as the preview may start it from another
folder; the server moves to the root of the checkout by itself.

The page switches the terms of the library's `Effects`, which the main checkout's own library doesn't have until the
merged library lands there: until then, add `"--library", "<a checkout whose library has them>"` to the entry's
`runtimeArgs` (without them, the page shows the inputs only, and says so, as does the server when it starts). Agents
that share the machine add `"--threads", "4"` and either `"--device", "cpu"` or `"--gpu-lock", "<the lock folder>"`.

- `--device auto|cuda|cpu`: where the closed form runs; `auto` picks CUDA when available. The page can switch device.
- `--threads N`: CPU threads of the closed form (8 by default).
- The library: `--library <folder>` if given, else the one on `PYTHONPATH` if set (e.g.
  `PYTHONPATH=E:/visionsim-wt/<branch>`), else that of the checkout the server sits in, i.e. the main checkout's (as
  installed) or, for a copy of this folder in a git worktree, the worktree's. The page shows which one is used.
- `--repeats N`: warm calls timed, of which the fastest is shown: 3 on the GPU, 1 on the CPU by default.
- `--root`: the checkout with `runs/fog-comparison` and the study's scripts (`common.py`, `analysis/`). These are
  untracked and live in the main checkout, which the server finds by itself when this folder sits in a git worktree.
- `--gpu-lock DIR`: a folder the server creates while it computes on the GPU, and waits for while it exists, to share
  the GPU with other jobs that use the same lock.
- `--no-warmup`, `--cache-frames N` (192 frames by default, about 1.3 MB each), `--host`, `--verbose`.

At start, the server loads both scenes (about 15-25 s) and then computes their default configuration in the
background: about 5-12 s by day on 4 CPU threads, about 2 s on the GPU; the night scene's first call computes the halo
of its lamp and the light that surfaces reflect, which takes about 1.5-2.5 min on 4 CPU threads and 6-15 s on the GPU.
The page says so while it lasts; requests made from the page go first.

The closed form runs in a single worker thread, one configuration at a time, so that requests don't compete for the
CPU; the page never waits for it, and shows what is being computed and for how long. A new request of the page
replaces those it made before: they leave the queue, and one already running stops before its next frame (a frame
being computed, e.g. the night's halo, can't be interrupted). Frames are kept in memory by their whole configuration,
so going back to a configuration is instant.

Each run of the server draws a token that names its images (`instance`), so that the browser never shows images that
an earlier run, maybe with another library, computed for the same configuration; when the server restarts, the page
reloads itself, keeping its configuration, which lives in the address. The page also shows a fingerprint of the
library's source, the same for identical copies in different checkouts.

## The page

**Switches** (left). One per field of the library's `Effects`, read from its JSON schema, grouped by the `group` it
declares (daylight, lamps, surfaces), with its description on hover. Switches added to the library appear by
themselves.

A switch whose prerequisites aren't met is greyed out, and its tooltip says what is missing: a switch it needs that is
off, or a condition on the inputs that doesn't hold (e.g. `lamps` in the day scene, `shadows` without shadow maps,
`multiple_scattering` without multiple scattering). This is computed by the library's own `Effects.missing` for the
current scene and inputs, on every change, never by a copy of the rules, and follows the switches a switch needs
recursively: turning `daylight_glow` off greys out `glow_shadows` and, through it, `glow_shadow_edges`. A greyed switch
keeps its own on/off state (its toggle shows it, muted), can't be toggled until its prerequisites are met, and the
library computes it exactly as a switch that is off, so that both share their frames. It stays focusable, and screen
readers read what it misses.

**Model inputs**: what describes the scene rather than chooses terms of the model: `Medium.multiple_scattering`,
shadow maps (the inputs `shadows` and `occlusion` of `Effects`) and the normals of surfaces, through which surfaces are
lit through the fog (`normals`), all on by default. For a library whose `apply_medium` (and `apply_medium_upsampled`)
takes the albedo of each pixel, the page also offers it (`albedo`): Blender's diffuse color pass of the frames without
fog (`runs/fog-comparison/clear/diffuse/color`, which both scenes share), from which the irradiance surfaces received is
measured rather than estimated from mean albedos. It is off by default, as in `eval_closed_form.py`, so that the
defaults keep its numbers; a scene whose pass isn't rendered greys it out. Every input is also given to `Effects`'
helpers (`shadows`, `occlusion`, `normals`, `albedo`, as far as the library's `EFFECT_GIVEN` names them), so that
greying follows them.

**Performance**: the resolution of the fog (`apply_medium_upsampled` at factor 1, 2 or 4), the frames without fog given
to the closed form (clean, i.e. rendered at many samples, or Cycles at 1/4/16/64 spp denoised by OIDN,
`gpu/clear_<scene>_spp<N>`) and the device.

**Presets**: all on, all off, the library's own presets if it defines any (`EFFECT_PRESETS` or `Effects.presets()`),
and the defaults (every switch on, inputs as by default, factor 1, clean frames).

The configuration is kept in the address, so that a view can be reloaded or shared.

**Images** (middle). A and B can each be any of:

- the closed form: whole frame, fog (its in-scattering) and surfaces (whole minus fog), the frame without fog it
  was given, and the albedo of each pixel when the library takes it;
- Cycles' reference: whole frame, fog (its volume passes, direct plus indirect) and surfaces;
- Cycles at the chosen samples per pixel (`gpu/day_spp<N>`, `gpu/night_full_spp<N>`): denoised, noisy, and its fog;
- the signed relative error of luminance, `(A - B) / B`, of the closed form, its fog or its surfaces against the
  reference's, or of Cycles at N spp against the reference: blue where too dark, red where too bright, within the
  chosen range, optionally after both are blurred 5x5. Where the reference is nearly black, B is kept above 2% of its
  mean so that the ratio stays finite.

Drag on the image (or use the wipe slider) to move the split between A and B. Images arrive as linear float16 and are
tone-mapped in the browser (exposure in stops, then sRGB), so exposure changes are instant; *Auto* puts the 97th
percentile of luminance at 90% of white (each scene keeps its own exposure).

## The numbers

All errors are those of `common.errors`, as in `analysis/eval_closed_form.py`:

- **rel L1**: the sum over pixels and channels of `|ours - reference|`, over the sum of the reference;
- **blurred**: the same once both images are box-blurred 5x5, which leaves out most of Monte Carlo noise, i.e. the
  bias;
- **ratio**: the total light of ours over the reference's (above 1: too bright).

They are given for whole frames, for the fog (the closed form's in-scattering against Cycles' volume passes) and for
surfaces (frames minus fog on both sides). References: by day, Cycles 4096 spp with 16 volume bounces
(`ms/cycles_ref`); at night, Cycles 65536 spp with 16 volume bounces without clamping (`gpu/night_full_spp65536`).

- **Frames 255 + 345**: both frames together, exactly as `eval_closed_form.py` (all on, factor 1, clean frames: day
  whole 3.92%, night 5.49% on merged2). **Frame N**: the frame shown alone. The day scene also has frames 256-262, for
  flicker.
- **Δ default**: points of rel L1 against the scene's default configuration (every switch on, inputs as by default,
  factor 1, clean frames); red when the error grows.
- While a new configuration is computed, or after it failed, the numbers of the previous one stay, dimmed, under a
  note that says so; the image keeps the previous result too, and says so.
- **Cycles N spp + OIDN**: Cycles at the samples chosen above the images, from the benchmark, for comparison.
- **By region**, over the same frames: the sky (pixels that see no surface), the ground (normals pointing up; by day
  split into the ground the sun lights and the ground in an object's shadow, as the sun's shadow map tells), objects
  (other surfaces), and at night the pixels whose fog lies in the shadow the cube casts from the lamp (Cycles' direct
  volume pass under 2% of its median, as `eval_closed_form.py`) and the rest. Each cell is the rel L1 within the region,
  against the reference's light in that region; hovering it gives the blurred error and the ratio.
- **Under the cursor**: rel L1 and ratio in a window around the cursor (1, 9x9 or 25x25 pixels), computed in the
  browser from the float16 images, of the closed form against the reference.
- **Time**: the warm call, i.e. the time of the frame shown once what is kept from one frame to the next is computed,
  with the inputs already on the device, and the first call of the configuration, which may include one-off work
  (tables, the halo of lamps, the light that surfaces reflect, CUDA's start): it is flagged when it took more than 1.3
  times the warm call. Times are measured on this machine, on the device and threads shown; they are noisy on a loaded
  CPU.
- **Cycles for the same error**: the time per frame and samples per pixel at which Cycles + OIDN reaches the closed
  form's whole-frame error (as is, and blurred), interpolated log-log along its curve of error against time, as
  `final_benchmark.py` does. The curve comes from the latest benchmark in `runs/fog-comparison/acceleration/results`
  (`final_gpu.json` if present, else the newest `final_benchmark*.json` made with Cycles' GPU renders), whose file and
  date are shown, with the device Cycles rendered on, as the logs of its renders name it (the benchmark's own `device`
  is where it ran the closed form, also shown). When the frames without fog come from Cycles at N spp, the
  **pipeline** is their render time (from the benchmark) plus the closed form's warm call, and the saving is Cycles'
  time at equal error over the pipeline's, as in `final_benchmark.py`. With clean frames without fog, which aren't
  timed, the page gives Cycles' time over the closed form's alone, and says that it leaves them out: no saving can be
  claimed from it. Mind that the closed form's time is measured here, while Cycles' comes from the benchmark's
  hardware: the page says when they differ.
- **Flicker** (day only, on demand): `common.flicker` over frames 255-262, the temporal error of log intensity that
  drives false events in event cameras, and the samples per pixel at which Cycles flickers as little (from the
  benchmark's `day_flicker`).
- **What each switch does** (below the images): from the ablations (`results/*ablation*.json` of
  `analysis/ablation.py`), each switch off alone against every switch on: the time saved per frame and the points of
  rel L1 it adds, at the factor chosen when the ablation has it, with the source of each row. An ablation is of the
  same model as the library's when its every switch on gives the library's errors of the scene's default configuration
  (factor 1, whole frames, fog and surfaces, within 0.001 points, as devices round differently); the folder an
  ablation names may hold another version of the library since, so its source isn't compared. Each switch is taken
  from ablations run on the same kind of device first, then those of the same model, the one whose every switch on
  took the least time first (its errors are the same, and work running beside it can only make it slower), then the
  others, the newest first, marked `≠` (another model) or `?` (not checked until the default configuration is
  computed). A switch that the library's own `Effects.missing` finds ineffective in the scene's default configuration
  is listed as such (source `lib`), whatever the ablations say; variants that are neither switches nor inputs of the
  library are left out. Click a switch's name to toggle it.

## API

JSON unless said otherwise; a configuration is
`{"scene", "frame", "effects": {switch: bool}, "inputs": {"multiple_scattering", "shadow_maps", "normals", "albedo"},
"factor", "clear": "clean" | spp, "device"}`, any field left out (or null) taking its default; `albedo` only for a
library that takes it.

- `GET /api/schema`: switches, groups, conditions, presets, scenes, devices, the library used, its `fingerprint` and
  the server's `instance`.
- `GET /api/status`: what is loading or computing and for how long, the queue, the warm-up and the `instance`.
- `POST /api/prerequisites` (a configuration): what each switch misses, from `Effects.missing`.
- `POST /api/compute` (a configuration, with `"flicker": true` to measure flicker, `"supersede": true` to drop the
  page's earlier requests): queues it unless its frames are known, and returns its state (`queued`, `running`, `done`,
  `error`), with its result once done. Switches, inputs and flags are JSON booleans; anything else is refused (400).
- `GET /api/image?id=&name=radiance|inscatter|surfaces`: the closed form's image of a result, float16 (h, w, 3).
- `GET /api/static?scene=&frame=&name=reference|reference_fog|reference_surfaces|input|albedo|cycles|cycles_noisy|cycles_fog&spp=&v=`:
  images of the scene, float16, which the browser may keep when `v` is the server's `instance`.
- `GET /api/ablation?scene=&factor=&device=`: the table's rows, each with its source and whether that ablation is of
  the same model (`model`: `same`, `other` or null), and `checked`, whether the default configuration is known yet.
- `GET /api/benchmark`.

## Smoke test

```sh
.conda/python.exe examples/medium/experiment/acceleration/viewer/smoke_test.py --device cpu --threads 4
```

starts the server on a free port, waits for its warm-up, and checks through the API that the default configuration of
both scenes gives the errors of `eval_closed_form.py` (evaluated in the same process with the same library, or read
from `--expected <its JSON>`, which must come from a library with the same fingerprint), that turning a switch off
changes them while turning off a greyed switch doesn't, that prerequisites follow the inputs, that images, factor 4,
Cycles' frames without fog, bad and malformed requests are handled, that ablations are checked against the library's
model, that the albedo of each pixel is offered, off by default, and changes the surfaces' error (for a library that
takes it), that a request answered from memory drops the page's earlier ones, and that a second run of the server
(`--skip-restart` to leave it out) names its images differently and gives the same numbers. It takes about 3 min on 4
CPU threads with `--expected` (the night's halo, about 1.5 min, dominates) and under 1 min on the GPU.
