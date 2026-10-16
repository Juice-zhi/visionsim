# Renders of the acceleration study: Cycles' Monte Carlo at several budgets, with denoising and path guiding, on the
# CPU, and EEVEE's volumetrics on the integrated GPU, of the day scene (sun, sky, objects, multiple scattering) and the
# night scene (a point light behind a cube, single scattering). Run from the repository root with visionsim's
# environment activated; frame times are printed in each log, as FRAME_TIME lines.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot/../../../..")
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = ""
$R = "runs/fog-comparison"
$A = "$R/acceleration"
$S = "examples/medium/experiment"
New-Item -ItemType Directory -Force "$A/logs" | Out-Null
$day = @(254, 255, 256, 257, 258, 259, 260, 261, 262, 345)  # 254 warms up Blender and is left out of timings
$pair = @(254, 255, 345)

function Mc($name, $blend, $frames, [string[]]$extra) {
    if (Test-Path "$A/logs/$name.log") { if (Select-String -Quiet "FRAME_TIME frame=$($frames[-1]) " "$A/logs/$name.log") { return } }
    & E:\Blender\blender.exe -b $blend --python-exit-code 1 --python "$S/acceleration/render_mc.py" -- "$A/$name" `
        --frame @frames --threads 12 @extra *> "$A/logs/$name.log"
    "$name exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
}

function Eevee($name, $blend, $frames, [string[]]$extra) {
    if (Test-Path "$A/logs/$name.log") { if (Select-String -Quiet "FRAME_TIME frame=$($frames[-1]) " "$A/logs/$name.log") { return } }
    $env:VK_LOADER_DRIVERS_DISABLE = "*nv-vk*"  # the integrated GPU only, leaving the NVIDIA GPU alone
    & E:\Blender\blender.exe -b --gpu-backend vulkan $blend --python-exit-code 1 --python "$S/acceleration/render_eevee.py" -- `
        "$A/$name" --frame @frames @extra *> "$A/logs/$name.log"
    Remove-Item Env:\VK_LOADER_DRIVERS_DISABLE
    "$name exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
}

# Scenes: the night scene without fog, as input of image-to-image networks, and the day scene with denser fog
if (-not (Test-Path "$A/clear_point_shadow.blend")) {
    & E:\Blender\blender.exe -b "$R/demo.blend" --python-exit-code 1 --python "$S/lamp_scene.py" -- `
        "$A/clear_point_shadow.blend" --position 4 21 1.5 --power 3000 --radius 0.05 *> "$A/logs/clear_point_shadow_blend.log"
}
if (-not (Test-Path "$A/fog_dense.blend")) {
    & E:\Blender\blender.exe -b "$R/demo.blend" --python-exit-code 1 --python "$S/add_fog.py" -- `
        "$S/acceleration/ground_fog_dense.json" "$A/fog_dense.blend" --volume-bounces 16 --no-adaptive *> "$A/logs/fog_dense_blend.log"
}

# Day scene, with light scattered up to 16 times as in the reference
$ms = @("--volume-bounces", "16")
foreach ($spp in 4, 16, 64, 256, 1024) { Mc "day_spp$spp" "$R/fog_ref.blend" $day ($ms + @("--samples", "$spp", "--denoise")) }
foreach ($spp in 16, 64, 256) { Mc "day_guided$spp" "$R/fog_ref.blend" $day ($ms + @("--samples", "$spp", "--denoise", "--guiding")) }
Mc "day_default" "$R/fog_ref.blend" $day ($ms + @("--samples", "256", "--adaptive", "0.05", "--denoise"))
Mc "day_spp64_seed1" "$R/fog_ref.blend" $pair ($ms + @("--samples", "64", "--denoise", "--seed", "1"))

# Night scene, single scattering as in its reference
$ss = @("--volume-bounces", "0")
$night = "$R/variants/fog_point_shadow.blend"
foreach ($spp in 16, 64, 256, 1024) { Mc "night_spp$spp" $night $pair ($ss + @("--samples", "$spp", "--denoise")) }
foreach ($spp in 64, 256) { Mc "night_guided$spp" $night $pair ($ss + @("--samples", "$spp", "--denoise", "--guiding")) }

# EEVEE on the integrated GPU: its defaults (8 pixel tiles, 64 slices up to 100 m), finer froxels further away, and a
# single sample per frame as in real time
Eevee "eevee_day_default" "$R/fog_ref.blend" $day @()
Eevee "eevee_day_fine" "$R/fog_ref.blend" $pair @("--tile", "2", "--slices", "256", "--end", "300")
Eevee "eevee_day_1sample" "$R/fog_ref.blend" $day @("--taa", "1")
Eevee "eevee_night_default" $night $pair @()
Eevee "eevee_night_fine" $night $pair @("--tile", "2", "--slices", "256", "--end", "300")

# References for learning methods out of their training distribution, and the night scene without fog
Mc "night_clear" "$A/clear_point_shadow.blend" $pair @("--samples", "256", "--denoise")
Mc "night_ms_ref" $night @(254, 255, 345) @("--volume-bounces", "16", "--samples", "2048")
Mc "dense_ref" "$A/fog_dense.blend" @(254, 255, 345) @("--volume-bounces", "16", "--samples", "2048")

# Timing at 1080p
foreach ($spp in 16, 64, 256) { Mc "hd_spp$spp" "$R/fog_ref.blend" @(254, 255) ($ms + @("--samples", "$spp", "--denoise", "--width", "1920", "--height", "1080")) }
"all done $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
