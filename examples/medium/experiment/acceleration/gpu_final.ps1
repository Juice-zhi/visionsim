# Renders for the final comparison of the closed form with Cycles (analysis/final_benchmark.py), on the NVIDIA GPU
# (OptiX, OIDN on the GPU) or on the CPU: frames without fog at a few samples per pixel, which are the closed form's
# input, so that their time and noise count towards its own, and Cycles with fog over the range of samples per pixel
# that brackets the closed form's error and flicker, at 320x180 and 1920x1080. Renders and logs go to
# runs/fog-comparison/acceleration/<Out>; renders whose log is complete are skipped, so that a new folder re-times
# everything in one session. Run from the repository root once the GPU (or the CPU) is free, e.g.
#     gpu_final.ps1                         # GPU, into gpu/ (fog renders at 4 spp and more come from gpu_rerun.ps1)
#     gpu_final.ps1 -Out gpu2 -All          # GPU, every render again, into gpu2/
#     gpu_final.ps1 -Device cpu -Out cpu2 -All
#     gpu_final.ps1 -All -RefsOnly -Lock <dir>   # only the 1080p references, taking a lock around each render
# With -Lock, each render first takes the lock, a directory that other GPU users create and remove too, and is skipped
# if someone else's jobs then use the GPU (more than 6 GB or 20%).
param([string]$Device = "optix", [string]$Out = "gpu", [switch]$All, [switch]$RefsOnly, [string]$Lock = "")
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot/../../../..")
$env:PYTHONUTF8 = "1"
if ($Device -eq "cpu") { $env:CUDA_VISIBLE_DEVICES = "-1" } else { Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
$R = "runs/fog-comparison"
$A = "$R/acceleration/$Out"
$S = "examples/medium/experiment"
New-Item -ItemType Directory -Force "$A/logs" | Out-Null
$day = @(254, 255, 256, 257, 258, 259, 260, 261, 262, 345)  # 254 warms up Blender and is left out of timings
$pair = @(254, 255, 345)
$hd = @(254, 255, 256)
$size = @("--width", "1920", "--height", "1080")
$deviceArgs = if ($Device -eq "cpu") { @("--device", "cpu", "--threads", "12") } else { @("--device", $Device, "--gpu-denoise") }

function Release { if ($Lock) { Remove-Item $Lock -Force -ErrorAction SilentlyContinue } }
function Acquire {
    if (-not $Lock) { return $true }
    while ($true) {
        try { New-Item -ItemType Directory $Lock -ErrorAction Stop | Out-Null; break } catch { Start-Sleep -Seconds 5 }
    }
    # Utilization is averaged over the last moments, so a job that just ended still shows: busy for 30 s means others
    for ($i = 0; $i -lt 6; $i++) {
        Start-Sleep -Seconds 5
        $gpu = (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ","
        if ([int]$gpu[0] -lt 6000 -and [int]$gpu[1] -lt 20) { return $true }
    }
    Release
    return $false
}

function Mc($name, $blend, $frames, [string[]]$extra) {
    if ($RefsOnly) { return }
    if (Test-Path "$A/logs/$name.log") { if (Select-String -Quiet "FRAME_TIME frame=$($frames[-1]) " "$A/logs/$name.log") { return } }
    if (-not (Acquire)) { "$name skipped: GPU busy $(Get-Date -Format s)" | Out-File "$A/logs/progress_final.txt" -Append -Encoding utf8; return }
    & E:\Blender\blender.exe -b $blend --python-exit-code 1 --python "$S/acceleration/render_mc.py" -- "$A/$name" `
        --frame @frames @deviceArgs --denoise @extra *> "$A/logs/$name.log"
    Release
    "$name exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress_final.txt" -Append -Encoding utf8
}

# Without fog: the day scene and the night scene's lamp behind a cube
$clearNight = "$R/acceleration/clear_point_shadow.blend"
foreach ($spp in 1, 4, 16, 64) {
    Mc "clear_day_spp$spp" "$R/demo.blend" $day @("--samples", "$spp")
    Mc "clear_night_spp$spp" $clearNight $pair @("--samples", "$spp")
}
foreach ($spp in 1, 4, 16) {
    Mc "clear_hd_spp$spp" "$R/demo.blend" $hd ($size + @("--samples", "$spp"))
    Mc "clear_night_hd_spp$spp" $clearNight $hd ($size + @("--samples", "$spp"))
}

# With fog, as the references: light scattered up to 16 times, unclamped at night
$night = "$R/variants/fog_point_shadow.blend"
$fogSpp = if ($All) { @(1, 2, 4, 16, 64, 256) } else { @(1, 2) }
$fogHdSpp = if ($All) { @(1, 4, 16) } else { @(1, 4) }
foreach ($spp in $fogSpp) {
    Mc "day_spp$spp" "$R/fog_ref.blend" $day @("--volume-bounces", "16", "--samples", "$spp")
    Mc "night_full_spp$spp" $night $pair @("--volume-bounces", "16", "--no-clamp", "--samples", "$spp")
}
foreach ($spp in $fogHdSpp) {
    Mc "hd_spp$spp" "$R/fog_ref.blend" $hd ($size + @("--volume-bounces", "16", "--samples", "$spp"))
    Mc "night_full_hd_spp$spp" $night $hd ($size + @("--volume-bounces", "16", "--no-clamp", "--samples", "$spp"))
}
# References at 1080p, on the GPU only, into hd_refs/, once: each one as two halves of different seeds, whose mean is the
# reference and whose difference measures its noise; with fog (as the references at 320x180), without fog at night
# (the closed form's input; by day it is hd/clear), and the normals of surfaces
if ($All -and $Device -ne "cpu") {
    $H = "$R/acceleration/hd_refs"
    New-Item -ItemType Directory -Force "$H/logs" | Out-Null
    $two = @(255, 256)
    function Ref($name, $blend, [string[]]$extra) {
        if (Test-Path "$H/logs/$name.log") { if (Select-String -Quiet "FRAME_TIME frame=256 " "$H/logs/$name.log") { return } }
        if (-not (Acquire)) { "$name skipped: GPU busy $(Get-Date -Format s)" | Out-File "$H/logs/progress.txt" -Append -Encoding utf8; return }
        & E:\Blender\blender.exe -b $blend --python-exit-code 1 --python "$S/acceleration/render_mc.py" -- "$H/$name" `
            --frame @two --device $Device @size @extra *> "$H/logs/$name.log"
        Release
        "$name exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$H/logs/progress.txt" -Append -Encoding utf8
    }
    foreach ($seed in 0, 1) {
        Ref "day_seed$seed" "$R/fog_ref.blend" @("--volume-bounces", "16", "--samples", "2048", "--seed", "$seed")
        Ref "night_seed$seed" $night @("--volume-bounces", "16", "--no-clamp", "--samples", "8192", "--seed", "$seed")
    }
    Ref "clear_night" $clearNight @("--samples", "1024", "--denoise", "--gpu-denoise")
    Ref "normals" "$R/demo.blend" @("--samples", "16", "--normals")
}
"all done $(Get-Date -Format s)" | Out-File "$A/logs/progress_final.txt" -Append -Encoding utf8
