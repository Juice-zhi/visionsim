# Time the acceleration study's methods on the NVIDIA GPU, once it is free (see wait_gpu.ps1): Cycles with OptiX,
# EEVEE, the PyTorch prototypes with CUDA, and Unreal Engine. Results only differ from the CPU's in time, so renders
# go to their own folders and are only used for timing. Run from the repository root.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot/../../../..")
$env:PYTHONUTF8 = "1"
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$R = "runs/fog-comparison"
$A = "$R/acceleration/gpu"
$S = "examples/medium/experiment"
New-Item -ItemType Directory -Force "$A/logs" | Out-Null
$day = @(254, 255, 256, 257, 258, 259, 260, 261, 262, 345)
$pair = @(254, 255, 345)

function Mc($name, $blend, $frames, [string[]]$extra) {
    & E:\Blender\blender.exe -b $blend --python-exit-code 1 --python "$S/acceleration/render_mc.py" -- "$A/$name" `
        --frame @frames --device optix --gpu-denoise @extra *> "$A/logs/$name.log"
    "$name exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
}

function Eevee($name, $blend, $frames, [string[]]$extra) {
    & E:\Blender\blender.exe -b $blend --python-exit-code 1 --python "$S/acceleration/render_eevee.py" -- `
        "$A/$name" --frame @frames @extra *> "$A/logs/$name.log"
    "$name exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
}

$ms = @("--volume-bounces", "16")
foreach ($spp in 4, 16, 64, 256, 1024) { Mc "day_spp$spp" "$R/fog_ref.blend" $day ($ms + @("--samples", "$spp", "--denoise")) }
Mc "day_default" "$R/fog_ref.blend" $day ($ms + @("--samples", "256", "--adaptive", "0.05", "--denoise"))
$night = "$R/variants/fog_point_shadow.blend"
foreach ($spp in 16, 64, 256, 1024) { Mc "night_spp$spp" $night $pair @("--volume-bounces", "0", "--samples", "$spp", "--denoise") }
foreach ($spp in 16, 64, 256) { Mc "hd_spp$spp" "$R/fog_ref.blend" @(254, 255, 256) ($ms + @("--samples", "$spp", "--denoise", "--width", "1920", "--height", "1080")) }
Mc "hd_default" "$R/fog_ref.blend" @(254, 255, 256) ($ms + @("--samples", "256", "--adaptive", "0.05", "--denoise", "--width", "1920", "--height", "1080"))

Eevee "eevee_day_default" "$R/fog_ref.blend" $day @()
Eevee "eevee_day_1sample" "$R/fog_ref.blend" $day @("--taa", "1")
Eevee "eevee_day_fine" "$R/fog_ref.blend" $pair @("--tile", "2", "--slices", "256", "--end", "300")
Eevee "eevee_night_default" $night $pair @()
Eevee "eevee_hd_1sample" "$R/fog_ref.blend" @(254, 255, 256) @("--taa", "1")

$env:ACCEL_DEVICE = "cuda"
& .conda/python.exe "$S/acceleration/exp_gpu.py" *> "$A/logs/exp_gpu.log"
"exp_gpu exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
Remove-Item Env:\ACCEL_DEVICE

& "$S/acceleration/unreal/run_unreal.ps1" -Steps time -Tag rtx5080
"unreal exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
"all done $(Get-Date -Format s)" | Out-File "$A/logs/progress.txt" -Append -Encoding utf8
