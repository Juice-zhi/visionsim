$ErrorActionPreference = "Continue"
# Run with visionsim's environment activated and Blender on the PATH
Set-Location (Resolve-Path "$PSScriptRoot/../../..")  # the repository root
$env:PYTHONUTF8 = "1"
$R = "runs/fog-comparison"
$log = "$R/render_times.txt"
"refs started $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8

function Timed($name, [scriptblock]$block) {
    $t = Measure-Command { & $block }
    "{0} exit={1} {2:n1}s" -f $name, $LASTEXITCODE, $t.TotalSeconds | Out-File $log -Append -Encoding utf8
}

# Reference: Cycles volumetric path tracing, 4096 spp, no adaptive sampling nor denoising, with volume passes,
# and a second seed to measure the reference's own Monte Carlo noise
Timed "cycles_ref" { blender -b "$R/fog_ref.blend" --python-exit-code 1 --python examples/medium/experiment/render_passes.py -- "$R/cycles_ref" --samples 4096 *> "$R/cycles_ref.log" }
Timed "cycles_ref_seed1" { blender -b "$R/fog_ref.blend" --python-exit-code 1 --python examples/medium/experiment/render_passes.py -- "$R/cycles_ref_seed1" --samples 4096 --seed 1 --no-volume-passes *> "$R/cycles_ref_seed1.log" }
"refs finished $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
