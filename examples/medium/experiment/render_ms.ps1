$ErrorActionPreference = "Continue"
# Run with visionsim's environment activated and Blender on the PATH
Set-Location (Resolve-Path "$PSScriptRoot/../../..")  # the repository root
$env:PYTHONUTF8 = "1"
$R = "runs/fog-comparison"
$M = "$R/ms"
$log = "$R/render_times.txt"
New-Item -ItemType Directory -Force $M | Out-Null
"ms started $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8

function Timed($name, [scriptblock]$block) {
    $t = Measure-Command { & $block }
    "{0} exit={1} {2:n1}s" -f $name, $LASTEXITCODE, $t.TotalSeconds | Out-File $log -Append -Encoding utf8
}

# Same renders as render_all.ps1 and render_refs.ps1, with multiple scattering: light may scatter up to 16 times in the
# fog (bounces.py shows that 8 are enough), instead of Blender's default of 0 volume bounces, i.e. single scattering
blender -b "$R/demo.blend" --python-exit-code 1 --python examples/medium/experiment/add_fog.py -- `
    examples/medium/media/ground_fog.json "$R/fog_default_ms.blend" --volume-bounces 16 *> "$M/add_fog.log"
$common = @("--config.keyframe-multiplier", "5", "--config.frames.file-format", "OPEN_EXR", "--config.frames.bit-depth", "32",
            "--config.frames.exr-codec", "ZIP", "--config.log-dir", "$M/logs")
Timed "ms_cycles_default" { visionsim blender.render-animation "$R/fog_default_ms.blend" "$M/cycles_default" @common *> "$M/cycles_default.log" }
Timed "ms_cycles_nodenoise" { visionsim blender.render-animation "$R/fog_default_ms.blend" "$M/cycles_nodenoise" @common --config.no-use-denoising *> "$M/cycles_nodenoise.log" }
Timed "ms_cycles_ref" { blender -b "$R/fog_ref.blend" --python-exit-code 1 --python examples/medium/experiment/render_passes.py -- "$M/cycles_ref" --samples 4096 --volume-bounces 16 *> "$M/cycles_ref.log" }
Timed "ms_cycles_ref_seed1" { blender -b "$R/fog_ref.blend" --python-exit-code 1 --python examples/medium/experiment/render_passes.py -- "$M/cycles_ref_seed1" --samples 4096 --volume-bounces 16 --seed 1 --no-volume-passes *> "$M/cycles_ref_seed1.log" }
"ms finished $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
