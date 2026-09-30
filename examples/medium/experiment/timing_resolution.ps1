param([int]$Width = 800, [int]$Height = 800)
$ErrorActionPreference = "Continue"
# Run with visionsim's environment activated and Blender on the PATH
Set-Location (Resolve-Path "$PSScriptRoot/../../..")  # the repository root
$env:PYTHONUTF8 = "1"
$R = "runs/fog-comparison"
$O = "$R/res${Width}x${Height}"
$log = "$O/timing.txt"
New-Item -ItemType Directory -Force $O | Out-Null
"started $(Get-Date -Format s)" | Out-File $log -Encoding utf8

function Timed($name, [scriptblock]$block) {
    $t = Measure-Command { & $block }
    "{0} exit={1} {2:n1}s for 10 frames" -f $name, $LASTEXITCODE, $t.TotalSeconds | Out-File $log -Append -Encoding utf8
}

# The same 10 frames at another resolution, e.g. VisionSIM-50's 800x800: the scene without fog (with the depths the
# medium needs), with the fog volume in Cycles with single and multiple scattering (render_ms.ps1), and a multiple
# scattering reference to measure quality at this resolution
$common = @("--config.keyframe-multiplier", "5", "--frame-start", "255", "--frame-end", "264", "--config.width", "$Width",
            "--config.height", "$Height", "--config.frames.file-format", "OPEN_EXR", "--config.frames.bit-depth", "32",
            "--config.frames.exr-codec", "ZIP", "--config.log-dir", "$O/logs")
Timed "clear" { visionsim blender.render-animation "$R/demo.blend" "$O/clear" @common --config.include-depths --config.depths.no-preview --config.depths.exr-codec ZIP *> "$O/clear.log" }
Timed "cycles_default" { visionsim blender.render-animation "$R/fog_default.blend" "$O/cycles_default" @common *> "$O/cycles_default.log" }
Timed "cycles_default_ms" { visionsim blender.render-animation "$R/fog_default_ms.blend" "$O/cycles_default_ms" @common *> "$O/cycles_default_ms.log" }
Timed "ms_cycles_ref" { blender -b "$R/fog_ref.blend" --python-exit-code 1 --python examples/medium/experiment/render_passes.py -- "$O/ms_cycles_ref" --samples 4096 --volume-bounces 16 --no-volume-passes --width $Width --height $Height --frame 255 256 257 258 259 260 261 262 263 264 *> "$O/ms_cycles_ref.log" }
"finished $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
