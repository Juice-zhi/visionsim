$ErrorActionPreference = "Continue"
# Run with visionsim's environment activated and Blender on the PATH
Set-Location (Resolve-Path "$PSScriptRoot/../../..")  # the repository root
$env:PYTHONUTF8 = "1"
$R = "runs/fog-comparison"
$log = "$R/render_times.txt"
"started $(Get-Date -Format s)" | Out-File $log -Encoding utf8

function Timed($name, [scriptblock]$block) {
    $t = Measure-Command { & $block }
    "{0} exit={1} {2:n1}s" -f $name, $LASTEXITCODE, $t.TotalSeconds | Out-File $log -Append -Encoding utf8
}

$common = @("--config.keyframe-multiplier", "5", "--config.frames.file-format", "OPEN_EXR", "--config.frames.bit-depth", "32",
            "--config.frames.exr-codec", "ZIP", "--config.log-dir", "$R/logs")

# Scene without fog, rendered with visionsim's default settings, with everything the medium and ToF sensor need
Timed "clear" { visionsim blender.render-animation "$R/demo.blend" "$R/clear" @common --config.include-depths --config.depths.no-preview `
    --config.depths.exr-codec ZIP --config.include-normals --config.normals.no-preview --config.normals.exr-codec ZIP `
    --config.include-diffuse-pass --config.diffuse-pass.exr-codec ZIP --config.include-lighting *> "$R/clear.log" }

# Same scene with a fog volume, rendered by Cycles with visionsim's default settings (256 spp, adaptive, denoised)
Timed "cycles_default" { visionsim blender.render-animation "$R/fog_default.blend" "$R/cycles_default" @common *> "$R/cycles_default.log" }
"finished $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
# The references are rendered by render_refs.ps1, which should run once these renders are done
