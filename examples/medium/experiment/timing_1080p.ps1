$ErrorActionPreference = "Continue"
# Run with visionsim's environment activated and Blender on the PATH
Set-Location (Resolve-Path "$PSScriptRoot/../../..")  # the repository root
$env:PYTHONUTF8 = "1"
$R = "runs/fog-comparison"
$log = "$R/timing_1080p.txt"
"started $(Get-Date -Format s)" | Out-File $log -Encoding utf8
$common = @("--config.keyframe-multiplier", "5", "--frame-start", "255", "--frame-end", "264", "--config.width", "1920",
            "--config.height", "1080", "--config.frames.file-format", "OPEN_EXR", "--config.frames.bit-depth", "32",
            "--config.frames.exr-codec", "ZIP", "--config.log-dir", "$R/logs_1080p")

# Same frames at 1920x1080: the scene without fog (with what the medium needs), and with the fog volume in Cycles
$t = Measure-Command { visionsim blender.render-animation "$R/demo.blend" "$R/hd/clear" @common --config.include-depths --config.depths.no-preview --config.depths.exr-codec ZIP *> "$R/hd_clear.log" }
"clear_1080p exit={0} {1:n1}s for 10 frames" -f $LASTEXITCODE, $t.TotalSeconds | Out-File $log -Append -Encoding utf8
$t = Measure-Command { visionsim blender.render-animation "$R/fog_default.blend" "$R/hd/cycles_default" @common *> "$R/hd_cycles_default.log" }
"cycles_default_1080p exit={0} {1:n1}s for 10 frames" -f $LASTEXITCODE, $t.TotalSeconds | Out-File $log -Append -Encoding utf8
"finished $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
