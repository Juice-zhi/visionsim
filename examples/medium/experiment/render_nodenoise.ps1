$ErrorActionPreference = "Continue"
# Run with visionsim's environment activated and Blender on the PATH
Set-Location (Resolve-Path "$PSScriptRoot/../../..")  # the repository root
$env:PYTHONUTF8 = "1"
$R = "runs/fog-comparison"
$common = @("--config.keyframe-multiplier", "5", "--config.frames.file-format", "OPEN_EXR", "--config.frames.bit-depth", "32",
            "--config.frames.exr-codec", "ZIP", "--config.log-dir", "$R/logs_nodenoise")
# Same as cycles_default, without the denoiser, i.e. the raw Monte Carlo noise of 256 spp with adaptive sampling
$t = Measure-Command { visionsim blender.render-animation "$R/fog_default.blend" "$R/cycles_nodenoise" @common --config.no-use-denoising *> "$R/cycles_nodenoise.log" }
"cycles_nodenoise exit={0} {1:n1}s" -f $LASTEXITCODE, $t.TotalSeconds | Out-File "$R/render_times.txt" -Append -Encoding utf8
