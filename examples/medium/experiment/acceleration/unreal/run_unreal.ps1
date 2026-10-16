# Build the fog scenes in Unreal Engine, render them with Movie Render Queue, and time the volumetric fog in game.
#
#   run_unreal.ps1 [-Adapter preferAMD] [-Steps build,render,time] [-Tag igpu]
#
# -Adapter picks the GPU (preferAMD for the integrated GPU, empty for the default, discrete one), and -Tag names the
# folder of this machine's renders and timings, e.g. igpu or rtx5080. Run from the repository root.
param(
    [string]$Adapter = "",
    [string[]]$Steps = @("build", "render", "time"),
    [string]$Tag = "default"
)
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot/../../../../..")
$env:PYTHONUTF8 = "1"
$Editor = "E:\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
$Here = (Resolve-Path "$PSScriptRoot").Path
$Out = (Resolve-Path "runs/fog-comparison/acceleration").Path + "\unreal"
$Project = "$Out\FogBench\FogBench.uproject"
$Logs = "$Out\logs_$Tag"
New-Item -ItemType Directory -Force $Logs | Out-Null
$gpu = @()
if ($Adapter) { $gpu = @("-$Adapter") }

function Stamp($text) { "$text $(Get-Date -Format s)" | Out-File "$Logs\progress.txt" -Append -Encoding utf8 }

if ($Steps -contains "build") {
    if (-not (Test-Path $Project)) {
        New-Item -ItemType Directory -Force "$Out\FogBench" | Out-Null
        Copy-Item -Recurse -Force "$Here\project\*" "$Out\FogBench\"
    }
    & .conda/python.exe "$Here\export_scene.py" *> "$Logs\export.log"
    $env:FOG_SCENE = $Out
    $t = Measure-Command {
        & $Editor $Project -run=pythonscript -script="$Here\build_scene.py" -unattended -nosplash -nopause -stdout -FullStdOutLogOutput @gpu *> "$Logs\build.log"
    }
    Stamp "build exit=$LASTEXITCODE seconds=$([int]$t.TotalSeconds)"
}

if ($Steps -contains "render") {
    foreach ($level in "Day", "Night") {
        foreach ($preset in "high", "epic", "cinematic", "nofog") {
            $t = Measure-Command {
                & $Editor $Project "/Game/Fog/$level" -game "-LevelSequence=/Game/Fog/Seq_$level.Seq_$level" `
                    "-MoviePipelineConfig=/Game/Fog/P_$preset.P_$preset" -windowed -resx=320 -resy=180 -RenderOffscreen `
                    -Unattended -NoTextureStreaming -nosplash -stdout -FullStdOutLogOutput @gpu *> "$Logs\render_${level}_$preset.log"
            }
            Stamp "render $level $preset exit=$LASTEXITCODE seconds=$([int]$t.TotalSeconds)"
        }
    }
}

if ($Steps -contains "time") {
    # Frame times in game, through the camera at frame 345, from Unreal's CSV profiler with GPU timings, with each
    # quality of volumetric fog and without fog, at two resolutions
    $qualities = @{
        "high" = "r.VolumetricFog 1, r.VolumetricFog.GridPixelSize 16, r.VolumetricFog.GridSizeZ 64";
        "epic" = "r.VolumetricFog 1, r.VolumetricFog.GridPixelSize 8, r.VolumetricFog.GridSizeZ 128";
        "cinematic" = "r.VolumetricFog 1, r.VolumetricFog.GridPixelSize 4, r.VolumetricFog.GridSizeZ 128";
        "nofog" = "r.Fog 0, r.VolumetricFog 0"
    }
    foreach ($level in "Day", "Night") {
        foreach ($res in @(@(320, 180), @(1920, 1080))) {
            foreach ($q in "nofog", "high", "epic", "cinematic") {
                $name = "${level}_$($res[0])_$q"
                $cmds = "r.VSync 0, t.MaxFPS 0, $($qualities[$q])"
                $t = Measure-Command {
                    & $Editor $Project "/Game/Fog/$level" -game -windowed "-resx=$($res[0])" "-resy=$($res[1])" -RenderOffscreen `
                        -csvCaptureFrames=900 -csvGpuStats -ExitAfterCsvProfiling "-csvMetadata=name=$name" `
                        "-ExecCmds=$cmds" -Unattended -NoTextureStreaming -nosplash -stdout @gpu *> "$Logs\time_$name.log"
                }
                # The profiler names the file it wrote in the log
                $line = Select-String -Path "$Logs\time_$name.log" -Pattern "Writing CSV to file : (.+\.csv)" | Select-Object -Last 1
                if ($line) { Copy-Item $line.Matches[0].Groups[1].Value "$Logs\time_$name.csv" }
                Stamp "time $name exit=$LASTEXITCODE seconds=$([int]$t.TotalSeconds)"
            }
        }
    }
}
Stamp "done"
