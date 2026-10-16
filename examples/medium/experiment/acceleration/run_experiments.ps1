# Run the acceleration study's experiments one after the other, on the CPU, once render_all.ps1 has finished.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot/../../../..")
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = ""
$A = "runs/fog-comparison/acceleration"
New-Item -ItemType Directory -Force "$A/results" | Out-Null
foreach ($name in $args) {
    $start = Get-Date
    & .conda/python.exe "examples/medium/experiment/acceleration/exp_$name.py" *> "$A/logs/exp_$name.log"
    "exp_$name exit=$LASTEXITCODE seconds=$([int]((Get-Date) - $start).TotalSeconds) $(Get-Date -Format s)" | Out-File "$A/logs/experiments.txt" -Append -Encoding utf8
}
