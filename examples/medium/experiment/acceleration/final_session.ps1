# The final comparison's timings, in one quiet session (analysis/final_benchmark.py): waits until the machine is free
# (the NVIDIA GPU under 15% utilization with under 6 GB used, and the CPU under 30%, for 10 minutes in a row, checked
# every minute), then on the GPU renders every Cycles configuration again (gpu_final.ps1 -All into gpu2/) and times the
# closed form, runs the tests that need CUDA, and does the same on the CPU (cpu2/, 12 threads). Before each further step
# it waits for 2 quiet minutes again, in case the machine got busy, e.g. when the user's own jobs resume. Progress goes to
# runs/fog-comparison/acceleration/results/final_session.txt. Run from the repository root.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot/../../../..")
$env:PYTHONUTF8 = "1"
$S = "examples/medium/experiment/acceleration"
$log = "runs/fog-comparison/acceleration/results/final_session.txt"

function Busy {
    $gpu = (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ","
    $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 2).CounterSamples |
        Measure-Object CookedValue -Average
    return ([int]$gpu[0] -ge 6000) -or ([int]$gpu[1] -ge 15) -or ($cpu.Average -ge 30)
}

function WaitFree($step, $minutes = 2) {
    $quiet = 0
    while ($quiet -lt $minutes) {
        if (Busy) { $quiet = 0 } else { $quiet += 1 }
        if ($quiet -lt $minutes) { Start-Sleep -Seconds 55 }
    }
    "$step start $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
}

function Done($step) { "$step done exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8 }

WaitFree "gpu renders" 10
& "$S/gpu_final.ps1" -Out gpu2 -All
Done "gpu renders"

WaitFree "gpu closed form"
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
& .conda/python.exe "$S/analysis/final_benchmark.py" --device cuda --renders gpu2 --factors 1 2 4 --repeats 5 `
    --out final_gpu.json *> "runs/fog-comparison/acceleration/results/final_gpu.log"
Done "gpu closed form"
& .conda/python.exe -m pytest tests/test_scattering.py tests/test_occlusion.py tests/test_surfaces.py `
    tests/test_upsampling.py tests/test_medium.py -q -p no:cacheprovider -rs *> "runs/fog-comparison/acceleration/results/final_cuda_tests.log"
Done "cuda tests"

WaitFree "cpu renders"
& "$S/gpu_final.ps1" -Device cpu -Out cpu2 -All
Done "cpu renders"

WaitFree "cpu closed form"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .conda/python.exe "$S/analysis/final_benchmark.py" --device cpu --renders cpu2 --factors 1 2 4 --hd-factors 4 `
    --repeats 3 --out final_cpu.json *> "runs/fog-comparison/acceleration/results/final_cpu.log"
Done "cpu closed form"
"all done $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
