# Wait until the NVIDIA GPU is free: less than 3 GB of memory used and under 10% utilization for 5 minutes in a row,
# checked every 30 s, for at most -Minutes minutes. Prints "idle" or "busy" when done.
param([int]$Minutes = 110)
$deadline = (Get-Date).AddMinutes($Minutes)
$quiet = 0
while ((Get-Date) -lt $deadline) {
    $line = (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ","
    if ([int]$line[0] -lt 3000 -and [int]$line[1] -lt 10) { $quiet += 1 } else { $quiet = 0 }
    if ($quiet -ge 10) { "idle $(Get-Date -Format s)"; exit 0 }
    Start-Sleep -Seconds 30
}
"busy $(Get-Date -Format s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
