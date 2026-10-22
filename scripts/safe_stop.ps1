# Pause the rep1 pipeline safely before sleep or shutdown (D86).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\safe_stop.ps1
#
# 1. Stops the orchestrators (resume_rep1.py, rep1_runbook.py, post_rep1_queue.py).
#    They only launch and gate stages; every stage they launch is resumable.
# 2. If the traces stage is running, waits for its next trace to be written and
#    flushed, then stops it, so at most the just-started problem is redone.
# 3. If another stage is running (acts, nla, forcing, Stage 1), it is NOT
#    stopped automatically: it prints what is running so you can decide.
#    Those stages resume per item; reconstruction restarts its phase.
$root = Split-Path -Parent $PSScriptRoot
$py = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($p in $py | Where-Object { $_.CommandLine -match 'resume_rep1|rep1_runbook|post_rep1_queue' }) {
    Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue
    "stopped orchestrator $($p.ProcessId)"
}
$traces = $py | Where-Object { $_.CommandLine -like '*generate_traces.py --config replication*' }
$other = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'scripts[\\/]' -and $_.CommandLine -notlike '*generate_traces.py*' }
if (-not $traces) {
    if ($other) { "a non-trace stage is running; not stopped:"; $other | ForEach-Object { "  $($_.ProcessId) $($_.CommandLine)" }; exit 2 }
    "nothing is running; safe to shut down"; exit 0
}
$f = Join-Path $root 'runs\rep1\traces\traces.jsonl'
$start = (Get-Content $f | Measure-Object -Line).Lines
"waiting for trace $($start + 1) to be written (up to 45 min)..."
$deadline = (Get-Date).AddMinutes(45)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if ((Get-Content $f | Measure-Object -Line).Lines -gt $start) {
        Start-Sleep -Milliseconds 300
        $traces = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*generate_traces.py --config replication*' }
        foreach ($p in $traces) { Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
        "traces stage stopped after trace $($start + 1); safe to shut down"
        exit 0
    }
}
"no trace finished within 45 minutes; nothing stopped"; exit 1
