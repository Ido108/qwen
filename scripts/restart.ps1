param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paths.ps1')
$runtimePath = Join-Path $appRoot 'runtime.json'
if (Test-Path -LiteralPath $runtimePath) {
    $saved = Get-Content -Raw -LiteralPath $runtimePath | ConvertFrom-Json
    $health = $null
    try { $health = Invoke-RestMethod -Uri "$($saved.url)/api/health" -TimeoutSec 3 } catch { }
    if ($health) {
        if ($health.app -ne 'image-studio-qwen21' -or $health.app_dir -ne $appRoot -or $health.data_dir -ne $dataRoot) {
            throw 'The running app does not match this Qwen folder.'
        }
        $queue = Invoke-RestMethod -Uri "$($saved.url)/api/queue" -TimeoutSec 3
        if ($queue.active -or @($queue.pending).Count -gt 0) {
            throw 'Qwen is processing or has queued jobs. Finish those jobs before restarting. Nothing was stopped.'
        }
        $listener = Get-NetTCPConnection -State Listen -LocalPort $saved.port -ErrorAction Stop | Select-Object -First 1
        $serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
        if ($serverProcess.CommandLine -notmatch 'uvicorn\s+backend\.server:app') {
            throw 'The listening process is not the Qwen server. Nothing was stopped.'
        }
        # Recheck immediately before stopping the verified idle server.
        $queue = Invoke-RestMethod -Uri "$($saved.url)/api/queue" -TimeoutSec 3
        if ($queue.active -or @($queue.pending).Count -gt 0) {
            throw 'A job was added. Finish it before restarting. Nothing was stopped.'
        }
        Stop-Process -Id $listener.OwningProcess -ErrorAction Stop
        Wait-Process -Id $listener.OwningProcess -Timeout 10 -ErrorAction SilentlyContinue
    }
}
& (Join-Path $PSScriptRoot 'launch.ps1') -NoBrowser:$NoBrowser
