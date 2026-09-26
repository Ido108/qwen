param([switch]$NoBrowser, [switch]$NoConsole)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paths.ps1')
if (-not (Test-QwenSetup)) {
    Write-Host 'Installing the local app environment...'
    & (Join-Path $PSScriptRoot 'setup.ps1')
}
function Test-Studio([string]$Address) {
    try {
        $health = Invoke-RestMethod -Uri "$Address/api/health" -TimeoutSec 2
        return $health.app -eq 'image-studio-qwen21' -and $health.app_dir -eq $appRoot -and $health.data_dir -eq $dataRoot
    } catch { return $false }
}
$studioPort = 47821
$runtimePath = Join-Path $appRoot 'runtime.json'
if (Test-Path -LiteralPath $runtimePath) {
    try {
        $old = Get-Content -Raw -LiteralPath $runtimePath | ConvertFrom-Json
        if ($old.port -ge 47821 -and $old.port -le 47840 -and (Test-Studio "http://127.0.0.1:$($old.port)")) { $studioPort = [int]$old.port }
    } catch { }
}
$studioUrl = "http://127.0.0.1:$studioPort"
if (-not (Test-Studio $studioUrl)) {
    while ($studioPort -le 47840) {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $studioPort -ErrorAction SilentlyContinue
        if (-not $listener) { break }
        $studioPort++
    }
    if ($studioPort -gt 47840) { throw 'No free local app port between 47821 and 47840.' }
    $studioUrl = "http://127.0.0.1:$studioPort"
    $env:QWEN_DATA_DIR = $dataRoot
    $env:PYTHONUTF8 = '1'
    $logDir = Join-Path $dataRoot 'logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $appProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u','-m','uvicorn','backend.server:app','--host','127.0.0.1','--port',"$studioPort") -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'server.out.log') -RedirectStandardError (Join-Path $logDir 'server.err.log')
    @{ port=$studioPort; pid=$appProcess.Id; url=$studioUrl } | ConvertTo-Json | Set-Content -LiteralPath $runtimePath -Encoding UTF8
    Write-Host 'Starting Qwen...'
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($appProcess.HasExited) { throw "Qwen stopped. Details: $(Join-Path $logDir 'server.err.log')" }
        if (Test-Studio $studioUrl) { $ready=$true; break }
        Start-Sleep -Milliseconds 1000
    }
    if (-not $ready) { throw "The app did not become ready. Details: $(Join-Path $logDir 'server.err.log')" }
}
if (-not $NoBrowser) {
    $edgePaths = @('C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','C:\Program Files\Microsoft\Edge\Application\msedge.exe')
    $edge = $edgePaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($edge) {
        Start-Process -FilePath $edge -ArgumentList @("--app=$studioUrl",'--window-size=1440,1000')
    } else { Start-Process $studioUrl }
}
Write-Host "Qwen is ready: $studioUrl"
if (-not $NoBrowser -and -not $NoConsole) {
    $Host.UI.RawUI.WindowTitle = 'Qwen - Activity'
    & $pythonPath -u (Join-Path $PSScriptRoot 'watch_logs.py') --url $studioUrl --data-dir $dataRoot
}
