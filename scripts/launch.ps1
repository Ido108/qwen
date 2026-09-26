param([switch]$NoBrowser, [switch]$NoConsole, [switch]$Restart, [switch]$Stop)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paths.ps1')
if (-not $Stop -and -not (Test-QwenSetup)) {
    Write-Host 'Installing the local app environment...'
    & (Join-Path $PSScriptRoot 'setup.ps1')
}
$Host.UI.RawUI.WindowTitle = 'Qwen - Activity'
$launchArgs = @('-u', (Join-Path $PSScriptRoot 'desktop.py'), '--data-dir', $dataRoot)
if ($NoBrowser) { $launchArgs += '--no-browser' }
if ($NoConsole) { $launchArgs += '--no-console' }
if ($Restart) { $launchArgs += '--restart' }
if ($Stop) { $launchArgs += '--stop' }
# Optional overrides isolate launcher verification from the real app.
if ($env:QWEN_APP_PORT) { $launchArgs += @('--port', $env:QWEN_APP_PORT) }
if ($env:QWEN_STATE_FILE) { $launchArgs += @('--runtime-file', $env:QWEN_STATE_FILE) }
& $pythonPath @launchArgs
exit $LASTEXITCODE
