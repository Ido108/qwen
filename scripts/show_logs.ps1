$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paths.ps1')
$studioUrl = 'http://127.0.0.1:47821'
$runtimePath = Join-Path $appRoot 'runtime.json'
if (Test-Path -LiteralPath $runtimePath) {
    try { $studioUrl = (Get-Content -Raw -LiteralPath $runtimePath | ConvertFrom-Json).url } catch { }
}
$Host.UI.RawUI.WindowTitle = 'Qwen - Activity'
& $pythonPath -u (Join-Path $PSScriptRoot 'desktop.py') --logs-only --data-dir $dataRoot
