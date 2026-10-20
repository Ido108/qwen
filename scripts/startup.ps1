param([ValidateSet('Qwen','MoiseGpuServer')][string]$Service='Qwen')
$ErrorActionPreference='Stop'
if ($Service -eq 'Qwen') {
    . (Join-Path $PSScriptRoot 'paths.ps1')
    $Host.UI.RawUI.WindowTitle='Qwen - Activity'
    # Startup must not reinstall or update packages. The normal launcher handles setup.
    & $pythonPath -u (Join-Path $PSScriptRoot 'desktop.py') --data-dir $dataRoot --no-browser
    exit $LASTEXITCODE
}
$moiseRoot=Join-Path $env:USERPROFILE 'REPOS\MoiseGpuServer'
if (Get-NetTCPConnection -State Listen -LocalPort 4000 -ErrorAction SilentlyContinue) { exit 0 }
$startupLog=Join-Path $moiseRoot '.local\startup.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $startupLog) -Force | Out-Null
Start-Transcript -Path $startupLog -Append | Out-Null
try {
    # Preserve the canonical launcher's normal handling of native tool warnings.
    $ErrorActionPreference='Continue'
    & (Join-Path $moiseRoot 'GPU-MOISE.ps1')
    $serverExit=$LASTEXITCODE
} finally { Stop-Transcript | Out-Null }
exit $serverExit
