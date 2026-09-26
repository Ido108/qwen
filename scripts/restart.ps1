param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'launch.ps1') -Restart -NoBrowser:$NoBrowser
exit $LASTEXITCODE
