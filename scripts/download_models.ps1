param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ModelIds)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paths.ps1')
if (-not (Test-QwenSetup)) { & (Join-Path $PSScriptRoot 'setup.ps1') }
& $pythonPath -u (Join-Path $PSScriptRoot 'install_models.py') @ModelIds
if ($LASTEXITCODE -ne 0) { throw 'Download did not finish. Run this BAT again to resume, or use Settings in Qwen.' }
