$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path (Split-Path -Parent $appRoot) 'Qwen.lnk'
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$env:SystemRoot\System32\cmd.exe"
$shortcut.Arguments = '/c ""' + (Join-Path $appRoot 'Start Qwen.bat') + '""'
$shortcut.WorkingDirectory = $appRoot
$shortcut.IconLocation = (Join-Path $appRoot 'public\branding\app.ico') + ',0'
$shortcut.Description = 'Local image creation and editing with Qwen-Image-2.1'
$shortcut.WindowStyle = 1
$shortcut.Save()
Write-Host "Created $shortcutPath"
