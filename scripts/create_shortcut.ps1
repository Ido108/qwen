$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path (Split-Path -Parent $appRoot) 'Qwen.lnk'
$shortcut = $shell.CreateShortcut($shortcutPath)
$launcher = Join-Path $appRoot 'Start Qwen.bat'
$icon = Join-Path $appRoot 'public\branding\app.ico'
foreach ($file in @($launcher, $icon)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file is missing: $file" }
}
$shortcut.TargetPath = "$env:SystemRoot\explorer.exe"
$shortcut.Arguments = '"' + $launcher + '"'
$shortcut.WorkingDirectory = $appRoot
$shortcut.IconLocation = $icon + ',0'
$shortcut.Description = 'Launch Qwen'
$shortcut.WindowStyle = 1
$shortcut.Save()
$saved = $shell.CreateShortcut($shortcutPath)
if ($saved.TargetPath -ne "$env:SystemRoot\explorer.exe" -or $saved.Arguments -ne ('"' + $launcher + '"') -or $saved.WorkingDirectory -ne $appRoot -or $saved.IconLocation -ne ($icon + ',0') -or $saved.WindowStyle -ne 1) {
    throw 'The Qwen shortcut could not be verified.'
}
Write-Host "Created $shortcutPath"
