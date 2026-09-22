$appRoot = Split-Path -Parent $PSScriptRoot
$settingsPath = Join-Path $appRoot 'local-settings.json'
$localSettings = @{}
if (Test-Path -LiteralPath $settingsPath) { $localSettings = Get-Content -Raw -LiteralPath $settingsPath | ConvertFrom-Json }
$dataRoot = $env:QWEN_DATA_DIR
if (-not $dataRoot) { $dataRoot = $localSettings.data_dir }
if (-not $dataRoot) {
    $existing = @('E:\AI\qwen-image-studio','D:\AI\qwen-image-studio') | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'installed-models.json') } | Select-Object -First 1
    if ($existing) { $dataRoot = $existing }
    else {
        $drive = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -gt 40GB } | Sort-Object Free -Descending | Select-Object -First 1
        if (-not $drive) { throw 'At least 40 GB of free space is needed for the app and its recommended image models.' }
        $dataRoot = Join-Path $drive.Root 'AI\qwen-image-studio'
    }
}
$runtimeRoot = $env:QWEN_RUNTIME_DIR
if (-not $runtimeRoot) { $runtimeRoot = $localSettings.runtime_dir }
if (-not $runtimeRoot) {
    if (Test-Path -LiteralPath 'D:\AI\qwen-image-studio\venv\Scripts\python.exe') { $runtimeRoot = 'D:\AI\qwen-image-studio' }
    elseif ((Get-PSDrive D -ErrorAction SilentlyContinue).Free -gt 12GB) { $runtimeRoot = 'D:\AI\qwen-image-studio' }
    else { $runtimeRoot = $dataRoot }
}
$dataRoot = [IO.Path]::GetFullPath($dataRoot)
$runtimeRoot = [IO.Path]::GetFullPath($runtimeRoot)
foreach ($location in @($dataRoot,$runtimeRoot)) {
    if (-not (Test-Path -LiteralPath ([IO.Path]::GetPathRoot($location)))) { throw "Connect the drive containing $location" }
    New-Item -ItemType Directory -Path $location -Force | Out-Null
}
$env:QWEN_DATA_DIR = $dataRoot
$env:QWEN_RUNTIME_DIR = $runtimeRoot
$env:PYTHONUTF8 = '1'
$env:UV_CACHE_DIR = Join-Path $dataRoot 'uv-cache'
$pythonInstallRoot = Join-Path $runtimeRoot 'python'
# uv creates version-directory links, which exFAT does not support. Keep only
# the small Python interpreter on NTFS; model files and caches can stay on E:.
if (([IO.DriveInfo]::new([IO.Path]::GetPathRoot($runtimeRoot))).DriveFormat -ne 'NTFS') {
    if ((Test-Path -LiteralPath 'D:\') -and ([IO.DriveInfo]::new('D:\')).DriveFormat -eq 'NTFS') { $pythonInstallRoot = 'D:\AI\qwen-image-studio\python' }
    else { $pythonInstallRoot = Join-Path $env:LOCALAPPDATA 'Qwen\python' }
}
$env:UV_PYTHON_INSTALL_DIR = $pythonInstallRoot
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:HF_HOME = Join-Path $dataRoot 'hf-cache'
$env:HF_XET_CACHE = Join-Path $dataRoot 'hf-cache\xet'
$env:UV_LINK_MODE = 'copy'
$env:npm_config_cache = Join-Path $dataRoot 'npm-cache'
$tempRoot = Join-Path $dataRoot 'temp'
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$pythonPath = Join-Path $runtimeRoot 'venv\Scripts\python.exe'
$toolsRoot = Join-Path $runtimeRoot 'tools'
$setupStamp = Join-Path $runtimeRoot 'setup-complete.json'

function Get-QwenSetupFingerprint {
    $files = @('requirements-lock.txt','package-lock.json','scripts\setup.ps1','scripts\paths.ps1','scripts\download_examples.py','scripts\check_runtime.py','reference\example-assets.json','index.html') | ForEach-Object { Join-Path $appRoot $_ }
    $files += Get-ChildItem -LiteralPath (Join-Path $appRoot 'src') -File -Recurse | Sort-Object FullName | ForEach-Object FullName
    $hashes = $files | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
    return $hashes -join ':'
}
function Test-QwenSetup {
    if (-not (Test-Path -LiteralPath $pythonPath) -or -not (Test-Path -LiteralPath $setupStamp) -or -not (Test-Path -LiteralPath (Join-Path $appRoot 'dist\index.html'))) { return $false }
    try { $stamp = Get-Content -Raw -LiteralPath $setupStamp | ConvertFrom-Json; return $stamp.fingerprint -eq (Get-QwenSetupFingerprint) } catch { return $false }
}
