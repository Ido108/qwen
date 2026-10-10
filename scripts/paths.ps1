$appRoot = Split-Path -Parent $PSScriptRoot
$settingsPath = Join-Path $appRoot 'local-settings.json'
$localSettings = @{}
if (Test-Path -LiteralPath $settingsPath) { $localSettings = Get-Content -Raw -LiteralPath $settingsPath | ConvertFrom-Json }
$dataRoot = $env:QWEN_DATA_DIR
if (-not $dataRoot) { $dataRoot = $localSettings.data_dir }
if (-not $dataRoot) {
    # First run: ask where models and generated files go. The default is drive C.
    $dataRoot = Join-Path $env:SystemDrive 'AI\qwen-image-studio'
    if (-not [Console]::IsInputRedirected) {
        Write-Host ''
        Write-Host 'Where should Qwen store its models, Python environment and generated files?'
        Write-Host 'Qwen needs at least 40 GB of free space. Available drives:'
        Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free } | ForEach-Object { Write-Host ('  {0}:  {1:N0} GB free' -f $_.Name, ($_.Free / 1GB)) }
        while ($true) {
            $answer = (Read-Host "Press Enter for $dataRoot, or type a drive letter or folder").Trim().Trim('"')
            if (-not $answer) { $answer = $dataRoot }
            if ($answer -match '^[A-Za-z]:?\\?$') { $answer = $answer.Substring(0,1).ToUpper() + ':\AI\qwen-image-studio' }
            if (-not ($answer -match '^[A-Za-z]:\\')) { Write-Host 'Type a drive letter (for example D) or a full folder path (for example D:\Qwen).'; continue }
            $root = [IO.Path]::GetPathRoot($answer)
            if (-not (Test-Path -LiteralPath $root)) { Write-Host "Drive $root is not available."; continue }
            $free = ([IO.DriveInfo]::new($root)).AvailableFreeSpace
            if ($free -lt 40GB) { Write-Host ('Drive {0} has only {1:N0} GB free. Choose a drive with at least 40 GB.' -f $root, ($free / 1GB)); continue }
            $dataRoot = [IO.Path]::GetFullPath($answer)
            break
        }
    }
    @{ data_dir=$dataRoot; runtime_dir=$dataRoot } | ConvertTo-Json | Set-Content -LiteralPath $settingsPath -Encoding UTF8
    $localSettings = Get-Content -Raw -LiteralPath $settingsPath | ConvertFrom-Json
}
$runtimeRoot = $env:QWEN_RUNTIME_DIR
if (-not $runtimeRoot) { $runtimeRoot = $localSettings.runtime_dir }
if (-not $runtimeRoot) { $runtimeRoot = $dataRoot }
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
# the small Python interpreter on NTFS; model files and caches can stay on exFAT.
if (([IO.DriveInfo]::new([IO.Path]::GetPathRoot($runtimeRoot))).DriveFormat -ne 'NTFS') { $pythonInstallRoot = Join-Path $env:LOCALAPPDATA 'Qwen\python' }
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
    $files = @('requirements\lock.txt','package-lock.json','scripts\setup.ps1','scripts\paths.ps1','scripts\download_examples.py','scripts\check_runtime.py','reference\example-assets.json','index.html') | ForEach-Object { Join-Path $appRoot $_ }
    $files += Get-ChildItem -LiteralPath (Join-Path $appRoot 'src') -File -Recurse | Sort-Object FullName | ForEach-Object FullName
    $hashes = $files | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
    return $hashes -join ':'
}
function Test-QwenSetup {
    if (-not (Test-Path -LiteralPath $pythonPath) -or -not (Test-Path -LiteralPath $setupStamp) -or -not (Test-Path -LiteralPath (Join-Path $appRoot 'dist\index.html'))) { return $false }
    try { $stamp = Get-Content -Raw -LiteralPath $setupStamp | ConvertFrom-Json; return $stamp.fingerprint -eq (Get-QwenSetupFingerprint) } catch { return $false }
}
