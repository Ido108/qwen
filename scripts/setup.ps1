param([switch]$Force)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paths.ps1')
if ((Test-QwenSetup) -and -not $Force) { Write-Host 'Qwen is already installed.'; return }
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Path $toolsRoot -Force | Out-Null
$setupLock = $null
try {
    try { $setupLock = [IO.File]::Open((Join-Path $runtimeRoot 'setup.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None) }
    catch { throw 'Another Qwen setup is running. Wait for that window to finish.' }
    Write-Host 'Qwen setup - installing the app. Models are downloaded from Settings in the app.'
    Write-Host "App files: $appRoot"
    Write-Host "Python environment: $runtimeRoot"
    Write-Host "Models and images: $dataRoot"
    function Get-VerifiedArchive([string]$Url,[string]$Name,[string]$Sha256) {
        $archive = Join-Path $toolsRoot $Name
        if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) { Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) { throw "Download verification failed: $Name. Run Setup.bat again to retry." }
        return $archive
    }
    $uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if (-not $uvPath) {
        $uvPath = Join-Path $toolsRoot 'uv\uv.exe'
        if (-not (Test-Path -LiteralPath $uvPath)) {
            Write-Host '[1/5] Downloading the Python installer...'
            $archive = Get-VerifiedArchive 'https://github.com/astral-sh/uv/releases/download/0.12.17/uv-x86_64-pc-windows-msvc.zip' 'uv-0.12.17.zip' 'a252121d5b59398fcb137c6ea448176459a44010f33f67e0072305a637119ca7'
            Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $toolsRoot 'uv') -Force
        }
    }
    Write-Host '[2/5] Preparing Python 3.12...'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        & $uvPath venv --python 3.12.10 (Join-Path $runtimeRoot 'venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment. Check the connection and free disk space, then run Setup.bat again.' }
    }
    Write-Host '[3/5] Installing the image tools...'
    Push-Location $appRoot
    try {
        & $uvPath pip install --python $pythonPath -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation did not finish. Run Setup.bat again to resume.' }
        & $uvPath pip check --python $pythonPath
        if ($LASTEXITCODE -ne 0) { throw 'Some installed dependencies are incompatible. See the package names above.' }
        Write-Host '[4/5] Building the interface...'
        $nodeReady = $false
        if ((Get-Command node -ErrorAction SilentlyContinue) -and (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
            $nodeVersion = [version]((& node -p 'process.versions.node').Trim())
            $nodeReady = $nodeVersion -ge [version]'22.12.0'
        }
        if (-not $nodeReady) {
            $nodeDir = Join-Path $toolsRoot 'node-v22.23.2-win-x64'
            if (-not (Test-Path -LiteralPath (Join-Path $nodeDir 'node.exe'))) {
                $archive = Get-VerifiedArchive 'https://nodejs.org/dist/v22.23.2/node-v22.23.2-win-x64.zip' 'node-v22.23.2.zip' '1177b4137ba5adaa56354ae40f1080c7450e8ae09cecb47da459d1c52ac99f97'
                Expand-Archive -LiteralPath $archive -DestinationPath $toolsRoot -Force
            }
            $env:PATH = $nodeDir + ';' + $env:PATH
        }
        & npm.cmd ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the interface packages. Run Setup.bat again to retry.' }
        & $pythonPath -u (Join-Path $PSScriptRoot 'download_examples.py')
        if ($LASTEXITCODE -ne 0) { throw 'Some example images could not be downloaded. Run Setup.bat again to retry.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Could not build the interface. See the error above.' }
        Write-Host '[5/5] Checking the graphics card and creating the shortcut...'
        & $pythonPath (Join-Path $PSScriptRoot 'check_runtime.py')
        if ($LASTEXITCODE -ne 0) { throw 'The graphics check failed. See the message above before generating images.' }
        & (Join-Path $PSScriptRoot 'create_shortcut.ps1')
        @{ fingerprint=(Get-QwenSetupFingerprint); completed_at=(Get-Date -Format o) } | ConvertTo-Json | Set-Content -LiteralPath $setupStamp -Encoding UTF8
    } finally { Pop-Location }
    Write-Host 'Setup complete. Choose and download models in Settings.'
} finally { if ($setupLock) { $setupLock.Dispose() } }
