# SAPIENT Windows Installer (PowerShell)
# Usage: irm https://raw.githubusercontent.com/SkidGod4444/sapient/main/install.ps1 | iex
#
# Installs the sapient CLI to %LOCALAPPDATA%\sapient\bin and adds it to PATH.

param(
    [string]$Version = "",
    [string]$InstallDir = "$env:LOCALAPPDATA\sapient\bin"
)

$ErrorActionPreference = "Stop"
$Repo = "SkidGod4444/sapient"
$BinaryName = "sapient.exe"

# ── UI helpers ────────────────────────────────────────────────────────────────
function Write-Banner {
    Write-Host ""
    Write-Host "  ___  _   ___ ___ ___ _  _ _____ " -ForegroundColor Cyan
    Write-Host " / __|| | | _ \ |_ _| __|\ | |_   _|" -ForegroundColor Cyan
    Write-Host " \__ \| |_|  _/ || || _|| .\`| | |  " -ForegroundColor Cyan
    Write-Host " |___/|___|_| |___|___|_||_|_| |_|  " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  LLM & SLM Inference Engine" -ForegroundColor White
    Write-Host ""
}

function Write-Step  { param($msg) Write-Host "  $([char]0x2192) $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  $([char]0x2713) $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  $([char]0x26A0) $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  $([char]0x2717) ERROR: $msg" -ForegroundColor Red; exit 1 }

# ── Detect architecture ───────────────────────────────────────────────────────
function Get-Platform {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64"   { return "x86_64-pc-windows-msvc" }
        "Arm64" { return "aarch64-pc-windows-msvc" }
        default { Write-Fail "Unsupported architecture: $arch" }
    }
}

# ── Get latest release ────────────────────────────────────────────────────────
function Get-LatestVersion {
    try {
        $release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
        return $release.tag_name
    } catch {
        Write-Fail "Could not fetch latest version. Check your internet connection."
    }
}

# ── Download ──────────────────────────────────────────────────────────────────
function Download-Binary {
    param($Platform, $Ver)

    $Filename = "sapient-$Platform.zip"
    $Url = "https://github.com/$Repo/releases/download/$Ver/$Filename"
    $TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
    New-Item -ItemType Directory -Path $TmpDir | Out-Null
    $TmpFile = Join-Path $TmpDir $Filename

    Write-Step "Downloading sapient $Ver for $Platform..."
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $TmpFile -UseBasicParsing
        $ProgressPreference = 'Continue'
    } catch {
        Write-Fail "Download failed from $Url`nError: $_"
    }

    # Verify archive checksum (matches release .sha256 files)
    $ChecksumUrl = "$Url.sha256"
    try {
        $Expected = (Invoke-RestMethod $ChecksumUrl -UseBasicParsing).Trim() -split '\s+' | Select-Object -First 1
        $Actual = (Get-FileHash $TmpFile -Algorithm SHA256).Hash.ToLower()
        if ($Expected -and ($Actual -ne $Expected.ToLower())) {
            Write-Fail "Checksum mismatch!`n  Expected: $Expected`n  Got:      $Actual"
        }
        Write-Ok "Checksum verified"
    } catch {
        Write-Warn "Could not verify checksum (non-fatal): $_"
    }

    Write-Step "Extracting..."
    Expand-Archive -Path $TmpFile -DestinationPath $TmpDir -Force

    $Binary = Get-ChildItem $TmpDir -Recurse -Filter $BinaryName | Select-Object -First 1
    if (-not $Binary) { Write-Fail "Binary not found after extraction" }

    return $Binary.FullName
}

# ── Install ───────────────────────────────────────────────────────────────────
function Install-Binary {
    param($BinaryPath, $Dir)

    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    }

    $Dest = Join-Path $Dir $BinaryName
    Copy-Item $BinaryPath $Dest -Force
    Write-Ok "Installed to $Dest"

    if (-not (Test-Path $Dest)) {
        Write-Fail "Install verification failed — binary missing at $Dest"
    }

    $UserPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if ([string]::IsNullOrEmpty($UserPath)) {
        $NewPath = $Dir
    } elseif ($UserPath -notlike "*$Dir*") {
        $NewPath = "$UserPath;$Dir"
    } else {
        $NewPath = $UserPath
    }

    if ($NewPath -ne $UserPath) {
        [System.Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
        Write-Ok "Added $Dir to your user PATH"
    }

    # Refresh PATH in the current session (helps `irm | iex` without restart)
    $MachinePath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
    $env:Path = "$NewPath;$MachinePath"

    if (-not (Get-Command $BinaryName -ErrorAction SilentlyContinue)) {
        Write-Warn "Restart your terminal if '$BinaryName' is not found"
    }
}

# ── Post-install ──────────────────────────────────────────────────────────────
function Write-PostInstall {
    param($Ver)
    Write-Host ""
    Write-Host "  SAPIENT $Ver installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Verify:" -ForegroundColor White
    Write-Host "    sapient --version" -ForegroundColor White
    Write-Host ""
    Write-Host "  Run your first model:" -ForegroundColor White
    Write-Host "    sapient chat microsoft/phi-2" -ForegroundColor White
    Write-Host ""
    Write-Host "  Other useful commands:" -ForegroundColor White
    Write-Host "    sapient pull TheBloke/Llama-2-7B-GGUF   " -NoNewline; Write-Host "# Download a model" -ForegroundColor DarkGray
    Write-Host "    sapient list                             " -NoNewline; Write-Host "# List cached models" -ForegroundColor DarkGray
    Write-Host "    sapient reset                            " -NoNewline; Write-Host "# Clear all cached models" -ForegroundColor DarkGray
    Write-Host "    sapient run microsoft/phi-2 --prompt ""Hello""" -NoNewline; Write-Host " # One-shot" -ForegroundColor DarkGray
    Write-Host "    sapient --help                           " -NoNewline; Write-Host "# Full help" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Docs: https://github.com/$Repo" -ForegroundColor Cyan
    Write-Host ""
}

# ── Main ──────────────────────────────────────────────────────────────────────
Write-Banner

$Platform = Get-Platform
Write-Step "Detected platform: $Platform"

$Ver = if ($Version) { $Version } else { Get-LatestVersion }
Write-Step "Version: $Ver"

$Binary = Download-Binary -Platform $Platform -Ver $Ver
Install-Binary -BinaryPath $Binary -Dir $InstallDir

Write-PostInstall -Ver $Ver

Remove-Item (Split-Path $Binary -Parent) -Recurse -Force -ErrorAction SilentlyContinue
