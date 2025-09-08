Param()

$ErrorActionPreference = "Stop"

# Always operate from the script's own folder, no matter where PowerShell was launched
Set-Location -LiteralPath $PSScriptRoot

Write-Host "=== Stage 2 setup: creating virtual environment (.venv_stage2) ==="

$venvPath = Join-Path $PSScriptRoot ".venv_stage2"
$reqFile  = Join-Path $PSScriptRoot "requirements_stage2.txt"

if (-not (Test-Path $reqFile)) {
    Write-Error "requirements_stage2.txt not found at $reqFile"
    exit 1
}

# Pick a Python to create the venv:
# Prefer 'python' on PATH; otherwise try 'py -3.11'; otherwise fail clearly.
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonCmd) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        $pythonCmd = "py -3.11"
    } else {
        Write-Error "Python not found. Install Python 3.11+ or ensure python/py is on PATH."
        exit 1
    }
} else {
    $pythonCmd = "python"
}

# Create venv if needed
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating venv at $venvPath"
    & $pythonCmd -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

# Use the venv's python explicitly from here on
$venvPy = Join-Path $venvPath "Scripts\python.exe"

# Upgrade pip and install requirements (using absolute paths)
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r $reqFile

Write-Host "=== Setup complete ==="
