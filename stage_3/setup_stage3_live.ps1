# setup_stage3_live.ps1
$ErrorActionPreference = "Stop"

$Here = $PSScriptRoot
$Venv = Join-Path $Here ".venv_stage3"
$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

if (-not (Test-Path $Venv)) {
  Write-Host "Creating venv at $Venv ..."
  py -3 -m venv $Venv
}

& $Py -m pip install --upgrade pip
# Core libs
& $Pip install pandas pyarrow pyyaml requests tqdm python-dateutil
# CAMS / NetCDF processing
& $Pip install cdsapi xarray netCDF4

Write-Host "Stage 3 live environment ready."
