<#
  setup_stage3.ps1 - Stage 3 setup (Windows PowerShell)
  - Prefer Stage 2 venv (..\\stage_2\\.venv_stage2), else use .\\venv_stage3
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Try Stage 2 venv python
$py = Join-Path (Join-Path $root "..") "stage_2\.venv_stage2\Scripts\python.exe"
if (-not (Test-Path $py)) {
  # fallback: local venv
  $localVenvPy = Join-Path $root ".venv_stage3\Scripts\python.exe"
  if (-not (Test-Path $localVenvPy)) {
    Write-Host "Creating local venv: .venv_stage3"
    py -3 -m venv .venv_stage3
  }
  $py = $localVenvPy
}

Write-Host "Using Python:" $py
$pipCmd = "$py -m pip install --upgrade pip"
cmd /c $pipCmd

$req = Join-Path $root "requirements_stage3.txt"
$installCmd = "$py -m pip install -r `"$req`""
cmd /c $installCmd
Write-Host "Stage 3 setup complete."
