Param(
  [string]$ConfigPath = "config/providers.yaml"
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
  if ($env:VIRTUAL_ENV) {
    $p = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path $p) { return $p }
  }
  $pythonOnPath = (Get-Command python -ErrorAction SilentlyContinue)
  if (Test-Command python -ErrorAction SilentlyContinue) {}
  if ($pythonOnPath) { return "python" }
  $fallback = "C:\aqf311\.venv\Scripts\python.exe"
  if (Test-Path $fallback) { return $fallback }
  throw "Could not locate python.exe."
}

$PythonExe = Resolve-Python

Write-Host "[CAMS history] Using Python at: $PythonExe"
& $PythonExe "scripts/providers/etl_cams_history.py" --config $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "[CAMS history] Done."
