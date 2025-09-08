# tests\test_validate.ps1
$ErrorActionPreference = "Stop"

$TestsDir   = $PSScriptRoot
$Stage3Root = Split-Path $TestsDir -Parent

# Resolve Python exe
$PyStage2  = Join-Path (Join-Path $Stage3Root "..") "stage_2\.venv_stage2\Scripts\python.exe"
$PythonExe = $null
if (Test-Path $PyStage2) {
  $PythonExe = $PyStage2
} else {
  $PythonExe = Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe"
}

$files = @(
  "data\providers_processed\cams_forecast.parquet",
  "data\providers_processed\aurora_forecast.parquet",
  "data\providers_processed\noaa_gefs_aerosol_forecast.parquet"
)

foreach ($rel in $files) {
  $p = Join-Path $Stage3Root $rel
  if (-not (Test-Path $p)) {
    throw "Missing expected file: $p. Run the ETLs first."
  }
  Write-Host "Validating $rel ..."
  & $PythonExe (Join-Path $Stage3Root "scripts\validate_provider.py") $p
  if ($LASTEXITCODE -ne 0) { throw "Validation failed for $rel" }
}

# Optional: merge then validate the merged parquet (if merge script exists)
$mergeScript = Join-Path $Stage3Root "scripts\merge_providers.py"
if (Test-Path $mergeScript) {
  Write-Host "Merging providers..."
  & $PythonExe $mergeScript
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: merge script returned non-zero exit code." -ForegroundColor Red
    exit 1
  }

  $merged = Join-Path $Stage3Root "data\providers_processed\all_providers.parquet"
  if (Test-Path $merged) {
    Write-Host "Validating merged parquet..."
    & $PythonExe (Join-Path $Stage3Root "scripts\validate_provider.py") $merged
    if ($LASTEXITCODE -ne 0) {
      Write-Host "ERROR: validation failed for merged parquet." -ForegroundColor Red
      exit 1
    }
  } else {
    Write-Host "ERROR: Merged parquet not found after merge script." -ForegroundColor Red
    exit 1
  }
}
Write-Host "All validations passed."

