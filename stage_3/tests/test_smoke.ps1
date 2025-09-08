# tests\test_smoke.ps1
# Smoke test for Stage 3:
# 1) Run CAMS ETL
# 2) Assert processed parquet exists
# 3) Assert required columns present

$ErrorActionPreference = "Stop"

# Paths
$TestsDir     = $PSScriptRoot
$Stage3Root   = Split-Path $TestsDir -Parent
$ProcessedDir = Join-Path $Stage3Root "data\providers_processed"
$ExpectedFile = Join-Path $ProcessedDir "cams_forecast.parquet"

# Choose Python: prefer Stage 2 venv if available, else Stage 3 venv
$PyStage2 = Join-Path (Join-Path $Stage3Root "..") "stage_2\.venv_stage2\Scripts\python.exe"
if (Test-Path $PyStage2) { $PythonExe = $PyStage2 } else { $PythonExe = Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe" }

Write-Host "Running CAMS ETL in sample mode..."
powershell -ExecutionPolicy Bypass -File (Join-Path $Stage3Root "etl_cams.ps1")

if (-not (Test-Path $ExpectedFile)) {
    throw "Smoke test failed: $ExpectedFile was not created."
}

# Validate columns using pandas
$CheckCmd = @"
import sys, pandas as pd
df = pd.read_parquet(r'$ExpectedFile')
required = {'city','date','pm25','pm10','no2','o3'}
missing = required - set(df.columns)
if missing:
    print(f'Missing columns: {missing}')
    sys.exit(2)
print('Smoke test OK: file exists and required columns present.')
"@

$CheckCmd | & $PythonExe -
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed: missing columns." }

Write-Host "Smoke test passed."
