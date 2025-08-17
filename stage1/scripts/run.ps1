param(
  [Parameter(Position=0, Mandatory=$false)]
  [string]$Target = "help"
)

# Single source of truth for python
$PY = "C:\aqf311\.venv\Scripts\python.exe"

function Show-Header($name) {
  Write-Host "== $name ==" -ForegroundColor Cyan
}

switch ($Target.ToLower()) {
  "help" {
    @"
Available commands (PowerShell runner):
  help              Show this help
  check-python      Show interpreter being used
  bootstrap         Install deps; copy .env.example -> .env if missing
  smoke             Run local smoke test
  lint              Run linter checks (if configured)
  fmt               Auto-format (if configured)
  test              Run unit tests

  ingest-berlin     Ingest inputs for Berlin
  ingest-munich     Ingest inputs for Munich
  ingest-hamburg    Ingest inputs for Hamburg
  features-berlin   Build features for Berlin
  features-munich   Build features for Munich
  features-hamburg  Build features for Hamburg
  train-pm25        Train PM2.5 model
  infer-hourly      Run hourly inference (all cities)
  verify-hourly     Verify forecasts vs obs & benchmarks
  publish-static    Export static CSV/JSON for dashboard

  clean             Remove caches/build artifacts (keeps data/models)
"@ | Write-Host
  }
  "check-python" {
    Show-Header "check-python"
    & $PY -V
  }
  "bootstrap" {
    Show-Header "bootstrap"
    & $PY -m pip install --upgrade pip setuptools wheel
    & $PY -m pip install -r requirements.txt
    if (-not (Test-Path ".\.env")) {
      Copy-Item "config\env\.env.example" ".\.env" -Force
    }
    Write-Host "bootstrap: done"
  }
  "smoke" {
    Show-Header "smoke"
    & $PY scripts\smoke_test.py
  }
  "lint" {
    Show-Header "lint"
    if (Test-Path ".\ruff.toml") { & $PY -m ruff check . }
    if (Test-Path ".\pyproject.toml") { & $PY -m black --check . }
    Write-Host "lint: done"
  }
  "fmt" {
    Show-Header "fmt"
    if (Test-Path ".\pyproject.toml") { & $PY -m black . }
    Write-Host "fmt: done"
  }
  "test" {
    Show-Header "test"
    & $PY -m pytest -q
  }
  "ingest-berlin"    { & $PY apps\etl\uba_obs.py --city berlin; & $PY apps\etl\cams_stage1.py --city berlin; Write-Host "ingest-berlin: OK" }
  "ingest-munich"    { & $PY apps\etl\uba_obs.py --city munich; & $PY apps\etl\cams_stage1.py --city munich; Write-Host "ingest-munich: OK" }
  "ingest-hamburg"   { & $PY apps\etl\uba_obs.py --city hamburg; & $PY apps\etl\cams_stage1.py --city hamburg; Write-Host "ingest-hamburg: OK" }
  "features-berlin"  { & $PY apps\features\build.py --city berlin; Write-Host "features-berlin: OK" }
  "features-munich"  { & $PY apps\features\build.py --city munich; Write-Host "features-munich: OK" }
  "features-hamburg" { & $PY apps\features\build.py --city hamburg; Write-Host "features-hamburg: OK" }
  "train-pm25"       { & $PY apps\train\train_xgb.py --pollutant pm25; Write-Host "train-pm25: OK" }
  "infer-hourly"     { & $PY apps\infer\infer_hourly.py --cities berlin hamburg munich --pollutant pm25; Write-Host "infer-hourly: OK" }
  "verify-hourly"    { & $PY apps\verify\verify_hourly.py --cities berlin hamburg munich --pollutant pm25; Write-Host "verify-hourly: OK" }
  "publish-static"   { & $PY apps\publish\export_static.py --cities berlin hamburg munich --pollutant pm25 --out web\public; Write-Host "publish-static: OK" }
  "clean" {
    Show-Header "clean"
    if (Test-Path ".\.pytest_cache") { Remove-Item ".\.pytest_cache" -Recurse -Force }
    if (Test-Path ".\build") { Remove-Item ".\build" -Recurse -Force }
    if (Test-Path ".\dist") { Remove-Item ".\dist" -Recurse -Force }
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "clean: done"
  }
  default {
    Write-Host "Unknown target '$Target'. Try: .\scripts\run.ps1 help" -ForegroundColor Yellow
    exit 1
  }
}
