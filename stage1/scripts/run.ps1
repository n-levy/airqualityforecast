param(
  [Parameter(Position=0, Mandatory=$false)]
  [string]$Target = "help"
)

$PY = "C:\aqf311\.venv\Scripts\python.exe"

function Header([string]$name) { Write-Host "== $name ==" -ForegroundColor Cyan }

switch ($Target.ToLower()) {
  "help" {
@"
Commands:
  etl-berlin  — Curate Berlin PM2.5 for a date range (uses fake data by default)
  validate    — Validate city YAMLs against schema
  clean       — Remove __pycache__ and .pytest_cache
Examples:
  .\scripts\run.ps1 etl-berlin 2025-07-01 2025-07-07
"@ | Write-Host
    exit 0
  }
  "etl-berlin" {
    param([string]$Since, [string]$Until)
    if (-not $Since -or -not $Until) {
      Write-Host "Usage: .\scripts\run.ps1 etl-berlin <YYYY-MM-DD> <YYYY-MM-DD>" -ForegroundColor Yellow
      exit 1
    }
    $env:OBS_FAKE = "1"
    & $PY apps\etl\obs_pm25.py --city berlin --since $Since --until $Until --verbose
    exit $LASTEXITCODE
  }
  "validate" {
    & $PY apps\tools\validate_cities.py
    exit $LASTEXITCODE
  }
  "clean" {
    Header "clean"
    if (Test-Path ".\.pytest_cache") { Remove-Item ".\.pytest_cache" -Recurse -Force }
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "clean: done."
    exit 0
  }
  default {
    Write-Host "Unknown target '$Target'. Try: .\scripts\run.ps1 help" -ForegroundColor Yellow
    exit 1
  }
}
