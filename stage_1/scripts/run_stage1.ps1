param(
  [ValidateSet("berlin","hamburg","munich")] [string]$City = "berlin",
  [ValidateSet("auto","openaq","openmeteo")] [string]$Provider = "auto",
  [int]$ObsHours = 168,
  [int]$ForecastHours = 24
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

# Your Python venv path (created earlier)
$Py = "C:\aqf311\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) { throw "Python not found at $Py. Adjust the path." }

# Make the stage1 package importable and unbuffer logs
$Stage1Root = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = $Stage1Root
$env:PYTHONUNBUFFERED = "1"

# Load .env (simple key=value) into this process
$EnvFile = Join-Path $Stage1Root "config\env\.env"
if (Test-Path $EnvFile) {
  Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $kv = $_ -split '=', 2
    if ($kv.Count -eq 2) { [Environment]::SetEnvironmentVariable($kv[0].Trim(), $kv[1].Trim(), "Process") }
  }
}

# Default roots if not set in .env
if (-not $env:DATA_ROOT)   { $env:DATA_ROOT   = "C:\aqf311\data" }
if (-not $env:MODELS_ROOT) { $env:MODELS_ROOT = "C:\aqf311\models" }
if (-not $env:CACHE_ROOT)  { $env:CACHE_ROOT  = "C:\aqf311\.cache" }

Write-Host ("Provider: {0} | OpenAQ key present: {1}" -f $Provider, [bool]$env:OPENAQ_API_KEY)

function Step([string]$name, [string[]]$ArgsList) {
  Write-Host "==> $name"
  if (-not $ArgsList -or -not (Test-Path -LiteralPath $ArgsList[0])) {
    throw "Step '$name' missing/invalid script path: $($ArgsList[0])"
  }
  Write-Host ("CMD: {0} {1}" -f $Py, ($ArgsList -join ' '))
  & $Py $ArgsList
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (exit $LASTEXITCODE)" }
}

Step "Smoke test" @("$PSScriptRoot\smoke_test.py")
Step "ETL"        @(".\stage1\apps\etl\obs_pm25.py","--city",$City,"--hours",$ObsHours,"--provider",$Provider)
Step "Features"   @(".\stage1\apps\features\build.py","--city",$City)
Step "Train"      @(".\stage1\apps\train\train_ridge.py","--city",$City)
Step "Inference"  @(".\stage1\apps\infer\infer_hourly.py","--city",$City,"--hours",$ForecastHours)

Write-Host "==> Verify"
& $Py ".\stage1\apps\verify\verify_hourly.py" --city $City
if ($LASTEXITCODE -ne 0) { Write-Warning "No overlap yet; metrics appear after some forecast hours pass." }

Step "Publish"    @(".\stage1\apps\publish\export_static.py","--city",$City)
Write-Host ("[stage1] Done for city={0} (provider={1})" -f $City, $Provider)
