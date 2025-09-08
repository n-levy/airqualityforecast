<# ============================
 run_all.ps1  — Stage 3 pipeline runner
 - No popups, no prompts
 - Runs: OpenAQ (S3-only) → CAMS → Join
 - Writes timestamped logs to .\logs\
 - Clear progress + exit codes

 USAGE (from stage_3 folder):
   powershell -ExecutionPolicy Bypass -File .\run_all.ps1

 Optional switches:
   -SkipOpenAQ    # skip OpenAQ step
   -SkipCAMS      # skip CAMS step
   -SkipJoin      # skip join step
   -Verbose       # show extra progress in console

 Requirements:
   - Python venv at .\.venv_stage3 (as set up in your repo)
   - OPENAQ_API_KEY in env (for location discovery)
================================ #>

[CmdletBinding()]
param(
  [switch]$SkipOpenAQ,
  [switch]$SkipCAMS,
  [switch]$SkipJoin
)

$ErrorActionPreference = 'Stop'

function Write-Header($text) {
  Write-Host ""
  Write-Host ("=" * 80)
  Write-Host ("=  {0}" -f $text) -ForegroundColor Cyan
  Write-Host ("=" * 80)
}

# --- Resolve important paths relative to this script ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$repoRoot   = Resolve-Path "$ScriptDir"
$logsDir    = Join-Path $repoRoot "logs"
$dataDir    = Join-Path $repoRoot "data"
$procDir    = Join-Path $dataDir "providers_processed"
$venvDir    = Join-Path $repoRoot ".venv_stage3"
$activatePs = Join-Path $venvDir "Scripts\Activate.ps1"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $procDir | Out-Null

# --- Prepare logging ---
$stamp      = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath    = Join-Path $logsDir "run_all_$stamp.log"
"Run started: $(Get-Date -Format s)" | Tee-Object -FilePath $logPath -Append | Out-Host

# --- Sanity checks ---
if (-not (Test-Path $activatePs)) {
  throw "Python venv not found at $activatePs . Please create/restore .venv_stage3."
}

if (-not $env:OPENAQ_API_KEY) {
  Write-Warning "OPENAQ_API_KEY not set in this session. OpenAQ location discovery will fail."
  Write-Warning "Set it in this session OR persist it:  [Environment]::SetEnvironmentVariable('OPENAQ_API_KEY','<key>','User')"
}

# --- Activate venv (for consistency of python libs used by the PS1 steps) ---
Write-Header "Activating Python venv"
. $activatePs
"Venv activated: $venvDir" | Tee-Object -FilePath $logPath -Append | Out-Host

# --- Helper to run a step with timing & logging ---
function Invoke-Step {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][scriptblock]$ScriptBlock
  )
  Write-Header "Step: $Name"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    & $ScriptBlock *>&1 | Tee-Object -FilePath $logPath -Append
    $sw.Stop()
    "✔ $Name completed in {0:n1} sec" -f $sw.Elapsed.TotalSeconds | Tee-Object -FilePath $logPath -Append | Out-Host
    return 0
  }
  catch {
    $sw.Stop()
    "✖ $Name failed after {0:n1} sec" -f $sw.Elapsed.TotalSeconds | Tee-Object -FilePath $logPath -Append | Out-Host
    $_ | Tee-Object -FilePath $logPath -Append
    return 1
  }
}

$exitOpenAQ = 0
$exitCAMS   = 0
$exitJoin   = 0

# --- 1) OpenAQ (S3-only) ---
if (-not $SkipOpenAQ) {
  $exitOpenAQ = Invoke-Step -Name "OpenAQ (S3-only)" -ScriptBlock {
    powershell -ExecutionPolicy Bypass -File .\etl_openaq_history.ps1
    if ($LASTEXITCODE -ne 0) { throw "OpenAQ ETL reported non-zero exit code: $LASTEXITCODE" }
    # Small post-check
    $out = Join-Path $procDir "openaq_hourly.parquet"
    if (-not (Test-Path $out)) { throw "Expected output not found: $out" }
    Write-Host "Output: $out"
  }
}
else {
  Write-Header "Skipping: OpenAQ (S3-only)"
}

# --- 2) CAMS ---
if (-not $SkipCAMS) {
  $exitCAMS = Invoke-Step -Name "CAMS forecast (historical live pull)" -ScriptBlock {
    powershell -ExecutionPolicy Bypass -File .\etl_cams_live.ps1
    if ($LASTEXITCODE -ne 0) { throw "CAMS ETL reported non-zero exit code: $LASTEXITCODE" }
    # Optional post-check(s) — adjust if your CAMS script writes specific files
    # e.g., Check for parquet output location if defined
  }
}
else {
  Write-Header "Skipping: CAMS"
}

# --- 3) Join ---
if (-not $SkipJoin) {
  $exitJoin = Invoke-Step -Name "Build comparable table (join OpenAQ + CAMS)" -ScriptBlock {
    powershell -ExecutionPolicy Bypass -File .\run_build_join_dataset.ps1
    if ($LASTEXITCODE -ne 0) { throw "Join step reported non-zero exit code: $LASTEXITCODE" }
    # Optional: verify output file(s) exist
  }
}
else {
  Write-Header "Skipping: Join"
}

# --- Summary ---
Write-Header "Summary"
"OpenAQ exit: $exitOpenAQ" | Tee-Object -FilePath $logPath -Append | Out-Host
"CAMS   exit: $exitCAMS"   | Tee-Object -FilePath $logPath -Append | Out-Host
"Join   exit: $exitJoin"   | Tee-Object -FilePath $logPath -Append | Out-Host

$overall = $exitOpenAQ + $exitCAMS + $exitJoin
if ($overall -eq 0) {
  "ALL STEPS SUCCEEDED" | Tee-Object -FilePath $logPath -Append | Out-Host
  exit 0
} else {
  "ONE OR MORE STEPS FAILED — see log: $logPath" | Tee-Object -FilePath $logPath -Append | Out-Host
  exit 1
}
