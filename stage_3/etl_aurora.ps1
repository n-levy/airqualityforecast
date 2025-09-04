<#
  ETL Aurora (Stage 3)
  Usage:
    cd C:\aqf311\Git_repo\stage_3
    powershell -ExecutionPolicy Bypass -File .\etl_aurora.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$stage2 = Join-Path (Join-Path $root "..") "stage_2\.venv_stage2\Scripts\python.exe"
$local  = Join-Path $root ".venv_stage3\Scripts\python.exe"
$py = if (Test-Path $stage2) { $stage2 } elseif (Test-Path $local) { $local } else { $null }
if (-not $py) { throw "Python venv not found. Run setup_stage3.ps1 first." }

$cfg = Join-Path $root "config\providers.yaml"
$script = Join-Path $root "scripts\providers\etl_aurora.py"
Write-Host "Command:" $py $script "--config" $cfg
& $py $script --config $cfg
if ($LASTEXITCODE -ne 0) { throw "ETL Aurora (Stage 3) failed." }
Write-Host "ETL Aurora (Stage 3) finished."
