# etl_noaa_gefs_aerosol.ps1
$ErrorActionPreference = "Stop"
$Stage3Root = $PSScriptRoot
$PyStage2   = Join-Path (Join-Path $Stage3Root "..") "stage_2\.venv_stage2\Scripts\python.exe"
if (Test-Path $PyStage2) { 
    $PythonExe = $PyStage2 
} elseif (Test-Path (Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe")) {
    $PythonExe = Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe"
} else { 
    $PythonExe = "python" 
}
$ScriptPath = Join-Path $Stage3Root "scripts\providers\etl_noaa_gefs_aerosol.py"
$ConfigPath = Join-Path $Stage3Root "config\providers.yaml"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = ($Stage3Root + "\scripts" + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })))
Write-Host "Command: $PythonExe $ScriptPath --config $ConfigPath"
Push-Location $Stage3Root
& $PythonExe $ScriptPath "--config" $ConfigPath
$exit = $LASTEXITCODE
Pop-Location
if ($exit -ne 0) { throw "ETL NOAA GEFS-Aerosol (Stage 3) failed." }
Write-Host "ETL NOAA GEFS-Aerosol (Stage 3) completed successfully."