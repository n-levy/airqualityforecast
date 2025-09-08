# run_build_join_dataset.ps1
$ErrorActionPreference = "Stop"

$Stage3Root = $PSScriptRoot
$PythonExe  = Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe"
$ScriptPath = Join-Path $Stage3Root "scripts\align_and_join.py"
$ConfigPath = Join-Path $Stage3Root "config\providers.yaml"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $Stage3Root + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }))

Write-Host "Command: $PythonExe $ScriptPath --config $ConfigPath"
Push-Location $Stage3Root
& $PythonExe $ScriptPath "--config" $ConfigPath
$e = $LASTEXITCODE
Pop-Location

if ($e -ne 0) { throw "Build & Join dataset failed." }
Write-Host "Build & Join dataset completed."
