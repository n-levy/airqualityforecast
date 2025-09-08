# run_all_providers.ps1
# Runs all Stage 3 ETLs sequentially. Stops on first failure.

$ErrorActionPreference = "Stop"

Write-Host "=== Stage 3: Running CAMS ==="
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\etl_cams.ps1"

Write-Host "=== Stage 3: Running Aurora ==="
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\etl_aurora.ps1"

Write-Host "=== Stage 3: Running NOAA GEFS-Aerosol ==="
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\etl_noaa_gefs_aerosol.ps1"

Write-Host "All Stage 3 providers completed successfully."
