# run_noaa.ps1
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\etl_noaa_gefs_aerosol.ps1"
