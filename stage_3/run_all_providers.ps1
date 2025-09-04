<#
  run_all_providers.ps1 - runs all three provider ETLs (sample mode)
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

.\etl_cams.ps1
.\etl_aurora.ps1
.\etl_noaa_gefs_aerosol.ps1
Write-Host "All three provider ETLs completed."
