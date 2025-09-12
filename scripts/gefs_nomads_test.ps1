# GEFS-Aerosols Data Collection via NOMADS gribfilter
# ===================================================
# Quick test script for downloading PM2.5 data from recent GEFS-chem runs

# Configuration
$outputDir = "C:\aqf311\data\gefs_chem_raw"
$baseUrl = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_chem_0p25.pl"

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Test cities (lat/lon bounding boxes)
$cities = @{
    "Delhi" = @{
        "leftlon" = 77.0; "rightlon" = 78.0
        "toplat" = 29.0; "bottomlat" = 28.0
    }
    "Beijing" = @{
        "leftlon" = 116.0; "rightlon" = 117.0
        "toplat" = 40.5; "bottomlat" = 39.5
    }
    "Los_Angeles" = @{
        "leftlon" = -119.0; "rightlon" = -118.0
        "toplat" = 34.5; "bottomlat" = 33.5
    }
}

# Recent dates and cycles to test
$testDate = "20250912"  # Today
$cycles = @("00", "12")
$fhours = @("000", "003", "006", "012", "024", "048")  # Sample forecast hours

Write-Host "Testing NOMADS GEFS-chem gribfilter access..." -ForegroundColor Green

foreach ($cycle in $cycles) {
    Write-Host "Processing cycle: ${testDate}/${cycle}Z" -ForegroundColor Cyan

    foreach ($fhour in $fhours) {
        $file = "gefs.chem.t${cycle}z.a2d_0p25.f${fhour}.grib2"
        $dir = "/gefs.${testDate}/${cycle}/chem/pgrb2ap25"

        Write-Host "  Forecast hour: f${fhour}" -ForegroundColor Yellow

        foreach ($cityName in $cities.Keys) {
            $city = $cities[$cityName]

            # Build query string for PM2.5 and PM10
            $qs = "file=${file}&dir=${dir}&var_PMTF=on&var_PMTC=on&lev_surface=on&subregion=" +
                  "&leftlon=$($city.leftlon)&rightlon=$($city.rightlon)" +
                  "&toplat=$($city.toplat)&bottomlat=$($city.bottomlat)"

            $url = "${baseUrl}?${qs}"
            $outFile = "${outputDir}\${testDate}_${cycle}_f${fhour}_${cityName}_PM.grib2"

            try {
                Write-Host "    Downloading ${cityName}..." -NoNewline

                # Download with curl
                $result = curl -s -f $url -o $outFile

                if (Test-Path $outFile) {
                    $size = (Get-Item $outFile).Length
                    if ($size -gt 1000) {  # Check if file has content
                        Write-Host " Success (${size} bytes)" -ForegroundColor Green
                    } else {
                        Write-Host " Empty file" -ForegroundColor Yellow
                        Remove-Item $outFile -ErrorAction SilentlyContinue
                    }
                } else {
                    Write-Host " Failed" -ForegroundColor Red
                }

            } catch {
                Write-Host " Error: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

# Summary
Write-Host "`nDownload Summary:" -ForegroundColor Green
$files = Get-ChildItem $outputDir -Filter "*.grib2"
Write-Host "Total files downloaded: $($files.Count)"
$totalSize = ($files | Measure-Object -Property Length -Sum).Sum
Write-Host "Total size: $([math]::Round($totalSize/1MB, 2)) MB"

if ($files.Count -gt 0) {
    Write-Host "`nSample files:" -ForegroundColor Cyan
    $files | Select-Object -First 5 | ForEach-Object {
        $sizeMB = [math]::Round($_.Length/1MB, 2)
        Write-Host "  $($_.Name) - ${sizeMB} MB"
    }

    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Install GRIB reading tools: pip install cfgrib eccodes"
    Write-Host "2. Run Python script: python scripts/backfill_gefs_pm25.py"
    Write-Host "3. Verify PM2.5 data extraction and city coordinates"
} else {
    Write-Host "`nNo files downloaded. Check:" -ForegroundColor Red
    Write-Host "1. Internet connectivity to nomads.ncep.noaa.gov"
    Write-Host "2. Date availability (NOMADS keeps ~7-14 days)"
    Write-Host "3. GEFS-chem run schedule (00Z, 06Z, 12Z, 18Z)"
}

Write-Host "`nGEFS-chem test complete!" -ForegroundColor Green
