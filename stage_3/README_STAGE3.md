
# Stage 3: External Forecast Benchmarks (CAMS, Aurora, NOAA GEFS-Aerosol)

Produces **comparable daily city-level** tables for {pm25, pm10, no2, o3}:
```
city,date,pm25,pm10,no2,o3
```
For fast tests, ETLs run in **sample mode** and read small CSVs in `data/samples/`.
Later switch to live mode by editing `config/providers.yaml` and adding real fetch code.

## One-liners (Windows PowerShell)

```powershell
# 1) Setup (installs Stage 3 deps into Stage 2 venv if present, else creates .venv_stage3)
cd C:\aqf311\Git_repo\stage_3
powershell -ExecutionPolicy Bypass -File .\setup_stage3.ps1

# Alternative (per provider setups – all call the same setup script)
powershell -ExecutionPolicy Bypass -File .\setup_cams.ps1
powershell -ExecutionPolicy Bypass -File .\setup_aurora.ps1
powershell -ExecutionPolicy Bypass -File .\setup_noaa_gefs_aerosol.ps1

# 2) Run each ETL
powershell -ExecutionPolicy Bypass -File .\etl_cams.ps1
powershell -ExecutionPolicy Bypass -File .\etl_aurora.ps1
powershell -ExecutionPolicy Bypass -File .\etl_noaa_gefs_aerosol.ps1

# 3) Run all three
powershell -ExecutionPolicy Bypass -File .\run_all_providers.ps1
```

## Outputs
- Raw CSVs: `data/providers_raw/raw_{provider}_*.csv`
- Processed parquet: `data/providers_processed/{provider}_forecast.parquet`

## Comparable to Stage 2 (OpenAQ)
- Same columns & types
- Same example cities: Berlin, Hamburg, München
- Daily granularity
- UTF-8 data, ASCII-only logs

## Live mode (later)
Switch `mode: live` in `config/providers.yaml` then add provider-specific fetch logic in `scripts/providers/etl_*.py`.

Outputs: data/providers_raw/raw_{provider}_*.csv, data/providers_processed/{provider}_forecast.parquet
