# Project Overview: Stages 1–3

## Stage 1 — Environment and Pipeline Skeleton
- Windows/PowerShell-first repo and venv scaffolding.
- Pipeline skeleton with scripts, config, data, logs.
- Core docs: PRD, NFRs, Roadmap, ADRs.

## Stage 2 — Observations ETL (OpenAQ v3) + Validation + Processing
- OpenAQ v3 fetch with 1-year lookback, capped sensors per city/parameter.
- Validate (warnings for negatives), process to parquet with summary stats.
- PowerShell orchestration with progress bar and ASCII logs.
- Output: data/processed/clean_air_quality.parquet

## Stage 3 — External Forecast Benchmarks (CAMS, Aurora, NOAA GEFS-Aerosol)
- Three provider ETLs producing schema identical to Stage 2 (city, date, pm25, pm10, no2, o3).
- Default to sample mode for fast runs; switchable to live mode later.
- PowerShell one-liners:
  - Setup (shared): .\setup_stage3.ps1
  - Per-provider setup shortcuts: .\setup_cams.ps1, .\setup_aurora.ps1, .\setup_noaa_gefs_aerosol.ps1
  - Run one: .\etl_cams.ps1 (and aurora/noaa variants)
  - Run all: .\run_all_providers.ps1
- Outputs:
  - data/providers_raw/raw_{provider}_*.csv
  - data/providers_processed/{provider}_forecast.parquet

Next:
- Implement live-mode fetch for each provider.
- Add evaluation scripts to score providers vs OpenAQ truth.
- Proceed to Stage 4 (feature engineering) and Stage 5 (modeling).
