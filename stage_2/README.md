# Stage 2 – Data Ingestion & Validation

This stage implements an end-to-end data ingestion and validation pipeline for the Air Quality Forecast project.
It is designed to be run entirely from **Windows PowerShell**.

## What you get here

- **Config**: `config/data_sources.yaml` to define data inputs (local CSV by default).
- **Scripts** (Python):
  - `scripts/fetch_data.py`: fetches/loads raw data (local CSV by default).
  - `scripts/validate_data.py`: validates raw data (schema + basic plausibility checks).
  - `scripts/process_data.py`: turns validated raw data into a cleaned/processed dataset ready for modeling.
  - `scripts/common.py`: shared utilities (logging, config loading, paths).
- **Tests**: quick sanity checks with `pytest`.
- **PowerShell**:
  - `setup_stage2.ps1`: sets up a Python venv for Stage 2 and installs dependencies.
  - `run_stage2.ps1`: orchestrates the full Stage 2 pipeline (fetch → validate → process).
- **Data folders**:
  - `data/raw` – raw inputs (created automatically).
  - `data/interim` – intermediate validated outputs.
  - `data/processed` – final cleaned dataset.
- **Logs**: pipeline logs under `logs/`.

## Default input (offline-friendly)

We provide a small example CSV at `data/raw/sample_air_quality.csv` and configure the pipeline
to use it by default. You can later switch to real providers by editing `config/data_sources.yaml`.

## Outputs

- `data/interim/validated_air_quality.csv`
- `data/processed/clean_air_quality.parquet`

## How to run (short version)

1. Open **Windows PowerShell** and `cd` to your repo root (the folder containing Stage 1).
2. Unzip the provided `stage_2.zip` next to Stage 1, so you have a sibling folder named `stage_2`.
3. Run:
   ```powershell
   cd .\stage_2
   .\setup_stage2.ps1
   .\run_stage2.ps1
   ```

See `docs/RUN_STAGE2_WINDOWS.md` for beginner-friendly, detailed, step-by-step instructions.


## Provider: OpenAQ (Live)

- Configure in `config/data_sources.yaml` under `sources.openaq_de.options`.
- **API key:** set `OPENAQ_API_KEY` environment variable (recommended) or edit the yaml (less secure).
- Defaults fetch: last 7 days, Berlin/Munich/Hamburg, pollutants pm25/pm10/no2/o3.
- Output schema from fetch: `city,date,(pm25|pm10|no2|o3)` (daily mean).

### Set API key (PowerShell)
```powershell
$env:OPENAQ_API_KEY = "<your_key_here>"
# Persist for this session; to set permanently for your user:
[Environment]::SetEnvironmentVariable("OPENAQ_API_KEY", "<your_key_here>", "User")
```
