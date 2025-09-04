# Run Stage 2 on Windows (PowerShell) – Step by Step

> This guide assumes **no prior software engineering experience**.

## 0) Where to put this folder
- Place the entire `stage_2` folder **next to** your existing Stage 1 folder in the same repo root.
  Your repo root should look like:
  ```text
  repo_root/
    stage_1/
    stage_2/
    (other files...)
  ```

## 1) Open Windows PowerShell
- Press **Win** key, type **PowerShell**, press **Enter**.

## 2) Navigate to your repo root
Replace the path below with your actual path:
```powershell
cd "C:\path\to\your\repo_root"
```

## 3) Move into Stage 2
```powershell
cd .\stage_2
```

## 4) Set up Python environment (only the first time)
This creates a local virtual environment just for Stage 2 and installs dependencies:
```powershell
.\setup_stage2.ps1
```

What it does:
- Creates `.venv_stage2` under `stage_2`.
- Upgrades `pip`.
- Installs packages from `requirements_stage2.txt`.

## 5) Run the full Stage 2 pipeline
```powershell
.\run_stage2.ps1
```

What happens:
1. **Fetch** – loads the configured raw dataset(s) into `data/raw`.
2. **Validate** – checks that the raw data has the expected columns, types, and plausible ranges. Output → `data/interim/validated_air_quality.csv`.
3. **Process** – cleans the data (handles missing values/outliers, normalizes city names, parses dates, etc.). Output → `data/processed/clean_air_quality.parquet`.

## 6) Verify outputs
After it finishes without errors, you should see:
- `data/interim/validated_air_quality.csv`
- `data/processed/clean_air_quality.parquet`
- Logs under `logs/`

## 7) (Optional) Run tests
```powershell
.\.venv_stage2\Scripts\Activate.ps1
pytest -q
deactivate
```

## 8) Switch to your real data (later)
Edit `config/data_sources.yaml`:
- Change `provider: csv_local` to another provider (e.g., `openaq`) and fill the config section.
- Re-run `run_stage2.ps1`.

That’s it! If anything fails, look in the `logs\` folder for the most recent `.log` file.
