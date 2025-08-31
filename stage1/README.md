# Stage 1 — PM2.5 Observations (Curated)

This folder contains a **working implementation** of Stage 1 for your Air Quality Forecast project.

It does one thing well: **create a curated, city‑level hourly PM2.5 time series** (UTC, RFC 3339) and write it as **partitioned Parquet** under your `DATA_ROOT`.

The code is designed for **Windows PowerShell** execution and assumes you will edit Python in **Visual Studio Code**.

---

## What you can run (quick start)

1) **Open PowerShell** and activate your project virtualenv  
```powershell
# Example — adapt to your path
C:\aqf311\.venv\Scripts\activate
```

2) **Set where data will be written** (can be anywhere — defaults are OK)  
Create a `.env` file next to this README or set env vars in the shell:

```
DATA_ROOT=%USERPROFILE%\stage1_data
LOGS_ROOT=%USERPROFILE%\stage1_logs
```

3) **Install Python deps**
```powershell
pip install -r requirements.txt
```

4) **Generate curated Berlin observations for a week (FAKE data so no internet required)**
```powershell
$env:OBS_FAKE = "1"
C:\aqf311\.venv\Scripts\python.exe apps\etl\obs_pm25.py --city berlin --since 2025-07-01 --until 2025-07-07
```

5) **(Optional) Validate city configs**
```powershell
C:\aqf311\.venv\Scripts\python.exe apps\tools\validate_cities.py
```

You should see files under:

```
%DATA_ROOT%\curated\obs\berlin\pm25\date=2025-07-01\data.parquet
...
```

---

## Design (very short)

- **Input**: hourly station PM2.5 (faked by default; can be extended to EPA/UBA/OpenAQ).  
- **Transform**: normalize to **UTC** `valid_time`, aggregate per city (= **mean across stations**, per hour).  
- **Output**: Parquet partitioned by `date=YYYY-MM-DD` at `%DATA_ROOT%/curated/obs/<city>/pm25/…`.  
- **Schema**: `config/schemas/curated/observations_pm25.json`.

---

## Windows PowerShell run cheatsheet

```powershell
# One day (fake)
$env:OBS_FAKE = "1"
C:\aqf311\.venv\Scripts\python.exe apps\etl\obs_pm25.py --city berlin --since 2025-07-01 --until 2025-07-01

# All three cities
$env:OBS_FAKE = "1"
foreach ($c in @("berlin","hamburg","munich")) {
  C:\aqf311\.venv\Scripts\python.exe apps\etl\obs_pm25.py --city $c --since 2025-07-01 --until 2025-07-07
}
```

> Tip: Once you are ready to fetch real measurements, set `$env:OBS_FAKE = "0"` and implement the `fetch_uba_pm25()` function stub (kept minimal on purpose).

---

## Repository layout (Stage 1)

- `apps/etl/obs_pm25.py` — main ETL app (CLI).  
- `apps/tools/validate_cities.py` — checks YAML city configs against schema.  
- `config/cities/*.yml` — city metadata (stations, tz).  
- `config/schemas/…` — JSON Schemas for raw and curated rows.  
- `requirements.txt` — Python deps (tiny).  
- `tests/` — two tests that check the curated dataset shape and timezone.  

---

## Why “curated” and not “raw” first?

We keep Stage 1 focused: a clean time series any model can consume. You can always persist raw station‑level pulls separately (`DATA_ROOT/raw/...`) if you add a data source.

---

## Troubleshooting

- **`ModuleNotFoundError: pyarrow`** → run `pip install -r requirements.txt`.  
- **No files under `DATA_ROOT`** → check that `%DATA_ROOT%` exists (create it), and that `$env:OBS_FAKE` is set to `1` if you’re offline.  
- **Timezone or schema validations** → use `apps/tools/validate_cities.py` and inspect one Parquet file in VS Code:  
  ```powershell
  C:\aqf311\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_parquet(r'%USERPROFILE%\stage1_data\curated\obs\berlin\pm25\date=2025-07-01\data.parquet').head())"
  ```

