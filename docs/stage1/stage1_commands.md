STAGE1_COMMANDS.md

Last updated: 2025-09-01

Setup / Environment
# Create Python 3.11 venv and install
py -3.11 -m venv C:\aqf311\.venv
C:\aqf311\.venv\Scripts\python.exe -m pip install --upgrade pip
C:\aqf311\.venv\Scripts\python.exe -m pip install -r .\stage1\requirements.txt

# Create .env from template and fill paths/API key (optional)
Copy-Item .\stage1\config\env\.env.example .\stage1\config\env\.env
notepad .\stage1\config\env\.env


.env example:

DATA_ROOT=C:\aqf311\data
MODELS_ROOT=C:\aqf311\models
CACHE_ROOT=C:\aqf311\.cache
OPENAQ_API_KEY=<your key or leave blank>

Run Stage 1 (one-shot runner)
# Allow just for this window (if policies block scripts)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

powershell -NoProfile -ExecutionPolicy Bypass -File ".\stage1\scripts\run_stage1.ps1" `
  -City berlin -Provider auto -ObsHours 168 -ForecastHours 24


-Provider openaq → force observations (requires key)

-Provider openmeteo → force modeled fallback

-Provider auto → OpenAQ first; fallback to Open-Meteo

Run step-by-step (for debugging)
$env:PYTHONPATH = (Resolve-Path ".\stage1").Path
$Py = "C:\aqf311\.venv\Scripts\python.exe"

& $Py .\stage1\apps\etl\obs_pm25.py --city berlin --hours 168 --provider auto
& $Py .\stage1\apps\features\build.py --city berlin
& $Py .\stage1\apps\train\train_ridge.py --city berlin
& $Py .\stage1\apps\infer\infer_hourly.py --city berlin --hours 24
& $Py .\stage1\apps\verify\verify_hourly.py --city berlin
& $Py .\stage1\apps\publish\export_static.py --city berlin

Quick checks (data landed where expected)
# Observation partitions
gci C:\aqf311\data\curated\obs\berlin\pm25 -Recurse -Filter data.parquet | measure

# Features / model / forecast
gci C:\aqf311\data\features\berlin\pm25\features.parquet
gci C:\aqf311\models\ridge\berlin\pm25\model.joblib
gci C:\aqf311\data\forecasts\ours\berlin\pm25\forecast.parquet

# Exports for BI
gci C:\aqf311\data\exports\berlin\

Peek at data (Python one-liners)
# Show last 5 forecast rows
C:\aqf311\.venv\Scripts\python.exe - << 'PY'
import pandas as pd, pathlib as P
p = P.Path(r"C:\aqf311\data\forecasts\ours\berlin\pm25\forecast.parquet")
df = pd.read_parquet(p).sort_values("valid_time").tail(5)
print(df.to_string(index=False))
PY

Common fixes
# Script policy (one window only)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Corporate proxy (before pip/http calls)
$env:HTTP_PROXY  = "http://user:pass@proxy:port"
$env:HTTPS_PROXY = "http://user:pass@proxy:port"

# Rebuild venv on correct interpreter
Remove-Item -Recurse -Force C:\aqf311\.venv -ErrorAction SilentlyContinue
py -3.11 -m venv C:\aqf311\.venv