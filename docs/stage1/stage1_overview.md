# Stage 1 Overview (Air Quality Forecast)
_Last updated: 2025-08-31_

## For Humans (manager/engineer-friendly)

**Goal.** A working end-to-end pipeline that ingests **real PM2.5 observations** via **OpenAQ v3** (API key) with a **keyless Open-Meteo fallback**, builds features, trains a baseline model, generates hourly forecasts, verifies basic metrics, and publishes CSV/JSON for hand-off to BI or dashboards.

### What Stage 1 does (high level)

1) **ETL → Observations**
   - Provider priority: **OpenAQ v3** → fallback **Open-Meteo** if OpenAQ is unavailable (401/429/no data).
   - Writes *partitioned Parquet* by date:  
     `C:\aqf311\data\curated\obs\<city>\pm25\date=YYYY-MM-DD\data.parquet`

2) **Feature engineering**
   - Lags: 1, 2, 3, 6, 12, 24, 48 hours  
   - Rolling means: 6h, 24h  
   - Calendar: hour of day, day of week  
   - Output: `C:\aqf311\data\features\<city>\pm25\features.parquet`

3) **Training**
   - `RidgeCV` with alphas {0.1, 1.0, 3.0, 10.0}, 5-fold CV.  
   - Output: `C:\aqf311\models\ridge\<city>\pm25\model.joblib`

4) **Inference**
   - Recursive next-N-hours forecast (default 24).  
   - Output: `C:\aqf311\data\forecasts\ours\<city>\pm25\forecast.parquet`

5) **Verification**
   - Joins forecast with overlapping observations; prints MAE & Bias (if overlap exists).

6) **Publish**
   - Recent observations and forecast exported for BI:  
     `C:\aqf311\data\exports\<city>\obs_pm25_recent.{csv,json}`  
     `C:\aqf311\data\exports\<city>\forecast_pm25.{csv,json}`

### Entrypoint & how to run (one-shot)
> Assumes Python **3.11** venv at `C:\aqf311\.venv` and `stage1\requirements.txt` installed.

```powershell
# Allow script just for this window:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# From repo root:
powershell -NoProfile -ExecutionPolicy Bypass -File ".\stage1\scripts\run_stage1.ps1" `
  -City berlin -Provider auto -ObsHours 168 -ForecastHours 24
```

- **Cities:** `berlin`, `hamburg`, `munich`  
- **Provider:** `openaq` | `openmeteo` | `auto` (OpenAQ first, then fallback to Open-Meteo)

### Environment & config

Create `stage1\config\env\.env` from `.env.example`:

```
DATA_ROOT=C:\aqf311\data
MODELS_ROOT=C:\aqf311\models
CACHE_ROOT=C:\aqf311\.cache
OPENAQ_API_KEY=<optional, only for OpenAQ>
```

The runner loads `.env` into the **process** environment and sets `PYTHONPATH` to `<repo>\stage1`. All timestamps are **UTC** and **hourly**.

### Provider behavior (resilience)

- **OpenAQ v3** client:
  - City bbox discovery + latest-active sensors.
  - Tries endpoints in order:
    1. `/sensors/{id}/hours`
    2. `/sensors/{id}/measurements`
    3. `/measurements?bbox=...` (city-level fallback)
  - 429 `Retry-After` honored with backoff; 401 raises a clear error.

- **Open-Meteo** fallback:
  - Keyless, deterministic; hourly `pm2_5` for city (UTC).

### Data contracts (schemas)

- **Curated observations** (`data.parquet`):  
  `city` (str, lower), `valid_time` (datetime64[ns, UTC], hourly), `value` (float, µg/m³), `unit` (str)

- **Features** (`features.parquet`):  
  `valid_time` (UTC), `y`, `lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48`, `roll_6`, `roll_24`, `hour`, `dow`

- **Forecast** (`forecast.parquet`):  
  `valid_time` (UTC), `yhat`

### Typical outputs to verify

- Parquet partitions under `...\curated\obs\<city>\pm25\date=*`  
- Features/model/forecast written under `C:\aqf311\...`  
- CSV/JSON exports for BI in `C:\aqf311\data\exports\<city>\`

---

## For ChatGPT (system-facing spec)

**Runner:** `stage1\scripts\run_stage1.ps1`  
**Order:** `smoke_test.py` → `apps\etl\obs_pm25.py` → `apps\features\build.py` → `apps\train\train_ridge.py` → `apps\infer\infer_hourly.py` → `apps\verify\verify_hourly.py` → `apps\publish\export_static.py`

**Key modules**

- `apps\etl\providers.py`  
  - `fetch_pm25_city(city: str, hours: int, prefer: str='auto') -> List[dict]`  
  - Classes: `OpenAQv3`, `OpenMeteoAQ`; emit rows: `datetime` (ISO, UTC), `value`, `unit` (+ optional `sensor_id`).  
  - Known cities (lat,lon): berlin(52.52,13.405), hamburg(53.55,9.993), munich(48.137,11.575).

- `apps\etl\obs_pm25.py` → normalize to (`city`,`valid_time` UTC hour, `value`,`unit`) and write date partitions.

- `apps\features\build.py` → build lags/rollups/calendar; keep `valid_time` & `y`.

- `apps\train\train_ridge.py` → `RidgeCV` baseline saved to `models\ridge\...`.

- `apps\infer\infer_hourly.py` → recursive N-step prediction; write `forecast.parquet`.

- `apps\verify\verify_hourly.py` → print MAE/Bias if obs overlap exists.

- `apps\publish\export_static.py` → write recent obs + forecast to CSV/JSON.

**Environment invariants**

- Python 3.11; Parquet via **pyarrow** only.  
- Heavy artifacts under `C:\aqf311\...` (avoid cloud-sync churn).  
- Timestamps **UTC**; hourly frequency; recursive forecasting.  
- Each script runnable with `PYTHONPATH=<repo>\stage1`.

**Extensibility (Stage 2 hooks)**

- Add cities → extend `CITY_LL` dicts in `providers.py`.  
- Add features → extend `build.py` (preserve contracts), or add modules.  
- Add models → new `apps\train\train_*.py` + runner update.  
- Backtests/metrics → `apps\verify\backtest_*.py` + runner call.
