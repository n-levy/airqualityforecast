# Stage 1 – Multi-City PM₂.₅ (Berlin, Hamburg, Munich) with CAMS, Aurora, GEFS + Lags
_Designed for ~1 hour per task, minimal coding experience, gradual MLOps learning_

## A) Orientation & Project Fundamentals (6 hrs)
1) Read the repo map [✅ Completed]  
Goal: Understand folders you’ll touch first (apps/etl, features, train, infer, verify, publish; config/*; tests/*).  
Deliverable: A short written summary (10 bullets) of what each folder does.

2) Read the Product & Non‑Functional targets [✅ Completed]  
Goal: Know success criteria, dashboard outputs, and performance/SLA targets.  
Deliverable: A checklist of PRD success metrics and NFR targets.

3) Read modeling ADR [✅ Completed]  
Goal: Why a global combiner + optional per‑city bias; pollutants; AQI computation approach.  
Deliverable: One‑page notes with “what/why/how” of the combiner model.

4) Read evaluation ADR [✅ Completed]  
Goal: Understand κ (quadratic), category accuracy, threshold hits/false alarms, MAE per pollutant.  
Deliverable: A table explaining when to use each metric and what it teaches.

5) Read storage & scheduler ADRs [✅ Completed]  
Goal: Know Parquet via fsspec; local vs cloud paths; cron locally, cloud jobs later.  
Deliverable: Diagram showing data flow file:// → (later) s3:// and how jobs run.

6) Read GEFS‑Aerosol ADR [✅ Completed]  
Goal: Understand why we add GEFS, adapter location, schema/tests/scheduling.  
Deliverable: Bullet plan of GEFS tasks to integrate alongside CAMS/Aurora.

## B) Environment & Tooling (6 hrs)
7) Create a clean Python env (no admin rights) [✅ Completed]  
Goal: Set up `pyenv` or `venv` and activate it; confirm `python -m pip list`.  
Deliverable: Commands + version notes saved in `docs/SETUP.md`.

8) Install project dependencies from `pyproject.toml` [✅ Completed]  
Goal: Install minimum working set; record any issues.  
Deliverable: `pip install -e .` (or equivalent) succeeds; `python -c "import pandas"` works.

9) Make targets dry run  
Goal: Read the `Makefile` and run non‑destructive targets (e.g., `make help`).  
Deliverable: Screenshot/log + 3 notes on what each target will eventually do.

10) Configure `.env` from example [✅ Completed]  
Goal: Copy `config/env/.env.example` → `.env` and set DATA_ROOT/MODELS_ROOT.  
Deliverable: Working `.env` with local `file://` paths.

11) Create local data/model directories [✅ Completed]  
Goal: Create `data/` and `models/` folders, confirm gitignored.  
Deliverable: Terminal log + a note on why artifacts stay out of git.

12) Sanity check: tiny script reads/writes Parquet via fsspec  [✅ Completed]  
Goal: Prove your env can round‑trip a small dataframe to `file://`.  
Deliverable: `data/smoke/test.parquet` present; code snippet saved to `tests/unit/test_io_smoke.py`.

## C) City Configuration (8 hrs)
13) Confirm the 3 cities  
Goal: Lock in Berlin, Hamburg, Munich.  
Deliverable: Short `docs/cities.md` with rationale and links.

14) Draft `config/schemas/providers.yml` & cities schema  
Goal: Ensure providers list includes cams, aurora, gefs_aerosol.  
Deliverable: Updated schema stubs (even if minimal).

15) Create `config/cities/berlin.yml` (update if exists)  
Goal: Stations, bounds, timezone, providers, holidays.  
Deliverable: Valid YAML for Berlin.

16) Create `config/cities/hamburg.yml`  
Goal: Same fields as Berlin.  
Deliverable: Valid YAML for Hamburg.

17) Create `config/cities/munich.yml`  
Goal: Same fields as Berlin.  
Deliverable: Valid YAML for Munich.

18) Add simple JSON Schema for city files  
Goal: Basic required fields + types.  
Deliverable: `config/schemas/cities.json` minimal validator.

19) Write a schema‑validation script  
Goal: Load every `config/cities/*.yml` and assert schema compliance.  
Deliverable: `apps/tools/validate_cities.py` + passing run.

20) Create a city‑config “README”  
Goal: Teach future-you how to add a new city.  
Deliverable: `docs/howto_add_city.md`.

## D) Observations & Benchmarks Wiring (6 hrs)
21) Decide observation source(s) for PM₂.₅  
Goal: Pick authoritative public source for the 3 cities; document format/latency.  
Deliverable: `docs/obs_source.md`.

22) Write observation schema stubs  
Goal: `config/schemas/raw/observations.json` and `curated/observations.json`.  
Deliverable: Both schema files.

23) Create obs ETL skeleton  
Goal: `apps/etl/obs_pm25.py` CLI args (city, date range), no logic yet.  
Deliverable: Runs with `--help`.

24) Implement obs ETL (single city, single day)  
Goal: Fetch→parse→normalize→write Parquet.  
Deliverable: Files under `data/raw/observations/...` and `data/curated/observations/...`.

25) Extend obs ETL to 3 cities + date span  
Goal: Loop cities; chunk by day; idempotent writes.  
Deliverable: Successful backfill for last 7 days.

26) Add obs unit & contract tests  
Goal: Validate schema and edge cases.  
Deliverable: Tests passing.

## E) Provider ETL – CAMS, Aurora, GEFS (12 hrs)
27) CAMS ETL skeleton  
Goal: `apps/etl/cams_pm25.py` CLI with city/date args.  
Deliverable: Script runs `--help`.

28) CAMS ETL for one city/day  
Goal: Subset to city bounds, write raw & curated Parquet.  
Deliverable: Day of data for Berlin.

29) CAMS ETL for 3 cities, 7 days  
Goal: Robust looping + retries.  
Deliverable: Data populated.

30) Aurora ETL skeleton  
Goal: `apps/etl/aurora_pm25.py` CLI parity with CAMS.  
Deliverable: `--help` works.

31) Aurora ETL for one city/day  
Goal: Parse/normalize units; write Parquet.  
Deliverable: Berlin day done.

32) Aurora ETL for 3 cities, 7 days  
Goal: Populate curated data.  
Deliverable: Success log.

33) GEFS ETL skeleton  
Goal: `apps/etl/gefs_aerosol.py` CLI; plan NetCDF/GRIB read.  
Deliverable: `--help` works.

34) GEFS ETL for one city/day  
Goal: Subset/interpolate; unit conversions; write raw/curated.  
Deliverable: Berlin day done.

35) GEFS ETL for 3 cities, 7 days  
Goal: Full loop + retries aligned with 6‑hour cycles.  
Deliverable: Populated data.

36) Provider ETL unit tests  
Goal: Metrics: row counts, NaN handling, tz alignment.  
Deliverable: Tests in `tests/unit/etl_*`.

37) Provider contract tests  
Goal: Validate curated schemas.  
Deliverable: `tests/contract/` passing.

38) ETL README  
Goal: Explain how to run any provider ETL.  
Deliverable: `docs/howto_run_etl.md`.

## F) Feature Engineering & Lags (8 hrs)
39) Feature builder skeleton  
Goal: `apps/features/build.py` to read curated obs + providers → features.  
Deliverable: CLI with `--city`, `--asof`, `--horizon`.

40) Join CAMS + Aurora + GEFS to obs (one city/day)  
Goal: Proper time alignment.  
Deliverable: Berlin features for 1 day.

41) Add calendar features  
Goal: DOW, month, holiday flag.  
Deliverable: New columns appear.

42) Add lag features from observations  
Goal: t‑1h/t‑24h or t‑1d, no leakage.  
Deliverable: Lags computed.

43) Add lag features from provider forecasts  
Goal: Previous cycle’s same‑lead forecast, no leakage.  
Deliverable: New lag columns.

44) Feature schema & tests  
Goal: Minimal `config/schemas/features.json`; unit tests.  
Deliverable: Tests passing.

45) Multi‑city build  
Goal: Loop 3 cities; last 14 days of features.  
Deliverable: Feature Parquets.

46) Feature quality report  
Goal: Nulls, value ranges, simple correlations.  
Deliverable: `docs/feature_qc.md`.

## G) Baseline Bias‑Correction Benchmark (6 hrs)
47) Rolling mean bias correction (per PRD)  
Goal: Implement per‑city rolling mean; subtract from forecast.  
Deliverable: Function + tests.

48) Optional scaling (a + b×forecast)  
Goal: Linear fit on recent window.  
Deliverable: Function + test.

49) Simple regime split (wind <2 vs ≥2 m/s)  
Goal: Two bias models by regime.  
Deliverable: Branching logic + test.

50) Physical constraints  
Goal: Enforce ≥0 and PM₂.₅ ≤ PM₁₀.  
Deliverable: Constraint function + test.

51) Apply bias correction across providers  
Goal: Produce corrected series for CAMS/Aurora/GEFS.  
Deliverable: Curated corrected outputs.

52) Quick bias‑benchmark evaluation  
Goal: MAE and bias vs raw forecasts.  
Deliverable: Table saved to `web/public/metrics/*.csv`.

## H) Global Combiner Model (6 hrs)
53) Prepare training folds  
Goal: Train/validation splits by time; prevent leakage.  
Deliverable: Fold files in `tests/eval/folds/`.

54) Train Ridge baseline (global)  
Goal: Features = three providers + calendar + lags.  
Deliverable: Model artifact + metrics log.

55) Train XGBoost (max_depth=3)  
Goal: Same features; compare to Ridge.  
Deliverable: Model artifact + comparison table.

56) Optional per‑city bias layer on top of global ŷ  
Goal: Fit y ≈ α_c + β_c·ŷ_global; fallback global.  
Deliverable: Coeffs per city + final predictions.

57) Save & register models  
Goal: `models/` holds artifacts + metadata JSON.  
Deliverable: Organized model registry folders.

58) Compute AQI from concentrations (stub)  
Goal: Implement per‑pollutant breakpoints and max‑subindex logic for PM₂.₅.  
Deliverable: `apps/common/aqi.py` + unit tests.

## I) Evaluation Protocol & Reports (6 hrs)
59) Implement ADR‑005 metrics  
Goal: Weighted κ, category accuracy, hit/false alarms, MAE per pollutant.  
Deliverable: `apps/eval/metrics.py` + unit tests.

60) Build evaluation script  
Goal: Compare: raw providers, bias‑corrected, Ridge, XGB, +/− lags.  
Deliverable: `apps/eval/run_eval.py` emits CSVs.

61) Create EvalProtocol.md  
Goal: Document folds, horizons, metrics, thresholds.  
Deliverable: `docs/EvalProtocol.md`.

62) Acceptance gate (Stage 1 criteria)  
Goal: Check day‑ahead MAE ≥15% improvement in ≥60% of days.  
Deliverable: One‑pager pass/fail summary.

## J) Inference, Verification, Publishing (8 hrs)
63) Inference script (multi‑city)  
Goal: `apps/infer/infer_hourly.py` to load latest features → predictions → AQI.  
Deliverable: Runs for the 3 cities.

64) Verification script  
Goal: `apps/verify/verify_hourly.py` to join predictions with observations.  
Deliverable: Outputs verification CSV/Parquet.

65) Export for static dashboard  
Goal: `apps/publish/export_static.py` writes tidy CSV/JSON.  
Deliverable: Files under `web/public/…`.

66) Minimal static dashboard stub  
Goal: A simple HTML/JS (or just CSV preview).  
Deliverable: `web/public/index.html` showing a simple line chart.

## K) Reliability, Scheduling, and Cost (4 hrs)
67) Add retries & logging  
Goal: Wrap network calls with retries; structured logs.  
Deliverable: Exceptions handled; retry count configurable.

68) Local scheduling plan  
Goal: Document cron timings for each stage.  
Deliverable: `docs/schedule_local.md`.

69) Smoke test end‑to‑end  
Goal: Run a mini‑pipeline for one day; verify outputs.  
Deliverable: A green “smoke run” checklist.

70) Cost/readiness check  
Goal: Note steps to flip `file://` → `s3://` and expected costs.  
Deliverable: `docs/cloud_ready.md`.

## L) Tests & Documentation Polish (2 hrs)
71) Expand unit/contract/eval tests coverage  
Goal: Add missing edge‑case tests.  
Deliverable: Coverage report improvement.

72) Contributor docs & run‑books  
Goal: HOWTOs for adding provider, adding city, debugging failures.  
Deliverable: Three short HOWTOs in `docs/`.
