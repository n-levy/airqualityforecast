# PM2.5 Forecasting Project – Context & Goals

## Why I’m Doing This
- I want to **learn how machine learning forecasting models are produced in the industry**, following modern **MLOps best practices**.
- I want **hands-on experience** building, deploying, and maintaining an ML system that delivers value to the public.
- At work, I have assumed responsibility for a team developing ML forecasting models. This project is a **practical sandbox** to deepen my understanding.
- Learn how forecasting models are built and maintained, applying MLOps best practices in a realistic setting.
- Gain hands-on experience with the full lifecycle: data ingestion, feature building, modeling, evaluation, deployment, and monitoring.
- Build a system that provides public value while serving as a personal learning sandbox.
- This is a solo, hobby project; operational simplicity and low cost are priorities.

## Core Objective
Produce city-level AQI forecasts for Germany that outperform public benchmark forecasts such as CAMS and Aurora, using a simple, maintainable modeling approach.

## Roadmap Vision
1. Stage 1 – Multi-city PM₂.₅ Stage 1
   - Forecast PM₂.₅ concentrations for the top 3 German cities (Berlin, Munich, Hamburg).
   - Combine CAMS, Aurora and NOAA GEFS-Aerosol predictions in a global model with optional per-city bias correction.
   - Display live and historical comparisons:
     - My forecasts
     - CAMS
     - Aurora
     - NOAA GEFS-Aerosol
     - Observations
     - Accuracy metrics (MAE, bias, category accuracy, κw, exceedance rates) over selectable periods.

2. Stage 2 – Cloud Migration
   - Move from local-hosted to cloud-based production.
   - Keep running costs below €10/month

3. Stage 3 – Per-pollutant AQI
   - Extend to PM₁₀, O₃, and NO₂.
   - Train separate global + bias-corrected models per pollutant.
   - Compute AQI from predicted concentrations using standard breakpoints.
   - Track dominant pollutant for each AQI forecast.

## Key Requirements
- Transparency: Verification against open, authoritative data.
- Accuracy Benchmarking: Side-by-side comparison with CAMS, Aurora and NOAA GEFS-Aerosol.
- Config-driven Design: New cities/pollutants added via config files, no code rewrite.
- Operational Simplicity: Minimal manual steps; light compute requirements.
- Low Cost: Target <€10/month after migration.
- Public Dashboard: Initially static; later API-backed.
- Extensibility: Architecture supports adding pollutants, locations, and features easily.

## Stage 1 Definition (v1.1)
- Scope: PM₂.₅ forecasts for 3 major German cities.
- Inputs: CAMS PM₂.₅, Aurora PM₂.₅, calendar features, lag features from observations and provider forecasts.
- Model: Global Ridge or XGBoost (depth=3) + optional per-city bias correction.
- Evaluation: MAE, bias, κw, category accuracy, exceedance hit/false alarm rates.
- Hosting: Local machine; output pushed to static dashboard.
- UI: Select city, lead time, and metric; view historical trends; download datasets.
- Database: Historical forecasts of the model, forecasts of the three benchmarks, and actuals, per city, hourly. 

## Technical Constraints
- No admin rights on development machine; all tooling must run in user space.
- No Docker Desktop locally; container builds via cloud CI/CD.
- Python envs via virtualenv or conda/mamba in user space.
- Scheduling via user-space cron or manual until cloud migration.
- Data and models stored in configurable user paths.
- Architecture must allow switching from file:// to s3:// or similar without code changes.

---

For scope and stages see `ROADMAP.md`; for metrics see `EVAL_METRICS.md`; for features see `FEATURES.md`; for providers see `PROVIDERS.md`.
