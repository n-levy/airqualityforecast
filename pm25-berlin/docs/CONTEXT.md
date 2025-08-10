# PM2.5 Forecasting Project – Context & Goals

## Why I’m Doing This
- I want to **learn how machine learning forecasting models are produced in the industry**, following modern **MLOps best practices**.
- I want **hands-on experience** building, deploying, and maintaining an ML system that delivers value to the public.
- At work, I have assumed responsibility for a team developing ML forecasting models. This project is a **practical sandbox** to deepen my understanding.

## Core Objective
Produce PM2.5 forecasts for **Berlin** that **outperform public benchmark forecasts** in measurable, transparent ways.

## Roadmap Vision
1. **Stage 1 – Berlin PM2.5 MVP**
   - Forecast PM2.5 concentrations in Berlin with better accuracy than benchmarks such as CAMS.
   - Show historical and live comparison of:
     - My predictions.
     - Benchmark predictions.
     - Observed actual PM2.5 values.
     - Accuracy metrics (MAE, bias, SMAPE, exceedance F1) over selectable periods and lead times.

2. **Stage 2 – Expansion**
   - Add other cities.
   - Add other AQI components (e.g., PM10, O₃, NO₂) to aim for **beating overall AQI forecasts**.
   - Possibly expand to related health/environmental metrics.

3. **Stage 3 – Cloud Migration**
   - Flip from local-hosted MVP (v1.1) to a cloud-based production system (v1.2).
   - Maintain low operational costs.
   - Monetize enough to cover expenses.

## Key Requirements
- **Transparency:** Verification against publicly available data (not user-reported).
- **Accuracy Benchmarking:** Show side-by-side performance vs. public forecasts.
- **Local-first Development:** MVP runs entirely on my local machine.
- **Cloud-ready Design:** Use architecture that makes migration trivial.
- **Low Cost:** Target <€10/month in cloud costs after migration.
- **Ease of Extension:** Adding locations, pollutants, and components should require minimal rework.
- **Public Dashboard:** Static site for MVP; later possibly API-backed for live queries.
- **Monetization:** Optional features or data products that cover running costs.

## MVP Definition (v1.1)
- **Scope:** Berlin only, PM2.5 only.
- **Hosting:** Local machine generates forecasts, pushes JSON/CSV to a static dashboard (e.g., Vercel/Netlify).
- **Benchmarks:** CAMS forecasts (and any other relevant public forecasts).
- **Metrics:** MAE, bias, SMAPE, exceedance metrics for lead times (0–24h, 48h).
- **UI Features:**
  - Select lead time and metric.
  - View historical accuracy trends.
  - Download datasets.

## Post-MVP (v1.2 and beyond)
- Add more cities and AQI components.
- Add live API for real-time queries.
- Deploy in the cloud (Cloud Run/Fly.io/S3 + Vercel).
- Explore monetization paths (API keys, premium features, partnerships).

## Technical Constraints
- **Work Computer:** This project will initially run on my work laptop, where I **do not have administrator permissions**.
- **Install Limitations:** Cannot install system-wide software like Docker Desktop; must use tools that can run in **user space**.
- **Python Environments:** Will use **virtualenv** or **conda/mamba** installed in my user directory.
- **Containerization:** Will design as if containerized, but may need to run scripts locally; container builds will be handled via **cloud CI/CD** (e.g., GitHub Actions).
- **Scheduling:** May not have access to systemd timers; will use user-space cron jobs or manual runs until migrated to cloud scheduling.
- **Data & Models:** Stored in user directories; paths must be configurable via environment variables.
- **Cloud Migration:** Architecture must support an easy switch from `file://` to `s3://` or equivalent without code changes.

---

**Last updated:** YYYY-MM-DD
