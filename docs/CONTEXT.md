# Global Air Quality Forecasting System – Context & Goals

## Why I'm Doing This
- I want to **learn how machine learning forecasting models are produced in the industry**, following modern **MLOps best practices**.
- I want **hands-on experience** building, deploying, and maintaining a global-scale ML system that delivers public health value worldwide.
- At work, I have assumed responsibility for a team developing ML forecasting models. This project serves as a **comprehensive practical sandbox** to deepen my understanding of complex, multi-regional systems.
- Learn how large-scale forecasting systems are architected, with continental standardization and local adaptations.
- Gain hands-on experience with the full lifecycle: global data ingestion, multi-standard feature engineering, ensemble modeling, regional validation, and health warning systems.
- Build a system that provides critical public health value across 5 continents while serving as an advanced learning platform.
- This remains a manageable project prioritizing operational simplicity, public data sources, and cost-effectiveness.

## Core Objective
Create a comprehensive global air quality forecasting system covering 100 cities across 5 continents that outperforms regional benchmark forecasts using local AQI standards, ensemble modeling, and health warning systems - all powered exclusively by public data sources.

## System Vision (Current - Stage 3 Complete)
**Global 100-City Framework**:
- **Coverage**: 100 cities with highest pollution levels across 5 continents
- **Standards**: 11 regional AQI calculation systems (EPA, EAQI, Indian, Chinese, etc.)
- **Data Sources**: Public APIs only - zero personal authentication required
- **Health Focus**: Sensitive group and general population warnings
- **Architecture**: Continental standardization with regional customization

**Continental Implementation**:
- **Europe (20 cities)**: EEA + CAMS + National networks → EAQI standard
- **North America (20 cities)**: EPA/Environment Canada + NOAA → EPA/Canadian/Mexican standards
- **Asia (20 cities)**: Government portals + WAQI + NASA → Local national standards
- **Africa (20 cities)**: WHO + NASA satellites + Research → WHO guidelines
- **South America (20 cities)**: Government portals + NASA + Research → EPA/Chilean adaptations

3. Stage 3 – Cloud Migration
   - Migrate to low-cost cloud execution (scheduled containers, object storage).
   - Keep running costs below €10/month.

## Key Requirements
- Transparency: Verification against open, authoritative data.
- Accuracy Benchmarking: Side-by-side comparison with CAMS and NOAA GEFS-Aerosol.
- Config-driven Design: New cities/pollutants added via config files, no code rewrite.
- Operational Simplicity: Minimal manual steps; light compute requirements.
- Low Cost: Target <€10/month after migration.
- Public Dashboard: Initially static; later API-backed.
- Extensibility: Architecture supports adding pollutants, locations, and features easily.

## Stage 1 Definition (v1.1)
- Scope: PM₂.₅ forecasts for 3 major German cities.
- Inputs: CAMS PM₂.₅, NOAA GEFS-Aerosol PM₂.₅, calendar features, lag features from observations and provider forecasts.
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
