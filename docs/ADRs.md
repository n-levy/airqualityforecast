# ADR 001 – Data Storage Format and Access

## Status : Accepted

## Context
We need to store raw, curated, and feature data in a way that is efficient, cloud-migratable, and supports both local and cloud execution.

## Decision
We will use **Apache Parquet** files stored via **fsspec** paths.
- Local Stage 1: `file://` paths in user directory.
- Cloud: `s3://` paths with the same interface.

## Consequences
- Easy migration from local to cloud by changing environment variables.
- Columnar storage for efficient analytics.
- Supports partitioning by city, date, and data type.

-------------------------------------------------------------------------------------------------------------------------------------------

# ADR 002 – Model Family
## Status: Accepted

## Context
We need a model that can handle tabular, temporal features, work with limited historical data, and run quickly for multiple lead times.

## Decision
We will use gradient-boosted trees (XGBoost) as a bias-correcting downscaler over public forecasts and meteorological inputs.

## Consequences
Strong performance for tabular features.
Interpretability through feature importance.
Fast retraining and inference.
Easy to implement in Python with open-source libraries.

-------------------------------------------------------------------------------------------------------------------------------------------

# ADR 003 – Job Scheduling
## Status: Accepted

## Context
We need to run ingestion, feature building, inference, verification, and publishing on a regular schedule.

## Decision
Local Stage 1: user-space cron jobs or manual runs.
Cloud: scheduled container jobs (e.g., Cloud Run Jobs, Fly.io Machines).

## Consequences
Minimal setup locally.
Easy migration to cloud-native scheduling.
Decoupled job definitions from infrastructure.

-------------------------------------------------------------------------------------------------------------------------------------------

# ADR 004 - Modeling Approach for AQI Forecasting
## Status: Accepted

## Context
We require a lightweight, solo-maintainable approach to improve air quality index (AQI) forecasts by combining CAMS, Aurora and NOAA GEFS-Aerosol (see `PROVIDERS.md` for details) model outputs with minimal additional features (calendar events such as holidays, lag features from observations and provider forecasts). The solution should be easy to extend to new cities and pollutants.

## Decision
We will implement a global combiner model per pollutant (starting with PM₂.₅, PM₁₀, NO₂, O₃), using CAMS, Aurora and NOAA GEFS-Aerosol (see `PROVIDERS.md` for details) concentrations along with basic calendar features and lag features (see `FEATURES.md` for details). A simple model (Ridge regression or XGBoost with max\_depth=3) will be used, followed by an optional per-city bias correction layer (y ≈ α\_c + β\_c·ŷ\_global) trained on recent data, with a global fallback.

AQI will be computed from predicted concentrations using a configuration-driven breakpoint table, taking the maximum sub-index across pollutants. This preserves pollutant-specific patterns, allows partial AQI when data is missing, and supports pollutant-level diagnostics.

## Consequences
1. Minimal computational and maintenance burden.
2. Modular design allows phased rollout: Stage 1 PM₂.₅ only, Phase 2 add O₃ and NO₂, Phase 3 add PM₁₀.
3. Can extend to additional pollutants without re-engineering the pipeline.

-------------------------------------------------------------------------------------------------------------------------------------------

# ADR 5: Evaluation Metrics for AQI Forecasting

## Status: Accepted

## Context
AQI is an ordinal categorical variable with public health implications. Metrics must capture both exact matches and near misses, as well as pollutant-level contributions to overall AQI accuracy.

## Decision
Example metrics are listed in `EVAL_METRICS.md`. These are **examples only**; the final set will be decided during Stage 1 evaluation.

## Consequences
1. Balanced view of forecast skill at both AQI and pollutant levels.
2. Ability to communicate results effectively to both technical and public audiences.
3. Supports targeted model improvements by pollutant.

### Additional Details on Metric Selection Criteria
The choice of final evaluation metrics will balance several priorities:
1. **Public Health Relevance** – Metrics should capture the model's ability to predict hazardous pollution levels accurately.
2. **Robustness Across Conditions** – The chosen metrics must produce stable and meaningful results across seasons and varying meteorological conditions.
3. **Transparency & Reproducibility** – Calculations will be open, reproducible, and well-documented to allow external verification.
4. **Operational Actionability** – Metrics should guide both model retraining schedules and real-time forecast adjustments.
5. **Trade-off Awareness** – In cases where higher exact match accuracy conflicts with broader tolerance for near misses, these trade-offs will be explicitly evaluated and documented.