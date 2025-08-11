ADR: Modeling Approach for AQI Forecasting

Status: Accepted

Context: We require a lightweight, solo-maintainable approach to improve air quality index (AQI) forecasts by combining CAMS, Aurora and NOAA GEFS-Aerosol (see `PROVIDERS.md` for details) model outputs with minimal additional features (calendar events such as holidays, lag features from observations and provider forecasts). The solution should be easy to extend to new cities and pollutants.

Decision:
We will implement a global combiner model per pollutant (starting with PM₂.₅, PM₁₀, NO₂, O₃), using CAMS, Aurora and NOAA GEFS-Aerosol (see `PROVIDERS.md` for details) concentrations along with basic calendar features and lag features (see `FEATURES.md` for details). A simple model (Ridge regression or XGBoost with max\_depth=3) will be used, followed by an optional per-city bias correction layer (y ≈ α\_c + β\_c·ŷ\_global) trained on recent data, with a global fallback.

AQI will be computed from predicted concentrations using a configuration-driven breakpoint table, taking the maximum sub-index across pollutants. This preserves pollutant-specific patterns, allows partial AQI when data is missing, and supports pollutant-level diagnostics.

Consequences:

1. Minimal computational and maintenance burden.
2. Modular design allows phased rollout: Stage 1 PM₂.₅ only, Phase 2 add O₃ and NO₂, Phase 3 add PM₁₀.
3. Can extend to additional pollutants without re-engineering the pipeline.

