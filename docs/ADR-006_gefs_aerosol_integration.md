# ADR 006 – Integration of NOAA GEFS-Aerosol as Additional Forecast Provider

## Status
Accepted

See `PROVIDERS.md` for general provider descriptions and metadata.

## Context
The current modeling approach (ADR-004) combines CAMS and Aurora forecasts with simple calendar features in a global combiner model, plus an optional per-city bias correction.  
To improve accuracy and portability to other countries, we will add **NOAA GEFS-Aerosol** as a third forecast provider:
- **Global coverage** → supports expansion beyond Germany.
- **Independent model physics** → increases diversity in the ensemble.
- **Open access** → consistent with project’s transparency and low-cost goals.

## Decision
- Implement a **source adapter** `apps/etl/gefs_aerosol.py`:
  - Ingest GEFS-Aerosol PM₂.₅, PM₁₀, O₃, NO₂ fields from open NetCDF/GRIB files.
  - Subset to city bounding box and interpolate to station locations.
  - Store **raw** and **curated** Parquet datasets using ADR-001 conventions.
- Update **config**:
  - Add provider block under `config/schemas/providers.yml`.
  - Add `gefs_aerosol` to `config/cities/<city>.yml` provider lists.
- Extend **feature builder** to join GEFS pollutant columns alongside CAMS/Aurora.
- Add **GEFS-only** and **multi-provider** A/B experiments to evaluation (ADR-005).
- Maintain model simplicity:
  - Use the same Ridge/XGB(max_depth=3) global combiner.
  - Treat GEFS variables as additional predictors; retrain bias layer unchanged.
- **Schema updates**:
  - Add `raw/noaa_gefs_aerosol.json` and `curated/noaa_gefs_aerosol.json` to `config/schemas/`.
- **Testing**:
  - Unit: variable mapping, unit conversions.
  - Contract: schema compliance for raw and curated data.
  - Eval: fixed-fold comparison of CAMS+Aurora vs CAMS+Aurora+GEFS.
- **Scheduling**: Run ingestion every 6h aligned with GEFS cycles (ADR-003).
- **Verification**:
  - Include GEFS in benchmark plots/tables for transparency.
  - Track KPIs per ADR-005 (κw, category accuracy, MAE per pollutant, hit/false alarms).

## Consequences
1. **Accuracy** – Added forecast diversity should reduce bias and improve category accuracy, especially for episodic pollution events.
2. **Portability** – Enables near-drop-in deployment to other Western countries where Aurora is unavailable.
3. **Complexity** – Minimal; adding a provider fits existing ETL → features → train → verify structure (REPO_TREE_STAGE0).
4. **Cost** – No increase; GEFS-Aerosol is public and free.

## References
- ADR-001 Storage
- ADR-002 Model Family
- ADR-003 Scheduler
- ADR-004 Modeling
- ADR-005 Evaluation
