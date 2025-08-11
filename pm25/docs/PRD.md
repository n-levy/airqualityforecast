# Product Requirements Document (PRD) – PM2.5 Forecasting Project

## 1. Background & Context
See `CONTEXT.md` for the full rationale, goals, and constraints.

## 2. Purpose
Produce PM2.5 forecasts for Berlin, Hamburg and Munich that outperform public benchmarks (e.g., CAMS) in measurable, transparent ways, with a public dashboard showing forecasts, actuals, and accuracy comparisons.

## 3. Scope (Stage 1)
- **Location:** Berlin, Hamburg, Munich
- **Metric:** PM2.5
- **Benchmarks:** CAMS forecasts, Aurora, NOAA GEFS-Aerosol
- **Outputs:**
  - Our predictions
  - Benchmark predictions
  - Actual PM2.5
  - Accuracy metrics (κw, category accuracy, hit/false alarms, MAE per pollutant)
- **UI:**
  - Static dashboard
  - User controls for lead time, date range, and metric selection
  - Downloadable CSV/JSON

## 4. Out of Scope (Stage 1)
- Other cities
- Other AQI components (PM10, O₃, NO₂)
- Live API
- Mobile app

## 5. Success Criteria
- Day-ahead MAE reduced by ≥15% vs. benchmark in ≥60% of days over last 60 days.
- Dashboard available with selectable metrics/periods.
- Public verification against open data.

## 6. Constraints
- Local-first implementation on work computer (no admin rights).
- Cloud-ready architecture for easy migration.
- Low-cost target: <€10/month after migration.

## 7. Risks & Mitigations
- **Data outages:** Use multiple public sources.
- **Model drift:** Regular retraining.
- **Cloud migration delays:** Build with fsspec paths and container-ready code.

## 8. Roadmap
1. Stage 1 (Berlin, Hamburg, Munich, PM2.5)
2. Add cities
3. Add AQI components
4. Cloud deployment
5. Monetization

### Baseline Bias-Correction Benchmark (planned for Stage 1)
We will implement a simple, transparent **per-pollutant rolling bias-correction** as an initial benchmark for Berlin, Hamburg and Munich.

**Purpose:**  
- Establish a reliable baseline against which all future models will be evaluated.  
- Provide early accuracy gains (expected 10–25 % MAE reduction for PM₂.₅) with minimal complexity.  

**Method:**  
- Apply rolling mean bias correction (window ≈ 21 days) per pollutant.  
- Add 
- Optional scale adjustment via linear regression (obs ≈ a + b × forecast).  
- Simple regime split on wind speed (< 2 m/s vs ≥ 2 m/s).  
- Enforce physical constraints (≥ 0; PM₂.₅ ≤ PM₁₀).  
- Recompute AQI from corrected pollutant values.  

**Evaluation metrics:**  
- MAE, RMSE, bias per pollutant.  
- AQI band accuracy and threshold-crossing skill.  

**Acceptance criteria:**  
- PM₂.₅ MAE reduced by ≥ 10 % vs raw forecast over recent 30-day window.  
- Bias within ± 1 µg/m³.  
- AQI categorical accuracy improved by ≥ 5 pp.

For metrics see `EVAL_METRICS.md`; for features see `FEATURES.md`; for providers see `PROVIDERS.md`.

For metrics see `EVAL_METRICS.md`; for features see `FEATURES.md`; for providers see `PROVIDERS.md`.

For metrics see `EVAL_METRICS.md`; for features see `FEATURES.md`; for providers see `PROVIDERS.md`.
