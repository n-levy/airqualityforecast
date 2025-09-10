# Product Requirements Document (PRD) – Global Air Quality Forecasting System

## 1. Background & Context
See `CONTEXT.md` for the full rationale, goals, and constraints.

## 2. Purpose
Create a comprehensive global air quality forecasting system covering 100 cities across 5 continents that outperforms regional benchmarks using ensemble forecasting methods, with local AQI standards and health warning systems for each region.

## 3. Current Scope (Global 100-City System)
- **Coverage:** 100 cities across 5 continents (20 cities each)
- **Continents:** Europe, North America, Asia, Africa, South America
- **Pollutants:** PM2.5, PM10, NO2, O3, SO2
- **AQI Standards:** 11 regional standards (EPA, EAQI, Indian, Chinese, Canadian, etc.)
- **Data Sources:** Public APIs only - no personal API keys required
- **Benchmarks:** 2+ benchmarks per city, standardized per continent
- **Models:** Simple Average + Ridge Regression + Advanced ensemble methods

### **Continental Data Architecture:**
- **Europe (20 cities):** EEA + CAMS + National networks → EAQI standard
- **North America (20 cities):** EPA/Environment Canada + NOAA → EPA/Canadian/Mexican standards
- **Asia (20 cities):** Government portals + WAQI + NASA satellites → Local national standards
- **Africa (20 cities):** WHO + NASA satellites + Research networks → WHO guidelines
- **South America (20 cities):** Government portals + NASA + Research → EPA/Chilean adaptations

### **Outputs:**
- City-specific ensemble predictions with local AQI calculations
- Health warnings for sensitive groups and general population
- Benchmark comparisons (improvement metrics vs. regional standards)
- Walk-forward validation results with health warning accuracy
- Dominant pollutant identification and tracking

## 4. Out of Scope (Current Implementation)
- Real-time data collection (Phase 2 implementation)
- Live dashboard (future enhancement)
- Mobile application
- Commercial API (monetization phase)
- Cities beyond the selected 100

## 5. Success Criteria (Updated for Global System)
- **Data Collection:** All 100 cities successfully collecting data from public sources
- **API Independence:** Zero personal API keys required across entire system
- **Accuracy:** Ensemble models outperform regional benchmarks by ≥10% MAE reduction
- **Health Warnings:** ≥90% recall for health warning detection (minimize false negatives)
- **Coverage:** 2+ validated benchmarks per city across all continents
- **Standards:** Local AQI calculations accurate for all 11 supported standards

## 6. Constraints (Updated)
- **Public Data Only:** No personal API keys, subscriptions, or proprietary data sources
- **Local Implementation:** Development on work computer (no admin rights)
- **Cost Target:** <€50/month for production deployment (100 cities)
- **Compliance:** Respect rate limits and terms of service for all public APIs
- **Attribution:** Proper attribution for scraped public data sources

## 7. Risks & Mitigations (Updated)
- **Data Source Changes:** Government websites may change structure
  - *Mitigation:* Robust scraping with monitoring and fallback sources
- **Rate Limiting:** Public APIs may impose usage restrictions
  - *Mitigation:* Distributed collection, caching, and respectful request patterns
- **Data Quality Variability:** Inconsistent quality across regions
  - *Mitigation:* Multi-source validation and quality scoring systems
- **Regulatory Changes:** API access policies may change
  - *Mitigation:* Multiple backup sources per continent, satellite data fallbacks

## 8. Implementation Roadmap (Current Status)

### ✅ **Completed Phases:**
1. **System Architecture** - Global 100-city framework designed
2. **Multi-Standard AQI Engine** - 11 regional standards implemented
3. **Continental Data Source Mapping** - Public APIs identified and validated
4. **Ensemble Framework** - Forecasting models and validation ready

### 📋 **Next Phase - Data Collection Implementation (8-12 weeks):**
1. **Phase 1:** Data Source Setup (1-2 weeks)
2. **Phase 2:** Collection Implementation (3-4 weeks)
3. **Phase 3:** Validation & QA (2-3 weeks)
4. **Phase 4:** Dataset Finalization (2-3 weeks)

### 🚀 **Future Phases:**
5. **Production Deployment** - Cloud migration and automation
6. **Expansion & Monetization** - Additional cities and commercial offerings

## 9. Technical Requirements

### **Data Requirements:**
- **Frequency:** Hourly data collection where available
- **Retention:** Minimum 1 year historical data per city
- **Quality:** Cross-source validation with outlier detection
- **Standards:** Consistent unit conversion and AQI calculations

### **Performance Requirements:**
- **Latency:** City forecasts generated within 5 minutes
- **Availability:** 99% uptime for data collection
- **Scalability:** Support for expansion to 500+ cities
- **Accuracy:** Health warning false negative rate <10%

### **Security & Compliance:**
- **Authentication:** No stored API keys or credentials
- **Privacy:** No personal data collection
- **Terms Compliance:** Adherence to all public API terms of service
- **Attribution:** Proper credit for all data sources

---

*Status: System fully specified and ready for Phase 1 implementation*
*Last Updated: 2025-09-10*
*Next Milestone: Begin data source setup for 100-city collection*

### Baseline Bias-Correction Benchmark (planned for Stage 1)
We will implement a simple, transparent **per-pollutant rolling bias-correction** as an initial benchmark for Berlin, Hamburg and Munich.

**Purpose:**
- Establish a reliable baseline against which all future models will be evaluated.
- Provide early accuracy gains (expected 10–25 % MAE reduction for PM₂.₅) with minimal complexity.

**Method:**
- Apply rolling mean bias correction (window ≈ 21 days) per pollutant.
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
