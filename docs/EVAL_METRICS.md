# GLOBAL AIR QUALITY FORECASTING - EVALUATION METRICS

## Current Status
The **comprehensive evaluation framework** has been defined in `EVALUATION_FRAMEWORK.md`. This document provides a quick reference to the core metrics used across all 100 cities.

## Core Evaluation Metrics

### Individual Pollutant Performance
- **Mean Absolute Error (MAE)** - Primary accuracy metric
- **Root Mean Square Error (RMSE)** - Penalty for large errors
- **Mean Absolute Percentage Error (MAPE)** - Relative accuracy
- **R² Score** - Coefficient of determination
- **Bias** - Systematic over/under-prediction

### AQI Performance (Local Standards)
- **Category Accuracy** - Correct AQI category prediction
- **Weighted Cohen's κ (quadratic)** - Account for category proximity
- **Health Warning Recall** - Sensitivity for health alerts (>90% target)
- **False Negative Rate** - Missed health warnings (<10% target)
- **Precision/Recall** - By AQI category and health threshold

### Health Warning Analysis
- **True Positive Rate** - Correctly identified health warnings
- **False Positive Rate** - Unnecessary health alerts
- **False Negative Rate** - **CRITICAL** - Missed health protection opportunities
- **Precision** - Positive predictive value for health alerts

## Model Comparison Framework

### Core Models
1. **Simple Average** - Baseline ensemble of 2 benchmarks per continent
2. **Ridge Regression** - Optimized ensemble with L2 regularization
3. **Gradient Boosting** - *(Future)* Advanced non-linear ensemble

### Benchmark Comparison
- **Continental Benchmarks** - Regional standard forecasting systems
- **Improvement Metrics** - Percentage improvement over best individual benchmark
- **Statistical Significance** - Paired t-tests and confidence intervals

## Regional Standards Coverage

### 11 AQI Standards Evaluated
- **US EPA AQI** (0-500) - USA cities
- **European EAQI** (1-6) - European cities
- **Indian National AQI** (0-500) - Indian cities
- **Chinese AQI** (0-500) - Chinese cities
- **Canadian AQHI** (1-10+) - Canadian cities
- **Mexican IMECA** - Mexican cities
- **Thai AQI** - Thailand
- **Indonesian ISPU** - Indonesia
- **Pakistani AQI** - Pakistani cities
- **WHO Guidelines** - African cities
- **Chilean ICA** - Chilean cities

## Implementation Priority

### Phase 4 (Current)
✅ **Primary Metrics**: MAE, Category Accuracy, Health Warning Performance
✅ **Core Models**: Simple Average, Ridge Regression
✅ **All Standards**: Local AQI calculations for all 100 cities
✅ **Health Focus**: False negative minimization

### Phase 5 (Future)
🔄 **Extended Metrics**: Additional stakeholder-defined measures
🔄 **Advanced Models**: Gradient Boosting integration
🔄 **Real-time Evaluation**: Live performance monitoring

---

**Reference**: See `EVALUATION_FRAMEWORK.md` for complete methodology
**Implementation**: Ready for Phase 4 global deployment
**Focus**: Public health protection through accurate health warnings

*Last Updated: 2025-09-10*
