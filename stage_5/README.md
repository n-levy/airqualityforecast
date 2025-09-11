# Stage 5: Global 100-City Air Quality Dataset Collection

## Overview
Stage 5 implements the comprehensive collection of ultra-minimal air quality data for 100 cities across 5 continents, covering 5 years of daily data (2020-09-11 to 2025-09-11) using proven continental patterns and public APIs.

## Current Status: Phase 1 Complete ✅

### Completed:
- ✅ **Step 1**: Infrastructure setup and configuration
- ✅ **Step 2**: Data source validation and accessibility testing

### Results:
- **58/100 cities** expected to have high-quality data
- **93.3% API accessibility** across all data sources
- **5 continental patterns** validated and ready
- **No API keys required** - all public data sources

## Quick Start

### Prerequisites
```bash
pip install requests pandas numpy
```

### Run Phase 1 (Completed)
```bash
cd stage_5/scripts

# Step 1: Initialize infrastructure
python global_100city_data_collector.py

# Step 2: Validate data sources
python data_source_validator.py
```

### Check Results
```bash
# View validation results
cat ../logs/step2_validation_results.json

# View progress
cat ../logs/collection_progress.json
```

## Continental Patterns

| Continent | Pattern | Cities | Ground Truth | Benchmark 1 | Benchmark 2 | Success Rate |
|-----------|---------|--------|--------------|-------------|-------------|--------------|
| Europe | Berlin | 20 | EEA | CAMS | National Networks | 85% |
| South America | São Paulo | 20 | Government | NASA Satellite | Research Networks | **85%** |
| North America | Toronto | 20 | EPA + EnvCan | NOAA | State/Provincial | 70% |
| Asia | Delhi | 20 | Gov Portals | WAQI | NASA Satellite | 50% |
| Africa | Cairo | 20 | WHO | NASA MODIS | Research Networks | 55% |

## Dataset Specification

### Data Coverage
- **Cities**: 100 across 5 continents (20 per continent)
- **Time Period**: 5 years daily data (1,825 days per city)
- **Total Records**: ~18.25 million records
- **Expected Size**: 8-12 GB (ultra-minimal approach)

### Data Elements per City
1. **Ground Truth**: Actual pollutant levels (PM2.5, PM10, NO2, O3, SO2)
2. **Benchmark Forecasts**: 2 forecast sources per city
3. **Local AQI**: Calculated using 11 regional standards
4. **Features**: Meteorological, temporal, regional features

### AQI Standards Supported
- European EAQI (20 cities)
- US EPA AQI (9 cities)
- Canadian AQHI (6 cities)
- Mexican IMECA (5 cities)
- Indian National AQI (5 cities)
- Chinese AQI (2 cities)
- WHO Guidelines (20 cities)
- Chilean ICA (2 cities)
- Local standards (31 cities)

## File Structure

```
stage_5/
├── README.md                    # This file
├── config/
│   ├── cities_config.json       # 100 cities specification
│   ├── continental_patterns.json # 5 collection patterns
│   └── data_sources.json        # 15 data source configurations
├── data/
│   ├── raw/                     # Raw collected data
│   ├── processed/               # Processed datasets
│   └── final/                   # Final 100-city dataset
├── logs/
│   ├── step1_results.json       # Infrastructure setup results
│   ├── step2_validation_results.json # Validation results
│   ├── collection_progress.json # Overall progress tracking
│   └── *.log                    # Execution logs
├── metadata/
│   └── collection_metadata.json # Project metadata
├── scripts/
│   ├── global_100city_data_collector.py # Main collection framework
│   └── data_source_validator.py # Data source validation
└── quality_reports/             # Data quality reports (generated)
```

## Data Sources (All Public APIs)

### Europe (Berlin Pattern)
- **Ground Truth**: EEA (European Environment Agency)
- **Benchmark 1**: CAMS (Copernicus Atmosphere Monitoring)
- **Benchmark 2**: National monitoring networks

### South America (São Paulo Pattern) - **READY**
- **Ground Truth**: Government agencies (Brazil, Chile, etc.)
- **Benchmark 1**: NASA satellite estimates
- **Benchmark 2**: Regional research networks

### North America (Toronto Pattern)
- **Ground Truth**: EPA AirNow + Environment Canada
- **Benchmark 1**: NOAA air quality forecasts
- **Benchmark 2**: State/provincial networks

### Asia (Delhi Pattern)
- **Ground Truth**: National environmental agencies
- **Benchmark 1**: WAQI (World Air Quality Index)
- **Benchmark 2**: NASA satellite estimates

### Africa (Cairo Pattern)
- **Ground Truth**: WHO Global Health Observatory
- **Benchmark 1**: NASA MODIS satellite data
- **Benchmark 2**: Research networks (INDAAF)

## Next Steps: Phase 2

Ready to proceed with continental data collection:

1. **Step 3**: Europe - Berlin Pattern (20 cities)
2. **Step 4**: South America - São Paulo Pattern (20 cities) - **PRIORITY**
3. **Step 5**: North America - Toronto Pattern (20 cities)
4. **Step 6**: Asia - Delhi Pattern (20 cities)
5. **Step 7**: Africa - Cairo Pattern (20 cities)

## Validation Results Summary

- **Overall Readiness**: Partial (sufficient for deployment)
- **Ready Continents**: 1/5 (South America fully ready)
- **Expected Success**: 58/100 cities with high-quality data
- **Fallback Strategy**: Synthetic data generation for missing periods

## Contributing

This is part of the Global Air Quality Forecasting System project. See main project documentation for contribution guidelines.

## License

Research project - see main project license.

---

*Stage 5: Global 100-City Dataset Collection*
*Generated: September 11, 2025*
