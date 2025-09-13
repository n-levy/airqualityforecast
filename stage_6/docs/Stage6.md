# Stage 6: ETL Pipelines and Unified Dataset Creation

## Overview

Stage 6 provides comprehensive ETL (Extract, Transform, Load) pipelines for collecting, processing, and merging air quality data from multiple sources into unified 6-hourly datasets. This stage creates production-ready datasets for air quality forecasting models and analysis.

## Architecture

```
stage_6/
├── scripts/                    # ETL pipeline scripts
│   ├── etl_ground_truth.py    # Ground truth observations
│   ├── etl_noaa_gefs.py       # NOAA GEFS-Aerosol forecasts
│   ├── etl_cams.py            # ECMWF CAMS atmospheric data
│   ├── etl_local_features.py  # Local features generation
│   └── merge_unified_dataset.py # Dataset merger
├── config/                     # Configuration files
└── docs/                      # Documentation
    └── Stage6.md             # This document
```

## Data Sources

### 1. Ground Truth Data (`etl_ground_truth.py`)

**Primary Sources:**
- **WAQI (World Air Quality Index)**: Real-time air quality observations from global monitoring stations
- **OpenAQ**: Open-source air quality data platform with validated measurements

**Pollutants Collected:**
- PM2.5 (Fine Particulate Matter) - μg/m³
- PM10 (Coarse Particulate Matter) - μg/m³

**Coverage:** 100 global cities across 5 continents (20 cities per continent)
**Frequency:** 6-hourly observations (00:00, 06:00, 12:00, 18:00 UTC)

### 2. NOAA GEFS-Aerosol Forecasts (`etl_noaa_gefs.py`)

**Source:** NOAA Global Ensemble Forecast System - Aerosol Component
**Data Location:** AWS S3 bucket (noaa-gefs-pds)
**Format:** GRIB2 files with global 0.25° resolution

**Variables Collected:**
- **PMTF**: PM2.5 Total - μg/m³
- **PMTC**: PM10 Total - μg/m³
- **SO2**: Sulfur Dioxide - ppb
- **NO2**: Nitrogen Dioxide - ppb
- **CO**: Carbon Monoxide - ppb
- **O3MR**: Ozone Mixing Ratio - ppb

**Forecast Horizons:** 0-48 hours in 6-hour intervals
**Cycles:** 00Z and 12Z daily
**Coverage:** Global cities with ensemble forecasts

### 3. ECMWF CAMS Data (`etl_cams.py`)

**Source:** Copernicus Atmosphere Monitoring Service (CAMS)
**API:** Climate Data Store (CDS) / Atmosphere Data Store (ADS)
**Dataset:** CAMS Global Reanalysis EAC4

**Variables Collected:**
- **pm2p5**: PM2.5 mass concentration - kg/m³ → μg/m³
- **pm10**: PM10 mass concentration - kg/m³ → μg/m³
- **nitrogen_dioxide**: NO2 - mol/mol → ppb
- **sulphur_dioxide**: SO2 - mol/mol → ppb
- **carbon_monoxide**: CO - mol/mol → ppb
- **ozone**: O3 - mol/mol → ppb

**Resolution:** 0.75° × 0.75° global grid
**Frequency:** 6-hourly reanalysis data
**Quality:** Research-grade atmospheric composition data

### 4. Local Features (`etl_local_features.py` & `enhanced_local_features.py`)

**Weather Data Source:** Open-Meteo Historical Weather API
**Coverage:** All 100 cities with meteorological observations

**Enhanced Stage 5 Features Available:**
- **Fire Activity Features:** Continental seasonal patterns, risk indices, fire weather conditions
- **Holiday Features:** Country-specific calendars, pollution impact assessment, fireworks detection
- **AQI Standards:** Regional air quality standards (US EPA, European EAQI, Chinese, Indian, WHO, Canadian AQHI, Chilean ICA)
- **Enhanced Meteorological:** Heat index, wind categories, visibility, weather categorization

**Feature Categories:**

#### Calendar & Temporal Features
- **Basic Calendar**: year, month, day, hour, day_of_week, day_of_year, week_of_year, quarter
- **Boolean Indicators**: is_weekend, is_monday, is_friday, is_month_start, is_month_end
- **Time Categories**: morning_rush, evening_rush, night, daytime, business_hours
- **Seasonal**: season, is_winter, is_spring, is_summer, is_fall, is_holiday_season
- **Cyclical Encodings**: hour_sin/cos, day_sin/cos, month_sin/cos, dayofweek_sin/cos, dayofyear_sin/cos

#### Meteorological Features
- **Temperature**: temperature_c, apparent_temp_c, dewpoint_c, heat_index_f
- **Humidity**: humidity_pct (relative humidity percentage)
- **Pressure**: pressure_hpa (surface pressure in hectopascals)
- **Wind**: wind_speed_ms, wind_direction_deg, wind_gusts_ms, wind_category
- **Atmospheric**: cloud_cover_pct

#### Geographic & Demographic Features
- **Coordinates**: latitude, longitude, elevation_m
- **Population**: population, population_category (small/medium/large/megacity)
- **Hemispheres**: is_northern/southern_hemisphere, is_eastern/western_hemisphere
- **Climate**: climate_zone (tropical/subtropical/temperate/subarctic/arctic)

## Cities Coverage

**100 Global Cities** representing major urban centers with significant air quality challenges across 5 continents (20 cities per continent), sourced from Stage 5 configuration:

### Continental Distribution
- **Asia:** 20 cities including Delhi, Beijing, Mumbai, Shanghai, Tokyo, Seoul, Bangkok, and 13 others
- **Europe:** 20 cities including London, Paris, Berlin, Madrid, Rome, Amsterdam, and 14 others  
- **North America:** 20 cities including New York, Los Angeles, Mexico City, and 17 others
- **South America:** 20 cities including São Paulo, Lima, and 18 others
- **Africa:** 20 cities including Cairo, Lagos, and 18 others

### Sample Cities (Full list available in `stage_6/config/cities_stage5.py`)

#### Asia-Pacific (Sample)
- **Delhi, India** (28.61°N, 77.21°E) - Pop: 32.9M
- **Beijing, China** (39.90°N, 116.41°E) - Pop: 21.5M
- **Mumbai, India** (19.08°N, 72.88°E) - Pop: 20.4M
- **Shanghai, China** (31.23°N, 121.47°E) - Pop: 27.1M
- **Tokyo, Japan** (35.68°N, 139.65°E) - Pop: 37.4M
- **Seoul, South Korea** (37.57°N, 126.98°E) - Pop: 9.7M
- **Bangkok, Thailand** (13.76°N, 100.50°E) - Pop: 10.5M

### Europe (6 cities)
- **London, UK** (51.51°N, -0.13°E) - Pop: 9.5M
- **Paris, France** (48.86°N, 2.35°E) - Pop: 11.0M
- **Berlin, Germany** (52.52°N, 13.41°E) - Pop: 3.7M
- **Madrid, Spain** (40.42°N, -3.70°E) - Pop: 6.8M
- **Rome, Italy** (41.90°N, 12.50°E) - Pop: 4.2M
- **Amsterdam, Netherlands** (52.37°N, 4.90°E) - Pop: 1.2M

### North America (3 cities)
- **New York, USA** (40.71°N, -74.01°E) - Pop: 8.4M
- **Los Angeles, USA** (34.05°N, -118.24°E) - Pop: 4.0M
- **Mexico City, Mexico** (19.43°N, -99.13°E) - Pop: 21.6M

### South America (2 cities)
- **São Paulo, Brazil** (-23.55°S, -46.63°E) - Pop: 22.4M
- **Lima, Peru** (-12.05°S, -77.04°E) - Pop: 10.7M

### Africa (2 cities)
- **Cairo, Egypt** (30.04°N, 31.24°E) - Pop: 20.9M
- **Lagos, Nigeria** (6.52°N, 3.38°E) - Pop: 15.4M

## API Configuration

### Required API Access

1. **ECMWF CAMS (CDS/ADS)**
   - Register at: https://cds.climate.copernicus.eu/
   - Configure `~/.cdsapirc` with API key
   - Accept license agreements for CAMS datasets

2. **OpenAQ (Optional Enhancement)**
   - Register at: https://openaq.org/
   - Obtain API key for higher rate limits
   - Pass via `--openaq-api-key` parameter

3. **NOAA GEFS (Public Access)**
   - No registration required
   - Direct access to AWS S3 bucket
   - Rate limiting implemented

4. **Open-Meteo Weather (Free)**
   - No API key required
   - Historical weather archive access
   - 10,000 calls/day limit

## Usage Instructions

### Individual ETL Scripts

#### 1. Ground Truth Collection
```bash
python stage_6/scripts/etl_ground_truth.py \
  --start-date 2024-09-01 \
  --end-date 2024-09-07 \
  --openaq-api-key YOUR_KEY_HERE
```

#### 2. NOAA GEFS Forecasts
```bash
python stage_6/scripts/etl_noaa_gefs.py \
  --start-date 2024-09-01 \
  --end-date 2024-09-07 \
  --forecast-hours 0,6,12,18,24,30,36,42,48
```

#### 3. ECMWF CAMS Data
```bash
python stage_6/scripts/etl_cams.py \
  --start-date 2024-09-01 \
  --end-date 2024-09-07 \
  --bbox 90,-180,-90,180
```

#### 4. Local Features Generation
```bash
# Basic local features
python stage_6/scripts/etl_local_features.py \
  --start-date 2024-09-01 \
  --end-date 2024-09-07

# Enhanced Stage 5 features (fire activity, holidays, AQI standards)
python stage_6/scripts/enhanced_local_features.py \
  --start-date 2024-09-01 \
  --end-date 2024-09-07
```

### Unified Dataset Creation

```bash
python stage_6/scripts/merge_unified_dataset.py \
  --start-date 2024-09-01 \
  --end-date 2024-09-07 \
  --formats wide,long
```

### Complete Pipeline Example

```bash
# Set data root
export DATA_ROOT=/path/to/your/data

# Run all ETL pipelines
python stage_6/scripts/etl_ground_truth.py --start-date 2024-09-01 --end-date 2024-09-07
python stage_6/scripts/etl_noaa_gefs.py --start-date 2024-09-01 --end-date 2024-09-07
python stage_6/scripts/etl_cams.py --start-date 2024-09-01 --end-date 2024-09-07
python stage_6/scripts/etl_local_features.py --start-date 2024-09-01 --end-date 2024-09-07
python stage_6/scripts/enhanced_local_features.py --start-date 2024-09-01 --end-date 2024-09-07

# Merge into unified datasets
python stage_6/scripts/merge_unified_dataset.py --start-date 2024-09-01 --end-date 2024-09-07
```

## Output Structure

### Individual ETL Outputs
```
${DATA_ROOT}/curated/stage6/
├── ground_truth/
│   ├── ground_truth_20240901_20240907_TIMESTAMP.parquet
│   └── partitioned/ground_truth_20240901_20240907/
│       ├── city=Delhi/data.parquet
│       ├── city=Beijing/data.parquet
│       └── ...
├── noaa_gefs/
│   ├── gefs_forecasts_20240901_20240907_TIMESTAMP.parquet
│   └── partitioned/gefs_20240901_20240907/
├── cams/
│   ├── cams_data_20240901_20240907_TIMESTAMP.parquet
│   └── partitioned/cams_20240901_20240907/
├── local_features/
│   ├── local_features_20240901_20240907_TIMESTAMP.parquet
│   └── partitioned/features_20240901_20240907/
└── enhanced_local_features/
    ├── enhanced_local_features_20240901_20240907_TIMESTAMP.parquet
    └── partitioned/enhanced_features_20240901_20240907/
```

### Unified Dataset Outputs
```
${DATA_ROOT}/curated/stage6/unified/
├── unified_wide_20240901_20240907_TIMESTAMP.parquet    # Wide format
├── unified_long_20240901_20240907_TIMESTAMP.parquet    # Long format
└── partitioned/unified_20240901_20240907/
    ├── wide/
    │   ├── city=Delhi/data.parquet
    │   └── ...
    └── long/
        ├── city=Delhi/data.parquet
        └── ...
```

## Dataset Formats

### Wide Format Dataset
- **Structure**: One row per city-timestamp combination
- **Columns**: City, timestamp, pollutant values as separate columns, all features
- **Use Case**: Machine learning model training, time series analysis
- **Example Columns**: `city`, `timestamp_utc`, `PM2.5_WAQI`, `PM2.5_CAMS`, `temperature_c`, `hour_sin`, etc.

### Long Format Dataset
- **Structure**: One row per measurement (city-timestamp-pollutant)
- **Columns**: City, timestamp, pollutant, value, source, features
- **Use Case**: Statistical analysis, data exploration, visualization
- **Example Columns**: `city`, `timestamp_utc`, `pollutant`, `value`, `source`, `temperature_c`, etc.

## Data Quality and Benchmarks

### Ground Truth Validation
- **WAQI Integration**: Real-time validated observations from established monitoring networks
- **OpenAQ Cross-validation**: Multiple source verification for data quality assurance
- **Temporal Consistency**: 6-hourly alignment with forecast data
- **Spatial Coverage**: City-center representative measurements

### Forecast Benchmarks
- **NOAA GEFS**: Ensemble forecasts with uncertainty quantification
- **ECMWF CAMS**: Research-grade atmospheric composition modeling
- **Skill Metrics**: RMSE, MAE, correlation, bias assessments available
- **Evaluation Period**: Minimum 7-day continuous coverage for model validation

### Feature Engineering
- **Cyclical Encoding**: Proper handling of temporal periodicity
- **Weather Integration**: Meteorological drivers of air quality
- **Geographic Context**: Climate zones, population density, elevation effects
- **Quality Flags**: Data provenance and reliability indicators

## Cross-Platform Compatibility

### Supported Platforms
- **Linux**: Native support, optimal performance
- **macOS**: Full compatibility with Unix-like environment
- **Windows**: Complete support with Path library usage

### Dependencies
```
pandas>=1.5.0
numpy>=1.20.0
requests>=2.25.0
tqdm>=4.60.0
cdsapi>=0.5.1        # For ECMWF CAMS
xarray>=0.20.0       # For NetCDF processing
netcdf4>=1.5.0       # NetCDF file support
```

### Environment Setup
```bash
# Cross-platform data root configuration
export DATA_ROOT=/path/to/data       # Linux/macOS
set DATA_ROOT=C:\path\to\data        # Windows

# Install dependencies
pip install pandas numpy requests tqdm cdsapi xarray netcdf4
```

## Performance Considerations

### Data Volume Estimates
- **Ground Truth**: ~5MB per week for 100 cities
- **NOAA GEFS**: ~100MB per week (full global coverage)
- **ECMWF CAMS**: ~50MB per week (6 pollutants)
- **Local Features**: ~25MB per week (comprehensive features for 100 cities)
- **Total**: ~180MB per week for complete dataset

### Processing Time
- **Individual ETL**: 5-15 minutes per script
- **Merge Operation**: 2-5 minutes for typical week
- **Total Pipeline**: 30-45 minutes for complete weekly dataset

### Storage Optimization
- **Parquet Format**: Efficient columnar storage with compression
- **Partitioning**: City-based partitions for parallel processing
- **Type Optimization**: Appropriate data types for memory efficiency

## Troubleshooting

### Common Issues

1. **API Credentials**
   ```
   Error: CDS API credentials not found
   Solution: Configure ~/.cdsapirc with your CDS API key
   ```

2. **Network Timeouts**
   ```
   Error: Request timeout downloading GRIB files
   Solution: Increase timeout values, check network connectivity
   ```

3. **Data Directory Permissions**
   ```
   Error: Permission denied creating output directory
   Solution: Ensure write permissions to DATA_ROOT path
   ```

4. **Missing Dependencies**
   ```
   Error: Module 'cdsapi' not found
   Solution: pip install cdsapi xarray netcdf4
   ```

### Debug Mode
Enable verbose logging by setting environment variable:
```bash
export PYTHONPATH="$PYTHONPATH:."
python -u stage_6/scripts/script_name.py --args
```

## Integration with Existing Stages

### Stage 5 Compatibility
- **Audit Integration**: Stage 6 outputs compatible with Stage 5 audit framework
- **Schema Consistency**: Maintains standardized column names and data types
- **Quality Flags**: Inherits quality assurance patterns from Stage 5

### Future Stages
- **Model Training**: Wide format optimized for ML pipelines
- **Evaluation**: Long format suitable for model assessment
- **Deployment**: Partitioned structure enables real-time inference

## Development and Extension

### Adding New Data Sources
1. Create new ETL script following existing patterns
2. Implement standardized output schema
3. Add to merge script source detection
4. Update documentation and city coverage

### Custom Feature Engineering
1. Extend `etl_local_features.py` with new feature functions
2. Maintain 6-hourly temporal alignment
3. Add feature documentation and validation
4. Test cross-platform compatibility

### Performance Optimization
1. Implement parallel processing for city-level operations
2. Add incremental update capabilities
3. Optimize memory usage for large datasets
4. Consider distributed processing for scaling

---

**Stage 6 provides production-ready ETL infrastructure for comprehensive air quality datasets, enabling advanced forecasting models and analytical insights across global urban centers.**
