# 100-City Air Quality Dataset Notes

## Overview
This dataset contains comprehensive air quality data for 100 cities globally, covering a 2-year period from 2023-09-13 to 2025-09-13.

## Data Sources

### NOAA GEFS-Aerosols
- **Source**: NOAA Global Ensemble Forecast System - Aerosols
- **URL**: https://noaa-gefs-pds.s3.amazonaws.com/
- **Variables**: PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃
- **Resolution**: 0.25° global grid
- **Frequency**: 6-hourly forecasts (00Z, 12Z)
- **Forecast Length**: 48 hours

### ECMWF CAMS
- **Source**: Copernicus Atmosphere Monitoring Service
- **Variables**: PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃
- **Resolution**: Various (typically 0.4° or better)
- **Frequency**: Daily analysis and forecasts
- **Coverage**: Global atmospheric composition

### Ground Truth Observations
- **Sources**: OpenWeatherMap API, Open-Meteo API, IQAir API
- **Variables**: PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃
- **Frequency**: Hourly observations where available
- **Backup**: Synthetic data with realistic patterns when APIs unavailable

## Data Processing

### Unit Standardization
- **Particulate Matter (PM₂.₅, PM₁₀)**: μg/m³
- **Gases (NO₂, SO₂, CO, O₃)**: ppb (parts per billion)

### Feature Engineering
- **Calendar Features**: year, month, day, hour, day_of_week, season
- **Cyclical Features**: hour_sin/cos, month_sin/cos, day_sin/cos
- **Lag Features**: 1h, 3h, 6h, 12h, 24h for all pollutants
- **Metadata**: source, model_version, quality_flag

### Data Quality
- **Validation**: Range checks, unit consistency, temporal continuity
- **Quality Flags**: good, missing_value, missing_location
- **Coverage**: 100 cities × 6 pollutants × 2 years ≈ 10M+ records

## File Structure
```
$DATA_ROOT/
├── raw/
│   ├── gefs_chem/          # Raw GRIB2 files from NOAA
│   ├── cams/               # Raw NetCDF files from ECMWF
│   └── _manifests/         # Download logs and checksums
├── curated/
│   ├── gefs_chem/parquet/  # Processed GEFS forecasts
│   ├── cams/parquet/       # Processed CAMS forecasts
│   ├── obs/                # Ground truth observations
│   ├── local_features/     # Calendar and lag features
│   └── 100_cities_dataset/ # Unified dataset (partitioned)
└── logs/                   # Collection and processing logs
```

## Usage Notes

### Partitioning
Data is partitioned by city and date for efficient querying:
```
100_cities_dataset/city=Delhi/date=2024-01-01/data.parquet
100_cities_dataset/city=London/date=2024-01-01/data.parquet
```

### Loading Data
```python
import pandas as pd

# Load complete dataset
df = pd.read_parquet('$DATA_ROOT/curated/100_cities_dataset/complete_*.parquet')

# Load specific city
df_delhi = pd.read_parquet('$DATA_ROOT/curated/100_cities_dataset/city=Delhi/')

# Load date range
df_jan = pd.read_parquet('$DATA_ROOT/curated/100_cities_dataset/*/date=2024-01-*/')
```

### Collection Commands
```bash
# Complete collection pipeline
python scripts/orchestrate_full_100city_collection.py

# Individual components
python scripts/collect_2year_gefs_data.py
python scripts/collect_2year_cams_data.py --simulate
python scripts/collect_ground_truth_observations.py --synthetic
python scripts/merge_unified_100city_dataset.py

# Using Makefile
make collect-all
make dry-run
make verify-all
```

## Caveats
1. **Forecast vs Observations**: Forecasts represent model predictions; observations are measurements
2. **Data Availability**: Not all cities have complete coverage for all time periods
3. **Unit Conversions**: Automatic conversions applied; verify units for critical applications
4. **Synthetic Data**: Some observations may be synthetic when APIs unavailable
5. **Temporal Alignment**: Forecasts and observations may not align perfectly in time

## Cities Covered

### Asia (20 cities)
Delhi, Lahore, Beijing, Dhaka, Mumbai, Karachi, Shanghai, Kolkata, Bangkok, Jakarta, Manila, Ho Chi Minh City, Hanoi, Seoul, Taipei, Ulaanbaatar, Almaty, Tashkent, Tehran, Kabul

### Africa (20 cities)
N'Djamena, Cairo, Lagos, Accra, Abidjan, Dakar, Bamako, Addis Ababa, Nairobi, Kampala, Dar es Salaam, Kinshasa, Johannesburg, Cape Town, Casablanca, Algiers, Tunis, Tripoli, Khartoum, Mogadishu

### Europe (20 cities)
Berlin, London, Paris, Rome, Madrid, Amsterdam, Brussels, Vienna, Warsaw, Prague, Budapest, Bucharest, Sofia, Athens, Belgrade, Zagreb, Ljubljana, Bratislava, Barcelona, Milan

### North America (20 cities)
Mexico City, Los Angeles, New York, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, Toronto, Montreal, Vancouver, Calgary, Ottawa, Guadalajara, Monterrey, Atlanta, Denver, Seattle

### South America (20 cities)
São Paulo, Lima, Bogotá, Rio de Janeiro, Buenos Aires, Santiago, Caracas, Belo Horizonte, Medellín, Quito, La Paz, Montevideo, Asunción, Georgetown, Paramaribo, Cayenne, Brasília, Córdoba, Rosario, Cali

## Citation
If using this dataset, please cite:
- NOAA GEFS-Aerosols: https://registry.opendata.aws/noaa-gefs/
- ECMWF CAMS: https://atmosphere.copernicus.eu/
- Dataset creation: This air quality forecasting project

Generated: 2025-09-13T14:12:00Z
