# Clean Data Sources Documentation

## Overview

The AQF311 Air Quality Forecasting project now uses a **clean data approach** with only external APIs and internal system features. All pattern-based synthetic modeling has been removed to ensure data authenticity and transparency.

## Clean Data Architecture

### Data Source Categories

1. **External APIs (Authenticated)**
   - OpenAQ: Real measured air quality data
   - NASA FIRMS: Real fire detection data
   - Open-Meteo: Weather-based air quality forecasts

2. **Internal System Features**
   - Temporal features: Date/time patterns
   - Geographic features: Location-based attributes
   - Holiday features: Major holidays only

3. **Excluded Sources (Removed)**
   - Pattern-based fire activity modeling
   - Synthetic WAQI fallback data
   - CAMS synthetic benchmarks
   - Extreme pollution scenarios
   - All pattern-based synthetic data

## Primary Data Sources

### 1. OpenAQ (Ground Truth Data)
- **Purpose**: Real measured air pollutant concentrations
- **API**: OpenAQ v3 (`https://api.openaq.org/v3`)
- **Authentication**: X-API-Key header (stored in `.config/api_keys.json`)
- **Data Type**: PM2.5, PM10, NO2, O3, CO, SO2 measurements
- **Coverage**: Global monitoring stations within 100km of cities
- **Update Frequency**: Real-time from ground monitoring stations
- **Quality**: Highest - authentic measured data from calibrated instruments

**API Usage:**
```python
# Find nearest stations
stations_url = "https://api.openaq.org/v3/locations"
stations_params = {
    "coordinates": f"{lat},{lon}",
    "radius": 100000,  # 100km
    "limit": 5
}

# Get measurements
measurements_url = "https://api.openaq.org/v3/measurements"
measurements_params = {
    "location_id": station_id,
    "date_from": "2025-09-04",
    "date_to": "2025-09-11",
    "limit": 100
}
```

### 2. NASA FIRMS (Fire Data)
- **Purpose**: Real fire detection and hotspot data
- **API**: NASA FIRMS (`https://firms.modaps.eosdis.nasa.gov/api/area/csv`)
- **Authentication**: MAP_KEY parameter (free registration required)
- **Data Type**: Fire hotspots from VIIRS satellite data
- **Coverage**: Global fire detection within 100km of cities
- **Update Frequency**: Near real-time satellite observations
- **Quality**: High - satellite-detected fire hotspots with confidence ratings

**API Usage:**
```python
firms_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
firms_params = {
    "MAP_KEY": nasa_firms_key,
    "source": "VIIRS_SNPP_NRT",
    "area": f"{west},{south},{east},{north}",
    "dayRange": 7,
    "date": "2025-09-11"
}
```

**Fire Data Processing:**
- Distance calculation from city centers
- Confidence level filtering (>70% for high confidence)
- PM2.5 impact assessment
- Fire risk level classification

### 3. Open-Meteo (Forecast Data)
- **Purpose**: Weather-based air quality predictions
- **API**: Open-Meteo Air Quality (`https://api.open-meteo.com/v1/air-quality`)
- **Authentication**: None required (free access)
- **Data Type**: Physics-based air quality forecasts
- **Coverage**: Global coverage for all cities
- **Update Frequency**: Hourly forecasts updated 4x daily
- **Quality**: High - based on numerical weather prediction models

**API Usage:**
```python
forecast_url = "https://api.open-meteo.com/v1/air-quality"
forecast_params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
               "sulphur_dioxide", "ozone", "european_aqi"],
    "past_days": 7,
    "forecast_days": 7
}
```

## Internal System Features

### 4. Temporal Features
- **Source**: System-generated from timestamps
- **Implementation**: Built-in date/time processing
- **Features Generated**:
  - Month (1-12)
  - Day of year (1-365/366)
  - Weekday (0-6)
  - Weekend flag (boolean)
  - Season (winter/spring/summer/autumn)
  - Quarter (Q1-Q4)

### 5. Geographic Features
- **Source**: System-generated from coordinates
- **Implementation**: Built-in geographic processing
- **Features Generated**:
  - Latitude/longitude coordinates
  - Continent classification
  - Hemisphere (Northern/Southern)
  - Climate zone (tropical/subtropical/temperate/polar)
  - Coastal proximity (coastal/near_coastal/inland)

### 6. Holiday Features (Simplified)
- **Source**: System-generated basic holiday calendar
- **Implementation**: Major holidays only (no pattern-based modeling)
- **Features Generated**:
  - New Year flag (January 1)
  - Christmas flag (December 25)
  - Holiday season flag (December 20 - January 5)

## API Key Management

### Secure Storage
- **Location**: `C:\aqf311\Git_repo\.config\api_keys.json`
- **Security**: Excluded from Git repository (in `.gitignore`)
- **Format**: JSON structure for multiple APIs

### API Key Structure
```json
{
  "apis": {
    "openaq": {
      "key": "your_openaq_api_key",
      "provider": "OpenAQ",
      "api_version": "v3"
    },
    "nasa_firms": {
      "key": "your_nasa_firms_map_key",
      "provider": "NASA FIRMS",
      "api_version": "4.0"
    }
  }
}
```

### Registration Requirements
1. **OpenAQ API Key**: Already obtained and stored
2. **NASA FIRMS MAP_KEY**: Registration at https://firms.modaps.eosdis.nasa.gov/api/map_key/
   - Free registration with email
   - Limit: 5,000 transactions per 10-minute interval

## Data Quality Levels

### Authenticity Hierarchy
1. **Real Measured Data**: OpenAQ ground monitoring stations
2. **Real Satellite Data**: NASA FIRMS fire detection
3. **Physics-Based Forecasts**: Open-Meteo numerical models
4. **System Features**: Internal temporal/geographic calculations

### Quality Assurance
- **No synthetic modeling**: All pattern-based generation removed
- **API validation**: Authentication and error handling
- **Data transparency**: Clear source documentation for every feature
- **Completeness tracking**: Success rates monitored per API

## Dataset Structure

### Clean Dataset Format
```json
{
  "generation_timestamp": "2025-09-11T...",
  "dataset_type": "CLEAN_100_CITY_DATASET",
  "data_sources": {
    "ground_truth": "OpenAQ Real Measured Data",
    "forecasts": "Open-Meteo Weather-Based Predictions",
    "fire_data": "NASA FIRMS Real Fire Detection",
    "internal_features": "System-generated features only"
  },
  "excluded_sources": [
    "Pattern-based fire modeling",
    "Synthetic WAQI fallback data",
    "CAMS synthetic benchmarks",
    "All pattern-based synthetic data"
  ],
  "cities_data": [...]
}
```

### City Data Structure
```json
{
  "city_metadata": {
    "name": "Delhi",
    "country": "IN",
    "continent": "Asia",
    "coordinates": {"lat": 28.6139, "lon": 77.2090}
  },
  "ground_truth": {
    "status": "success|no_data|error",
    "data": {...}
  },
  "forecasts": {
    "status": "success|no_data|error",
    "data": {...}
  },
  "fire_data": {
    "status": "success|no_fires|error",
    "data": {...}
  },
  "internal_features": {
    "temporal": {...},
    "geographic": {...},
    "holidays": {...}
  },
  "data_completeness": "complete|partial_with_fire|partial|incomplete"
}
```

## Implementation

### Clean Dataset Generator
- **Script**: `stage_5/scripts/clean_100_city_dataset_generator.py`
- **Purpose**: Generate datasets using only external APIs and internal features
- **Output**: JSON files with complete data provenance
- **Logging**: Comprehensive progress and error tracking

### Usage
```bash
cd stage_5/scripts
python clean_100_city_dataset_generator.py
```

## Migration from Previous Approach

### Removed Components
- ❌ Pattern-based fire activity synthesis
- ❌ WAQI fallback synthetic data
- ❌ CAMS synthetic benchmark modeling
- ❌ Extreme pollution scenario generation
- ❌ All algorithmic air quality synthesis

### Retained Components
- ✅ OpenAQ real measured data (enhanced)
- ✅ Open-Meteo physics-based forecasts
- ✅ Internal temporal/geographic features
- ✅ Basic holiday indicators

### New Additions
- ✅ NASA FIRMS real fire detection data
- ✅ Fire impact assessment algorithms
- ✅ Enhanced API error handling
- ✅ Comprehensive data source documentation

## Future Enhancements

### Additional Real Data APIs
- Weather station APIs for meteorological data
- Government air quality monitoring APIs
- Satellite-based air quality products
- Traffic and emission monitoring systems

### Data Source Expansion
- Regional air quality networks
- Industrial emission monitoring
- Agricultural activity tracking
- Transportation density data

## Documentation Updates

**Last Updated**: September 11, 2025
**Status**: Clean data architecture implemented
**API Keys**: OpenAQ configured, NASA FIRMS pending registration
**Success Rate**: TBD (will be measured during first collection run)

---

This clean data approach ensures full transparency, removes all synthetic modeling, and relies only on authentic external data sources combined with basic internal system features.
