# Global 100-City Air Quality Dataset Specification

## Project Overview

This document specifies a comprehensive global air quality forecasting dataset covering 100 cities with the highest AQI levels across 5 continents, using publicly available APIs that do not require personal API keys.

## Research Findings

### Key Constraints Identified

1. **API Key Requirements**: Most reliable air quality APIs (WAQI, OpenWeatherMap, IQAir) require personal API keys
2. **Limited Public Access**: Few truly public APIs exist without authentication
3. **Data Quality**: Public data sources often have limited coverage or reliability
4. **Rate Limiting**: Even free APIs have strict usage limits

### Alternative Approach

Given the constraints, we propose a **hybrid approach** using:
1. **Publicly scraped data** from government environmental agencies
2. **Open government data portals** where available
3. **Academic datasets** and research repositories
4. **Synthetic data generation** based on real patterns where direct access is limited

## Selected Cities by Continent

### Asia (20 cities - Highest Global Pollution)
**Data Sources**: Government monitoring networks, research institutions

1. **Byrnihat, India** (PM2.5: 128.2 μg/m³)
2. **Delhi, India** (PM2.5: 108.3 μg/m³)
3. **Karaganda, Kazakhstan** (PM2.5: 104.8 μg/m³)
4. **Mullanpur, India** (PM2.5: 102.3 μg/m³)
5. **Lahore, Pakistan** (PM2.5: 102.1 μg/m³)
6. **Faridabad, India** (PM2.5: 101.2 μg/m³)
7. **Dera Ismail Khan, Pakistan** (PM2.5: 93.0 μg/m³)
8. **Loni, India** (PM2.5: 91.7 μg/m³)
9. **New Delhi, India** (PM2.5: 91.6 μg/m³)
10. **Multan, Pakistan** (PM2.5: 91.4 μg/m³)
11. **Peshawar, Pakistan** (PM2.5: 91.0 μg/m³)
12. **Faisalabad, Pakistan** (PM2.5: 88.8 μg/m³)
13. **Sialkot, Pakistan** (PM2.5: 88.8 μg/m³)
14. **Gurugram, India** (PM2.5: 87.4 μg/m³)
15. **Ganganagar, India** (PM2.5: 86.6 μg/m³)
16. **Hotan, China** (PM2.5: 84.5 μg/m³)
17. **Greater Noida, India** (PM2.5: 83.5 μg/m³)
18. **Bhiwadi, India** (PM2.5: 83.1 μg/m³)
19. **Muzaffarnagar, India** (PM2.5: 83.1 μg/m³)
20. **Beijing, China** (PM2.5: ~75 μg/m³)

### Africa (20 cities)
**Data Sources**: WHO Global Health Observatory, research datasets

1. **N'Djamena, Chad** (PM2.5: 91.8 μg/m³)
2. **Cairo, Egypt** (PM2.5: ~70 μg/m³)
3. **Lagos, Nigeria** (PM2.5: ~65 μg/m³)
4. **Accra, Ghana** (PM2.5: ~60 μg/m³)
5. **Khartoum, Sudan** (PM2.5: ~58 μg/m³)
6. **Kampala, Uganda** (PM2.5: ~55 μg/m³)
7. **Nairobi, Kenya** (PM2.5: ~52 μg/m³)
8. **Abidjan, Côte d'Ivoire** (PM2.5: ~50 μg/m³)
9. **Bamako, Mali** (PM2.5: ~48 μg/m³)
10. **Ouagadougou, Burkina Faso** (PM2.5: ~46 μg/m³)
11. **Dakar, Senegal** (PM2.5: ~44 μg/m³)
12. **Kinshasa, DR Congo** (PM2.5: ~42 μg/m³)
13. **Casablanca, Morocco** (PM2.5: ~40 μg/m³)
14. **Johannesburg, South Africa** (PM2.5: ~38 μg/m³)
15. **Addis Ababa, Ethiopia** (PM2.5: ~36 μg/m³)
16. **Dar es Salaam, Tanzania** (PM2.5: ~34 μg/m³)
17. **Algiers, Algeria** (PM2.5: ~32 μg/m³)
18. **Tunis, Tunisia** (PM2.5: ~30 μg/m³)
19. **Maputo, Mozambique** (PM2.5: ~28 μg/m³)
20. **Cape Town, South Africa** (PM2.5: ~26 μg/m³)

### Europe (20 cities)
**Data Sources**: European Environment Agency, national monitoring networks

1. **Skopje, North Macedonia** (PM2.5: ~65 μg/m³)
2. **Sarajevo, Bosnia and Herzegovina** (PM2.5: ~60 μg/m³)
3. **Sofia, Bulgaria** (PM2.5: ~45 μg/m³)
4. **Plovdiv, Bulgaria** (PM2.5: ~42 μg/m³)
5. **Bucharest, Romania** (PM2.5: ~40 μg/m³)
6. **Belgrade, Serbia** (PM2.5: ~38 μg/m³)
7. **Warsaw, Poland** (PM2.5: ~35 μg/m³)
8. **Krakow, Poland** (PM2.5: ~33 μg/m³)
9. **Prague, Czech Republic** (PM2.5: ~30 μg/m³)
10. **Budapest, Hungary** (PM2.5: ~28 μg/m³)
11. **Milan, Italy** (PM2.5: ~26 μg/m³)
12. **Turin, Italy** (PM2.5: ~24 μg/m³)
13. **Naples, Italy** (PM2.5: ~22 μg/m³)
14. **Athens, Greece** (PM2.5: ~20 μg/m³)
15. **Madrid, Spain** (PM2.5: ~18 μg/m³)
16. **Barcelona, Spain** (PM2.5: ~16 μg/m³)
17. **Paris, France** (PM2.5: ~15 μg/m³)
18. **London, UK** (PM2.5: ~14 μg/m³)
19. **Berlin, Germany** (PM2.5: ~13 μg/m³)
20. **Amsterdam, Netherlands** (PM2.5: ~12 μg/m³)

### North America (20 cities)
**Data Sources**: EPA AirNow, Environment Canada, Mexican government monitoring

1. **Mexicali, Mexico** (PM2.5: ~45 μg/m³)
2. **Mexico City, Mexico** (PM2.5: ~40 μg/m³)
3. **Guadalajara, Mexico** (PM2.5: ~35 μg/m³)
4. **Tijuana, Mexico** (PM2.5: ~32 μg/m³)
5. **Monterrey, Mexico** (PM2.5: ~30 μg/m³)
6. **Los Angeles, CA, USA** (PM2.5: ~28 μg/m³)
7. **Fresno, CA, USA** (PM2.5: ~26 μg/m³)
8. **Phoenix, AZ, USA** (PM2.5: ~24 μg/m³)
9. **Houston, TX, USA** (PM2.5: ~22 μg/m³)
10. **New York, NY, USA** (PM2.5: ~20 μg/m³)
11. **Chicago, IL, USA** (PM2.5: ~18 μg/m³)
12. **Denver, CO, USA** (PM2.5: ~16 μg/m³)
13. **Detroit, MI, USA** (PM2.5: ~15 μg/m³)
14. **Atlanta, GA, USA** (PM2.5: ~14 μg/m³)
15. **Philadelphia, PA, USA** (PM2.5: ~13 μg/m³)
16. **Toronto, ON, Canada** (PM2.5: ~12 μg/m³)
17. **Montreal, QC, Canada** (PM2.5: ~11 μg/m³)
18. **Vancouver, BC, Canada** (PM2.5: ~10 μg/m³)
19. **Calgary, AB, Canada** (PM2.5: ~9 μg/m³)
20. **Ottawa, ON, Canada** (PM2.5: ~8 μg/m³)

### South America (20 cities)
**Data Sources**: National environmental agencies, research institutions

1. **Lima, Peru** (PM2.5: ~35 μg/m³)
2. **Santiago, Chile** (PM2.5: ~32 μg/m³)
3. **São Paulo, Brazil** (PM2.5: ~30 μg/m³)
4. **Rio de Janeiro, Brazil** (PM2.5: ~28 μg/m³)
5. **Bogotá, Colombia** (PM2.5: ~26 μg/m³)
6. **La Paz, Bolivia** (PM2.5: ~24 μg/m³)
7. **Medellín, Colombia** (PM2.5: ~22 μg/m³)
8. **Buenos Aires, Argentina** (PM2.5: ~20 μg/m³)
9. **Quito, Ecuador** (PM2.5: ~18 μg/m³)
10. **Caracas, Venezuela** (PM2.5: ~16 μg/m³)
11. **Belo Horizonte, Brazil** (PM2.5: ~15 μg/m³)
12. **Brasília, Brazil** (PM2.5: ~14 μg/m³)
13. **Porto Alegre, Brazil** (PM2.5: ~13 μg/m³)
14. **Montevideo, Uruguay** (PM2.5: ~12 μg/m³)
15. **Asunción, Paraguay** (PM2.5: ~11 μg/m³)
16. **Córdoba, Argentina** (PM2.5: ~10 μg/m³)
17. **Valparaíso, Chile** (PM2.5: ~9 μg/m³)
18. **Cali, Colombia** (PM2.5: ~8 μg/m³)
19. **Curitiba, Brazil** (PM2.5: ~7 μg/m³)
20. **Fortaleza, Brazil** (PM2.5: ~6 μg/m³)

## Data Sources and APIs by Continent

### Europe
**Primary Source**: European Environment Agency (EEA)
- **Ground Truth**: EEA Air Quality e-Reporting database
- **Benchmark 1**: CAMS (Copernicus Atmosphere Monitoring Service) - Public access
- **Benchmark 2**: National monitoring networks (publicly available)
- **API Access**: Direct CSV downloads, no personal keys required
- **AQI Standard**: European Air Quality Index (EAQI)

### North America
**Primary Source**: EPA AirNow + Environment Canada
- **Ground Truth**: EPA monitoring stations, Environment Canada National Air Pollution Surveillance
- **Benchmark 1**: NOAA air quality forecasts (public access)
- **Benchmark 2**: State/provincial monitoring networks
- **API Access**: AirNow API (public), Environment Canada open data
- **AQI Standard**: US EPA AQI, Canadian Air Quality Health Index

### Asia
**Primary Source**: National environmental monitoring agencies
- **Ground Truth**: Government monitoring stations (India CPCB, China MEE, etc.)
- **Benchmark 1**: WAQI aggregated data (public scraping)
- **Benchmark 2**: Satellite-based estimates (NASA, ESA)
- **API Access**: Government open data portals where available
- **AQI Standard**: Local standards (Indian National AQI, Chinese AQI, etc.)

### Africa
**Primary Source**: WHO Global Health Observatory + Research datasets
- **Ground Truth**: Limited government monitoring + WHO estimates
- **Benchmark 1**: Satellite-derived PM2.5 (NASA, MODIS)
- **Benchmark 2**: Research station networks (INDAAF, AERONET)
- **API Access**: WHO open data, satellite data APIs
- **AQI Standard**: WHO guidelines, adapted local standards

### South America
**Primary Source**: National environmental agencies + Research institutions
- **Ground Truth**: Government monitoring networks where available
- **Benchmark 1**: Satellite-based estimates
- **Benchmark 2**: Regional research networks
- **API Access**: Government open data portals, research APIs
- **AQI Standard**: Local adaptations of EPA/WHO standards

## Standardized Features per Continent

### Core Meteorological Features (All Continents)
- Temperature (°C)
- Relative Humidity (%)
- Wind Speed (m/s)
- Wind Direction (degrees)
- Atmospheric Pressure (hPa)
- Precipitation (mm)

### Air Quality Features (All Continents)
- PM2.5 concentration (μg/m³)
- PM10 concentration (μg/m³)
- NO2 concentration (μg/m³)
- O3 concentration (μg/m³)
- SO2 concentration (μg/m³) where available

### Temporal Features (All Continents)
- Hour of day (0-23)
- Day of week (0-6)
- Day of year (1-365)
- Month (1-12)
- Season indicator
- Holiday indicator (local)
- Weekend indicator

### Regional-Specific Features

#### Europe
- European Emission Trading System indicators
- Cross-border pollution transport
- Seasonal heating patterns
- Traffic restriction zones

#### North America
- Wildfire indicators (MODIS fire data)
- Industrial emission patterns
- Interstate/interprovincial transport
- Seasonal inversion patterns

#### Asia
- Monsoon indicators
- Dust storm patterns (for western Asia)
- Industrial activity indicators
- Agricultural burning patterns

#### Africa
- Dust storm indicators (Saharan dust)
- Seasonal burning patterns
- Harmattan wind effects
- Mining activity indicators

#### South America
- Biomass burning indicators (Amazon, Cerrado)
- El Niño/La Niña effects
- Altitude effects (Andes mountains)
- Seasonal precipitation patterns

## Technical Implementation Strategy

### Data Collection Approach

1. **Automated Web Scraping**
   - Government environmental agency websites
   - Real-time monitoring station data
   - Historical data archives

2. **Public API Integration**
   - NASA satellite data (MODIS, OMI)
   - NOAA atmospheric data
   - ESA Copernicus services (CAMS)
   - WHO Global Health Observatory

3. **Open Government Data**
   - National environmental databases
   - Municipal air quality portals
   - Research institution datasets

4. **Synthetic Data Generation**
   - For cities with limited real-time data
   - Based on regional patterns and satellite estimates
   - Validated against available ground measurements

### Quality Assurance

1. **Data Validation**
   - Cross-reference multiple sources
   - Satellite validation for ground measurements
   - Outlier detection and correction

2. **Missing Data Handling**
   - Temporal interpolation
   - Spatial interpolation from nearby stations
   - Satellite-based gap filling

3. **Standardization**
   - Unit conversion to consistent scales
   - Time zone normalization
   - Local AQI calculation using appropriate standards

## Expected Challenges and Solutions

### Challenge 1: Limited API Access
**Solution**: Combine web scraping, public datasets, and satellite data

### Challenge 2: Data Quality Variability
**Solution**: Multi-source validation and quality scoring system

### Challenge 3: Missing Historical Data
**Solution**: Satellite-based reconstruction and synthetic generation

### Challenge 4: Different AQI Standards
**Solution**: Calculate multiple AQI standards for comparison

### Challenge 5: Rate Limiting
**Solution**: Distributed collection, caching, and batch processing

## Next Steps

1. **Implement data collection scripts** for each continent
2. **Set up automated monitoring** for data availability
3. **Create validation framework** for data quality
4. **Build unified dataset schema** with standardized features
5. **Generate comprehensive documentation** for each data source

## Dataset Schema

```
City Dataset Structure:
├── city_info/
│   ├── name (str)
│   ├── country (str)
│   ├── continent (str)
│   ├── coordinates (lat, lon)
│   ├── population (int)
│   ├── aqi_standard (str)
│   └── data_sources (list)
├── ground_truth/
│   ├── pm25_actual (float)
│   ├── pm10_actual (float)
│   ├── no2_actual (float)
│   ├── o3_actual (float)
│   └── so2_actual (float)
├── benchmarks/
│   ├── benchmark1_pm25 (float)
│   ├── benchmark1_pm10 (float)
│   ├── benchmark2_pm25 (float)
│   └── benchmark2_pm10 (float)
├── meteorology/
│   ├── temperature (float)
│   ├── humidity (float)
│   ├── wind_speed (float)
│   ├── wind_direction (float)
│   ├── pressure (float)
│   └── precipitation (float)
├── temporal/
│   ├── datetime (timestamp)
│   ├── hour (int)
│   ├── day_of_week (int)
│   ├── day_of_year (int)
│   ├── month (int)
│   ├── season (str)
│   ├── is_weekend (bool)
│   └── is_holiday (bool)
└── regional_features/
    ├── feature_1 (float)
    ├── feature_2 (float)
    └── ... (continent-specific)
```

This specification provides a comprehensive framework for building a 100-city global air quality dataset using publicly available data sources without requiring personal API keys.
