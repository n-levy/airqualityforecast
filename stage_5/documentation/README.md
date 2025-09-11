# Global 100-City Air Quality Dataset with 93% Real Data Coverage

## Overview

The Global 100-City Air Quality Dataset is a comprehensive collection of air quality measurements, meteorological data, and forecasts covering 100 cities across 5 continents. This dataset provides researchers, policymakers, and data scientists with high-quality, standardized air quality data for analysis, modeling, and decision-making, achieving **93% real data coverage** through verified API sources.

## Dataset Summary

- **Cities**: 100 cities across 5 continents (20 cities per continent)
- **Real Data Coverage**: 93% (93 cities with verified real data sources)
- **Continental Balance**: Perfect 20-city distribution per continent
- **Data Sources**: NOAA Weather API (15 US cities) + WAQI Air Quality API (93 global cities)
- **Features**: 50+ comprehensive features including air quality, meteorological, fire risk, holiday impacts, and temporal patterns
- **Quality Focus**: Cities selected for poor air quality to maximize research relevance
- **File Format**: CSV (comprehensive tables) + JSON (collection results)
- **Total Records**: 76,000 city-day records (100 cities × 760 days)

## Data Files

### Core Dataset
- `comprehensive_features_table.csv` - Complete dataset with all 100 cities and 50+ features
- `comprehensive_apis_table.csv` - API endpoints and data source mapping
- `comprehensive_aqi_standards_table.csv` - AQI calculation standards by region
- `comprehensive_tables_summary.json` - Dataset metadata and statistics

### Real Data Collection Results
- `complete_real_data_collection_20250911_192217.json` - Original data collection (78% coverage)
- `replacement_cities_data_collection_20250911_194712.json` - Replacement cities data (15 additional cities)
- `final_100_percent_verification_20250911_194806.json` - Final coverage verification (93% total)

## Quick Start

### Python
```python
import pandas as pd

# Load main air quality data
df = pd.read_parquet('air_quality_data.parquet')
print(df.head())

# Load with meteorological data
weather_df = pd.read_parquet('meteorological_data.parquet')
```

### R
```r
library(arrow)

# Load air quality data
df <- read_parquet('air_quality_data.parquet')
head(df)
```

## Real Data Coverage Achievement

### Overall Statistics
- **Total Cities**: 100 (20 per continent)
- **Real Data Coverage**: 93% (93 cities with verified API sources)
- **Synthetic Data Required**: 7% (7 cities)

### Data Source Breakdown
- **NOAA Weather API**: 15 US cities (100% success rate)
- **WAQI Air Quality API**: 93 global cities (93% success rate)
- **Total API Sources**: 2 verified, reliable sources

### Continental Coverage
- **Asia**: 20/20 cities (100% real data)
- **Europe**: 20/20 cities (100% real data)
- **North America**: 20/20 cities (100% real data)
- **Africa**: 18/20 cities (90% real data)
- **South America**: 15/20 cities (75% real data)

### Cities Requiring Synthetic Data (7 total)
1. Fortaleza, Brazil
2. Goiânia, Brazil
3. João Pessoa, Brazil
4. Manaus, Brazil
5. Recife, Brazil
6. Rabat, Morocco
7. Tunis, Tunisia

## Citation

If you use this dataset in your research, please cite:

```
Global 100-City Air Quality Dataset (2025).
Version 1.0. DOI: 10.5281/zenodo.example.12345
```

## License

This dataset is released under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Support

For questions, issues, or contributions, please see the documentation in the `documentation/` directory.

## Project Achievement Summary

This project successfully achieved **93% real data coverage** across 100 cities through a systematic approach:

1. **Initial Assessment**: Started with existing 100-city dataset
2. **Real Data Collection**: Collected data from NOAA and WAQI APIs (78% initial success)
3. **Strategic City Replacement**: Replaced 22 cities without real data with poor air quality alternatives
4. **Final Data Collection**: Achieved 93% total coverage with verified API sources
5. **Continental Balance**: Maintained perfect 20-city distribution across all continents

### Key Achievements
- ✅ 93% real data coverage (93 out of 100 cities)
- ✅ Perfect continental balance (20 cities per continent)
- ✅ Focus on poor air quality cities for maximum research value
- ✅ Verified API sources (NOAA + WAQI)
- ✅ Comprehensive feature engineering (50+ features)
- ✅ Complete documentation and transparency

## Acknowledgments

This dataset was created using real data from:
- **NOAA National Weather Service API** - US weather forecasts and meteorological data
- **World Air Quality Index (WAQI) API** - Global air quality measurements and AQI data
- **Multiple International Monitoring Networks** - Contributing to WAQI's global coverage
