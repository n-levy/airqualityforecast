# Global 100-City Air Quality Dataset

## Overview

The Global 100-City Air Quality Dataset is a comprehensive collection of air quality measurements, meteorological data, and forecasts covering 92 cities across 5 continents over a 5-year period (2020-2025). This dataset provides researchers, policymakers, and data scientists with high-quality, standardized air quality data for analysis, modeling, and decision-making.

## Dataset Summary

- **Cities**: 92 cities across 5 continents
- **Time Period**: September 12, 2020 - September 11, 2025 (5 years)
- **Records**: 251,343 validated daily measurements
- **Features**: 215+ engineered features across 6 categories
- **AQI Standards**: 7 regional standards implemented
- **File Format**: Apache Parquet (optimized for analysis)
- **Size**: 0.09 GB (raw), 0.03 GB (compressed)

## Data Files

- `air_quality_data.parquet` - Core air quality measurements (PM2.5, PM10, NO2, O3, SO2, CO, AQI)
- `meteorological_data.parquet` - Weather and meteorological features
- `temporal_features.parquet` - Engineered temporal features (seasonality, trends)
- `spatial_features.parquet` - Geographic and spatial characteristics
- `forecast_data.parquet` - Integrated forecast data from multiple sources
- `dataset_metadata.json` - Comprehensive dataset metadata

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

## Data Quality

- **Overall Quality Score**: 88.7%
- **Data Completeness**: 98.6%
- **Validation Status**: Passed all critical tests
- **Missing Data**: < 2%
- **Duplicate Records**: < 0.1%

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

## Acknowledgments

This dataset was created using data from multiple sources including EPA AirNow, Environment Canada, European Environment Agency, NASA satellite data, and various national monitoring networks.
