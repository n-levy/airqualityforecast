# Data Dictionary

## Air Quality Data (`air_quality_data.parquet`)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| city | string | - | City name |
| date | date | - | Measurement date (YYYY-MM-DD) |
| PM2.5 | float | μg/m³ | Fine particulate matter (≤2.5 micrometers) |
| PM10 | float | μg/m³ | Particulate matter (≤10 micrometers) |
| NO2 | float | μg/m³ | Nitrogen dioxide |
| O3 | float | μg/m³ | Ozone |
| SO2 | float | μg/m³ | Sulfur dioxide |
| CO | float | mg/m³ | Carbon monoxide |
| AQI | integer | - | Air Quality Index value |
| AQI_category | string | - | AQI category (Good, Moderate, Unhealthy, etc.) |
| AQI_standard | string | - | AQI standard used (EPA, EAQI, etc.) |

## Meteorological Data (`meteorological_data.parquet`)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| city | string | - | City name |
| date | date | - | Measurement date |
| temperature | float | °C | Daily average temperature |
| humidity | float | % | Relative humidity |
| pressure | float | hPa | Atmospheric pressure |
| wind_speed | float | m/s | Wind speed |
| wind_direction | float | degrees | Wind direction (0-360°) |
| precipitation | float | mm | Daily precipitation |
| cloud_cover | float | % | Cloud coverage |
| visibility | float | km | Atmospheric visibility |

## Temporal Features (`temporal_features.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| city | string | City name |
| date | date | Date |
| hour_of_day | integer | Hour (0-23) |
| day_of_week | integer | Day of week (0=Monday, 6=Sunday) |
| month | integer | Month (1-12) |
| season | string | Season (Spring, Summer, Fall, Winter) |
| is_weekend | boolean | Weekend indicator |
| is_holiday | boolean | Holiday indicator |
| day_of_year | integer | Day of year (1-365/366) |

## Spatial Features (`spatial_features.parquet`)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| city | string | - | City name |
| latitude | float | degrees | Geographic latitude |
| longitude | float | degrees | Geographic longitude |
| elevation | float | meters | Elevation above sea level |
| population_density | float | people/km² | Population density |
| urban_area_index | float | - | Urbanization index (0-1) |
| distance_to_coast | float | km | Distance to nearest coast |

## Forecast Data (`forecast_data.parquet`)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| city | string | - | City name |
| date | date | - | Forecast date |
| forecast_horizon | string | - | Forecast horizon (1h, 6h, 24h, 48h) |
| forecast_PM2.5 | float | μg/m³ | Forecasted PM2.5 |
| forecast_PM10 | float | μg/m³ | Forecasted PM10 |
| forecast_NO2 | float | μg/m³ | Forecasted NO2 |
| forecast_O3 | float | μg/m³ | Forecasted O3 |
| forecast_source | string | - | Forecast data source |

## Missing Values

Missing values are represented as `null` in Parquet files. The dataset has been cleaned with < 2% missing values overall.

## Data Types

- **Dates**: ISO 8601 format (YYYY-MM-DD)
- **Floating point**: 64-bit precision
- **Strings**: UTF-8 encoded
- **Booleans**: True/False values
