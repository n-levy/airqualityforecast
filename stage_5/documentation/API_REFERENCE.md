# API Reference and Usage Guide

## Data Loading

### Python (pandas)
```python
import pandas as pd

# Load main dataset
air_quality = pd.read_parquet('air_quality_data.parquet')
weather = pd.read_parquet('meteorological_data.parquet')
features = pd.read_parquet('temporal_features.parquet')
spatial = pd.read_parquet('spatial_features.parquet')
forecasts = pd.read_parquet('forecast_data.parquet')

# Merge datasets
full_data = air_quality.merge(weather, on=['city', 'date'])
full_data = full_data.merge(features, on=['city', 'date'])
```

### Python (PyArrow)
```python
import pyarrow.parquet as pq

# Load with PyArrow for better performance
table = pq.read_table('air_quality_data.parquet')
df = table.to_pandas()

# Filter while reading
filtered = pq.read_table(
    'air_quality_data.parquet',
    filters=[('city', '=', 'Berlin')]
)
```

### R (arrow)
```r
library(arrow)
library(dplyr)

# Load data
air_quality <- read_parquet('air_quality_data.parquet')
weather <- read_parquet('meteorological_data.parquet')

# Join datasets
full_data <- air_quality %>%
  left_join(weather, by = c('city', 'date'))
```

### Spark (PySpark)
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AirQuality").getOrCreate()

# Load data
df = spark.read.parquet('air_quality_data.parquet')
df.show()

# Query data
df.filter(df.city == 'Berlin').select('date', 'PM25', 'AQI').show()
```

## Common Queries

### Time Series Analysis
```python
# Get time series for specific city
city_data = air_quality[air_quality['city'] == 'Berlin'].copy()
city_data['date'] = pd.to_datetime(city_data['date'])
city_data.set_index('date', inplace=True)

# Calculate monthly averages
monthly_avg = city_data.resample('M')['PM2.5'].mean()
```

### Multi-City Comparison
```python
# Compare cities
cities = ['Berlin', 'Delhi', 'São Paulo', 'Toronto', 'Cairo']
comparison = air_quality[air_quality['city'].isin(cities)]
comparison.groupby('city')['AQI'].describe()
```

### Seasonal Analysis
```python
# Load temporal features
temporal = pd.read_parquet('temporal_features.parquet')
merged = air_quality.merge(temporal, on=['city', 'date'])

# Seasonal analysis
seasonal_avg = merged.groupby(['city', 'season'])['PM2.5'].mean().unstack()
```

## Data Schema

### Air Quality Data Schema
```python
# Expected schema
{
    'city': 'string',
    'date': 'date32[day]',
    'PM2.5': 'double',
    'PM10': 'double',
    'NO2': 'double',
    'O3': 'double',
    'SO2': 'double',
    'CO': 'double',
    'AQI': 'int32',
    'AQI_category': 'string',
    'AQI_standard': 'string'
}
```

## Performance Tips

### Memory Optimization
```python
# Read specific columns only
columns = ['city', 'date', 'PM2.5', 'AQI']
df = pd.read_parquet('air_quality_data.parquet', columns=columns)

# Use categorical data types for cities
df['city'] = df['city'].astype('category')
```

### Query Optimization
```python
# Use PyArrow for filtering large datasets
import pyarrow.compute as pc

table = pq.read_table('air_quality_data.parquet')
filtered = table.filter(
    pc.and_(
        pc.equal(table['city'], 'Berlin'),
        pc.greater(table['PM2.5'], 35)
    )
)
```

## Error Handling

### Common Issues
```python
# Handle missing data
df = pd.read_parquet('air_quality_data.parquet')
print(f"Missing data: {df.isnull().sum()}")

# Handle date parsing
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Validate data ranges
assert df['PM2.5'].min() >= 0, "Negative PM2.5 values found"
assert df['AQI'].max() <= 500, "AQI values exceed maximum"
```

## Integration Examples

### Machine Learning Pipeline
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Prepare features
features = ['temperature', 'humidity', 'pressure', 'wind_speed']
X = merged[features]
y = merged['PM2.5']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Time series plot
plt.figure(figsize=(12, 6))
city_data['PM2.5'].plot()
plt.title('PM2.5 Time Series - Berlin')
plt.ylabel('PM2.5 (μg/m³)')
plt.show()

# Correlation heatmap
correlation = merged[['PM2.5', 'temperature', 'humidity', 'pressure']].corr()
sns.heatmap(correlation, annot=True)
```

## Batch Processing

### Process Multiple Cities
```python
def process_city(city_name):
    city_data = air_quality[air_quality['city'] == city_name]
    # Your processing logic here
    return city_data.describe()

# Process all cities
results = {}
for city in air_quality['city'].unique():
    results[city] = process_city(city)
```

## Support

For technical support or questions about the API, please refer to the documentation or create an issue in the project repository.
