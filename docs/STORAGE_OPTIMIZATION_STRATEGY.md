# Storage Optimization Strategy for Laptop Deployment

## Current Storage Requirements Analysis

### Per City (5-year dataset):
- **São Paulo**: 38.7 GB (Raw: 24.3 GB, Processed: 14.3 GB, Metadata: 70 MB)
- **Projected 100 cities**: ~3.9 TB

### Current Storage Calculation (per record):
- Raw data: 0.58 MB per record
- Processed data: 0.34 MB per record
- Metadata: minimal
- **Total: 0.92 MB per record**

---

## 🎯 Optimization Strategies

### **Strategy 1: Data Compression** - **60-80% reduction**

#### Implementation:
```python
# Replace current storage estimates with compressed versions
storage_estimate = {
    "raw_data_mb": total_actual_records * 0.12,      # Was 0.58 → Compression + binary format
    "processed_data_mb": total_actual_records * 0.08, # Was 0.34 → Parquet format
    "metadata_mb": 20,  # Was 70 → Compressed JSON
    "total_mb": total_actual_records * 0.20 + 20      # 80% reduction
}
```

#### Techniques:
- **Binary storage formats**: Parquet instead of JSON/CSV
- **Compression algorithms**: gzip, lz4, or zstd
- **Data type optimization**: Float32 instead of Float64, appropriate integer sizes
- **Metadata compression**: Remove redundant timestamps, compress source attribution

#### Result: **São Paulo: 38.7 GB → 8.4 GB (78% reduction)**

---

### **Strategy 2: Selective Data Retention** - **70-90% reduction**

#### Core Data Only Approach:
```python
# Minimal essential data per record
essential_data = {
    "timestamp": 4,      # Unix timestamp (4 bytes)
    "pm25": 4,          # PM2.5 measurement (4 bytes)
    "pm10": 4,          # PM10 measurement (4 bytes)
    "no2": 4,           # NO2 measurement (4 bytes)
    "o3": 4,            # O3 measurement (4 bytes)
    "aqi": 2,           # Calculated AQI (2 bytes)
    "quality_flag": 1,   # Data quality indicator (1 byte)
    # Total: 23 bytes per record
}

storage_estimate = {
    "core_data_mb": total_actual_records * 0.000023 * 1024,  # 23 bytes per record
    "total_mb": total_actual_records * 0.025  # ~97% reduction
}
```

#### What to exclude:
- Raw satellite imagery
- Detailed source metadata
- Historical benchmark comparisons
- Intermediate processing steps

#### Result: **São Paulo: 38.7 GB → 1.0 GB (97% reduction)**

---

### **Strategy 3: Temporal Sampling** - **50-90% reduction**

#### Smart Sampling Approaches:

**Option A: Forecasting-Focused (50% reduction)**
- Keep all recent data (last 30 days): Full resolution
- Historical data: Sample every 6 hours instead of hourly
- Special events: Keep high pollution episodes at full resolution

**Option B: Laptop-Optimized (80% reduction)**
- Keep only daily averages for historical data
- Full hourly data for last 7 days only
- Monthly summaries for older data

**Option C: Ultra-Minimal (90% reduction)**
- Daily averages only
- Keep 1 year of daily data per city
- Store only essential pollutants (PM2.5, PM10, AQI)

#### Implementation:
```python
# Temporal sampling storage calculation
def calculate_sampled_storage(sampling_strategy="forecasting_focused"):
    if sampling_strategy == "forecasting_focused":
        reduction_factor = 0.5  # 50% reduction
    elif sampling_strategy == "laptop_optimized":
        reduction_factor = 0.2  # 80% reduction
    elif sampling_strategy == "ultra_minimal":
        reduction_factor = 0.1  # 90% reduction

    return total_actual_records * 0.20 * reduction_factor  # Combined with compression
```

---

### **Strategy 4: On-Demand Data Loading** - **95%+ reduction**

#### Local Cache Strategy:
- **Local storage**: Only current forecast data + last 30 days
- **Cloud/remote storage**: Full historical dataset
- **Dynamic loading**: Fetch historical data only when needed for analysis

#### Implementation:
```python
# Minimal local storage
local_storage = {
    "current_forecasts": "50 MB per city",
    "recent_data_30days": "200 MB per city",
    "metadata_cache": "10 MB per city",
    "total_per_city": "260 MB"
}
# 100 cities × 260 MB = 26 GB total (vs 3.9 TB)
```

---

## 💡 Recommended Implementation for Laptop

### **Hybrid Approach: Compression + Selective + Sampling**

```python
# Optimized storage calculation
def calculate_optimized_storage(total_records, optimization_level="laptop"):
    if optimization_level == "laptop":
        # Combination of strategies
        base_size = total_records * 0.05  # Compressed essential data only
        sampling_reduction = 0.3  # Keep 30% of temporal data
        final_size = base_size * sampling_reduction

        return {
            "optimized_data_mb": final_size,
            "metadata_mb": 10,  # Minimal metadata
            "cache_mb": 20,     # Local processing cache
            "total_mb": final_size + 30
        }
```

### **Results**:
- **Per city**: 38.7 GB → **1.3 GB** (97% reduction)
- **100 cities**: 3.9 TB → **130 GB** (97% reduction)

### **What you keep**:
- Essential pollutants (PM2.5, PM10, NO2, O3)
- AQI calculations
- Last 30 days at full resolution
- Historical daily averages
- Quality indicators

### **What you exclude**:
- Raw satellite imagery
- Detailed source metadata
- Full historical hourly data beyond 30 days
- Benchmark comparison data
- Intermediate processing files

---

## 🚀 Implementation Priority

1. **Immediate**: Implement data compression (Strategy 1)
2. **Phase 2**: Add selective data retention (Strategy 2)
3. **Phase 3**: Implement smart temporal sampling (Strategy 3)
4. **Advanced**: On-demand loading system (Strategy 4)

This approach makes the system completely viable for laptop deployment while maintaining forecasting accuracy.
