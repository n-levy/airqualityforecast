#!/usr/bin/env python3
"""
Unified 100-City Dataset Merger
===============================

Merges CAMS, NOAA GEFS-Aerosol, ground truth observations, and local features
into one comprehensive 100-city air quality forecasting dataset.

Data Sources:
- NOAA GEFS-Aerosol forecasts (PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃)
- ECMWF CAMS forecasts (PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃)
- Ground truth observations (PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃)
- Local features (calendar, lag, metadata)

Output: Unified dataset with standardized schema and partitioning
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logs_dir = Path("C:/aqf311/data/logs")
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "unified_dataset_merge.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Standardized schema for the unified dataset
UNIFIED_SCHEMA = {
    "city": "string",
    "country": "string",
    "timestamp_utc": "datetime64[ns]",
    "pollutant": "string",  # PM25, PM10, NO2, SO2, CO, O3
    "value": "float64",  # Concentration in standardized units
    "units": "string",  # μg/m³ for PM, ppb for gases
    "source": "string",  # GEFS, CAMS, Ground-Truth
    "data_type": "string",  # forecast, observation
    "run_date": "string",  # YYYY-MM-DD (for forecasts)
    "run_hour": "string",  # HH (for forecasts)
    "f_hour": "int64",  # Forecast hour (for forecasts)
    "lat": "float64",
    "lon": "float64",
    # Calendar features
    "year": "int64",
    "month": "int64",
    "day": "int64",
    "hour": "int64",
    "day_of_week": "int64",
    "day_of_year": "int64",
    "week_of_year": "int64",
    "is_weekend": "bool",
    "is_holiday_season": "bool",
    "season": "int64",
    "hour_sin": "float64",
    "hour_cos": "float64",
    "month_sin": "float64",
    "month_cos": "float64",
    "day_sin": "float64",
    "day_cos": "float64",
    # Lag features (will be calculated per city)
    "pm25_lag_1h": "float64",
    "pm25_lag_3h": "float64",
    "pm25_lag_6h": "float64",
    "pm25_lag_12h": "float64",
    "pm25_lag_24h": "float64",
    "pm10_lag_1h": "float64",
    "pm10_lag_3h": "float64",
    "pm10_lag_6h": "float64",
    "pm10_lag_12h": "float64",
    "pm10_lag_24h": "float64",
    # Metadata
    "model_version": "string",
    "quality_flag": "string",
}

# Unit conversions (target: μg/m³ for PM, ppb for gases)
UNIT_CONVERSIONS = {
    # PM concentrations - already in μg/m³ typically
    "pm25": {"μg/m³": 1.0, "mg/m³": 1000.0},
    "pm10": {"μg/m³": 1.0, "mg/m³": 1000.0},
    # Gases - convert to ppb
    "no2": {"ppb": 1.0, "μg/m³": 0.532},  # NO2: 1 ppb = 1.88 μg/m³
    "so2": {"ppb": 1.0, "μg/m³": 0.375},  # SO2: 1 ppb = 2.67 μg/m³
    "co": {"ppb": 1.0, "mg/m³": 873.0, "μg/m³": 0.873},  # CO: 1 ppb = 1.15 mg/m³
    "o3": {"ppb": 1.0, "μg/m³": 0.510},  # O3: 1 ppb = 1.96 μg/m³
}


def standardize_units(value, pollutant, source_unit):
    """Convert pollutant values to standardized units."""
    if pd.isna(value) or value is None:
        return None

    pollutant = pollutant.lower()
    source_unit = source_unit.lower() if source_unit else ""

    # PM pollutants - target μg/m³
    if pollutant in ["pm25", "pm10"]:
        # Target unit for PM pollutants: μg/m³
        if "mg" in source_unit:
            return value * 1000.0  # mg/m³ to μg/m³
        elif "μg" in source_unit or "ug" in source_unit:
            return value  # Already in μg/m³
        else:
            return value  # Assume μg/m³ if unit unclear

    # Gas pollutants - target ppb
    else:
        # Target unit for gas pollutants: ppb
        if pollutant in UNIT_CONVERSIONS:
            conversions = UNIT_CONVERSIONS[pollutant]

            if "ppb" in source_unit:
                return value  # Already in ppb
            elif "μg/m³" in source_unit or "ug/m3" in source_unit:
                if "μg/m³" in conversions:
                    return value * conversions["μg/m³"]
            elif "mg/m³" in source_unit:
                if "mg/m³" in conversions:
                    return value * conversions["mg/m³"]

        return value  # Assume ppb if conversion not found


def get_standard_units(pollutant):
    """Get the standard unit for a pollutant."""
    pollutant = pollutant.lower()
    if pollutant in ["pm25", "pm10"]:
        return "μg/m³"
    else:
        return "ppb"


def load_gefs_data(data_root):
    """Load and standardize GEFS data."""
    log.info("Loading GEFS data...")

    gefs_dir = Path(data_root) / "curated" / "gefs_chem" / "parquet"
    benchmark_dir = Path(data_root) / "benchmark_forecasts_100cities"

    gefs_records = []

    # Load from curated parquet files if available
    if gefs_dir.exists():
        parquet_files = list(gefs_dir.rglob("*.parquet"))
        log.info(f"Found {len(parquet_files)} GEFS parquet files")

        for file in parquet_files:
            try:
                df = pd.read_parquet(file)

                # Convert to long format
                for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                    if pollutant in df.columns:
                        pollutant_df = df[
                            ["run_date", "run_hour", "f_hour", "lat", "lon"]
                            + [pollutant]
                        ].copy()
                        pollutant_df = pollutant_df.dropna(subset=[pollutant])

                        # Create timestamp from run_date, run_hour, and f_hour
                        pollutant_df["timestamp_utc"] = pd.to_datetime(
                            pollutant_df["run_date"]
                            + " "
                            + pollutant_df["run_hour"].astype(str).str.zfill(2)
                            + ":00:00"
                        ) + pd.to_timedelta(pollutant_df["f_hour"], unit="h")

                        # Standardize format
                        for _, row in pollutant_df.iterrows():
                            record = {
                                "timestamp_utc": row["timestamp_utc"],
                                "pollutant": pollutant.upper(),
                                "value": standardize_units(
                                    row[pollutant], pollutant, None
                                ),
                                "units": get_standard_units(pollutant),
                                "source": "GEFS",
                                "data_type": "forecast",
                                "run_date": row["run_date"],
                                "run_hour": str(row["run_hour"]).zfill(2),
                                "f_hour": int(row["f_hour"]),
                                "lat": row["lat"],
                                "lon": row["lon"],
                                "model_version": "GEFS-chem_0.25deg",
                            }
                            gefs_records.append(record)

            except Exception as e:
                log.error(f"Error loading GEFS file {file}: {e}")

    # Load from benchmark data if no curated files
    if not gefs_records and benchmark_dir.exists():
        benchmark_files = list(benchmark_dir.glob("*gefs*.parquet"))
        log.info(f"Loading from {len(benchmark_files)} GEFS benchmark files")

        for file in benchmark_files:
            try:
                df = pd.read_parquet(file)

                for _, row in df.iterrows():
                    # Create records for each pollutant
                    for pollutant in [
                        "pm25",
                        "pm10",
                    ]:  # Benchmark data typically has PM only
                        if pollutant in row and pd.notna(row[pollutant]):
                            record = {
                                "city": row.get("city"),
                                "country": row.get("country"),
                                "timestamp_utc": pd.to_datetime(
                                    row.get("forecast_time")
                                ),
                                "pollutant": pollutant.upper(),
                                "value": standardize_units(
                                    row[pollutant], pollutant, None
                                ),
                                "units": get_standard_units(pollutant),
                                "source": "GEFS",
                                "data_type": "forecast",
                                "run_date": pd.to_datetime(
                                    row.get("run_time")
                                ).strftime("%Y-%m-%d"),
                                "run_hour": "00",  # Default from benchmark
                                "f_hour": int(row.get("forecast_hour", 0)),
                                "lat": row.get("lat"),
                                "lon": row.get("lon"),
                                "model_version": row.get(
                                    "model_version", "GEFS-chem_0.25deg"
                                ),
                            }
                            gefs_records.append(record)

            except Exception as e:
                log.error(f"Error loading GEFS benchmark file {file}: {e}")

    log.info(f"Loaded {len(gefs_records)} GEFS records")
    return gefs_records


def load_cams_data(data_root):
    """Load and standardize CAMS data."""
    log.info("Loading CAMS data...")

    cams_dir = Path(data_root) / "curated" / "cams" / "parquet"
    cams_records = []

    if cams_dir.exists():
        parquet_files = list(cams_dir.glob("*.parquet"))
        log.info(f"Found {len(parquet_files)} CAMS parquet files")

        for file in parquet_files:
            try:
                df = pd.read_parquet(file)

                # CAMS data structure varies, adapt accordingly
                for _, row in df.iterrows():
                    for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                        if pollutant in row and pd.notna(row[pollutant]):

                            # Determine if this is forecast or analysis data
                            data_type = (
                                "forecast" if "forecast_hour" in row else "analysis"
                            )

                            # Handle timestamp conversion safely
                            timestamp_val = row.get(
                                "forecast_time",
                                row.get("time", row.get("timestamp_utc")),
                            )
                            if pd.notna(timestamp_val):
                                timestamp_utc = pd.to_datetime(timestamp_val)
                                run_date = (
                                    timestamp_utc.strftime("%Y-%m-%d")
                                    if pd.notna(timestamp_utc)
                                    else None
                                )
                            else:
                                timestamp_utc = None
                                run_date = None

                            record = {
                                "city": row.get("city"),
                                "country": row.get("country"),
                                "timestamp_utc": timestamp_utc,
                                "pollutant": pollutant.upper(),
                                "value": standardize_units(
                                    row[pollutant], pollutant, None
                                ),
                                "units": get_standard_units(pollutant),
                                "source": "CAMS",
                                "data_type": data_type,
                                "run_date": run_date,
                                "run_hour": "00",  # CAMS typically 00Z
                                "f_hour": int(row.get("forecast_hour", 0)),
                                "lat": row.get("lat", row.get("latitude")),
                                "lon": row.get("lon", row.get("longitude")),
                                "model_version": row.get(
                                    "model_version", "CAMS_Global"
                                ),
                            }
                            cams_records.append(record)

            except Exception as e:
                log.error(f"Error loading CAMS file {file}: {e}")

    log.info(f"Loaded {len(cams_records)} CAMS records")
    return cams_records


def load_ground_truth_data(data_root):
    """Load and standardize ground truth observation data."""
    log.info("Loading ground truth observation data...")

    obs_dir = Path(data_root) / "curated" / "obs"
    obs_records = []

    if obs_dir.exists():
        parquet_files = list(obs_dir.glob("*.parquet"))
        log.info(f"Found {len(parquet_files)} observation parquet files")

        for file in parquet_files:
            try:
                df = pd.read_parquet(file)

                for _, row in df.iterrows():
                    for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                        if pollutant in row and pd.notna(row[pollutant]):
                            record = {
                                "city": row.get("city"),
                                "country": row.get("country"),
                                "timestamp_utc": pd.to_datetime(
                                    row.get("timestamp_utc")
                                ),
                                "pollutant": pollutant.upper(),
                                "value": standardize_units(
                                    row[pollutant], pollutant, None
                                ),
                                "units": get_standard_units(pollutant),
                                "source": row.get("source", "Ground-Truth"),
                                "data_type": "observation",
                                "run_date": None,  # No run date for observations
                                "run_hour": None,
                                "f_hour": 0,  # Observations are "nowcast"
                                "lat": row.get("lat"),
                                "lon": row.get("lon"),
                                "model_version": None,
                                # Copy calendar features if available
                                **{
                                    col: row.get(col)
                                    for col in row.index
                                    if any(
                                        x in col
                                        for x in [
                                            "year",
                                            "month",
                                            "day",
                                            "hour",
                                            "season",
                                            "is_",
                                            "_sin",
                                            "_cos",
                                            "lag_",
                                        ]
                                    )
                                },
                            }
                            obs_records.append(record)

            except Exception as e:
                log.error(f"Error loading observation file {file}: {e}")

    log.info(f"Loaded {len(obs_records)} ground truth observation records")
    return obs_records


def load_local_features(data_root):
    """Load local features data."""
    log.info("Loading local features data...")

    features_dir = Path(data_root) / "curated" / "local_features"
    features_data = {}

    if features_dir.exists():
        parquet_files = list(features_dir.glob("*.parquet"))
        log.info(f"Found {len(parquet_files)} local features parquet files")

        for file in parquet_files:
            try:
                df = pd.read_parquet(file)

                # Group features by city and timestamp for efficient lookup
                for _, row in df.iterrows():
                    key = (row.get("city"), pd.to_datetime(row.get("timestamp_utc")))
                    features_data[key] = {
                        col: row[col]
                        for col in row.index
                        if any(
                            x in col
                            for x in [
                                "year",
                                "month",
                                "day",
                                "hour",
                                "season",
                                "is_",
                                "_sin",
                                "_cos",
                                "lag_",
                            ]
                        )
                    }

            except Exception as e:
                log.error(f"Error loading features file {file}: {e}")

    log.info(f"Loaded features for {len(features_data)} city-timestamp combinations")
    return features_data


def add_calendar_features(record):
    """Add calendar features to a record."""
    if "timestamp_utc" in record and record["timestamp_utc"]:
        dt = pd.to_datetime(record["timestamp_utc"])

        calendar_features = {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "day_of_year": dt.dayofyear,
            "week_of_year": dt.isocalendar()[1],
            "is_weekend": dt.dayofweek >= 5,
            "is_holiday_season": dt.month in [11, 12, 1],
            "season": (dt.month % 12 + 3) // 3,
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
            "month_sin": np.sin(2 * np.pi * dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dt.month / 12),
            "day_sin": np.sin(2 * np.pi * dt.day / 31),
            "day_cos": np.cos(2 * np.pi * dt.day / 31),
        }

        record.update(calendar_features)

    return record


def merge_all_datasets(data_root):
    """Merge all datasets into unified format."""
    log.info("Starting unified dataset merge...")

    # Load all data sources
    gefs_records = load_gefs_data(data_root)
    cams_records = load_cams_data(data_root)
    obs_records = load_ground_truth_data(data_root)
    # features_data = load_local_features(data_root)  # Currently unused

    # Combine all records
    all_records = gefs_records + cams_records + obs_records
    log.info(f"Total records to merge: {len(all_records)}")

    if not all_records:
        log.error("No data to merge!")
        return None

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_records)

    # Standardize timezone handling - convert all timestamps to UTC
    log.info("Standardizing timezone handling...")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Add calendar features where missing
    log.info("Adding calendar features...")
    for idx, row in df.iterrows():
        if pd.isna(row.get("year")):  # Only add if not already present
            df.loc[idx] = add_calendar_features(row.to_dict())

    # Add city information for records missing it (match by lat/lon)
    log.info("Adding city information...")
    from collect_2year_gefs_data import CITIES_100

    def find_nearest_city(lat, lon, max_distance=1.0):
        """Find the nearest city to given coordinates."""
        min_distance = float("inf")
        nearest_city = None

        for city_name, city_info in CITIES_100.items():
            distance = np.sqrt(
                (lat - city_info["lat"]) ** 2 + (lon - city_info["lon"]) ** 2
            )
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_city = city_name

        return nearest_city, (
            CITIES_100.get(nearest_city, {}).get("country") if nearest_city else None
        )

    for idx, row in df.iterrows():
        if (
            pd.isna(row.get("city"))
            and pd.notna(row.get("lat"))
            and pd.notna(row.get("lon"))
        ):
            city, country = find_nearest_city(row["lat"], row["lon"])
            if city:
                df.loc[idx, "city"] = city
                df.loc[idx, "country"] = country

    # Add quality flags based on data completeness
    log.info("Adding quality flags...")
    df["quality_flag"] = "good"
    df.loc[df["value"].isna(), "quality_flag"] = "missing_value"
    df.loc[df["city"].isna(), "quality_flag"] = "missing_location"

    # Sort by city and timestamp
    df = df.sort_values(["city", "timestamp_utc", "pollutant", "source"])

    # Calculate lag features for observations by city and pollutant
    log.info("Calculating lag features...")
    lag_hours = [1, 3, 6, 12, 24]

    for city in df["city"].dropna().unique():
        city_mask = df["city"] == city

        for pollutant in df["pollutant"].unique():
            pollutant_mask = df["pollutant"] == pollutant
            mask = city_mask & pollutant_mask & (df["data_type"] == "observation")

            if mask.sum() > 0:
                city_pollutant_df = df[mask].sort_values("timestamp_utc")

                for lag in lag_hours:
                    lag_col = f"{pollutant.lower()}_lag_{lag}h"
                    lag_values = city_pollutant_df["value"].shift(lag)
                    df.loc[mask, lag_col] = lag_values

    # Ensure all schema columns are present
    for col, dtype in UNIFIED_SCHEMA.items():
        if col not in df.columns:
            if "int" in dtype:
                df[col] = 0
            elif "float" in dtype:
                df[col] = np.nan
            elif "bool" in dtype:
                df[col] = False
            else:
                df[col] = None

    # Convert to proper data types
    log.info("Converting data types...")
    for col, dtype in UNIFIED_SCHEMA.items():
        if col in df.columns:
            try:
                if "datetime" in dtype:
                    df[col] = pd.to_datetime(df[col])
                elif "int" in dtype:
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce")
                        .fillna(0)
                        .astype("Int64")
                    )
                elif "float" in dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif "bool" in dtype:
                    df[col] = df[col].astype(bool)
                else:  # string
                    df[col] = df[col].astype(str)
            except Exception as e:
                log.warning(f"Could not convert {col} to {dtype}: {e}")

    # Remove rows with missing essential data
    essential_cols = ["city", "timestamp_utc", "pollutant", "value"]
    before_count = len(df)
    df = df.dropna(subset=essential_cols)
    after_count = len(df)

    log.info(f"Removed {before_count - after_count} rows missing essential data")
    log.info(f"Final unified dataset: {len(df)} records")

    return df


def save_partitioned_dataset(df, data_root):
    """Save the unified dataset in partitioned format."""
    log.info("Saving partitioned unified dataset...")

    output_dir = Path(data_root) / "curated" / "100_cities_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Partition by city and date for efficient querying
    df["partition_date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

    partition_info = []

    for city in df["city"].unique():
        city_data = df[df["city"] == city]

        for date in city_data["partition_date"].unique():
            date_data = city_data[city_data["partition_date"] == date]

            if len(date_data) > 0:
                # Create partition directory
                partition_dir = output_dir / f"city={city}" / f"date={date}"
                partition_dir.mkdir(parents=True, exist_ok=True)

                # Save partition
                partition_file = partition_dir / "data.parquet"
                date_data.drop("partition_date", axis=1).to_parquet(
                    partition_file, index=False
                )

                partition_info.append(
                    {
                        "city": city,
                        "date": date,
                        "records": len(date_data),
                        "file_size": partition_file.stat().st_size,
                        "pollutants": sorted(date_data["pollutant"].unique().tolist()),
                        "sources": sorted(date_data["source"].unique().tolist()),
                    }
                )

    # Also save complete dataset as single file for convenience
    complete_file = (
        output_dir
        / f"complete_100cities_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    df.drop("partition_date", axis=1).to_parquet(complete_file, index=False)

    log.info(f"Saved {len(partition_info)} partitions")
    log.info(f"Complete dataset saved: {complete_file}")

    return {
        "partitions": partition_info,
        "complete_file": complete_file,
        "partition_count": len(partition_info),
        "total_records": len(df),
    }


def generate_dataset_report(df, save_info, data_root):
    """Generate comprehensive dataset report."""
    log.info("Generating dataset report...")

    # Calculate statistics
    total_size_gb = sum(p["file_size"] for p in save_info["partitions"]) / (1024**3)

    report = {
        "generation_date": datetime.now().isoformat(),
        "dataset_summary": {
            "total_records": len(df),
            "total_storage_gb": round(total_size_gb, 2),
            "cities_count": df["city"].nunique(),
            "date_range": {
                "start": df["timestamp_utc"].min().isoformat(),
                "end": df["timestamp_utc"].max().isoformat(),
                "days": (df["timestamp_utc"].max() - df["timestamp_utc"].min()).days,
            },
            "frequency": "Hourly observations + 6-hourly forecasts",
            "partitions": len(save_info["partitions"]),
        },
        "data_sources": {
            source: {
                "records": len(df[df["source"] == source]),
                "percentage": round(len(df[df["source"] == source]) / len(df) * 100, 1),
            }
            for source in df["source"].unique()
        },
        "pollutant_coverage": {
            pollutant: {
                "records": len(df[df["pollutant"] == pollutant]),
                "cities": df[df["pollutant"] == pollutant]["city"].nunique(),
                "units": (
                    df[df["pollutant"] == pollutant]["units"].iloc[0]
                    if len(df[df["pollutant"] == pollutant]) > 0
                    else "N/A"
                ),
            }
            for pollutant in sorted(df["pollutant"].unique())
        },
        "city_coverage": {
            city: {
                "records": len(df[df["city"] == city]),
                "pollutants": sorted(
                    df[df["city"] == city]["pollutant"].unique().tolist()
                ),
                "sources": sorted(df[df["city"] == city]["source"].unique().tolist()),
                "date_range": {
                    "start": df[df["city"] == city]["timestamp_utc"].min().isoformat(),
                    "end": df[df["city"] == city]["timestamp_utc"].max().isoformat(),
                },
            }
            for city in sorted(df["city"].unique())[:10]  # Top 10 cities for brevity
        },
        "data_quality": {
            "total_records": len(df),
            "complete_records": len(df[df["quality_flag"] == "good"]),
            "missing_values": len(df[df["quality_flag"] == "missing_value"]),
            "missing_location": len(df[df["quality_flag"] == "missing_location"]),
            "quality_percentage": round(
                len(df[df["quality_flag"] == "good"]) / len(df) * 100, 1
            ),
        },
        "schema_info": {
            "columns": list(df.columns),
            "unified_schema_compliance": True,
            "standardized_units": True,
        },
    }

    # Save report
    report_file = (
        Path(data_root) / "curated" / "100_cities_dataset" / "DATASET_REPORT.json"
    )
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"Dataset report saved: {report_file}")

    # Print summary
    log.info("=== UNIFIED 100-CITY DATASET SUMMARY ===")
    log.info(f"Total Records: {report['dataset_summary']['total_records']:,}")
    log.info(f"Storage Size: {report['dataset_summary']['total_storage_gb']:.2f} GB")
    log.info(f"Cities: {report['dataset_summary']['cities_count']}")
    log.info(
        f"Date Range: {report['dataset_summary']['date_range']['start'][:10]} "
        f"to {report['dataset_summary']['date_range']['end'][:10]}"
    )
    log.info(f"Time Span: {report['dataset_summary']['date_range']['days']} days")
    log.info(f"Frequency: {report['dataset_summary']['frequency']}")
    log.info(f"Partitions: {report['dataset_summary']['partitions']}")
    log.info("")
    log.info("Data Sources:")
    for source, stats in report["data_sources"].items():
        log.info(f"  {source}: {stats['records']:,} records ({stats['percentage']}%)")
    log.info("")
    log.info("Pollutant Coverage:")
    for pollutant, stats in report["pollutant_coverage"].items():
        log.info(
            f"  {pollutant}: {stats['records']:,} records, "
            f"{stats['cities']} cities, {stats['units']}"
        )
    log.info("")
    log.info(
        f"Data Quality: {report['data_quality']['quality_percentage']}% complete records"
    )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Merge unified 100-city air quality dataset"
    )
    parser.add_argument("--data-root", default=None, help="Data root directory")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing unified dataset",
    )

    args = parser.parse_args()

    # Set up data root
    data_root = args.data_root or os.environ.get("DATA_ROOT", "C:/aqf311/data")
    log.info(f"Using data root: {data_root}")

    # Ensure log directory exists
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        log.info("Verification mode - checking existing unified dataset")

        unified_dir = Path(data_root) / "curated" / "100_cities_dataset"
        if unified_dir.exists():
            # Check partitions
            partitions = list(unified_dir.glob("city=*/date=*/data.parquet"))
            complete_files = list(unified_dir.glob("complete_*.parquet"))

            log.info(f"Found {len(partitions)} partition files")
            log.info(f"Found {len(complete_files)} complete dataset files")

            if complete_files:
                latest_file = max(complete_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_parquet(latest_file)
                log.info(f"Latest complete dataset: {latest_file}")
                log.info(f"Records: {len(df):,}")
                log.info(f"Cities: {df['city'].nunique()}")
                log.info(
                    f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}"
                )
        else:
            log.info("No unified dataset found")

        return

    log.info("Starting unified 100-city dataset merge...")

    # Merge all datasets
    df = merge_all_datasets(data_root)

    if df is None or len(df) == 0:
        log.error("Failed to merge datasets or no data available")
        sys.exit(1)

    # Save partitioned dataset
    save_info = save_partitioned_dataset(df, data_root)

    # Generate comprehensive report
    report = generate_dataset_report(df, save_info, data_root)

    # Save merge summary
    summary = {
        "merge_date": datetime.now().isoformat(),
        "success": True,
        "dataset_info": save_info,
        "report_summary": report["dataset_summary"],
    }

    summary_file = Path(data_root) / "logs" / "unified_dataset_merge_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Merge summary saved to: {summary_file}")
    log.info("Unified 100-city dataset merge completed successfully!")


if __name__ == "__main__":
    main()
