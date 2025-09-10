#!/usr/bin/env python3
"""
Generate 3-Year Hourly Air Quality Forecasting Dataset - Optimized Version

Creates a comprehensive synthetic dataset spanning 3 years (2022-2025) with hourly frequency.
Uses vectorized operations for better performance with large datasets.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import math

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# City coordinates and characteristics
CITY_COORDS = {
    "Berlin": (52.5200, 13.4050),
    "Hamburg": (53.5511, 9.9937),
    "Munich": (48.1351, 11.5820),
}

CITY_CHARACTERISTICS = {
    "Berlin": {
        "population": 3677000,
        "urban_intensity": 0.9,
        "coastal_influence": 0.1,
        "industrial_score": 0.7,
        "traffic_base": 0.8,
        "elevation": 34,
        "temp_base": 12,
        "pollution_base": {"pm25": 12, "pm10": 20, "no2": 25, "o3": 35},
    },
    "Hamburg": {
        "population": 1945000,
        "urban_intensity": 0.8,
        "coastal_influence": 0.9,
        "industrial_score": 0.8,
        "traffic_base": 0.7,
        "elevation": 6,
        "temp_base": 10,
        "pollution_base": {"pm25": 11, "pm10": 18, "no2": 22, "o3": 33},
    },
    "Munich": {
        "population": 1488000,
        "urban_intensity": 0.8,
        "coastal_influence": 0.0,
        "industrial_score": 0.6,
        "traffic_base": 0.75,
        "elevation": 519,
        "temp_base": 13,
        "pollution_base": {"pm25": 10, "pm10": 17, "no2": 20, "o3": 37},
    },
}


def generate_optimized_dataset(
    start_date: str, end_date: str, output_path: Path
) -> pd.DataFrame:
    """Generate complete dataset using vectorized operations."""
    log.info(
        f"Generating optimized 3-year hourly dataset from {start_date} to {end_date}"
    )

    # Generate time range
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    cities = list(CITY_COORDS.keys())

    log.info(
        f"Time range: {len(date_range):,} hours, {len(cities)} cities = {len(date_range) * len(cities):,} total rows"
    )

    # Create base dataframe efficiently
    data = []
    for city in cities:
        city_data = pd.DataFrame(
            {
                "city": city,
                "datetime": date_range,
                "date": date_range.strftime("%Y-%m-%d"),
                "hour": date_range.hour,
                "year": date_range.year,
                "month": date_range.month,
                "day": date_range.day,
                "dayofweek": date_range.dayofweek,
                "dayofyear": date_range.dayofyear,
                "latitude": CITY_COORDS[city][0],
                "longitude": CITY_COORDS[city][1],
            }
        )
        data.append(city_data)

    df = pd.concat(data, ignore_index=True)
    log.info(f"Base dataframe created: {df.shape}")

    # Add city characteristics
    df["urban_intensity"] = df["city"].map(
        {k: v["urban_intensity"] for k, v in CITY_CHARACTERISTICS.items()}
    )
    df["coastal_influence"] = df["city"].map(
        {k: v["coastal_influence"] for k, v in CITY_CHARACTERISTICS.items()}
    )
    df["industrial_score"] = df["city"].map(
        {k: v["industrial_score"] for k, v in CITY_CHARACTERISTICS.items()}
    )
    df["traffic_base"] = df["city"].map(
        {k: v["traffic_base"] for k, v in CITY_CHARACTERISTICS.items()}
    )

    # Vectorized meteorological patterns
    log.info("Adding meteorological patterns (vectorized)...")
    np.random.seed(42)

    # Temperature patterns (vectorized)
    temp_base = df["city"].map(
        {k: v["temp_base"] for k, v in CITY_CHARACTERISTICS.items()}
    )
    seasonal_temp = 15 * np.cos(2 * np.pi * (df["dayofyear"] - 200) / 365)
    daily_temp = 8 * np.cos(2 * np.pi * (df["hour"] - 14) / 24)
    climate_trend = 0.02 * (df["year"] - 2022)
    temp_noise = np.random.normal(0, 3, len(df))
    df["temperature"] = (
        temp_base + seasonal_temp + daily_temp + climate_trend + temp_noise
    )

    # Wind speed (vectorized)
    wind_base = df["coastal_influence"] * 2 + 3
    wind_seasonal = 1.5 * np.cos(2 * np.pi * (df["dayofyear"] - 60) / 365)
    wind_daily = 1.5 * np.sin(2 * np.pi * df["hour"] / 24)
    df["wind_speed"] = np.clip(
        wind_base + wind_seasonal + wind_daily + np.random.normal(0, 1.5, len(df)),
        0.5,
        25,
    )

    # Wind direction (vectorized)
    prevailing = 270
    seasonal_shift = 30 * np.sin(2 * np.pi * df["dayofyear"] / 365)
    daily_shift = 20 * np.sin(2 * np.pi * df["hour"] / 24)
    df["wind_direction"] = (
        prevailing + seasonal_shift + daily_shift + np.random.normal(0, 45, len(df))
    ) % 360

    # Humidity (vectorized)
    base_humidity = 70 - (df["temperature"] - 10) * 1.5
    humidity_daily = 10 * np.cos(2 * np.pi * (df["hour"] - 6) / 24)
    coastal_effect = df["coastal_influence"] * 10
    df["humidity"] = np.clip(
        base_humidity
        + humidity_daily
        + coastal_effect
        + np.random.normal(0, 8, len(df)),
        20,
        95,
    )

    # Pressure (vectorized)
    pressure_seasonal = 8 * np.cos(2 * np.pi * (df["dayofyear"] - 30) / 365)
    pressure_daily = 2 * np.cos(2 * np.pi * (df["hour"] - 10) / 24)
    df["pressure"] = (
        1013 + pressure_seasonal + pressure_daily + np.random.normal(0, 5, len(df))
    )

    # Solar radiation (vectorized)
    daylight_mask = (df["hour"] >= 6) & (df["hour"] <= 18)
    max_solar = np.where(
        daylight_mask, 800 * np.sin(2 * np.pi * (df["dayofyear"] - 80) / 365), 0
    )
    daily_solar = np.where(
        daylight_mask, max_solar * np.sin(np.pi * (df["hour"] - 6) / 12), 0
    )
    cloud_factor = np.clip(1 - (df["humidity"] - 40) / 100, 0.1, 1)
    df["solar_radiation"] = np.clip(
        daily_solar * cloud_factor + np.random.normal(0, 50, len(df)), 0, 1000
    )

    # Boundary layer height (vectorized)
    temp_effect = (df["temperature"] - 10) * 25
    wind_effect = df["wind_speed"] * 40
    time_effect = np.where(
        (df["hour"] >= 6) & (df["hour"] <= 18),
        200 * np.sin(2 * np.pi * (df["hour"] - 12) / 24),
        -100,
    )
    df["boundary_layer_height"] = np.clip(
        600
        + temp_effect
        + wind_effect
        + time_effect
        + np.random.normal(0, 100, len(df)),
        100,
        3000,
    )

    # Precipitation (vectorized)
    precip_prob = 0.1 + 0.05 * np.cos(2 * np.pi * (df["dayofyear"] - 30) / 365)
    precip_random = np.random.random(len(df))
    precip_values = np.random.exponential(2.0, len(df))
    df["precipitation"] = np.where(precip_random < precip_prob, precip_values, 0)

    log.info("Meteorological patterns added")

    # Vectorized pollutant concentrations
    log.info("Adding pollutant concentrations (vectorized)...")
    np.random.seed(43)

    # Traffic patterns (vectorized)
    weekday_mask = df["dayofweek"] < 5
    rush_morning = (df["hour"] >= 7) & (df["hour"] <= 9)
    rush_evening = (df["hour"] >= 17) & (df["hour"] <= 19)
    business_hours = (df["hour"] >= 9) & (df["hour"] <= 17)
    night_hours = (df["hour"] < 6) | (df["hour"] > 22)

    traffic_factor = np.ones(len(df))
    # Weekday patterns
    traffic_factor = np.where(weekday_mask & rush_morning, 1.6, traffic_factor)
    traffic_factor = np.where(weekday_mask & rush_evening, 1.4, traffic_factor)
    traffic_factor = np.where(
        weekday_mask & business_hours & ~rush_morning & ~rush_evening,
        1.2,
        traffic_factor,
    )
    traffic_factor = np.where(weekday_mask & night_hours, 0.6, traffic_factor)
    # Weekend patterns
    traffic_factor = np.where(~weekday_mask, 0.7, traffic_factor)
    weekend_active = ~weekday_mask & (df["hour"] >= 10) & (df["hour"] <= 16)
    traffic_factor = np.where(weekend_active, 0.9, traffic_factor)

    # Seasonal patterns (vectorized)
    pm_seasonal = 1 + 0.3 * np.cos(2 * np.pi * (df["dayofyear"] - 330) / 365)
    no2_seasonal = 1 + 0.4 * np.cos(2 * np.pi * (df["dayofyear"] - 330) / 365)
    o3_seasonal = 1 + 0.5 * np.sin(2 * np.pi * (df["dayofyear"] - 80) / 365)

    # Dispersion effects (vectorized)
    dispersion_factor = 1.5 / (
        1 + df["wind_speed"] * df["boundary_layer_height"] / 1000
    )
    temp_factor_o3 = 1 + 0.02 * (df["temperature"] - 15)
    temp_factor_pm = 1 - 0.01 * (df["temperature"] - 15)
    photo_factor = 1 + df["solar_radiation"] / 1000 * 0.3

    # Calculate pollutants for each city
    for pollutant in ["pm25", "pm10", "no2", "o3"]:
        base_levels = df["city"].map(
            {k: v["pollution_base"][pollutant] for k, v in CITY_CHARACTERISTICS.items()}
        )

        if pollutant in ["pm25", "pm10"]:
            # PM patterns
            conc_traffic = base_levels * traffic_factor * df["traffic_base"]
            conc_seasonal = conc_traffic * pm_seasonal
            conc_met = conc_seasonal * dispersion_factor * temp_factor_pm
            noise_scale = 2 if pollutant == "pm25" else 3
            max_val = 100 if pollutant == "pm25" else 150
            df[f"actual_{pollutant}"] = np.clip(
                conc_met + np.random.normal(0, noise_scale, len(df)), 1, max_val
            )

        elif pollutant == "no2":
            # NO2 patterns
            conc_traffic = base_levels * traffic_factor * df["traffic_base"]
            conc_seasonal = conc_traffic * no2_seasonal
            conc_met = conc_seasonal * dispersion_factor
            conc_photo = conc_met * (1 - df["solar_radiation"] / 1000 * 0.2)
            df[f"actual_{pollutant}"] = np.clip(
                conc_photo + np.random.normal(0, 3, len(df)), 2, 120
            )

        elif pollutant == "o3":
            # O3 patterns (calculated after NO2)
            no2_titration = np.clip(1 - (df["actual_no2"] - 10) / 100 * 0.3, 0.3, 1.5)
            conc_seasonal = base_levels * o3_seasonal * no2_titration
            conc_temp = conc_seasonal * temp_factor_o3
            conc_photo = conc_temp * photo_factor
            # O3 daily pattern (peaks in afternoon)
            afternoon_boost = np.where(
                (df["hour"] >= 8) & (df["hour"] <= 20),
                1 + 0.3 * np.sin(2 * np.pi * (df["hour"] - 14) / 24),
                0.8,
            )
            conc_daily = conc_photo * afternoon_boost
            df[f"actual_{pollutant}"] = np.clip(
                conc_daily + np.random.normal(0, 4, len(df)), 5, 200
            )

    log.info("Pollutant concentrations added")

    # Add forecast data (vectorized)
    log.info("Adding forecast data (vectorized)...")
    np.random.seed(44)

    model_bias = {
        "cams": {"pm25": 0.02, "pm10": -0.05, "no2": 0.08, "o3": -0.03},
        "noaa_gefs_aerosol": {"pm25": -0.01, "pm10": 0.03, "no2": -0.06, "o3": 0.02},
    }

    model_rmse = {
        "cams": {"pm25": 0.15, "pm10": 0.18, "no2": 0.22, "o3": 0.12},
        "noaa_gefs_aerosol": {"pm25": 0.17, "pm10": 0.16, "no2": 0.20, "o3": 0.14},
    }

    for provider in ["cams", "noaa_gefs_aerosol"]:
        for pollutant in ["pm25", "pm10", "no2", "o3"]:
            actual_col = f"actual_{pollutant}"
            forecast_col = f"forecast_{provider}_{pollutant}"

            bias = model_bias[provider][pollutant]
            rmse = model_rmse[provider][pollutant]

            # Base forecast with bias
            forecast_base = df[actual_col] * (1 + bias)

            # Random errors
            random_errors = np.random.normal(0, rmse * df[actual_col].mean(), len(df))

            df[forecast_col] = np.clip(forecast_base + random_errors, 0.5, None)

    # Add forecast metadata
    df["forecast_made_date"] = (df["datetime"] - timedelta(hours=24)).dt.strftime(
        "%Y-%m-%d"
    )
    df["forecast_lead_hours"] = 24

    log.info("Forecast data added")

    # Add basic external features (vectorized)
    log.info("Adding external features (vectorized)...")
    np.random.seed(45)

    # Fire activity (seasonal, random)
    fire_season_factor = 1 + 0.5 * np.sin(2 * np.pi * (df["dayofyear"] - 150) / 365)
    fire_prob = 0.02 * fire_season_factor
    fire_random = np.random.random(len(df))
    fire_count = np.where(fire_random < fire_prob, np.random.poisson(2, len(df)), 0)
    df["fire_fire_count"] = fire_count
    df["fire_total_frp"] = np.where(
        fire_count > 0, fire_count * np.random.exponential(50, len(df)), 0
    )

    # Construction activity
    construction_base = df["urban_intensity"] * 20
    construction_seasonal = 1 + 0.3 * np.sin(2 * np.pi * (df["dayofyear"] - 120) / 365)
    working_hours = weekday_mask & (df["hour"] >= 7) & (df["hour"] <= 17)
    construction_factor = np.where(working_hours, 1.0, 0.1)
    df["construction_site_count"] = (
        construction_base * construction_seasonal * construction_factor
        + np.random.poisson(2, len(df))
    )

    # Infrastructure (static by city)
    df["infrastructure_major_roads"] = df["traffic_base"] * 50 + np.random.poisson(
        5, len(df)
    )
    df["infrastructure_railways"] = df["urban_intensity"] * 10 + np.random.poisson(
        2, len(df)
    )
    df["infrastructure_industrial_areas"] = df[
        "industrial_score"
    ] * 15 + np.random.poisson(3, len(df))

    # Holidays (simplified)
    major_holidays_mask = (
        ((df["month"] == 1) & (df["day"] == 1))
        | ((df["month"] == 5) & (df["day"] == 1))
        | ((df["month"] == 10) & (df["day"] == 3))
        | ((df["month"] == 12) & (df["day"] == 25))
    )
    df["is_public_holiday"] = major_holidays_mask.astype(int)

    school_holidays_mask = (
        (df["month"].isin([7, 8]))
        | ((df["month"] == 12) & (df["day"] > 20))
        | ((df["month"] == 1) & (df["day"] < 10))
    )
    df["is_school_holiday"] = school_holidays_mask.astype(int)

    # Traffic intensity
    traffic_intensity = df["traffic_base"] * traffic_factor + np.random.normal(
        0, 0.1, len(df)
    )
    df["traffic_intensity"] = traffic_intensity

    # Economic activity
    economic_base = 0.8
    business_hours_all = (df["hour"] >= 8) & (df["hour"] <= 18)
    friday_mask = df["dayofweek"] == 5
    saturday_mask = df["dayofweek"] == 6
    sunday_mask = df["dayofweek"] == 0

    economic_activity = np.full(len(df), economic_base)
    economic_activity = np.where(
        weekday_mask & business_hours_all, economic_base, economic_activity
    )
    economic_activity = np.where(friday_mask, economic_base * 0.9, economic_activity)
    economic_activity = np.where(saturday_mask, economic_base * 0.6, economic_activity)
    economic_activity = np.where(sunday_mask, economic_base * 0.3, economic_activity)
    df["economic_activity"] = economic_activity + np.random.normal(0, 0.05, len(df))

    log.info("External features added")

    # Organize columns
    id_cols = [
        "city",
        "datetime",
        "date",
        "hour",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "latitude",
        "longitude",
        "forecast_made_date",
        "forecast_lead_hours",
    ]
    actual_cols = [col for col in df.columns if col.startswith("actual_")]
    forecast_cols = [col for col in df.columns if col.startswith("forecast_")]
    met_cols = [
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "pressure",
        "solar_radiation",
        "precipitation",
        "boundary_layer_height",
    ]
    external_cols = [
        col
        for col in df.columns
        if col not in id_cols + actual_cols + forecast_cols + met_cols
    ]

    # Reorder columns
    column_order = id_cols + actual_cols + forecast_cols + met_cols + external_cols
    df = df[column_order]

    # Save dataset
    log.info("Saving dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet first (more efficient)
    df.to_parquet(output_path.with_suffix(".parquet"), index=False)

    # Save CSV for compatibility
    df.to_csv(output_path, index=False)

    log.info(f"Dataset saved: {df.shape}")
    log.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized 3-year hourly air quality dataset"
    )
    parser.add_argument(
        "--output",
        default="data/analysis/3year_hourly_dataset_optimized.csv",
        help="Output path for the dataset",
    )
    parser.add_argument(
        "--start-date", default="2022-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Generate dataset
    df = generate_optimized_dataset(args.start_date, args.end_date, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("3-YEAR HOURLY DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print(f"Total hours: {len(df['datetime'].unique()):,}")

    # File sizes
    csv_size = output_path.stat().st_size / 1024 / 1024
    parquet_size = output_path.with_suffix(".parquet").stat().st_size / 1024 / 1024
    print(f"File size (CSV): {csv_size:.1f} MB")
    print(f"File size (Parquet): {parquet_size:.1f} MB")
    print(f"Compression ratio: {csv_size/parquet_size:.1f}x")

    # Show sample data
    print(f"\nSample data (first 5 rows):")
    sample_cols = [
        "city",
        "datetime",
        "actual_pm25",
        "forecast_cams_pm25",
        "temperature",
        "traffic_intensity",
    ]
    print(df[sample_cols].head())

    # Basic statistics
    print(f"\nBasic statistics:")
    stats_cols = [
        "actual_pm25",
        "actual_pm10",
        "actual_no2",
        "actual_o3",
        "temperature",
    ]
    print(df[stats_cols].describe())

    return 0


if __name__ == "__main__":
    exit(main())
