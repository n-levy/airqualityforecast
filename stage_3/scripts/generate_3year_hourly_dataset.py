#!/usr/bin/env python3
"""
Generate 3-Year Hourly Air Quality Forecasting Dataset

Creates a comprehensive synthetic dataset spanning 3 years (2022-2025) with hourly frequency.
Includes realistic temporal patterns, seasonal variations, weather effects, and pollutant relationships.
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
from scipy import stats
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
    },
    "Hamburg": {
        "population": 1945000,
        "urban_intensity": 0.8,
        "coastal_influence": 0.9,
        "industrial_score": 0.8,
        "traffic_base": 0.7,
        "elevation": 6,
    },
    "Munich": {
        "population": 1488000,
        "urban_intensity": 0.8,
        "coastal_influence": 0.0,
        "industrial_score": 0.6,
        "traffic_base": 0.75,
        "elevation": 519,
    },
}


def generate_base_time_series(
    start_date: str, end_date: str, freq: str = "H"
) -> pd.DataFrame:
    """Generate base time series with city combinations."""
    log.info(
        f"Generating time series from {start_date} to {end_date} with frequency {freq}"
    )

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    cities = list(CITY_COORDS.keys())

    # Create all combinations of cities and dates
    data = []
    for city in cities:
        for dt in date_range:
            data.append(
                {
                    "city": city,
                    "datetime": dt,
                    "date": dt.strftime("%Y-%m-%d"),
                    "hour": dt.hour,
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "dayofweek": dt.dayofweek,
                    "dayofyear": dt.dayofyear,
                    "latitude": CITY_COORDS[city][0],
                    "longitude": CITY_COORDS[city][1],
                }
            )

    df = pd.DataFrame(data)
    log.info(f"Generated base time series: {len(df)} rows, {len(cities)} cities")
    return df


def add_meteorological_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic meteorological patterns with hourly, daily, and seasonal variations."""
    log.info("Adding meteorological patterns...")

    df = df.copy()
    np.random.seed(42)

    for i, row in df.iterrows():
        city = row["city"]
        dt = row["datetime"]
        hour = row["hour"]
        day_of_year = row["dayofyear"]

        city_char = CITY_CHARACTERISTICS[city]

        # Temperature patterns
        # Seasonal cycle (peaks in July-August)
        seasonal_temp = 15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)

        # Daily cycle (peaks around 2-3 PM)
        daily_temp = 8 * np.cos(2 * np.pi * (hour - 14) / 24)

        # City-specific temperature baseline
        temp_baselines = {"Berlin": 12, "Hamburg": 10, "Munich": 13}
        base_temp = temp_baselines[city] + seasonal_temp + daily_temp

        # Add weather variability and trends
        temp_noise = np.random.normal(0, 3)
        climate_trend = 0.02 * (dt.year - 2022)  # Warming trend

        df.loc[i, "temperature"] = base_temp + temp_noise + climate_trend

        # Humidity (inverse relationship with temperature, higher at night/morning)
        base_humidity = 70 - (df.loc[i, "temperature"] - 10) * 1.5
        humidity_daily = 10 * np.cos(2 * np.pi * (hour - 6) / 24)  # Peak at dawn
        coastal_effect = city_char["coastal_influence"] * 10
        df.loc[i, "humidity"] = np.clip(
            base_humidity + humidity_daily + coastal_effect + np.random.normal(0, 8),
            20,
            95,
        )

        # Wind speed (higher during day, seasonal variations)
        wind_base = city_char["coastal_influence"] * 2 + 3  # Coastal cities windier
        wind_seasonal = 1.5 * np.cos(
            2 * np.pi * (day_of_year - 60) / 365
        )  # Higher in winter
        wind_daily = 1.5 * np.sin(2 * np.pi * hour / 24)  # Higher during day
        df.loc[i, "wind_speed"] = np.clip(
            wind_base + wind_seasonal + wind_daily + np.random.normal(0, 1.5), 0.5, 25
        )

        # Wind direction (prevailing westerlies with daily and seasonal variation)
        prevailing = 270  # Westerly
        seasonal_shift = 30 * np.sin(2 * np.pi * day_of_year / 365)
        daily_shift = 20 * np.sin(2 * np.pi * hour / 24)
        df.loc[i, "wind_direction"] = (
            prevailing + seasonal_shift + daily_shift + np.random.normal(0, 45)
        ) % 360

        # Pressure (higher in winter, daily cycle)
        pressure_seasonal = 8 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
        pressure_daily = 2 * np.cos(2 * np.pi * (hour - 10) / 24)
        df.loc[i, "pressure"] = (
            1013 + pressure_seasonal + pressure_daily + np.random.normal(0, 5)
        )

        # Solar radiation (seasonal and daily cycles)
        if 6 <= hour <= 18:  # Daylight hours
            max_solar = 800 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal
            daily_solar = max_solar * np.sin(np.pi * (hour - 6) / 12)  # Daily curve
            cloud_factor = 1 - (df.loc[i, "humidity"] - 40) / 100
            df.loc[i, "solar_radiation"] = np.clip(
                daily_solar * np.clip(cloud_factor, 0.1, 1) + np.random.normal(0, 50),
                0,
                1000,
            )
        else:
            df.loc[i, "solar_radiation"] = 0

        # Precipitation (higher probability in winter, stochastic)
        precip_prob = 0.1 + 0.05 * np.cos(
            2 * np.pi * (day_of_year - 30) / 365
        )  # Winter peak
        if np.random.random() < precip_prob:
            df.loc[i, "precipitation"] = np.random.exponential(2.0)
        else:
            df.loc[i, "precipitation"] = 0

        # Boundary layer height (temperature and wind dependent)
        temp_effect = (df.loc[i, "temperature"] - 10) * 25
        wind_effect = df.loc[i, "wind_speed"] * 40
        time_effect = (
            200 * np.sin(2 * np.pi * (hour - 12) / 24)
            if hour >= 6 and hour <= 18
            else -100
        )
        df.loc[i, "boundary_layer_height"] = np.clip(
            600 + temp_effect + wind_effect + time_effect + np.random.normal(0, 100),
            100,
            3000,
        )

    log.info("Added meteorological patterns with hourly and seasonal variations")
    return df


def add_pollutant_concentrations(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic pollutant concentrations with complex temporal patterns."""
    log.info("Adding pollutant concentrations...")

    df = df.copy()
    np.random.seed(43)

    for i, row in df.iterrows():
        city = row["city"]
        hour = row["hour"]
        dayofweek = row["dayofweek"]
        month = row["month"]
        day_of_year = row["dayofyear"]

        city_char = CITY_CHARACTERISTICS[city]

        # Base pollution levels by city
        pollution_baselines = {
            "Berlin": {"pm25": 12, "pm10": 20, "no2": 25, "o3": 35},
            "Hamburg": {"pm25": 11, "pm10": 18, "no2": 22, "o3": 33},
            "Munich": {"pm25": 10, "pm10": 17, "no2": 20, "o3": 37},
        }

        base_levels = pollution_baselines[city]

        # Traffic-related patterns (NO2, PM) - rush hours
        if dayofweek < 5:  # Weekdays
            traffic_factor = 1.0
            # Morning rush (7-9 AM)
            if 7 <= hour <= 9:
                traffic_factor = 1.6
            # Evening rush (5-7 PM)
            elif 17 <= hour <= 19:
                traffic_factor = 1.4
            # Midday moderate
            elif 11 <= hour <= 14:
                traffic_factor = 1.2
            # Night time low
            elif hour < 6 or hour > 22:
                traffic_factor = 0.6
        else:  # Weekends
            traffic_factor = 0.7
            if 10 <= hour <= 16:  # Weekend activity
                traffic_factor = 0.9

        # Seasonal variations
        # PM higher in winter (heating), lower in summer (less combustion)
        pm_seasonal = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 330) / 365)
        # NO2 higher in winter (more combustion, less photolysis)
        no2_seasonal = 1 + 0.4 * np.cos(2 * np.pi * (day_of_year - 330) / 365)
        # O3 higher in summer (photochemical production)
        o3_seasonal = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Meteorological effects
        temp = row["temperature"]
        wind_speed = row["wind_speed"]
        boundary_layer_height = row["boundary_layer_height"]
        solar_radiation = row["solar_radiation"]

        # Dispersion factor (higher wind and BLH = lower concentrations)
        dispersion_factor = 1.5 / (1 + wind_speed * boundary_layer_height / 1000)

        # Temperature effects on chemistry
        temp_factor_o3 = 1 + 0.02 * (temp - 15)  # Higher temp = more O3
        temp_factor_pm = 1 - 0.01 * (
            temp - 15
        )  # Higher temp = less PM (volatilization)

        # Photochemical effects
        photo_factor = 1 + solar_radiation / 1000 * 0.3

        # Calculate concentrations
        # PM2.5
        pm25_base = base_levels["pm25"]
        pm25_traffic = pm25_base * traffic_factor * city_char["traffic_base"]
        pm25_seasonal = pm25_traffic * pm_seasonal
        pm25_met = pm25_seasonal * dispersion_factor * temp_factor_pm
        df.loc[i, "actual_pm25"] = np.clip(pm25_met + np.random.normal(0, 2), 1, 100)

        # PM10
        pm10_base = base_levels["pm10"]
        pm10_traffic = pm10_base * traffic_factor * city_char["traffic_base"]
        pm10_seasonal = pm10_traffic * pm_seasonal
        pm10_met = pm10_seasonal * dispersion_factor * temp_factor_pm
        df.loc[i, "actual_pm10"] = np.clip(pm10_met + np.random.normal(0, 3), 2, 150)

        # NO2
        no2_base = base_levels["no2"]
        no2_traffic = no2_base * traffic_factor * city_char["traffic_base"]
        no2_seasonal = no2_traffic * no2_seasonal
        no2_met = no2_seasonal * dispersion_factor
        # NO2 decreases with sunlight (photolysis)
        no2_photo = no2_met * (1 - solar_radiation / 1000 * 0.2)
        df.loc[i, "actual_no2"] = np.clip(no2_photo + np.random.normal(0, 3), 2, 120)

        # O3
        o3_base = base_levels["o3"]
        # O3 is inversely related to NO2 in urban areas (titration)
        no2_titration = 1 - (df.loc[i, "actual_no2"] - 10) / 100 * 0.3
        o3_seasonal = o3_base * o3_seasonal * np.clip(no2_titration, 0.3, 1.5)
        o3_temp = o3_seasonal * temp_factor_o3
        o3_photo = o3_temp * photo_factor
        # O3 peaks in afternoon
        o3_daily = (
            o3_photo * (1 + 0.3 * np.sin(2 * np.pi * (hour - 14) / 24))
            if 8 <= hour <= 20
            else o3_photo * 0.8
        )
        df.loc[i, "actual_o3"] = np.clip(o3_daily + np.random.normal(0, 4), 5, 200)

    log.info("Added realistic pollutant concentrations with temporal patterns")
    return df


def add_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic forecast data from CAMS and NOAA with realistic errors."""
    log.info("Adding forecast data with realistic model errors...")

    df = df.copy()
    np.random.seed(44)

    pollutants = ["pm25", "pm10", "no2", "o3"]

    # Model characteristics
    model_bias = {
        "cams": {"pm25": 0.02, "pm10": -0.05, "no2": 0.08, "o3": -0.03},
        "noaa_gefs_aerosol": {"pm25": -0.01, "pm10": 0.03, "no2": -0.06, "o3": 0.02},
    }

    model_rmse = {
        "cams": {"pm25": 0.15, "pm10": 0.18, "no2": 0.22, "o3": 0.12},
        "noaa_gefs_aerosol": {"pm25": 0.17, "pm10": 0.16, "no2": 0.20, "o3": 0.14},
    }

    for provider in ["cams", "noaa_gefs_aerosol"]:
        for pollutant in pollutants:
            actual_col = f"actual_{pollutant}"
            forecast_col = f"forecast_{provider}_{pollutant}"

            # Base forecast = actual + bias + random error
            bias = model_bias[provider][pollutant]
            rmse = model_rmse[provider][pollutant]

            # Add systematic bias
            forecast_base = df[actual_col] * (1 + bias)

            # Add random error with temporal correlation
            n_rows = len(df)
            random_errors = np.random.normal(0, rmse * df[actual_col].mean(), n_rows)

            # Add some temporal autocorrelation to errors
            for i in range(1, n_rows):
                if df.iloc[i]["city"] == df.iloc[i - 1]["city"]:  # Same city
                    random_errors[i] = (
                        0.3 * random_errors[i - 1] + 0.7 * random_errors[i]
                    )

            df[forecast_col] = forecast_base + random_errors

            # Ensure non-negative values
            df[forecast_col] = np.clip(df[forecast_col], 0.5, None)

    # Add forecast metadata
    df["forecast_made_date"] = (df["datetime"] - timedelta(hours=24)).dt.strftime(
        "%Y-%m-%d"
    )
    df["forecast_lead_hours"] = 24

    log.info("Added forecast data for CAMS and NOAA with realistic model errors")
    return df


def add_external_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic external data features (weather, fire, infrastructure, etc.)."""
    log.info("Adding external data features...")

    df = df.copy()
    np.random.seed(45)

    for i, row in df.iterrows():
        city = row["city"]
        hour = row["hour"]
        dayofweek = row["dayofweek"]
        month = row["month"]
        temp = row["temperature"]

        city_char = CITY_CHARACTERISTICS[city]

        # Fire activity (seasonal, random events)
        fire_season_factor = 1 + 0.5 * np.sin(
            2 * np.pi * (row["dayofyear"] - 150) / 365
        )  # Peak in summer
        fire_prob = 0.02 * fire_season_factor
        if np.random.random() < fire_prob:
            df.loc[i, "fire_fire_count"] = np.random.poisson(2)
            df.loc[i, "fire_total_frp"] = df.loc[
                i, "fire_fire_count"
            ] * np.random.exponential(50)
            df.loc[i, "fire_avg_distance"] = np.random.uniform(10, 100)
        else:
            df.loc[i, "fire_fire_count"] = 0
            df.loc[i, "fire_total_frp"] = 0
            df.loc[i, "fire_avg_distance"] = 0

        # Construction activity (seasonal, city-dependent)
        construction_base = city_char["urban_intensity"] * 20
        construction_seasonal = 1 + 0.3 * np.sin(
            2 * np.pi * (row["dayofyear"] - 120) / 365
        )  # Peak in summer
        if dayofweek < 5 and 7 <= hour <= 17:  # Working hours
            construction_factor = 1.0
        else:
            construction_factor = 0.1
        df.loc[i, "construction_site_count"] = (
            construction_base * construction_seasonal * construction_factor
            + np.random.poisson(2)
        )

        # Infrastructure (static by city)
        df.loc[i, "infrastructure_major_roads"] = city_char[
            "traffic_base"
        ] * 50 + np.random.poisson(5)
        df.loc[i, "infrastructure_railways"] = city_char[
            "urban_intensity"
        ] * 10 + np.random.poisson(2)
        df.loc[i, "infrastructure_industrial_areas"] = city_char[
            "industrial_score"
        ] * 15 + np.random.poisson(3)

        # Earthquake activity (rare, random)
        if np.random.random() < 0.001:  # Very low probability
            df.loc[i, "earthquake_count"] = 1
            df.loc[i, "max_earthquake_magnitude"] = np.random.uniform(2, 5)
        else:
            df.loc[i, "earthquake_count"] = 0
            df.loc[i, "max_earthquake_magnitude"] = 0

        # Holiday flags (simplified)
        # Major holidays
        major_holidays = [
            "01-01",
            "05-01",
            "10-03",
            "12-25",
            "12-26",
        ]  # New Year, Labor Day, Unity Day, Christmas
        date_str = f"{month:02d}-{row['day']:02d}"
        df.loc[i, "is_public_holiday"] = 1 if date_str in major_holidays else 0

        # School holidays (summer: July-August, winter: late Dec-early Jan)
        school_holiday = 0
        if (
            month in [7, 8]
            or (month == 12 and row["day"] > 20)
            or (month == 1 and row["day"] < 10)
        ):
            school_holiday = 1
        df.loc[i, "is_school_holiday"] = school_holiday

        # Traffic patterns
        traffic_base = city_char["traffic_base"]
        if dayofweek < 5:  # Weekdays
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                traffic_intensity = traffic_base * 1.6
            elif 9 <= hour <= 17:  # Business hours
                traffic_intensity = traffic_base * 1.2
            else:  # Off hours
                traffic_intensity = traffic_base * 0.6
        else:  # Weekends
            traffic_intensity = traffic_base * 0.7

        df.loc[i, "traffic_intensity"] = traffic_intensity + np.random.normal(0, 0.1)

        # Economic activity
        economic_base = 0.8
        if dayofweek < 5 and 8 <= hour <= 18:  # Business hours
            economic_activity = economic_base
        elif dayofweek == 5:  # Friday
            economic_activity = economic_base * 0.9
        elif dayofweek == 6:  # Saturday
            economic_activity = economic_base * 0.6
        else:  # Sunday
            economic_activity = economic_base * 0.3

        df.loc[i, "economic_activity"] = economic_activity + np.random.normal(0, 0.05)

    log.info("Added external data features with realistic patterns")
    return df


def generate_3year_hourly_dataset(output_path: Path) -> pd.DataFrame:
    """Generate complete 3-year hourly dataset."""
    log.info("Generating comprehensive 3-year hourly dataset...")

    # Generate base time series (2022-2025)
    df = generate_base_time_series("2022-01-01", "2024-12-31", "H")

    # Add meteorological patterns
    df = add_meteorological_patterns(df)

    # Add pollutant concentrations
    df = add_pollutant_concentrations(df)

    # Add forecast data
    df = add_forecast_data(df)

    # Add external data features
    df = add_external_data_features(df)

    # Clean up and organize columns
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df.to_parquet(output_path.with_suffix(".parquet"), index=False)

    log.info(f"Generated 3-year hourly dataset: {df.shape}")
    log.info(f"Saved to {output_path} and {output_path.with_suffix('.parquet')}")
    log.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-year hourly air quality forecasting dataset"
    )
    parser.add_argument(
        "--output",
        default="data/analysis/3year_hourly_dataset.csv",
        help="Output path for the dataset",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Generate dataset
    df = generate_3year_hourly_dataset(output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("3-YEAR HOURLY DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print(f"Total hours: {len(df['datetime'].unique()):,}")
    print(f"File size (CSV): {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(
        f"File size (Parquet): {output_path.with_suffix('.parquet').stat().st_size / 1024 / 1024:.1f} MB"
    )

    # Show sample data
    print(f"\nSample data (first 5 rows):")
    print(
        df[
            [
                "city",
                "datetime",
                "actual_pm25",
                "forecast_cams_pm25",
                "temperature",
                "traffic_intensity",
            ]
        ].head()
    )

    return 0


if __name__ == "__main__":
    exit(main())
