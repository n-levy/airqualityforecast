#!/usr/bin/env python3
"""
Generate 5-Year Hourly Air Quality Forecasting Dataset

Creates a comprehensive synthetic dataset spanning 5+ years (2020-01-01 to 2025-09-08) with hourly frequency.
Includes realistic temporal patterns, seasonal variations, weather effects, and pollutant relationships.
Extended from 3-year version to provide more robust validation data for improved validation strategies.
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


def generate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic weather data with seasonal patterns."""
    log.info("Generating weather data...")

    # Set random seed for reproducibility
    np.random.seed(42)

    for city in df["city"].unique():
        city_mask = df["city"] == city
        city_data = df[city_mask].copy()

        # Base temperature with seasonal variation
        day_of_year = city_data["dayofyear"]
        hour = city_data["hour"]

        # Seasonal temperature pattern (sine wave with peak in summer)
        seasonal_temp = 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)

        # Daily temperature variation (sine wave with peak in afternoon)
        daily_temp = 8 * np.sin((hour - 6) * 2 * np.pi / 24)

        # Base temperature by city (latitude effect)
        base_temp = {
            "Munich": 8,  # Southern, slightly warmer
            "Berlin": 6,  # Central
            "Hamburg": 4,  # Northern, coastal, cooler
        }[city]

        # Add random variation and year-to-year differences
        random_temp = np.random.normal(0, 2, len(city_data))
        year_effect = (city_data["year"] - 2020) * 0.3  # Slight warming trend

        temperature = base_temp + seasonal_temp + daily_temp + random_temp + year_effect

        # Humidity (inverse correlation with temperature, higher in winter)
        base_humidity = 70 - temperature * 0.8 + np.random.normal(0, 5, len(city_data))
        humidity = np.clip(base_humidity, 20, 95)

        # Wind speed with seasonal and random components
        seasonal_wind = 2 * np.sin((day_of_year - 60) * 2 * np.pi / 365) + 3
        wind_speed = seasonal_wind + np.random.exponential(2, len(city_data))
        wind_speed = np.clip(wind_speed, 0.1, 25)

        # Wind direction (more westerly in winter)
        wind_direction = (
            180
            + 90 * np.sin((day_of_year - 60) * 2 * np.pi / 365)
            + np.random.normal(0, 45, len(city_data))
        ) % 360

        # Pressure with weather system variations
        base_pressure = 1013 + np.random.normal(0, 15, len(city_data))
        pressure = np.clip(base_pressure, 980, 1040)

        # Solar radiation (dependent on time of day and season)
        solar_elevation = np.maximum(
            0,
            np.sin((hour - 12) * np.pi / 12)
            * (0.5 + 0.5 * np.sin((day_of_year - 172) * 2 * np.pi / 365)),
        )
        solar_radiation = (
            solar_elevation * 800 * (1 + np.random.normal(0, 0.2, len(city_data)))
        )
        solar_radiation = np.clip(solar_radiation, 0, 1000)

        # Precipitation (more in winter, random events)
        precip_prob = 0.15 + 0.1 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        precipitation = np.where(
            np.random.random(len(city_data)) < precip_prob,
            np.random.exponential(3, len(city_data)),
            0,
        )

        # Boundary layer height (higher in summer, during day)
        blh_seasonal = 800 + 400 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        blh_diurnal = 200 * np.sin((hour - 6) * 2 * np.pi / 24)
        boundary_layer_height = (
            blh_seasonal + blh_diurnal + np.random.normal(0, 100, len(city_data))
        )
        boundary_layer_height = np.clip(boundary_layer_height, 100, 2000)

        # Assign to dataframe
        df.loc[city_mask, "temperature"] = temperature
        df.loc[city_mask, "humidity"] = humidity
        df.loc[city_mask, "wind_speed"] = wind_speed
        df.loc[city_mask, "wind_direction"] = wind_direction
        df.loc[city_mask, "pressure"] = pressure
        df.loc[city_mask, "solar_radiation"] = solar_radiation
        df.loc[city_mask, "precipitation"] = precipitation
        df.loc[city_mask, "boundary_layer_height"] = boundary_layer_height

    log.info("Weather data generated successfully")
    return df


def generate_traffic_and_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Generate traffic patterns and activity indicators."""
    log.info("Generating traffic and activity data...")

    # Traffic intensity based on hour, day of week, and city
    for city in df["city"].unique():
        city_mask = df["city"] == city
        city_data = df[city_mask].copy()

        hour = city_data["hour"]
        dayofweek = city_data["dayofweek"]

        # Base traffic pattern (rush hours)
        morning_rush = np.exp(-((hour - 8) ** 2) / 8) * 0.8
        evening_rush = np.exp(-((hour - 18) ** 2) / 8) * 0.9
        base_traffic = 0.3 + morning_rush + evening_rush

        # Weekend reduction
        weekend_factor = np.where(dayofweek >= 5, 0.6, 1.0)

        # City-specific multiplier
        city_multiplier = CITY_CHARACTERISTICS[city]["traffic_base"]

        traffic_intensity = base_traffic * weekend_factor * city_multiplier
        traffic_intensity += np.random.normal(0, 0.1, len(city_data))
        traffic_intensity = np.clip(traffic_intensity, 0, 2)

        # Public holidays (simplified - major holidays)
        is_public_holiday = (
            ((city_data["month"] == 1) & (city_data["day"] == 1))  # New Year
            | ((city_data["month"] == 12) & (city_data["day"] == 25))  # Christmas
            | ((city_data["month"] == 5) & (city_data["day"] == 1))  # Labor Day
        ).astype(int)

        # School holidays (simplified - summer break)
        is_school_holiday = (
            (city_data["month"] >= 7) & (city_data["month"] <= 8)
        ).astype(int)

        df.loc[city_mask, "traffic_intensity"] = traffic_intensity
        df.loc[city_mask, "is_public_holiday"] = is_public_holiday
        df.loc[city_mask, "is_school_holiday"] = is_school_holiday

    log.info("Traffic and activity data generated successfully")
    return df


def generate_external_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Generate external factors like fire activity, construction, economic indicators."""
    log.info("Generating external factors...")

    np.random.seed(123)  # Different seed for external factors

    for city in df["city"].unique():
        city_mask = df["city"] == city
        n_records = city_mask.sum()

        # Fire activity (higher in summer, drought conditions)
        fire_base = 0.1 * (
            1 + 0.5 * np.sin((df.loc[city_mask, "dayofyear"] - 172) * 2 * np.pi / 365)
        )
        fire_random = np.random.exponential(0.05, n_records)
        fire_activity = fire_base + fire_random
        fire_activity = np.clip(fire_activity, 0, 1)

        # Construction activity (lower in winter, business hours)
        construction_seasonal = 0.7 + 0.3 * np.sin(
            (df.loc[city_mask, "dayofyear"] - 100) * 2 * np.pi / 365
        )
        construction_hourly = np.where(
            (df.loc[city_mask, "hour"] >= 7)
            & (df.loc[city_mask, "hour"] <= 18)
            & (df.loc[city_mask, "dayofweek"] < 5),
            1.0,
            0.2,
        )
        construction_activity = construction_seasonal * construction_hourly * 0.3
        construction_activity += np.random.normal(0, 0.1, n_records)
        construction_activity = np.clip(construction_activity, 0, 1)

        # Economic activity (higher during business hours, weekdays)
        economic_hourly = np.where(
            (df.loc[city_mask, "hour"] >= 8) & (df.loc[city_mask, "hour"] <= 20),
            0.8,
            0.3,
        )
        economic_weekly = np.where(df.loc[city_mask, "dayofweek"] < 5, 1.0, 0.4)
        # COVID effect (reduced activity in 2020-2021)
        covid_effect = np.where(df.loc[city_mask, "year"] <= 2021, 0.7, 1.0)
        economic_activity = economic_hourly * economic_weekly * covid_effect * 0.8
        economic_activity += np.random.normal(0, 0.1, n_records)
        economic_activity = np.clip(economic_activity, 0, 1)

        df.loc[city_mask, "fire_activity"] = fire_activity
        df.loc[city_mask, "construction_activity"] = construction_activity
        df.loc[city_mask, "economic_activity"] = economic_activity

    log.info("External factors generated successfully")
    return df


def generate_pollutant_concentrations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic air pollutant concentrations."""
    log.info("Generating pollutant concentrations...")

    for city in df["city"].unique():
        city_mask = df["city"] == city
        city_data = df[city_mask].copy()

        # Get city characteristics
        chars = CITY_CHARACTERISTICS[city]

        # Base pollution levels by city
        base_pm25 = chars["urban_intensity"] * 8 + chars["industrial_score"] * 3
        base_pm10 = base_pm25 * 1.8 + chars["industrial_score"] * 2
        base_no2 = chars["traffic_base"] * 15 + chars["urban_intensity"] * 8
        base_o3 = 25 + chars["urban_intensity"] * 5  # Background O3

        # Seasonal variations
        day_of_year = city_data["dayofyear"]

        # PM2.5: higher in winter (heating), inversely related to mixing height
        pm25_seasonal = -3 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        pm25_meteorological = (
            -city_data["boundary_layer_height"] / 200 + city_data["humidity"] / 20
        )
        pm25_traffic = city_data["traffic_intensity"] * 2
        pm25_construction = city_data["construction_activity"] * 1.5

        pm25 = (
            base_pm25
            + pm25_seasonal
            + pm25_meteorological
            + pm25_traffic
            + pm25_construction
            + np.random.normal(0, 1.5, len(city_data))
        )
        pm25 = np.clip(pm25, 0.5, 50)

        # PM10: similar to PM2.5 but higher baseline
        pm10_seasonal = -4 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        pm10_meteorological = (
            -city_data["boundary_layer_height"] / 150 + city_data["humidity"] / 25
        )
        pm10_traffic = city_data["traffic_intensity"] * 3
        pm10_construction = city_data["construction_activity"] * 3

        pm10 = (
            base_pm10
            + pm10_seasonal
            + pm10_meteorological
            + pm10_traffic
            + pm10_construction
            + np.random.normal(0, 2, len(city_data))
        )
        pm10 = np.clip(pm10, 1, 80)

        # NO2: traffic-dominated, higher in winter
        no2_seasonal = 5 * np.sin(
            (day_of_year - 60) * 2 * np.pi / 365
        )  # Higher in winter
        no2_meteorological = (
            -city_data["boundary_layer_height"] / 100 - city_data["wind_speed"] / 2
        )
        no2_traffic = city_data["traffic_intensity"] * 8
        no2_economic = city_data["economic_activity"] * 3

        no2 = (
            base_no2
            + no2_seasonal
            + no2_meteorological
            + no2_traffic
            + no2_economic
            + np.random.normal(0, 2, len(city_data))
        )
        no2 = np.clip(no2, 1, 100)

        # O3: photochemically produced, higher in summer
        o3_seasonal = 15 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        o3_diurnal = 10 * np.sin((city_data["hour"] - 6) * 2 * np.pi / 24)
        o3_solar = city_data["solar_radiation"] / 50
        o3_temperature = city_data["temperature"] / 5
        # Inverse NO2 relationship (NOx titration)
        o3_nox_titration = -no2 / 10

        o3 = (
            base_o3
            + o3_seasonal
            + o3_diurnal
            + o3_solar
            + o3_temperature
            + o3_nox_titration
            + np.random.normal(0, 3, len(city_data))
        )
        o3 = np.clip(o3, 5, 120)

        # Assign actual concentrations
        df.loc[city_mask, "actual_pm25"] = pm25
        df.loc[city_mask, "actual_pm10"] = pm10
        df.loc[city_mask, "actual_no2"] = no2
        df.loc[city_mask, "actual_o3"] = o3

    log.info("Pollutant concentrations generated successfully")
    return df


def generate_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate forecast data from CAMS and NOAA GEFS-Aerosol with realistic errors."""
    log.info("Generating forecast data...")

    # Set forecast made date (24 hours before)
    df["forecast_made_date"] = (
        pd.to_datetime(df["datetime"]) - timedelta(hours=24)
    ).dt.strftime("%Y-%m-%d")
    df["forecast_lead_hours"] = 24

    # CAMS forecasts (generally good for European cities)
    for pollutant in ["pm25", "pm10", "no2", "o3"]:
        actual_col = f"actual_{pollutant}"
        forecast_col = f"forecast_cams_{pollutant}"

        # CAMS bias and errors
        if pollutant == "pm25":
            bias = 0.2  # Slight positive bias
            random_error = 0.15
        elif pollutant == "pm10":
            bias = -0.1  # Slight negative bias
            random_error = 0.18
        elif pollutant == "no2":
            bias = 0.3  # Positive bias in urban areas
            random_error = 0.20
        else:  # o3
            bias = -0.05  # Slight negative bias
            random_error = 0.12

        # Generate forecast with bias and random error
        cams_forecast = df[actual_col] * (1 + bias) + np.random.normal(
            0, df[actual_col] * random_error, len(df)
        )
        df[forecast_col] = np.maximum(cams_forecast, 0.1)  # Ensure positive values

    # NOAA GEFS-Aerosol forecasts (more variable, different biases)
    for pollutant in ["pm25", "pm10", "no2", "o3"]:
        actual_col = f"actual_{pollutant}"
        forecast_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

        # NOAA GEFS-Aerosol bias and errors
        if pollutant == "pm25":
            bias = -0.1  # Slight negative bias
            random_error = 0.20
        elif pollutant == "pm10":
            bias = 0.15  # Positive bias
            random_error = 0.22
        elif pollutant == "no2":
            bias = -0.2  # Negative bias
            random_error = 0.25
        else:  # o3
            bias = 0.1  # Positive bias
            random_error = 0.15

        # Generate forecast with bias and random error
        noaa_forecast = df[actual_col] * (1 + bias) + np.random.normal(
            0, df[actual_col] * random_error, len(df)
        )
        df[forecast_col] = np.maximum(noaa_forecast, 0.1)  # Ensure positive values

    log.info("Forecast data generated successfully")
    return df


def add_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive engineered features for enhanced modeling."""
    log.info("Adding comprehensive engineered features...")

    # Temporal features
    df["temp_squared"] = df["temperature"] ** 2
    df["wind_u"] = df["wind_speed"] * np.cos(np.radians(df["wind_direction"]))
    df["wind_v"] = df["wind_speed"] * np.sin(np.radians(df["wind_direction"]))
    df["ventilation_index"] = df["wind_speed"] * df["boundary_layer_height"]
    df["stability_indicator"] = df["temperature"] / (df["boundary_layer_height"] + 1)

    # Traffic patterns
    df["simulated_morning_rush"] = np.exp(-((df["hour"] - 8) ** 2) / 8)
    df["simulated_evening_rush"] = np.exp(-((df["hour"] - 18) ** 2) / 8)
    df["simulated_midday_low"] = np.exp(-((df["hour"] - 14) ** 2) / 16) * 0.5

    # Calendar features
    df["day_of_month"] = df["day"]
    df["week_of_year"] = pd.to_datetime(df["datetime"]).dt.isocalendar().week

    # Add more temporal features
    df["days_to_weekend"] = np.where(df["dayofweek"] < 5, 5 - df["dayofweek"], 0)
    df["days_from_weekend"] = np.where(df["dayofweek"] > 0, df["dayofweek"], 0)
    df["week_position"] = (
        "weekend"
        if pd.to_datetime(df["datetime"]).iloc[0].dayofweek >= 5
        else "weekday"
    )

    # Trigonometric encoding
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    # Pollutant cross-relationships for CAMS
    df["cams_pm25_pm10_ratio"] = df["forecast_cams_pm25"] / (
        df["forecast_cams_pm10"] + 0.1
    )
    df["cams_coarse_pm"] = df["forecast_cams_pm10"] - df["forecast_cams_pm25"]
    df["cams_no2_o3_ratio"] = df["forecast_cams_no2"] / (df["forecast_cams_o3"] + 0.1)
    df["cams_photochem_potential"] = df["forecast_cams_o3"] / (
        df["forecast_cams_no2"] + 0.1
    )
    df["cams_secondary_aerosol_proxy"] = (
        df["forecast_cams_pm25"] - df["forecast_cams_no2"] * 0.1
    )
    df["cams_total_load"] = (
        df["forecast_cams_pm25"] + df["forecast_cams_pm10"] + df["forecast_cams_no2"]
    )
    df["cams_fresh_emissions"] = (
        df["forecast_cams_no2"] + df["forecast_cams_pm10"] * 0.5
    )
    df["cams_aged_air"] = df["forecast_cams_o3"] + df["forecast_cams_pm25"]

    # Pollutant cross-relationships for NOAA GEFS-Aerosol
    df["noaa_gefs_aerosol_pm25_pm10_ratio"] = df["forecast_noaa_gefs_aerosol_pm25"] / (
        df["forecast_noaa_gefs_aerosol_pm10"] + 0.1
    )
    df["noaa_gefs_aerosol_coarse_pm"] = (
        df["forecast_noaa_gefs_aerosol_pm10"] - df["forecast_noaa_gefs_aerosol_pm25"]
    )
    df["noaa_gefs_aerosol_no2_o3_ratio"] = df["forecast_noaa_gefs_aerosol_no2"] / (
        df["forecast_noaa_gefs_aerosol_o3"] + 0.1
    )
    df["noaa_gefs_aerosol_photochem_potential"] = df[
        "forecast_noaa_gefs_aerosol_o3"
    ] / (df["forecast_noaa_gefs_aerosol_no2"] + 0.1)
    df["noaa_gefs_aerosol_secondary_aerosol_proxy"] = (
        df["forecast_noaa_gefs_aerosol_pm25"]
        - df["forecast_noaa_gefs_aerosol_no2"] * 0.1
    )
    df["noaa_gefs_aerosol_total_load"] = (
        df["forecast_noaa_gefs_aerosol_pm25"]
        + df["forecast_noaa_gefs_aerosol_pm10"]
        + df["forecast_noaa_gefs_aerosol_no2"]
    )
    df["noaa_gefs_aerosol_fresh_emissions"] = (
        df["forecast_noaa_gefs_aerosol_no2"]
        + df["forecast_noaa_gefs_aerosol_pm10"] * 0.5
    )
    df["noaa_gefs_aerosol_aged_air"] = (
        df["forecast_noaa_gefs_aerosol_o3"] + df["forecast_noaa_gefs_aerosol_pm25"]
    )

    # Cross-provider comparisons
    df["cross_provider_pm25_agreement"] = 1 - abs(
        df["forecast_cams_pm25"] - df["forecast_noaa_gefs_aerosol_pm25"]
    ) / (df["forecast_cams_pm25"] + df["forecast_noaa_gefs_aerosol_pm25"] + 0.1)
    df["cross_provider_pm10_agreement"] = 1 - abs(
        df["forecast_cams_pm10"] - df["forecast_noaa_gefs_aerosol_pm10"]
    ) / (df["forecast_cams_pm10"] + df["forecast_noaa_gefs_aerosol_pm10"] + 0.1)
    df["cross_provider_no2_agreement"] = 1 - abs(
        df["forecast_cams_no2"] - df["forecast_noaa_gefs_aerosol_no2"]
    ) / (df["forecast_cams_no2"] + df["forecast_noaa_gefs_aerosol_no2"] + 0.1)
    df["cross_provider_o3_agreement"] = 1 - abs(
        df["forecast_cams_o3"] - df["forecast_noaa_gefs_aerosol_o3"]
    ) / (df["forecast_cams_o3"] + df["forecast_noaa_gefs_aerosol_o3"] + 0.1)

    # Meteorological interactions
    df["temp_o3_interaction"] = df["temperature"] * df["forecast_cams_o3"] / 100
    df["solar_no2_interaction"] = df["solar_radiation"] * df["forecast_cams_no2"] / 1000
    df["biogenic_emission_proxy"] = df["temperature"] * df["solar_radiation"] / 100

    # Add city-specific features
    for city in df["city"].unique():
        city_mask = df["city"] == city
        df.loc[city_mask, f"city_{city}_temp_interaction"] = (
            df.loc[city_mask, "temperature"]
            * CITY_CHARACTERISTICS[city]["urban_intensity"]
        )

    log.info("Comprehensive features added successfully")
    return df


def main():
    """Main function to generate the 5-year dataset."""
    parser = argparse.ArgumentParser(
        description="Generate 5-year hourly air quality dataset"
    )
    parser.add_argument(
        "--output",
        default="data/analysis/5year_hourly_comprehensive_dataset.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2025-09-08", help="End date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting 5-year dataset generation...")

    # Generate base time series
    df = generate_base_time_series(args.start_date, args.end_date)

    # Add weather data
    df = generate_weather_data(df)

    # Add traffic and activities
    df = generate_traffic_and_activities(df)

    # Add external factors
    df = generate_external_factors(df)

    # Generate pollutant concentrations
    df = generate_pollutant_concentrations(df)

    # Generate forecast data
    df = generate_forecast_data(df)

    # Add comprehensive features
    df = add_comprehensive_features(df)

    # Save dataset
    log.info(f"Saving dataset to {output_path}...")
    df.to_csv(output_path, index=False)

    # Print summary
    log.info("Dataset generation completed!")
    log.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    log.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    log.info(f"Cities: {', '.join(df['city'].unique())}")
    log.info(f"Output saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
