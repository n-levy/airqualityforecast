#!/usr/bin/env python3
"""
Generate Complete 3-Year Hourly Air Quality Dataset
Simple, efficient approach for large-scale dataset generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def generate_3year_dataset():
    """Generate complete 3-year hourly dataset efficiently."""
    log.info("Generating 3-year hourly dataset...")

    # Generate full date range
    date_range = pd.date_range("2022-01-01", "2024-12-31", freq="h")
    cities = ["Berlin", "Hamburg", "Munich"]

    log.info(f"Date range: {len(date_range):,} hours, {len(cities)} cities")
    log.info(f"Total rows: {len(date_range) * len(cities):,}")

    # City characteristics
    city_params = {
        "Berlin": {
            "pm25_base": 12,
            "pm10_base": 20,
            "no2_base": 25,
            "o3_base": 35,
            "traffic_base": 0.8,
            "temp_base": 12,
        },
        "Hamburg": {
            "pm25_base": 11,
            "pm10_base": 18,
            "no2_base": 22,
            "o3_base": 33,
            "traffic_base": 0.7,
            "temp_base": 10,
        },
        "Munich": {
            "pm25_base": 10,
            "pm10_base": 17,
            "no2_base": 20,
            "o3_base": 37,
            "traffic_base": 0.75,
            "temp_base": 13,
        },
    }

    np.random.seed(42)

    all_data = []

    for city in cities:
        log.info(f"Processing {city}...")
        params = city_params[city]

        city_data = []
        for i, dt in enumerate(date_range):
            if i % 10000 == 0:
                log.info(
                    f"  Progress: {i:,}/{len(date_range):,} ({i/len(date_range)*100:.1f}%)"
                )

            hour = dt.hour
            day_of_year = dt.timetuple().tm_yday
            day_of_week = dt.weekday()

            # Traffic patterns
            if day_of_week < 5:  # Weekdays
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                    traffic_factor = 1.6
                elif 9 <= hour <= 17:  # Business hours
                    traffic_factor = 1.2
                else:  # Off hours
                    traffic_factor = 0.6
            else:  # Weekends
                traffic_factor = 0.8 if 10 <= hour <= 16 else 0.6

            # Weather patterns
            # Temperature: seasonal + daily + noise
            seasonal_temp = 15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
            daily_temp = 8 * np.cos(2 * np.pi * (hour - 14) / 24)
            temp = (
                params["temp_base"]
                + seasonal_temp
                + daily_temp
                + np.random.normal(0, 3)
            )

            # Wind and other met
            wind_speed = max(
                0.5, 3 + 2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1.5)
            )
            humidity = np.clip(70 - (temp - 10) * 1.5 + np.random.normal(0, 8), 20, 95)

            # Pollutant concentrations
            # PM2.5
            pm25_seasonal = 1 + 0.3 * np.cos(
                2 * np.pi * (day_of_year - 330) / 365
            )  # Higher in winter
            pm25_base = (
                params["pm25_base"]
                * traffic_factor
                * params["traffic_base"]
                * pm25_seasonal
            )
            dispersion = 1.5 / (1 + wind_speed * 0.1)  # Simplified dispersion
            pm25 = max(1, pm25_base * dispersion + np.random.normal(0, 2))

            # PM10
            pm10 = max(2, pm25 * 1.7 + np.random.normal(0, 2))

            # NO2
            no2_seasonal = 1 + 0.4 * np.cos(2 * np.pi * (day_of_year - 330) / 365)
            no2_base = (
                params["no2_base"]
                * traffic_factor
                * params["traffic_base"]
                * no2_seasonal
            )
            no2 = max(2, no2_base * dispersion + np.random.normal(0, 3))

            # O3 (higher in summer, daily peak in afternoon)
            o3_seasonal = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            o3_daily = (
                1 + 0.3 * np.sin(2 * np.pi * (hour - 14) / 24)
                if 8 <= hour <= 20
                else 0.8
            )
            o3_temp_factor = 1 + 0.02 * (temp - 15)  # Higher with temperature
            o3 = max(
                5,
                params["o3_base"] * o3_seasonal * o3_daily * o3_temp_factor
                + np.random.normal(0, 4),
            )

            # Forecast data (add realistic model errors)
            cams_bias = {"pm25": 1.02, "pm10": 0.98, "no2": 1.05, "o3": 0.97}
            noaa_bias = {"pm25": 0.99, "pm10": 1.01, "no2": 0.96, "o3": 1.01}

            forecast_cams_pm25 = max(
                1, pm25 * cams_bias["pm25"] + np.random.normal(0, 1.5)
            )
            forecast_cams_pm10 = max(
                2, pm10 * cams_bias["pm10"] + np.random.normal(0, 2)
            )
            forecast_cams_no2 = max(
                2, no2 * cams_bias["no2"] + np.random.normal(0, 2.5)
            )
            forecast_cams_o3 = max(5, o3 * cams_bias["o3"] + np.random.normal(0, 3))

            forecast_noaa_pm25 = max(
                1, pm25 * noaa_bias["pm25"] + np.random.normal(0, 1.7)
            )
            forecast_noaa_pm10 = max(
                2, pm10 * noaa_bias["pm10"] + np.random.normal(0, 1.8)
            )
            forecast_noaa_no2 = max(
                2, no2 * noaa_bias["no2"] + np.random.normal(0, 2.2)
            )
            forecast_noaa_o3 = max(5, o3 * noaa_bias["o3"] + np.random.normal(0, 3.2))

            city_data.append(
                {
                    "city": city,
                    "datetime": dt,
                    "date": dt.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "dayofweek": day_of_week,
                    "dayofyear": day_of_year,
                    # Actuals
                    "actual_pm25": round(pm25, 2),
                    "actual_pm10": round(pm10, 2),
                    "actual_no2": round(no2, 2),
                    "actual_o3": round(o3, 2),
                    # CAMS forecasts
                    "forecast_cams_pm25": round(forecast_cams_pm25, 2),
                    "forecast_cams_pm10": round(forecast_cams_pm10, 2),
                    "forecast_cams_no2": round(forecast_cams_no2, 2),
                    "forecast_cams_o3": round(forecast_cams_o3, 2),
                    # NOAA forecasts
                    "forecast_noaa_gefs_aerosol_pm25": round(forecast_noaa_pm25, 2),
                    "forecast_noaa_gefs_aerosol_pm10": round(forecast_noaa_pm10, 2),
                    "forecast_noaa_gefs_aerosol_no2": round(forecast_noaa_no2, 2),
                    "forecast_noaa_gefs_aerosol_o3": round(forecast_noaa_o3, 2),
                    # Meteorology
                    "temperature": round(temp, 1),
                    "humidity": round(humidity, 1),
                    "wind_speed": round(wind_speed, 1),
                    "wind_direction": round(np.random.uniform(0, 360), 1),
                    "pressure": round(1013 + np.random.normal(0, 5), 1),
                    "solar_radiation": round(
                        max(
                            0,
                            500
                            * np.sin(2 * np.pi * (hour - 12) / 24)
                            * (1 if 6 <= hour <= 18 else 0),
                        ),
                        1,
                    ),
                    "precipitation": round(
                        max(
                            0,
                            np.random.exponential(1) if np.random.random() < 0.1 else 0,
                        ),
                        1,
                    ),
                    "boundary_layer_height": round(
                        max(100, 800 + (temp - 10) * 20 + wind_speed * 40), 1
                    ),
                    # External features
                    "traffic_intensity": round(
                        traffic_factor * params["traffic_base"]
                        + np.random.normal(0, 0.1),
                        2,
                    ),
                    "is_public_holiday": (
                        1
                        if (dt.month == 1 and dt.day == 1)
                        or (dt.month == 12 and dt.day == 25)
                        else 0
                    ),
                    "is_school_holiday": 1 if dt.month in [7, 8] else 0,
                    "fire_activity": round(
                        max(
                            0,
                            (
                                np.random.exponential(0.5)
                                if np.random.random() < 0.02
                                else 0
                            ),
                        ),
                        1,
                    ),
                    "construction_activity": round(
                        max(
                            0,
                            (
                                10 + np.random.poisson(5)
                                if day_of_week < 5 and 7 <= hour <= 17
                                else np.random.poisson(2)
                            ),
                        ),
                        1,
                    ),
                    "economic_activity": round(
                        0.8 if day_of_week < 5 and 8 <= hour <= 18 else 0.4, 2
                    ),
                    # Forecast metadata
                    "forecast_made_date": (dt - timedelta(hours=24)).strftime(
                        "%Y-%m-%d"
                    ),
                    "forecast_lead_hours": 24,
                }
            )

        all_data.extend(city_data)
        log.info(f"  {city} complete: {len(city_data):,} rows")

    df = pd.DataFrame(all_data)
    log.info(f"Dataset created: {df.shape}")

    return df


def main():
    output_path = Path("data/analysis/3year_hourly_complete_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    df = generate_3year_dataset()

    # Save
    log.info("Saving dataset...")
    df.to_csv(output_path, index=False)

    # Summary
    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print("\n" + "=" * 80)
    print("3-YEAR HOURLY DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Columns: {len(df.columns)}")

    # Sample data
    print(f"\nSample data:")
    sample_cols = [
        "city",
        "datetime",
        "actual_pm25",
        "forecast_cams_pm25",
        "temperature",
        "traffic_intensity",
    ]
    print(df[sample_cols].head(10))

    # Basic statistics
    print(f"\nPollutant statistics:")
    stats_cols = ["actual_pm25", "actual_pm10", "actual_no2", "actual_o3"]
    print(df[stats_cols].describe())

    return 0


if __name__ == "__main__":
    exit(main())
