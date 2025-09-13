#!/usr/bin/env python3
"""
Stage 6 ETL: Local Features Generation
======================================

Generates comprehensive local features including:
- Calendar and temporal features (cyclical encodings)
- Meteorological data (temperature, humidity, wind, pressure)
- Geographic and demographic features
- Holiday and special event indicators

Cross-platform implementation supporting Linux/macOS/Windows.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Cross-platform data root
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "aqf_data"))
OUTPUT_DIR = DATA_ROOT / "curated" / "stage6" / "local_features"


class LocalFeaturesETL:
    """ETL pipeline for local feature generation."""

    def __init__(self, cities_config: Optional[Dict] = None):
        """Initialize with cities configuration."""
        self.cities = cities_config or self.get_default_cities()
        self.setup_output_directory()

    def get_default_cities(self) -> Dict[str, Dict]:
        """Get Stage 5 cities configuration (100 cities, 20 per continent)."""
        return load_stage5_cities()

    def setup_output_directory(self):
        """Create output directory structure."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {OUTPUT_DIR}")

    def generate_calendar_features(self, timestamp: pd.Timestamp) -> Dict:
        """Generate comprehensive calendar and temporal features."""
        dt = timestamp

        # Basic calendar features
        features = {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "day_of_week": dt.dayofweek,  # 0 = Monday
            "day_of_year": dt.dayofyear,
            "week_of_year": dt.isocalendar()[1],
            "quarter": dt.quarter,
        }

        # Boolean temporal features
        features.update(
            {
                "is_weekend": dt.dayofweek >= 5,
                "is_monday": dt.dayofweek == 0,
                "is_friday": dt.dayofweek == 4,
                "is_month_start": dt.day <= 7,
                "is_month_end": dt.day >= 24,
                "is_quarter_start": dt.month in [1, 4, 7, 10] and dt.day <= 7,
                "is_quarter_end": dt.month in [3, 6, 9, 12] and dt.day >= 24,
            }
        )

        # Time of day categories
        if dt.hour in [6, 7, 8]:
            features["time_category"] = "morning_rush"
        elif dt.hour in [17, 18, 19]:
            features["time_category"] = "evening_rush"
        elif dt.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
            features["time_category"] = "night"
        elif dt.hour in [9, 10, 11, 12, 13, 14, 15, 16]:
            features["time_category"] = "daytime"
        else:
            features["time_category"] = "transition"

        # Rush hour indicators
        features.update(
            {
                "is_morning_rush": dt.hour in [7, 8, 9],
                "is_evening_rush": dt.hour in [17, 18, 19],
                "is_rush_hour": dt.hour in [7, 8, 9, 17, 18, 19],
                "is_business_hours": 9 <= dt.hour <= 17 and dt.dayofweek < 5,
                "is_night": dt.hour in [22, 23, 0, 1, 2, 3, 4, 5],
            }
        )

        # Seasonal features
        features.update(
            {
                "season": (dt.month % 12 + 3)
                // 3,  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
                "is_winter": dt.month in [12, 1, 2],
                "is_spring": dt.month in [3, 4, 5],
                "is_summer": dt.month in [6, 7, 8],
                "is_fall": dt.month in [9, 10, 11],
                "is_holiday_season": dt.month in [11, 12, 1],
            }
        )

        # Cyclical encodings (for neural networks)
        features.update(
            {
                "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
                "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
                "day_sin": np.sin(2 * np.pi * dt.day / 31),
                "day_cos": np.cos(2 * np.pi * dt.day / 31),
                "month_sin": np.sin(2 * np.pi * dt.month / 12),
                "month_cos": np.cos(2 * np.pi * dt.month / 12),
                "dayofweek_sin": np.sin(2 * np.pi * dt.dayofweek / 7),
                "dayofweek_cos": np.cos(2 * np.pi * dt.dayofweek / 7),
                "dayofyear_sin": np.sin(2 * np.pi * dt.dayofyear / 365),
                "dayofyear_cos": np.cos(2 * np.pi * dt.dayofyear / 365),
            }
        )

        return features

    def collect_weather_data(
        self, city_name: str, city_info: Dict, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        """Collect weather data from Open-Meteo API."""
        weather_records = []

        try:
            # Open-Meteo historical weather API (free, no key required)
            url = "https://archive-api.open-meteo.com/v1/archive"

            params = {
                "latitude": city_info["lat"],
                "longitude": city_info["lon"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "dewpoint_2m",
                    "apparent_temperature",
                    "surface_pressure",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                ],
                "timezone": "UTC",
            }

            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()
                hourly_data = data.get("hourly", {})

                if hourly_data and "time" in hourly_data:
                    times = pd.to_datetime(hourly_data["time"], utc=True)

                    # Filter to 6-hourly intervals
                    for i, timestamp in enumerate(times):
                        if timestamp.hour % 6 == 0:  # 0, 6, 12, 18 hours
                            weather_record = {
                                "city": city_name,
                                "timestamp_utc": timestamp,
                                "temperature_c": hourly_data.get(
                                    "temperature_2m", [None] * len(times)
                                )[i],
                                "humidity_pct": hourly_data.get(
                                    "relative_humidity_2m", [None] * len(times)
                                )[i],
                                "dewpoint_c": hourly_data.get(
                                    "dewpoint_2m", [None] * len(times)
                                )[i],
                                "apparent_temp_c": hourly_data.get(
                                    "apparent_temperature", [None] * len(times)
                                )[i],
                                "pressure_hpa": hourly_data.get(
                                    "surface_pressure", [None] * len(times)
                                )[i],
                                "cloud_cover_pct": hourly_data.get(
                                    "cloud_cover", [None] * len(times)
                                )[i],
                                "wind_speed_ms": hourly_data.get(
                                    "wind_speed_10m", [None] * len(times)
                                )[i],
                                "wind_direction_deg": hourly_data.get(
                                    "wind_direction_10m", [None] * len(times)
                                )[i],
                                "wind_gusts_ms": hourly_data.get(
                                    "wind_gusts_10m", [None] * len(times)
                                )[i],
                            }

                            # Add derived weather features
                            if (
                                weather_record["temperature_c"] is not None
                                and weather_record["humidity_pct"] is not None
                            ):
                                # Heat index approximation
                                temp_f = weather_record["temperature_c"] * 9 / 5 + 32
                                rh = weather_record["humidity_pct"]
                                if temp_f >= 80 and rh >= 40:
                                    heat_index = (
                                        -42.379
                                        + 2.04901523 * temp_f
                                        + 10.14333127 * rh
                                        - 0.22475541 * temp_f * rh
                                        - 6.83783e-3 * temp_f**2
                                        - 5.481717e-2 * rh**2
                                        + 1.22874e-3 * temp_f**2 * rh
                                        + 8.5282e-4 * temp_f * rh**2
                                        - 1.99e-6 * temp_f**2 * rh**2
                                    )
                                    weather_record["heat_index_f"] = heat_index
                                else:
                                    weather_record["heat_index_f"] = temp_f

                            # Wind categories
                            if weather_record["wind_speed_ms"] is not None:
                                wind_speed = weather_record["wind_speed_ms"]
                                if wind_speed < 2:
                                    weather_record["wind_category"] = "calm"
                                elif wind_speed < 6:
                                    weather_record["wind_category"] = "light"
                                elif wind_speed < 12:
                                    weather_record["wind_category"] = "moderate"
                                else:
                                    weather_record["wind_category"] = "strong"

                            weather_records.append(weather_record)

            else:
                log.warning(
                    f"Weather API failed for {city_name}: {response.status_code}"
                )

        except Exception as e:
            log.error(f"Error collecting weather for {city_name}: {e}")

        return weather_records

    def generate_geographic_features(self, city_name: str, city_info: Dict) -> Dict:
        """Generate geographic and demographic features."""
        features = {
            "latitude": city_info["lat"],
            "longitude": city_info["lon"],
            "elevation_m": city_info.get("elevation", 0),
            "population": city_info.get("population", 0),
            "country_code": city_info.get("country_code", ""),
        }

        # Hemisphere indicators
        features.update(
            {
                "is_northern_hemisphere": city_info["lat"] > 0,
                "is_southern_hemisphere": city_info["lat"] < 0,
                "is_eastern_hemisphere": city_info["lon"] > 0,
                "is_western_hemisphere": city_info["lon"] < 0,
            }
        )

        # Climate zone approximation
        lat = abs(city_info["lat"])
        if lat < 23.5:
            features["climate_zone"] = "tropical"
        elif lat < 35:
            features["climate_zone"] = "subtropical"
        elif lat < 50:
            features["climate_zone"] = "temperate"
        elif lat < 66.5:
            features["climate_zone"] = "subarctic"
        else:
            features["climate_zone"] = "arctic"

        # Population density category
        pop = city_info.get("population", 0)
        if pop < 1000000:
            features["population_category"] = "small"
        elif pop < 5000000:
            features["population_category"] = "medium"
        elif pop < 10000000:
            features["population_category"] = "large"
        else:
            features["population_category"] = "megacity"

        return features

    def run_etl(self, start_date: datetime, end_date: datetime) -> str:
        """Run complete local features ETL pipeline."""
        log.info("=== LOCAL FEATURES ETL PIPELINE ===")
        log.info(f"Period: {start_date.date()} to {end_date.date()}")
        log.info(f"Cities: {len(self.cities)}")

        all_records = []

        # Generate timestamps for 6-hourly intervals
        timestamps = []
        current_date = start_date
        while current_date <= end_date:
            for hour in [0, 6, 12, 18]:
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                timestamps.append(pd.Timestamp(timestamp, tz="UTC"))
            current_date += timedelta(days=1)

        log.info(f"Generating features for {len(timestamps)} timestamps")

        # Collect weather data for each city
        weather_data = {}
        for city_name, city_info in tqdm(
            self.cities.items(), desc="Collecting weather"
        ):
            weather_records = self.collect_weather_data(
                city_name, city_info, start_date, end_date
            )
            weather_data[city_name] = {
                rec["timestamp_utc"]: rec for rec in weather_records
            }

        # Generate features for each city and timestamp
        for city_name, city_info in tqdm(
            self.cities.items(), desc="Generating features"
        ):
            # Get geographic features (constant for each city)
            geo_features = self.generate_geographic_features(city_name, city_info)

            for timestamp in timestamps:
                # Generate calendar features
                calendar_features = self.generate_calendar_features(timestamp)

                # Get weather features if available
                weather_features = weather_data.get(city_name, {}).get(timestamp, {})

                # Combine all features
                record = {
                    "city": city_name,
                    "country": city_info["country"],
                    "timestamp_utc": timestamp,
                    "source": "LocalFeatures",
                    "data_type": "features",
                    "quality_flag": "generated",
                }

                record.update(geo_features)
                record.update(calendar_features)
                record.update(weather_features)

                all_records.append(record)

        if not all_records:
            log.error("No local features generated!")
            return ""

        # Convert to DataFrame
        df = pd.DataFrame(all_records)

        # Ensure consistent timestamps
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        # Sort data
        df = df.sort_values(["city", "timestamp_utc"])

        # Create partitioned output
        output_file = self.save_partitioned_data(df, start_date, end_date)

        log.info("=== LOCAL FEATURES ETL COMPLETE ===")
        log.info(f"Total records: {len(df):,}")
        log.info(f"Cities: {df['city'].nunique()}")
        log.info(f"Features: {len(df.columns)} columns")
        log.info(
            f"Time range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}"
        )
        log.info(f"Output: {output_file}")

        return str(output_file)

    def save_partitioned_data(
        self, df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> Path:
        """Save data as partitioned Parquet files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_start = start_date.strftime("%Y%m%d")
        date_end = end_date.strftime("%Y%m%d")
        output_file = (
            OUTPUT_DIR / f"local_features_{date_start}_{date_end}_{timestamp}.parquet"
        )

        # Save main file
        df.to_parquet(output_file, index=False)

        # Create partitioned structure by city
        partition_dir = (
            OUTPUT_DIR
            / "partitioned"
            / f"features_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        for city in df["city"].unique():
            city_df = df[df["city"] == city]
            city_file = partition_dir / f"city={city}" / "data.parquet"
            city_file.parent.mkdir(parents=True, exist_ok=True)
            city_df.to_parquet(city_file, index=False)

        log.info(f"Partitioned data saved to: {partition_dir}")
        return output_file


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Local Features ETL Pipeline")
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        etl = LocalFeaturesETL()
        output_file = etl.run_etl(start_date, end_date)

        if output_file:
            log.info("Local Features ETL completed successfully!")
            return 0
        else:
            log.error("Local Features ETL failed!")
            return 1

    except Exception as e:
        log.error(f"ETL execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
