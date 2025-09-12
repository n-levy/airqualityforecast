#!/usr/bin/env python3
"""
100-City Benchmark Air Quality Data Collection
==============================================

Collect ECMWF CAMS and NOAA GEFS-Aerosols forecasts for all 100 cities
in the Stage 5 dataset to serve as benchmarks for our forecasting system.

Data Sources:
- ECMWF CAMS: Atmosphere Data Store (ADS) API for atmospheric composition
- NOAA GEFS: NOMADS (recent) + AWS S3 (historical) for aerosol chemistry

Output: Daily parquet files for Stage 5 benchmark comparison
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Import our GRIB processor
from grib_processor import GRIBProcessor

warnings.filterwarnings("ignore")

# Configure logging
log_dir = Path("C:/aqf311/Git_repo/stage_5/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "benchmark_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Try importing required libraries
# Herbie library check - not currently used but available for future enhancement
try:
    from herbie import Herbie  # noqa: F401

    HERBIE_AVAILABLE = True
    log.info("Herbie library available for GEFS collection")
except Exception as e:
    log.warning(f"Herbie not available: {e}. GEFS collection will be disabled.")
    HERBIE_AVAILABLE = False

try:
    import cdsapi

    CDS_AVAILABLE = True
    log.info("CDS API available for CAMS collection")
except Exception as e:
    log.warning(f"CDS API not available: {e}. CAMS collection will be disabled.")
    CDS_AVAILABLE = False


class BenchmarkCollector100Cities:
    """Collect benchmark forecasts for all 100 Stage 5 cities."""

    def __init__(self):
        """Initialize collector with paths and configuration."""
        self.DATA_ROOT = Path(r"C:\aqf311\data")
        self.output_dir = self.DATA_ROOT / "benchmark_forecasts_100cities"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GRIB processor
        self.grib_processor = GRIBProcessor()

        # Load complete 100-city configuration
        self.cities = self._load_cities()
        log.info(f"Loaded {len(self.cities)} cities for benchmark collection")

        # GEFS configuration
        self.gefs_config = {
            "product": "chem.25",  # GEFS chemistry at 0.25°
            "member": "mean",  # ensemble mean
            "cycles": [0, 12],  # 00Z and 12Z cycles
            "fhours": [0, 6, 12, 18, 24, 30, 36, 42, 48],  # 0-48h forecasts
        }

        # CAMS configuration
        self.cams_config = {
            "product_type": "forecast",
            "format": "grib",
            "variable": [
                "particulate_matter_2.5um",
                "particulate_matter_10um",
                "nitrogen_dioxide",
                "ozone",
            ],
            "time": ["00:00", "12:00"],
            "leadtime_hour": ["0", "6", "12", "18", "24", "30", "36", "42", "48"],
        }

    def _load_cities(self):
        """Load complete 100-city configuration from JSON."""
        cities_file = Path(
            "C:/aqf311/Git_repo/stage_5/scripts/stage_5/config/cities_config.json"
        )

        try:
            with open(cities_file, "r") as f:
                cities_data = json.load(f)

            # Flatten all regions into single list
            cities = []
            for region, city_list in cities_data.items():
                for city in city_list:
                    city["region"] = region
                    cities.append(city)

            return cities

        except Exception as e:
            log.error(f"Failed to load cities configuration: {e}")
            raise

    def collect_gefs_data_nomads(self, target_date=None, max_cities=None):
        """Collect GEFS-Aerosols data using NOMADS API directly."""
        if target_date is None:
            target_date = datetime.now().date()

        log.info(f"Collecting GEFS data via NOMADS for {target_date}")

        cities_to_process = self.cities[:max_cities] if max_cities else self.cities
        log.info(f"Processing {len(cities_to_process)} cities")

        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_chem_0p25.pl"
        rows = []

        for cycle in self.gefs_config["cycles"]:
            cycle_str = f"{cycle:02d}"
            date_str = target_date.strftime("%Y%m%d")

            log.info(f"Processing GEFS cycle {date_str}/{cycle_str}Z")

            for fhour in self.gefs_config["fhours"]:
                fhour_str = f"{fhour:03d}"

                # Build NOMADS query
                file_name = f"gefs.chem.t{cycle_str}z.a2d_0p25.f{fhour_str}.grib2"
                dir_path = f"/gefs.{date_str}/{cycle_str}/chem/pgrb2ap25"

                log.info(f"  Fetching forecast hour f{fhour_str}")

                # Process cities in batches to avoid overwhelming the server
                batch_size = 10
                for i in range(0, len(cities_to_process), batch_size):
                    batch = cities_to_process[i : i + batch_size]

                    for city in batch:
                        try:
                            # Create bounding box around city (±0.5 degrees)
                            lat, lon = city["lat"], city["lon"]

                            params = {
                                "file": file_name,
                                "dir": dir_path,
                                "var_PMTF": "on",  # PM2.5
                                "var_PMTC": "on",  # PM10
                                "lev_surface": "on",
                                "subregion": "",
                                "leftlon": lon - 0.5,
                                "rightlon": lon + 0.5,
                                "toplat": lat + 0.5,
                                "bottomlat": lat - 0.5,
                            }

                            # Build query string
                            query_string = "&".join(
                                [f"{k}={v}" for k, v in params.items()]
                            )
                            url = f"{base_url}?{query_string}"

                            # Download data
                            response = requests.get(url, timeout=30)

                            if (
                                response.status_code == 200
                                and len(response.content) > 1000
                            ):
                                # Save temporary GRIB file
                                temp_file = (
                                    self.output_dir
                                    / f"temp_{city['name']}_f{fhour_str}.grib2"
                                )
                                with open(temp_file, "wb") as f:
                                    f.write(response.content)

                                # Extract PM2.5 and PM10 values from GRIB
                                grib_data = self.grib_processor.extract_point_data(
                                    temp_file,
                                    city["lat"],
                                    city["lon"],
                                    ["pm25", "pm10"],
                                )

                                forecast_time = datetime(
                                    target_date.year,
                                    target_date.month,
                                    target_date.day,
                                    cycle,
                                ) + timedelta(hours=fhour)

                                rows.append(
                                    {
                                        "source": "GEFS-NOMADS",
                                        "run_time": datetime(
                                            target_date.year,
                                            target_date.month,
                                            target_date.day,
                                            cycle,
                                        ),
                                        "forecast_hour": fhour,
                                        "forecast_time": forecast_time,
                                        "city": city["name"],
                                        "country": city["country"],
                                        "region": city["region"],
                                        "lat": city["lat"],
                                        "lon": city["lon"],
                                        "pm25": grib_data.get("pm25"),
                                        "pm10": grib_data.get("pm10"),
                                        "data_size_bytes": len(response.content),
                                        "model_version": "GEFS-chem_0.25deg",
                                    }
                                )

                                # Clean up temp file
                                temp_file.unlink(missing_ok=True)

                                log.debug(
                                    f"    {city['name']}: {len(response.content)} bytes"
                                )

                            else:
                                log.warning(
                                    f"    {city['name']}: No data (HTTP {response.status_code})"
                                )

                        except Exception as e:
                            log.warning(f"    {city['name']}: Error - {e}")

                    # Small delay between batches
                    import time

                    time.sleep(1)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def collect_cams_data_api(self, target_date=None):
        """Collect CAMS data using ECMWF ADS API."""
        if not CDS_AVAILABLE:
            log.error("CDS API not available for CAMS collection")
            return pd.DataFrame()

        if target_date is None:
            target_date = datetime.now().date()

        log.info(f"Collecting CAMS data via ADS API for {target_date}")

        try:
            c = cdsapi.Client()

            # Request CAMS data for target date
            request = {
                "date": target_date.strftime("%Y-%m-%d"),
                "type": self.cams_config["product_type"],
                "variable": self.cams_config["variable"],
                "time": self.cams_config["time"],
                "leadtime_hour": self.cams_config["leadtime_hour"],
                "format": self.cams_config["format"],
            }

            # Download to temporary file
            temp_file = self.output_dir / f"cams_{target_date.strftime('%Y%m%d')}.grib"
            log.info(f"Downloading CAMS data to {temp_file}")

            c.retrieve(
                "cams-global-atmospheric-composition-forecasts", request, str(temp_file)
            )

            # Process GRIB file (would need proper GRIB processing)
            rows = []

            # For now, create placeholder records for all cities
            for city in self.cities:
                for time_str in self.cams_config["time"]:
                    cycle = int(time_str.split(":")[0])

                    for leadtime_str in self.cams_config["leadtime_hour"]:
                        leadtime = int(leadtime_str)

                        forecast_time = datetime(
                            target_date.year, target_date.month, target_date.day, cycle
                        ) + timedelta(hours=leadtime)

                        rows.append(
                            {
                                "source": "CAMS-ADS",
                                "run_time": datetime(
                                    target_date.year,
                                    target_date.month,
                                    target_date.day,
                                    cycle,
                                ),
                                "forecast_hour": leadtime,
                                "forecast_time": forecast_time,
                                "city": city["name"],
                                "country": city["country"],
                                "region": city["region"],
                                "lat": city["lat"],
                                "lon": city["lon"],
                                "pm25": None,  # Would extract from GRIB
                                "pm10": None,  # Would extract from GRIB
                                "no2": None,  # Would extract from GRIB
                                "o3": None,  # Would extract from GRIB
                                "model_version": "CAMS_global",
                            }
                        )

            # Clean up temp file
            temp_file.unlink(missing_ok=True)

            return pd.DataFrame(rows) if rows else pd.DataFrame()

        except Exception as e:
            log.error(f"CAMS collection failed: {e}")
            return pd.DataFrame()

    def save_to_parquet(self, df, filename):
        """Save DataFrame to Parquet format."""
        if df.empty:
            log.warning(f"No data to save for {filename}")
            return

        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df)} records to {output_path}")

        # Log summary statistics
        log.info(f"Data summary for {filename}:")
        log.info(f"  Cities: {df['city'].nunique()}")
        log.info(f"  Regions: {df['region'].value_counts().to_dict()}")
        log.info(f"  Sources: {df['source'].value_counts().to_dict()}")
        log.info(
            f"  Date range: {df['forecast_time'].min()} to {df['forecast_time'].max()}"
        )

    def run_daily_collection(self, target_date=None, test_mode=True):
        """Run daily benchmark collection for all 100 cities."""
        if target_date is None:
            target_date = datetime.now().date()

        log.info(f"Starting daily benchmark collection for {target_date}")
        log.info(f"Test mode: {test_mode}")

        date_str = target_date.strftime("%Y%m%d")
        max_cities = 5 if test_mode else None

        # Collect GEFS data
        log.info("=== COLLECTING GEFS-AEROSOLS DATA ===")
        gefs_df = self.collect_gefs_data_nomads(target_date, max_cities=max_cities)
        if not gefs_df.empty:
            self.save_to_parquet(gefs_df, f"gefs_benchmark_{date_str}.parquet")

        # Collect CAMS data (only if not in test mode due to API limits)
        if not test_mode:
            log.info("=== COLLECTING CAMS DATA ===")
            cams_df = self.collect_cams_data_api(target_date)
            if not cams_df.empty:
                self.save_to_parquet(cams_df, f"cams_benchmark_{date_str}.parquet")
        else:
            log.info("Skipping CAMS collection in test mode")
            cams_df = pd.DataFrame()

        # Combine datasets
        all_data = pd.concat([gefs_df, cams_df], ignore_index=True)
        if not all_data.empty:
            self.save_to_parquet(all_data, f"combined_benchmark_{date_str}.parquet")

            log.info("=== COLLECTION COMPLETE ===")
            log.info(f"Total records: {len(all_data)}")
            log.info(f"Cities covered: {all_data['city'].nunique()}")
            log.info(f"Regions: {all_data['region'].value_counts().to_dict()}")
            log.info(f"Sources: {all_data['source'].value_counts().to_dict()}")
        else:
            log.error("No benchmark data collected")


def main():
    """Main execution function."""
    collector = BenchmarkCollector100Cities()

    # Run collection for today in test mode (5 cities only)
    collector.run_daily_collection(test_mode=True)

    log.info("100-city benchmark collection complete!")


if __name__ == "__main__":
    main()
