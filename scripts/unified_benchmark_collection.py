#!/usr/bin/env python3
"""
Unified Benchmark Air Quality Data Collection
=============================================

Collect both ECMWF CAMS and NOAA GEFS-Aerosols forecasts for benchmarking
against our Stage-5 air quality forecasting system.

Data Sources:
- ECMWF CAMS: Atmosphere Data Store (ADS) API for atmospheric composition
- NOAA GEFS: NOMADS (recent) + AWS S3 (historical) for aerosol chemistry

Output: Parquet files compatible with Stage-1 architecture
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("C:/aqf311/Git_repo/stage_5/logs/benchmark_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Try importing required libraries
try:
    from herbie import Herbie

    HERBIE_AVAILABLE = True
except Exception as e:
    log.warning(f"Herbie not available: {e}. GEFS collection will be disabled.")
    HERBIE_AVAILABLE = False

try:
    import cdsapi

    CDS_AVAILABLE = True
except Exception as e:
    log.warning(f"CDS API not available: {e}. CAMS collection will be disabled.")
    CDS_AVAILABLE = False


class UnifiedBenchmarkCollector:
    """Unified collector for CAMS and GEFS air quality benchmarks."""

    def __init__(self):
        """Initialize collector with paths and configuration."""
        self.DATA_ROOT = Path(r"C:\aqf311\data")
        self.output_dir = self.DATA_ROOT / "benchmark_forecasts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load 100-city configuration
        self.cities = self._load_cities()

        # GEFS configuration
        self.gefs_config = {
            "product": "chem.25",  # GEFS chemistry at 0.25°
            "member": "mean",  # ensemble mean
            "cycles": [0, 12],  # 00Z and 12Z cycles
            "fhours": list(range(0, 121, 3)),  # 0-120h every 3h
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
            "leadtime_hour": [str(h) for h in range(0, 121, 3)],
        }

    def _load_cities(self):
        """Load 100-city configuration from JSON."""
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

            log.info(f"Loaded {len(cities)} cities from configuration")
            return cities[:20]  # Limit to 20 cities for testing

        except Exception as e:
            log.error(f"Failed to load cities configuration: {e}")
            # Fallback to hardcoded sample
            return [
                {"name": "Delhi", "lat": 28.6139, "lon": 77.209, "country": "India"},
                {
                    "name": "Beijing",
                    "lat": 39.9042,
                    "lon": 116.4074,
                    "country": "China",
                },
                {
                    "name": "Los_Angeles",
                    "lat": 34.0522,
                    "lon": -118.2437,
                    "country": "USA",
                },
                {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "UK"},
                {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "Egypt"},
            ]

    def collect_gefs_data(self, days_back=7):
        """Collect GEFS-Aerosols data using NOMADS + S3."""
        if not HERBIE_AVAILABLE:
            log.error("Herbie not available for GEFS collection")
            return pd.DataFrame()

        log.info(f"Collecting GEFS data for past {days_back} days")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        rows = []
        for date in self._daterange(start_date, end_date, 24):
            for cycle in self.gefs_config["cycles"]:
                valid = datetime(date.year, date.month, date.day, cycle)

                for f in self.gefs_config["fhours"][:17]:  # Limit to 48h for testing
                    try:
                        log.info(
                            f"Fetching GEFS {valid.strftime('%Y%m%d_%H')} f{f:03d}"
                        )

                        H = Herbie(
                            valid,
                            model="gefs",
                            product=self.gefs_config["product"],
                            member=self.gefs_config["member"],
                            fxx=f,
                        )

                        if not H.grib:
                            log.warning(f"No GRIB file for {valid} f{f:03d}")
                            continue

                        # Load with xarray via cfgrib
                        ds = H.xarray()

                        # Find PM2.5 variable (common names)
                        pm25_var = None
                        for var in ds.data_vars:
                            if any(
                                pm in var.lower() for pm in ["pm2p5", "pmtf", "pm25"]
                            ):
                                pm25_var = var
                                break

                        if pm25_var is None:
                            log.warning(f"No PM2.5 variable found in {valid} f{f:03d}")
                            continue

                        # Extract data for cities
                        pm25_data = ds[pm25_var]

                        for city in self.cities:
                            try:
                                # Extract nearest grid point
                                city_data = pm25_data.sel(
                                    latitude=city["lat"],
                                    longitude=city["lon"],
                                    method="nearest",
                                )

                                rows.append(
                                    {
                                        "source": "GEFS",
                                        "run_time": valid,
                                        "forecast_hour": f,
                                        "forecast_time": valid + timedelta(hours=f),
                                        "city": city["name"],
                                        "country": city.get("country", "Unknown"),
                                        "lat": city["lat"],
                                        "lon": city["lon"],
                                        "pm25": float(city_data.values),
                                        "model_version": "GEFS-chem_0.25deg",
                                    }
                                )

                            except Exception as e:
                                log.warning(
                                    f"City extraction failed for {city['name']}: {e}"
                                )

                    except Exception as e:
                        log.warning(f"Failed to fetch GEFS {valid} f{f:03d}: {e}")

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def collect_cams_data(self, days_back=7):
        """Collect CAMS data using ECMWF ADS API."""
        if not CDS_AVAILABLE:
            log.error("CDS API not available for CAMS collection")
            return pd.DataFrame()

        log.info(f"Collecting CAMS data for past {days_back} days")

        try:
            c = cdsapi.Client()

            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)

            date_list = []
            current = start_date
            while current <= end_date:
                date_list.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

            # Request CAMS data
            request = {
                "date": date_list,
                "type": self.cams_config["product_type"],
                "variable": self.cams_config["variable"],
                "time": self.cams_config["time"],
                "leadtime_hour": self.cams_config["leadtime_hour"][:17],  # Limit to 48h
                "format": self.cams_config["format"],
            }

            # Download to temporary file
            temp_file = self.output_dir / "temp_cams.grib"
            c.retrieve(
                "cams-global-atmospheric-composition-forecasts", request, str(temp_file)
            )

            # Process GRIB file
            ds = xr.open_dataset(temp_file, engine="cfgrib")
            log.info(f"CAMS variables: {list(ds.data_vars.keys())}")

            # Extract PM2.5 data for cities
            rows = []
            if "pm2p5" in ds.data_vars:
                pm25_data = ds["pm2p5"]

                for city in self.cities:
                    try:
                        city_data = pm25_data.sel(
                            latitude=city["lat"],
                            longitude=city["lon"],
                            method="nearest",
                        )

                        # Convert to DataFrame and iterate through time/step
                        for time_idx, time_val in enumerate(city_data.time):
                            for step_idx, step_val in enumerate(city_data.step):
                                forecast_time = pd.to_datetime(
                                    time_val.values
                                ) + pd.to_timedelta(step_val.values)

                                rows.append(
                                    {
                                        "source": "CAMS",
                                        "run_time": pd.to_datetime(time_val.values),
                                        "forecast_hour": int(
                                            step_val.values / pd.Timedelta(hours=1)
                                        ),
                                        "forecast_time": forecast_time,
                                        "city": city["name"],
                                        "country": city.get("country", "Unknown"),
                                        "lat": city["lat"],
                                        "lon": city["lon"],
                                        "pm25": float(
                                            city_data.isel(
                                                time=time_idx, step=step_idx
                                            ).values
                                        ),
                                        "model_version": "CAMS_global",
                                    }
                                )

                    except Exception as e:
                        log.warning(
                            f"CAMS city extraction failed for {city['name']}: {e}"
                        )

            # Clean up temp file
            temp_file.unlink(missing_ok=True)

            return pd.DataFrame(rows) if rows else pd.DataFrame()

        except Exception as e:
            log.error(f"CAMS collection failed: {e}")
            return pd.DataFrame()

    def _daterange(self, start, end, step_hours=24):
        """Generate date range with step."""
        t = start
        while t <= end:
            yield t
            t += timedelta(hours=step_hours)

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
        log.info(
            f"  Date range: {df['forecast_time'].min()} to {df['forecast_time'].max()}"
        )
        log.info(f"  Cities: {df['city'].nunique()}")
        log.info(f"  Sources: {df['source'].value_counts().to_dict()}")
        log.info(
            f"  PM2.5 range: {df['pm25'].min():.2f} - {df['pm25'].max():.2f} μg/m³"
        )
        log.info(f"  Mean PM2.5: {df['pm25'].mean():.2f} μg/m³")


def main():
    """Main execution function."""
    collector = UnifiedBenchmarkCollector()

    log.info("Starting unified benchmark air quality data collection")

    # Collect GEFS data (7 days)
    gefs_df = collector.collect_gefs_data(days_back=3)  # Reduced for testing
    if not gefs_df.empty:
        collector.save_to_parquet(gefs_df, "gefs_benchmark_3days.parquet")

    # Collect CAMS data (7 days)
    cams_df = collector.collect_cams_data(days_back=3)  # Reduced for testing
    if not cams_df.empty:
        collector.save_to_parquet(cams_df, "cams_benchmark_3days.parquet")

    # Combine datasets
    all_data = pd.concat([gefs_df, cams_df], ignore_index=True)
    if not all_data.empty:
        collector.save_to_parquet(all_data, "combined_benchmark_3days.parquet")

        log.info("Benchmark collection complete!")
        log.info(f"Total records: {len(all_data)}")
        log.info(f"Sources: {all_data['source'].value_counts().to_dict()}")
        log.info(f"Models: {all_data['model_version'].value_counts().to_dict()}")
    else:
        log.error(
            "No benchmark data collected - check API connectivity and credentials"
        )


if __name__ == "__main__":
    main()
