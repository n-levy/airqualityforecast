#!/usr/bin/env python3
"""
GEFS-Aerosols PM2.5/PM10 Data Collection
========================================

Collect GEFS chemistry forecasts from NOMADS (recent) and AWS S3 (historical)
for air quality benchmark comparison with ECMWF CAMS.

Uses Herbie for clean API access to GEFS-chem 0.25° data.
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("C:/aqf311/Git_repo/stage_5/logs/gefs_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

try:
    from herbie import Herbie
except ImportError:
    log.error("Herbie not installed. Install with: pip install herbie-data")
    exit(1)


class GEFSAerosolCollector:
    """Collect GEFS-Aerosols data for air quality benchmarking."""

    def __init__(self):
        """Initialize collector with paths and configuration."""
        self.DATA_ROOT = Path(r"C:\aqf311\data")
        self.output_dir = self.DATA_ROOT / "gefs_pm25_0p25"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Global coordinates for 100 cities (sample for now)
        self.cities = [
            {"name": "Delhi", "lat": 28.6139, "lon": 77.209},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
            {"name": "Los_Angeles", "lat": 34.0522, "lon": -118.2437},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        ]

        # GEFS configuration
        self.product = "chem.25"  # GEFS chemistry at 0.25°
        self.member = "mean"  # ensemble mean
        self.cycles = [0, 12]  # 00Z and 12Z cycles
        self.fhours = list(range(0, 121, 3))  # 0-120h every 3h

    def collect_recent_data(self, days_back=7):
        """Collect recent GEFS-chem data using NOMADS."""
        log.info(f"Collecting recent {days_back} days from NOMADS")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        rows = []
        for date in self._daterange(start_date, end_date, 24):
            for cycle in self.cycles:
                valid = datetime(date.year, date.month, date.day, cycle)

                for f in self.fhours[:17]:  # Limit to 48h for testing
                    try:
                        log.info(f"Fetching {valid.strftime('%Y%m%d_%H')} f{f:03d}")

                        H = Herbie(
                            valid,
                            model="gefs",
                            product=self.product,
                            member=self.member,
                            fxx=f,
                        )

                        # Check if data is available
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
                            log.info(f"Available vars: {list(ds.data_vars.keys())}")
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
                                        "run": valid,
                                        "f_hour": f,
                                        "forecast_time": valid + timedelta(hours=f),
                                        "city": city["name"],
                                        "lat": city["lat"],
                                        "lon": city["lon"],
                                        "pm25": float(city_data.values),
                                        "source": "NOMADS",
                                    }
                                )

                            except Exception as e:
                                log.warning(
                                    f"City extraction failed for {city['name']}: {e}"
                                )

                    except Exception as e:
                        log.warning(f"Failed to fetch {valid} f{f:03d}: {e}")

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def collect_historical_data(self, start_date, end_date):
        """Collect historical GEFS-chem data from AWS S3."""
        log.info(f"Collecting historical data from {start_date} to {end_date}")

        rows = []
        for date in self._daterange(start_date, end_date, 24):
            for cycle in self.cycles:
                valid = datetime(date.year, date.month, date.day, cycle)

                # Limit forecast hours for historical data
                for f in self.fhours[::4]:  # Every 12h to reduce volume
                    try:
                        log.info(
                            f"Fetching historical {valid.strftime('%Y%m%d_%H')} f{f:03d}"
                        )

                        H = Herbie(
                            valid,
                            model="gefs",
                            product=self.product,
                            member=self.member,
                            fxx=f,
                        )

                        # Herbie automatically uses AWS S3 for historical data
                        ds = H.xarray()

                        # Find PM2.5 variable
                        pm25_var = None
                        for var in ds.data_vars:
                            if any(
                                pm in var.lower() for pm in ["pm2p5", "pmtf", "pm25"]
                            ):
                                pm25_var = var
                                break

                        if pm25_var is None:
                            continue

                        # Extract data for cities
                        pm25_data = ds[pm25_var]

                        for city in self.cities:
                            try:
                                city_data = pm25_data.sel(
                                    latitude=city["lat"],
                                    longitude=city["lon"],
                                    method="nearest",
                                )

                                rows.append(
                                    {
                                        "run": valid,
                                        "f_hour": f,
                                        "forecast_time": valid + timedelta(hours=f),
                                        "city": city["name"],
                                        "lat": city["lat"],
                                        "lon": city["lon"],
                                        "pm25": float(city_data.values),
                                        "source": "AWS_S3",
                                    }
                                )

                            except Exception as e:
                                log.warning(f"Historical city extraction failed: {e}")

                    except Exception as e:
                        log.warning(f"Failed historical fetch {valid} f{f:03d}: {e}")

        return pd.DataFrame(rows) if rows else pd.DataFrame()

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
        log.info("Data summary:")
        log.info(
            f"  Date range: {df['forecast_time'].min()} to {df['forecast_time'].max()}"
        )
        log.info(f"  Cities: {df['city'].nunique()}")
        log.info(
            f"  PM2.5 range: {df['pm25'].min():.2f} - {df['pm25'].max():.2f} μg/m³"
        )
        log.info(f"  Mean PM2.5: {df['pm25'].mean():.2f} μg/m³")


def main():
    """Main execution function."""
    collector = GEFSAerosolCollector()

    log.info("Starting GEFS-Aerosols PM2.5 collection")

    # Collect recent data (7 days)
    recent_df = collector.collect_recent_data(days_back=7)
    if not recent_df.empty:
        collector.save_to_parquet(recent_df, "gefs_pm25_recent_7days.parquet")

    # Collect sample historical data (30 days from 6 months ago)
    end_hist = datetime.now() - timedelta(days=180)
    start_hist = end_hist - timedelta(days=30)
    historical_df = collector.collect_historical_data(start_hist, end_hist)
    if not historical_df.empty:
        collector.save_to_parquet(historical_df, "gefs_pm25_historical_30days.parquet")

    # Combine datasets
    all_data = pd.concat([recent_df, historical_df], ignore_index=True)
    if not all_data.empty:
        collector.save_to_parquet(all_data, "gefs_pm25_combined.parquet")

        log.info("GEFS-Aerosols collection complete!")
        log.info(f"Total records: {len(all_data)}")
        log.info(f"Sources: {all_data['source'].value_counts().to_dict()}")
    else:
        log.error("No data collected - check API connectivity and variable names")


if __name__ == "__main__":
    main()
