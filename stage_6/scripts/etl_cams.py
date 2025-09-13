#!/usr/bin/env python3
"""
Stage 6 ETL: ECMWF CAMS Data Collection
=======================================

Collects ECMWF CAMS (Copernicus Atmosphere Monitoring Service) data via CDS/ADS API.
Processes atmospheric composition forecasts and reanalysis data.

Cross-platform implementation supporting Linux/macOS/Windows.
"""

import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import Stage 5 cities configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.cities_stage5 import load_stage5_cities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Cross-platform data root
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "aqf_data"))
OUTPUT_DIR = DATA_ROOT / "curated" / "stage6" / "cams"

# CAMS variable configuration
CAMS_VARIABLES = {
    "PM2.5": "pm2p5",  # PM2.5 mass concentration
    "PM10": "pm10",  # PM10 mass concentration
    "NO2": "nitrogen_dioxide",  # Nitrogen dioxide
    "SO2": "sulphur_dioxide",  # Sulphur dioxide
    "CO": "carbon_monoxide",  # Carbon monoxide
    "O3": "ozone",  # Ozone
}


class CAMSETL:
    """ETL pipeline for ECMWF CAMS atmospheric composition data."""

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

    def check_cdsapi_credentials(self) -> bool:
        """Check if CDS API credentials are configured."""
        cdsapi_rc = Path.home() / ".cdsapirc"
        if not cdsapi_rc.exists():
            log.warning("CDS API credentials not found. Please configure ~/.cdsapirc")
            return False
        return True

    def create_cams_request(
        self,
        start_date: datetime,
        end_date: datetime,
        bbox: Optional[List[float]] = None,
    ) -> Dict:
        """Create CAMS data request parameters."""
        # Generate date strings
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        # Times for 6-hourly data
        times = ["00:00", "06:00", "12:00", "18:00"]

        # Bounding box (global by default)
        if bbox is None:
            bbox = [90, -180, -90, 180]  # North, West, South, East

        request = {
            "variable": list(CAMS_VARIABLES.values()),
            "date": dates,
            "time": times,
            "area": bbox,
            "format": "netcdf",
        }

        return request

    def download_cams_data(self, request: Dict, output_path: Path) -> bool:
        """Download CAMS data using CDS API."""
        try:
            # Check if cdsapi is available
            try:
                import cdsapi
            except ImportError:
                log.error("cdsapi module not found. Please install: pip install cdsapi")
                return False

            log.info("Initializing CDS API client...")
            client = cdsapi.Client()

            log.info("Downloading CAMS data...")
            log.info(f"Date range: {request['date'][0]} to {request['date'][-1]}")
            log.info(f"Variables: {len(request['variable'])}")

            # Use reanalysis dataset (more reliable)
            dataset = "cams-global-reanalysis-eac4"

            client.retrieve(dataset, request, str(output_path))

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            log.info(f"Downloaded: {output_path.name} ({file_size_mb:.2f} MB)")
            return True

        except Exception as e:
            log.error(f"CAMS download failed: {e}")
            return False

    def extract_city_data_from_netcdf(self, nc_path: Path) -> List[Dict]:
        """Extract city-specific data from NetCDF file."""
        records = []

        try:
            # Check if xarray is available
            try:
                import xarray as xr
            except ImportError:
                log.error(
                    "xarray not found. Please install: pip install xarray netcdf4"
                )
                return records

            log.info(f"Processing NetCDF file: {nc_path.name}")

            with xr.open_dataset(nc_path) as ds:
                # Get time coordinates
                times = pd.to_datetime(ds.time.values)

                # Extract data for each city
                for city_name, city_info in tqdm(
                    self.cities.items(), desc="Extracting cities"
                ):
                    lat = city_info["lat"]
                    lon = city_info["lon"]

                    # Find nearest grid point
                    lat_idx = abs(ds.latitude - lat).argmin()
                    lon_idx = abs(ds.longitude - lon).argmin()

                    for time_idx, timestamp in enumerate(times):
                        timestamp_utc = pd.Timestamp(timestamp, tz="UTC")

                        for pollutant, var_name in CAMS_VARIABLES.items():
                            if var_name in ds.variables:
                                # Extract value at city location
                                value = float(
                                    ds[var_name][time_idx, lat_idx, lon_idx].values
                                )

                                # Convert units if needed
                                if pollutant in ["PM2.5", "PM10"]:
                                    # Convert from kg/m³ to μg/m³
                                    value = value * 1e9
                                    units = "μg/m³"
                                else:
                                    # Convert from mol/mol to ppb
                                    value = value * 1e9
                                    units = "ppb"

                                if not np.isnan(value) and value >= 0:
                                    records.append(
                                        {
                                            "city": city_name,
                                            "country": city_info["country"],
                                            "latitude": lat,
                                            "longitude": lon,
                                            "timestamp_utc": timestamp_utc,
                                            "pollutant": pollutant,
                                            "value": value,
                                            "units": units,
                                            "source": "CAMS",
                                            "data_type": "reanalysis",
                                            "quality_flag": "verified",
                                            "cams_variable": var_name,
                                            "grid_lat": float(
                                                ds.latitude[lat_idx].values
                                            ),
                                            "grid_lon": float(
                                                ds.longitude[lon_idx].values
                                            ),
                                        }
                                    )

        except Exception as e:
            log.error(f"Error processing NetCDF file: {e}")

        return records

    def collect_cams_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bbox: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Collect CAMS atmospheric composition data."""
        log.info("Collecting ECMWF CAMS data...")

        all_records = []

        # Check API credentials
        if not self.check_cdsapi_credentials():
            log.warning("Using simulated CAMS data (no API credentials)")
            return self.simulate_cams_data(start_date, end_date)

        # Create request
        request = self.create_cams_request(start_date, end_date, bbox)

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Download data
            if self.download_cams_data(request, temp_path):
                # Extract city data
                all_records = self.extract_city_data_from_netcdf(temp_path)

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

        log.info(f"Collected {len(all_records)} CAMS records")
        return all_records

    def simulate_cams_data(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        """Simulate CAMS data for demonstration purposes."""
        log.info("Simulating CAMS data...")

        records = []

        current_date = start_date
        while current_date <= end_date:
            # Create 6-hourly data
            for hour in [0, 6, 12, 18]:
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                timestamp_utc = pd.Timestamp(timestamp, tz="UTC")

                for city_name, city_info in self.cities.items():
                    for pollutant, var_name in CAMS_VARIABLES.items():
                        # Simulate realistic values
                        if pollutant == "PM2.5":
                            base_value = 15.0
                            units = "μg/m³"
                        elif pollutant == "PM10":
                            base_value = 25.0
                            units = "μg/m³"
                        elif pollutant == "NO2":
                            base_value = 20.0
                            units = "ppb"
                        elif pollutant == "SO2":
                            base_value = 5.0
                            units = "ppb"
                        elif pollutant == "CO":
                            base_value = 200.0
                            units = "ppb"
                        else:  # O3
                            base_value = 50.0
                            units = "ppb"

                        # Add diurnal and random variation
                        diurnal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)
                        value = base_value * diurnal_factor + np.random.normal(
                            0, base_value * 0.1
                        )

                        records.append(
                            {
                                "city": city_name,
                                "country": city_info["country"],
                                "latitude": city_info["lat"],
                                "longitude": city_info["lon"],
                                "timestamp_utc": timestamp_utc,
                                "pollutant": pollutant,
                                "value": max(0, value),
                                "units": units,
                                "source": "CAMS-Simulated",
                                "data_type": "reanalysis",
                                "quality_flag": "simulated",
                                "cams_variable": var_name,
                            }
                        )

            current_date += timedelta(days=1)

        return records

    def run_etl(
        self,
        start_date: datetime,
        end_date: datetime,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """Run complete CAMS ETL pipeline."""
        log.info("=== ECMWF CAMS ETL PIPELINE ===")
        log.info(f"Period: {start_date.date()} to {end_date.date()}")
        log.info(f"Cities: {len(self.cities)}")
        log.info(f"Variables: {list(CAMS_VARIABLES.keys())}")

        # Collect CAMS data
        records = self.collect_cams_data(start_date, end_date, bbox)

        if not records:
            log.error("No CAMS data collected!")
            return ""

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Ensure consistent timestamps
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        # Sort data
        df = df.sort_values(["city", "timestamp_utc", "pollutant"])

        # Create partitioned output
        output_file = self.save_partitioned_data(df, start_date, end_date)

        log.info("=== CAMS ETL COMPLETE ===")
        log.info(f"Total records: {len(df):,}")
        log.info(f"Cities: {df['city'].nunique()}")
        log.info(f"Pollutants: {list(df['pollutant'].unique())}")
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
            OUTPUT_DIR / f"cams_data_{date_start}_{date_end}_{timestamp}.parquet"
        )

        # Save main file
        df.to_parquet(output_file, index=False)

        # Create partitioned structure by city
        partition_dir = (
            OUTPUT_DIR
            / "partitioned"
            / f"cams_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
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

    parser = argparse.ArgumentParser(description="ECMWF CAMS ETL Pipeline")
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument("--bbox", type=str, help="Bounding box: north,west,south,east")

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        bbox = None
        if args.bbox:
            bbox = [float(x) for x in args.bbox.split(",")]

        etl = CAMSETL()
        output_file = etl.run_etl(start_date, end_date, bbox)

        if output_file:
            log.info("CAMS ETL completed successfully!")
            return 0
        else:
            log.error("CAMS ETL failed!")
            return 1

    except Exception as e:
        log.error(f"ETL execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
