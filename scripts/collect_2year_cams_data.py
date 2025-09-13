#!/usr/bin/env python3
"""
2-Year ECMWF CAMS Data Collection for 100-City Dataset
=====================================================

Collects comprehensive ECMWF CAMS atmospheric composition forecasts for all 100 cities over 2 years.
Uses the Climate Data Store (CDS) API for historical CAMS reanalysis and forecasts.

Data Sources: ECMWF Atmosphere Data Store (ADS)
Pollutants: PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃
Time Range: 2023-09-13 to 2025-09-13 (2 years)
Coverage: Global 100 cities across 5 continents
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
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
        logging.FileHandler(logs_dir / "cams_2year_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Check for CAMS API availability
try:
    import cdsapi

    CDS_AVAILABLE = True
    log.info("CDS API available for CAMS data collection")
except ImportError:
    CDS_AVAILABLE = False
    log.warning("CDS API not available. Install with: pip install cdsapi")

# Import cities from the GEFS collection script
sys.path.append(str(Path(__file__).parent))
try:
    from collect_2year_gefs_data import CITIES_100
except ImportError:
    log.error("Cannot import CITIES_100. Ensure collect_2year_gefs_data.py exists.")
    sys.exit(1)

# CAMS variable mapping to our standard names
CAMS_VARIABLES = {
    "particulate_matter_2.5um": "pm25",
    "particulate_matter_10um": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
    "ozone": "o3",
}


def get_city_bbox(city_name, buffer_deg=0.5):
    """Get bounding box around a city with buffer."""
    city_info = CITIES_100[city_name]
    lat, lon = city_info["lat"], city_info["lon"]

    return [
        lat - buffer_deg,  # South
        lon - buffer_deg,  # West
        lat + buffer_deg,  # North
        lon + buffer_deg,  # East
    ]


def collect_cams_forecast_data(start_date, end_date, data_root):
    """Collect CAMS forecast data using the CDS API."""
    if not CDS_AVAILABLE:
        log.error("CDS API not available. Cannot collect CAMS data.")
        return False

    try:
        c = cdsapi.Client()

        # CAMS Forecast API parameters
        request_params = {
            "format": "netcdf",
            "variable": list(CAMS_VARIABLES.keys()),
            "date": f"{start_date}/{end_date}",
            "time": ["00:00", "12:00"],  # Two forecasts per day
            "leadtime_hour": ["0", "6", "12", "18", "24", "30", "36", "42", "48"],
            "type": "forecast",
            "area": [85, -180, -60, 180],  # Global coverage
        }

        output_dir = Path(data_root) / "raw" / "cams"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"cams_forecast_{start_date}_{end_date}.nc"

        log.info(f"Requesting CAMS forecast data: {start_date} to {end_date}")
        log.info(f"Output file: {output_file}")

        c.retrieve(
            "cams-global-atmospheric-composition-forecasts",
            request_params,
            str(output_file),
        )

        log.info(f"CAMS forecast data downloaded successfully: {output_file}")
        return True

    except Exception as e:
        log.error(f"Failed to collect CAMS forecast data: {e}")
        return False


def collect_cams_reanalysis_data(start_date, end_date, data_root):
    """Collect CAMS reanalysis data for historical periods."""
    if not CDS_AVAILABLE:
        log.error("CDS API not available. Cannot collect CAMS data.")
        return False

    try:
        c = cdsapi.Client()

        # For historical data, use CAMS reanalysis
        years = []
        months = []
        days = []

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            years.append(str(current.year))
            months.append(f"{current.month:02d}")
            days.append(f"{current.day:02d}")
            current += timedelta(days=1)

        years = sorted(list(set(years)))
        months = sorted(list(set(months)))
        days = sorted(list(set(days)))

        request_params = {
            "format": "netcdf",
            "variable": list(CAMS_VARIABLES.keys()),
            "year": years,
            "month": months,
            "day": days,
            "time": ["00:00", "12:00"],
            "area": [85, -180, -60, 180],  # Global coverage
        }

        output_dir = Path(data_root) / "raw" / "cams"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"cams_reanalysis_{start_date}_{end_date}.nc"

        log.info(f"Requesting CAMS reanalysis data: {start_date} to {end_date}")
        log.info(f"Output file: {output_file}")

        c.retrieve("cams-global-reanalysis-eac4", request_params, str(output_file))

        log.info(f"CAMS reanalysis data downloaded successfully: {output_file}")
        return True

    except Exception as e:
        log.error(f"Failed to collect CAMS reanalysis data: {e}")
        return False


def extract_cams_city_data(netcdf_file, data_root):
    """Extract CAMS data for each of the 100 cities and convert to Parquet."""
    try:
        import xarray as xr

        log.info(f"Processing CAMS NetCDF file: {netcdf_file}")
        ds = xr.open_dataset(netcdf_file)

        # Create output directory
        output_dir = Path(data_root) / "curated" / "cams" / "parquet"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_city_data = []

        for city_name, city_info in CITIES_100.items():
            log.info(f"Extracting data for {city_name}")

            # Find nearest grid point to city
            lat_target = city_info["lat"]
            lon_target = city_info["lon"]

            # Select nearest point
            city_ds = ds.sel(
                latitude=lat_target, longitude=lon_target, method="nearest"
            )

            # Convert to DataFrame
            city_df = city_ds.to_dataframe().reset_index()

            # Add city metadata
            city_df["city"] = city_name
            city_df["country"] = city_info["country"]
            city_df["city_lat"] = lat_target
            city_df["city_lon"] = lon_target

            # Rename variables to standard names
            for cams_var, std_var in CAMS_VARIABLES.items():
                if cams_var in city_df.columns:
                    city_df[std_var] = city_df[cams_var]
                    city_df.drop(columns=[cams_var], inplace=True)

            # Add source information
            city_df["source"] = "CAMS"
            city_df["model_version"] = "CAMS_Global"

            all_city_data.append(city_df)

        # Combine all cities into one DataFrame
        combined_df = pd.concat(all_city_data, ignore_index=True)

        # Save as partitioned Parquet
        filename = f"cams_100cities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        output_file = output_dir / filename

        combined_df.to_parquet(output_file, index=False)
        log.info(f"CAMS city data saved: {output_file}")
        log.info(f"Data shape: {combined_df.shape}")
        log.info(
            f"Date range: {combined_df['time'].min()} to {combined_df['time'].max()}"
        )

        return output_file

    except Exception as e:
        log.error(f"Failed to extract CAMS city data: {e}")
        return None


def simulate_cams_collection(start_date, end_date, data_root):
    """Simulate CAMS data collection for development/testing when API unavailable."""
    log.warning("Simulating CAMS data collection (API not available)")

    # Create realistic fake data based on the GEFS benchmark structure
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    forecast_hours = [0, 6, 12, 18, 24, 30, 36, 42, 48]

    all_data = []

    for city_name, city_info in CITIES_100.items():
        for date in date_range:
            for run_time in ["00:00", "12:00"]:
                for f_hour in forecast_hours:
                    forecast_time = pd.to_datetime(f"{date} {run_time}") + pd.Timedelta(
                        hours=f_hour
                    )

                    # Generate realistic pollutant values with some randomness
                    base_values = {
                        "pm25": np.random.lognormal(2.5, 0.8),  # ~12-15 μg/m³ typical
                        "pm10": np.random.lognormal(3.0, 0.7),  # ~20-25 μg/m³ typical
                        "no2": np.random.lognormal(2.8, 0.6),  # ~16-20 ppb typical
                        "so2": np.random.lognormal(1.5, 1.0),  # ~4-5 ppb typical
                        "co": np.random.lognormal(5.5, 0.5),  # ~200-300 ppb typical
                        "o3": np.random.lognormal(3.5, 0.4),  # ~30-35 ppb typical
                    }

                    record = {
                        "source": "CAMS-Simulated",
                        "run_time": date.strftime("%Y-%m-%d"),
                        "run_hour": run_time.split(":")[0],
                        "forecast_hour": f_hour,
                        "forecast_time": forecast_time,
                        "city": city_name,
                        "country": city_info["country"],
                        "lat": city_info["lat"],
                        "lon": city_info["lon"],
                        "model_version": "CAMS_Global_Simulated",
                        **base_values,
                    }

                    all_data.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save as Parquet
    output_dir = Path(data_root) / "curated" / "cams" / "parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"cams_simulated_100cities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    output_file = output_dir / filename

    df.to_parquet(output_file, index=False)
    log.info(f"Simulated CAMS data saved: {output_file}")
    log.info(f"Data shape: {df.shape}")

    return output_file


def verify_cams_data_integrity(data_root):
    """Verify the integrity and completeness of collected CAMS data."""
    raw_dir = Path(data_root) / "raw" / "cams"
    curated_dir = Path(data_root) / "curated" / "cams" / "parquet"

    # Check raw NetCDF files
    netcdf_files = list(raw_dir.glob("*.nc")) if raw_dir.exists() else []
    total_raw_size = sum(f.stat().st_size for f in netcdf_files) if netcdf_files else 0

    # Check curated Parquet files
    parquet_files = list(curated_dir.glob("*.parquet")) if curated_dir.exists() else []
    total_curated_size = (
        sum(f.stat().st_size for f in parquet_files) if parquet_files else 0
    )

    log.info(
        f"Raw CAMS files: {len(netcdf_files)}, total size: {total_raw_size / (1024**3):.2f} GB"
    )
    log.info(
        f"Curated CAMS files: {len(parquet_files)}, "
        f"total size: {total_curated_size / (1024**2):.2f} MB"
    )

    # Check data quality if files exist
    if parquet_files:
        sample_df = pd.read_parquet(parquet_files[0])
        log.info(f"Sample CAMS data shape: {sample_df.shape}")
        log.info(f"Sample columns: {list(sample_df.columns)}")

        # Check for expected pollutants
        expected_pollutants = set(CAMS_VARIABLES.values())
        available_pollutants = set(sample_df.columns) & expected_pollutants
        log.info(f"Available pollutants: {available_pollutants}")

        if len(available_pollutants) < len(expected_pollutants):
            missing = expected_pollutants - available_pollutants
            log.warning(f"Missing pollutants: {missing}")

    return {
        "raw_files": len(netcdf_files),
        "raw_size_gb": total_raw_size / (1024**3),
        "curated_files": len(parquet_files),
        "curated_size_mb": total_curated_size / (1024**2),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect 2-year ECMWF CAMS data")
    parser.add_argument(
        "--start-date", default="2023-09-13", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2025-09-13", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument("--data-root", default=None, help="Data root directory")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate data collection if API unavailable",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing data"
    )

    args = parser.parse_args()

    # Set up data root
    data_root = args.data_root or os.environ.get("DATA_ROOT", "C:/aqf311/data")
    log.info(f"Using data root: {data_root}")

    # Ensure log directory exists
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        log.info("Verification mode - checking existing CAMS data")
        stats = verify_cams_data_integrity(data_root)
        log.info(f"CAMS data verification complete: {stats}")
        return

    log.info("Starting 2-year CAMS data collection")
    log.info(f"Date range: {args.start_date} to {args.end_date}")
    log.info(f"Cities: {len(CITIES_100)} global cities")
    log.info(f"Pollutants: {', '.join(CAMS_VARIABLES.values())}")

    # Check if we should simulate or use real API
    if not CDS_AVAILABLE or args.simulate:
        log.info("Using simulation mode for CAMS data collection")
        output_file = simulate_cams_collection(
            args.start_date, args.end_date, data_root
        )
        success = output_file is not None
    else:
        # Use real CAMS API - split into manageable chunks
        current_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        success = True

        # Collect data in monthly chunks to avoid API limits
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=30), end_date)

            chunk_start_str = current_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            # Try forecast first, then reanalysis for historical periods
            if current_date.date() <= datetime.now().date():
                chunk_success = collect_cams_reanalysis_data(
                    chunk_start_str, chunk_end_str, data_root
                )
            else:
                chunk_success = collect_cams_forecast_data(
                    chunk_start_str, chunk_end_str, data_root
                )

            if not chunk_success:
                success = False
                log.error(
                    f"Failed to collect CAMS data for {chunk_start_str} to {chunk_end_str}"
                )
                break

            current_date = chunk_end + timedelta(days=1)

        # Extract city data from collected NetCDF files
        if success:
            raw_dir = Path(data_root) / "raw" / "cams"
            netcdf_files = list(raw_dir.glob("*.nc"))

            for nc_file in netcdf_files:
                extract_cams_city_data(nc_file, data_root)

    # Final verification and reporting
    stats = verify_cams_data_integrity(data_root)
    log.info(f"Final CAMS data statistics: {stats}")

    # Save collection summary
    summary = {
        "collection_date": datetime.now().isoformat(),
        "date_range": f"{args.start_date} to {args.end_date}",
        "cities_count": len(CITIES_100),
        "data_source": (
            "CAMS-Simulated" if (not CDS_AVAILABLE or args.simulate) else "CAMS-Real"
        ),
        "success": success,
        "data_statistics": stats,
    }

    summary_file = Path(data_root) / "logs" / "cams_2year_collection_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"CAMS collection summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
