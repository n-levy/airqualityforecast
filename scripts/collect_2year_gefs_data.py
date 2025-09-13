#!/usr/bin/env python3
"""
2-Year NOAA GEFS-Aerosol Data Collection for 100-City Dataset
=============================================================

Collects comprehensive NOAA GEFS-Aerosol forecasts for all 100 cities over 2 years.
Builds upon existing infrastructure for historical data collection.

Data Sources: NOAA GEFS-PDS S3 Bucket
Pollutants: PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃
Time Range: 2023-09-13 to 2025-09-13 (2 years)
Coverage: Global 100 cities across 5 continents
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Configure logging
logs_dir = Path("C:/aqf311/data/logs")
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "gefs_2year_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# 100-City coordinates for global coverage
CITIES_100 = {
    # Asia (20 cities)
    "Delhi": {"lat": 28.61, "lon": 77.21, "country": "India"},
    "Lahore": {"lat": 31.55, "lon": 74.34, "country": "Pakistan"},
    "Beijing": {"lat": 39.90, "lon": 116.41, "country": "China"},
    "Dhaka": {"lat": 23.81, "lon": 90.41, "country": "Bangladesh"},
    "Mumbai": {"lat": 19.08, "lon": 72.88, "country": "India"},
    "Karachi": {"lat": 24.86, "lon": 67.00, "country": "Pakistan"},
    "Shanghai": {"lat": 31.23, "lon": 121.47, "country": "China"},
    "Kolkata": {"lat": 22.57, "lon": 88.36, "country": "India"},
    "Bangkok": {"lat": 14.60, "lon": 100.50, "country": "Thailand"},
    "Jakarta": {"lat": -6.21, "lon": 106.85, "country": "Indonesia"},
    "Manila": {"lat": 14.60, "lon": 120.98, "country": "Philippines"},
    "Ho Chi Minh City": {"lat": 10.82, "lon": 106.63, "country": "Vietnam"},
    "Hanoi": {"lat": 21.03, "lon": 105.85, "country": "Vietnam"},
    "Seoul": {"lat": 37.57, "lon": 126.98, "country": "South Korea"},
    "Taipei": {"lat": 25.03, "lon": 121.57, "country": "Taiwan"},
    "Ulaanbaatar": {"lat": 47.89, "lon": 106.91, "country": "Mongolia"},
    "Almaty": {"lat": 43.26, "lon": 76.93, "country": "Kazakhstan"},
    "Tashkent": {"lat": 41.30, "lon": 69.24, "country": "Uzbekistan"},
    "Tehran": {"lat": 35.70, "lon": 51.42, "country": "Iran"},
    "Kabul": {"lat": 34.56, "lon": 69.21, "country": "Afghanistan"},
    # Africa (20 cities)
    "N'Djamena": {"lat": 12.13, "lon": 15.06, "country": "Chad"},
    "Cairo": {"lat": 30.04, "lon": 31.24, "country": "Egypt"},
    "Lagos": {"lat": 6.52, "lon": 3.38, "country": "Nigeria"},
    "Accra": {"lat": 5.60, "lon": -0.19, "country": "Ghana"},
    "Abidjan": {"lat": 5.32, "lon": -4.03, "country": "Ivory Coast"},
    "Dakar": {"lat": 14.69, "lon": -17.44, "country": "Senegal"},
    "Bamako": {"lat": 12.65, "lon": -8.00, "country": "Mali"},
    "Addis Ababa": {"lat": 9.03, "lon": 38.74, "country": "Ethiopia"},
    "Nairobi": {"lat": -1.29, "lon": 36.82, "country": "Kenya"},
    "Kampala": {"lat": 0.35, "lon": 32.58, "country": "Uganda"},
    "Dar es Salaam": {"lat": -6.79, "lon": 39.21, "country": "Tanzania"},
    "Kinshasa": {"lat": -4.44, "lon": 15.27, "country": "DR Congo"},
    "Johannesburg": {"lat": -26.20, "lon": 28.05, "country": "South Africa"},
    "Cape Town": {"lat": -33.93, "lon": 18.42, "country": "South Africa"},
    "Casablanca": {"lat": 33.57, "lon": -7.59, "country": "Morocco"},
    "Algiers": {"lat": 36.75, "lon": 3.04, "country": "Algeria"},
    "Tunis": {"lat": 36.81, "lon": 10.18, "country": "Tunisia"},
    "Tripoli": {"lat": 32.89, "lon": 13.19, "country": "Libya"},
    "Khartoum": {"lat": 15.50, "lon": 32.56, "country": "Sudan"},
    "Mogadishu": {"lat": 2.04, "lon": 45.34, "country": "Somalia"},
    # Europe (20 cities)
    "Berlin": {"lat": 52.52, "lon": 13.41, "country": "Germany"},
    "London": {"lat": 51.51, "lon": -0.13, "country": "United Kingdom"},
    "Paris": {"lat": 48.86, "lon": 2.35, "country": "France"},
    "Rome": {"lat": 41.90, "lon": 12.50, "country": "Italy"},
    "Madrid": {"lat": 40.42, "lon": -3.70, "country": "Spain"},
    "Amsterdam": {"lat": 52.37, "lon": 4.90, "country": "Netherlands"},
    "Brussels": {"lat": 50.85, "lon": 4.35, "country": "Belgium"},
    "Vienna": {"lat": 48.21, "lon": 16.37, "country": "Austria"},
    "Warsaw": {"lat": 52.23, "lon": 21.01, "country": "Poland"},
    "Prague": {"lat": 50.09, "lon": 14.42, "country": "Czech Republic"},
    "Budapest": {"lat": 47.50, "lon": 19.04, "country": "Hungary"},
    "Bucharest": {"lat": 44.43, "lon": 26.11, "country": "Romania"},
    "Sofia": {"lat": 42.70, "lon": 23.32, "country": "Bulgaria"},
    "Athens": {"lat": 37.98, "lon": 23.73, "country": "Greece"},
    "Belgrade": {"lat": 44.79, "lon": 20.45, "country": "Serbia"},
    "Zagreb": {"lat": 45.81, "lon": 15.98, "country": "Croatia"},
    "Ljubljana": {"lat": 46.06, "lon": 14.51, "country": "Slovenia"},
    "Bratislava": {"lat": 48.15, "lon": 17.11, "country": "Slovakia"},
    "Barcelona": {"lat": 41.39, "lon": 2.16, "country": "Spain"},
    "Milan": {"lat": 45.46, "lon": 9.19, "country": "Italy"},
    # North America (20 cities)
    "Mexico City": {"lat": 19.43, "lon": -99.13, "country": "Mexico"},
    "Los Angeles": {"lat": 34.05, "lon": -118.24, "country": "USA"},
    "New York": {"lat": 40.71, "lon": -74.01, "country": "USA"},
    "Chicago": {"lat": 41.88, "lon": -87.63, "country": "USA"},
    "Houston": {"lat": 29.76, "lon": -95.37, "country": "USA"},
    "Phoenix": {"lat": 33.45, "lon": -112.07, "country": "USA"},
    "Philadelphia": {"lat": 39.95, "lon": -75.17, "country": "USA"},
    "San Antonio": {"lat": 29.42, "lon": -98.49, "country": "USA"},
    "San Diego": {"lat": 32.72, "lon": -117.16, "country": "USA"},
    "Dallas": {"lat": 32.78, "lon": -96.80, "country": "USA"},
    "Toronto": {"lat": 43.65, "lon": -79.38, "country": "Canada"},
    "Montreal": {"lat": 45.50, "lon": -73.57, "country": "Canada"},
    "Vancouver": {"lat": 49.25, "lon": -123.12, "country": "Canada"},
    "Calgary": {"lat": 51.05, "lon": -114.07, "country": "Canada"},
    "Ottawa": {"lat": 45.42, "lon": -75.70, "country": "Canada"},
    "Guadalajara": {"lat": 20.67, "lon": -103.35, "country": "Mexico"},
    "Monterrey": {"lat": 25.67, "lon": -100.32, "country": "Mexico"},
    "Atlanta": {"lat": 33.75, "lon": -84.39, "country": "USA"},
    "Denver": {"lat": 39.74, "lon": -104.99, "country": "USA"},
    "Seattle": {"lat": 47.61, "lon": -122.33, "country": "USA"},
    # South America (20 cities)
    "São Paulo": {"lat": -23.55, "lon": -46.63, "country": "Brazil"},
    "Lima": {"lat": -12.05, "lon": -77.04, "country": "Peru"},
    "Bogotá": {"lat": 4.61, "lon": -74.08, "country": "Colombia"},
    "Rio de Janeiro": {"lat": -22.91, "lon": -43.17, "country": "Brazil"},
    "Buenos Aires": {"lat": -34.61, "lon": -58.38, "country": "Argentina"},
    "Santiago": {"lat": -33.46, "lon": -70.65, "country": "Chile"},
    "Caracas": {"lat": 10.49, "lon": -66.88, "country": "Venezuela"},
    "Belo Horizonte": {"lat": -19.92, "lon": -43.94, "country": "Brazil"},
    "Medellín": {"lat": 6.24, "lon": -75.59, "country": "Colombia"},
    "Quito": {"lat": -0.18, "lon": -78.47, "country": "Ecuador"},
    "La Paz": {"lat": -16.50, "lon": -68.15, "country": "Bolivia"},
    "Montevideo": {"lat": -34.90, "lon": -56.16, "country": "Uruguay"},
    "Asunción": {"lat": -25.26, "lon": -57.58, "country": "Paraguay"},
    "Georgetown": {"lat": 6.80, "lon": -58.16, "country": "Guyana"},
    "Paramaribo": {"lat": 5.87, "lon": -55.17, "country": "Suriname"},
    "Cayenne": {"lat": 4.93, "lon": -52.33, "country": "French Guiana"},
    "Brasília": {"lat": -15.83, "lon": -47.86, "country": "Brazil"},
    "Córdoba": {"lat": -31.42, "lon": -64.18, "country": "Argentina"},
    "Rosario": {"lat": -32.94, "lon": -60.65, "country": "Argentina"},
    "Cali": {"lat": 3.39, "lon": -76.52, "country": "Colombia"},
}

# Global bounding box for comprehensive coverage
GLOBAL_BBOX = "-180,-60,180,85"  # Longitude min/max, Latitude min/max


def calculate_date_chunks(start_date, end_date, chunk_months=3):
    """Split date range into manageable chunks for collection."""
    chunks = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=90), end_date)  # ~3 months
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)

    return chunks


def run_gefs_orchestrator(start_date, end_date, data_root):
    """Run the GEFS orchestrator script for a specific date range."""
    cmd = [
        sys.executable,
        "scripts/orchestrate_gefs_https.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--cycles",
        "00,12",  # Two cycles per day for comprehensive coverage
        "--fhours",
        "0:6:48",  # 48-hour forecasts in 6-hour intervals
        f"--bbox={GLOBAL_BBOX}",
        "--pollutants",
        "PM25,PM10,NO2,SO2,CO,O3",
        "--data-root",
        data_root,
        "--workers",
        "4",
        "--force",
    ]

    log.info(f"Running GEFS collection: {start_date} to {end_date}")
    log.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"Collection successful for {start_date} to {end_date}")
        log.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Collection failed for {start_date} to {end_date}")
        log.error(f"Error: {e.stderr}")
        return False


def verify_data_integrity(data_root):
    """Verify the integrity and completeness of collected data."""
    raw_dir = Path(data_root) / "raw" / "gefs_chem"
    curated_dir = Path(data_root) / "curated" / "gefs_chem" / "parquet"

    # Check manifest files
    manifest_file = raw_dir / "_manifests" / "download_manifest.csv"
    extract_manifest = curated_dir.parent / "extract_manifest.csv"

    total_size = 0
    total_files = 0

    if manifest_file.exists():
        manifest_df = pd.read_csv(manifest_file)
        total_files = len(manifest_df)
        total_size = (
            manifest_df["file_size_bytes"].sum()
            if "file_size_bytes" in manifest_df.columns
            else 0
        )
        log.info(
            f"Downloaded {total_files} GRIB2 files, total size: {total_size / (1024**3):.2f} GB"
        )

    # Check curated data
    if curated_dir.exists():
        parquet_files = list(curated_dir.rglob("*.parquet"))
        log.info(f"Created {len(parquet_files)} Parquet files")

        # Sample data quality check
        if parquet_files:
            sample_df = pd.read_parquet(parquet_files[0])
            log.info(f"Sample data shape: {sample_df.shape}")
            log.info(f"Sample columns: {list(sample_df.columns)}")
            log.info(f"Sample data types: {sample_df.dtypes.to_dict()}")

    return {
        "raw_files": total_files,
        "raw_size_gb": total_size / (1024**3),
        "curated_files": len(parquet_files) if "parquet_files" in locals() else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect 2-year NOAA GEFS-Aerosol data"
    )
    parser.add_argument(
        "--start-date", default="2023-09-13", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2025-09-13", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument("--data-root", default=None, help="Data root directory")
    parser.add_argument(
        "--chunk-months", type=int, default=3, help="Months per collection chunk"
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
        log.info("Verification mode - checking existing data")
        stats = verify_data_integrity(data_root)
        log.info(f"Data verification complete: {stats}")
        return

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    log.info("Starting 2-year GEFS data collection")
    log.info(
        f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    log.info(f"Cities: {len(CITIES_100)} global cities")
    log.info("Pollutants: PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃")

    # Calculate collection chunks
    chunks = calculate_date_chunks(start_date, end_date, args.chunk_months)
    log.info(f"Collection will be done in {len(chunks)} chunks")

    # Collect data in chunks
    successful_chunks = 0
    failed_chunks = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        log.info(f"Processing chunk {i}/{len(chunks)}: {chunk_start} to {chunk_end}")

        success = run_gefs_orchestrator(chunk_start, chunk_end, data_root)
        if success:
            successful_chunks += 1
        else:
            failed_chunks.append((chunk_start, chunk_end))

    # Final verification and reporting
    log.info(
        f"Collection completed: {successful_chunks}/{len(chunks)} chunks successful"
    )
    if failed_chunks:
        log.warning(f"Failed chunks: {failed_chunks}")

    stats = verify_data_integrity(data_root)
    log.info(f"Final data statistics: {stats}")

    # Save collection summary
    summary = {
        "collection_date": datetime.now().isoformat(),
        "date_range": f"{args.start_date} to {args.end_date}",
        "cities_count": len(CITIES_100),
        "successful_chunks": successful_chunks,
        "total_chunks": len(chunks),
        "failed_chunks": failed_chunks,
        "data_statistics": stats,
    }

    summary_file = Path(data_root) / "logs" / "gefs_2year_collection_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Collection summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
