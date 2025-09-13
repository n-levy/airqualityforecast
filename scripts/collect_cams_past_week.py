#!/usr/bin/env python3
"""
Collect CAMS reanalysis data for all 100 cities for the past week with 6-hour intervals.

This script downloads CAMS global reanalysis (EAC4) data for all pollutants
(PM2.5, PM10, NO2, O3, SO2, CO) for all 100 cities over the past 7 days
at 6-hour intervals (00:00, 06:00, 12:00, 18:00).
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time as time_module

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cams_ads_downloader import CAMSADSDownloader  # noqa: E402


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cams_past_week_collection_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_cities_data():
    """Load the 100 cities from the comprehensive features table."""
    cities_file = Path("stage_5/comprehensive_tables/comprehensive_features_table.csv")

    if not cities_file.exists():
        raise FileNotFoundError(f"Cities data file not found: {cities_file}")

    df = pd.read_csv(cities_file)

    # Extract relevant columns
    cities = df[["City", "Country", "Continent", "Latitude", "Longitude"]].copy()

    logging.info(f"Loaded {len(cities)} cities from {cities_file}")
    return cities


def generate_date_time_combinations():
    """Generate date and time combinations for the past week with 6-hour intervals."""
    today = datetime.now()
    start_date = today - timedelta(days=7)

    dates = []
    current_date = start_date
    while current_date <= today:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    times = ["00:00", "06:00", "12:00", "18:00"]  # 6-hour intervals

    logging.info(f"Date range: {dates[0]} to {dates[-1]}")
    logging.info(f"Times: {times}")
    logging.info(f"Total time points: {len(dates) * len(times)}")

    return dates, times


def get_pollutant_variables():
    """Get the list of pollutant variables to collect from CAMS."""
    variables = [
        "particulate_matter_2.5um",  # PM2.5
        "particulate_matter_10um",  # PM10
        "nitrogen_dioxide",  # NO2
        "ozone",  # O3
        "sulphur_dioxide",  # SO2
        "carbon_monoxide",  # CO
    ]

    logging.info(f"Pollutant variables: {variables}")
    return variables


def create_city_bounding_box(lat, lon, buffer=0.1):
    """Create a small bounding box around a city point."""
    # Small buffer to get point data while satisfying CAMS area requirements
    north = lat + buffer
    south = lat - buffer
    west = lon - buffer
    east = lon + buffer

    return [north, west, south, east]


def collect_city_data(downloader, city_info, dates, times, variables, output_dir):
    """Collect CAMS data for a single city."""
    city_name = city_info["City"]
    country = city_info["Country"]
    lat = city_info["Latitude"]
    lon = city_info["Longitude"]

    logger = logging.getLogger(__name__)

    # Create city-specific output directory
    safe_city_name = "".join(
        c for c in f"{city_name}_{country}" if c.isalnum() or c in "._-"
    )
    city_output_dir = output_dir / safe_city_name
    city_output_dir.mkdir(parents=True, exist_ok=True)

    # Create bounding box
    area = create_city_bounding_box(lat, lon)

    logger.info(f"Collecting data for {city_name}, {country} (lat={lat}, lon={lon})")
    logger.info(f"Bounding box: {area}")

    collected_files = []
    total_attempts = len(dates) * len(times)
    successful_downloads = 0

    for date in dates:
        for time in times:
            try:
                # Create output filename
                timestamp = f"{date}_{time.replace(':', '')}"
                output_file = (
                    city_output_dir / f"cams_data_{safe_city_name}_{timestamp}.nc"
                )

                # Skip if file already exists and is valid
                if output_file.exists():
                    logger.info(f"File already exists: {output_file.name}")
                    collected_files.append(str(output_file))
                    successful_downloads += 1
                    continue

                # Download data
                logger.info(f"Downloading {date} {time} for {city_name}")

                result_file = downloader.download_reanalysis(
                    dates=[date],
                    times=[time],
                    variables=variables,
                    area=area,
                    output_file=str(output_file),
                )

                collected_files.append(result_file)
                successful_downloads += 1
                logger.info(f"Successfully downloaded: {output_file.name}")

                # Small delay to avoid overwhelming the API
                time_module.sleep(2)

            except Exception as e:
                logger.error(f"Failed to download {date} {time} for {city_name}: {e}")
                continue

    success_rate = successful_downloads / total_attempts
    logger.info(
        f"City {city_name} collection complete: "
        f"{successful_downloads}/{total_attempts} files ({success_rate:.1%})"
    )

    return {
        "city": city_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "files_collected": collected_files,
        "success_count": successful_downloads,
        "total_attempts": total_attempts,
        "success_rate": success_rate,
    }


def save_collection_summary(results, output_dir):
    """Save a summary of the collection process."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"collection_summary_{timestamp}.json"

    # Calculate overall statistics
    total_cities = len(results)
    total_files = sum(r["success_count"] for r in results)
    total_attempts = sum(r["total_attempts"] for r in results)
    overall_success_rate = total_files / total_attempts if total_attempts > 0 else 0

    summary = {
        "collection_timestamp": timestamp,
        "total_cities": total_cities,
        "total_files_collected": total_files,
        "total_attempts": total_attempts,
        "overall_success_rate": overall_success_rate,
        "pollutant_variables": get_pollutant_variables(),
        "date_range": (
            f"{generate_date_time_combinations()[0][0]} to "
            f"{generate_date_time_combinations()[0][-1]}"
        ),
        "time_intervals": generate_date_time_combinations()[1],
        "city_results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Collection summary saved to: {summary_file}")
    return summary


def main():
    """Main execution function."""
    logger = setup_logging()
    logger.info("Starting CAMS past week data collection for all 100 cities")

    try:
        # Initialize downloader
        logger.info("Initializing CAMS ADS downloader...")
        downloader = CAMSADSDownloader()

        # Load cities data
        logger.info("Loading cities data...")
        cities = load_cities_data()

        # Generate date/time combinations
        logger.info("Generating date/time combinations...")
        dates, times = generate_date_time_combinations()

        # Get pollutant variables
        variables = get_pollutant_variables()

        # Create output directory
        output_dir = Path("data/cams_past_week_collection")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Collect data for all cities
        logger.info(f"Starting collection for {len(cities)} cities...")
        results = []

        for i, (_, city_info) in enumerate(cities.iterrows(), 1):
            logger.info(
                f"Processing city {i}/{len(cities)}: {city_info['City']}, {city_info['Country']}"
            )

            try:
                result = collect_city_data(
                    downloader, city_info, dates, times, variables, output_dir
                )
                results.append(result)

                # Progress update
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(cities)} cities completed")

            except Exception as e:
                logger.error(f"Failed to process city {city_info['City']}: {e}")
                continue

        # Save collection summary
        logger.info("Saving collection summary...")
        summary = save_collection_summary(results, output_dir)

        # Final summary
        logger.info("=== COLLECTION COMPLETE ===")
        logger.info(f"Cities processed: {summary['total_cities']}")
        logger.info(f"Files collected: {summary['total_files_collected']}")
        logger.info(f"Success rate: {summary['overall_success_rate']:.1%}")
        logger.info(f"Output directory: {output_dir}")

        return True

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

