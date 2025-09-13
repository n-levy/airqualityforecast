#!/usr/bin/env python3
"""
Test CAMS collection with a small sample (3 cities, 2 time points) to verify implementation.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cams_ads_downloader import CAMSADSDownloader  # noqa: E402


def test_sample_collection():
    """Test collection with a small sample."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Testing CAMS collection with small sample")

    try:
        # Initialize downloader
        downloader = CAMSADSDownloader()

        # Load cities (take first 3)
        cities_file = Path(
            "../stage_5/comprehensive_tables/comprehensive_features_table.csv"
        )
        df = pd.read_csv(cities_file)
        sample_cities = df[["City", "Country", "Latitude", "Longitude"]].head(3)

        # Use just 2 recent time points
        today = datetime.now()
        dates = [(today - timedelta(days=1)).strftime("%Y-%m-%d")]
        times = ["00:00", "12:00"]

        variables = [
            "particulate_matter_2.5um",
            "nitrogen_dioxide",
        ]  # Just 2 variables for test

        logger.info(
            f"Testing with {len(sample_cities)} cities, {len(dates)} dates, {len(times)} times"
        )
        logger.info(f"Variables: {variables}")

        # Create output directory
        output_dir = Path("data/cams_test_sample")
        output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        total_attempts = len(sample_cities) * len(dates) * len(times)

        for _, city_info in sample_cities.iterrows():
            city_name = city_info["City"]
            lat = city_info["Latitude"]
            lon = city_info["Longitude"]

            logger.info(f"Testing {city_name} (lat={lat}, lon={lon})")

            # Create small bounding box
            buffer = 0.1
            area = [lat + buffer, lon - buffer, lat - buffer, lon + buffer]

            for date in dates:
                for time in times:
                    try:
                        output_file = (
                            output_dir
                            / f"test_{city_name}_{date}_{time.replace(':', '')}.nc"
                        )

                        logger.info(f"Downloading {date} {time} for {city_name}")

                        result_file = downloader.download_reanalysis(
                            dates=[date],
                            times=[time],
                            variables=variables,
                            area=area,
                            output_file=str(output_file),
                        )

                        logger.info(f"SUCCESS: {result_file}")
                        success_count += 1

                    except Exception as e:
                        logger.error(f"FAILED {date} {time} for {city_name}: {e}")

        success_rate = success_count / total_attempts
        logger.info(
            f"Sample test results: {success_count}/{total_attempts} successful ({success_rate:.1%})"
        )

        if success_rate > 0:
            logger.info("✓ Sample test PASSED - implementation works!")
            return True
        else:
            logger.error("✗ Sample test FAILED - no successful downloads")
            return False

    except Exception as e:
        logger.error(f"Sample test error: {e}")
        return False


if __name__ == "__main__":
    success = test_sample_collection()
    sys.exit(0 if success else 1)
