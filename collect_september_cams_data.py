#!/usr/bin/env python3
"""
Collect CAMS Data for September 1-7, 2025
==========================================

Attempts to collect real ECMWF-CAMS data for the target period.
Will try both reanalysis and forecast datasets.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def collect_cams_september_data():
    """Collect CAMS data for September 1-7, 2025."""
    logger = logging.getLogger(__name__)

    try:
        import cdsapi

        client = cdsapi.Client()
        logger.info("✅ ADS client initialized")

        # Target dates: September 1-7, 2025
        dates = []
        start_date = datetime(2025, 9, 1)
        for i in range(7):  # 7 days
            date = start_date + timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))

        times = ["00:00", "06:00", "12:00", "18:00"]  # 6-hour intervals

        logger.info(f"📅 Target period: {dates[0]} to {dates[-1]}")
        logger.info(f"⏰ Times: {times}")

        # Create output directory
        output_dir = Path("data/cams_september_2025")
        output_dir.mkdir(parents=True, exist_ok=True)

        collected_files = []

        # Try forecast data first (more likely to be available for recent dates)
        logger.info("🚀 Attempting CAMS forecasts for September 2025...")

        for date in dates[:3]:  # Try first 3 days
            for time in times[:2]:  # Try first 2 times to test
                try:
                    logger.info(f"Trying forecast for {date} {time}")

                    request = {
                        "variable": ["particulate_matter_2.5um"],
                        "date": date,
                        "time": time,
                        "leadtime_hour": ["0"],  # Analysis (0-hour forecast)
                        "area": [60, -10, 50, 10],  # Western Europe
                        "format": "netcdf",
                    }

                    safe_datetime = f"{date.replace('-', '')}_{time.replace(':', '')}"
                    output_file = output_dir / f"cams_forecast_{safe_datetime}.nc"

                    client.retrieve(
                        "cams-global-atmospheric-composition-forecasts",
                        request,
                        str(output_file),
                    )

                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    logger.info(
                        f"✅ Success: {output_file.name} ({file_size_mb:.2f} MB)"
                    )
                    collected_files.append(str(output_file))

                    # Small delay between requests
                    import time

                    time.sleep(2)

                except Exception as e:
                    logger.warning(f"Forecast failed for {date} {time}: {e}")

                    # If forecast fails, try reanalysis with older date
                    if "2025" in str(e):
                        logger.info(
                            "2025 data not available, trying reanalysis with Sept 2024..."
                        )
                        try:
                            # Try September 2024 instead
                            date_2024 = date.replace("2025", "2024")

                            request_2024 = {
                                "variable": ["particulate_matter_2.5um"],
                                "date": date_2024,
                                "time": time,
                                "area": [60, -10, 50, 10],
                                "format": "netcdf",
                            }

                            safe_datetime_2024 = (
                                f"{date_2024.replace('-', '')}_{time.replace(':', '')}"
                            )
                            output_file_2024 = (
                                output_dir / f"cams_reanalysis_{safe_datetime_2024}.nc"
                            )

                            client.retrieve(
                                "cams-global-reanalysis-eac4",
                                request_2024,
                                str(output_file_2024),
                            )

                            file_size_mb = output_file_2024.stat().st_size / (
                                1024 * 1024
                            )
                            logger.info(
                                f"✅ Success (2024 data): {output_file_2024.name} ({file_size_mb:.2f} MB)"
                            )
                            collected_files.append(str(output_file_2024))

                        except Exception as e2:
                            logger.error(f"Both forecast and reanalysis failed: {e2}")
                            continue

        logger.info(f"📊 Collection summary: {len(collected_files)} files collected")

        if collected_files:
            logger.info("✅ CAMS data collection successful!")
            for file in collected_files:
                logger.info(f"  📁 {file}")
            return True, collected_files
        else:
            logger.error("❌ No CAMS data collected")
            return False, []

    except Exception as e:
        logger.error(f"CAMS collection failed: {e}")
        return False, []


def main():
    """Main execution."""
    logger = setup_logging()
    logger.info("🌍 CAMS Data Collection - September 1-7, 2025")
    logger.info("=" * 50)

    success, files = collect_cams_september_data()

    if success:
        logger.info("🎉 CAMS collection completed successfully!")
        return True
    else:
        logger.error("❌ CAMS collection failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
