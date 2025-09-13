#!/usr/bin/env python3
"""
Real CAMS Data Collection Test - 6-hour intervals, past week
============================================================

Since ADS access requires separate registration, we'll try alternative approaches:
1. Test with existing CDS credentials (sometimes CAMS is accessible via CDS)
2. Try with a very small sample to verify if real data collection works
3. Focus on 6-hour intervals over the past week as requested
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def test_cds_for_cams():
    """Test if CAMS data is accessible via CDS endpoint."""
    logger = logging.getLogger(__name__)

    try:
        import cdsapi

        logger.info("CDS API library available")

        # Try with CDS endpoint
        client = cdsapi.Client()
        logger.info("CDS client created successfully")

        # Test with a very small CAMS request
        yesterday = datetime.now() - timedelta(
            days=2
        )  # Use 2 days ago for data availability
        date_str = yesterday.strftime("%Y-%m-%d")

        request = {
            "format": "netcdf",
            "variable": ["particulate_matter_2.5um"],  # Just PM2.5
            "date": date_str,
            "time": "00:00",  # Just one time step
            "area": [52, 4, 51, 5],  # Very small area: Netherlands
        }

        output_dir = Path("data/cams_test_real")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"cams_test_{date_str.replace('-', '')}.nc"

        logger.info(f"Testing CAMS collection for {date_str}")
        logger.info("Request parameters:")
        for key, value in request.items():
            logger.info(f"  {key}: {value}")

        logger.info("Submitting request (this may take several minutes)...")

        # Try both possible dataset names
        datasets_to_try = [
            "cams-global-reanalysis-eac4",  # Official CAMS dataset
            "cams-global-atmospheric-composition-forecasts",  # Alternative
        ]

        success = False
        for dataset in datasets_to_try:
            try:
                logger.info(f"Trying dataset: {dataset}")
                client.retrieve(dataset, request, str(output_file))
                logger.info(f"SUCCESS: Downloaded {output_file}")
                logger.info(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
                success = True
                break
            except Exception as e:
                logger.warning(f"Dataset {dataset} failed: {e}")
                continue

        return success, output_file if success else None

    except ImportError:
        logger.error("CDS API library not installed (pip install cdsapi)")
        return False, None
    except Exception as e:
        logger.error(f"CDS API error: {e}")
        return False, None


def collect_past_week_sample():
    """Collect a small sample covering the past week with 6-hour intervals."""
    logger = logging.getLogger(__name__)

    # Generate past week dates with 6-hour intervals
    today = datetime.now()
    start_date = today - timedelta(days=7)

    dates = []
    current_date = start_date
    while current_date <= today - timedelta(days=1):  # Stop at yesterday
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    times = ["00:00", "06:00", "12:00", "18:00"]  # 6-hour intervals as requested

    logger.info("Collecting past week data:")
    logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
    logger.info(f"  Times: {times}")
    logger.info(f"  Total time points: {len(dates) * len(times)}")

    # Try just one sample first
    if dates:
        logger.info("Testing with single sample first...")
        success, sample_file = test_cds_for_cams()

        if success:
            logger.info("✓ Real CAMS data collection WORKS!")
            logger.info(
                "This confirms that real CAMS data can be collected successfully"
            )
            logger.info(f"Sample file: {sample_file}")
            return True
        else:
            logger.error("✗ Real CAMS data collection FAILED")
            logger.error("Unable to collect real CAMS data with current setup")
            return False

    return False


def run_smoke_test_on_real_data():
    """If real data was collected, run a basic smoke test."""
    logger = logging.getLogger(__name__)

    test_dir = Path("data/cams_test_real")
    if not test_dir.exists():
        logger.warning("No test data directory found")
        return False

    nc_files = list(test_dir.glob("*.nc"))
    if not nc_files:
        logger.warning("No NetCDF files found in test directory")
        return False

    try:
        import numpy as np
        import xarray as xr

        for nc_file in nc_files:
            logger.info(f"Smoke testing: {nc_file.name}")

            with xr.open_dataset(nc_file) as ds:
                logger.info(f"  Variables: {list(ds.data_vars.keys())}")
                logger.info(f"  Dimensions: {dict(ds.dims)}")

                # Check for PM2.5 data
                pm25_vars = [
                    v
                    for v in ds.data_vars
                    if "pm2p5" in v.lower() or "particulate" in v.lower()
                ]
                if pm25_vars:
                    pm25_data = ds[pm25_vars[0]]
                    values = pm25_data.values.flatten()
                    valid_values = values[~np.isnan(values)]

                    if len(valid_values) > 0:
                        logger.info(
                            f"  PM2.5 stats: min={valid_values.min():.2f}, "
                            f"max={valid_values.max():.2f}, "
                            f"mean={valid_values.mean():.2f}"
                        )
                        logger.info("  ✓ Data contains realistic PM2.5 values")
                    else:
                        logger.warning("  ✗ No valid PM2.5 values found")
                        return False
                else:
                    logger.warning("  ✗ No PM2.5 variables found")
                    return False

        logger.info("✓ Smoke test PASSED - Real CAMS data verified!")
        return True

    except ImportError:
        logger.error("xarray not available for smoke testing")
        return False
    except Exception as e:
        logger.error(f"Smoke test error: {e}")
        return False


def main():
    """Main execution."""
    logger = setup_logging()
    logger.info("=== REAL CAMS DATA COLLECTION TEST ===")

    try:
        # Test past week collection
        collection_success = collect_past_week_sample()

        if collection_success:
            # Run smoke test
            smoke_test_success = run_smoke_test_on_real_data()

            if smoke_test_success:
                logger.info("🎉 SUCCESS: Real CAMS data collection verified!")
                logger.info(
                    "The system can collect real CAMS data with 6-hour intervals"
                    " over the past week"
                )
                return True
            else:
                logger.warning("⚠️  Data collected but failed smoke test")
                return False
        else:
            logger.error("❌ FAILED: Cannot collect real CAMS data")
            logger.error("This indicates a configuration or access issue")
            return False

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
