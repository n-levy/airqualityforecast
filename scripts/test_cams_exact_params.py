#!/usr/bin/env python3
"""
Test CAMS with exact parameter formats based on ADS documentation.
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


def test_cams_forecasts():
    """Test CAMS forecasts with exact parameters."""
    logger = logging.getLogger(__name__)

    try:
        import cdsapi

        client = cdsapi.Client()
        logger.info("✅ ADS client ready")

        # Use a date that should definitely be available
        test_date = datetime.now() - timedelta(days=2)
        date_str = test_date.strftime("%Y-%m-%d")

        logger.info(f"Testing CAMS forecasts for {date_str}")

        # Try minimal parameters first
        request = {
            "variable": ["particulate_matter_2.5um"],  # Back to list format
            "date": date_str,
            "time": "00:00",
            "leadtime_hour": ["0"],  # List format
            "format": "netcdf",
        }

        # Remove area for global test

        output_dir = Path("data/cams_exact_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"cams_global_{date_str.replace('-', '')}.nc"

        logger.info("Request parameters (global, minimal):")
        for key, value in request.items():
            logger.info(f"  {key}: {value}")

        logger.info("🚀 Submitting global request...")

        try:
            client.retrieve(
                "cams-global-atmospheric-composition-forecasts",
                request,
                str(output_file),
            )

            logger.info(f"🎉 SUCCESS: Global CAMS data downloaded!")
            logger.info(f"📁 File: {output_file}")
            logger.info(f"📏 Size: {output_file.stat().st_size / (1024*1024):.2f} MB")

            return True, output_file

        except Exception as e:
            logger.error(f"Global request failed: {e}")

            # Try with area but different format
            request_with_area = {
                "variable": ["particulate_matter_2.5um"],
                "date": date_str,
                "time": "00:00",
                "leadtime_hour": ["0"],
                "area": [60, -10, 50, 10],  # Larger area covering Western Europe
                "format": "netcdf",
            }

            logger.info("Trying with larger European area...")
            output_file_area = (
                output_dir / f"cams_europe_{date_str.replace('-', '')}.nc"
            )

            client.retrieve(
                "cams-global-atmospheric-composition-forecasts",
                request_with_area,
                str(output_file_area),
            )

            logger.info(f"🎉 SUCCESS: European CAMS data downloaded!")
            logger.info(f"📁 File: {output_file_area}")
            logger.info(
                f"📏 Size: {output_file_area.stat().st_size / (1024*1024):.2f} MB"
            )

            return True, output_file_area

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False, None


def test_cams_reanalysis_old_date():
    """Test CAMS reanalysis with a date we know should work."""
    logger = logging.getLogger(__name__)

    try:
        import cdsapi

        client = cdsapi.Client()

        # Use a date from mid-2024 that should be in reanalysis
        test_date = datetime(2024, 6, 1)
        date_str = test_date.strftime("%Y-%m-%d")

        logger.info(f"Testing CAMS reanalysis for {date_str}")

        request = {
            "variable": ["particulate_matter_2.5um"],
            "date": date_str,
            "time": "00:00",
            "area": [60, -10, 50, 10],  # Western Europe
            "format": "netcdf",
        }

        output_dir = Path("data/cams_exact_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"cams_reanalysis_{date_str.replace('-', '')}.nc"

        logger.info("Request parameters (reanalysis, June 2024):")
        for key, value in request.items():
            logger.info(f"  {key}: {value}")

        logger.info("🚀 Submitting reanalysis request...")

        client.retrieve("cams-global-reanalysis-eac4", request, str(output_file))

        logger.info(f"🎉 SUCCESS: CAMS reanalysis data downloaded!")
        logger.info(f"📁 File: {output_file}")
        logger.info(f"📏 Size: {output_file.stat().st_size / (1024*1024):.2f} MB")

        return True, output_file

    except Exception as e:
        logger.error(f"❌ Reanalysis test failed: {e}")
        return False, None


def verify_cams_data(nc_file):
    """Verify the CAMS data content."""
    logger = logging.getLogger(__name__)

    try:
        import numpy as np
        import xarray as xr

        logger.info(f"🔍 Verifying CAMS data: {nc_file.name}")

        with xr.open_dataset(nc_file) as ds:
            logger.info("📊 Dataset structure:")
            logger.info(f"  Variables: {list(ds.data_vars.keys())}")
            logger.info(f"  Dimensions: {dict(ds.dims)}")
            logger.info(f"  Coordinates: {list(ds.coords.keys())}")

            if hasattr(ds, "time"):
                logger.info(f"  Time: {ds.time.values}")

            # Get first data variable
            first_var = list(ds.data_vars.keys())[0]
            data = ds[first_var]
            values = data.values.flatten()
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                logger.info(f"🎉 REAL CAMS DATA VERIFIED!")
                logger.info(f"  Variable: {first_var}")
                logger.info(f"  Valid points: {len(valid_values):,}")
                logger.info(
                    f"  Value range: {valid_values.min():.8f} to {valid_values.max():.8f}"
                )
                logger.info(f"  Mean: {valid_values.mean():.8f}")

                # Check units and attributes
                if hasattr(data, "units"):
                    logger.info(f"  Units: {data.units}")
                if hasattr(data, "long_name"):
                    logger.info(f"  Description: {data.long_name}")

                return True
            else:
                logger.error("❌ No valid data found")
                return False

    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


def main():
    """Main execution."""
    logger = setup_logging()
    logger.info("🧪 Testing CAMS with Exact Parameters")
    logger.info("=" * 50)

    # Try forecasts first (more recent data)
    logger.info("1️⃣  Testing CAMS forecasts...")
    success1, file1 = test_cams_forecasts()

    if success1 and file1:
        if verify_cams_data(file1):
            logger.info("🎉 SUCCESS: Real CAMS forecast data verified!")
            return True

    # If forecasts fail, try reanalysis with old date
    logger.info("\n2️⃣  Testing CAMS reanalysis with historical date...")
    success2, file2 = test_cams_reanalysis_old_date()

    if success2 and file2:
        if verify_cams_data(file2):
            logger.info("🎉 SUCCESS: Real CAMS reanalysis data verified!")
            return True

    logger.error("❌ All tests failed")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
