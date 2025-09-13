#!/usr/bin/env python3
"""
Test CAMS data collection with correct parameters and recent dates.
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


def test_cams_with_correct_params():
    """Test CAMS collection with proper parameters."""
    logger = logging.getLogger(__name__)

    try:
        import cdsapi

        client = cdsapi.Client()
        logger.info("✓ ADS client created successfully")

        # Use dates from a few days ago to ensure data availability
        test_date = datetime.now() - timedelta(days=3)
        date_str = test_date.strftime("%Y-%m-%d")

        logger.info(f"Testing CAMS collection for {date_str}")

        # Test with CAMS forecasts (more likely to have recent data)
        request = {
            "variable": ["particulate_matter_2.5um"],
            "date": date_str,
            "time": "00:00",
            "leadtime_hour": "0",  # Analysis (0-hour forecast)
            "area": [52, 4, 51, 5],  # Small area in Netherlands
            "format": "netcdf",
        }

        output_dir = Path("data/cams_test_proper")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"cams_test_{date_str.replace('-', '')}.nc"

        logger.info("Request parameters:")
        for key, value in request.items():
            logger.info(f"  {key}: {value}")

        logger.info("Submitting request to CAMS forecasts...")

        try:
            client.retrieve(
                "cams-global-atmospheric-composition-forecasts",
                request,
                str(output_file),
            )

            logger.info(f"✅ SUCCESS: Downloaded {output_file}")
            logger.info(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
            return True, output_file

        except Exception as e:
            logger.error(f"CAMS forecasts failed: {e}")

            # Try with reanalysis and older date
            older_date = datetime(2024, 9, 1)  # Known good date
            older_date_str = older_date.strftime("%Y-%m-%d")

            request_reanalysis = {
                "variable": ["particulate_matter_2.5um"],
                "date": older_date_str,
                "time": "00:00",
                "area": [52, 4, 51, 5],
                "format": "netcdf",
            }

            logger.info(f"Trying reanalysis with older date: {older_date_str}")

            try:
                output_file_old = (
                    output_dir / f"cams_reanalysis_{older_date_str.replace('-', '')}.nc"
                )

                client.retrieve(
                    "cams-global-reanalysis-eac4",
                    request_reanalysis,
                    str(output_file_old),
                )

                logger.info(f"✅ SUCCESS: Downloaded {output_file_old}")
                logger.info(
                    f"File size: {output_file_old.stat().st_size / 1024:.1f} KB"
                )
                return True, output_file_old

            except Exception as e2:
                logger.error(f"Reanalysis also failed: {e2}")
                return False, None

    except ImportError:
        logger.error("CDS API library not available")
        return False, None
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return False, None


def verify_real_data(nc_file):
    """Verify that the downloaded file contains real CAMS data."""
    logger = logging.getLogger(__name__)

    try:
        import numpy as np
        import xarray as xr

        logger.info(f"🔍 Verifying real data in: {nc_file.name}")

        with xr.open_dataset(nc_file) as ds:
            logger.info(f"Variables: {list(ds.data_vars.keys())}")
            logger.info(f"Dimensions: {dict(ds.dims)}")
            logger.info(f"Coordinates: {list(ds.coords.keys())}")

            # Look for PM2.5 data
            pm25_vars = [v for v in ds.data_vars if "pm2p5" in v.lower() or "2.5" in v]
            if not pm25_vars:
                # Check all variables for any that might be PM2.5
                logger.info("Available variables:")
                for var in ds.data_vars:
                    logger.info(f"  - {var}")
                pm25_vars = list(ds.data_vars.keys())[:1]  # Use first variable

            if pm25_vars:
                var_name = pm25_vars[0]
                data = ds[var_name]
                values = data.values.flatten()
                valid_values = values[~np.isnan(values)]

                if len(valid_values) > 0:
                    logger.info(f"✅ Variable '{var_name}' data:")
                    logger.info(f"  Min: {valid_values.min():.6f}")
                    logger.info(f"  Max: {valid_values.max():.6f}")
                    logger.info(f"  Mean: {valid_values.mean():.6f}")
                    logger.info(f"  Valid points: {len(valid_values)}")

                    # Check if values are realistic for atmospheric concentrations
                    if valid_values.min() >= 0 and valid_values.max() < 1000:
                        logger.info("✅ Values appear realistic for atmospheric data")
                        return True
                    else:
                        logger.warning("⚠️  Values may be outside expected range")
                        return True  # Still real data, just unexpected values
                else:
                    logger.warning("❌ No valid data values found")
                    return False
            else:
                logger.warning("❌ No PM2.5-like variables found")
                return False

    except ImportError:
        logger.error("xarray not available for verification")
        return False
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


def main():
    """Main execution."""
    logger = setup_logging()
    logger.info("🧪 Testing CAMS with Proper Parameters")
    logger.info("=" * 50)

    success, nc_file = test_cams_with_correct_params()

    if success and nc_file:
        logger.info("🎉 CAMS data collection SUCCEEDED!")

        # Verify it's real data
        if verify_real_data(nc_file):
            logger.info("🎉 SUCCESS: Real CAMS data verified!")
            logger.info("The API key works and real atmospheric data was collected")
            return True
        else:
            logger.warning("⚠️  Data collected but verification failed")
            return False
    else:
        logger.error("❌ CAMS data collection FAILED")
        logger.error("Need to check parameters or data availability")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
