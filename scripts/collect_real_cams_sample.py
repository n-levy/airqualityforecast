#!/usr/bin/env python3
"""
Real ECMWF CAMS Data Collection (Limited Sample)
===============================================

Attempts to collect a small sample of real ECMWF CAMS data via the CDS API.
Limited to a small geographic area and short time period to minimize download time.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logs_dir = Path("C:/aqf311/data/logs")
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "real_cams_collection.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_cams_api_access():
    """Test if CAMS CDS API is properly configured."""
    try:
        import cdsapi

        log.info("CDS API library available")

        # Try to create client
        c = cdsapi.Client()
        log.info("CDS API client created successfully")

        # Check if configuration exists
        config_file = Path.home() / ".cdsapirc"
        if config_file.exists():
            log.info(f"CDS API configuration found: {config_file}")
            return True, c
        else:
            log.warning("CDS API configuration file not found")
            log.warning("To use real CAMS data, you need:")
            log.warning("1. Register at https://cds.climate.copernicus.eu/")
            log.warning("2. Get your API key")
            log.warning("3. Create ~/.cdsapirc with your credentials")
            return False, None

    except ImportError:
        log.error("CDS API library not installed (pip install cdsapi)")
        return False, None
    except Exception as e:
        log.error(f"CDS API error: {e}")
        return False, None


def collect_cams_sample(data_root):
    """Attempt to collect a small CAMS data sample."""
    log.info("=== ATTEMPTING REAL CAMS DATA COLLECTION ===")

    # Test API access first
    api_available, client = test_cams_api_access()

    if not api_available or not client:
        log.error("CAMS API not properly configured - cannot collect real data")
        return None

    # Calculate yesterday's date (CAMS reanalysis has a delay)
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")

    log.info(f"Attempting to collect CAMS data for {date_str}")

    # Very small request - just Europe, PM2.5 only, one time step
    request_params = {
        "format": "netcdf",
        "variable": ["particulate_matter_2.5um"],  # Just PM2.5
        "date": date_str,
        "time": "00:00",  # Just one time step
        "type": "analysis",  # Analysis data (not forecast)
        "area": [55, 0, 50, 10],  # Small area: North Germany/Netherlands
    }

    log.info("CAMS request parameters:")
    for key, value in request_params.items():
        log.info(f"  {key}: {value}")

    output_dir = Path(data_root) / "raw" / "cams"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"cams_sample_{date_str.replace('-', '')}.nc"

    try:
        log.info("Submitting CAMS data request...")
        log.info("NOTE: This may take 5-30 minutes depending on queue")
        log.info(
            "The request will be queued at ECMWF and processed when resources are available"
        )

        # This is the actual API call that retrieves real CAMS data
        client.retrieve("cams-global-reanalysis-eac4", request_params, str(output_file))

        log.info(f"CAMS data downloaded successfully: {output_file}")
        log.info(f"File size: {output_file.stat().st_size / (1024**2):.1f} MB")

        return output_file

    except Exception as e:
        log.error(f"CAMS data collection failed: {e}")

        if "Missing/incomplete configuration" in str(e):
            log.error("CDS API credentials are not properly configured")
            log.error("Please set up your CDS API key at ~/.cdsapirc")
        elif "Invalid request" in str(e):
            log.error("Request parameters may be invalid")
        elif "Request timeout" in str(e):
            log.error("Request timed out - CAMS queue may be busy")

        return None


def main():
    """Main execution."""
    data_root = os.environ.get("DATA_ROOT", "C:/aqf311/data")

    log.info("Starting real CAMS data collection attempt...")

    try:
        output_file = collect_cams_sample(data_root)

        if output_file:
            log.info("CAMS data collection completed successfully!")
            log.info(f"Output file: {output_file}")
            return True
        else:
            log.warning("CAMS data collection failed or not configured")
            log.warning("Real CAMS data requires:")
            log.warning("1. ECMWF CDS account registration")
            log.warning("2. API key configuration")
            log.warning("3. Significant download time")
            return False

    except Exception as e:
        log.error(f"Collection failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
