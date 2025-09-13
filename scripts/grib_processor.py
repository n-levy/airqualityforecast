#!/usr/bin/env python3
"""
GRIB File Processor for Air Quality Data
========================================

Extract PM2.5, PM10, and other air quality variables from GRIB2 files
downloaded from NOMADS and other sources.
"""

import logging
import warnings
from pathlib import Path

# import numpy as np  # Available for future use

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    log.warning("xarray not available for GRIB processing")
    XARRAY_AVAILABLE = False

try:
    import cfgrib  # noqa: F401

    CFGRIB_AVAILABLE = True
except ImportError:
    log.warning("cfgrib not available for GRIB processing")
    CFGRIB_AVAILABLE = False


class GRIBProcessor:
    """Process GRIB files to extract air quality data."""

    def __init__(self):
        """Initialize processor."""
        self.available = XARRAY_AVAILABLE and CFGRIB_AVAILABLE

        # Variable name mappings for GEFS-Aerosols
        self.var_mappings = {
            "pm25": [
                "PMTF",
                "pm2p5",
                "pm25",
                "particulate_matter_2.5um",
                "PMTF_surface",
            ],
            "pm10": ["PMTC", "pm10", "particulate_matter_10um", "PMTC_surface"],
            "no2": ["NO2", "nitrogen_dioxide", "NO2_surface", "OZCON1"],
            "o3": ["O3", "ozone", "OZCON", "OZCON_surface", "ozone_concentration"],
        }

    def extract_point_data(self, grib_file, lat, lon, variables=None):
        """Extract data at specific lat/lon point from GRIB file."""
        if not self.available:
            log.warning("GRIB processing not available")
            return {}

        if variables is None:
            variables = ["pm25", "pm10", "no2", "o3"]

        try:
            # Open GRIB file with xarray
            ds = xr.open_dataset(grib_file, engine="cfgrib")

            results = {}
            available_vars = list(ds.data_vars.keys())
            log.debug(f"Available variables in GRIB: {available_vars}")

            for var_name in variables:
                var_found = None

                # Try to find the variable by different names
                for possible_name in self.var_mappings.get(var_name, [var_name]):
                    for grib_var in available_vars:
                        if possible_name.lower() in grib_var.lower():
                            var_found = grib_var
                            break
                    if var_found:
                        break

                if var_found:
                    try:
                        # Extract data at nearest grid point
                        data_var = ds[var_found]

                        # Handle different coordinate names
                        lat_coord = None
                        lon_coord = None

                        for coord in data_var.coords:
                            if "lat" in coord.lower():
                                lat_coord = coord
                            elif "lon" in coord.lower():
                                lon_coord = coord

                        if lat_coord and lon_coord:
                            point_data = data_var.sel(
                                {lat_coord: lat, lon_coord: lon}, method="nearest"
                            )

                            # Convert to float
                            value = float(point_data.values)
                            results[var_name] = value

                            log.debug(f"Extracted {var_name}: {value}")
                        else:
                            log.warning(
                                f"Could not find lat/lon coordinates for {var_found}"
                            )
                            results[var_name] = None

                    except Exception as e:
                        log.warning(f"Error extracting {var_name}: {e}")
                        results[var_name] = None
                else:
                    log.debug(f"Variable {var_name} not found in GRIB file")
                    results[var_name] = None

            return results

        except Exception as e:
            log.error(f"Error processing GRIB file {grib_file}: {e}")
            return {var: None for var in variables}

    def get_file_info(self, grib_file):
        """Get basic information about GRIB file."""
        if not self.available:
            return {}

        try:
            ds = xr.open_dataset(grib_file, engine="cfgrib")

            info = {
                "variables": list(ds.data_vars.keys()),
                "coordinates": list(ds.coords.keys()),
                "dims": dict(ds.dims),
                "valid_time": str(ds.attrs.get("valid_time", "unknown")),
                "forecast_reference_time": str(
                    ds.attrs.get("forecast_reference_time", "unknown")
                ),
            }

            return info

        except Exception as e:
            log.error(f"Error getting GRIB file info: {e}")
            return {}


def test_grib_processor():
    """Test GRIB processor with sample data."""
    processor = GRIBProcessor()

    if not processor.available:
        print("GRIB processing not available - install xarray and cfgrib")
        return

    # Test with actual GRIB file if available
    test_file = Path("C:/aqf311/data/gefs_chem_raw/20250912_00_f000_Delhi_PM.grib2")

    if test_file.exists():
        print(f"Testing with {test_file}")

        # Get file info
        info = processor.get_file_info(test_file)
        print(f"File info: {info}")

        # Extract data for Delhi
        data = processor.extract_point_data(test_file, 28.6139, 77.209)
        print(f"Extracted data: {data}")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_grib_processor()
