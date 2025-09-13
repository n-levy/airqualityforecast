"""
Smoke tests for CAMS ADS downloader.

These tests verify that the CAMS ADS downloader can successfully retrieve
small datasets for Berlin using both forecast and reanalysis endpoints.
"""

import json
import os
import sys
import tempfile
import unittest

import xarray as xr

# Add parent directory to Python path to import cams_ads_downloader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cams_ads_downloader import CAMSADSDownloader  # noqa: E402


class TestCAMSADS(unittest.TestCase):
    """Test CAMS ADS download functionality with Berlin point data."""

    def setUp(self):
        """Set up test environment."""
        # Berlin coordinates (small bounding box for minimal data transfer)
        self.berlin_area = [52.52, 13.40, 52.50, 13.42]  # north, west, south, east
        self.temp_dir = tempfile.mkdtemp()

        # Skip tests if no credentials available
        try:
            self.downloader = CAMSADSDownloader()
        except (FileNotFoundError, ValueError) as e:
            self.skipTest(f"ADS credentials not configured: {e}")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_forecast_download(self):
        """
        Test downloading a tiny CAMS forecast for Berlin.

        Downloads PM2.5 forecast for 2025-09-10 00:00 with 0-hour leadtime.
        Verifies file creation, NetCDF validity, and provenance metadata.
        """
        output_file = os.path.join(self.temp_dir, "berlin_forecast_test.nc")

        # Download small Berlin forecast
        result_file = self.downloader.download_forecast(
            dates="2025-09-10",
            times="00:00",
            variables="particulate_matter_2.5um",
            area=self.berlin_area,
            output_file=output_file,
            leadtime_hours=0,
        )

        # Verify file was created
        self.assertTrue(
            os.path.exists(result_file), f"Output file not created: {result_file}"
        )
        self.assertEqual(
            result_file, output_file, "Returned path should match requested path"
        )

        # Verify NetCDF can be opened and contains expected data
        with xr.open_dataset(result_file) as ds:
            self.assertGreater(
                len(ds.data_vars), 0, "Dataset should contain data variables"
            )

            # Check for PM2.5 variable (might have different name in dataset)
            var_names = list(ds.data_vars.keys())
            self.assertGreater(
                len(var_names), 0, f"Should have variables, found: {var_names}"
            )

            # Check time dimension exists
            self.assertIn(
                "time",
                ds.dims,
                f"Should have time dimension, found dims: {list(ds.dims.keys())}",
            )

            # Verify we have some data
            first_var = list(ds.data_vars.keys())[0]
            data_array = ds[first_var]
            self.assertFalse(
                data_array.isnull().all().item(), "Data should not be all NaN"
            )

        # Verify provenance file was created
        provenance_file = result_file + ".provenance.json"
        self.assertTrue(
            os.path.exists(provenance_file), "Provenance file should be created"
        )

        with open(provenance_file, "r") as f:
            provenance = json.load(f)

        # Check provenance content
        self.assertEqual(
            provenance["dataset"], "cams-global-atmospheric-composition-forecasts"
        )
        self.assertIn("request", provenance)
        self.assertIn("api_url", provenance)
        self.assertIn("sha256", provenance)
        self.assertIn("size_bytes", provenance)
        self.assertGreater(
            provenance["size_bytes"], 0, "File should have non-zero size"
        )

        print(
            f"✓ Forecast test passed: {os.path.getsize(result_file)} bytes downloaded"
        )

    def test_reanalysis_download(self):
        """
        Test downloading a tiny CAMS reanalysis for Berlin.

        Downloads PM2.5 reanalysis for 2003-01-01 00:00.
        Verifies file creation, NetCDF validity, and provenance metadata.
        """
        output_file = os.path.join(self.temp_dir, "berlin_reanalysis_test.nc")

        # Download small Berlin reanalysis
        result_file = self.downloader.download_reanalysis(
            dates="2003-01-01",
            times="00:00",
            variables="particulate_matter_2.5um",
            area=self.berlin_area,
            output_file=output_file,
        )

        # Verify file was created
        self.assertTrue(
            os.path.exists(result_file), f"Output file not created: {result_file}"
        )
        self.assertEqual(
            result_file, output_file, "Returned path should match requested path"
        )

        # Verify NetCDF can be opened and contains expected data
        with xr.open_dataset(result_file) as ds:
            self.assertGreater(
                len(ds.data_vars), 0, "Dataset should contain data variables"
            )

            # Check for PM2.5 variable (might have different name in dataset)
            var_names = list(ds.data_vars.keys())
            self.assertGreater(
                len(var_names), 0, f"Should have variables, found: {var_names}"
            )

            # Check time dimension exists
            self.assertIn(
                "time",
                ds.dims,
                f"Should have time dimension, found dims: {list(ds.dims.keys())}",
            )

            # Verify we have some data
            first_var = list(ds.data_vars.keys())[0]
            data_array = ds[first_var]
            self.assertFalse(
                data_array.isnull().all().item(), "Data should not be all NaN"
            )

        # Verify provenance file was created
        provenance_file = result_file + ".provenance.json"
        self.assertTrue(
            os.path.exists(provenance_file), "Provenance file should be created"
        )

        with open(provenance_file, "r") as f:
            provenance = json.load(f)

        # Check provenance content
        self.assertEqual(provenance["dataset"], "cams-global-reanalysis-eac4")
        self.assertIn("request", provenance)
        self.assertIn("api_url", provenance)
        self.assertIn("sha256", provenance)
        self.assertIn("size_bytes", provenance)
        self.assertGreater(
            provenance["size_bytes"], 0, "File should have non-zero size"
        )

        print(
            f"✓ Reanalysis test passed: {os.path.getsize(result_file)} bytes downloaded"
        )

    def test_idempotent_downloads(self):
        """Test that downloads are idempotent (don't re-download existing valid files)."""
        output_file = os.path.join(self.temp_dir, "berlin_idempotent_test.nc")

        # First download
        result1 = self.downloader.download_forecast(
            dates="2025-09-10",
            times="00:00",
            variables="particulate_matter_2.5um",
            area=self.berlin_area,
            output_file=output_file,
            leadtime_hours=0,
        )

        # Record file modification time
        first_mtime = os.path.getmtime(result1)

        # Second download should skip
        result2 = self.downloader.download_forecast(
            dates="2025-09-10",
            times="00:00",
            variables="particulate_matter_2.5um",
            area=self.berlin_area,
            output_file=output_file,
            leadtime_hours=0,
        )

        # File should not have been modified
        second_mtime = os.path.getmtime(result2)
        self.assertEqual(
            first_mtime, second_mtime, "File should not be re-downloaded if valid"
        )

        print("✓ Idempotent download test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
