"""
ECMWF CAMS ADS (Atmosphere Data Store) downloader with retries and provenance tracking.

This module provides a wrapper around the cdsapi.Client for downloading CAMS atmospheric
composition forecasts and reanalysis data from the ADS API with proper error handling,
retries, and metadata tracking.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import cdsapi
import xarray as xr


class CAMSADSDownloader:
    """Wrapper for cdsapi.Client with ADS endpoints and robust error handling."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize CAMS ADS client.

        Args:
            config_file: Path to .cdsapirc file. If None, uses default %USERPROFILE%\\.cdsapirc
        """
        if config_file is None:
            config_file = os.path.join(os.path.expanduser("~"), ".cdsapirc")

        # Read configuration
        self.config = self._read_config(config_file)
        self.client = None
        self._initialize_client()

    def _read_config(self, config_file: str) -> Dict[str, str]:
        """Read ADS API configuration from .cdsapirc file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Please create ~/.cdsapirc with:\n"
                f"url: https://ads.atmosphere.copernicus.eu/api\n"
                f"key: <YOUR_ADS_API_KEY>"
            )

        config = {}
        with open(config_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()

        if "url" not in config or "key" not in config:
            raise ValueError(
                f"Invalid configuration file: {config_file}\n"
                f"Must contain 'url' and 'key' entries"
            )

        return config

    def _initialize_client(self):
        """Initialize cdsapi client with ADS endpoint."""
        try:
            # Try primary ADS endpoint first
            self.client = cdsapi.Client(url=self.config["url"], key=self.config["key"])
            self.api_url = self.config["url"]
        except Exception as e:
            # Fallback to beta endpoint if specified in requirements
            if "ads.atmosphere.copernicus.eu" in self.config["url"]:
                beta_url = self.config["url"].replace(
                    "ads.atmosphere.copernicus.eu", "ads-beta.atmosphere.copernicus.eu"
                )
                try:
                    self.client = cdsapi.Client(url=beta_url, key=self.config["key"])
                    self.api_url = beta_url
                    print(f"Warning: Fell back to beta endpoint: {beta_url}")
                except Exception as beta_e:
                    raise RuntimeError(
                        f"Failed to connect to both ADS endpoints: {e}, {beta_e}"
                    )
            else:
                raise e

    def _retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0):
        """Execute function with exponential backoff on 429/5xx errors."""
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                error_msg = str(e).lower()
                if attempt == max_retries:
                    raise e

                # Retry on rate limiting (429) or server errors (5xx)
                if "429" in error_msg or "5" in error_msg[:3] or "timeout" in error_msg:
                    delay = base_delay * (2**attempt)
                    print(
                        f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    raise e

    def _file_exists_and_valid(self, filepath: str) -> bool:
        """Check if NetCDF file exists and is valid."""
        if not os.path.exists(filepath):
            return False

        try:
            # Quick validation - try to open with xarray
            with xr.open_dataset(filepath) as ds:
                return len(ds.data_vars) > 0
        except Exception:
            return False

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _write_provenance(
        self,
        filepath: str,
        dataset: str,
        request: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
    ):
        """Write provenance metadata JSON file."""
        file_size = os.path.getsize(filepath)
        file_hash = self._compute_file_hash(filepath)

        provenance = {
            "dataset": dataset,
            "request": request,
            "api_url": self.api_url,
            "sha256": file_hash,
            "size_bytes": file_size,
            "download_start": start_time.isoformat(),
            "download_end": end_time.isoformat(),
            "cdsapi_version": cdsapi.__version__,
        }

        provenance_file = filepath + ".provenance.json"
        with open(provenance_file, "w") as f:
            json.dump(provenance, f, indent=2)

    def download_forecast(
        self,
        dates: Union[str, List[str]],
        times: Union[str, List[str]],
        variables: Union[str, List[str]],
        area: List[float],
        output_file: str,
        leadtime_hours: Union[int, List[int]] = 0,
        format: str = "netcdf",
    ) -> str:
        """
        Download CAMS global atmospheric composition forecasts.

        Args:
            dates: Date(s) in YYYY-MM-DD format
            times: Time(s) in HH:MM format
            variables: Variable name(s) (e.g., 'particulate_matter_2.5um')
            area: [north, west, south, east] bounding box
            output_file: Path for output NetCDF file
            leadtime_hours: Forecast leadtime(s) in hours
            format: Output format (netcdf)

        Returns:
            Path to downloaded file
        """
        # Skip if valid file already exists (idempotent)
        if self._file_exists_and_valid(output_file):
            print(f"File already exists and is valid: {output_file}")
            return output_file

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Normalize inputs to lists
        if isinstance(dates, str):
            dates = [dates]
        if isinstance(times, str):
            times = [times]
        if isinstance(variables, str):
            variables = [variables]
        if isinstance(leadtime_hours, int):
            leadtime_hours = [leadtime_hours]

        request = {
            "date": dates,
            "time": times,
            "variable": variables,
            "area": area,
            "leadtime_hour": leadtime_hours,
            "format": format,
        }

        dataset = "cams-global-atmospheric-composition-forecasts"

        def download_func():
            start_time = datetime.now()
            self.client.retrieve(dataset, request, output_file)
            end_time = datetime.now()

            # Write provenance metadata
            self._write_provenance(output_file, dataset, request, start_time, end_time)
            return output_file

        return self._retry_with_backoff(download_func)

    def download_reanalysis(
        self,
        dates: Union[str, List[str]],
        times: Union[str, List[str]],
        variables: Union[str, List[str]],
        area: List[float],
        output_file: str,
        format: str = "netcdf",
    ) -> str:
        """
        Download CAMS global reanalysis (EAC4) data.

        Args:
            dates: Date(s) in YYYY-MM-DD format
            times: Time(s) in HH:MM format
            variables: Variable name(s) (e.g., 'particulate_matter_2.5um')
            area: [north, west, south, east] bounding box
            output_file: Path for output NetCDF file
            format: Output format (netcdf)

        Returns:
            Path to downloaded file
        """
        # Skip if valid file already exists (idempotent)
        if self._file_exists_and_valid(output_file):
            print(f"File already exists and is valid: {output_file}")
            return output_file

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Normalize inputs to lists
        if isinstance(dates, str):
            dates = [dates]
        if isinstance(times, str):
            times = [times]
        if isinstance(variables, str):
            variables = [variables]

        request = {
            "date": dates,
            "time": times,
            "variable": variables,
            "area": area,
            "format": format,
        }

        dataset = "cams-global-reanalysis-eac4"

        def download_func():
            start_time = datetime.now()
            self.client.retrieve(dataset, request, output_file)
            end_time = datetime.now()

            # Write provenance metadata
            self._write_provenance(output_file, dataset, request, start_time, end_time)
            return output_file

        return self._retry_with_backoff(download_func)
