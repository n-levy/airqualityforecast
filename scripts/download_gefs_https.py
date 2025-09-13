#!/usr/bin/env python3
"""
NOAA GEFS-Aerosols HTTPS Downloader

Downloads GRIB2 files from NOAA GEFS-Aerosols via HTTPS to S3 website endpoint.
Handles 404s gracefully, resumes partial downloads, and maintains manifest.
Cross-platform Python implementation using requests library.
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("ERROR: Missing requests library")
    print("Install with: pip install requests")
    sys.exit(1)


class GefsDownloader:
    """HTTPS downloader for GEFS-Aerosols GRIB2 files."""

    BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"

    def __init__(self, data_root: Path, verbose: bool = False):
        self.data_root = Path(data_root)
        self.raw_root = self.data_root / "raw" / "gefs_chem"
        self.manifest_dir = self.data_root / "raw" / "gefs_chem" / "_manifests"
        self.manifest_file = self.manifest_dir / "download_manifest.csv"
        self.verbose = verbose

        # Create directories
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        # Setup requests session with retry strategy
        self.session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set reasonable timeout and headers
        self.session.headers.update({"User-Agent": "NOAA-GEFS-Downloader/1.0"})

        # Initialize manifest if not exists
        self._init_manifest()

    def _init_manifest(self):
        """Initialize manifest file with headers."""
        if not self.manifest_file.exists():
            with open(self.manifest_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_date",
                        "run_hour",
                        "f_hour",
                        "url",
                        "status",
                        "http_code",
                        "bytes",
                        "path",
                        "ts_utc",
                    ]
                )

    def _log_to_manifest(
        self,
        run_date: str,
        run_hour: str,
        f_hour: str,
        url: str,
        status: str,
        http_code: int,
        bytes_downloaded: int,
        local_path: str,
    ):
        """Log download attempt to manifest."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        with open(self.manifest_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    run_date,
                    run_hour,
                    f_hour,
                    url,
                    status,
                    http_code,
                    bytes_downloaded,
                    local_path,
                    timestamp,
                ]
            )

    def download_file(
        self, url: str, local_path: Path, run_date: str, run_hour: str, f_hour: str
    ) -> bool:
        """
        Download a single file with resume capability.

        Args:
            url: URL to download
            local_path: Local file path
            run_date: Run date for manifest
            run_hour: Run hour for manifest
            f_hour: Forecast hour for manifest

        Returns:
            True if successful (including if file already exists), False on error
        """
        status = "unknown"
        http_code = 0
        bytes_downloaded = 0

        try:
            # Check if file already exists and is non-empty
            if local_path.exists() and local_path.stat().st_size > 0:
                status = "exists"
                bytes_downloaded = local_path.stat().st_size
                if self.verbose:
                    print(f"EXISTS: {local_path.name} ({bytes_downloaded:,} bytes)")

                self._log_to_manifest(
                    run_date,
                    run_hour,
                    f_hour,
                    url,
                    status,
                    http_code,
                    bytes_downloaded,
                    str(local_path),
                )
                return True

            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine resume position
            resume_pos = 0
            if local_path.exists():
                resume_pos = local_path.stat().st_size

            # Setup headers for resume
            headers = {}
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
                if self.verbose:
                    print(f"RESUMING: {url} from byte {resume_pos:,}")
            else:
                if self.verbose:
                    print(f"DOWNLOADING: {url}")

            # Download with streaming
            response = self.session.get(url, headers=headers, stream=True, timeout=30)
            http_code = response.status_code

            if response.status_code == 404:
                status = "not_found"
                if self.verbose:
                    print(f"NOT FOUND (404): {url}")
                self._log_to_manifest(
                    run_date,
                    run_hour,
                    f_hour,
                    url,
                    status,
                    http_code,
                    bytes_downloaded,
                    str(local_path),
                )
                return True  # 404 is expected for some dates

            response.raise_for_status()

            # Handle partial content response
            if response.status_code == 206:  # Partial content
                mode = "ab"  # Append binary
            else:
                mode = "wb"  # Write binary
                resume_pos = 0  # Start from beginning if not partial

            # Write to temporary file first for atomic operation
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, dir=local_path.parent, prefix=local_path.name + ".tmp"
                ) as temp_file:
                    temp_path = Path(temp_file.name)

                    # Copy existing content if resuming
                    if resume_pos > 0 and local_path.exists():
                        with open(local_path, "rb") as existing:
                            shutil.copyfileobj(existing, temp_file)

                    # Download new content
                    bytes_this_session = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            bytes_this_session += len(chunk)

                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                # Atomic move to final location
                shutil.move(str(temp_path), str(local_path))

                bytes_downloaded = resume_pos + bytes_this_session
                status = "success"

                if self.verbose:
                    print(
                        f"SUCCESS: {local_path.name} ({bytes_downloaded:,} bytes, HTTP {http_code})"
                    )

            except Exception as e:
                # Clean up temp file on error
                if temp_file and Path(temp_file.name).exists():
                    Path(temp_file.name).unlink()
                raise e

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                status = "not_found"
                http_code = 404
                if self.verbose:
                    print(f"NOT FOUND (404): {url}")
            else:
                status = "http_error"
                http_code = e.response.status_code
                print(f"HTTP ERROR: {url} - {e}")

        except requests.exceptions.RequestException as e:
            status = "network_error"
            print(f"NETWORK ERROR: {url} - {e}")

        except Exception as e:
            status = "error"
            print(f"ERROR: {url} - {e}")

        self._log_to_manifest(
            run_date,
            run_hour,
            f_hour,
            url,
            status,
            http_code,
            bytes_downloaded,
            str(local_path),
        )

        # Return True for success, exists, or 404 (expected missing files)
        return status in ["success", "exists", "not_found"]

    def build_url(self, date_str: str, cycle: str, fhour: str) -> str:
        """Build GEFS URL for given parameters."""
        return f"{self.BASE_URL}/gefs.{date_str}/{cycle}/chem/pgrb2ap25/gefs.chem.t{cycle}z.a2d_0p25.f{fhour}.grib2"

    def build_local_path(self, date_str: str, cycle: str, fhour: str) -> Path:
        """Build local file path for given parameters."""
        return (
            self.raw_root
            / date_str
            / cycle
            / f"gefs.chem.t{cycle}z.a2d_0p25.f{fhour}.grib2"
        )


def parse_forecast_hours(fhours_str: str) -> List[str]:
    """Parse forecast hours string."""
    if ":" in fhours_str:
        parts = fhours_str.split(":")
        if len(parts) != 3:
            raise ValueError("Forecast hours format should be start:step:end")
        f_start, f_step, f_end = map(int, parts)
        return [f"{f:03d}" for f in range(f_start, f_end + 1, f_step)]
    else:
        return [f"{int(f.strip()):03d}" for f in fhours_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="NOAA GEFS-Aerosols HTTPS Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--start-date", required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--cycles",
        default="00,12",
        help='Comma-separated forecast cycles (e.g., "00,12")',
    )
    parser.add_argument(
        "--fhours",
        default="0:6:120",
        help="Forecast hours as start:step:end or comma-separated list",
    )
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=== NOAA GEFS-Aerosols HTTPS Downloader ===")
    print(f"Base URL: {GefsDownloader.BASE_URL}")
    print(f"Data root: {args.data_root}")

    # Parse inputs
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid date format - {e}")
        sys.exit(1)

    cycles = [c.strip() for c in args.cycles.split(",")]

    try:
        fhours = parse_forecast_hours(args.fhours)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Cycles: {', '.join(cycles)}")
    print(f"Forecast hours: {', '.join(fhours)}")

    # Initialize downloader
    downloader = GefsDownloader(args.data_root, verbose=args.verbose)

    # Download files
    total_files = 0
    success_files = 0
    not_found_files = 0
    error_files = 0

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        iso_date_str = current_date.strftime("%Y-%m-%d")

        if args.verbose:
            print(f"\nProcessing date: {iso_date_str}")

        for cycle in cycles:
            for fhour in fhours:
                total_files += 1

                url = downloader.build_url(date_str, cycle, fhour)
                local_path = downloader.build_local_path(date_str, cycle, fhour)

                success = downloader.download_file(
                    url, local_path, iso_date_str, cycle, fhour
                )

                if success:
                    # Read last manifest entry to determine success type
                    try:
                        with open(downloader.manifest_file, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # Has header + at least one entry
                                last_entry = lines[-1].strip()
                                if ",not_found," in last_entry:
                                    not_found_files += 1
                                else:
                                    success_files += 1
                            else:
                                success_files += 1
                    except Exception:
                        success_files += 1
                else:
                    error_files += 1

        current_date += timedelta(days=1)

    # Summary
    print(f"\n=== DOWNLOAD SUMMARY ===")
    print(f"Total files requested: {total_files:,}")
    print(f"Successfully downloaded/existed: {success_files:,}")
    print(f"Not found (404): {not_found_files:,}")
    print(f"Errors: {error_files:,}")
    print(f"Manifest file: {downloader.manifest_file}")

    # Exit code: 0 if all files either succeeded or 404'd, non-zero if there were other errors
    if error_files == 0:
        print("OK Download completed successfully (including expected 404s)")
        sys.exit(0)
    else:
        print(f"WARNING Download completed with {error_files:,} errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
