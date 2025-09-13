#!/usr/bin/env python3
"""
NOAA GEFS-Aerosols Data Collection Orchestrator

Downloads, extracts, and verifies two years of NOAA GEFS-Aerosols pollutant data
(PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃) for a configurable region via HTTPS only.

Creates partitioned Parquet files ready for ML modeling.
Cross-platform Python implementation.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def check_dependencies():
    """Check if required Python packages are available."""
    required_packages = ["requests", "xarray", "cfgrib", "pandas", "pyarrow"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("ERROR: Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements.txt")
        return False

    return True


def check_data_root(data_root):
    """Check and create data root directory structure."""
    if not data_root:
        # Try environment variable first
        data_root = os.environ.get("DATA_ROOT")
        if not data_root:
            # Use platform-neutral default location
            data_root = Path.home() / "gefs_data"

    data_root = Path(data_root)

    # Create directory structure
    raw_root = data_root / "raw" / "gefs_chem"
    curated_root = data_root / "curated" / "gefs_chem" / "parquet"
    manifest_dir = data_root / "raw" / "gefs_chem" / "_manifests"

    raw_root.mkdir(parents=True, exist_ok=True)
    curated_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    return data_root


def parse_forecast_hours(fhours_str):
    """Parse forecast hours string."""
    if ":" in fhours_str:
        parts = fhours_str.split(":")
        if len(parts) != 3:
            raise ValueError("Forecast hours format should be start:step:end")
        f_start, f_step, f_end = map(int, parts)
        return [f"{f:03d}" for f in range(f_start, f_end + 1, f_step)]
    else:
        return [f"{int(f.strip()):03d}" for f in fhours_str.split(",")]


def calculate_planned_files(start_date, end_date, cycles, fhours):
    """Calculate number of planned files."""
    date_count = (end_date - start_date).days + 1
    return date_count * len(cycles) * len(fhours)


def main():
    parser = argparse.ArgumentParser(
        description="NOAA GEFS-Aerosols Data Collection Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (1 day, one cycle, one forecast hour)
  python orchestrate_gefs_https.py --start-date 2024-01-12 --end-date 2024-01-12 --cycles 00 --fhours 24:24:24

  # Full two years
  python orchestrate_gefs_https.py --start-date 2022-01-01 --end-date 2024-01-01 --force
        """,
    )

    # Date parameters
    default_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        "--start-date",
        default=default_start,
        help="Start date in YYYY-MM-DD format (default: today minus 730 days)",
    )
    parser.add_argument(
        "--end-date",
        default=default_end,
        help="End date in YYYY-MM-DD format (default: today)",
    )

    # Forecast parameters
    parser.add_argument(
        "--cycles",
        default="00,12",
        help='Comma-separated forecast cycles (default: "00,12")',
    )
    parser.add_argument(
        "--fhours",
        default="0:6:120",
        help='Forecast hours as start:step:end (default: "0:6:120")',
    )

    # Geographic parameters
    parser.add_argument(
        "--bbox",
        default="5,16,47,56",
        help='Bounding box as lon_min,lon_max,lat_min,lat_max (default: "5,16,47,56" for Germany)',
    )
    parser.add_argument(
        "--pollutants",
        default="PM25,PM10,NO2,SO2,CO,O3",
        help='Comma-separated pollutants (default: "PM25,PM10,NO2,SO2,CO,O3")',
    )

    # System parameters
    parser.add_argument(
        "--data-root",
        help="Data root directory (default: from DATA_ROOT env or ~/gefs_data)",
    )
    parser.add_argument(
        "--max-planned-files",
        type=int,
        default=50000,
        help="Safety limit for planned file count (default: 50000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Python extraction parallelism (default: 4)",
    )

    # Control parameters
    parser.add_argument(
        "--force", action="store_true", help="Allow exceeding max-planned-files limit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=== NOAA GEFS-Aerosols Data Collection Orchestrator ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

    # Step 1: Dependency Check
    print("\n[1/5] Dependency Check...")
    if not check_dependencies():
        sys.exit(1)
    print("OK All required packages available")

    # Step 2: Environment Check
    print("\n[2/5] Environment Check...")
    data_root = check_data_root(args.data_root)
    raw_root = data_root / "raw" / "gefs_chem"
    curated_root = data_root / "curated" / "gefs_chem" / "parquet"

    print(f"OK Data root: {data_root}")
    print(f"OK Raw GRIB2: {raw_root}")
    print(f"OK Curated Parquet: {curated_root}")

    # Step 3: Parse and validate parameters
    print("\n[3/5] Planning Download...")

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid date format - {e}")
        sys.exit(1)

    # Parse cycles
    cycles = [c.strip() for c in args.cycles.split(",")]
    for cycle in cycles:
        if cycle not in ["00", "06", "12", "18"]:
            print(f"ERROR: Invalid cycle: {cycle}. Must be one of: 00, 06, 12, 18")
            sys.exit(1)

    # Parse forecast hours
    try:
        fhours = parse_forecast_hours(args.fhours)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Calculate planned files
    planned_files = calculate_planned_files(start_date, end_date, cycles, fhours)

    print(
        f"Date range: {args.start_date} to {args.end_date} ({(end_date - start_date).days + 1} days)"
    )
    print(f"Cycles: {', '.join(cycles)} ({len(cycles)} cycles)")
    print(f"Forecast hours: {', '.join(fhours)} ({len(fhours)} hours)")
    print(f"Planned files: {planned_files:,}")

    if planned_files > args.max_planned_files and not args.force:
        print(
            f"ERROR: Planned files ({planned_files:,}) exceeds safety limit ({args.max_planned_files:,})"
        )
        print("Use --force to override")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {planned_files:,} files")
        print(f"Raw GRIB2 would be stored under: {raw_root}")
        print(f"Curated Parquet would be stored under: {curated_root}")
        sys.exit(0)

    # Step 4: Download
    print("\n[4/5] Downloading GRIB2 Files...")

    script_dir = Path(__file__).parent
    download_script = script_dir / "download_gefs_https.py"

    download_cmd = [
        sys.executable,
        str(download_script),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--cycles",
        args.cycles,
        "--fhours",
        args.fhours,
        "--data-root",
        str(data_root),
    ]

    if args.verbose:
        download_cmd.append("--verbose")

    print(f"Executing: {' '.join(download_cmd)}")
    result = subprocess.run(download_cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Download failed with exit code: {result.returncode}")
        sys.exit(result.returncode)

    print("OK Download completed successfully")

    # Step 5: Extract
    print("\n[5/5] Extracting to Parquet...")

    extract_script = script_dir / "extract_gefs_pollutants.py"

    extract_cmd = [
        sys.executable,
        str(extract_script),
        "--raw-root",
        str(raw_root),
        "--out-root",
        str(curated_root),
        "--bbox",
        args.bbox,
        "--pollutants",
        args.pollutants,
        "--runs-from",
        args.start_date,
        "--runs-to",
        args.end_date,
        "--cycles",
        args.cycles,
        "--fhours",
        args.fhours,
        "--workers",
        str(args.workers),
    ]

    if args.verbose:
        extract_cmd.append("--verbose")

    print(f"Executing Python extraction...")
    result = subprocess.run(extract_cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Extraction failed with exit code: {result.returncode}")
        sys.exit(result.returncode)

    print("OK Extraction completed successfully")

    # Step 6: Summary
    print("\n=== FINAL SUMMARY ===")

    # Count files
    try:
        grib_files = list(raw_root.rglob("*.grib2"))
        parquet_files = list(curated_root.rglob("*.parquet"))

        print(f"GRIB2 files found: {len(grib_files):,}")
        print(f"Parquet files created: {len(parquet_files):,}")

        # Check manifests
        download_manifest = (
            data_root / "raw" / "gefs_chem" / "_manifests" / "download_manifest.csv"
        )
        extract_manifest = curated_root / "extract_manifest.csv"

        download_entries = 0
        extract_entries = 0

        if download_manifest.exists():
            with open(download_manifest, "r") as f:
                download_entries = sum(1 for _ in f) - 1  # Subtract header

        if extract_manifest.exists():
            with open(extract_manifest, "r") as f:
                extract_entries = sum(1 for _ in f) - 1  # Subtract header

        print(f"Download manifest entries: {download_entries:,}")
        print(f"Extract manifest entries: {extract_entries:,}")

    except Exception as e:
        print(f"Warning: Could not count files - {e}")

    print(f"\nData locations:")
    print(f"  Raw GRIB2: {raw_root}")
    print(f"  Curated Parquet: {curated_root}")
    print(f"  Download manifest: {download_manifest}")
    print(f"  Extract manifest: {extract_manifest}")

    print("\n=== ORCHESTRATION COMPLETE ===")
    print("OK All steps completed successfully")


if __name__ == "__main__":
    main()
