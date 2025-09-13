#!/usr/bin/env python3
"""
CAMS ADS CLI - Command line interface for downloading CAMS atmospheric data.

Provides commands for downloading CAMS forecasts and reanalysis data with
full parameter control, dry-run capability, and file validation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import xarray as xr

from cams_ads_downloader import CAMSADSDownloader


def parse_area(area_str: str) -> List[float]:
    """Parse area string 'north,west,south,east' to list of floats."""
    try:
        coords = [float(x.strip()) for x in area_str.split(",")]
        if len(coords) != 4:
            raise ValueError("Area must have exactly 4 coordinates")
        return coords
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid area format: {e}")


def parse_list(value: str) -> List[str]:
    """Parse comma-separated string to list."""
    return [item.strip() for item in value.split(",")]


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated string to list of integers."""
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {e}")


def validate_netcdf(filepath: str) -> bool:
    """
    Validate NetCDF file by checking if it opens and contains expected structure.

    Args:
        filepath: Path to NetCDF file

    Returns:
        True if valid, False otherwise
    """
    try:
        with xr.open_dataset(filepath) as ds:
            print(f"✓ File opens successfully: {filepath}")
            print(f"  Variables: {list(ds.data_vars.keys())}")
            print(f"  Dimensions: {dict(ds.dims)}")

            # Check for time dimension
            if "time" in ds.dims:
                print(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")

            # Check provenance file
            provenance_file = filepath + ".provenance.json"
            if Path(provenance_file).exists():
                with open(provenance_file, "r") as f:
                    provenance = json.load(f)
                print(f"  Dataset: {provenance.get('dataset', 'unknown')}")
                print(f"  File size: {provenance.get('size_bytes', 0)} bytes")
                print(f"  SHA256: {provenance.get('sha256', 'unknown')[:16]}...")

            return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def cmd_forecast(args):
    """Handle forecast download command."""
    if hasattr(args, "dry_run") and args.dry_run:
        request = {
            "date": parse_list(args.dates),
            "time": parse_list(args.times),
            "variable": parse_list(args.variables),
            "area": args.area,
            "leadtime_hour": args.leadtime_hours,
            "format": args.format,
        }
        print("Dry run - would submit request:")
        print(json.dumps(request, indent=2))
        return

    try:
        downloader = CAMSADSDownloader(args.config)
        output_file = downloader.download_forecast(
            dates=parse_list(args.dates),
            times=parse_list(args.times),
            variables=parse_list(args.variables),
            area=args.area,
            output_file=args.output,
            leadtime_hours=args.leadtime_hours,
            format=args.format,
        )
        print(f"✓ Downloaded: {output_file}")

    except Exception as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)


def cmd_reanalysis(args):
    """Handle reanalysis download command."""
    if hasattr(args, "dry_run") and args.dry_run:
        request = {
            "date": parse_list(args.dates),
            "time": parse_list(args.times),
            "variable": parse_list(args.variables),
            "area": args.area,
            "format": args.format,
        }
        print("Dry run - would submit request:")
        print(json.dumps(request, indent=2))
        return

    try:
        downloader = CAMSADSDownloader(args.config)
        output_file = downloader.download_reanalysis(
            dates=parse_list(args.dates),
            times=parse_list(args.times),
            variables=parse_list(args.variables),
            area=args.area,
            output_file=args.output,
            format=args.format,
        )
        print(f"✓ Downloaded: {output_file}")

    except Exception as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)


def cmd_validate(args):
    """Handle file validation command."""
    success = validate_netcdf(args.file)
    sys.exit(0 if success else 1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CAMS ADS CLI - Download atmospheric composition data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download PM2.5 forecast for Berlin
  python cams_ads_cli.py forecast \\
    --dates 2025-09-10 \\
    --times 00:00 \\
    --variables particulate_matter_2.5um \\
    --area 52.52,13.40,52.50,13.42 \\
    --leadtime-hours 0 \\
    --output berlin_forecast.nc

  # Download PM2.5 reanalysis for Berlin
  python cams_ads_cli.py reanalysis \\
    --dates 2003-01-01 \\
    --times 00:00 \\
    --variables particulate_matter_2.5um \\
    --area 52.52,13.40,52.50,13.42 \\
    --output berlin_reanalysis.nc

  # Validate a downloaded file
  python cams_ads_cli.py validate berlin_forecast.nc
        """,
    )

    parser.add_argument(
        "--config", help="Path to .cdsapirc config file (default: ~/.cdsapirc)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Download CAMS forecasts")
    forecast_parser.add_argument(
        "--dry-run", action="store_true", help="Print request JSON without downloading"
    )
    forecast_parser.add_argument(
        "--dates", required=True, help="Comma-separated dates (YYYY-MM-DD)"
    )
    forecast_parser.add_argument(
        "--times", required=True, help="Comma-separated times (HH:MM)"
    )
    forecast_parser.add_argument(
        "--variables", required=True, help="Comma-separated variable names"
    )
    forecast_parser.add_argument(
        "--area",
        type=parse_area,
        required=True,
        help="Bounding box: north,west,south,east",
    )
    forecast_parser.add_argument(
        "--leadtime-hours",
        type=parse_int_list,
        default=[0],
        help="Comma-separated leadtime hours (default: 0)",
    )
    forecast_parser.add_argument(
        "--format", default="netcdf", help="Output format (default: netcdf)"
    )
    forecast_parser.add_argument("--output", required=True, help="Output file path")
    forecast_parser.set_defaults(func=cmd_forecast)

    # Reanalysis command
    reanalysis_parser = subparsers.add_parser(
        "reanalysis", help="Download CAMS reanalysis"
    )
    reanalysis_parser.add_argument(
        "--dry-run", action="store_true", help="Print request JSON without downloading"
    )
    reanalysis_parser.add_argument(
        "--dates", required=True, help="Comma-separated dates (YYYY-MM-DD)"
    )
    reanalysis_parser.add_argument(
        "--times", required=True, help="Comma-separated times (HH:MM)"
    )
    reanalysis_parser.add_argument(
        "--variables", required=True, help="Comma-separated variable names"
    )
    reanalysis_parser.add_argument(
        "--area",
        type=parse_area,
        required=True,
        help="Bounding box: north,west,south,east",
    )
    reanalysis_parser.add_argument(
        "--format", default="netcdf", help="Output format (default: netcdf)"
    )
    reanalysis_parser.add_argument("--output", required=True, help="Output file path")
    reanalysis_parser.set_defaults(func=cmd_reanalysis)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate NetCDF file")
    validate_parser.add_argument("file", help="NetCDF file to validate")
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
