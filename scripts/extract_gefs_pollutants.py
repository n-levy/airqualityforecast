#!/usr/bin/env python3
"""
NOAA GEFS-Aerosols Pollutant Extraction Script

Extracts pollutants (PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃) from GRIB2 files
and converts to partitioned Parquet format for ML modeling.

Handles robust variable detection with fuzzy mapping and surface level preference.
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pyarrow import Table
    from pyarrow import parquet as pq
except ImportError as e:
    print(f"ERROR: Missing required package - {e}")
    print("Install with: pip install xarray cfgrib eccodes pandas pyarrow")
    sys.exit(1)

# Suppress warnings from cfgrib/eccodes
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PollutantExtractor:
    """Extracts and processes pollutant data from GEFS GRIB2 files."""

    # Pollutant mapping with fuzzy matching rules
    POLLUTANT_MAP = {
        "PM25": {
            "patterns": ["pm2", "5", "pmtf"],  # pm2.5, pm2p5, PMTF
            "exclude": [],
            "priority_patterns": ["pm2p5", "pmtf"],
        },
        "PM10": {
            "patterns": ["pm10", "pmtc"],
            "exclude": [],
            "priority_patterns": ["pm10", "pmtc"],
        },
        "NO2": {"patterns": ["no2"], "exclude": [], "priority_patterns": ["no2"]},
        "SO2": {"patterns": ["so2"], "exclude": [], "priority_patterns": ["so2"]},
        "CO": {
            "patterns": ["co"],
            "exclude": ["co2", "cos"],
            "priority_patterns": ["co"],
        },
        "O3": {"patterns": ["o3"], "exclude": [], "priority_patterns": ["o3"]},
    }

    def __init__(self, bbox: Tuple[float, float, float, float]):
        """
        Initialize extractor.

        Args:
            bbox: Bounding box as (lon_min, lon_max, lat_min, lat_max)
        """
        self.bbox = bbox
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = bbox

    def find_pollutant_variables(self, ds: xr.Dataset, pollutant: str) -> List[str]:
        """
        Find variables that match a pollutant using fuzzy matching.

        Args:
            ds: xarray Dataset
            pollutant: Pollutant name (PM25, PM10, etc.)

        Returns:
            List of variable names matching the pollutant
        """
        if pollutant not in self.POLLUTANT_MAP:
            return []

        mapping = self.POLLUTANT_MAP[pollutant]
        patterns = mapping["patterns"]
        exclude = mapping["exclude"]
        priority_patterns = mapping["priority_patterns"]

        matches = []
        priority_matches = []

        # Check all variables and their attributes
        for var_name in ds.data_vars:
            var = ds[var_name]

            # Get shortName from attributes if available
            short_name = getattr(var, "GRIB_shortName", var_name).lower()
            var_name_lower = var_name.lower()

            # Check for exclusions first
            excluded = any(
                excl in short_name or excl in var_name_lower for excl in exclude
            )
            if excluded:
                continue

            # Check for pattern matches
            pattern_match = any(
                pattern in short_name or pattern in var_name_lower
                for pattern in patterns
            )

            if pattern_match:
                matches.append(var_name)

                # Check for priority patterns
                priority_match = any(
                    pattern in short_name or pattern in var_name_lower
                    for pattern in priority_patterns
                )
                if priority_match:
                    priority_matches.append(var_name)

        # Return priority matches if available, otherwise all matches
        return priority_matches if priority_matches else matches

    def select_surface_variable(
        self, ds: xr.Dataset, var_names: List[str]
    ) -> Optional[str]:
        """
        Select surface-level variable from candidates.

        Args:
            ds: xarray Dataset
            var_names: List of candidate variable names

        Returns:
            Best variable name or None
        """
        if not var_names:
            return None

        surface_vars = []

        for var_name in var_names:
            var = ds[var_name]

            # Check if variable has vertical levels
            has_levels = any(
                dim in ["isobaricInhPa", "hybrid", "heightAboveGround"]
                for dim in var.dims
            )

            if not has_levels:
                # No vertical dimension - likely surface
                surface_vars.append(var_name)
            else:
                # Check for surface level indicators
                for dim in var.dims:
                    if dim in var.coords:
                        coord_vals = var.coords[dim].values
                        # Look for surface pressure (1000+ hPa) or low height (< 10m)
                        if dim == "isobaricInhPa" and np.any(coord_vals >= 1000):
                            surface_vars.append(var_name)
                        elif dim == "heightAboveGround" and np.any(coord_vals <= 10):
                            surface_vars.append(var_name)

        # Return first surface variable, or first variable if none identified as surface
        return surface_vars[0] if surface_vars else var_names[0]

    def extract_pollutant_data(
        self, ds: xr.Dataset, pollutant: str
    ) -> Optional[xr.DataArray]:
        """
        Extract pollutant data from dataset.

        Args:
            ds: xarray Dataset
            pollutant: Pollutant name

        Returns:
            xarray DataArray or None if not found
        """
        var_names = self.find_pollutant_variables(ds, pollutant)
        if not var_names:
            return None

        var_name = self.select_surface_variable(ds, var_names)
        if not var_name:
            return None

        var_data = ds[var_name]

        # Select surface level if multi-level
        for dim in var_data.dims:
            if dim in ["isobaricInhPa", "hybrid", "heightAboveGround"]:
                if dim in var_data.coords:
                    coord_vals = var_data.coords[dim].values
                    if dim == "isobaricInhPa":
                        # Select highest pressure (surface)
                        surface_idx = np.argmax(coord_vals)
                    else:
                        # Select lowest height
                        surface_idx = np.argmin(coord_vals)
                    var_data = var_data.isel({dim: surface_idx})

        return var_data

    def subset_to_bbox(self, data: xr.DataArray) -> xr.DataArray:
        """
        Subset data to bounding box.

        Args:
            data: Input DataArray

        Returns:
            Subsetted DataArray
        """
        # Handle different coordinate names
        lat_coord = None
        lon_coord = None

        for coord in data.coords:
            coord_vals = data.coords[coord].values
            if len(coord_vals.shape) == 1:  # 1D coordinate
                if "lat" in coord.lower():
                    lat_coord = coord
                elif "lon" in coord.lower():
                    lon_coord = coord

        if lat_coord is None or lon_coord is None:
            print(
                f"WARNING: Could not identify lat/lon coordinates in {list(data.coords.keys())}"
            )
            return data

        # Subset to bounding box
        lat_mask = (data.coords[lat_coord] >= self.lat_min) & (
            data.coords[lat_coord] <= self.lat_max
        )
        lon_mask = (data.coords[lon_coord] >= self.lon_min) & (
            data.coords[lon_coord] <= self.lon_max
        )

        data_subset = data.where(lat_mask & lon_mask, drop=True)

        # Normalize coordinate names
        data_subset = data_subset.rename({lat_coord: "lat", lon_coord: "lon"})

        # Ensure proper lat orientation (ascending)
        if "lat" in data_subset.dims:
            if data_subset.lat[0] > data_subset.lat[-1]:
                data_subset = data_subset.isel(lat=slice(None, None, -1))

        return data_subset

    def process_file(
        self,
        file_path: Path,
        run_date: str,
        run_hour: str,
        f_hour: str,
        pollutants: List[str],
    ) -> Dict:
        """
        Process a single GRIB2 file.

        Args:
            file_path: Path to GRIB2 file
            run_date: Run date (YYYY-MM-DD)
            run_hour: Run hour (HH)
            f_hour: Forecast hour (FFF)
            pollutants: List of pollutants to extract

        Returns:
            Processing result dictionary
        """
        result = {
            "path_in": str(file_path),
            "rows_out": 0,
            "pollutants_found": "",
            "ts_utc": datetime.utcnow().isoformat() + "Z",
            "status": "unknown",
            "error": "",
        }

        try:
            if not file_path.exists():
                result["status"] = "file_not_found"
                return result

            # Open GRIB2 file with cfgrib
            with xr.open_dataset(file_path, engine="cfgrib") as ds:
                extracted_data = []
                found_pollutants = []

                for pollutant in pollutants:
                    pollutant_data = self.extract_pollutant_data(ds, pollutant)

                    if pollutant_data is not None:
                        # Subset to bounding box
                        subset_data = self.subset_to_bbox(pollutant_data)

                        if subset_data.size > 0:
                            # Convert to DataFrame
                            df = subset_data.to_dataframe().reset_index()

                            # Ensure we have the right columns
                            value_col = [
                                col
                                for col in df.columns
                                if col not in ["lat", "lon", "time"]
                            ]
                            if value_col:
                                df = df[["lat", "lon"] + value_col[:1]]
                                df.columns = ["lat", "lon", "value"]
                                df["pollutant"] = pollutant
                                df["run_date"] = run_date
                                df["run_hour"] = run_hour
                                df["f_hour"] = f_hour

                                extracted_data.append(df)
                                found_pollutants.append(pollutant)

                if extracted_data:
                    # Combine all pollutants
                    combined_df = pd.concat(extracted_data, ignore_index=True)

                    # Reorder columns
                    column_order = [
                        "run_date",
                        "run_hour",
                        "f_hour",
                        "lat",
                        "lon",
                        "pollutant",
                        "value",
                    ]
                    combined_df = combined_df[column_order]

                    result["rows_out"] = len(combined_df)
                    result["pollutants_found"] = ",".join(found_pollutants)
                    result["status"] = "success"
                    result["data"] = combined_df
                else:
                    result["status"] = "no_pollutants_found"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result


def write_parquet_partitioned(
    df: pd.DataFrame, out_root: Path, run_date: str, run_hour: str, f_hour: str
) -> Path:
    """
    Write DataFrame to partitioned Parquet.

    Args:
        df: DataFrame to write
        out_root: Output root directory
        run_date: Run date for partitioning
        run_hour: Run hour for partitioning
        f_hour: Forecast hour for partitioning

    Returns:
        Path to written parquet file
    """
    # Create partition directory
    partition_dir = (
        out_root / f"run_date={run_date}" / f"run_hour={run_hour}" / f"f_hour={f_hour}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary file for atomic write
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", delete=False, dir=partition_dir
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        # Write to temporary file
        table = Table.from_pandas(df)
        pq.write_table(table, temp_path)

        # Atomic move to final location
        final_path = partition_dir / f"part-{run_hour}{f_hour}.parquet"
        shutil.move(str(temp_path), str(final_path))

        return final_path

    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def worker_process_file(args):
    """Worker function for parallel processing."""
    extractor, file_path, run_date, run_hour, f_hour, pollutants, out_root = args

    # Process file
    result = extractor.process_file(file_path, run_date, run_hour, f_hour, pollutants)

    # Write to Parquet if successful
    if result["status"] == "success" and "data" in result:
        try:
            parquet_path = write_parquet_partitioned(
                result["data"], out_root, run_date, run_hour, f_hour
            )
            result["parquet_path"] = str(parquet_path)
            print(
                f"PROCESSED: {file_path.name} -> {result['rows_out']} rows "
                f"({result['pollutants_found']})"
            )
        except Exception as e:
            result["status"] = "parquet_error"
            result["error"] = str(e)
            print(f"PARQUET ERROR: {file_path.name} - {e}")
    else:
        if result["status"] == "no_pollutants_found":
            print(f"NO POLLUTANTS: {file_path.name}")
        else:
            print(f"ERROR: {file_path.name} - {result.get('error', result['status'])}")

    # Remove data from result to save memory
    if "data" in result:
        del result["data"]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract GEFS pollutants to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-root", required=True, help="Raw GRIB2 root directory")
    parser.add_argument(
        "--out-root", required=True, help="Output Parquet root directory"
    )
    parser.add_argument(
        "--bbox",
        default="5,16,47,56",
        help="Bounding box: lon_min,lon_max,lat_min,lat_max",
    )
    parser.add_argument(
        "--pollutants",
        default="PM25,PM10,NO2,SO2,CO,O3",
        help="Comma-separated pollutants",
    )
    parser.add_argument("--runs-from", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--runs-to", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cycles", default="00,12", help="Comma-separated cycles")
    parser.add_argument(
        "--fhours",
        default="0:6:120",
        help="Forecast hours (start:step:end or comma-separated)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Parse inputs
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    # Parse bounding box
    try:
        bbox_parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(bbox_parts) != 4:
            raise ValueError("Bounding box must have 4 values")
        bbox = tuple(bbox_parts)
    except ValueError as e:
        print(f"ERROR: Invalid bounding box format - {e}")
        sys.exit(1)

    # Parse pollutants
    pollutants = [p.strip().upper() for p in args.pollutants.split(",")]

    # Parse dates
    try:
        start_date = datetime.strptime(args.runs_from, "%Y-%m-%d")
        end_date = datetime.strptime(args.runs_to, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid date format - {e}")
        sys.exit(1)

    # Parse cycles
    cycles = [c.strip() for c in args.cycles.split(",")]

    # Parse forecast hours
    if ":" in args.fhours:
        parts = args.fhours.split(":")
        if len(parts) != 3:
            print("ERROR: Forecast hours format should be start:step:end")
            sys.exit(1)
        f_start, f_step, f_end = map(int, parts)
        fhours = [f"{f:03d}" for f in range(f_start, f_end + 1, f_step)]
    else:
        fhours = [f"{int(f.strip()):03d}" for f in args.fhours.split(",")]

    print("=== GEFS Pollutant Extraction ===")
    print(f"Platform: {sys.platform}")
    print(f"Raw root: {raw_root}")
    print(f"Output root: {out_root}")
    print(f"Bounding box: {bbox}")
    print(f"Pollutants: {pollutants}")
    print(f"Date range: {args.runs_from} to {args.runs_to}")
    print(f"Cycles: {cycles}")
    print(f"Forecast hours: {fhours}")
    print(f"Workers: {args.workers}")

    if args.verbose:
        print("Verbose mode enabled")

    # Create output directory
    out_root.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = PollutantExtractor(bbox)

    # Find all GRIB2 files to process
    files_to_process = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        iso_date_str = current_date.strftime("%Y-%m-%d")

        for cycle in cycles:
            for fhour in fhours:
                file_path = (
                    raw_root
                    / date_str
                    / cycle
                    / f"gefs.chem.t{cycle}z.a2d_0p25.f{fhour}.grib2"
                )

                if file_path.exists():
                    files_to_process.append((file_path, iso_date_str, cycle, fhour))

        current_date += timedelta(days=1)

    print(f"\nFound {len(files_to_process)} GRIB2 files to process")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for file_path, run_date, run_hour, f_hour in files_to_process[:10]:
            print(f"  {file_path}")
        if len(files_to_process) > 10:
            print(f"  ... and {len(files_to_process) - 10} more files")
        sys.exit(0)

    if not files_to_process:
        print("No GRIB2 files found to process")
        sys.exit(0)

    # Prepare manifest file
    manifest_file = out_root / "extract_manifest.csv"

    # Initialize manifest if it doesn't exist
    if not manifest_file.exists():
        with open(manifest_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["path_in", "rows_out", "pollutants_found", "ts_utc", "status", "error"]
            )

    # Process files in parallel
    print(f"\nProcessing with {args.workers} workers...")

    total_rows = 0
    processed_files = 0
    error_files = 0

    # Prepare worker arguments
    worker_args = []
    for file_path, run_date, run_hour, f_hour in files_to_process:
        worker_args.append(
            (extractor, file_path, run_date, run_hour, f_hour, pollutants, out_root)
        )

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        future_to_args = {
            executor.submit(worker_process_file, args): args for args in worker_args
        }

        # Collect results
        results = []
        for future in as_completed(future_to_args):
            result = future.result()
            results.append(result)

            if result["status"] == "success":
                total_rows += result["rows_out"]
                processed_files += 1
            else:
                error_files += 1

    # Write all results to manifest
    with open(manifest_file, "a", newline="") as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow(
                [
                    result["path_in"],
                    result["rows_out"],
                    result["pollutants_found"],
                    result["ts_utc"],
                    result["status"],
                    result["error"],
                ]
            )

    # Final summary
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Files processed successfully: {processed_files}")
    print(f"Files with errors: {error_files}")
    print(f"Total rows extracted: {total_rows:,}")
    print(f"Output directory: {out_root}")
    print(f"Manifest file: {manifest_file}")

    # Count unique pollutants found
    all_pollutants = set()
    for result in results:
        if result["pollutants_found"]:
            all_pollutants.update(result["pollutants_found"].split(","))

    if all_pollutants:
        print(f"Pollutants found: {', '.join(sorted(all_pollutants))}")

    # Exit with appropriate code
    sys.exit(0 if error_files == 0 else 1)


if __name__ == "__main__":
    main()
