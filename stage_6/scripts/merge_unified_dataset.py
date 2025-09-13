#!/usr/bin/env python3
"""
Stage 6 Merge: Unified Dataset Creation
======================================

Merges outputs from all Stage 6 ETL pipelines into a unified 6-hourly dataset:
- Ground truth observations (WAQI, OpenAQ)
- NOAA GEFS-Aerosol forecasts
- ECMWF CAMS atmospheric composition
- Local features (calendar, weather, geographic)

Creates comprehensive air quality datasets for model training and analysis.
Cross-platform implementation supporting Linux/macOS/Windows.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Cross-platform data root
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "aqf_data"))
STAGE6_DIR = DATA_ROOT / "curated" / "stage6"
OUTPUT_DIR = STAGE6_DIR / "unified"


class UnifiedDatasetMerger:
    """Merges all Stage 6 ETL outputs into unified datasets."""

    def __init__(self):
        """Initialize the merger."""
        self.setup_output_directory()

    def setup_output_directory(self):
        """Create output directory structure."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {OUTPUT_DIR}")

    def find_latest_etl_outputs(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Path]:
        """Find the latest ETL output files for the date range."""
        log.info("Finding latest ETL outputs...")

        date_prefix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        outputs = {}

        # Ground truth data
        ground_truth_dir = STAGE6_DIR / "ground_truth"
        if ground_truth_dir.exists():
            gt_files = list(
                ground_truth_dir.glob(f"ground_truth_{date_prefix}_*.parquet")
            )
            if gt_files:
                outputs["ground_truth"] = sorted(gt_files)[-1]  # Latest file
                log.info(f"Found ground truth: {outputs['ground_truth'].name}")

        # NOAA GEFS forecasts
        gefs_dir = STAGE6_DIR / "noaa_gefs"
        if gefs_dir.exists():
            gefs_files = list(gefs_dir.glob(f"gefs_forecasts_{date_prefix}_*.parquet"))
            if gefs_files:
                outputs["noaa_gefs"] = sorted(gefs_files)[-1]
                log.info(f"Found NOAA GEFS: {outputs['noaa_gefs'].name}")

        # ECMWF CAMS data
        cams_dir = STAGE6_DIR / "cams"
        if cams_dir.exists():
            cams_files = list(cams_dir.glob(f"cams_data_{date_prefix}_*.parquet"))
            if cams_files:
                outputs["cams"] = sorted(cams_files)[-1]
                log.info(f"Found ECMWF CAMS: {outputs['cams'].name}")

        # Local features
        features_dir = STAGE6_DIR / "local_features"
        if features_dir.exists():
            feat_files = list(
                features_dir.glob(f"local_features_{date_prefix}_*.parquet")
            )
            if feat_files:
                outputs["local_features"] = sorted(feat_files)[-1]
                log.info(f"Found local features: {outputs['local_features'].name}")

        log.info(f"Found {len(outputs)} ETL output files")
        return outputs

    def load_and_standardize_dataframe(
        self, file_path: Path, source_type: str
    ) -> pd.DataFrame:
        """Load and standardize a dataframe from ETL output."""
        log.info(f"Loading {source_type} from {file_path.name}")

        try:
            df = pd.read_parquet(file_path)

            # Ensure UTC timestamps
            if "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

            # Add source metadata
            df["etl_source"] = source_type
            df["etl_file"] = file_path.name

            log.info(f"Loaded {len(df):,} records from {source_type}")
            return df

        except Exception as e:
            log.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def align_timestamps_to_6hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align all timestamps to 6-hourly intervals (00:00, 06:00, 12:00, 18:00)."""
        if "timestamp_utc" not in df.columns:
            return df

        # Round timestamps to nearest 6-hour interval
        def round_to_6hourly(timestamp):
            hour = timestamp.hour
            if hour < 3:
                new_hour = 0
            elif hour < 9:
                new_hour = 6
            elif hour < 15:
                new_hour = 12
            elif hour < 21:
                new_hour = 18
            else:
                new_hour = 0
                timestamp = timestamp + timedelta(days=1)

            return timestamp.replace(hour=new_hour, minute=0, second=0, microsecond=0)

        df["timestamp_utc"] = df["timestamp_utc"].apply(round_to_6hourly)
        return df

    def merge_pollutant_data(
        self, ground_truth_df: pd.DataFrame, forecast_dfs: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge pollutant data from ground truth and forecasts."""
        log.info("Merging pollutant data...")

        pollutant_data = []

        # Add ground truth data
        if not ground_truth_df.empty:
            gt_subset = ground_truth_df[
                [
                    "city",
                    "country",
                    "latitude",
                    "longitude",
                    "timestamp_utc",
                    "pollutant",
                    "value",
                    "units",
                    "source",
                    "data_type",
                    "quality_flag",
                ]
            ].copy()
            gt_subset["measurement_type"] = "observation"
            pollutant_data.append(gt_subset)

        # Add forecast data
        for forecast_df in forecast_dfs:
            if not forecast_df.empty and "pollutant" in forecast_df.columns:
                fc_subset = forecast_df[
                    [
                        "city",
                        "country",
                        "latitude",
                        "longitude",
                        "timestamp_utc",
                        "pollutant",
                        "value",
                        "units",
                        "source",
                        "data_type",
                        "quality_flag",
                    ]
                ].copy()
                fc_subset["measurement_type"] = "forecast"
                pollutant_data.append(fc_subset)

        if pollutant_data:
            combined_df = pd.concat(pollutant_data, ignore_index=True)
            combined_df = self.align_timestamps_to_6hourly(combined_df)

            # Remove duplicates (keep first occurrence)
            combined_df = combined_df.drop_duplicates(
                subset=["city", "timestamp_utc", "pollutant", "source"], keep="first"
            )

            log.info(f"Merged pollutant data: {len(combined_df):,} records")
            return combined_df

        return pd.DataFrame()

    def create_wide_format_dataset(
        self, pollutant_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create wide-format dataset with pollutants as columns and features."""
        log.info("Creating wide-format dataset...")

        # Pivot pollutant data to wide format
        pollutant_wide = pollutant_df.pivot_table(
            index=["city", "country", "timestamp_utc"],
            columns=["pollutant", "source"],
            values="value",
            aggfunc="first",  # Take first value if duplicates
        )

        # Flatten column names
        pollutant_wide.columns = [
            f"{pol}_{src}".replace(" ", "_") for pol, src in pollutant_wide.columns
        ]
        pollutant_wide = pollutant_wide.reset_index()

        # Merge with features
        if not features_df.empty:
            # Select relevant feature columns
            feature_cols = [
                col
                for col in features_df.columns
                if col
                not in [
                    "city",
                    "country",
                    "timestamp_utc",
                    "source",
                    "data_type",
                    "quality_flag",
                ]
            ]

            features_subset = features_df[
                ["city", "timestamp_utc"] + feature_cols
            ].copy()
            features_subset = self.align_timestamps_to_6hourly(features_subset)

            # Remove duplicates
            features_subset = features_subset.drop_duplicates(
                subset=["city", "timestamp_utc"], keep="first"
            )

            # Merge pollutants with features
            wide_df = pollutant_wide.merge(
                features_subset, on=["city", "timestamp_utc"], how="left"
            )
        else:
            wide_df = pollutant_wide

        log.info(
            f"Wide-format dataset: {len(wide_df):,} rows × {len(wide_df.columns)} columns"
        )
        return wide_df

    def create_long_format_dataset(
        self,
        ground_truth_df: pd.DataFrame,
        forecast_dfs: List[pd.DataFrame],
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create long-format dataset with one row per measurement."""
        log.info("Creating long-format dataset...")

        # Merge all pollutant data
        pollutant_df = self.merge_pollutant_data(ground_truth_df, forecast_dfs)

        if pollutant_df.empty:
            log.warning("No pollutant data available for long format")
            return pd.DataFrame()

        # Merge with features
        if not features_df.empty:
            # Select non-overlapping feature columns
            feature_cols = [
                col
                for col in features_df.columns
                if col not in pollutant_df.columns or col in ["city", "timestamp_utc"]
            ]

            features_subset = features_df[feature_cols].copy()
            features_subset = self.align_timestamps_to_6hourly(features_subset)

            # Remove duplicates
            features_subset = features_subset.drop_duplicates(
                subset=["city", "timestamp_utc"], keep="first"
            )

            # Merge
            long_df = pollutant_df.merge(
                features_subset, on=["city", "timestamp_utc"], how="left"
            )
        else:
            long_df = pollutant_df

        # Sort by city, timestamp, pollutant
        long_df = long_df.sort_values(["city", "timestamp_utc", "pollutant"])

        log.info(
            f"Long-format dataset: {len(long_df):,} rows × {len(long_df.columns)} columns"
        )
        return long_df

    def run_merge(
        self, start_date: datetime, end_date: datetime, output_formats: List[str] = None
    ) -> Dict[str, str]:
        """Run complete merge pipeline."""
        log.info("=== UNIFIED DATASET MERGE PIPELINE ===")
        log.info(f"Period: {start_date.date()} to {end_date.date()}")

        if output_formats is None:
            output_formats = ["wide", "long"]

        # Find ETL outputs
        etl_outputs = self.find_latest_etl_outputs(start_date, end_date)

        if not etl_outputs:
            log.error("No ETL outputs found!")
            return {}

        # Load dataframes
        dfs = {}
        for source_type, file_path in etl_outputs.items():
            df = self.load_and_standardize_dataframe(file_path, source_type)
            if not df.empty:
                dfs[source_type] = df

        if not dfs:
            log.error("No valid dataframes loaded!")
            return {}

        # Separate by data type
        ground_truth_df = dfs.get("ground_truth", pd.DataFrame())
        features_df = dfs.get("local_features", pd.DataFrame())
        forecast_dfs = [
            df
            for key, df in dfs.items()
            if key in ["noaa_gefs", "cams"] and not df.empty
        ]

        # Create output files
        output_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_suffix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        # Wide format dataset
        if "wide" in output_formats:
            pollutant_df = self.merge_pollutant_data(ground_truth_df, forecast_dfs)
            if not pollutant_df.empty:
                wide_df = self.create_wide_format_dataset(pollutant_df, features_df)
                wide_file = (
                    OUTPUT_DIR / f"unified_wide_{date_suffix}_{timestamp}.parquet"
                )
                wide_df.to_parquet(wide_file, index=False)
                output_files["wide"] = str(wide_file)
                log.info(f"Wide format saved: {wide_file.name}")

        # Long format dataset
        if "long" in output_formats:
            long_df = self.create_long_format_dataset(
                ground_truth_df, forecast_dfs, features_df
            )
            if not long_df.empty:
                long_file = (
                    OUTPUT_DIR / f"unified_long_{date_suffix}_{timestamp}.parquet"
                )
                long_df.to_parquet(long_file, index=False)
                output_files["long"] = str(long_file)
                log.info(f"Long format saved: {long_file.name}")

        # Create partitioned outputs
        self.create_partitioned_outputs(output_files, start_date, end_date)

        log.info("=== UNIFIED DATASET MERGE COMPLETE ===")
        log.info(f"Output files: {len(output_files)}")
        for format_type, file_path in output_files.items():
            log.info(f"  {format_type}: {Path(file_path).name}")

        return output_files

    def create_partitioned_outputs(
        self, output_files: Dict[str, str], start_date: datetime, end_date: datetime
    ):
        """Create partitioned versions of the unified datasets."""
        log.info("Creating partitioned outputs...")

        date_suffix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        partition_base = OUTPUT_DIR / "partitioned" / f"unified_{date_suffix}"

        for format_type, file_path in output_files.items():
            try:
                df = pd.read_parquet(file_path)

                # Create city partitions
                format_dir = partition_base / format_type
                format_dir.mkdir(parents=True, exist_ok=True)

                for city in df["city"].unique():
                    city_df = df[df["city"] == city]
                    city_dir = format_dir / f"city={city}"
                    city_dir.mkdir(parents=True, exist_ok=True)

                    city_file = city_dir / "data.parquet"
                    city_df.to_parquet(city_file, index=False)

                log.info(f"Partitioned {format_type} data saved to: {format_dir}")

            except Exception as e:
                log.error(f"Error creating partitions for {format_type}: {e}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Dataset Merge Pipeline")
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="wide,long",
        help="Output formats: wide,long (default: both)",
    )

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        output_formats = args.formats.split(",")

        merger = UnifiedDatasetMerger()
        output_files = merger.run_merge(start_date, end_date, output_formats)

        if output_files:
            log.info("Unified dataset merge completed successfully!")
            return 0
        else:
            log.error("Unified dataset merge failed!")
            return 1

    except Exception as e:
        log.error(f"Merge execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
