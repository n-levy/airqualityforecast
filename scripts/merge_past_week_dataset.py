#!/usr/bin/env python3
"""
Past Week Dataset Merger
========================

Merges the collected past week data (GEFS, CAMS, ground truth, local features)
into a unified dataset with 6-hour frequency and standardized schema.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logs_dir = Path("C:/aqf311/data/logs")
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "past_week_merge.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Unit conversion functions
def standardize_units(value, pollutant):
    """Convert pollutant values to standardized units."""
    if pd.isna(value) or value is None:
        return None

    pollutant = pollutant.lower()

    # PM pollutants - target μg/m³ (already in correct units from simulation)
    if pollutant in ["pm25", "pm10"]:
        return value

    # Gas pollutants - target ppb (already in correct units from simulation)
    else:
        return value


def get_standard_units(pollutant):
    """Get the standard unit for a pollutant."""
    pollutant = pollutant.lower()
    if pollutant in ["pm25", "pm10"]:
        return "μg/m³"
    else:
        return "ppb"


def load_and_process_gefs_data(data_root):
    """Load and process GEFS data."""
    log.info("Loading GEFS data...")

    gefs_dir = Path(data_root) / "curated" / "gefs_chem" / "parquet"
    parquet_files = list(gefs_dir.glob("gefs_past_week_*.parquet"))

    if not parquet_files:
        log.warning("No GEFS past week files found")
        return []

    all_records = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            log.info(f"Loaded GEFS file: {file} ({len(df)} records)")

            # Convert to long format for each pollutant
            for _, row in df.iterrows():
                for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                    if pollutant in row and pd.notna(row[pollutant]):
                        record = {
                            "city": row["city"],
                            "country": row["country"],
                            "timestamp_utc": pd.to_datetime(row["forecast_time"]),
                            "pollutant": pollutant.upper(),
                            "value": standardize_units(row[pollutant], pollutant),
                            "units": get_standard_units(pollutant),
                            "source": "GEFS",
                            "data_type": "forecast",
                            "run_date": row["run_date"],
                            "run_hour": row["run_hour"],
                            "f_hour": row["forecast_hour"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "model_version": row["model_version"],
                            "quality_flag": "good",
                        }
                        all_records.append(record)

        except Exception as e:
            log.error(f"Error loading GEFS file {file}: {e}")

    log.info(f"Processed {len(all_records)} GEFS records")
    return all_records


def load_and_process_cams_data(data_root):
    """Load and process CAMS data."""
    log.info("Loading CAMS data...")

    cams_dir = Path(data_root) / "curated" / "cams" / "parquet"
    parquet_files = list(cams_dir.glob("cams_past_week_*.parquet"))

    if not parquet_files:
        log.warning("No CAMS past week files found")
        return []

    all_records = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            log.info(f"Loaded CAMS file: {file} ({len(df)} records)")

            # Convert to long format for each pollutant
            for _, row in df.iterrows():
                for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                    if pollutant in row and pd.notna(row[pollutant]):
                        record = {
                            "city": row["city"],
                            "country": row["country"],
                            "timestamp_utc": pd.to_datetime(row["forecast_time"]),
                            "pollutant": pollutant.upper(),
                            "value": standardize_units(row[pollutant], pollutant),
                            "units": get_standard_units(pollutant),
                            "source": "CAMS",
                            "data_type": (
                                "forecast" if row["forecast_hour"] > 0 else "analysis"
                            ),
                            "run_date": row["run_date"],
                            "run_hour": row["run_hour"],
                            "f_hour": row["forecast_hour"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "model_version": row["model_version"],
                            "quality_flag": "good",
                        }
                        all_records.append(record)

        except Exception as e:
            log.error(f"Error loading CAMS file {file}: {e}")

    log.info(f"Processed {len(all_records)} CAMS records")
    return all_records


def load_and_process_obs_data(data_root):
    """Load and process ground truth observation data."""
    log.info("Loading ground truth observation data...")

    obs_dir = Path(data_root) / "curated" / "obs"
    parquet_files = list(obs_dir.glob("obs_past_week_*.parquet"))

    if not parquet_files:
        log.warning("No observation past week files found")
        return []

    all_records = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            log.info(f"Loaded observation file: {file} ({len(df)} records)")

            # Convert to long format for each pollutant
            for _, row in df.iterrows():
                for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                    if pollutant in row and pd.notna(row[pollutant]):
                        record = {
                            "city": row["city"],
                            "country": row["country"],
                            "timestamp_utc": pd.to_datetime(row["timestamp_utc"]),
                            "pollutant": pollutant.upper(),
                            "value": standardize_units(row[pollutant], pollutant),
                            "units": get_standard_units(pollutant),
                            "source": "Ground-Truth",
                            "data_type": "observation",
                            "run_date": None,
                            "run_hour": None,
                            "f_hour": 0,
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "model_version": None,
                            "quality_flag": "good",
                            # Calendar features from observation data
                            "year": row.get("year"),
                            "month": row.get("month"),
                            "day": row.get("day"),
                            "hour": row.get("hour"),
                            "day_of_week": row.get("day_of_week"),
                            "day_of_year": row.get("day_of_year"),
                            "week_of_year": row.get("week_of_year"),
                            "is_weekend": row.get("is_weekend"),
                            "is_holiday_season": row.get("is_holiday_season"),
                            "season": row.get("season"),
                            "hour_sin": row.get("hour_sin"),
                            "hour_cos": row.get("hour_cos"),
                            "month_sin": row.get("month_sin"),
                            "month_cos": row.get("month_cos"),
                            "day_sin": row.get("day_sin"),
                            "day_cos": row.get("day_cos"),
                        }
                        all_records.append(record)

        except Exception as e:
            log.error(f"Error loading observation file {file}: {e}")

    log.info(f"Processed {len(all_records)} observation records")
    return all_records


def add_calendar_features(record):
    """Add calendar features to a record if missing."""
    if (
        "timestamp_utc" in record
        and record["timestamp_utc"]
        and pd.isna(record.get("year"))
    ):
        dt = pd.to_datetime(record["timestamp_utc"])

        record.update(
            {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "day_of_week": dt.dayofweek,
                "day_of_year": dt.dayofyear,
                "week_of_year": dt.isocalendar()[1],
                "is_weekend": dt.dayofweek >= 5,
                "is_holiday_season": dt.month in [11, 12, 1],
                "season": (dt.month % 12 + 3) // 3,
                "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
                "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
                "month_sin": np.sin(2 * np.pi * dt.month / 12),
                "month_cos": np.cos(2 * np.pi * dt.month / 12),
                "day_sin": np.sin(2 * np.pi * dt.day / 31),
                "day_cos": np.cos(2 * np.pi * dt.day / 31),
            }
        )

    return record


def merge_past_week_data(data_root):
    """Merge all past week data sources."""
    log.info("=== MERGING PAST WEEK DATASET ===")

    # Load all data sources
    gefs_records = load_and_process_gefs_data(data_root)
    cams_records = load_and_process_cams_data(data_root)
    obs_records = load_and_process_obs_data(data_root)

    # Combine all records
    all_records = gefs_records + cams_records + obs_records
    log.info(f"Total records to merge: {len(all_records)}")

    if not all_records:
        log.error("No data to merge!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Standardize timezone handling
    log.info("Standardizing timezone handling...")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Add calendar features where missing
    log.info("Adding calendar features...")
    for idx, row in df.iterrows():
        if pd.isna(row.get("year")):
            df.loc[idx] = add_calendar_features(row.to_dict())

    # Sort by city, timestamp, pollutant, source
    df = df.sort_values(["city", "timestamp_utc", "pollutant", "source"])

    # Remove rows with missing essential data
    essential_cols = ["city", "timestamp_utc", "pollutant", "value"]
    before_count = len(df)
    df = df.dropna(subset=essential_cols)
    after_count = len(df)

    log.info(f"Removed {before_count - after_count} rows missing essential data")
    log.info(f"Final unified dataset: {len(df)} records")

    return df


def save_unified_dataset(df, data_root):
    """Save the unified past week dataset."""
    log.info("Saving unified past week dataset...")

    output_dir = Path(data_root) / "curated" / "100_cities_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    complete_file = output_dir / f"past_week_unified_dataset_{timestamp}.parquet"

    df.to_parquet(complete_file, index=False)
    log.info(f"Unified dataset saved: {complete_file}")

    # Generate summary statistics
    summary = {
        "generation_date": datetime.now().isoformat(),
        "total_records": len(df),
        "cities_count": df["city"].nunique(),
        "date_range": {
            "start": df["timestamp_utc"].min().isoformat(),
            "end": df["timestamp_utc"].max().isoformat(),
        },
        "frequency": "6-hour intervals",
        "data_sources": {
            source: len(df[df["source"] == source]) for source in df["source"].unique()
        },
        "pollutant_coverage": {
            pollutant: {
                "records": len(df[df["pollutant"] == pollutant]),
                "cities": df[df["pollutant"] == pollutant]["city"].nunique(),
            }
            for pollutant in sorted(df["pollutant"].unique())
        },
        "output_file": str(complete_file),
        "file_size_mb": complete_file.stat().st_size / (1024**2),
    }

    # Save summary
    summary_file = output_dir / f"past_week_dataset_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Dataset summary saved: {summary_file}")

    # Print summary
    log.info("=== PAST WEEK DATASET SUMMARY ===")
    log.info(f"Total Records: {summary['total_records']:,}")
    log.info(f"Cities: {summary['cities_count']}")
    log.info(
        f"Date Range: {summary['date_range']['start'][:19]} to {summary['date_range']['end'][:19]}"
    )
    log.info(f"Frequency: {summary['frequency']}")
    log.info(f"File Size: {summary['file_size_mb']:.2f} MB")
    log.info("")
    log.info("Data Sources:")
    for source, count in summary["data_sources"].items():
        percentage = count / summary["total_records"] * 100
        log.info(f"  {source}: {count:,} records ({percentage:.1f}%)")
    log.info("")
    log.info("Pollutant Coverage:")
    for pollutant, stats in summary["pollutant_coverage"].items():
        log.info(
            f"  {pollutant}: {stats['records']:,} records, {stats['cities']} cities"
        )

    return complete_file, summary


def main():
    """Main execution function."""
    data_root = os.environ.get("DATA_ROOT", "C:/aqf311/data")

    log.info(f"Using data root: {data_root}")

    try:
        # Merge all data
        df = merge_past_week_data(data_root)

        if df is None or len(df) == 0:
            log.error("Failed to merge data or no data available")
            return False

        # Save unified dataset
        output_file, summary = save_unified_dataset(df, data_root)

        log.info("Past week dataset merge completed successfully!")
        return True

    except Exception as e:
        log.error(f"Merge failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
