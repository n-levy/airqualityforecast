#!/usr/bin/env python3
"""
Dual AQI Standards Demonstration

Shows EPA vs European AQI calculation differences using real data.
"""

from __future__ import annotations
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from calculate_aqi_dual_standard import (
    process_dataset_with_dual_aqi,
    compare_aqi_standards,
    calculate_individual_aqi,
    get_aqi_category,
    convert_units_for_eaqi,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def dual_aqi_demonstration():
    """Demonstrate both EPA and European AQI calculations with real data."""

    log.info("Starting dual AQI standards demonstration...")

    # Load sample of the dataset
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Heavy sampling for quick demo
    df_sampled = df.iloc[::100].copy().reset_index(drop=True)  # Every 100th record
    log.info(f"Using {len(df_sampled)} samples for quick dual-standard demo")

    # Process with both standards
    df_dual = process_dataset_with_dual_aqi(df_sampled, standards=["EPA", "EAQI"])

    # Generate comparison report
    compare_aqi_standards(df_dual)

    # Show sample conversions
    print(f"\nSAMPLE CONVERSIONS AND COMPARISONS:")
    print(
        f"{'Date':>12} {'PM2.5':>8} {'EPA AQI':>8} {'EPA Cat':>25} {'EAQI':>6} {'EAQI Cat':>15}"
    )
    print("-" * 85)

    # Show first 10 samples with valid data
    valid_samples = df_dual[
        (df_dual["aqi_composite_epa"].notna()) & (df_dual["aqi_composite_eaqi"].notna())
    ].head(10)

    for _, row in valid_samples.iterrows():
        date_str = (
            row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "Unknown"
        )
        pm25_val = row.get("actual_pm25", 0)
        epa_aqi = row["aqi_composite_epa"]
        epa_cat = row["aqi_level_epa"]
        eaqi_val = row["aqi_composite_eaqi"]
        eaqi_cat = row["aqi_level_eaqi"]

        print(
            f"{date_str:>12} {pm25_val:>8.1f} {epa_aqi:>8.0f} {epa_cat:>25} {eaqi_val:>6.0f} {eaqi_cat:>15}"
        )

    # Detailed unit conversion examples
    print(f"\nUNIT CONVERSION EXAMPLES:")
    print(
        f"{'Pollutant':>10} {'Original':>10} {'Unit':>6} {'EAQI Unit':>10} {'Converted':>10}"
    )
    print("-" * 60)

    conversion_examples = [
        ("NO2", 50, "ppb"),
        ("NO2", 100, "ppb"),
        ("O3", 60, "ppb"),
        ("O3", 120, "ppb"),
    ]

    for pollutant, value, unit in conversion_examples:
        converted = convert_units_for_eaqi(value, pollutant.lower(), unit)
        print(
            f"{pollutant:>10} {value:>10.1f} {unit:>6} {'ug/m3':>10} {converted:>10.1f}"
        )

    # Health warning threshold comparison
    print(f"\nHEALTH WARNING THRESHOLD COMPARISON:")
    print("EPA Thresholds:")
    print("  - Sensitive Groups: AQI >= 101 (Orange level)")
    print("  - General Population: AQI >= 151 (Red level)")
    print("\nEAQI Thresholds:")
    print("  - Sensitive Groups: Level >= 4 (Poor)")
    print("  - General Population: Level >= 5 (Very Poor)")

    # Summary statistics
    if (
        "aqi_composite_epa" in df_dual.columns
        and "aqi_composite_eaqi" in df_dual.columns
    ):
        epa_mean = df_dual["aqi_composite_epa"].mean()
        eaqi_mean = df_dual["aqi_composite_eaqi"].mean()
        correlation = (
            df_dual[["aqi_composite_epa", "aqi_composite_eaqi"]].corr().iloc[0, 1]
        )

        print(f"\nSUMMARY STATISTICS:")
        print(f"EPA AQI Mean: {epa_mean:.1f}")
        print(f"EAQI Mean: {eaqi_mean:.2f}")
        print(f"Correlation: {correlation:.3f}")

    print(f"\n" + "=" * 80)
    print("DUAL AQI DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Key Findings:")
    print("- EPA uses 0-500 scale with linear interpolation")
    print("- EAQI uses 1-6 scale with step functions")
    print("- Different pollutant units and breakpoints")
    print("- Both methods identify air quality trends similarly")
    print("- Choose standard based on regional requirements")
    print("=" * 80)


def main():
    """Main execution function."""
    dual_aqi_demonstration()
    return 0


if __name__ == "__main__":
    exit(main())
