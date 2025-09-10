#!/usr/bin/env python3
"""
Air Quality Index (AQI) Calculation Module

Implements standard EPA Air Quality Index calculation for multiple pollutants.
Supports PM2.5, PM10, NO2, and O3 with health-based categorical classification.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# EPA Air Quality Index Breakpoints
# Format: {pollutant: [(C_lo, C_hi, I_lo, I_hi), ...]}
AQI_BREAKPOINTS = {
    "pm25": [
        (0.0, 12.0, 0, 50),  # Good
        (12.1, 35.4, 51, 100),  # Moderate
        (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),  # Unhealthy
        (150.5, 250.4, 201, 300),  # Very Unhealthy
        (250.5, 500.4, 301, 500),  # Hazardous
    ],
    "pm10": [
        (0, 54, 0, 50),  # Good
        (55, 154, 51, 100),  # Moderate
        (155, 254, 101, 150),  # Unhealthy for Sensitive Groups
        (255, 354, 151, 200),  # Unhealthy
        (355, 424, 201, 300),  # Very Unhealthy
        (425, 604, 301, 500),  # Hazardous
    ],
    "no2": [
        (0, 53, 0, 50),  # Good
        (54, 100, 51, 100),  # Moderate
        (101, 360, 101, 150),  # Unhealthy for Sensitive Groups
        (361, 649, 151, 200),  # Unhealthy
        (650, 1249, 201, 300),  # Very Unhealthy
        (1250, 2049, 301, 500),  # Hazardous
    ],
    "o3": [
        (0, 54, 0, 50),  # Good (8-hour average)
        (55, 70, 51, 100),  # Moderate
        (71, 85, 101, 150),  # Unhealthy for Sensitive Groups
        (86, 105, 151, 200),  # Unhealthy
        (106, 200, 201, 300),  # Very Unhealthy
        (201, 504, 301, 500),  # Hazardous
    ],
}

# AQI Category Definitions
AQI_CATEGORIES = {
    (0, 50): {
        "level": "Good",
        "color": "Green",
        "health_message": "Air quality is satisfactory",
    },
    (51, 100): {
        "level": "Moderate",
        "color": "Yellow",
        "health_message": "Air quality is acceptable for most people",
    },
    (101, 150): {
        "level": "Unhealthy for Sensitive Groups",
        "color": "Orange",
        "health_message": "Sensitive groups may experience health effects",
    },
    (151, 200): {
        "level": "Unhealthy",
        "color": "Red",
        "health_message": "Everyone may experience health effects",
    },
    (201, 300): {
        "level": "Very Unhealthy",
        "color": "Purple",
        "health_message": "Health alert: serious health effects for everyone",
    },
    (301, 500): {
        "level": "Hazardous",
        "color": "Maroon",
        "health_message": "Health warning: emergency conditions",
    },
}


def calculate_individual_aqi(concentration: float, pollutant: str) -> float:
    """
    Calculate AQI for a single pollutant concentration.

    Args:
        concentration: Pollutant concentration in μg/m³
        pollutant: Pollutant name ('pm25', 'pm10', 'no2', 'o3')

    Returns:
        AQI value (0-500+)
    """
    if pollutant not in AQI_BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")

    if pd.isna(concentration) or concentration < 0:
        return np.nan

    breakpoints = AQI_BREAKPOINTS[pollutant]

    # Find appropriate breakpoint
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            # Linear interpolation formula
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
            return round(aqi)

    # Handle concentrations above highest breakpoint
    # Use highest category (Hazardous)
    c_lo, c_hi, i_lo, i_hi = breakpoints[-1]
    if concentration > c_hi:
        # Extrapolate linearly
        aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
        return round(aqi)

    return np.nan


def calculate_composite_aqi(concentrations: Dict[str, float]) -> Tuple[float, str]:
    """
    Calculate composite AQI from multiple pollutant concentrations.

    Args:
        concentrations: Dictionary of {pollutant: concentration}

    Returns:
        Tuple of (composite_aqi, dominant_pollutant)
    """
    individual_aqis = {}

    for pollutant, concentration in concentrations.items():
        if pollutant in AQI_BREAKPOINTS:
            aqi = calculate_individual_aqi(concentration, pollutant)
            if not pd.isna(aqi):
                individual_aqis[pollutant] = aqi

    if not individual_aqis:
        return np.nan, "unknown"

    # Composite AQI is the maximum of individual AQIs
    max_aqi = max(individual_aqis.values())
    dominant_pollutant = max(individual_aqis, key=individual_aqis.get)

    return max_aqi, dominant_pollutant


def get_aqi_category(aqi_value: float) -> Dict[str, str]:
    """
    Get AQI category information for a given AQI value.

    Args:
        aqi_value: AQI value (0-500+)

    Returns:
        Dictionary with level, color, and health_message
    """
    if pd.isna(aqi_value):
        return {
            "level": "Unknown",
            "color": "Gray",
            "health_message": "Data unavailable",
        }

    for (min_aqi, max_aqi), category_info in AQI_CATEGORIES.items():
        if min_aqi <= aqi_value <= max_aqi:
            return category_info

    # Handle values above 500 (Hazardous+)
    if aqi_value > 500:
        return {
            "level": "Hazardous",
            "color": "Maroon",
            "health_message": "Health warning: emergency conditions",
        }

    return {"level": "Unknown", "color": "Gray", "health_message": "Invalid AQI value"}


def is_health_warning_required(aqi_value: float, sensitive_groups: bool = True) -> bool:
    """
    Determine if health warning is required based on AQI value.

    Args:
        aqi_value: AQI value
        sensitive_groups: If True, warn for sensitive groups (AQI >= 101)
                         If False, warn for general population (AQI >= 151)

    Returns:
        Boolean indicating if warning is required
    """
    if pd.isna(aqi_value):
        return False

    threshold = 101 if sensitive_groups else 151
    return aqi_value >= threshold


def process_dataset_with_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process entire dataset to add AQI calculations.

    Args:
        df: DataFrame with pollutant concentration columns

    Returns:
        DataFrame with added AQI columns
    """
    log.info("Processing dataset with AQI calculations...")

    df_aqi = df.copy()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    # Calculate individual AQIs
    for pollutant in pollutants:
        actual_col = f"actual_{pollutant}"
        if actual_col in df_aqi.columns:
            aqi_col = f"aqi_{pollutant}"
            df_aqi[aqi_col] = df_aqi[actual_col].apply(
                lambda x: calculate_individual_aqi(x, pollutant)
            )

    # Calculate composite AQI for actual values
    log.info("Calculating composite AQI...")

    def calculate_row_aqi(row):
        concentrations = {}
        for pollutant in pollutants:
            actual_col = f"actual_{pollutant}"
            if actual_col in row and not pd.isna(row[actual_col]):
                concentrations[pollutant] = row[actual_col]

        if concentrations:
            composite_aqi, dominant_pollutant = calculate_composite_aqi(concentrations)
            return pd.Series(
                {
                    "aqi_composite": composite_aqi,
                    "aqi_dominant_pollutant": dominant_pollutant,
                }
            )
        else:
            return pd.Series(
                {"aqi_composite": np.nan, "aqi_dominant_pollutant": "unknown"}
            )

    aqi_results = df_aqi.apply(calculate_row_aqi, axis=1)
    df_aqi = pd.concat([df_aqi, aqi_results], axis=1)

    # Add categorical information
    log.info("Adding AQI categorical information...")

    def get_category_info(aqi_value):
        category = get_aqi_category(aqi_value)
        return pd.Series(
            {
                "aqi_level": category["level"],
                "aqi_color": category["color"],
                "aqi_health_message": category["health_message"],
            }
        )

    category_results = df_aqi["aqi_composite"].apply(get_category_info)
    df_aqi = pd.concat([df_aqi, category_results], axis=1)

    # Add health warning flags
    df_aqi["health_warning_sensitive"] = df_aqi["aqi_composite"].apply(
        lambda x: is_health_warning_required(x, sensitive_groups=True)
    )
    df_aqi["health_warning_general"] = df_aqi["aqi_composite"].apply(
        lambda x: is_health_warning_required(x, sensitive_groups=False)
    )

    log.info(
        f"AQI processing complete. Added {len([col for col in df_aqi.columns if col.startswith('aqi_') or col.startswith('health_')])} new columns"
    )

    return df_aqi


def generate_aqi_summary_report(df_aqi: pd.DataFrame) -> None:
    """
    Generate summary report of AQI distribution and health warnings.

    Args:
        df_aqi: DataFrame with AQI calculations
    """
    print("\n" + "=" * 80)
    print("AIR QUALITY INDEX (AQI) ANALYSIS SUMMARY")
    print("=" * 80)

    if "aqi_composite" not in df_aqi.columns:
        print("ERROR: No AQI data found in dataset")
        return

    # Overall AQI statistics
    aqi_stats = df_aqi["aqi_composite"].describe()
    print(f"\nOVERALL AQI STATISTICS:")
    print(f"Count: {int(aqi_stats['count']):,} observations")
    print(f"Mean AQI: {aqi_stats['mean']:.1f}")
    print(f"Median AQI: {aqi_stats['50%']:.1f}")
    print(f"Min AQI: {aqi_stats['min']:.1f}")
    print(f"Max AQI: {aqi_stats['max']:.1f}")
    print(f"Std Dev: {aqi_stats['std']:.1f}")

    # AQI category distribution
    print(f"\nAQI CATEGORY DISTRIBUTION:")
    if "aqi_level" in df_aqi.columns:
        category_counts = df_aqi["aqi_level"].value_counts()
        total_valid = category_counts.sum()

        for category, count in category_counts.items():
            percentage = (count / total_valid) * 100
            print(f"{category:30}: {count:6,} observations ({percentage:5.1f}%)")

    # Health warning analysis
    print(f"\nHEALTH WARNING ANALYSIS:")

    if "health_warning_sensitive" in df_aqi.columns:
        sensitive_warnings = df_aqi["health_warning_sensitive"].sum()
        sensitive_pct = (sensitive_warnings / len(df_aqi)) * 100
        print(
            f"Sensitive Group Warnings (AQI >= 101): {sensitive_warnings:,} days ({sensitive_pct:.1f}%)"
        )

    if "health_warning_general" in df_aqi.columns:
        general_warnings = df_aqi["health_warning_general"].sum()
        general_pct = (general_warnings / len(df_aqi)) * 100
        print(
            f"General Population Warnings (AQI >= 151): {general_warnings:,} days ({general_pct:.1f}%)"
        )

    # Dominant pollutant analysis
    if "aqi_dominant_pollutant" in df_aqi.columns:
        print(f"\nDOMINANT POLLUTANT ANALYSIS:")
        dominant_counts = df_aqi["aqi_dominant_pollutant"].value_counts()
        for pollutant, count in dominant_counts.items():
            percentage = (count / len(df_aqi)) * 100
            print(
                f"{pollutant.upper():10}: {count:6,} times dominant ({percentage:5.1f}%)"
            )

    # Seasonal analysis if date information available
    if "date" in df_aqi.columns or "datetime" in df_aqi.columns:
        print(f"\nSEASONAL AQI PATTERNS:")

        date_col = "date" if "date" in df_aqi.columns else "datetime"
        if not pd.api.types.is_datetime64_any_dtype(df_aqi[date_col]):
            df_aqi[date_col] = pd.to_datetime(df_aqi[date_col])

        df_aqi["month"] = df_aqi[date_col].dt.month
        monthly_aqi = (
            df_aqi.groupby("month")["aqi_composite"].agg(["mean", "count"]).round(1)
        )

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for month in range(1, 13):
            if month in monthly_aqi.index:
                month_name = month_names[month - 1]
                avg_aqi = monthly_aqi.loc[month, "mean"]
                count = int(monthly_aqi.loc[month, "count"])
                print(
                    f"{month_name}: Average AQI {avg_aqi:5.1f} ({count:,} observations)"
                )

    print("\n" + "=" * 80)
    print("AQI ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    """Test AQI calculation functions with sample data."""

    # Test individual AQI calculations
    print("Testing individual AQI calculations:")
    test_concentrations = {
        "pm25": [5.0, 15.0, 40.0, 65.0, 200.0],
        "pm10": [25.0, 75.0, 200.0, 300.0, 450.0],
        "no2": [30.0, 80.0, 200.0, 500.0, 1000.0],
        "o3": [40.0, 60.0, 80.0, 95.0, 150.0],
    }

    for pollutant, concentrations in test_concentrations.items():
        print(f"\n{pollutant.upper()} AQI calculations:")
        for conc in concentrations:
            aqi = calculate_individual_aqi(conc, pollutant)
            category = get_aqi_category(aqi)
            print(f"  {conc:6.1f} ug/m3 -> AQI {aqi:3.0f} ({category['level']})")

    # Test composite AQI
    print(f"\nTesting composite AQI calculation:")
    test_composite = {
        "pm25": 25.0,  # AQI ~88
        "pm10": 100.0,  # AQI ~66
        "no2": 150.0,  # AQI ~126
        "o3": 65.0,  # AQI ~80
    }

    composite_aqi, dominant = calculate_composite_aqi(test_composite)
    category = get_aqi_category(composite_aqi)

    print(f"Concentrations: {test_composite}")
    print(f"Composite AQI: {composite_aqi} (dominant: {dominant})")
    print(f"Category: {category['level']} - {category['health_message']}")

    # Test health warnings
    print(f"\nHealth warning tests:")
    test_aqis = [45, 85, 115, 165, 225]
    for aqi in test_aqis:
        sensitive_warning = is_health_warning_required(aqi, sensitive_groups=True)
        general_warning = is_health_warning_required(aqi, sensitive_groups=False)
        print(
            f"AQI {aqi:3d}: Sensitive warning: {sensitive_warning}, General warning: {general_warning}"
        )


if __name__ == "__main__":
    main()
