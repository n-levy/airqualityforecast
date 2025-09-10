#!/usr/bin/env python3
"""
Dual-Standard Air Quality Index (AQI) Calculation Module

Supports both EPA (US) and European Air Quality Index (EAQI) calculation methods.
Implements standard calculations for PM2.5, PM10, NO2, and O3 with health-based classification.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple, Union, Literal
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# Type definitions
AQIStandard = Literal["EPA", "EAQI"]

# EPA Air Quality Index Breakpoints (US Standard)
# Format: {pollutant: [(C_lo, C_hi, I_lo, I_hi), ...]}
EPA_BREAKPOINTS = {
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
        (0, 53, 0, 50),  # Good (ppb)
        (54, 100, 51, 100),  # Moderate
        (101, 360, 101, 150),  # Unhealthy for Sensitive Groups
        (361, 649, 151, 200),  # Unhealthy
        (650, 1249, 201, 300),  # Very Unhealthy
        (1250, 2049, 301, 500),  # Hazardous
    ],
    "o3": [
        (0, 54, 0, 50),  # Good (ppb, 8-hour)
        (55, 70, 51, 100),  # Moderate
        (71, 85, 101, 150),  # Unhealthy for Sensitive Groups
        (86, 105, 151, 200),  # Unhealthy
        (106, 200, 201, 300),  # Very Unhealthy
        (201, 504, 301, 500),  # Hazardous
    ],
}

# European Air Quality Index Breakpoints (EAQI)
# Format: {pollutant: [(C_lo, C_hi, EAQI_level), ...]}
EAQI_BREAKPOINTS = {
    "pm25": [
        (0, 10, 1),  # Very Good
        (10, 20, 2),  # Good
        (20, 25, 3),  # Medium
        (25, 50, 4),  # Poor
        (50, 75, 5),  # Very Poor
        (75, 800, 6),  # Extremely Poor
    ],
    "pm10": [
        (0, 20, 1),  # Very Good
        (20, 40, 2),  # Good
        (40, 50, 3),  # Medium
        (50, 100, 4),  # Poor
        (100, 150, 5),  # Very Poor
        (150, 1200, 6),  # Extremely Poor
    ],
    "no2": [
        # NO2 in ug/m3 for EAQI (different from EPA ppb)
        (0, 40, 1),  # Very Good
        (40, 90, 2),  # Good
        (90, 120, 3),  # Medium
        (120, 230, 4),  # Poor
        (230, 340, 5),  # Very Poor
        (340, 1000, 6),  # Extremely Poor
    ],
    "o3": [
        # O3 in ug/m3 for EAQI (8-hour average)
        (0, 50, 1),  # Very Good
        (50, 100, 2),  # Good
        (100, 130, 3),  # Medium
        (130, 240, 4),  # Poor
        (240, 380, 5),  # Very Poor
        (380, 800, 6),  # Extremely Poor
    ],
}

# EPA AQI Category Definitions
EPA_CATEGORIES = {
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

# European AQI Category Definitions
EAQI_CATEGORIES = {
    1: {
        "level": "Very Good",
        "color": "Dark Blue",
        "health_message": "Excellent air quality",
    },
    2: {
        "level": "Good",
        "color": "Blue",
        "health_message": "Good air quality",
    },
    3: {
        "level": "Medium",
        "color": "Green",
        "health_message": "Medium air quality",
    },
    4: {
        "level": "Poor",
        "color": "Yellow",
        "health_message": "Poor air quality",
    },
    5: {
        "level": "Very Poor",
        "color": "Orange",
        "health_message": "Very poor air quality",
    },
    6: {
        "level": "Extremely Poor",
        "color": "Red",
        "health_message": "Extremely poor air quality",
    },
}


def calculate_individual_aqi(
    concentration: float, pollutant: str, standard: AQIStandard = "EPA"
) -> float:
    """
    Calculate AQI for a single pollutant concentration using specified standard.

    Args:
        concentration: Pollutant concentration (units depend on standard)
        pollutant: Pollutant name ('pm25', 'pm10', 'no2', 'o3')
        standard: 'EPA' for US AQI or 'EAQI' for European index

    Returns:
        AQI value (EPA: 0-500+, EAQI: 1-6)
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan

    # Select appropriate breakpoints
    if standard == "EPA":
        if pollutant not in EPA_BREAKPOINTS:
            raise ValueError(f"Unsupported pollutant for EPA: {pollutant}")
        breakpoints = EPA_BREAKPOINTS[pollutant]

        # EPA uses linear interpolation
        for c_lo, c_hi, i_lo, i_hi in breakpoints:
            if c_lo <= concentration <= c_hi:
                aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
                return round(aqi)

        # Handle concentrations above highest breakpoint (extrapolate)
        c_lo, c_hi, i_lo, i_hi = breakpoints[-1]
        if concentration > c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
            return round(aqi)

    elif standard == "EAQI":
        if pollutant not in EAQI_BREAKPOINTS:
            raise ValueError(f"Unsupported pollutant for EAQI: {pollutant}")
        breakpoints = EAQI_BREAKPOINTS[pollutant]

        # EAQI uses step function (no interpolation)
        for c_lo, c_hi, eaqi_level in breakpoints:
            if c_lo <= concentration <= c_hi:
                return float(eaqi_level)

        # If above highest breakpoint, return highest level
        return 6.0

    else:
        raise ValueError(f"Unsupported standard: {standard}")

    return np.nan


def calculate_composite_aqi(
    concentrations: Dict[str, float], standard: AQIStandard = "EPA"
) -> Tuple[float, str]:
    """
    Calculate composite AQI from multiple pollutant concentrations.

    Args:
        concentrations: Dictionary of {pollutant: concentration}
        standard: 'EPA' for US AQI or 'EAQI' for European index

    Returns:
        Tuple of (composite_aqi, dominant_pollutant)
    """
    individual_aqis = {}

    for pollutant, concentration in concentrations.items():
        if (standard == "EPA" and pollutant in EPA_BREAKPOINTS) or (
            standard == "EAQI" and pollutant in EAQI_BREAKPOINTS
        ):
            aqi = calculate_individual_aqi(concentration, pollutant, standard)
            if not pd.isna(aqi):
                individual_aqis[pollutant] = aqi

    if not individual_aqis:
        return np.nan, "unknown"

    if standard == "EPA":
        # EPA: Composite AQI is the maximum of individual AQIs
        max_aqi = max(individual_aqis.values())
        dominant_pollutant = max(individual_aqis, key=individual_aqis.get)
    elif standard == "EAQI":
        # EAQI: Also uses maximum (worst) of individual indices
        max_aqi = max(individual_aqis.values())
        dominant_pollutant = max(individual_aqis, key=individual_aqis.get)

    return max_aqi, dominant_pollutant


def get_aqi_category(aqi_value: float, standard: AQIStandard = "EPA") -> Dict[str, str]:
    """
    Get AQI category information for a given AQI value.

    Args:
        aqi_value: AQI value (EPA: 0-500+, EAQI: 1-6)
        standard: 'EPA' for US AQI or 'EAQI' for European index

    Returns:
        Dictionary with level, color, and health_message
    """
    if pd.isna(aqi_value):
        return {
            "level": "Unknown",
            "color": "Gray",
            "health_message": "Data unavailable",
        }

    if standard == "EPA":
        for (min_aqi, max_aqi), category_info in EPA_CATEGORIES.items():
            if min_aqi <= aqi_value <= max_aqi:
                return category_info

        # Handle values above 500 (Hazardous+)
        if aqi_value > 500:
            return {
                "level": "Hazardous",
                "color": "Maroon",
                "health_message": "Health warning: emergency conditions",
            }

    elif standard == "EAQI":
        # EAQI uses integer levels 1-6
        eaqi_level = int(round(aqi_value))
        if 1 <= eaqi_level <= 6:
            return EAQI_CATEGORIES[eaqi_level]
        elif eaqi_level > 6:
            return EAQI_CATEGORIES[6]  # Extremely Poor

    return {"level": "Unknown", "color": "Gray", "health_message": "Invalid AQI value"}


def is_health_warning_required(
    aqi_value: float, sensitive_groups: bool = True, standard: AQIStandard = "EPA"
) -> bool:
    """
    Determine if health warning is required based on AQI value.

    Args:
        aqi_value: AQI value
        sensitive_groups: If True, warn for sensitive groups
        standard: 'EPA' for US AQI or 'EAQI' for European index

    Returns:
        Boolean indicating if warning is required
    """
    if pd.isna(aqi_value):
        return False

    if standard == "EPA":
        # EPA thresholds
        threshold = 101 if sensitive_groups else 151  # Orange vs Red
        return aqi_value >= threshold
    elif standard == "EAQI":
        # EAQI thresholds (European levels)
        threshold = 4 if sensitive_groups else 5  # Poor vs Very Poor
        return aqi_value >= threshold

    return False


def convert_units_for_eaqi(
    concentration: float, pollutant: str, from_unit: str
) -> float:
    """
    Convert pollutant concentrations to appropriate units for EAQI calculation.

    Args:
        concentration: Original concentration value
        pollutant: Pollutant name
        from_unit: Current unit ('ppb' or 'ug/m3')

    Returns:
        Concentration in ug/m3 (EAQI standard unit)
    """
    if pd.isna(concentration):
        return np.nan

    if from_unit == "ug/m3":
        return concentration
    elif from_unit == "ppb" and pollutant == "no2":
        # Convert NO2 from ppb to ug/m3 (at 20°C, 1 atm)
        # 1 ppb NO2 ≈ 1.88 ug/m3
        return concentration * 1.88
    elif from_unit == "ppb" and pollutant == "o3":
        # Convert O3 from ppb to ug/m3 (at 20°C, 1 atm)
        # 1 ppb O3 ≈ 1.96 ug/m3
        return concentration * 1.96
    else:
        # PM2.5 and PM10 are already in ug/m3
        return concentration


def process_dataset_with_dual_aqi(
    df: pd.DataFrame, standards: List[AQIStandard] = ["EPA", "EAQI"]
) -> pd.DataFrame:
    """
    Process entire dataset to add AQI calculations for both standards.

    Args:
        df: DataFrame with pollutant concentration columns
        standards: List of standards to calculate ('EPA', 'EAQI', or both)

    Returns:
        DataFrame with added AQI columns for each standard
    """
    log.info(f"Processing dataset with dual AQI calculations: {standards}")

    df_aqi = df.copy()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    for standard in standards:
        log.info(f"Calculating {standard} AQI values...")

        # Calculate individual AQIs for each standard
        for pollutant in pollutants:
            actual_col = f"actual_{pollutant}"
            if actual_col in df_aqi.columns:
                aqi_col = f"aqi_{pollutant}_{standard.lower()}"

                if standard == "EAQI" and pollutant in ["no2", "o3"]:
                    # Convert units for EAQI if needed (assume input is in ppb for NO2/O3)
                    df_aqi[f"{actual_col}_ug_m3"] = df_aqi[actual_col].apply(
                        lambda x: convert_units_for_eaqi(x, pollutant, "ppb")
                    )
                    df_aqi[aqi_col] = df_aqi[f"{actual_col}_ug_m3"].apply(
                        lambda x: calculate_individual_aqi(x, pollutant, standard)
                    )
                else:
                    df_aqi[aqi_col] = df_aqi[actual_col].apply(
                        lambda x: calculate_individual_aqi(x, pollutant, standard)
                    )

        # Calculate composite AQI for each standard
        log.info(f"Calculating composite {standard} AQI...")

        def calculate_row_aqi(row, std):
            concentrations = {}
            for pollutant in pollutants:
                actual_col = f"actual_{pollutant}"
                if actual_col in row and not pd.isna(row[actual_col]):
                    if std == "EAQI" and pollutant in ["no2", "o3"]:
                        # Use converted units for EAQI
                        conc = convert_units_for_eaqi(row[actual_col], pollutant, "ppb")
                    else:
                        conc = row[actual_col]
                    concentrations[pollutant] = conc

            if concentrations:
                composite_aqi, dominant_pollutant = calculate_composite_aqi(
                    concentrations, std
                )
                return pd.Series(
                    {
                        f"aqi_composite_{std.lower()}": composite_aqi,
                        f"aqi_dominant_pollutant_{std.lower()}": dominant_pollutant,
                    }
                )
            else:
                return pd.Series(
                    {
                        f"aqi_composite_{std.lower()}": np.nan,
                        f"aqi_dominant_pollutant_{std.lower()}": "unknown",
                    }
                )

        aqi_results = df_aqi.apply(lambda row: calculate_row_aqi(row, standard), axis=1)
        df_aqi = pd.concat([df_aqi, aqi_results], axis=1)

        # Add categorical information for each standard
        log.info(f"Adding {standard} categorical information...")

        composite_col = f"aqi_composite_{standard.lower()}"

        def get_category_info(aqi_value, std):
            category = get_aqi_category(aqi_value, std)
            return pd.Series(
                {
                    f"aqi_level_{std.lower()}": category["level"],
                    f"aqi_color_{std.lower()}": category["color"],
                    f"aqi_health_message_{std.lower()}": category["health_message"],
                }
            )

        category_results = df_aqi[composite_col].apply(
            lambda x: get_category_info(x, standard)
        )
        df_aqi = pd.concat([df_aqi, category_results], axis=1)

        # Add health warning flags for each standard
        df_aqi[f"health_warning_sensitive_{standard.lower()}"] = df_aqi[
            composite_col
        ].apply(
            lambda x: is_health_warning_required(
                x, sensitive_groups=True, standard=standard
            )
        )
        df_aqi[f"health_warning_general_{standard.lower()}"] = df_aqi[
            composite_col
        ].apply(
            lambda x: is_health_warning_required(
                x, sensitive_groups=False, standard=standard
            )
        )

    new_cols = [
        col for col in df_aqi.columns if any(std.lower() in col for std in standards)
    ]
    log.info(f"Dual AQI processing complete. Added {len(new_cols)} new columns")

    return df_aqi


def compare_aqi_standards(df_aqi: pd.DataFrame) -> None:
    """
    Generate comparison report between EPA and EAQI standards.

    Args:
        df_aqi: DataFrame with both EPA and EAQI calculations
    """
    print("\n" + "=" * 80)
    print("DUAL-STANDARD AQI COMPARISON REPORT")
    print("EPA (US) vs EAQI (European) Air Quality Index")
    print("=" * 80)

    if (
        "aqi_composite_epa" not in df_aqi.columns
        or "aqi_composite_eaqi" not in df_aqi.columns
    ):
        print("ERROR: Missing AQI data for comparison")
        return

    # Overall statistics comparison
    print(f"\nOVERALL AQI STATISTICS COMPARISON:")

    epa_stats = df_aqi["aqi_composite_epa"].describe()
    eaqi_stats = df_aqi["aqi_composite_eaqi"].describe()

    print(f"\nEPA AQI (0-500 scale):")
    print(f"  Count: {int(epa_stats['count']):,} observations")
    print(f"  Mean: {epa_stats['mean']:.1f}")
    print(f"  Median: {epa_stats['50%']:.1f}")
    print(f"  Range: {epa_stats['min']:.1f} - {epa_stats['max']:.1f}")

    print(f"\nEAQI (1-6 scale):")
    print(f"  Count: {int(eaqi_stats['count']):,} observations")
    print(f"  Mean: {eaqi_stats['mean']:.2f}")
    print(f"  Median: {eaqi_stats['50%']:.2f}")
    print(f"  Range: {eaqi_stats['min']:.0f} - {eaqi_stats['max']:.0f}")

    # Category distribution comparison
    print(f"\nCATEGORY DISTRIBUTION COMPARISON:")

    if "aqi_level_epa" in df_aqi.columns:
        print(f"\nEPA Categories:")
        epa_counts = df_aqi["aqi_level_epa"].value_counts()
        total_epa = epa_counts.sum()
        for category, count in epa_counts.items():
            pct = (count / total_epa) * 100
            print(f"  {category}: {count:,} ({pct:.1f}%)")

    if "aqi_level_eaqi" in df_aqi.columns:
        print(f"\nEAQI Categories:")
        eaqi_counts = df_aqi["aqi_level_eaqi"].value_counts()
        total_eaqi = eaqi_counts.sum()
        for category, count in eaqi_counts.items():
            pct = (count / total_eaqi) * 100
            print(f"  {category}: {count:,} ({pct:.1f}%)")

    # Health warning comparison
    print(f"\nHEALTH WARNING COMPARISON:")

    # EPA warnings
    if "health_warning_sensitive_epa" in df_aqi.columns:
        epa_sensitive = df_aqi["health_warning_sensitive_epa"].sum()
        epa_general = df_aqi["health_warning_general_epa"].sum()
        print(f"\nEPA Warnings:")
        print(
            f"  Sensitive Groups (AQI >= 101): {epa_sensitive:,} days ({epa_sensitive/len(df_aqi)*100:.1f}%)"
        )
        print(
            f"  General Population (AQI >= 151): {epa_general:,} days ({epa_general/len(df_aqi)*100:.1f}%)"
        )

    # EAQI warnings
    if "health_warning_sensitive_eaqi" in df_aqi.columns:
        eaqi_sensitive = df_aqi["health_warning_sensitive_eaqi"].sum()
        eaqi_general = df_aqi["health_warning_general_eaqi"].sum()
        print(f"\nEAQI Warnings:")
        print(
            f"  Sensitive Groups (Level >= 4): {eaqi_sensitive:,} days ({eaqi_sensitive/len(df_aqi)*100:.1f}%)"
        )
        print(
            f"  General Population (Level >= 5): {eaqi_general:,} days ({eaqi_general/len(df_aqi)*100:.1f}%)"
        )

    # Correlation analysis
    if "aqi_composite_epa" in df_aqi.columns and "aqi_composite_eaqi" in df_aqi.columns:
        correlation = (
            df_aqi[["aqi_composite_epa", "aqi_composite_eaqi"]].corr().iloc[0, 1]
        )
        print(f"\nCORRELATION BETWEEN STANDARDS:")
        print(f"  EPA vs EAQI correlation: {correlation:.3f}")

    print("\n" + "=" * 80)
    print("STANDARDS COMPARISON COMPLETE")
    print("=" * 80)


def main():
    """Test dual-standard AQI calculation functions with sample data."""

    print("Testing dual-standard AQI calculations:")

    # Test individual AQI calculations for both standards
    test_concentrations = {
        "pm25": [5.0, 15.0, 30.0, 60.0],  # ug/m3
        "pm10": [25.0, 75.0, 120.0, 200.0],  # ug/m3
        "no2": [30.0, 80.0, 150.0, 300.0],  # ppb for EPA, will convert for EAQI
        "o3": [40.0, 60.0, 90.0, 150.0],  # ppb for EPA, will convert for EAQI
    }

    for pollutant, concentrations in test_concentrations.items():
        print(f"\n{pollutant.upper()} AQI calculations:")
        print(
            f"{'Conc':>8} {'EPA AQI':>8} {'EPA Cat':>25} {'EAQI':>6} {'EAQI Cat':>15}"
        )
        print("-" * 70)

        for conc in concentrations:
            # EPA calculation
            epa_aqi = calculate_individual_aqi(conc, pollutant, "EPA")
            epa_category = get_aqi_category(epa_aqi, "EPA")

            # EAQI calculation (convert units if needed)
            if pollutant in ["no2", "o3"]:
                eaqi_conc = convert_units_for_eaqi(conc, pollutant, "ppb")
            else:
                eaqi_conc = conc
            eaqi_val = calculate_individual_aqi(eaqi_conc, pollutant, "EAQI")
            eaqi_category = get_aqi_category(eaqi_val, "EAQI")

            print(
                f"{conc:8.1f} {epa_aqi:8.0f} {epa_category['level']:>25} {eaqi_val:6.0f} {eaqi_category['level']:>15}"
            )

    # Test composite AQI for both standards
    print(f"\nTesting composite AQI calculation:")
    test_composite = {
        "pm25": 25.0,  # ug/m3
        "pm10": 100.0,  # ug/m3
        "no2": 150.0,  # ppb
        "o3": 65.0,  # ppb
    }

    print(f"Test concentrations: {test_composite}")

    # EPA composite
    epa_composite_aqi, epa_dominant = calculate_composite_aqi(test_composite, "EPA")
    epa_composite_category = get_aqi_category(epa_composite_aqi, "EPA")

    # EAQI composite (convert units)
    eaqi_concentrations = {}
    for pollutant, conc in test_composite.items():
        if pollutant in ["no2", "o3"]:
            eaqi_concentrations[pollutant] = convert_units_for_eaqi(
                conc, pollutant, "ppb"
            )
        else:
            eaqi_concentrations[pollutant] = conc

    eaqi_composite_val, eaqi_dominant = calculate_composite_aqi(
        eaqi_concentrations, "EAQI"
    )
    eaqi_composite_category = get_aqi_category(eaqi_composite_val, "EAQI")

    print(f"\nEPA Results:")
    print(f"  Composite AQI: {epa_composite_aqi:.0f} (dominant: {epa_dominant})")
    print(
        f"  Category: {epa_composite_category['level']} - {epa_composite_category['health_message']}"
    )

    print(f"\nEAQI Results:")
    print(f"  Composite EAQI: {eaqi_composite_val:.0f} (dominant: {eaqi_dominant})")
    print(
        f"  Category: {eaqi_composite_category['level']} - {eaqi_composite_category['health_message']}"
    )

    # Test health warnings
    print(f"\nHealth warning tests:")
    test_values = [
        (45, 2),  # Good/Good
        (85, 3),  # Moderate/Medium
        (115, 4),  # Unhealthy Sensitive/Poor
        (165, 5),  # Unhealthy/Very Poor
        (225, 6),  # Very Unhealthy/Extremely Poor
    ]

    print(
        f"{'Value':>6} {'EPA Sensitive':>13} {'EPA General':>12} {'EAQI Sensitive':>15} {'EAQI General':>13}"
    )
    print("-" * 70)

    for epa_val, eaqi_val in test_values:
        epa_sens = is_health_warning_required(epa_val, True, "EPA")
        epa_gen = is_health_warning_required(epa_val, False, "EPA")
        eaqi_sens = is_health_warning_required(eaqi_val, True, "EAQI")
        eaqi_gen = is_health_warning_required(eaqi_val, False, "EAQI")

        print(
            f"{epa_val:>6} {str(epa_sens):>13} {str(epa_gen):>12} {str(eaqi_sens):>15} {str(eaqi_gen):>13}"
        )


if __name__ == "__main__":
    main()
