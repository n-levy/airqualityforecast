#!/usr/bin/env python3
"""
Multi-Standard Air Quality Index (AQI) Calculation Module

Supports AQI calculation for 10 global cities using their local standards:
- Delhi: Indian National AQI
- Beijing: Chinese AQI
- Bangkok: Thai AQI (EPA-based)
- Mexico City: Mexican IMECA
- Santiago: Chilean ICA
- Krakow: European EAQI
- Los Angeles: US EPA AQI
- Milan: European EAQI
- Jakarta: Indonesian ISPU
- Lahore: Pakistani AQI (EPA-based)
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
AQIStandardType = Literal[
    "EPA",
    "EAQI",
    "Indian",
    "Chinese",
    "Thai",
    "Mexican",
    "Chilean",
    "Indonesian",
    "Pakistani",
]

# City to AQI Standard Mapping
CITY_AQI_STANDARDS = {
    "delhi": "Indian",
    "beijing": "Chinese",
    "bangkok": "Thai",
    "mexico_city": "Mexican",
    "santiago": "Chilean",
    "krakow": "EAQI",
    "los_angeles": "EPA",
    "milan": "EAQI",
    "jakarta": "Indonesian",
    "lahore": "Pakistani",
}

# EPA Air Quality Index Breakpoints (US Standard)
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
        (0, 40, 1),  # Very Good (ug/m3)
        (40, 90, 2),  # Good
        (90, 120, 3),  # Medium
        (120, 230, 4),  # Poor
        (230, 340, 5),  # Very Poor
        (340, 1000, 6),  # Extremely Poor
    ],
    "o3": [
        (0, 50, 1),  # Very Good (ug/m3)
        (50, 100, 2),  # Good
        (100, 130, 3),  # Medium
        (130, 240, 4),  # Poor
        (240, 380, 5),  # Very Poor
        (380, 800, 6),  # Extremely Poor
    ],
}

# Indian National AQI Breakpoints
INDIAN_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),  # Good
        (31, 60, 51, 100),  # Satisfactory
        (61, 90, 101, 200),  # Moderate
        (91, 120, 201, 300),  # Poor
        (121, 250, 301, 400),  # Very Poor
        (251, 500, 401, 500),  # Severe
    ],
    "pm10": [
        (0, 50, 0, 50),  # Good
        (51, 100, 51, 100),  # Satisfactory
        (101, 250, 101, 200),  # Moderate
        (251, 350, 201, 300),  # Poor
        (351, 430, 301, 400),  # Very Poor
        (431, 800, 401, 500),  # Severe
    ],
    "no2": [
        (0, 40, 0, 50),  # Good (ug/m3)
        (41, 80, 51, 100),  # Satisfactory
        (81, 180, 101, 200),  # Moderate
        (181, 280, 201, 300),  # Poor
        (281, 400, 301, 400),  # Very Poor
        (401, 800, 401, 500),  # Severe
    ],
    "o3": [
        (0, 50, 0, 50),  # Good (ug/m3)
        (51, 100, 51, 100),  # Satisfactory
        (101, 168, 101, 200),  # Moderate
        (169, 208, 201, 300),  # Poor
        (209, 748, 301, 400),  # Very Poor
        (749, 1000, 401, 500),  # Severe
    ],
    "so2": [
        (0, 40, 0, 50),  # Good (ug/m3)
        (41, 80, 51, 100),  # Satisfactory
        (81, 380, 101, 200),  # Moderate
        (381, 800, 201, 300),  # Poor
        (801, 1600, 301, 400),  # Very Poor
        (1601, 2620, 401, 500),  # Severe
    ],
}

# Chinese AQI Breakpoints
CHINESE_BREAKPOINTS = {
    "pm25": [
        (0, 35, 0, 50),  # Excellent
        (36, 75, 51, 100),  # Good
        (76, 115, 101, 150),  # Lightly Polluted
        (116, 150, 151, 200),  # Moderately Polluted
        (151, 250, 201, 300),  # Heavily Polluted
        (251, 500, 301, 500),  # Severely Polluted
    ],
    "pm10": [
        (0, 50, 0, 50),  # Excellent
        (51, 150, 51, 100),  # Good
        (151, 250, 101, 150),  # Lightly Polluted
        (251, 350, 151, 200),  # Moderately Polluted
        (351, 420, 201, 300),  # Heavily Polluted
        (421, 600, 301, 500),  # Severely Polluted
    ],
    "no2": [
        (0, 40, 0, 50),  # Excellent (ug/m3)
        (41, 80, 51, 100),  # Good
        (81, 180, 101, 150),  # Lightly Polluted
        (181, 280, 151, 200),  # Moderately Polluted
        (281, 565, 201, 300),  # Heavily Polluted
        (566, 940, 301, 500),  # Severely Polluted
    ],
    "o3": [
        (0, 100, 0, 50),  # Excellent (ug/m3)
        (101, 160, 51, 100),  # Good
        (161, 215, 101, 150),  # Lightly Polluted
        (216, 265, 151, 200),  # Moderately Polluted
        (266, 800, 201, 300),  # Heavily Polluted
        (801, 1000, 301, 500),  # Severely Polluted
    ],
}

# Thai AQI (EPA-based but slightly different breakpoints)
THAI_BREAKPOINTS = EPA_BREAKPOINTS  # Thailand uses EPA-similar system

# Mexican IMECA (similar to EPA)
MEXICAN_BREAKPOINTS = EPA_BREAKPOINTS  # Mexico uses EPA-similar system

# Chilean ICA (similar to EPA)
CHILEAN_BREAKPOINTS = EPA_BREAKPOINTS  # Chile uses EPA-similar system

# Indonesian ISPU
INDONESIAN_BREAKPOINTS = {
    "pm25": [
        (0, 15, 0, 50),  # Good
        (16, 55, 51, 100),  # Moderate
        (56, 150, 101, 200),  # Unhealthy
        (151, 250, 201, 300),  # Very Unhealthy
        (251, 500, 301, 500),  # Hazardous
    ],
    "pm10": [
        (0, 50, 0, 50),  # Good
        (51, 150, 51, 100),  # Moderate
        (151, 350, 101, 200),  # Unhealthy
        (351, 420, 201, 300),  # Very Unhealthy
        (421, 600, 301, 500),  # Hazardous
    ],
    "no2": [
        (0, 80, 0, 50),  # Good (ug/m3)
        (81, 200, 51, 100),  # Moderate
        (201, 1130, 101, 200),  # Unhealthy
        (1131, 2260, 201, 300),  # Very Unhealthy
        (2261, 3000, 301, 500),  # Hazardous
    ],
    "o3": [
        (0, 120, 0, 50),  # Good (ug/m3)
        (121, 235, 51, 100),  # Moderate
        (236, 400, 101, 200),  # Unhealthy
        (401, 800, 201, 300),  # Very Unhealthy
        (801, 1000, 301, 500),  # Hazardous
    ],
}

# Pakistani AQI (EPA-based)
PAKISTANI_BREAKPOINTS = EPA_BREAKPOINTS  # Pakistan uses EPA-similar system

# Consolidated breakpoints dictionary
AQI_BREAKPOINTS = {
    "EPA": EPA_BREAKPOINTS,
    "EAQI": EAQI_BREAKPOINTS,
    "Indian": INDIAN_BREAKPOINTS,
    "Chinese": CHINESE_BREAKPOINTS,
    "Thai": THAI_BREAKPOINTS,
    "Mexican": MEXICAN_BREAKPOINTS,
    "Chilean": CHILEAN_BREAKPOINTS,
    "Indonesian": INDONESIAN_BREAKPOINTS,
    "Pakistani": PAKISTANI_BREAKPOINTS,
}

# Health warning thresholds for each standard
HEALTH_WARNING_THRESHOLDS = {
    "EPA": {"sensitive": 101, "general": 151},
    "EAQI": {"sensitive": 4, "general": 5},
    "Indian": {"sensitive": 101, "general": 201},
    "Chinese": {"sensitive": 101, "general": 151},
    "Thai": {"sensitive": 101, "general": 151},
    "Mexican": {"sensitive": 101, "general": 151},
    "Chilean": {"sensitive": 101, "general": 151},
    "Indonesian": {"sensitive": 101, "general": 201},
    "Pakistani": {"sensitive": 101, "general": 151},
}

# AQI Categories for each standard
AQI_CATEGORIES = {
    "EPA": {
        (0, 50): {"level": "Good", "color": "Green"},
        (51, 100): {"level": "Moderate", "color": "Yellow"},
        (101, 150): {"level": "Unhealthy for Sensitive Groups", "color": "Orange"},
        (151, 200): {"level": "Unhealthy", "color": "Red"},
        (201, 300): {"level": "Very Unhealthy", "color": "Purple"},
        (301, 500): {"level": "Hazardous", "color": "Maroon"},
    },
    "EAQI": {
        1: {"level": "Very Good", "color": "Dark Blue"},
        2: {"level": "Good", "color": "Blue"},
        3: {"level": "Medium", "color": "Green"},
        4: {"level": "Poor", "color": "Yellow"},
        5: {"level": "Very Poor", "color": "Orange"},
        6: {"level": "Extremely Poor", "color": "Red"},
    },
    "Indian": {
        (0, 50): {"level": "Good", "color": "Green"},
        (51, 100): {"level": "Satisfactory", "color": "Light Green"},
        (101, 200): {"level": "Moderate", "color": "Yellow"},
        (201, 300): {"level": "Poor", "color": "Orange"},
        (301, 400): {"level": "Very Poor", "color": "Red"},
        (401, 500): {"level": "Severe", "color": "Maroon"},
    },
    "Chinese": {
        (0, 50): {"level": "Excellent", "color": "Green"},
        (51, 100): {"level": "Good", "color": "Yellow"},
        (101, 150): {"level": "Lightly Polluted", "color": "Orange"},
        (151, 200): {"level": "Moderately Polluted", "color": "Red"},
        (201, 300): {"level": "Heavily Polluted", "color": "Purple"},
        (301, 500): {"level": "Severely Polluted", "color": "Maroon"},
    },
}

# For other standards that use EPA-like categories
for standard in ["Thai", "Mexican", "Chilean", "Indonesian", "Pakistani"]:
    AQI_CATEGORIES[standard] = AQI_CATEGORIES["EPA"]


def calculate_individual_aqi(
    concentration: float, pollutant: str, standard: AQIStandardType
) -> float:
    """Calculate AQI for a single pollutant using specified standard."""

    if pd.isna(concentration) or concentration < 0:
        return np.nan

    if standard not in AQI_BREAKPOINTS:
        raise ValueError(f"Unsupported AQI standard: {standard}")

    if pollutant not in AQI_BREAKPOINTS[standard]:
        raise ValueError(f"Unsupported pollutant {pollutant} for standard {standard}")

    breakpoints = AQI_BREAKPOINTS[standard][pollutant]

    if standard == "EAQI":
        # EAQI uses step function (no interpolation)
        for c_lo, c_hi, eaqi_level in breakpoints:
            if c_lo <= concentration <= c_hi:
                return float(eaqi_level)
        return 6.0  # Above highest breakpoint
    else:
        # All other standards use linear interpolation
        for c_lo, c_hi, i_lo, i_hi in breakpoints:
            if c_lo <= concentration <= c_hi:
                aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
                return round(aqi)

        # Handle concentrations above highest breakpoint
        c_lo, c_hi, i_lo, i_hi = breakpoints[-1]
        if concentration > c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
            return round(aqi)

    return np.nan


def calculate_composite_aqi(
    concentrations: Dict[str, float], standard: AQIStandardType
) -> Tuple[float, str]:
    """Calculate composite AQI from multiple pollutant concentrations."""

    individual_aqis = {}

    for pollutant, concentration in concentrations.items():
        if pollutant in AQI_BREAKPOINTS[standard]:
            aqi = calculate_individual_aqi(concentration, pollutant, standard)
            if not pd.isna(aqi):
                individual_aqis[pollutant] = aqi

    if not individual_aqis:
        return np.nan, "unknown"

    # Composite AQI is the maximum of individual AQIs for all standards
    max_aqi = max(individual_aqis.values())
    dominant_pollutant = max(individual_aqis, key=individual_aqis.get)

    return max_aqi, dominant_pollutant


def get_aqi_category(aqi_value: float, standard: AQIStandardType) -> Dict[str, str]:
    """Get AQI category information for a given AQI value."""

    if pd.isna(aqi_value):
        return {"level": "Unknown", "color": "Gray"}

    if standard not in AQI_CATEGORIES:
        return {"level": "Unknown", "color": "Gray"}

    categories = AQI_CATEGORIES[standard]

    if standard == "EAQI":
        # EAQI uses integer levels 1-6
        eaqi_level = int(round(aqi_value))
        if 1 <= eaqi_level <= 6:
            return categories[eaqi_level]
        elif eaqi_level > 6:
            return categories[6]
    else:
        # Range-based categories for other standards
        for (min_aqi, max_aqi), category_info in categories.items():
            if min_aqi <= aqi_value <= max_aqi:
                return category_info

        # Handle values above highest range
        if aqi_value > 500:
            return list(categories.values())[-1]  # Return highest category

    return {"level": "Unknown", "color": "Gray"}


def is_health_warning_required(
    aqi_value: float, sensitive_groups: bool = True, standard: AQIStandardType = "EPA"
) -> bool:
    """Determine if health warning is required based on AQI value."""

    if pd.isna(aqi_value):
        return False

    if standard not in HEALTH_WARNING_THRESHOLDS:
        return False

    thresholds = HEALTH_WARNING_THRESHOLDS[standard]
    threshold = thresholds["sensitive"] if sensitive_groups else thresholds["general"]

    return aqi_value >= threshold


def convert_pollutant_units(
    concentration: float, pollutant: str, from_unit: str, to_unit: str
) -> float:
    """Convert pollutant concentrations between units."""

    if pd.isna(concentration):
        return np.nan

    if from_unit == to_unit:
        return concentration

    # Conversion factors at standard conditions (20°C, 1 atm)
    conversion_factors = {
        "no2": {"ppb_to_ugm3": 1.88, "ugm3_to_ppb": 1 / 1.88},
        "o3": {"ppb_to_ugm3": 1.96, "ugm3_to_ppb": 1 / 1.96},
    }

    if pollutant in conversion_factors:
        if from_unit == "ppb" and to_unit == "ugm3":
            return concentration * conversion_factors[pollutant]["ppb_to_ugm3"]
        elif from_unit == "ugm3" and to_unit == "ppb":
            return concentration * conversion_factors[pollutant]["ugm3_to_ppb"]

    return concentration  # No conversion needed for PM2.5, PM10


def process_city_data_with_local_aqi(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Process city data with local AQI standard."""

    if city not in CITY_AQI_STANDARDS:
        raise ValueError(f"Unsupported city: {city}")

    standard = CITY_AQI_STANDARDS[city]
    log.info(f"Processing {city} data with {standard} AQI standard")

    df_aqi = df.copy()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    # Calculate individual AQIs
    for pollutant in pollutants:
        actual_col = f"actual_{pollutant}"
        if actual_col in df_aqi.columns:
            aqi_col = f"aqi_{pollutant}"

            # Convert units if needed for specific standards
            if standard in [
                "EAQI",
                "Indian",
                "Chinese",
                "Indonesian",
            ] and pollutant in ["no2", "o3"]:
                # Convert from ppb to ug/m3
                df_aqi[f"{actual_col}_converted"] = df_aqi[actual_col].apply(
                    lambda x: convert_pollutant_units(x, pollutant, "ppb", "ugm3")
                )
                df_aqi[aqi_col] = df_aqi[f"{actual_col}_converted"].apply(
                    lambda x: calculate_individual_aqi(x, pollutant, standard)
                )
            else:
                df_aqi[aqi_col] = df_aqi[actual_col].apply(
                    lambda x: calculate_individual_aqi(x, pollutant, standard)
                )

    # Calculate composite AQI
    def calculate_row_aqi(row):
        concentrations = {}
        for pollutant in pollutants:
            actual_col = f"actual_{pollutant}"
            if actual_col in row and not pd.isna(row[actual_col]):
                conc = row[actual_col]

                # Apply unit conversion if needed
                if standard in [
                    "EAQI",
                    "Indian",
                    "Chinese",
                    "Indonesian",
                ] and pollutant in ["no2", "o3"]:
                    conc = convert_pollutant_units(conc, pollutant, "ppb", "ugm3")

                concentrations[pollutant] = conc

        if concentrations:
            composite_aqi, dominant_pollutant = calculate_composite_aqi(
                concentrations, standard
            )
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
    def get_category_info(aqi_value):
        category = get_aqi_category(aqi_value, standard)
        return pd.Series(
            {
                "aqi_level": category["level"],
                "aqi_color": category["color"],
            }
        )

    category_results = df_aqi["aqi_composite"].apply(get_category_info)
    df_aqi = pd.concat([df_aqi, category_results], axis=1)

    # Add health warning flags
    df_aqi["health_warning_sensitive"] = df_aqi["aqi_composite"].apply(
        lambda x: is_health_warning_required(
            x, sensitive_groups=True, standard=standard
        )
    )
    df_aqi["health_warning_general"] = df_aqi["aqi_composite"].apply(
        lambda x: is_health_warning_required(
            x, sensitive_groups=False, standard=standard
        )
    )

    # Add city and standard info
    df_aqi["city"] = city
    df_aqi["aqi_standard"] = standard

    log.info(f"Processed {city} with {standard} standard: {len(df_aqi)} records")
    return df_aqi


def main():
    """Test multi-standard AQI calculation functions."""

    print("Testing multi-standard AQI calculations:")

    # Test concentrations for different pollutants
    test_concentrations = {
        "pm25": 45.0,  # High PM2.5
        "pm10": 120.0,  # High PM10
        "no2": 80.0,  # ppb
        "o3": 90.0,  # ppb
    }

    cities_to_test = ["delhi", "beijing", "krakow", "bangkok", "jakarta"]

    print(f"\nTest concentrations: {test_concentrations}")
    print(
        f"{'City':<15} {'Standard':<10} {'AQI':<6} {'Category':<25} {'Sensitive':<10} {'General':<8}"
    )
    print("-" * 85)

    for city in cities_to_test:
        standard = CITY_AQI_STANDARDS[city]

        # Convert units for standards that need it
        test_conc = test_concentrations.copy()
        if standard in ["EAQI", "Indian", "Chinese", "Indonesian"]:
            test_conc["no2"] = convert_pollutant_units(
                test_conc["no2"], "no2", "ppb", "ugm3"
            )
            test_conc["o3"] = convert_pollutant_units(
                test_conc["o3"], "o3", "ppb", "ugm3"
            )

        composite_aqi, dominant = calculate_composite_aqi(test_conc, standard)
        category = get_aqi_category(composite_aqi, standard)

        sensitive_warning = is_health_warning_required(composite_aqi, True, standard)
        general_warning = is_health_warning_required(composite_aqi, False, standard)

        print(
            f"{city:<15} {standard:<10} {composite_aqi:<6.0f} {category['level']:<25} {str(sensitive_warning):<10} {str(general_warning):<8}"
        )

    print(f"\nDominant pollutant in all cases: {dominant}")


if __name__ == "__main__":
    main()
