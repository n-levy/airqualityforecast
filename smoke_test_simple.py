#!/usr/bin/env python3
"""Simple smoke test to check model features data without Unicode issues."""

import json
from pathlib import Path


def main():
    dataset_path = Path(
        "stage_5/enhanced_features/enhanced_worst_air_quality_with_features.json"
    )

    if not dataset_path.exists():
        print("ERROR: Enhanced features dataset not found")
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=== MODEL FEATURES SMOKE TEST RESULTS ===")
    print(f"Total cities: {len(dataset.get('city_results', {}))}")

    # Count features
    cities_with_fire = 0
    cities_with_holidays = 0
    total_fire_features = 0
    total_holiday_features = 0

    # Sample some fire feature values for analysis
    fire_weather_indices = []
    fire_distances = []
    holiday_names = set()
    countries = set()

    for city_name, city_data in dataset["city_results"].items():
        country = city_data.get("country", "Unknown")
        countries.add(country)

        city_has_fire = False
        city_has_holidays = False

        for source_name, source_data in city_data.get("data_sources", {}).items():
            if "data_sample" in source_data:
                for record in source_data["data_sample"]:
                    if "fire_features" in record:
                        city_has_fire = True
                        total_fire_features += 1
                        fire_features = record["fire_features"]

                        if "fire_weather_index" in fire_features:
                            fire_weather_indices.append(
                                fire_features["fire_weather_index"]
                            )
                        if "fire_distance_km" in fire_features:
                            fire_distances.append(fire_features["fire_distance_km"])

                    if "holiday_features" in record:
                        city_has_holidays = True
                        total_holiday_features += 1
                        holiday_features = record["holiday_features"]

                        if holiday_features.get("holiday_name"):
                            holiday_names.add(holiday_features["holiday_name"])

        if city_has_fire:
            cities_with_fire += 1
        if city_has_holidays:
            cities_with_holidays += 1

    print(f"Cities with fire features: {cities_with_fire}")
    print(f"Cities with holiday features: {cities_with_holidays}")
    print(f"Total fire feature records: {total_fire_features}")
    print(f"Total holiday feature records: {total_holiday_features}")
    print(f"Countries represented: {len(countries)}")
    print(f"Unique holiday names found: {len(holiday_names)}")

    # Data quality checks
    print("\n=== DATA QUALITY ANALYSIS ===")

    # Check fire weather index variability
    if fire_weather_indices:
        unique_fwi = len(set(fire_weather_indices))
        total_fwi = len(fire_weather_indices)
        variability_ratio = unique_fwi / total_fwi
        print(
            f"Fire weather index variability: {variability_ratio:.2%} ({unique_fwi}/{total_fwi})"
        )

        # Sample values
        print(f"Sample fire weather indices: {sorted(set(fire_weather_indices))[:10]}")

    # Check fire distance variability
    if fire_distances:
        unique_distances = len(set(fire_distances))
        total_distances = len(fire_distances)
        distance_variability = unique_distances / total_distances
        print(
            f"Fire distance variability: {distance_variability:.2%} ({unique_distances}/{total_distances})"
        )

        # Sample values
        print(f"Sample fire distances: {sorted(set(fire_distances))[:10]}")

    # Show sample holidays
    print(f"\nSample holidays found: {list(holiday_names)[:10]}")
    print(f"Sample countries: {list(countries)[:10]}")

    # Check for real vs synthetic patterns
    print("\n=== AUTHENTICITY ANALYSIS ===")

    # Check if we have country-specific holidays
    authentic_patterns = 0
    known_holidays = {
        "Independence Day",
        "Christmas Day",
        "New Year's Day",
        "Diwali",
        "Chinese New Year",
        "Eid al-Fitr",
        "Thanksgiving",
    }

    for holiday in holiday_names:
        if holiday in known_holidays:
            authentic_patterns += 1

    print(f"Authentic holiday patterns found: {authentic_patterns}")

    # Final assessment
    if total_fire_features > 0 and total_holiday_features > 0:
        if authentic_patterns > 0 and variability_ratio > 0.3:
            print("\nCONCLUSION: Data appears to be REAL")
            return 0
        else:
            print("\nCONCLUSION: Data may contain synthetic elements")
            return 1
    else:
        print("\nCONCLUSION: No features found - data collection failed")
        return 2


if __name__ == "__main__":
    exit(main())
