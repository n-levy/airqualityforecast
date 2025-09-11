#!/usr/bin/env python3
"""
Extract cities with verified real city-level API data availability
Only include cities that have actual city-specific WAQI stations (no Shanghai fallbacks)
"""
import json
import sys


def extract_verified_cities():
    """Extract cities with verified city-specific API data"""

    try:
        with open(
            "../final_dataset/complete_real_data_collection_20250911_192217.json",
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("ERROR: Could not find the complete real data collection file")
        return []

    print("=== CITIES WITH VERIFIED REAL WAQI API DATA ===")
    verified_cities = []
    shanghai_fallback = []
    city_specific = []

    waqi_data = data.get("waqi_data", {})
    for city, city_data in waqi_data.items():
        if city_data.get("api_status") == "SUCCESS":
            aqi = city_data.get("current_aqi", "N/A")
            station = city_data.get("station_info", {}).get("name", "Unknown")
            quality = city_data.get("quality_rating", "Unknown")

            verified_cities.append(city)

            # Check if it's Shanghai fallback (avoid Unicode issues)
            safe_station = station.encode("ascii", "replace").decode("ascii")[:30]
            if "shanghai" in station.lower() or "Shanghai" in station:
                shanghai_fallback.append(city)
                print(
                    f"{city}: AQI={aqi}, Station={safe_station}... [SHANGHAI FALLBACK]"
                )
            else:
                city_specific.append(city)
                print(f"{city}: AQI={aqi}, Station={safe_station}... [CITY-SPECIFIC]")

    print(f"\nSUMMARY:")
    print(f"Total cities with API SUCCESS: {len(verified_cities)}")
    print(f"Cities with city-specific stations: {len(city_specific)}")
    print(f"Cities with Shanghai fallback: {len(shanghai_fallback)}")

    print(f"\nCITY-SPECIFIC STATIONS ({len(city_specific)} cities):")
    for i, city in enumerate(city_specific, 1):
        print(f"{i:2d}. {city}")

    print(
        f"\nThese {len(city_specific)} cities have verified real city-level API data."
    )

    # Save the verified cities list
    verified_data = {
        "extraction_timestamp": "2025-09-11T21:53:00",
        "source_file": "complete_real_data_collection_20250911_192217.json",
        "total_tested": len(waqi_data),
        "api_success_count": len(verified_cities),
        "city_specific_count": len(city_specific),
        "shanghai_fallback_count": len(shanghai_fallback),
        "city_specific_stations": city_specific,
        "shanghai_fallback_cities": shanghai_fallback,
        "criteria": "Only cities with SUCCESS API status and city-specific stations (no Shanghai fallbacks)",
    }

    with open(
        "../final_dataset/verified_cities_with_real_api_data.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(verified_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: verified_cities_with_real_api_data.json")
    return city_specific


if __name__ == "__main__":
    verified_cities = extract_verified_cities()
