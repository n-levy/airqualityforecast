#!/usr/bin/env python3
"""
Verify Real Data Coverage

Verify actual real vs synthetic data coverage for all 100 cities.
"""

import json
from datetime import datetime

import pandas as pd


def verify_real_data_coverage():
    """Verify the actual real data coverage."""

    print("REAL DATA COVERAGE VERIFICATION")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 50)

    # Load collection results
    with open(
        "../final_dataset/complete_real_data_collection_20250911_192217.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # Load cities
    cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")

    # Find cities with real data
    real_cities = set()

    # Add NOAA cities (weather data)
    noaa_count = 0
    for city, info in data["noaa_data"].items():
        if info.get("api_status") == "SUCCESS":
            real_cities.add(city)
            noaa_count += 1

    # Add WAQI cities (air quality data)
    waqi_count = 0
    for city, info in data["waqi_data"].items():
        if info.get("api_status") == "SUCCESS":
            real_cities.add(city)
            waqi_count += 1

    all_cities = set(cities_df["City"].tolist())
    cities_without_real = all_cities - real_cities

    print("FINAL VERIFICATION RESULTS:")
    print("=" * 30)
    print(f"Total cities in dataset: {len(all_cities)}")
    print(f"Cities with REAL data: {len(real_cities)}")
    print(f"Cities requiring SYNTHETIC data: {len(cities_without_real)}")
    print(f"Real data coverage: {len(real_cities)/len(all_cities)*100:.1f}%")
    print()
    print("Data Source Breakdown:")
    print(f"  NOAA weather forecasts: {noaa_count} US cities")
    print(f"  WAQI air quality data: {waqi_count} global cities")
    print(f"  Total unique real cities: {len(real_cities)}")

    print(f"\nCONCLUSION:")
    print("=" * 15)
    if len(real_cities) == 100:
        print("✅ 100% REAL DATA COVERAGE ACHIEVED!")
        print("All 100 cities have verified real data sources.")
    else:
        print(f"⚠️  {len(real_cities)}/100 cities have real data ({len(real_cities)}%)")
        print(f"❌ {len(cities_without_real)} cities require synthetic data")
        print("\nCITIES REQUIRING SYNTHETIC DATA:")
        print("-" * 35)
        for i, city in enumerate(sorted(cities_without_real), 1):
            # Handle Unicode safely
            try:
                city_clean = city.encode("ascii", "ignore").decode("ascii")
                if city_clean != city:
                    print(f"  {i:2d}. {city_clean} (original: contains special chars)")
                else:
                    print(f"  {i:2d}. {city}")
            except:
                print(f"  {i:2d}. [Unicode city name]")

    # Save verification report
    verification_report = {
        "verification_time": datetime.now().isoformat(),
        "total_cities": len(all_cities),
        "cities_with_real_data": len(real_cities),
        "cities_requiring_synthetic": len(cities_without_real),
        "real_data_percentage": len(real_cities) / len(all_cities) * 100,
        "noaa_cities": noaa_count,
        "waqi_cities": waqi_count,
        "is_100_percent_real": len(real_cities) == 100,
        "cities_with_real_data_list": sorted(list(real_cities)),
        "cities_requiring_synthetic_list": sorted(list(cities_without_real)),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../final_dataset/real_data_verification_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nVerification report saved to: {output_file}")

    return verification_report


if __name__ == "__main__":
    results = verify_real_data_coverage()
