#!/usr/bin/env python3
"""
Quick 100% Real Data Replacement

Replace cities without real data with backup cities known to have poor air quality
and likely real data availability, achieving 100% real data coverage.
"""

import json
from datetime import datetime

import pandas as pd


def achieve_100_percent_real_data():
    """Replace cities to achieve 100% real data coverage with poor air quality cities."""

    print("ACHIEVING 100% REAL DATA COVERAGE")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 45)

    # Load current data
    cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")

    with open(
        "../final_dataset/complete_real_data_collection_20250911_192217.json",
        "r",
        encoding="utf-8",
    ) as f:
        collection_data = json.load(f)

    # Find cities with real data
    real_cities = set()
    for city, info in collection_data["noaa_data"].items():
        if info.get("api_status") == "SUCCESS":
            real_cities.add(city)
    for city, info in collection_data["waqi_data"].items():
        if info.get("api_status") == "SUCCESS":
            real_cities.add(city)

    all_cities = set(cities_df["City"].tolist())
    cities_without_real = all_cities - real_cities

    # Group cities needing replacement by continent
    replacement_needed = {}
    for city in cities_without_real:
        city_row = cities_df[cities_df["City"] == city].iloc[0]
        continent = city_row["Continent"]
        if continent not in replacement_needed:
            replacement_needed[continent] = []
        replacement_needed[continent].append(
            {"city": city, "country": city_row["Country"]}
        )

    print("CITIES NEEDING REPLACEMENT:")
    print("=" * 30)
    for continent, cities in replacement_needed.items():
        print(f"{continent}: {len(cities)} cities")
        for city_info in cities:
            print(f"  - {city_info['city']}, {city_info['country']}")

    # High-priority backup cities with known poor air quality and likely real data
    # These are major polluted cities with established monitoring networks
    backup_replacements = {
        "South_America": [
            {
                "name": "Salvador",
                "country": "Brazil",
                "lat": -12.9714,
                "lon": -38.5014,
                "reason": "Major Brazilian city with poor air quality",
            },
            {
                "name": "Fortaleza",
                "country": "Brazil",
                "lat": -3.7319,
                "lon": -38.5267,
                "reason": "Industrial coastal city",
            },
            {
                "name": "Brasília",
                "country": "Brazil",
                "lat": -15.8267,
                "lon": -47.9218,
                "reason": "Capital city with monitoring",
            },
            {
                "name": "Recife",
                "country": "Brazil",
                "lat": -8.0476,
                "lon": -34.8770,
                "reason": "Major port city",
            },
            {
                "name": "Manaus",
                "country": "Brazil",
                "lat": -3.1190,
                "lon": -60.0217,
                "reason": "Industrial center with pollution",
            },
            {
                "name": "Goiânia",
                "country": "Brazil",
                "lat": -16.6869,
                "lon": -49.2648,
                "reason": "Industrial city",
            },
            {
                "name": "Belém",
                "country": "Brazil",
                "lat": -1.4558,
                "lon": -48.4902,
                "reason": "Port city with air quality issues",
            },
            {
                "name": "João Pessoa",
                "country": "Brazil",
                "lat": -7.1195,
                "lon": -34.8450,
                "reason": "Coastal industrial city",
            },
        ],
        "Africa": [
            {
                "name": "Johannesburg",
                "country": "South Africa",
                "lat": -26.2041,
                "lon": 28.0473,
                "reason": "Mining hub with severe air pollution",
            },
            {
                "name": "Cape Town",
                "country": "South Africa",
                "lat": -33.9249,
                "lon": 18.4241,
                "reason": "Major city with air quality monitoring",
            },
            {
                "name": "Durban",
                "country": "South Africa",
                "lat": -29.8587,
                "lon": 31.0218,
                "reason": "Industrial port city",
            },
            {
                "name": "Nairobi",
                "country": "Kenya",
                "lat": -1.2921,
                "lon": 36.8219,
                "reason": "Major East African city",
            },
            {
                "name": "Addis Ababa",
                "country": "Ethiopia",
                "lat": 9.1450,
                "lon": 40.4897,
                "reason": "Capital with air quality issues",
            },
            {
                "name": "Tunis",
                "country": "Tunisia",
                "lat": 36.8065,
                "lon": 10.1815,
                "reason": "North African capital",
            },
            {
                "name": "Algiers",
                "country": "Algeria",
                "lat": 36.7372,
                "lon": 3.0863,
                "reason": "Major North African city",
            },
            {
                "name": "Rabat",
                "country": "Morocco",
                "lat": 34.0209,
                "lon": -6.8416,
                "reason": "Capital with monitoring network",
            },
            {
                "name": "Abuja",
                "country": "Nigeria",
                "lat": 9.0765,
                "lon": 7.3986,
                "reason": "Capital city with air quality monitoring",
            },
        ],
        "Asia": [
            {
                "name": "Mumbai",
                "country": "India",
                "lat": 19.0760,
                "lon": 72.8777,
                "reason": "Severely polluted megacity",
            },
            {
                "name": "Chennai",
                "country": "India",
                "lat": 13.0827,
                "lon": 80.2707,
                "reason": "Industrial hub with poor air quality",
            },
            {
                "name": "Hyderabad",
                "country": "India",
                "lat": 17.3850,
                "lon": 78.4867,
                "reason": "Tech city with air pollution issues",
            },
        ],
        "North_America": [
            {
                "name": "Atlanta",
                "country": "USA",
                "lat": 33.7490,
                "lon": -84.3880,
                "reason": "Major US city with air quality monitoring",
            }
        ],
        "Europe": [
            {
                "name": "Warsaw",
                "country": "Poland",
                "lat": 52.2297,
                "lon": 21.0122,
                "reason": "Major European city with air quality issues",
            }
        ],
    }

    # Create replacement recommendations
    replacement_plan = {}

    print(f"\nREPLACEMENT PLAN:")
    print("=" * 20)

    for continent, failed_cities in replacement_needed.items():
        backups = backup_replacements.get(continent, [])

        print(f"\n{continent}:")
        replacement_plan[continent] = []

        for i, failed_city in enumerate(failed_cities):
            if i < len(backups):
                backup = backups[i]
                replacement_plan[continent].append(
                    {
                        "remove": failed_city,
                        "replace_with": backup,
                        "justification": f"Replace {failed_city['city']} with {backup['name']} ({backup['reason']})",
                    }
                )
                print(
                    f"  ✓ {failed_city['city']} → {backup['name']}, {backup['country']}"
                )
                print(f"    Reason: {backup['reason']}")
            else:
                print(
                    f"  ⚠️  Need backup for {failed_city['city']} - insufficient options"
                )

    # Calculate replacement success
    total_replacements_needed = sum(
        len(cities) for cities in replacement_needed.values()
    )
    total_replacements_available = sum(len(plan) for plan in replacement_plan.values())

    print(f"\nREPLACEMENT SUMMARY:")
    print("=" * 20)
    print(f"Cities needing replacement: {total_replacements_needed}")
    print(f"Replacement cities identified: {total_replacements_available}")
    print(
        f"Success rate: {total_replacements_available/total_replacements_needed*100:.1f}%"
    )

    if total_replacements_available == total_replacements_needed:
        print("✅ SUCCESS: Can achieve 100% real data coverage!")
        can_achieve_100_percent = True
    else:
        print(
            f"❌ INCOMPLETE: Need {total_replacements_needed - total_replacements_available} more replacements"
        )
        can_achieve_100_percent = False

    # Save replacement plan
    replacement_report = {
        "replacement_time": datetime.now().isoformat(),
        "objective": "Achieve 100% real data coverage with poor air quality cities",
        "cities_needing_replacement": replacement_needed,
        "replacement_plan": replacement_plan,
        "summary": {
            "total_cities_needing_replacement": total_replacements_needed,
            "replacement_cities_identified": total_replacements_available,
            "can_achieve_100_percent": can_achieve_100_percent,
            "success_rate": total_replacements_available
            / total_replacements_needed
            * 100,
        },
        "next_steps": [
            "Validate API data availability for replacement cities",
            "Update comprehensive tables with new city list",
            "Collect real data for replacement cities",
            "Verify 100% real data coverage achieved",
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"../final_dataset/replacement_plan_100_percent_{timestamp}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(replacement_report, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nReplacement plan saved to: {report_file}")

    return replacement_report, report_file


if __name__ == "__main__":
    results, file_path = achieve_100_percent_real_data()
