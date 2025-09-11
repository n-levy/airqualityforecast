#!/usr/bin/env python3
"""
Achieve 100% Real Data Coverage

Replace cities without real data sources with backup cities that have verified real data,
achieving 100% real data coverage across all 100 cities (20 per continent).
"""

import json
import time
import warnings
from datetime import datetime

import pandas as pd
import requests

warnings.filterwarnings("ignore")


class CityReplacementFor100PercentReal:
    """Replace cities to achieve 100% real data coverage."""

    def __init__(self):
        """Initialize city replacement system."""

        # Load current data
        self.cities_df = pd.read_csv(
            "../comprehensive_tables/comprehensive_features_table.csv"
        )

        with open(
            "../final_dataset/complete_real_data_collection_20250911_192217.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.collection_data = json.load(f)

        # Backup cities with known poor air quality by continent
        self.backup_cities = {
            "Asia": [
                {"name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
                {
                    "name": "Karachi",
                    "country": "Pakistan",
                    "lat": 24.8607,
                    "lon": 67.0011,
                },
                {"name": "Chennai", "country": "India", "lat": 13.0827, "lon": 80.2707},
                {
                    "name": "Hyderabad",
                    "country": "India",
                    "lat": 17.3850,
                    "lon": 78.4867,
                },
                {
                    "name": "Ahmedabad",
                    "country": "India",
                    "lat": 23.0225,
                    "lon": 72.5714,
                },
                {"name": "Pune", "country": "India", "lat": 18.5204, "lon": 73.8567},
                {"name": "Surat", "country": "India", "lat": 21.1702, "lon": 72.8311},
                {"name": "Kanpur", "country": "India", "lat": 26.4499, "lon": 80.3319},
                {"name": "Jaipur", "country": "India", "lat": 26.9124, "lon": 75.7873},
                {"name": "Indore", "country": "India", "lat": 22.7196, "lon": 75.8577},
                {
                    "name": "Bangkok",
                    "country": "Thailand",
                    "lat": 13.7563,
                    "lon": 100.5018,
                },
                {
                    "name": "Jakarta",
                    "country": "Indonesia",
                    "lat": -6.2088,
                    "lon": 106.8456,
                },
                {
                    "name": "Manila",
                    "country": "Philippines",
                    "lat": 14.5995,
                    "lon": 120.9842,
                },
                {
                    "name": "Hanoi",
                    "country": "Vietnam",
                    "lat": 21.0285,
                    "lon": 105.8542,
                },
                {
                    "name": "Seoul",
                    "country": "South Korea",
                    "lat": 37.5665,
                    "lon": 126.9780,
                },
                {
                    "name": "Chengdu",
                    "country": "China",
                    "lat": 30.5728,
                    "lon": 104.0668,
                },
                {"name": "Wuhan", "country": "China", "lat": 30.5928, "lon": 114.3055},
                {
                    "name": "Chongqing",
                    "country": "China",
                    "lat": 29.4316,
                    "lon": 106.9123,
                },
                {"name": "Xi'an", "country": "China", "lat": 34.3416, "lon": 108.9398},
                {
                    "name": "Tianjin",
                    "country": "China",
                    "lat": 39.3434,
                    "lon": 117.3616,
                },
            ],
            "Africa": [
                {
                    "name": "Johannesburg",
                    "country": "South Africa",
                    "lat": -26.2041,
                    "lon": 28.0473,
                },
                {
                    "name": "Cape Town",
                    "country": "South Africa",
                    "lat": -33.9249,
                    "lon": 18.4241,
                },
                {
                    "name": "Durban",
                    "country": "South Africa",
                    "lat": -29.8587,
                    "lon": 31.0218,
                },
                {"name": "Nairobi", "country": "Kenya", "lat": -1.2921, "lon": 36.8219},
                {
                    "name": "Addis Ababa",
                    "country": "Ethiopia",
                    "lat": 9.1450,
                    "lon": 40.4897,
                },
                {"name": "Tunis", "country": "Tunisia", "lat": 36.8065, "lon": 10.1815},
                {
                    "name": "Algiers",
                    "country": "Algeria",
                    "lat": 36.7372,
                    "lon": 3.0863,
                },
                {"name": "Rabat", "country": "Morocco", "lat": 34.0209, "lon": -6.8416},
                {"name": "Luanda", "country": "Angola", "lat": -8.8390, "lon": 13.2894},
                {"name": "Abuja", "country": "Nigeria", "lat": 9.0765, "lon": 7.3986},
                {"name": "Kano", "country": "Nigeria", "lat": 12.0022, "lon": 8.5920},
                {"name": "Ibadan", "country": "Nigeria", "lat": 7.3775, "lon": 3.9470},
                {
                    "name": "Port Harcourt",
                    "country": "Nigeria",
                    "lat": 4.8156,
                    "lon": 7.0498,
                },
                {
                    "name": "Benin City",
                    "country": "Nigeria",
                    "lat": 6.3350,
                    "lon": 5.6037,
                },
                {
                    "name": "Maputo",
                    "country": "Mozambique",
                    "lat": -25.9692,
                    "lon": 32.5732,
                },
                {
                    "name": "Harare",
                    "country": "Zimbabwe",
                    "lat": -17.8292,
                    "lon": 31.0522,
                },
                {
                    "name": "Lusaka",
                    "country": "Zambia",
                    "lat": -15.3875,
                    "lon": 28.3228,
                },
                {
                    "name": "Dar es Salaam",
                    "country": "Tanzania",
                    "lat": -6.7924,
                    "lon": 39.2083,
                },
                {
                    "name": "Libreville",
                    "country": "Gabon",
                    "lat": 0.4162,
                    "lon": 9.4673,
                },
                {
                    "name": "Port Elizabeth",
                    "country": "South Africa",
                    "lat": -33.9180,
                    "lon": 25.5701,
                },
            ],
            "Europe": [
                {"name": "Warsaw", "country": "Poland", "lat": 52.2297, "lon": 21.0122},
                {
                    "name": "Prague",
                    "country": "Czech Republic",
                    "lat": 50.0755,
                    "lon": 14.4378,
                },
                {
                    "name": "Budapest",
                    "country": "Hungary",
                    "lat": 47.4979,
                    "lon": 19.0402,
                },
                {"name": "Athens", "country": "Greece", "lat": 37.9838, "lon": 23.7275},
                {"name": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964},
                {"name": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038},
                {
                    "name": "Ljubljana",
                    "country": "Slovenia",
                    "lat": 46.0569,
                    "lon": 14.5058,
                },
                {
                    "name": "Bratislava",
                    "country": "Slovakia",
                    "lat": 48.1486,
                    "lon": 17.1077,
                },
                {
                    "name": "Zagreb",
                    "country": "Croatia",
                    "lat": 45.8150,
                    "lon": 15.9819,
                },
                {
                    "name": "Podgorica",
                    "country": "Montenegro",
                    "lat": 42.4304,
                    "lon": 19.2594,
                },
            ],
            "North_America": [
                {
                    "name": "Mexico City",
                    "country": "Mexico",
                    "lat": 19.4326,
                    "lon": -99.1332,
                },
                {
                    "name": "Los Angeles",
                    "country": "USA",
                    "lat": 34.0522,
                    "lon": -118.2437,
                },
                {"name": "Houston", "country": "USA", "lat": 29.7604, "lon": -95.3698},
                {"name": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298},
                {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
                {
                    "name": "Philadelphia",
                    "country": "USA",
                    "lat": 39.9526,
                    "lon": -75.1652,
                },
                {
                    "name": "Toronto",
                    "country": "Canada",
                    "lat": 43.6532,
                    "lon": -79.3832,
                },
                {
                    "name": "Vancouver",
                    "country": "Canada",
                    "lat": 49.2827,
                    "lon": -123.1207,
                },
                {
                    "name": "Montreal",
                    "country": "Canada",
                    "lat": 45.5017,
                    "lon": -73.5673,
                },
                {"name": "Atlanta", "country": "USA", "lat": 33.7490, "lon": -84.3880},
            ],
            "South_America": [
                {
                    "name": "São Paulo",
                    "country": "Brazil",
                    "lat": -23.5505,
                    "lon": -46.6333,
                },
                {
                    "name": "Rio de Janeiro",
                    "country": "Brazil",
                    "lat": -22.9068,
                    "lon": -43.1729,
                },
                {"name": "Lima", "country": "Peru", "lat": -12.0464, "lon": -77.0428},
                {
                    "name": "Bogotá",
                    "country": "Colombia",
                    "lat": 4.7110,
                    "lon": -74.0721,
                },
                {
                    "name": "Santiago",
                    "country": "Chile",
                    "lat": -33.4489,
                    "lon": -70.6693,
                },
                {
                    "name": "Buenos Aires",
                    "country": "Argentina",
                    "lat": -34.6118,
                    "lon": -58.3960,
                },
                {
                    "name": "Caracas",
                    "country": "Venezuela",
                    "lat": 10.4806,
                    "lon": -66.9036,
                },
                {
                    "name": "Montevideo",
                    "country": "Uruguay",
                    "lat": -34.9011,
                    "lon": -56.1645,
                },
                {
                    "name": "Asunción",
                    "country": "Paraguay",
                    "lat": -25.2637,
                    "lon": -57.5759,
                },
                {
                    "name": "La Paz",
                    "country": "Bolivia",
                    "lat": -16.5000,
                    "lon": -68.1193,
                },
                {
                    "name": "Salvador",
                    "country": "Brazil",
                    "lat": -12.9714,
                    "lon": -38.5014,
                },
                {
                    "name": "Fortaleza",
                    "country": "Brazil",
                    "lat": -3.7319,
                    "lon": -38.5267,
                },
                {
                    "name": "Brasília",
                    "country": "Brazil",
                    "lat": -15.8267,
                    "lon": -47.9218,
                },
                {
                    "name": "Recife",
                    "country": "Brazil",
                    "lat": -8.0476,
                    "lon": -34.8770,
                },
                {
                    "name": "Manaus",
                    "country": "Brazil",
                    "lat": -3.1190,
                    "lon": -60.0217,
                },
            ],
        }

    def get_cities_needing_replacement(self):
        """Get cities without real data grouped by continent."""

        # Find cities with real data
        real_cities = set()
        for city, info in self.collection_data["noaa_data"].items():
            if info.get("api_status") == "SUCCESS":
                real_cities.add(city)
        for city, info in self.collection_data["waqi_data"].items():
            if info.get("api_status") == "SUCCESS":
                real_cities.add(city)

        all_cities = set(self.cities_df["City"].tolist())
        cities_without_real = all_cities - real_cities

        # Group by continent
        replacement_needed = {}
        for city in cities_without_real:
            city_row = self.cities_df[self.cities_df["City"] == city].iloc[0]
            continent = city_row["Continent"]
            if continent not in replacement_needed:
                replacement_needed[continent] = []
            replacement_needed[continent].append(
                {
                    "city": city,
                    "country": city_row["Country"],
                    "lat": city_row["Latitude"],
                    "lon": city_row["Longitude"],
                }
            )

        return replacement_needed, real_cities

    def test_city_data_availability(self, city_info):
        """Test if a backup city has real data available."""

        city_name = city_info["name"]
        country = city_info["country"]
        lat = city_info["lat"]
        lon = city_info["lon"]

        print(f"    Testing {city_name}, {country}...")

        data_sources = {
            "noaa_available": False,
            "waqi_available": False,
            "data_quality_score": 0,
        }

        # Test NOAA if US city
        if country == "USA":
            try:
                grid_url = f"https://api.weather.gov/points/{lat},{lon}"
                response = requests.get(grid_url, timeout=5)
                if response.status_code == 200:
                    grid_data = response.json()
                    # Test forecast availability
                    forecast_url = grid_data["properties"]["forecast"]
                    forecast_response = requests.get(forecast_url, timeout=5)
                    if forecast_response.status_code == 200:
                        data_sources["noaa_available"] = True
                        data_sources["data_quality_score"] += 50
                        print(f"      NOAA: Available")
                    else:
                        print(f"      NOAA: Forecast unavailable")
                else:
                    print(f"      NOAA: Grid unavailable")
            except Exception as e:
                print(f"      NOAA: Error - {str(e)}")
            time.sleep(0.5)

        # Test WAQI
        try:
            city_queries = [
                city_name.lower().replace(" ", "-").replace("'", ""),
                city_name.lower().replace(" ", "").replace("'", ""),
                f"{city_name.lower().replace(' ', '-')}-{country.lower()}",
            ]

            waqi_success = False
            for query in city_queries:
                if waqi_success:
                    break

                url = f"https://api.waqi.info/feed/{query}/?token=demo"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok" and "data" in data:
                        aqi_data = data["data"]
                        current_aqi = aqi_data.get("aqi", -1)

                        if current_aqi > 0:
                            data_sources["waqi_available"] = True
                            data_sources["data_quality_score"] += 50
                            data_sources["current_aqi"] = current_aqi
                            waqi_success = True
                            print(f"      WAQI: Available (AQI={current_aqi})")

            if not waqi_success:
                print(f"      WAQI: Unavailable")

        except Exception as e:
            print(f"      WAQI: Error - {str(e)}")

        time.sleep(1.0)  # Rate limiting
        return data_sources

    def find_replacement_cities(self):
        """Find replacement cities with verified real data sources."""

        print("FINDING REPLACEMENT CITIES FOR 100% REAL DATA")
        print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 55)

        cities_needing_replacement, current_real_cities = (
            self.get_cities_needing_replacement()
        )

        print("CITIES NEEDING REPLACEMENT:")
        print("=" * 30)
        total_to_replace = 0
        for continent, cities in cities_needing_replacement.items():
            print(f"{continent}: {len(cities)} cities")
            total_to_replace += len(cities)
            for city_info in cities:
                print(f"  - {city_info['city']}, {city_info['country']}")

        print(f"\nTotal cities to replace: {total_to_replace}")

        # Find replacements for each continent
        recommended_replacements = {}

        for continent, failed_cities in cities_needing_replacement.items():
            print(f"\nTESTING BACKUP CITIES FOR {continent}:")
            print("-" * 40)

            backup_cities = self.backup_cities.get(continent, [])
            tested_backups = []

            for backup_city in backup_cities:
                # Skip if already in current dataset
                if backup_city["name"] in self.cities_df["City"].values:
                    continue

                data_availability = self.test_city_data_availability(backup_city)

                tested_backups.append(
                    {"city_info": backup_city, "data_availability": data_availability}
                )

                # Stop testing once we have enough viable options
                viable_count = sum(
                    1
                    for t in tested_backups
                    if t["data_availability"]["data_quality_score"] > 0
                )
                if viable_count >= len(failed_cities) + 2:  # Extra buffer
                    break

            # Sort by data quality score
            tested_backups.sort(
                key=lambda x: x["data_availability"]["data_quality_score"], reverse=True
            )

            # Create specific replacement recommendations
            recommended_swaps = []
            for i, failed_city in enumerate(failed_cities):
                if (
                    i < len(tested_backups)
                    and tested_backups[i]["data_availability"]["data_quality_score"] > 0
                ):
                    recommended_swaps.append(
                        {
                            "remove": failed_city,
                            "replace_with": tested_backups[i]["city_info"],
                            "data_score": tested_backups[i]["data_availability"][
                                "data_quality_score"
                            ],
                            "data_sources": tested_backups[i]["data_availability"],
                        }
                    )

            recommended_replacements[continent] = {
                "failed_cities": failed_cities,
                "tested_backups": tested_backups,
                "recommended_swaps": recommended_swaps,
            }

        return recommended_replacements

    def generate_replacement_report(self, recommended_replacements):
        """Generate comprehensive replacement report."""

        print(f"\nGENERATING 100% REAL DATA REPLACEMENT REPORT")
        print("=" * 50)

        replacement_report = {
            "replacement_time": datetime.now().isoformat(),
            "objective": "Achieve 100% real data coverage across 100 cities (20 per continent)",
            "recommended_replacements": recommended_replacements,
            "summary": {
                "total_replacements": 0,
                "continents_affected": list(recommended_replacements.keys()),
                "replacement_feasibility": {},
            },
        }

        total_successful_replacements = 0

        print("REPLACEMENT RECOMMENDATIONS:")
        print("=" * 35)

        for continent, replacements in recommended_replacements.items():
            failed_count = len(replacements["failed_cities"])
            successful_swaps = len(replacements["recommended_swaps"])

            print(
                f"\n{continent}: {successful_swaps}/{failed_count} replacements found"
            )

            for swap in replacements["recommended_swaps"]:
                remove_city = swap["remove"]["city"]
                replace_city = swap["replace_with"]["name"]
                replace_country = swap["replace_with"]["country"]
                score = swap["data_score"]

                print(
                    f"  ✓ Replace {remove_city} with {replace_city}, {replace_country} (Score: {score})"
                )

            total_successful_replacements += successful_swaps

            replacement_report["summary"]["replacement_feasibility"][continent] = {
                "cities_to_replace": failed_count,
                "successful_replacements": successful_swaps,
                "success_rate": (
                    successful_swaps / failed_count if failed_count > 0 else 1.0
                ),
            }

        replacement_report["summary"][
            "total_replacements"
        ] = total_successful_replacements

        print(f"\nSUMMARY:")
        print("=" * 10)
        print(f"Total cities needing replacement: 22")
        print(f"Successful replacements found: {total_successful_replacements}")
        print(f"Replacement success rate: {total_successful_replacements/22*100:.1f}%")

        if total_successful_replacements == 22:
            print("✅ SUCCESS: Can achieve 100% real data coverage!")
        else:
            print(
                f"⚠️  WARNING: Only {total_successful_replacements}/22 replacements found"
            )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            f"../final_dataset/100_percent_real_data_replacement_{timestamp}.json"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(replacement_report, f, indent=2, default=str, ensure_ascii=False)

        print(f"\nReplacement report saved to: {report_file}")

        return replacement_report, report_file


def main():
    """Main function to achieve 100% real data coverage."""

    replacer = CityReplacementFor100PercentReal()

    # Find replacement cities
    recommended_replacements = replacer.find_replacement_cities()

    # Generate comprehensive report
    report, report_file = replacer.generate_replacement_report(recommended_replacements)

    return report, report_file


if __name__ == "__main__":
    results, file_path = main()
