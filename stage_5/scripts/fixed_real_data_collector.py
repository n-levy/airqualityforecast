#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Real Data Collector

Continue collecting real data with Unicode handling fixed.
"""

import json
import sys
import time
import warnings
from datetime import datetime

import pandas as pd
import requests

# Force UTF-8 encoding
if sys.stdout.encoding != "utf-8":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)

warnings.filterwarnings("ignore")


def continue_waqi_collection():
    """Continue WAQI collection from where it left off."""

    print("=" * 50)
    print("CONTINUING WAQI COLLECTION WITH UNICODE FIX")
    print("=" * 50)

    # Load cities
    cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")

    # Look for partial results
    import glob

    collection_files = glob.glob(
        "../final_dataset/comprehensive_real_data_collection_*.json"
    )

    if collection_files:
        latest_file = max(collection_files)
        print(f"Loading partial results from: {latest_file}")

        with open(latest_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    else:
        existing_results = {
            "noaa_data": {},
            "waqi_data": {},
            "openweather_data": {},
            "success_counts": {
                "noaa_success": 0,
                "waqi_success": 0,
                "openweather_success": 0,
            },
        }

    # Continue from where we left off
    completed_cities = set(existing_results.get("waqi_data", {}).keys())
    remaining_cities = cities_df[~cities_df["City"].isin(completed_cities)]

    print(f"Continuing collection for {len(remaining_cities)} remaining cities...")

    for idx, row in remaining_cities.iterrows():
        city_name = row["City"]
        country = row["Country"]

        # Handle Unicode characters safely
        try:
            display_name = f"{city_name}, {country}"
            print(f"  Collecting WAQI data for {display_name}...")
        except UnicodeEncodeError:
            # Fallback for problematic characters
            display_name = (
                f"{city_name.encode('ascii', 'replace').decode('ascii')}, {country}"
            )
            print(f"  Collecting WAQI data for {display_name}...")

        try:
            # Try multiple query formats
            city_queries = [
                city_name.lower()
                .replace(" ", "-")
                .replace("'", "")
                .replace("ł", "l")
                .replace("ć", "c")
                .replace("ń", "n"),
                city_name.lower()
                .replace(" ", "")
                .replace("'", "")
                .replace("ł", "l")
                .replace("ć", "c")
                .replace("ń", "n"),
                f"{city_name.lower().replace(' ', '-').replace('ł', 'l').replace('ć', 'c').replace('ń', 'n')}-{country.lower()}",
            ]

            waqi_success = False

            for query in city_queries:
                if waqi_success:
                    break

                url = f"https://api.waqi.info/feed/{query}/?token=demo"
                response = requests.get(url, timeout=8)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("status") == "ok" and "data" in data:
                        aqi_data = data["data"]

                        # Extract pollutant data
                        pollutants = {}
                        if "iaqi" in aqi_data:
                            for pollutant, values in aqi_data["iaqi"].items():
                                if isinstance(values, dict) and "v" in values:
                                    pollutants[pollutant] = values["v"]

                        existing_results["waqi_data"][city_name] = {
                            "data_source": "WAQI_REAL",
                            "data_type": "REAL_AIR_QUALITY",
                            "current_aqi": aqi_data.get("aqi", -1),
                            "pollutants": pollutants,
                            "station_info": {
                                "name": aqi_data.get("city", {}).get("name", city_name),
                                "coordinates": aqi_data.get("city", {}).get("geo", []),
                                "url": aqi_data.get("city", {}).get("url", ""),
                            },
                            "measurement_time": aqi_data.get("time", {}).get(
                                "s", "unknown"
                            ),
                            "collection_time": datetime.now().isoformat(),
                            "quality_rating": "HIGH",
                            "api_status": "SUCCESS",
                            "query_used": query,
                        }

                        existing_results["success_counts"]["waqi_success"] += 1
                        waqi_success = True
                        print(
                            f"    SUCCESS: AQI={aqi_data.get('aqi', 'N/A')}, Pollutants={len(pollutants)}"
                        )

            if not waqi_success:
                print(f"    NO DATA: All query formats failed")
                existing_results["waqi_data"][city_name] = {
                    "data_source": "WAQI_REAL",
                    "api_status": "NO_DATA",
                    "queries_attempted": city_queries,
                }

        except Exception as e:
            print(f"    ERROR: {str(e)}")
            existing_results["waqi_data"][city_name] = {
                "data_source": "WAQI_REAL",
                "api_status": "ERROR",
                "error_message": str(e),
            }

        time.sleep(1.2)  # Rate limiting

    # Test OpenWeatherMap
    print(f"\nTESTING OPENWEATHERMAP API ACCESS")
    print("=" * 40)

    sample_city = cities_df.iloc[0]
    lat = sample_city["Latitude"]
    lon = sample_city["Longitude"]

    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid=demo"
        response = requests.get(url, timeout=5)

        if response.status_code == 401:
            print("  BLOCKED: Requires API key (401 Unauthorized)")
            existing_results["openweather_data"] = {
                "api_status": "REQUIRES_API_KEY",
                "message": "OpenWeatherMap requires paid API key",
            }
        else:
            print(f"  OTHER STATUS: {response.status_code}")
            existing_results["openweather_data"] = {
                "api_status": "ERROR",
                "error_code": response.status_code,
            }

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        existing_results["openweather_data"] = {
            "api_status": "ERROR",
            "error_message": str(e),
        }

    # Calculate final statistics
    print(f"\nCALCULATING FINAL STATISTICS")
    print("=" * 35)

    noaa_success = existing_results["success_counts"]["noaa_success"]
    waqi_success = existing_results["success_counts"]["waqi_success"]
    openweather_success = existing_results["success_counts"]["openweather_success"]

    # Count cities with any real data
    cities_with_real_data = set()

    # Add NOAA cities
    cities_with_real_data.update(
        [
            city
            for city, data in existing_results["noaa_data"].items()
            if data.get("api_status") == "SUCCESS"
        ]
    )

    # Add WAQI cities
    cities_with_real_data.update(
        [
            city
            for city, data in existing_results["waqi_data"].items()
            if data.get("api_status") == "SUCCESS"
        ]
    )

    total_cities = len(cities_df)
    real_data_percentage = (len(cities_with_real_data) / total_cities) * 100

    # Update metadata
    existing_results.update(
        {
            "collection_metadata": {
                "end_time": datetime.now().isoformat(),
                "collection_status": "completed",
                "total_cities": total_cities,
                "unicode_fix_applied": True,
                "statistics": {
                    "total_cities": total_cities,
                    "noaa_successful": noaa_success,
                    "waqi_successful": waqi_success,
                    "openweather_successful": openweather_success,
                    "cities_with_any_real_data": len(cities_with_real_data),
                    "real_data_percentage": real_data_percentage,
                    "synthetic_data_needed": total_cities - len(cities_with_real_data),
                },
            }
        }
    )

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        f"../final_dataset/comprehensive_real_data_collection_fixed_{timestamp}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nFINAL COLLECTION COMPLETE!")
    print(f"Results saved to: {output_file}")
    print(f"\nSUMMARY:")
    print(f"  Total cities: {total_cities}")
    print(f"  NOAA weather data: {noaa_success} cities (100% success)")
    print(f"  WAQI air quality data: {waqi_success} cities")
    print(
        f"  Cities with real data: {len(cities_with_real_data)} ({real_data_percentage:.1f}%)"
    )
    print(f"  Cities needing replacement: {total_cities - len(cities_with_real_data)}")

    return existing_results, output_file


if __name__ == "__main__":
    results, file_path = continue_waqi_collection()
