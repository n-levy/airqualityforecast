#!/usr/bin/env python3
"""
Simple API Test for Alternative Air Quality APIs
Test basic functionality and data availability
"""
import json
import time
from datetime import datetime

import requests


def test_openaq_simple():
    """Test OpenAQ API with simple request"""
    print("=== TESTING OPENAQ API ===")
    try:
        # Test getting locations for major cities
        cities = ["London", "Paris", "Tokyo", "New York"]
        successful = 0

        for city in cities:
            url = "https://api.openaq.org/v2/locations"
            params = {"limit": 5, "city": city}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                locations = data.get("results", [])
                if locations:
                    location = locations[0]
                    print(
                        f"  SUCCESS {city}: Found location {location.get('name')} with {len(location.get('parameters', []))} parameters"
                    )
                    successful += 1
                else:
                    print(f"  FAILED {city}: No locations found")
            else:
                print(f"  FAILED {city}: API error {response.status_code}")

            time.sleep(0.5)

        print(f"OpenAQ Results: {successful}/{len(cities)} cities successful")
        return successful > 0

    except Exception as e:
        print(f"OpenAQ Error: {str(e)}")
        return False


def test_airnow_simple():
    """Test AirNow API with simple request"""
    print("\n=== TESTING AIRNOW API ===")
    try:
        # Test US cities only
        cities = [
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        ]
        successful = 0

        for city in cities:
            url = "https://www.airnowapi.org/aq/observation/latLong/current/"
            params = {
                "format": "application/json",
                "latitude": city["lat"],
                "longitude": city["lon"],
                "distance": 25,
                "API_KEY": "guest",
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    obs = data[0]
                    print(
                        f"  SUCCESS {city['name']}: AQI={obs.get('AQI')} {obs.get('ParameterName')} in {obs.get('ReportingArea')}"
                    )
                    successful += 1
                else:
                    print(f"  FAILED {city['name']}: No data available")
            else:
                print(f"  FAILED {city['name']}: API error {response.status_code}")

            time.sleep(1)

        print(f"AirNow Results: {successful}/{len(cities)} cities successful")
        return successful > 0

    except Exception as e:
        print(f"AirNow Error: {str(e)}")
        return False


def test_iqair_simple():
    """Test IQAir API with simple request"""
    print("\n=== TESTING IQAIR API ===")
    try:
        # Test nearest city API
        cities = [
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        ]
        successful = 0

        for city in cities:
            url = "https://api.airvisual.com/v2/nearest_city"
            params = {"lat": city["lat"], "lon": city["lon"], "key": "demo"}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    city_data = data["data"]
                    pollution = city_data.get("current", {}).get("pollution", {})
                    print(
                        f"  SUCCESS {city['name']} -> {city_data.get('city')}: AQI={pollution.get('aqius')}"
                    )
                    successful += 1
                else:
                    print(f"  FAILED {city['name']}: API returned error")
            else:
                print(f"  FAILED {city['name']}: API error {response.status_code}")

            time.sleep(1)

        print(f"IQAir Results: {successful}/{len(cities)} cities successful")
        return successful > 0

    except Exception as e:
        print(f"IQAir Error: {str(e)}")
        return False


def main():
    """Main test execution"""
    print("SIMPLE ALTERNATIVE API TEST")
    print("Testing basic functionality of alternative air quality APIs")
    print("=" * 60)

    results = {"timestamp": datetime.now().isoformat(), "apis": {}}

    # Test each API
    results["apis"]["openaq"] = test_openaq_simple()
    results["apis"]["airnow"] = test_airnow_simple()
    results["apis"]["iqair"] = test_iqair_simple()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    working_apis = [api for api, working in results["apis"].items() if working]
    print(f"Working APIs: {len(working_apis)}/{len(results['apis'])}")
    print(f"Functional APIs: {', '.join(working_apis)}")

    if len(working_apis) >= 2:
        print("Status: VIABLE - Multi-API approach possible")
        print("Recommendation: Proceed with detailed testing of working APIs")
    else:
        print("Status: LIMITED - Few working APIs found")
        print("Recommendation: Focus on cities with verified data from working APIs")

    # Save results
    with open("../final_dataset/simple_api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: simple_api_test_results.json")


if __name__ == "__main__":
    main()
