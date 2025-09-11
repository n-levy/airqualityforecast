#!/usr/bin/env python3
"""
OpenAQ API Test Script
Test OpenAQ API for real measured air quality data from monitoring stations
"""
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


class OpenAQAPITest:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v2"
        self.headers = {
            "User-Agent": "AQF311-Research/1.0",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["X-API-Key"] = api_key

        self.test_cities = [
            {"name": "Beijing", "country": "CN", "lat": 39.9042, "lon": 116.4074},
            {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "country": "GB", "lat": 51.5074, "lon": -0.1278},
            {"name": "Delhi", "country": "IN", "lat": 28.6139, "lon": 77.2090},
            {"name": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503},
        ]

        self.results = {}

    def test_api_availability(self):
        """Test if API is accessible without key"""
        print("=== TESTING OPENAQ API AVAILABILITY ===")

        try:
            # Test basic API endpoint
            url = f"{self.base_url}/locations"
            params = {"limit": 1}

            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                total_locations = data.get("meta", {}).get("found", 0)
                print(f"SUCCESS: API Access works")
                print(f"   Total monitoring locations: {total_locations:,}")
                return True
            elif response.status_code == 401:
                print(f"KEY REQUIRED: API Access needs authentication")
                print(f"   Status: {response.status_code}")
                return False
            else:
                print(f"ERROR: API Access failed {response.status_code}")
                return False

        except Exception as e:
            print(f"EXCEPTION: API Access failed - {str(e)}")
            return False

    def find_nearby_stations(self, city_name, country, lat, lon, radius_km=50):
        """Find monitoring stations near a city"""
        print(f"\n--- Finding stations near {city_name}, {country} ---")

        try:
            url = f"{self.base_url}/locations"
            params = {
                "coordinates": f"{lat},{lon}",
                "radius": radius_km * 1000,  # Convert to meters
                "limit": 10,
                "sort": "lastUpdated",
            }

            response = requests.get(
                url, headers=self.headers, params=params, timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                stations = data.get("results", [])

                if stations:
                    print(f"  Found {len(stations)} stations within {radius_km}km")

                    # Get details of best station
                    best_station = stations[0]
                    station_id = best_station.get("id")
                    station_name = best_station.get("name", "Unknown")
                    last_updated = best_station.get("lastUpdated", "Unknown")
                    parameters = [
                        p.get("parameter") for p in best_station.get("parameters", [])
                    ]

                    print(f"  Best station: {station_name} (ID: {station_id})")
                    print(f"  Last updated: {last_updated}")
                    print(f"  Parameters: {', '.join(parameters)}")

                    return {
                        "station_found": True,
                        "station_id": station_id,
                        "station_name": station_name,
                        "parameters": parameters,
                        "last_updated": last_updated,
                        "total_stations": len(stations),
                    }
                else:
                    print(f"  No stations found within {radius_km}km")
                    return {"station_found": False, "total_stations": 0}

            else:
                print(f"  API Error: {response.status_code}")
                return {"station_found": False, "error": response.status_code}

        except Exception as e:
            print(f"  Exception: {str(e)}")
            return {"station_found": False, "error": str(e)}

    def get_recent_measurements(self, station_id, city_name):
        """Get recent measurements from a station"""
        print(f"\n--- Getting recent data for {city_name} (Station: {station_id}) ---")

        try:
            url = f"{self.base_url}/measurements"
            params = {
                "location_id": station_id,
                "limit": 100,
                "sort": "datetime",
                "order": "desc",
            }

            response = requests.get(
                url, headers=self.headers, params=params, timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                measurements = data.get("results", [])

                if measurements:
                    print(f"  Retrieved {len(measurements)} recent measurements")

                    # Analyze pollutants available
                    pollutants = {}
                    for m in measurements:
                        param = m.get("parameter")
                        value = m.get("value")
                        unit = m.get("unit")
                        date = m.get("date", {}).get("utc", "Unknown")

                        if param not in pollutants:
                            pollutants[param] = {
                                "count": 0,
                                "latest_value": None,
                                "latest_date": None,
                                "unit": unit,
                            }

                        pollutants[param]["count"] += 1
                        if (
                            pollutants[param]["latest_date"] is None
                            or date > pollutants[param]["latest_date"]
                        ):
                            pollutants[param]["latest_value"] = value
                            pollutants[param]["latest_date"] = date

                    print(f"  Available pollutants:")
                    for param, info in pollutants.items():
                        print(
                            f"    {param}: {info['latest_value']} {info['unit']} ({info['count']} measurements)"
                        )

                    return {
                        "data_available": True,
                        "total_measurements": len(measurements),
                        "pollutants": pollutants,
                    }
                else:
                    print(f"  No recent measurements available")
                    return {"data_available": False}

            else:
                print(f"  API Error: {response.status_code}")
                return {"data_available": False, "error": response.status_code}

        except Exception as e:
            print(f"  Exception: {str(e)}")
            return {"data_available": False, "error": str(e)}

    def test_comprehensive_coverage(self):
        """Test API coverage across different cities"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE OPENAQ API COVERAGE TEST")
        print("=" * 60)

        # First test API availability
        api_available = self.test_api_availability()

        if not api_available:
            print("\nAPI KEY REQUIRED - Cannot proceed with free access")
            return False

        # Test coverage across cities
        successful_cities = 0
        total_stations = 0
        total_pollutants = set()

        for city in self.test_cities:
            # Find stations
            station_result = self.find_nearby_stations(
                city["name"], city["country"], city["lat"], city["lon"]
            )

            if station_result.get("station_found"):
                successful_cities += 1
                total_stations += station_result.get("total_stations", 0)

                # Get measurements
                measurements_result = self.get_recent_measurements(
                    station_result["station_id"], city["name"]
                )

                if measurements_result.get("data_available"):
                    pollutants = measurements_result.get("pollutants", {})
                    total_pollutants.update(pollutants.keys())

                # Store results
                self.results[city["name"]] = {
                    "station_result": station_result,
                    "measurements_result": measurements_result,
                }

            time.sleep(1)  # Rate limiting

        # Summary
        print(f"\n" + "=" * 60)
        print("OPENAQ API TEST SUMMARY")
        print("=" * 60)
        print(f"Cities tested: {len(self.test_cities)}")
        print(f"Cities with stations: {successful_cities}")
        print(f"Success rate: {successful_cities/len(self.test_cities)*100:.1f}%")
        print(f"Total stations found: {total_stations}")
        print(f"Unique pollutants: {len(total_pollutants)}")
        print(f"Available pollutants: {', '.join(sorted(total_pollutants))}")

        # Check AQI component coverage
        aqi_components = {"pm25", "pm10", "no2", "o3", "co", "so2"}
        found_components = {p.lower().replace(".", "") for p in total_pollutants}
        missing_components = aqi_components - found_components

        print(f"\nAQI Component Coverage:")
        for component in aqi_components:
            status = "YES" if component in found_components else "NO"
            print(f"  {status}: {component.upper()}")

        if missing_components:
            print(f"\nMissing AQI components: {', '.join(missing_components).upper()}")

        return successful_cities >= 3  # Consider successful if >=3 cities have data

    def save_results(self):
        """Save test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../final_dataset/openaq_api_test_results_{timestamp}.json"

        test_summary = {
            "test_timestamp": datetime.now().isoformat(),
            "api_key_used": self.api_key is not None,
            "test_cities": self.test_cities,
            "results": self.results,
            "summary": {
                "cities_tested": len(self.test_cities),
                "cities_with_data": len(
                    [
                        r
                        for r in self.results.values()
                        if r.get("station_result", {}).get("station_found")
                    ]
                ),
                "recommendation": (
                    "Use OpenAQ API for real monitoring station data"
                    if self.results
                    else "API key required"
                ),
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(test_summary, f, indent=2, default=str)

        print(f"\nResults saved: {filename}")
        return filename


def main():
    """Main execution - test without API key first"""
    print("OPENAQ API TEST - Testing free access first")
    print("If free access fails, will notify about API key requirement")

    # Test without API key first
    tester = OpenAQAPITest()  # No API key
    success = tester.test_comprehensive_coverage()
    tester.save_results()

    if success:
        print(f"\nSUCCESS: OpenAQ API works without authentication!")
        print(f"Can proceed with free real monitoring station data")
    else:
        print(f"\nAPI KEY REQUIRED: OpenAQ needs authentication")
        print(f"Please provide your OpenAQ API key to access real monitoring data")
        print(
            f"This will give access to authentic measured pollutant data from ground stations"
        )


if __name__ == "__main__":
    main()
