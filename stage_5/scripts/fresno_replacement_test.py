#!/usr/bin/env python3
"""
Fresno Replacement Test
Replace Fresno with another North American city that works with Open-Meteo
"""
import json
import time
from datetime import datetime, timedelta

import requests


class FresnoReplacementTest:
    def __init__(self):
        # Candidate replacement cities in North America
        self.candidates = [
            {
                "name": "Sacramento",
                "country": "USA",
                "lat": 38.5816,
                "lon": -121.4944,
                "continent": "North America",
            },
            {
                "name": "San Jose",
                "country": "USA",
                "lat": 37.3382,
                "lon": -121.8863,
                "continent": "North America",
            },
            {
                "name": "Oakland",
                "country": "USA",
                "lat": 37.8044,
                "lon": -122.2712,
                "continent": "North America",
            },
            {
                "name": "Portland",
                "country": "USA",
                "lat": 45.5152,
                "lon": -122.6784,
                "continent": "North America",
            },
            {
                "name": "Seattle",
                "country": "USA",
                "lat": 47.6062,
                "lon": -122.3321,
                "continent": "North America",
            },
            {
                "name": "San Diego",
                "country": "USA",
                "lat": 32.7157,
                "lon": -117.1611,
                "continent": "North America",
            },
            {
                "name": "Las Vegas",
                "country": "USA",
                "lat": 36.1699,
                "lon": -115.1398,
                "continent": "North America",
            },
            {
                "name": "Tucson",
                "country": "USA",
                "lat": 32.2226,
                "lon": -110.9747,
                "continent": "North America",
            },
        ]

    def test_open_meteo_city(self, city):
        """Test if a city works with Open-Meteo historical data"""
        print(f"\n--- Testing {city['name']}, {city['country']} ---")

        try:
            # Test current weather first
            current_url = "https://api.open-meteo.com/v1/forecast"
            current_params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "timezone": "auto",
            }

            current_response = requests.get(
                current_url, params=current_params, timeout=10
            )
            if current_response.status_code != 200:
                print(
                    f"  FAILED: Current weather API error {current_response.status_code}"
                )
                return False

            print(f"  SUCCESS: Current weather API working")

            # Test historical data (most critical part)
            end_date = datetime.now() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=30)  # 30 days ago

            historical_url = "https://archive-api.open-meteo.com/v1/archive"
            historical_params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
                "timezone": "auto",
            }

            print(
                f"  Testing historical data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})..."
            )
            historical_response = requests.get(
                historical_url, params=historical_params, timeout=20
            )

            if historical_response.status_code != 200:
                print(
                    f"  FAILED: Historical API error {historical_response.status_code}"
                )
                return False

            historical_data = historical_response.json()
            hourly = historical_data.get("hourly", {})

            if not hourly or not hourly.get("time"):
                print(f"  FAILED: No historical data available")
                return False

            historical_records = len(hourly["time"])
            expected_records = 30 * 24  # 30 days * 24 hours

            print(
                f"  SUCCESS: {historical_records} historical records (expected ~{expected_records})"
            )

            if (
                historical_records < expected_records * 0.8
            ):  # At least 80% of expected data
                print(
                    f"  WARNING: Low data completeness ({historical_records}/{expected_records})"
                )
                return False

            # Test air quality data
            air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            air_quality_params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone",
                "timezone": "auto",
            }

            air_quality_response = requests.get(
                air_quality_url, params=air_quality_params, timeout=10
            )
            if air_quality_response.status_code == 200:
                aq_data = air_quality_response.json()
                current_aq = aq_data.get("current", {})
                if current_aq:
                    print(f"  SUCCESS: Air quality data available")
                else:
                    print(f"  WARNING: Air quality API works but no current data")
            else:
                print(
                    f"  WARNING: Air quality API error {air_quality_response.status_code}"
                )

            print(f"  OVERALL: {city['name']} is SUITABLE as replacement")
            return True

        except Exception as e:
            print(f"  FAILED: Exception {str(e)}")
            return False

    def find_best_replacement(self):
        """Find the best replacement city for Fresno"""
        print("FRESNO REPLACEMENT TEST")
        print("Finding suitable North American city to replace Fresno")
        print("Testing Open-Meteo compatibility (weather + historical + air quality)")
        print("=" * 70)

        working_cities = []

        for city in self.candidates:
            is_working = self.test_open_meteo_city(city)
            if is_working:
                working_cities.append(city)
            time.sleep(1)  # Rate limiting

        print(f"\n" + "=" * 70)
        print("REPLACEMENT TEST RESULTS")
        print("=" * 70)
        print(f"Candidates tested: {len(self.candidates)}")
        print(f"Working cities: {len(working_cities)}")

        if working_cities:
            best_replacement = working_cities[0]  # First working city
            print(f"\nRECOMMENDED REPLACEMENT:")
            print(f"  Replace: Fresno, USA")
            print(f"  With: {best_replacement['name']}, {best_replacement['country']}")
            print(
                f"  Coordinates: {best_replacement['lat']}, {best_replacement['lon']}"
            )
            print(f"  Continent: {best_replacement['continent']}")

            return best_replacement
        else:
            print(f"\nERROR: No suitable replacement found")
            print(f"All candidates failed Open-Meteo compatibility test")
            return None

    def test_replacement_collection(self, replacement_city):
        """Test full data collection for the replacement city"""
        print(f"\n" + "=" * 70)
        print(f"TESTING FULL DATA COLLECTION FOR {replacement_city['name'].upper()}")
        print("=" * 70)

        # This would simulate the full collection process
        # For now, just confirm it works with the same parameters as the main script

        print(f"City: {replacement_city['name']}, {replacement_city['country']}")
        print(f"Coordinates: {replacement_city['lat']}, {replacement_city['lon']}")
        print(f"Continent: {replacement_city['continent']}")

        # Test 2-year historical data (like the main script)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=730)  # 2 years

        print(f"Testing 2-year data collection ({start_date} to {end_date})...")

        try:
            historical_url = "https://archive-api.open-meteo.com/v1/archive"
            historical_params = {
                "latitude": replacement_city["lat"],
                "longitude": replacement_city["lon"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
                "timezone": "auto",
            }

            response = requests.get(
                historical_url, params=historical_params, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                daily_records = len(data.get("daily", {}).get("time", []))
                hourly_records = len(data.get("hourly", {}).get("time", []))

                print(f"SUCCESS: Full 2-year collection test passed")
                print(f"  Daily records: {daily_records}")
                print(f"  Hourly records: {hourly_records}")
                print(f"  Expected daily: ~730")
                print(f"  Expected hourly: ~17520")

                if daily_records >= 700 and hourly_records >= 16000:
                    print(f"EXCELLENT: Data completeness sufficient for replacement")
                    return True
                else:
                    print(f"WARNING: Lower data completeness than expected")
                    return False
            else:
                print(f"FAILED: 2-year collection test failed {response.status_code}")
                return False

        except Exception as e:
            print(f"FAILED: Exception during 2-year test: {str(e)}")
            return False


def main():
    """Main execution"""
    tester = FresnoReplacementTest()

    # Find best replacement
    replacement = tester.find_best_replacement()

    if replacement:
        # Test full collection
        collection_works = tester.test_replacement_collection(replacement)

        if collection_works:
            print(f"\n🎉 SUCCESS: Found perfect replacement for Fresno!")
            print(f"✅ Replace Fresno with {replacement['name']}")
            print(f"✅ Full Open-Meteo compatibility confirmed")
            print(f"✅ 2-year historical data collection ready")

            # Save replacement info
            replacement_info = {
                "original_city": {
                    "name": "Fresno",
                    "country": "USA",
                    "issue": "Historical data timeout",
                },
                "replacement_city": replacement,
                "test_timestamp": datetime.now().isoformat(),
                "test_results": "SUCCESS - Full compatibility confirmed",
            }

            with open("../final_dataset/fresno_replacement.json", "w") as f:
                json.dump(replacement_info, f, indent=2)

            print(f"\nReplacement info saved: fresno_replacement.json")
        else:
            print(
                f"\n⚠️ WARNING: {replacement['name']} works but has data completeness issues"
            )
    else:
        print(f"\n❌ CRITICAL: No suitable replacement found")


if __name__ == "__main__":
    main()
