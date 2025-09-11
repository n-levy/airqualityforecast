#!/usr/bin/env python3
"""
Quick OpenAQ API v3 Test
Test your API key with the correct v3 endpoints and authentication method
"""
import json

import requests


def test_openaq_v3_api(api_key):
    """Quick test of OpenAQ v3 API"""
    print("=== OPENAQ API v3 QUICK TEST ===")

    base_url = "https://api.openaq.org/v3"
    headers = {
        "User-Agent": "AQF311-Research/1.0",
        "Accept": "application/json",
        "X-API-Key": api_key,  # Based on OpenAPI spec showing APIKeyHeader
    }

    # Test 1: Get locations (basic test)
    print("1. Testing /v3/locations endpoint...")
    try:
        response = requests.get(
            f"{base_url}/locations", headers=headers, params={"limit": 5}, timeout=10
        )
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"   SUCCESS: Found {len(results)} locations")

            if results:
                first_location = results[0]
                location_id = first_location.get("id")
                location_name = first_location.get("name", "Unknown")
                coordinates = first_location.get("coordinates", {})
                print(f"   Sample location: {location_name} (ID: {location_id})")
                print(f"   Coordinates: {coordinates}")
                return True, location_id
        else:
            print(f"   ERROR: {response.status_code} - {response.text[:100]}")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")

    return False, None


def test_sensor_measurements(api_key, location_id):
    """Test getting sensor measurements"""
    print(f"\n2. Testing sensor measurements for location {location_id}...")

    base_url = "https://api.openaq.org/v3"
    headers = {
        "User-Agent": "AQF311-Research/1.0",
        "Accept": "application/json",
        "X-API-Key": api_key,
    }

    try:
        # First get sensors for this location
        response = requests.get(
            f"{base_url}/locations/{location_id}", headers=headers, timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            # Look for sensors in the location data
            print(f"   Location data retrieved successfully")

            # Try to get measurements endpoint (might need different approach)
            response2 = requests.get(
                f"{base_url}/sensors", headers=headers, params={"limit": 1}, timeout=10
            )
            print(f"   Sensors endpoint status: {response2.status_code}")

            if response2.status_code == 200:
                sensors_data = response2.json()
                sensors = sensors_data.get("results", [])
                if sensors:
                    sensor_id = sensors[0].get("id")
                    print(f"   Found sensor ID: {sensor_id}")

                    # Try to get measurements
                    response3 = requests.get(
                        f"{base_url}/sensors/{sensor_id}/measurements",
                        headers=headers,
                        params={"limit": 5},
                        timeout=10,
                    )
                    print(f"   Measurements status: {response3.status_code}")

                    if response3.status_code == 200:
                        measurements_data = response3.json()
                        measurements = measurements_data.get("results", [])
                        print(f"   SUCCESS: Found {len(measurements)} measurements")

                        if measurements:
                            sample = measurements[0]
                            parameter = sample.get("parameter", {}).get(
                                "name", "Unknown"
                            )
                            value = sample.get("value")
                            unit = sample.get("parameter", {}).get("units", "Unknown")
                            datetime = sample.get("datetime")
                            print(
                                f"   Sample: {parameter} = {value} {unit} at {datetime}"
                            )

                        return True
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")

    return False


def main():
    api_key = "8c71a560478a03671edd9be444571ba70afbe82d9fd3a9d9b2612e8d806287f8"

    print("OPENAQ API v3 QUICK TEST")
    print("Testing your API key with correct v3 endpoints")
    print("=" * 50)

    # Test basic API access
    api_works, location_id = test_openaq_v3_api(api_key)

    if api_works and location_id:
        # Test measurements
        measurements_work = test_sensor_measurements(api_key, location_id)

        if measurements_work:
            print(f"\n🎉 SUCCESS: OpenAQ API v3 is working with your key!")
            print(f"✅ Locations: Accessible")
            print(f"✅ Measurements: Accessible")
            print(f"Ready to collect real air quality data from monitoring stations")
        else:
            print(
                f"\n⚠️  PARTIAL SUCCESS: Locations work, measurements need investigation"
            )
    else:
        print(f"\n❌ FAILED: API key authentication issue")
        print(
            f"Please check if the API key is correct or if there are access restrictions"
        )


if __name__ == "__main__":
    main()
