#!/usr/bin/env python3
"""
Open-Meteo Model Parameter Tester
=================================

Tests Open-Meteo API to determine correct model parameters for CAMS, ECMWF, and GFS forecasts.
"""

import json
from datetime import datetime

import requests


def test_open_meteo_models():
    """Test different model parameters with Open-Meteo API."""

    print("TESTING OPEN-METEO AIR QUALITY API MODELS")
    print("=" * 50)

    # Test location: Delhi
    test_coords = {"lat": 28.6139, "lon": 77.2090}

    # Base parameters
    base_params = {
        "latitude": test_coords["lat"],
        "longitude": test_coords["lon"],
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "forecast_days": 1,
        "timezone": "auto",
    }

    # Test different model configurations
    test_configurations = [
        # Test 1: No model specified (default)
        {"name": "Default (no models parameter)", "params": base_params.copy()},
        # Test 2: Try CAMS
        {"name": "CAMS Global", "params": {**base_params, "models": "cams_global"}},
        # Test 3: Try ECMWF
        {"name": "ECMWF IFS", "params": {**base_params, "models": "ecmwf_ifs04"}},
        # Test 4: Try GFS
        {"name": "GFS Global", "params": {**base_params, "models": "gfs_global"}},
        # Test 5: Try alternative model names
        {"name": "CAMS Alternative", "params": {**base_params, "models": "cams"}},
        {"name": "ECMWF Alternative", "params": {**base_params, "models": "ecmwf"}},
        {"name": "GFS Alternative", "params": {**base_params, "models": "gfs"}},
        # Test 6: Multiple models at once
        {
            "name": "Multiple models",
            "params": {**base_params, "models": "cams_global,ecmwf_ifs04,gfs_global"},
        },
    ]

    results = []

    for i, config in enumerate(test_configurations, 1):
        print(f"\n[{i}/{len(test_configurations)}] Testing: {config['name']}")

        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/air-quality",
                params=config["params"],
                timeout=15,
            )

            print(f"  Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # Analyze response
                has_hourly = "hourly" in data
                hourly_keys = list(data.get("hourly", {}).keys()) if has_hourly else []
                hourly_length = (
                    len(data.get("hourly", {}).get("time", [])) if has_hourly else 0
                )

                result = {
                    "config_name": config["name"],
                    "status": "SUCCESS",
                    "status_code": response.status_code,
                    "has_hourly_data": has_hourly,
                    "hourly_parameters": hourly_keys,
                    "hourly_data_points": hourly_length,
                    "response_size": len(response.text),
                    "params_used": config["params"],
                }

                print(
                    f"  ✓ SUCCESS - Hourly data: {has_hourly}, Parameters: {len(hourly_keys)}, Points: {hourly_length}"
                )

                # Check if we got the specific pollutants we requested
                requested_pollutants = set(config["params"]["hourly"])
                received_pollutants = set(hourly_keys) - {"time"}
                missing_pollutants = requested_pollutants - received_pollutants

                if missing_pollutants:
                    print(f"  ⚠ Missing pollutants: {missing_pollutants}")
                    result["missing_pollutants"] = list(missing_pollutants)
                else:
                    print(f"  ✓ All requested pollutants received")

            else:
                result = {
                    "config_name": config["name"],
                    "status": "FAILED",
                    "status_code": response.status_code,
                    "error_response": response.text[:200],
                    "params_used": config["params"],
                }
                print(f"  ✗ FAILED - Status: {response.status_code}")
                print(f"    Error: {response.text[:100]}")

        except Exception as e:
            result = {
                "config_name": config["name"],
                "status": "ERROR",
                "error": str(e),
                "params_used": config["params"],
            }
            print(f"  ✗ ERROR - {str(e)}")

        results.append(result)

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY OF RESULTS")
    print(f"{'='*50}")

    successful_configs = [r for r in results if r["status"] == "SUCCESS"]
    failed_configs = [r for r in results if r["status"] != "SUCCESS"]

    print(f"Successful configurations: {len(successful_configs)}/{len(results)}")
    print(f"Failed configurations: {len(failed_configs)}/{len(results)}")

    if successful_configs:
        print(f"\n✓ WORKING CONFIGURATIONS:")
        for config in successful_configs:
            models_param = config["params_used"].get("models", "default")
            print(f"  - {config['config_name']} (models='{models_param}')")

    if failed_configs:
        print(f"\n✗ FAILED CONFIGURATIONS:")
        for config in failed_configs:
            models_param = config["params_used"].get("models", "default")
            print(f"  - {config['config_name']} (models='{models_param}')")

    # Save detailed results
    results_file = "stage_5/logs/open_meteo_model_test_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "test_timestamp": datetime.now().isoformat(),
                "test_location": test_coords,
                "total_configurations_tested": len(results),
                "successful_configurations": len(successful_configs),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    test_open_meteo_models()
