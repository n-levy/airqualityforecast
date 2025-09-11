#!/usr/bin/env python3
"""
Real Benchmark API Research and Testing

Research and test real forecast APIs that can provide actual benchmark
forecast data for CAMS, NOAA, and other operational forecast systems.
"""

import json
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import requests

warnings.filterwarnings("ignore")


class RealBenchmarkAPIResearch:
    """Research and test real benchmark forecast APIs."""

    def __init__(self):
        """Initialize with comprehensive API endpoints for real forecasts."""

        # Real operational forecast APIs
        self.real_apis = {
            "openweathermap_pollution": {
                "name": "OpenWeatherMap Air Pollution API",
                "base_url": "http://api.openweathermap.org/data/2.5/air_pollution",
                "forecast_url": "http://api.openweathermap.org/data/2.5/air_pollution/forecast",
                "requires_key": True,
                "free_tier": "1,000 calls/day",
                "pollutants": ["CO", "NO", "NO2", "O3", "SO2", "PM2.5", "PM10", "NH3"],
                "forecast_horizon": 5,  # days
                "global_coverage": True,
                "quality": "High - Real operational forecasts",
                "update_frequency": "Hourly",
                "data_source": "European weather model integration",
            },
            "waqi_forecast": {
                "name": "World Air Quality Index Forecast",
                "base_url": "https://api.waqi.info/feed/",
                "forecast_url": "https://api.waqi.info/forecast/",
                "requires_key": True,
                "free_tier": "Limited calls/day",
                "pollutants": ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"],
                "forecast_horizon": 3,  # days
                "global_coverage": True,
                "quality": "High - Real station data + forecasts",
                "update_frequency": "Hourly",
                "data_source": "Global monitoring network + forecast models",
            },
            "iqair_forecast": {
                "name": "IQAir AirVisual API",
                "base_url": "https://api.airvisual.com/v2/",
                "forecast_url": "https://api.airvisual.com/v2/city",
                "requires_key": True,
                "free_tier": "10,000 calls/month",
                "pollutants": ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"],
                "forecast_horizon": 1,  # day
                "global_coverage": True,
                "quality": "High - Real-time + 24h forecast",
                "update_frequency": "Real-time",
                "data_source": "Government monitoring stations + forecasts",
            },
            "epa_airnow": {
                "name": "EPA AirNow API",
                "base_url": "https://www.airnowapi.org/aq/",
                "forecast_url": "https://www.airnowapi.org/aq/forecast/",
                "requires_key": True,
                "free_tier": "Available",
                "pollutants": ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"],
                "forecast_horizon": 2,  # days
                "global_coverage": False,  # US only
                "quality": "Excellent - Official EPA forecasts",
                "update_frequency": "Hourly",
                "data_source": "US EPA operational forecast system",
            },
            "copernicus_ads": {
                "name": "Copernicus Atmosphere Data Store",
                "base_url": "https://ads.atmosphere.copernicus.eu/api/v2/",
                "forecast_url": "https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts",
                "requires_key": True,
                "free_tier": "Free registration required",
                "pollutants": ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"],
                "forecast_horizon": 5,  # days
                "global_coverage": True,
                "quality": "Excellent - Official CAMS forecasts",
                "update_frequency": "Daily",
                "data_source": "ECMWF CAMS global forecast system",
            },
            "noaa_air_quality": {
                "name": "NOAA Air Quality Forecast",
                "base_url": "https://www.weather.gov/api/",
                "forecast_url": "https://graphical.weather.gov/xml/",
                "requires_key": False,
                "free_tier": "Free",
                "pollutants": ["PM2.5", "PM10", "O3"],
                "forecast_horizon": 3,  # days
                "global_coverage": False,  # US only
                "quality": "Good - NOAA operational forecasts",
                "update_frequency": "Daily",
                "data_source": "NOAA National Weather Service",
            },
        }

    def test_api_connectivity(self):
        """Test connectivity to real forecast APIs without API keys."""

        print("REAL BENCHMARK API CONNECTIVITY TEST")
        print("=" * 45)

        # Test coordinates for major cities
        test_locations = [
            {"city": "Delhi", "lat": 28.6139, "lon": 77.209},
            {"city": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
            {"city": "London", "lat": 51.5074, "lon": -0.1278},
            {"city": "Beijing", "lat": 39.9042, "lon": 116.4074},
        ]

        api_test_results = {}

        for api_name, api_info in self.real_apis.items():
            print(f"\nTesting {api_info['name']}:")
            print(f"  Requires API Key: {api_info['requires_key']}")
            print(f"  Global Coverage: {api_info['global_coverage']}")
            print(f"  Forecast Horizon: {api_info['forecast_horizon']} days")

            api_test_results[api_name] = {
                "name": api_info["name"],
                "requires_key": api_info["requires_key"],
                "test_results": {},
                "accessible_without_key": False,
            }

            # Test basic connectivity (without API key)
            if api_name == "openweathermap_pollution":
                # Test OpenWeatherMap with demo key
                for location in test_locations[:2]:  # Test 2 cities
                    try:
                        # Current air pollution (free without key sometimes)
                        url = f"{api_info['base_url']}?lat={location['lat']}&lon={location['lon']}&appid=demo"
                        response = requests.get(url, timeout=10)

                        api_test_results[api_name]["test_results"][location["city"]] = {
                            "status_code": response.status_code,
                            "accessible": response.status_code == 200,
                            "response_size": len(response.text) if response.text else 0,
                        }

                        if response.status_code == 200:
                            print(
                                f"    OK {location['city']}: Accessible (Status {response.status_code})"
                            )
                            api_test_results[api_name]["accessible_without_key"] = True
                        else:
                            print(
                                f"    BLOCKED {location['city']}: Blocked (Status {response.status_code})"
                            )

                    except Exception as e:
                        print(f"    ERROR {location['city']}: Error - {str(e)}")
                        api_test_results[api_name]["test_results"][location["city"]] = {
                            "status_code": 0,
                            "accessible": False,
                            "error": str(e),
                        }

                    time.sleep(1)  # Rate limiting

            elif api_name == "noaa_air_quality":
                # Test NOAA (US only, no key required)
                us_locations = [
                    loc for loc in test_locations if loc["city"] in ["Los Angeles"]
                ]
                for location in us_locations:
                    try:
                        # NOAA weather API test
                        url = f"https://api.weather.gov/points/{location['lat']},{location['lon']}"
                        response = requests.get(url, timeout=10)

                        api_test_results[api_name]["test_results"][location["city"]] = {
                            "status_code": response.status_code,
                            "accessible": response.status_code == 200,
                            "response_size": len(response.text) if response.text else 0,
                        }

                        if response.status_code == 200:
                            print(
                                f"    OK {location['city']}: Accessible (Status {response.status_code})"
                            )
                            api_test_results[api_name]["accessible_without_key"] = True
                        else:
                            print(
                                f"    BLOCKED {location['city']}: Blocked (Status {response.status_code})"
                            )

                    except Exception as e:
                        print(f"    ERROR {location['city']}: Error - {str(e)}")
                        api_test_results[api_name]["test_results"][location["city"]] = {
                            "status_code": 0,
                            "accessible": False,
                            "error": str(e),
                        }

                    time.sleep(1)

            else:
                # Other APIs require valid keys
                print(
                    f"    WARNING API Key Required - Cannot test without valid credentials"
                )
                api_test_results[api_name]["test_results"]["key_required"] = True

        return api_test_results

    def research_free_alternatives(self):
        """Research free/demo forecast data alternatives."""

        print(f"\nFREE FORECAST DATA ALTERNATIVES")
        print("=" * 35)

        free_alternatives = {
            "openweather_bulk": {
                "name": "OpenWeatherMap Bulk Download",
                "description": "Historical and forecast data files",
                "url": "https://openweathermap.org/api/statistics-api",
                "access": "Free tier available",
                "data_format": "JSON/CSV bulk files",
                "update_frequency": "Daily",
                "coverage": "Global",
                "quality": "High - same as API data",
            },
            "copernicus_public": {
                "name": "Copernicus Climate Data Store",
                "description": "Public access to CAMS forecast data",
                "url": "https://cds.climate.copernicus.eu/",
                "access": "Free registration required",
                "data_format": "NetCDF/GRIB files",
                "update_frequency": "Daily",
                "coverage": "Global",
                "quality": "Excellent - official CAMS data",
            },
            "epa_data_download": {
                "name": "EPA Air Quality Data Download",
                "description": "Historical and forecast AQI data",
                "url": "https://www.epa.gov/outdoor-air-quality-data",
                "access": "Free public access",
                "data_format": "CSV files",
                "update_frequency": "Daily",
                "coverage": "United States",
                "quality": "Excellent - official EPA data",
            },
            "nasa_giovanni": {
                "name": "NASA Giovanni Air Quality",
                "description": "Satellite-based air quality data",
                "url": "https://giovanni.gsfc.nasa.gov/giovanni/",
                "access": "Free web interface",
                "data_format": "NetCDF/Images",
                "update_frequency": "Daily",
                "coverage": "Global",
                "quality": "High - satellite observations",
            },
        }

        for alt_name, alt_info in free_alternatives.items():
            print(f"\n{alt_info['name']}:")
            print(f"  Description: {alt_info['description']}")
            print(f"  Access: {alt_info['access']}")
            print(f"  Coverage: {alt_info['coverage']}")
            print(f"  Quality: {alt_info['quality']}")

        return free_alternatives

    def create_real_data_collection_strategy(self):
        """Create strategy for collecting real benchmark forecast data."""

        print(f"\nREAL DATA COLLECTION STRATEGY")
        print("=" * 35)

        strategy = {
            "primary_approach": {
                "name": "OpenWeatherMap Air Pollution API",
                "rationale": "Most accessible real forecast API with global coverage",
                "implementation": "Use free tier with careful rate limiting",
                "coverage": "All 100 cities globally",
                "data_quality": "High - real operational forecasts",
                "cost": "Free tier: 1000 calls/day",
            },
            "secondary_approach": {
                "name": "WAQI API with Demo Token",
                "rationale": "Established air quality network with forecast capability",
                "implementation": "Use demo token for testing, request full access",
                "coverage": "Major cities with WAQI stations",
                "data_quality": "High - real monitoring + forecasts",
                "cost": "Demo access available",
            },
            "tertiary_approach": {
                "name": "Public Data Downloads",
                "rationale": "Free access to official forecast data",
                "implementation": "Download bulk forecast files from agencies",
                "coverage": "Regional (US: EPA, Europe: CAMS)",
                "data_quality": "Excellent - official agency data",
                "cost": "Free",
            },
            "validation_approach": {
                "name": "Cross-Reference Multiple Sources",
                "rationale": "Ensure data authenticity and accuracy",
                "implementation": "Compare forecasts across different APIs",
                "coverage": "Subset of cities with multiple sources",
                "data_quality": "Validated through cross-checking",
                "cost": "Uses existing approaches",
            },
        }

        for approach_name, approach_info in strategy.items():
            print(f"\n{approach_name.replace('_', ' ').title()}:")
            print(f"  Method: {approach_info['name']}")
            print(f"  Rationale: {approach_info['rationale']}")
            print(f"  Coverage: {approach_info['coverage']}")
            print(f"  Quality: {approach_info['data_quality']}")

        return strategy

    def save_research_results(self, api_results, alternatives, strategy):
        """Save research results to file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        research_data = {
            "timestamp": timestamp,
            "research_type": "Real Benchmark API Research",
            "api_connectivity_results": api_results,
            "free_alternatives": alternatives,
            "collection_strategy": strategy,
            "summary": {
                "total_apis_researched": len(self.real_apis),
                "apis_accessible_without_key": sum(
                    1
                    for api in api_results.values()
                    if api.get("accessible_without_key", False)
                ),
                "recommended_primary": strategy["primary_approach"]["name"],
                "next_steps": [
                    "Implement OpenWeatherMap Air Pollution API collection",
                    "Test WAQI demo access for additional cities",
                    "Download public forecast data for validation",
                    "Create real data validation framework",
                ],
            },
        }

        output_file = f"../final_dataset/real_benchmark_api_research_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(research_data, f, indent=2, default=str)

        print(f"\nResearch results saved to: {output_file}")
        return output_file


def main():
    """Main research execution."""

    print("REAL BENCHMARK FORECAST API RESEARCH")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 55)

    researcher = RealBenchmarkAPIResearch()

    # Test API connectivity
    api_results = researcher.test_api_connectivity()

    # Research free alternatives
    alternatives = researcher.research_free_alternatives()

    # Create collection strategy
    strategy = researcher.create_real_data_collection_strategy()

    # Save results
    output_file = researcher.save_research_results(api_results, alternatives, strategy)

    print(f"\nRESEARCH SUMMARY:")
    print(f"OK Researched {len(researcher.real_apis)} real forecast APIs")
    print(f"OK Tested connectivity for accessible APIs")
    print(f"OK Identified free data alternatives")
    print(f"OK Created real data collection strategy")
    print(f"OK Next: Implement OpenWeatherMap Air Pollution API collection")

    research_data = {
        "api_results": api_results,
        "alternatives": alternatives,
        "strategy": strategy,
    }

    return research_data, output_file


if __name__ == "__main__":
    results, file_path = main()
