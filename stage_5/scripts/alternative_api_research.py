#!/usr/bin/env python3
"""
Alternative Air Quality API Research
Find APIs that provide real city-specific data for 100 cities (20 per continent)
Acceptance criteria: unique real values, no fallbacks, no synthetic data
"""
import json
import time
import warnings
from datetime import datetime

import pandas as pd
import requests

warnings.filterwarnings("ignore")


class AlternativeAPIResearch:
    def __init__(self):
        self.timestamp = datetime.now()
        self.test_cities = {
            "Asia": [
                {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
                {"name": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090},
                {
                    "name": "Shanghai",
                    "country": "China",
                    "lat": 31.2304,
                    "lon": 121.4737,
                },
                {
                    "name": "Bangkok",
                    "country": "Thailand",
                    "lat": 13.7563,
                    "lon": 100.5018,
                },
            ],
            "Europe": [
                {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
                {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
                {
                    "name": "Berlin",
                    "country": "Germany",
                    "lat": 52.5200,
                    "lon": 13.4050,
                },
                {"name": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038},
            ],
            "North America": [
                {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
                {
                    "name": "Los Angeles",
                    "country": "USA",
                    "lat": 34.0522,
                    "lon": -118.2437,
                },
                {"name": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298},
                {
                    "name": "Toronto",
                    "country": "Canada",
                    "lat": 43.6532,
                    "lon": -79.3832,
                },
            ],
            "South America": [
                {
                    "name": "São Paulo",
                    "country": "Brazil",
                    "lat": -23.5558,
                    "lon": -46.6396,
                },
                {
                    "name": "Buenos Aires",
                    "country": "Argentina",
                    "lat": -34.6118,
                    "lon": -58.3960,
                },
                {"name": "Lima", "country": "Peru", "lat": -12.0464, "lon": -77.0428},
                {
                    "name": "Bogotá",
                    "country": "Colombia",
                    "lat": 4.7110,
                    "lon": -74.0721,
                },
            ],
            "Africa": [
                {"name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
                {"name": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792},
                {
                    "name": "Johannesburg",
                    "country": "South Africa",
                    "lat": -26.2041,
                    "lon": 28.0473,
                },
                {"name": "Nairobi", "country": "Kenya", "lat": -1.2921, "lon": 36.8219},
            ],
        }

        self.api_results = {}

    def test_openaq_api(self):
        """Test OpenAQ API for city-specific data"""
        print("=== TESTING OPENAQ API ===")
        results = {
            "api_name": "OpenAQ",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        for continent, cities in self.test_cities.items():
            for city in cities:
                try:
                    # OpenAQ locations endpoint
                    url = f"https://api.openaq.org/v2/locations"
                    params = {
                        "limit": 100,
                        "city": city["name"],
                        "country": city["country"][:2].upper(),  # ISO code
                    }

                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        locations = data.get("results", [])

                        if locations:
                            # Get latest measurements
                            location_id = locations[0]["id"]
                            measurements_url = f"https://api.openaq.org/v2/measurements"
                            measurements_params = {
                                "location_id": location_id,
                                "limit": 1,
                                "order_by": "datetime",
                                "sort": "desc",
                            }

                            measurements_response = requests.get(
                                measurements_url, params=measurements_params, timeout=10
                            )
                            if measurements_response.status_code == 200:
                                measurements_data = measurements_response.json()
                                measurements = measurements_data.get("results", [])

                                if measurements:
                                    measurement = measurements[0]
                                    results["successful_cities"].append(
                                        {
                                            "city": city["name"],
                                            "country": city["country"],
                                            "continent": continent,
                                            "location_id": location_id,
                                            "parameter": measurement.get("parameter"),
                                            "value": measurement.get("value"),
                                            "unit": measurement.get("unit"),
                                            "date": measurement.get("date", {}).get(
                                                "utc"
                                            ),
                                            "coordinates": measurement.get(
                                                "coordinates"
                                            ),
                                        }
                                    )
                                    print(
                                        f"  SUCCESS {city['name']}: {measurement.get('parameter')}={measurement.get('value')} {measurement.get('unit')}"
                                    )
                                else:
                                    results["failed_cities"].append(
                                        f"{city['name']} - No measurements"
                                    )
                                    print(
                                        f"  FAILED {city['name']}: No measurements available"
                                    )
                            else:
                                results["failed_cities"].append(
                                    f"{city['name']} - Measurements API error"
                                )
                                print(
                                    f"  FAILED {city['name']}: Measurements API error"
                                )
                        else:
                            results["failed_cities"].append(
                                f"{city['name']} - No locations found"
                            )
                            print(f"  FAILED {city['name']}: No locations found")
                    else:
                        results["failed_cities"].append(
                            f"{city['name']} - API error {response.status_code}"
                        )
                        print(
                            f"  FAILED {city['name']}: API error {response.status_code}"
                        )

                    results["cities_tested"] += 1
                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    results["failed_cities"].append(
                        f"{city['name']} - Exception: {str(e)}"
                    )
                    print(f"  FAILED {city['name']}: Exception: {str(e)}")

        return results

    def test_airnow_api(self):
        """Test AirNow API for US cities"""
        print("\n=== TESTING AIRNOW API ===")
        results = {
            "api_name": "AirNow",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        # Test only US cities
        us_cities = self.test_cities["North America"]
        for city in us_cities:
            if city["country"] == "USA":
                try:
                    # AirNow current observations by lat/lon
                    url = "https://www.airnowapi.org/aq/observation/latLong/current/"
                    params = {
                        "format": "application/json",
                        "latitude": city["lat"],
                        "longitude": city["lon"],
                        "distance": 25,
                        "API_KEY": "guest",  # Guest key for testing
                    }

                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            observation = data[0]
                            results["successful_cities"].append(
                                {
                                    "city": city["name"],
                                    "country": city["country"],
                                    "continent": "North America",
                                    "parameter": observation.get("ParameterName"),
                                    "aqi": observation.get("AQI"),
                                    "category_name": observation.get(
                                        "Category", {}
                                    ).get("Name"),
                                    "reporting_area": observation.get("ReportingArea"),
                                    "date": observation.get("DateObserved"),
                                    "hour": observation.get("HourObserved"),
                                }
                            )
                            print(
                                f"  SUCCESS {city['name']}: {observation.get('ParameterName')} AQI={observation.get('AQI')} ({observation.get('ReportingArea')})"
                            )
                        else:
                            results["failed_cities"].append(
                                f"{city['name']} - No data available"
                            )
                            print(f"  FAILED {city['name']}: No data available")
                    else:
                        results["failed_cities"].append(
                            f"{city['name']} - API error {response.status_code}"
                        )
                        print(
                            f"  FAILED {city['name']}: API error {response.status_code}"
                        )

                    results["cities_tested"] += 1
                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    results["failed_cities"].append(
                        f"{city['name']} - Exception: {str(e)}"
                    )
                    print(f"  FAILED {city['name']}: Exception: {str(e)}")

        return results

    def test_european_api(self):
        """Test European Environment Agency API"""
        print("\n=== TESTING EUROPEAN ENVIRONMENT AGENCY API ===")
        results = {
            "api_name": "EEA",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        # Test European cities
        eu_cities = self.test_cities["Europe"]
        for city in eu_cities:
            try:
                # European Air Quality Index API
                url = "https://discomap.eea.europa.eu/map/fme/AirQualityExport.fmw"
                params = {
                    "CountryCode": city["country"][:2].upper(),
                    "CityName": city["name"],
                    "Pollutant": "PM2.5",
                    "Year": "2024",
                    "Output": "JSON",
                }

                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data and len(data) > 0:
                            station = data[0]
                            results["successful_cities"].append(
                                {
                                    "city": city["name"],
                                    "country": city["country"],
                                    "continent": "Europe",
                                    "station_name": station.get("StationName"),
                                    "pollutant": station.get("Pollutant"),
                                    "value": station.get("Concentration"),
                                    "unit": station.get("Unit"),
                                    "assessment_type": station.get("AssessmentType"),
                                }
                            )
                            print(
                                f"  ✅ {city['name']}: {station.get('Pollutant')}={station.get('Concentration')} {station.get('Unit')} ({station.get('StationName')})"
                            )
                        else:
                            results["failed_cities"].append(
                                f"{city['name']} - No data available"
                            )
                            print(f"  ❌ {city['name']}: No data available")
                    except json.JSONDecodeError:
                        results["failed_cities"].append(
                            f"{city['name']} - Invalid JSON response"
                        )
                        print(f"  ❌ {city['name']}: Invalid JSON response")
                else:
                    results["failed_cities"].append(
                        f"{city['name']} - API error {response.status_code}"
                    )
                    print(f"  ❌ {city['name']}: API error {response.status_code}")

                results["cities_tested"] += 1
                time.sleep(1)  # Rate limiting

            except Exception as e:
                results["failed_cities"].append(f"{city['name']} - Exception: {str(e)}")
                print(f"  ❌ {city['name']}: Exception: {str(e)}")

        return results

    def test_iqair_api(self):
        """Test IQAir API (alternative to WAQI)"""
        print("\n=== TESTING IQAIR API ===")
        results = {
            "api_name": "IQAir",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        # Test sample cities
        test_sample = []
        for continent, cities in self.test_cities.items():
            test_sample.extend(cities[:2])  # 2 cities per continent

        for city in test_sample:
            try:
                # IQAir nearest city API
                url = "https://api.airvisual.com/v2/nearest_city"
                params = {
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "key": "demo",  # Demo key
                }

                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        city_data = data["data"]
                        current = city_data.get("current", {})
                        pollution = current.get("pollution", {})

                        results["successful_cities"].append(
                            {
                                "requested_city": city["name"],
                                "actual_city": city_data.get("city"),
                                "country": city_data.get("country"),
                                "state": city_data.get("state"),
                                "aqi_us": pollution.get("aqius"),
                                "main_pollutant_us": pollution.get("mainus"),
                                "aqi_cn": pollution.get("aqicn"),
                                "main_pollutant_cn": pollution.get("maincn"),
                                "timestamp": pollution.get("ts"),
                            }
                        )
                        print(
                            f"  ✅ {city['name']} -> {city_data.get('city')}: AQI_US={pollution.get('aqius')} ({pollution.get('mainus')})"
                        )
                    else:
                        results["failed_cities"].append(
                            f"{city['name']} - API returned error"
                        )
                        print(f"  ❌ {city['name']}: API returned error")
                else:
                    results["failed_cities"].append(
                        f"{city['name']} - API error {response.status_code}"
                    )
                    print(f"  ❌ {city['name']}: API error {response.status_code}")

                results["cities_tested"] += 1
                time.sleep(1)  # Rate limiting

            except Exception as e:
                results["failed_cities"].append(f"{city['name']} - Exception: {str(e)}")
                print(f"  ❌ {city['name']}: Exception: {str(e)}")

        return results

    def run_comprehensive_research(self):
        """Run comprehensive API research"""
        print("ALTERNATIVE AIR QUALITY API RESEARCH")
        print("Objective: Find APIs with real city-specific data for 100 cities")
        print(
            "Acceptance criteria: 20 cities per continent, unique real values, no fallbacks"
        )
        print("=" * 80)

        # Test all APIs
        self.api_results["openaq"] = self.test_openaq_api()
        self.api_results["airnow"] = self.test_airnow_api()
        self.api_results["eea"] = self.test_european_api()
        self.api_results["iqair"] = self.test_iqair_api()

        # Generate summary
        self.generate_summary()

        # Save results
        self.save_results()

    def generate_summary(self):
        """Generate comprehensive summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE API RESEARCH SUMMARY")
        print("=" * 80)

        total_apis = len(self.api_results)
        viable_apis = []

        for api_name, results in self.api_results.items():
            successful = len(results["successful_cities"])
            tested = results["cities_tested"]
            success_rate = (successful / tested * 100) if tested > 0 else 0

            print(f"\n{results['api_name']} API:")
            print(f"  Cities tested: {tested}")
            print(f"  Successful cities: {successful}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Failed cities: {len(results['failed_cities'])}")

            if success_rate >= 50:  # Consider viable if 50%+ success rate
                viable_apis.append(api_name)
                print(f"  Status: ✅ VIABLE for multi-API approach")
            else:
                print(f"  Status: ❌ Not suitable for 100-city coverage")

        print(f"\nVIABLE APIs: {len(viable_apis)}/{total_apis}")
        print(f"APIs with good coverage: {', '.join(viable_apis)}")

        # Calculate potential coverage
        print(f"\n=== MULTI-API COVERAGE ANALYSIS ===")
        if len(viable_apis) >= 2:
            print("✅ Multi-API approach feasible")
            print("Recommended strategy:")
            print("  - OpenAQ: Global coverage, good for diverse cities")
            print("  - AirNow: US cities with high reliability")
            print("  - EEA: European cities with official data")
            print("  - IQAir: Backup for missing cities")
            print("\nPotential to achieve 100 cities with real data: HIGH")
        else:
            print("❌ Multi-API approach may be challenging")
            print("Consider focusing on cities with verified data availability")

    def save_results(self):
        """Save research results"""
        results_data = {
            "research_timestamp": self.timestamp.isoformat(),
            "objective": "Find APIs with real city-specific data for 100 cities (20 per continent)",
            "acceptance_criteria": "Unique real values, no fallbacks, no synthetic data",
            "test_cities": self.test_cities,
            "api_results": self.api_results,
            "summary": {
                "total_apis_tested": len(self.api_results),
                "viable_apis": [
                    api
                    for api, results in self.api_results.items()
                    if len(results["successful_cities"])
                    / max(results["cities_tested"], 1)
                    >= 0.5
                ],
                "recommendation": "Multi-API approach combining OpenAQ, AirNow, and EEA",
            },
        }

        filename = f"../final_dataset/alternative_api_research_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filename}")
        return filename


def main():
    """Main execution"""
    researcher = AlternativeAPIResearch()
    researcher.run_comprehensive_research()
    print("\n✅ ALTERNATIVE API RESEARCH COMPLETE!")


if __name__ == "__main__":
    main()
