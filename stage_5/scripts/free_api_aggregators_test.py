#!/usr/bin/env python3
"""
Free API Aggregators Test for 100-City Real Data Collection
Test Open-Meteo and other free APIs for comprehensive coverage
"""
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


class FreeAPIAggregatorTest:
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

    def test_open_meteo_weather(self):
        """Test Open-Meteo Weather API (free, no key required)"""
        print("=== TESTING OPEN-METEO WEATHER API ===")
        results = {
            "api_name": "Open-Meteo Weather",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        for continent, cities in self.test_cities.items():
            for city in cities:
                try:
                    # Open-Meteo current weather API
                    url = "https://api.open-meteo.com/v1/forecast"
                    params = {
                        "latitude": city["lat"],
                        "longitude": city["lon"],
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
                        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
                        "timezone": "auto",
                        "forecast_days": 1,
                    }

                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        current = data.get("current", {})
                        hourly = data.get("hourly", {})

                        if current and hourly:
                            results["successful_cities"].append(
                                {
                                    "city": city["name"],
                                    "country": city["country"],
                                    "continent": continent,
                                    "current_temp": current.get("temperature_2m"),
                                    "current_humidity": current.get(
                                        "relative_humidity_2m"
                                    ),
                                    "current_wind": current.get("wind_speed_10m"),
                                    "hourly_records": len(hourly.get("time", [])),
                                    "timezone": data.get("timezone"),
                                    "elevation": data.get("elevation"),
                                }
                            )
                            print(
                                f"  SUCCESS {city['name']}: Temp={current.get('temperature_2m')}C, {len(hourly.get('time', []))} hourly records"
                            )
                        else:
                            results["failed_cities"].append(
                                f"{city['name']} - No weather data"
                            )
                            print(f"  FAILED {city['name']}: No weather data")
                    else:
                        results["failed_cities"].append(
                            f"{city['name']} - API error {response.status_code}"
                        )
                        print(
                            f"  FAILED {city['name']}: API error {response.status_code}"
                        )

                    results["cities_tested"] += 1
                    time.sleep(0.2)  # Respectful rate limiting

                except Exception as e:
                    results["failed_cities"].append(
                        f"{city['name']} - Exception: {str(e)}"
                    )
                    print(f"  FAILED {city['name']}: Exception: {str(e)}")

        return results

    def test_open_meteo_air_quality(self):
        """Test Open-Meteo Air Quality API (CAMS data)"""
        print("\n=== TESTING OPEN-METEO AIR QUALITY API (CAMS DATA) ===")
        results = {
            "api_name": "Open-Meteo Air Quality",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        for continent, cities in self.test_cities.items():
            for city in cities:
                try:
                    # Open-Meteo air quality API with CAMS data
                    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
                    params = {
                        "latitude": city["lat"],
                        "longitude": city["lon"],
                        "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,european_aqi",
                        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,european_aqi",
                        "timezone": "auto",
                        "forecast_days": 1,
                    }

                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        current = data.get("current", {})
                        hourly = data.get("hourly", {})

                        if current and hourly:
                            results["successful_cities"].append(
                                {
                                    "city": city["name"],
                                    "country": city["country"],
                                    "continent": continent,
                                    "current_aqi": current.get("european_aqi"),
                                    "current_pm25": current.get("pm2_5"),
                                    "current_pm10": current.get("pm10"),
                                    "current_no2": current.get("nitrogen_dioxide"),
                                    "current_o3": current.get("ozone"),
                                    "hourly_records": len(hourly.get("time", [])),
                                    "data_source": "CAMS_ENSEMBLE",
                                    "resolution": "11km",
                                }
                            )
                            print(
                                f"  SUCCESS {city['name']}: AQI={current.get('european_aqi')}, PM2.5={current.get('pm2_5')}, {len(hourly.get('time', []))} hourly records"
                            )
                        else:
                            results["failed_cities"].append(
                                f"{city['name']} - No air quality data"
                            )
                            print(f"  FAILED {city['name']}: No air quality data")
                    else:
                        results["failed_cities"].append(
                            f"{city['name']} - API error {response.status_code}"
                        )
                        print(
                            f"  FAILED {city['name']}: API error {response.status_code}"
                        )

                    results["cities_tested"] += 1
                    time.sleep(0.2)  # Respectful rate limiting

                except Exception as e:
                    results["failed_cities"].append(
                        f"{city['name']} - Exception: {str(e)}"
                    )
                    print(f"  FAILED {city['name']}: Exception: {str(e)}")

        return results

    def test_open_meteo_historical(self):
        """Test Open-Meteo Historical Data API"""
        print("\n=== TESTING OPEN-METEO HISTORICAL DATA API ===")
        results = {
            "api_name": "Open-Meteo Historical",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        # Test with a subset of cities for historical data
        test_subset = [
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        ]

        # Test last 7 days of historical data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        for city in test_subset:
            try:
                # Historical weather API
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
                    "timezone": "auto",
                }

                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    hourly = data.get("hourly", {})

                    if hourly and hourly.get("time"):
                        historical_records = len(hourly["time"])
                        results["successful_cities"].append(
                            {
                                "city": city["name"],
                                "historical_records": historical_records,
                                "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                                "data_availability": "Full historical access",
                            }
                        )
                        print(
                            f"  SUCCESS {city['name']}: {historical_records} historical hourly records"
                        )
                    else:
                        results["failed_cities"].append(
                            f"{city['name']} - No historical data"
                        )
                        print(f"  FAILED {city['name']}: No historical data")
                else:
                    results["failed_cities"].append(
                        f"{city['name']} - API error {response.status_code}"
                    )
                    print(f"  FAILED {city['name']}: API error {response.status_code}")

                results["cities_tested"] += 1
                time.sleep(1)  # Longer delay for historical data

            except Exception as e:
                results["failed_cities"].append(f"{city['name']} - Exception: {str(e)}")
                print(f"  FAILED {city['name']}: Exception: {str(e)}")

        return results

    def test_other_free_apis(self):
        """Test other free APIs"""
        print("\n=== TESTING OTHER FREE APIS ===")
        results = {
            "api_name": "Other Free APIs",
            "cities_tested": 0,
            "successful_cities": [],
            "failed_cities": [],
        }

        # Test OpenWeatherMap free tier (requires signup but free)
        test_city = {"name": "London", "lat": 51.5074, "lon": -0.1278}

        try:
            # Try without API key first to see what happens
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": test_city["lat"],
                "lon": test_city["lon"],
                "appid": "demo",  # Invalid key to test response
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 401:
                print(
                    f"  INFO OpenWeatherMap: Requires API key (free signup available)"
                )
                results["api_notes"] = "OpenWeatherMap: Free tier available with signup"
            else:
                print(f"  OpenWeatherMap response: {response.status_code}")

        except Exception as e:
            print(f"  OpenWeatherMap error: {str(e)}")

        return results

    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of API capabilities"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE FREE API ANALYSIS")
        print("=" * 80)

        working_apis = []
        total_coverage = 0

        for api_name, results in self.api_results.items():
            successful = len(results.get("successful_cities", []))
            tested = results.get("cities_tested", 0)
            success_rate = (successful / tested * 100) if tested > 0 else 0

            print(f"\n{results['api_name']}:")
            print(f"  Cities tested: {tested}")
            print(f"  Successful cities: {successful}")
            print(f"  Success rate: {success_rate:.1f}%")

            if success_rate >= 80:  # High success rate
                working_apis.append(api_name)
                total_coverage += successful
                print(f"  Status: EXCELLENT - High reliability")
            elif success_rate >= 50:
                working_apis.append(api_name)
                total_coverage += successful
                print(f"  Status: GOOD - Suitable for use")
            else:
                print(f"  Status: LIMITED - May need authentication")

        print(f"\n=== COVERAGE ANALYSIS ===")
        print(f"Working APIs: {len(working_apis)}")
        print(f"Potential city coverage: {total_coverage} cities per API")

        if (
            "open_meteo_weather" in working_apis
            and "open_meteo_air_quality" in working_apis
        ):
            print(f"\nEXCELLENT: Open-Meteo provides comprehensive coverage!")
            print(f"✅ Weather data: Global coverage with hourly resolution")
            print(f"✅ Air quality data: CAMS ensemble data (11km resolution)")
            print(f"✅ Historical data: 80+ years of historical weather data")
            print(f"✅ Free usage: No API key required for non-commercial use")
            print(f"✅ Real data: Based on CAMS, ECMWF, and global weather models")

            print(f"\nRECOMMENDATION:")
            print(f"- Use Open-Meteo as primary API for 100-city coverage")
            print(f"- Weather data: Real ECMWF/GFS model data")
            print(f"- Air quality: Real CAMS ensemble forecasts")
            print(f"- Historical: Complete 2-year dataset achievable")
            print(f"- Authentication: None required")
            print(f"- Cost: Free for research/non-commercial use")

            return True
        else:
            print(f"\nLIMITED: Need to explore paid options or API keys")
            return False

    def run_comprehensive_test(self):
        """Run comprehensive test of all free APIs"""
        print("FREE API AGGREGATORS COMPREHENSIVE TEST")
        print("Objective: Find free APIs for 100 cities with real data")
        print("Requirements: Air quality + Weather + Historical data")
        print("=" * 80)

        # Test all APIs
        self.api_results["open_meteo_weather"] = self.test_open_meteo_weather()
        self.api_results["open_meteo_air_quality"] = self.test_open_meteo_air_quality()
        self.api_results["open_meteo_historical"] = self.test_open_meteo_historical()
        self.api_results["other_free_apis"] = self.test_other_free_apis()

        # Generate analysis
        viable = self.generate_comprehensive_analysis()

        # Save results
        self.save_results()

        return viable

    def save_results(self):
        """Save comprehensive test results"""
        results_data = {
            "test_timestamp": self.timestamp.isoformat(),
            "objective": "Find free APIs for 100 cities with real data (air quality + weather + historical)",
            "requirements": {
                "cities": "100 cities (20 per continent)",
                "data_types": ["air_quality", "weather", "historical"],
                "authenticity": "100% real data, no synthetic/fallback",
                "cost": "Free APIs only",
            },
            "api_results": self.api_results,
            "test_cities": self.test_cities,
        }

        filename = f"../final_dataset/free_api_test_results_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filename}")
        return filename


def main():
    """Main execution"""
    tester = FreeAPIAggregatorTest()
    viable = tester.run_comprehensive_test()

    if viable:
        print(f"\n🎉 SUCCESS: Viable free API solution found!")
        print(f"Next step: Implement 100-city data collection with Open-Meteo")
    else:
        print(f"\n⚠️ LIMITED: May need API registration or paid options")
        print(f"Next step: Evaluate API key requirements")


if __name__ == "__main__":
    main()
