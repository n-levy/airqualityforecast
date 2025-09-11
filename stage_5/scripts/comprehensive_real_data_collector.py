#!/usr/bin/env python3
"""
Comprehensive Real Data Collector

Collect real forecast and air quality data for all 100 cities from available APIs:
- NOAA National Weather Service (US cities)
- WAQI World Air Quality Index (global cities)
- OpenWeatherMap Air Pollution API (if accessible)
"""

import json
import time
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests

warnings.filterwarnings("ignore")


class ComprehensiveRealDataCollector:
    """Collect real data from all available APIs for all 100 cities."""

    def __init__(self):
        """Initialize comprehensive data collector."""

        # Load all 100 cities
        self.cities_df = pd.read_csv(
            "../comprehensive_tables/comprehensive_features_table.csv"
        )

        # Results storage
        self.collection_results = {
            "noaa_data": {},
            "waqi_data": {},
            "openweather_data": {},
            "collection_metadata": {
                "start_time": datetime.now().isoformat(),
                "total_cities": len(self.cities_df),
                "collection_status": "in_progress",
            },
            "success_counts": {
                "noaa_success": 0,
                "waqi_success": 0,
                "openweather_success": 0,
                "total_real_data_cities": 0,
            },
        }

        # Rate limiting settings
        self.rate_limits = {
            "noaa": 0.5,  # NOAA: 0.5 second delay
            "waqi": 1.2,  # WAQI: 1.2 second delay (demo token)
            "openweather": 0.1,  # OpenWeatherMap: 0.1 second delay
        }

    def collect_noaa_weather_data(self):
        """Collect NOAA weather forecast data for US cities."""

        print("COLLECTING NOAA WEATHER DATA FOR US CITIES")
        print("=" * 50)

        us_cities = self.cities_df[self.cities_df["Country"] == "USA"].copy()
        print(f"Found {len(us_cities)} US cities for NOAA data collection")

        for idx, row in us_cities.iterrows():
            city_name = row["City"]
            lat = row["Latitude"]
            lon = row["Longitude"]

            print(f"  Collecting NOAA data for {city_name}...")

            try:
                # Get NOAA grid point
                grid_url = f"https://api.weather.gov/points/{lat},{lon}"
                response = requests.get(grid_url, timeout=10)

                if response.status_code == 200:
                    grid_data = response.json()

                    # Get forecast
                    forecast_url = grid_data["properties"]["forecast"]
                    forecast_response = requests.get(forecast_url, timeout=10)

                    if forecast_response.status_code == 200:
                        forecast_data = forecast_response.json()
                        periods = forecast_data["properties"]["periods"]

                        # Get hourly forecast for more detailed data
                        hourly_url = grid_data["properties"]["forecastHourly"]
                        hourly_response = requests.get(hourly_url, timeout=10)

                        hourly_data = []
                        if hourly_response.status_code == 200:
                            hourly_forecast = hourly_response.json()
                            hourly_data = hourly_forecast["properties"]["periods"][
                                :48
                            ]  # 48 hours

                        # Store comprehensive NOAA data
                        self.collection_results["noaa_data"][city_name] = {
                            "data_source": "NOAA_REAL",
                            "data_type": "REAL_WEATHER_FORECAST",
                            "grid_office": grid_data["properties"]["forecastOffice"],
                            "grid_coordinates": f"{grid_data['properties']['gridX']},{grid_data['properties']['gridY']}",
                            "forecast_periods": periods[:7],  # 7-day forecast
                            "hourly_forecast": hourly_data,
                            "forecast_urls": {
                                "daily": forecast_url,
                                "hourly": hourly_url,
                            },
                            "collection_time": datetime.now().isoformat(),
                            "quality_rating": "EXCELLENT",
                            "api_status": "SUCCESS",
                        }

                        self.collection_results["success_counts"]["noaa_success"] += 1
                        print(
                            f"    SUCCESS: {len(periods)} daily + {len(hourly_data)} hourly periods"
                        )

                    else:
                        print(
                            f"    FORECAST FAILED: Status {forecast_response.status_code}"
                        )
                        self.collection_results["noaa_data"][city_name] = {
                            "data_source": "NOAA_REAL",
                            "api_status": "FORECAST_FAILED",
                            "error_code": forecast_response.status_code,
                        }

                else:
                    print(f"    GRID FAILED: Status {response.status_code}")
                    self.collection_results["noaa_data"][city_name] = {
                        "data_source": "NOAA_REAL",
                        "api_status": "GRID_FAILED",
                        "error_code": response.status_code,
                    }

            except Exception as e:
                print(f"    ERROR: {str(e)}")
                self.collection_results["noaa_data"][city_name] = {
                    "data_source": "NOAA_REAL",
                    "api_status": "ERROR",
                    "error_message": str(e),
                }

            time.sleep(self.rate_limits["noaa"])

        print(
            f"\nNOAA Collection Complete: {self.collection_results['success_counts']['noaa_success']}/{len(us_cities)} cities"
        )

    def collect_waqi_air_quality_data(self):
        """Collect WAQI air quality data for all cities."""

        print(f"\nCOLLECTING WAQI AIR QUALITY DATA FOR ALL CITIES")
        print("=" * 55)

        print(f"Attempting WAQI collection for all {len(self.cities_df)} cities")

        for idx, row in self.cities_df.iterrows():
            city_name = row["City"]
            country = row["Country"]

            print(f"  Collecting WAQI data for {city_name}, {country}...")

            try:
                # Try multiple query formats for better success rate
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
                    response = requests.get(url, timeout=8)

                    if response.status_code == 200:
                        data = response.json()

                        if data.get("status") == "ok" and "data" in data:
                            aqi_data = data["data"]

                            # Extract comprehensive pollutant data
                            pollutants = {}
                            if "iaqi" in aqi_data:
                                for pollutant, values in aqi_data["iaqi"].items():
                                    if isinstance(values, dict) and "v" in values:
                                        pollutants[pollutant] = values["v"]

                            # Extract forecast data if available
                            forecast_data = []
                            if "forecast" in aqi_data:
                                forecast_data = aqi_data["forecast"]

                            self.collection_results["waqi_data"][city_name] = {
                                "data_source": "WAQI_REAL",
                                "data_type": "REAL_AIR_QUALITY",
                                "current_aqi": aqi_data.get("aqi", -1),
                                "pollutants": pollutants,
                                "station_info": {
                                    "name": aqi_data.get("city", {}).get(
                                        "name", city_name
                                    ),
                                    "coordinates": aqi_data.get("city", {}).get(
                                        "geo", []
                                    ),
                                    "url": aqi_data.get("city", {}).get("url", ""),
                                },
                                "measurement_time": aqi_data.get("time", {}).get(
                                    "s", "unknown"
                                ),
                                "forecast_data": forecast_data,
                                "attributions": aqi_data.get("attributions", []),
                                "collection_time": datetime.now().isoformat(),
                                "quality_rating": "HIGH",
                                "api_status": "SUCCESS",
                                "query_used": query,
                            }

                            self.collection_results["success_counts"][
                                "waqi_success"
                            ] += 1
                            waqi_success = True
                            print(
                                f"    SUCCESS: AQI={aqi_data.get('aqi', 'N/A')}, Pollutants={len(pollutants)}"
                            )

                if not waqi_success:
                    print(f"    NO DATA: All query formats failed")
                    self.collection_results["waqi_data"][city_name] = {
                        "data_source": "WAQI_REAL",
                        "api_status": "NO_DATA",
                        "queries_attempted": city_queries,
                    }

            except Exception as e:
                print(f"    ERROR: {str(e)}")
                self.collection_results["waqi_data"][city_name] = {
                    "data_source": "WAQI_REAL",
                    "api_status": "ERROR",
                    "error_message": str(e),
                }

            time.sleep(self.rate_limits["waqi"])

        print(
            f"\nWAQI Collection Complete: {self.collection_results['success_counts']['waqi_success']}/{len(self.cities_df)} cities"
        )

    def test_openweathermap_access(self):
        """Test OpenWeatherMap API access (requires API key)."""

        print(f"\nTESTING OPENWEATHERMAP API ACCESS")
        print("=" * 40)

        # Test with a sample city to see if we have access
        sample_city = self.cities_df.iloc[0]
        lat = sample_city["Latitude"]
        lon = sample_city["Longitude"]

        print(f"Testing OpenWeatherMap access with {sample_city['City']}...")

        try:
            # Test air pollution API without key first
            url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid=demo"
            response = requests.get(url, timeout=5)

            if response.status_code == 401:
                print("  BLOCKED: Requires API key (401 Unauthorized)")
                self.collection_results["openweather_data"][
                    "api_status"
                ] = "REQUIRES_API_KEY"
                self.collection_results["openweather_data"][
                    "message"
                ] = "OpenWeatherMap requires paid API key"

            elif response.status_code == 200:
                print("  UNEXPECTED SUCCESS: Demo access worked!")
                # If demo access works, collect for all cities
                self.collect_openweathermap_data()

            else:
                print(f"  OTHER ERROR: Status {response.status_code}")
                self.collection_results["openweather_data"]["api_status"] = "ERROR"
                self.collection_results["openweather_data"][
                    "error_code"
                ] = response.status_code

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            self.collection_results["openweather_data"]["api_status"] = "ERROR"
            self.collection_results["openweather_data"]["error_message"] = str(e)

    def collect_openweathermap_data(self):
        """Collect OpenWeatherMap data if access is available."""

        print("COLLECTING OPENWEATHERMAP DATA FOR ALL CITIES")
        print("=" * 50)

        for idx, row in self.cities_df.iterrows():
            city_name = row["City"]
            lat = row["Latitude"]
            lon = row["Longitude"]

            print(f"  Collecting OpenWeatherMap data for {city_name}...")

            try:
                # Current air pollution
                current_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid=demo"
                current_response = requests.get(current_url, timeout=8)

                # Forecast air pollution
                forecast_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid=demo"
                forecast_response = requests.get(forecast_url, timeout=8)

                if current_response.status_code == 200:
                    current_data = current_response.json()

                    forecast_data = []
                    if forecast_response.status_code == 200:
                        forecast_data = forecast_response.json()

                    self.collection_results["openweather_data"][city_name] = {
                        "data_source": "OPENWEATHERMAP_REAL",
                        "data_type": "REAL_AIR_POLLUTION",
                        "current_pollution": current_data,
                        "forecast_pollution": forecast_data,
                        "collection_time": datetime.now().isoformat(),
                        "quality_rating": "HIGH",
                        "api_status": "SUCCESS",
                    }

                    self.collection_results["success_counts"][
                        "openweather_success"
                    ] += 1
                    print(f"    SUCCESS: Current + Forecast data collected")

                else:
                    print(f"    FAILED: Status {current_response.status_code}")

            except Exception as e:
                print(f"    ERROR: {str(e)}")

            time.sleep(self.rate_limits["openweather"])

    def calculate_collection_statistics(self):
        """Calculate comprehensive collection statistics."""

        print(f"\nCALCULATING COLLECTION STATISTICS")
        print("=" * 40)

        # Count successful collections
        noaa_success = self.collection_results["success_counts"]["noaa_success"]
        waqi_success = self.collection_results["success_counts"]["waqi_success"]
        openweather_success = self.collection_results["success_counts"][
            "openweather_success"
        ]

        # Count cities with any real data
        cities_with_real_data = set()
        cities_with_real_data.update(
            [
                city
                for city, data in self.collection_results["noaa_data"].items()
                if data.get("api_status") == "SUCCESS"
            ]
        )
        cities_with_real_data.update(
            [
                city
                for city, data in self.collection_results["waqi_data"].items()
                if data.get("api_status") == "SUCCESS"
            ]
        )
        cities_with_real_data.update(
            [
                city
                for city, data in self.collection_results["openweather_data"].items()
                if data.get("api_status") == "SUCCESS"
            ]
        )

        total_cities = len(self.cities_df)
        real_data_percentage = (len(cities_with_real_data) / total_cities) * 100

        # Update collection metadata
        self.collection_results["collection_metadata"].update(
            {
                "end_time": datetime.now().isoformat(),
                "collection_status": "completed",
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
        )

        # Update success counts
        self.collection_results["success_counts"]["total_real_data_cities"] = len(
            cities_with_real_data
        )

        # Print summary
        print(f"Collection Statistics:")
        print(f"  Total cities: {total_cities}")
        print(f"  NOAA successful: {noaa_success}")
        print(f"  WAQI successful: {waqi_success}")
        print(f"  OpenWeatherMap successful: {openweather_success}")
        print(
            f"  Cities with any real data: {len(cities_with_real_data)} ({real_data_percentage:.1f}%)"
        )
        print(
            f"  Cities needing synthetic data: {total_cities - len(cities_with_real_data)}"
        )

    def save_collection_results(self):
        """Save comprehensive collection results."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            f"../final_dataset/comprehensive_real_data_collection_{timestamp}.json"
        )

        with open(output_file, "w") as f:
            json.dump(self.collection_results, f, indent=2, default=str)

        print(f"\nCollection results saved to: {output_file}")
        return output_file


def main():
    """Main comprehensive data collection."""

    print("COMPREHENSIVE REAL DATA COLLECTION FOR ALL 100 CITIES")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)

    collector = ComprehensiveRealDataCollector()

    # Collect NOAA data for US cities
    collector.collect_noaa_weather_data()

    # Collect WAQI data for all cities
    collector.collect_waqi_air_quality_data()

    # Test OpenWeatherMap access
    collector.test_openweathermap_access()

    # Calculate statistics
    collector.calculate_collection_statistics()

    # Save results
    output_file = collector.save_collection_results()

    # Final summary
    stats = collector.collection_results["collection_metadata"]["statistics"]
    print(f"\nFINAL COLLECTION SUMMARY:")
    print(
        f"Real data collected for {stats['cities_with_any_real_data']}/{stats['total_cities']} cities ({stats['real_data_percentage']:.1f}%)"
    )
    print(f"NOAA weather data: {stats['noaa_successful']} US cities")
    print(f"WAQI air quality data: {stats['waqi_successful']} global cities")
    print(f"OpenWeatherMap data: {stats['openweather_successful']} cities")

    return collector.collection_results, output_file


if __name__ == "__main__":
    results, file_path = main()
