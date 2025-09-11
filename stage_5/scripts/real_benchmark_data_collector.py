#!/usr/bin/env python3
"""
Real Benchmark Data Collector

Collect real forecast data from accessible APIs identified in research phase.
Focus on NOAA (US cities) and publicly available forecast data sources.
"""

import json
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


class RealBenchmarkDataCollector:
    """Collect real benchmark forecast data from accessible APIs."""

    def __init__(self):
        """Initialize with accessible API configurations."""

        # Load city information
        self.cities_df = pd.read_csv(
            "../comprehensive_tables/comprehensive_features_table.csv"
        )

        # Accessible APIs from research
        self.accessible_apis = {
            "noaa_weather": {
                "name": "NOAA National Weather Service API",
                "base_url": "https://api.weather.gov",
                "requires_key": False,
                "coverage": "United States only",
                "data_available": ["weather_forecast", "air_quality_index"],
                "update_frequency": "Hourly",
                "quality": "Excellent - Official US government data",
            },
            "waqi_demo": {
                "name": "World Air Quality Index (Demo)",
                "base_url": "https://api.waqi.info",
                "requires_key": False,  # Demo token sometimes works
                "coverage": "Global cities with monitoring stations",
                "data_available": ["current_aqi", "limited_forecast"],
                "update_frequency": "Hourly",
                "quality": "High - Real monitoring station data",
            },
        }

        # US cities that can use NOAA API
        self.us_cities = self.cities_df[self.cities_df["Country"] == "USA"].copy()

        # Global cities for WAQI demo attempts
        self.global_cities = self.cities_df.copy()

    def collect_noaa_forecast_data(self):
        """Collect real forecast data from NOAA API for US cities."""

        print("COLLECTING NOAA REAL FORECAST DATA")
        print("=" * 38)

        noaa_results = {}

        for idx, row in self.us_cities.iterrows():
            city_name = row["City"]
            lat = row["Latitude"]
            lon = row["Longitude"]

            print(f"Processing {city_name} ({lat}, {lon})...")

            try:
                # Get NOAA grid point
                grid_url = f"https://api.weather.gov/points/{lat},{lon}"
                grid_response = requests.get(grid_url, timeout=10)

                if grid_response.status_code == 200:
                    grid_data = grid_response.json()

                    # Extract forecast URLs
                    forecast_url = grid_data["properties"]["forecast"]
                    hourly_url = grid_data["properties"]["forecastHourly"]

                    # Get forecast data
                    forecast_response = requests.get(forecast_url, timeout=10)

                    if forecast_response.status_code == 200:
                        forecast_data = forecast_response.json()

                        # Process forecast periods
                        periods = forecast_data["properties"]["periods"][
                            :5
                        ]  # Next 5 periods

                        processed_forecasts = []
                        for period in periods:
                            processed_forecasts.append(
                                {
                                    "period_name": period["name"],
                                    "start_time": period["startTime"],
                                    "temperature": period["temperature"],
                                    "temperature_unit": period["temperatureUnit"],
                                    "wind_speed": period["windSpeed"],
                                    "wind_direction": period["windDirection"],
                                    "short_forecast": period["shortForecast"],
                                    "detailed_forecast": period.get(
                                        "detailedForecast", ""
                                    ),
                                }
                            )

                        noaa_results[city_name] = {
                            "status": "success",
                            "grid_point": f"{grid_data['properties']['gridX']},{grid_data['properties']['gridY']}",
                            "forecast_office": grid_data["properties"][
                                "forecastOffice"
                            ],
                            "forecast_periods": processed_forecasts,
                            "collection_time": datetime.now().isoformat(),
                            "data_quality": "excellent",
                            "source": "NOAA National Weather Service",
                        }

                        print(
                            f"  SUCCESS: Collected {len(processed_forecasts)} forecast periods"
                        )

                    else:
                        print(
                            f"  FORECAST ERROR: Status {forecast_response.status_code}"
                        )
                        noaa_results[city_name] = {
                            "status": "forecast_error",
                            "error_code": forecast_response.status_code,
                            "collection_time": datetime.now().isoformat(),
                        }

                else:
                    print(f"  GRID ERROR: Status {grid_response.status_code}")
                    noaa_results[city_name] = {
                        "status": "grid_error",
                        "error_code": grid_response.status_code,
                        "collection_time": datetime.now().isoformat(),
                    }

            except Exception as e:
                print(f"  EXCEPTION: {str(e)}")
                noaa_results[city_name] = {
                    "status": "exception",
                    "error": str(e),
                    "collection_time": datetime.now().isoformat(),
                }

            time.sleep(1)  # Rate limiting

        return noaa_results

    def collect_waqi_demo_data(self):
        """Attempt to collect data from WAQI using demo/public endpoints."""

        print(f"\nCOLLECTING WAQI DEMO DATA")
        print("=" * 30)

        waqi_results = {}

        # Test major cities first (more likely to have data)
        major_cities = self.global_cities.head(20)  # Top 20 cities

        for idx, row in major_cities.iterrows():
            city_name = row["City"]
            country = row["Country"]

            print(f"Processing {city_name}, {country}...")

            try:
                # Try different WAQI endpoint formats
                city_query = city_name.lower().replace(" ", "-")

                # Method 1: City name search
                search_url = f"https://api.waqi.info/feed/{city_query}/?token=demo"
                response = requests.get(search_url, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("status") == "ok":
                        aqi_data = data["data"]

                        # Extract current AQI and pollutant data
                        current_aqi = aqi_data.get("aqi", "N/A")

                        # Extract individual pollutants
                        pollutants = {}
                        if "iaqi" in aqi_data:
                            for pollutant, values in aqi_data["iaqi"].items():
                                if isinstance(values, dict) and "v" in values:
                                    pollutants[pollutant] = values["v"]

                        waqi_results[city_name] = {
                            "status": "success",
                            "station_name": aqi_data.get("city", {}).get(
                                "name", city_name
                            ),
                            "current_aqi": current_aqi,
                            "pollutants": pollutants,
                            "measurement_time": aqi_data.get("time", {}).get(
                                "s", "unknown"
                            ),
                            "data_quality": "high",
                            "source": "WAQI Network",
                            "collection_time": datetime.now().isoformat(),
                        }

                        print(
                            f"  SUCCESS: AQI={current_aqi}, Pollutants={len(pollutants)}"
                        )

                    else:
                        print(f"  NO DATA: WAQI status={data.get('status')}")
                        waqi_results[city_name] = {
                            "status": "no_data",
                            "waqi_status": data.get("status"),
                            "collection_time": datetime.now().isoformat(),
                        }

                else:
                    print(f"  API ERROR: Status {response.status_code}")
                    waqi_results[city_name] = {
                        "status": "api_error",
                        "error_code": response.status_code,
                        "collection_time": datetime.now().isoformat(),
                    }

            except Exception as e:
                print(f"  EXCEPTION: {str(e)}")
                waqi_results[city_name] = {
                    "status": "exception",
                    "error": str(e),
                    "collection_time": datetime.now().isoformat(),
                }

            time.sleep(1.5)  # Conservative rate limiting

        return waqi_results

    def collect_public_forecast_sources(self):
        """Collect data from public forecast data sources."""

        print(f"\nCOLLECTING PUBLIC FORECAST SOURCES")
        print("=" * 40)

        public_sources = {}

        # EPA AirNow public data (no API key required for some endpoints)
        try:
            print("Testing EPA AirNow public data...")

            # Try to access current observations (sometimes public)
            epa_url = "https://www.airnowapi.org/aq/observation/latLong/current/?format=json&latitude=34.0522&longitude=-118.2437&distance=25&API_KEY=demo"
            response = requests.get(epa_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                public_sources["epa_airnow"] = {
                    "status": "accessible",
                    "sample_data": len(data) if isinstance(data, list) else "object",
                    "data_quality": "excellent",
                    "coverage": "US only",
                }
                print("  SUCCESS: EPA AirNow data accessible")
            else:
                public_sources["epa_airnow"] = {
                    "status": "blocked",
                    "error_code": response.status_code,
                }
                print(f"  BLOCKED: EPA AirNow Status {response.status_code}")

        except Exception as e:
            public_sources["epa_airnow"] = {"status": "exception", "error": str(e)}
            print(f"  EXCEPTION: EPA AirNow - {str(e)}")

        # Test other public sources...
        # (Would expand this section with more public APIs)

        return public_sources

    def create_real_benchmark_dataset(self, noaa_data, waqi_data, public_data):
        """Create a dataset with real benchmark forecasts where available."""

        print(f"\nCREATING REAL BENCHMARK DATASET")
        print("=" * 35)

        benchmark_dataset = {}

        # Process each city in the original dataset
        for idx, row in self.cities_df.iterrows():
            city_name = row["City"]
            country = row["Country"]
            continent = row["Continent"]

            city_benchmark = {
                "city": city_name,
                "country": country,
                "continent": continent,
                "latitude": row["Latitude"],
                "longitude": row["Longitude"],
                "real_forecasts": {},
                "data_sources": [],
                "collection_timestamp": datetime.now().isoformat(),
            }

            # Add NOAA data if available (US cities)
            if city_name in noaa_data and noaa_data[city_name]["status"] == "success":
                city_benchmark["real_forecasts"]["noaa_weather"] = noaa_data[city_name]
                city_benchmark["data_sources"].append("NOAA National Weather Service")

            # Add WAQI data if available
            if city_name in waqi_data and waqi_data[city_name]["status"] == "success":
                city_benchmark["real_forecasts"]["waqi_current"] = waqi_data[city_name]
                city_benchmark["data_sources"].append("WAQI Network")

            # Determine benchmark availability
            has_real_data = len(city_benchmark["real_forecasts"]) > 0
            city_benchmark["has_real_benchmarks"] = has_real_data
            city_benchmark["benchmark_quality"] = (
                "high" if has_real_data else "synthetic_required"
            )

            benchmark_dataset[city_name] = city_benchmark

        return benchmark_dataset

    def save_real_benchmark_data(
        self, benchmark_dataset, noaa_data, waqi_data, public_data
    ):
        """Save real benchmark data to files."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive dataset
        output_file = f"../final_dataset/real_benchmark_data_{timestamp}.json"

        full_dataset = {
            "timestamp": timestamp,
            "collection_type": "Real Benchmark Forecast Data",
            "benchmark_dataset": benchmark_dataset,
            "raw_data": {
                "noaa_forecasts": noaa_data,
                "waqi_current": waqi_data,
                "public_sources": public_data,
            },
            "summary": {
                "total_cities": len(benchmark_dataset),
                "cities_with_real_data": sum(
                    1
                    for city in benchmark_dataset.values()
                    if city["has_real_benchmarks"]
                ),
                "noaa_cities": len(
                    [
                        city
                        for city in benchmark_dataset.values()
                        if "noaa_weather" in city["real_forecasts"]
                    ]
                ),
                "waqi_cities": len(
                    [
                        city
                        for city in benchmark_dataset.values()
                        if "waqi_current" in city["real_forecasts"]
                    ]
                ),
                "coverage_percentage": (
                    sum(
                        1
                        for city in benchmark_dataset.values()
                        if city["has_real_benchmarks"]
                    )
                    / len(benchmark_dataset)
                )
                * 100,
            },
        }

        with open(output_file, "w") as f:
            json.dump(full_dataset, f, indent=2, default=str)

        print(f"\nReal benchmark data saved to: {output_file}")
        return output_file, full_dataset


def main():
    """Main data collection execution."""

    print("REAL BENCHMARK DATA COLLECTION")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 50)

    collector = RealBenchmarkDataCollector()

    # Collect from accessible APIs
    noaa_data = collector.collect_noaa_forecast_data()
    waqi_data = collector.collect_waqi_demo_data()
    public_data = collector.collect_public_forecast_sources()

    # Create benchmark dataset
    benchmark_dataset = collector.create_real_benchmark_dataset(
        noaa_data, waqi_data, public_data
    )

    # Save results
    output_file, full_dataset = collector.save_real_benchmark_data(
        benchmark_dataset, noaa_data, waqi_data, public_data
    )

    # Print summary
    summary = full_dataset["summary"]
    print(f"\nCOLLECTION SUMMARY:")
    print(f"Total cities processed: {summary['total_cities']}")
    print(f"Cities with real benchmark data: {summary['cities_with_real_data']}")
    print(f"NOAA weather forecasts: {summary['noaa_cities']} cities")
    print(f"WAQI current data: {summary['waqi_cities']} cities")
    print(f"Real data coverage: {summary['coverage_percentage']:.1f}%")
    print(f"Next: Validate collected data and update tables")

    return full_dataset, output_file


if __name__ == "__main__":
    results, file_path = main()
