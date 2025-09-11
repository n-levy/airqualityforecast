#!/usr/bin/env python3
"""
Real Measurement Data Collector
Collects ACTUAL measured pollutant data from ground monitoring stations
Uses multiple APIs: OpenAQ, EPA AirNow, EEA, PurpleAir, etc.
Target: PM2.5, PM10, NO2, O3, CO, SO2 measurements for 100 cities
"""
import json
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


class RealMeasurementCollector:
    def __init__(self):
        self.generation_timestamp = datetime.now()
        self.start_date = datetime.now() - timedelta(days=1)  # Yesterday
        self.end_date = self.start_date - timedelta(days=730)  # Two years ago

        # Complete 100-city list
        self.complete_100_cities = [
            # ASIA (20 Cities)
            {
                "name": "Delhi",
                "country": "India",
                "continent": "Asia",
                "lat": 28.61,
                "lon": 77.21,
            },
            {
                "name": "Lahore",
                "country": "Pakistan",
                "continent": "Asia",
                "lat": 31.55,
                "lon": 74.34,
            },
            {
                "name": "Beijing",
                "country": "China",
                "continent": "Asia",
                "lat": 39.90,
                "lon": 116.41,
            },
            {
                "name": "Dhaka",
                "country": "Bangladesh",
                "continent": "Asia",
                "lat": 23.81,
                "lon": 90.41,
            },
            {
                "name": "Mumbai",
                "country": "India",
                "continent": "Asia",
                "lat": 19.08,
                "lon": 72.88,
            },
            {
                "name": "Karachi",
                "country": "Pakistan",
                "continent": "Asia",
                "lat": 24.86,
                "lon": 67.00,
            },
            {
                "name": "Shanghai",
                "country": "China",
                "continent": "Asia",
                "lat": 31.23,
                "lon": 121.47,
            },
            {
                "name": "Kolkata",
                "country": "India",
                "continent": "Asia",
                "lat": 22.57,
                "lon": 88.36,
            },
            {
                "name": "Bangkok",
                "country": "Thailand",
                "continent": "Asia",
                "lat": 14.60,
                "lon": 100.50,
            },
            {
                "name": "Jakarta",
                "country": "Indonesia",
                "continent": "Asia",
                "lat": -6.21,
                "lon": 106.85,
            },
            {
                "name": "Manila",
                "country": "Philippines",
                "continent": "Asia",
                "lat": 14.60,
                "lon": 120.98,
            },
            {
                "name": "Ho Chi Minh City",
                "country": "Vietnam",
                "continent": "Asia",
                "lat": 10.82,
                "lon": 106.63,
            },
            {
                "name": "Hanoi",
                "country": "Vietnam",
                "continent": "Asia",
                "lat": 21.03,
                "lon": 105.85,
            },
            {
                "name": "Seoul",
                "country": "South Korea",
                "continent": "Asia",
                "lat": 37.57,
                "lon": 126.98,
            },
            {
                "name": "Taipei",
                "country": "Taiwan",
                "continent": "Asia",
                "lat": 25.03,
                "lon": 121.57,
            },
            {
                "name": "Ulaanbaatar",
                "country": "Mongolia",
                "continent": "Asia",
                "lat": 47.89,
                "lon": 106.91,
            },
            {
                "name": "Almaty",
                "country": "Kazakhstan",
                "continent": "Asia",
                "lat": 43.26,
                "lon": 76.93,
            },
            {
                "name": "Tashkent",
                "country": "Uzbekistan",
                "continent": "Asia",
                "lat": 41.30,
                "lon": 69.24,
            },
            {
                "name": "Tehran",
                "country": "Iran",
                "continent": "Asia",
                "lat": 35.70,
                "lon": 51.42,
            },
            {
                "name": "Kabul",
                "country": "Afghanistan",
                "continent": "Asia",
                "lat": 34.56,
                "lon": 69.21,
            },
            # AFRICA (20 Cities)
            {
                "name": "N'Djamena",
                "country": "Chad",
                "continent": "Africa",
                "lat": 12.13,
                "lon": 15.06,
            },
            {
                "name": "Cairo",
                "country": "Egypt",
                "continent": "Africa",
                "lat": 30.04,
                "lon": 31.24,
            },
            {
                "name": "Lagos",
                "country": "Nigeria",
                "continent": "Africa",
                "lat": 6.52,
                "lon": 3.38,
            },
            {
                "name": "Accra",
                "country": "Ghana",
                "continent": "Africa",
                "lat": 5.60,
                "lon": -0.19,
            },
            {
                "name": "Khartoum",
                "country": "Sudan",
                "continent": "Africa",
                "lat": 15.50,
                "lon": 32.56,
            },
            {
                "name": "Kampala",
                "country": "Uganda",
                "continent": "Africa",
                "lat": 0.35,
                "lon": 32.58,
            },
            {
                "name": "Nairobi",
                "country": "Kenya",
                "continent": "Africa",
                "lat": -1.29,
                "lon": 36.82,
            },
            {
                "name": "Abidjan",
                "country": "Côte d'Ivoire",
                "continent": "Africa",
                "lat": 5.36,
                "lon": -4.01,
            },
            {
                "name": "Bamako",
                "country": "Mali",
                "continent": "Africa",
                "lat": 12.64,
                "lon": -8.00,
            },
            {
                "name": "Ouagadougou",
                "country": "Burkina Faso",
                "continent": "Africa",
                "lat": 12.37,
                "lon": -1.52,
            },
            {
                "name": "Dakar",
                "country": "Senegal",
                "continent": "Africa",
                "lat": 14.72,
                "lon": -17.47,
            },
            {
                "name": "Kinshasa",
                "country": "DR Congo",
                "continent": "Africa",
                "lat": -4.44,
                "lon": 15.27,
            },
            {
                "name": "Casablanca",
                "country": "Morocco",
                "continent": "Africa",
                "lat": 33.57,
                "lon": -7.59,
            },
            {
                "name": "Johannesburg",
                "country": "South Africa",
                "continent": "Africa",
                "lat": -26.20,
                "lon": 28.05,
            },
            {
                "name": "Addis Ababa",
                "country": "Ethiopia",
                "continent": "Africa",
                "lat": 9.15,
                "lon": 38.75,
            },
            {
                "name": "Dar es Salaam",
                "country": "Tanzania",
                "continent": "Africa",
                "lat": -6.79,
                "lon": 39.21,
            },
            {
                "name": "Algiers",
                "country": "Algeria",
                "continent": "Africa",
                "lat": 36.75,
                "lon": 3.06,
            },
            {
                "name": "Tunis",
                "country": "Tunisia",
                "continent": "Africa",
                "lat": 36.81,
                "lon": 10.18,
            },
            {
                "name": "Maputo",
                "country": "Mozambique",
                "continent": "Africa",
                "lat": -25.97,
                "lon": 32.57,
            },
            {
                "name": "Cape Town",
                "country": "South Africa",
                "continent": "Africa",
                "lat": -33.92,
                "lon": 18.42,
            },
            # EUROPE (20 Cities)
            {
                "name": "Skopje",
                "country": "North Macedonia",
                "continent": "Europe",
                "lat": 42.00,
                "lon": 21.43,
            },
            {
                "name": "Sarajevo",
                "country": "Bosnia and Herzegovina",
                "continent": "Europe",
                "lat": 43.86,
                "lon": 18.41,
            },
            {
                "name": "Sofia",
                "country": "Bulgaria",
                "continent": "Europe",
                "lat": 42.70,
                "lon": 23.32,
            },
            {
                "name": "Plovdiv",
                "country": "Bulgaria",
                "continent": "Europe",
                "lat": 42.14,
                "lon": 24.75,
            },
            {
                "name": "Bucharest",
                "country": "Romania",
                "continent": "Europe",
                "lat": 44.43,
                "lon": 26.10,
            },
            {
                "name": "Belgrade",
                "country": "Serbia",
                "continent": "Europe",
                "lat": 44.79,
                "lon": 20.45,
            },
            {
                "name": "Warsaw",
                "country": "Poland",
                "continent": "Europe",
                "lat": 52.23,
                "lon": 21.01,
            },
            {
                "name": "Krakow",
                "country": "Poland",
                "continent": "Europe",
                "lat": 50.06,
                "lon": 19.95,
            },
            {
                "name": "Prague",
                "country": "Czech Republic",
                "continent": "Europe",
                "lat": 50.08,
                "lon": 14.44,
            },
            {
                "name": "Budapest",
                "country": "Hungary",
                "continent": "Europe",
                "lat": 47.50,
                "lon": 19.04,
            },
            {
                "name": "Milan",
                "country": "Italy",
                "continent": "Europe",
                "lat": 45.46,
                "lon": 9.19,
            },
            {
                "name": "Turin",
                "country": "Italy",
                "continent": "Europe",
                "lat": 45.07,
                "lon": 7.69,
            },
            {
                "name": "Naples",
                "country": "Italy",
                "continent": "Europe",
                "lat": 40.85,
                "lon": 14.27,
            },
            {
                "name": "Athens",
                "country": "Greece",
                "continent": "Europe",
                "lat": 37.98,
                "lon": 23.73,
            },
            {
                "name": "Madrid",
                "country": "Spain",
                "continent": "Europe",
                "lat": 40.42,
                "lon": -3.70,
            },
            {
                "name": "Barcelona",
                "country": "Spain",
                "continent": "Europe",
                "lat": 41.39,
                "lon": 2.17,
            },
            {
                "name": "Paris",
                "country": "France",
                "continent": "Europe",
                "lat": 48.86,
                "lon": 2.35,
            },
            {
                "name": "London",
                "country": "UK",
                "continent": "Europe",
                "lat": 51.51,
                "lon": -0.13,
            },
            {
                "name": "Berlin",
                "country": "Germany",
                "continent": "Europe",
                "lat": 52.52,
                "lon": 13.41,
            },
            {
                "name": "Amsterdam",
                "country": "Netherlands",
                "continent": "Europe",
                "lat": 52.37,
                "lon": 4.90,
            },
            # NORTH AMERICA (20 Cities)
            {
                "name": "Mexicali",
                "country": "Mexico",
                "continent": "North America",
                "lat": 32.65,
                "lon": -115.47,
            },
            {
                "name": "Mexico City",
                "country": "Mexico",
                "continent": "North America",
                "lat": 19.43,
                "lon": -99.13,
            },
            {
                "name": "Guadalajara",
                "country": "Mexico",
                "continent": "North America",
                "lat": 20.66,
                "lon": -103.35,
            },
            {
                "name": "Tijuana",
                "country": "Mexico",
                "continent": "North America",
                "lat": 32.51,
                "lon": -117.04,
            },
            {
                "name": "Monterrey",
                "country": "Mexico",
                "continent": "North America",
                "lat": 25.69,
                "lon": -100.32,
            },
            {
                "name": "Los Angeles",
                "country": "USA",
                "continent": "North America",
                "lat": 34.05,
                "lon": -118.24,
            },
            {
                "name": "Fresno",
                "country": "USA",
                "continent": "North America",
                "lat": 36.74,
                "lon": -119.79,
            },
            {
                "name": "Phoenix",
                "country": "USA",
                "continent": "North America",
                "lat": 33.45,
                "lon": -112.07,
            },
            {
                "name": "Houston",
                "country": "USA",
                "continent": "North America",
                "lat": 29.76,
                "lon": -95.37,
            },
            {
                "name": "New York",
                "country": "USA",
                "continent": "North America",
                "lat": 40.71,
                "lon": -74.01,
            },
            {
                "name": "Chicago",
                "country": "USA",
                "continent": "North America",
                "lat": 41.88,
                "lon": -87.63,
            },
            {
                "name": "Denver",
                "country": "USA",
                "continent": "North America",
                "lat": 39.74,
                "lon": -104.99,
            },
            {
                "name": "Detroit",
                "country": "USA",
                "continent": "North America",
                "lat": 42.33,
                "lon": -83.05,
            },
            {
                "name": "Atlanta",
                "country": "USA",
                "continent": "North America",
                "lat": 33.75,
                "lon": -84.39,
            },
            {
                "name": "Philadelphia",
                "country": "USA",
                "continent": "North America",
                "lat": 39.95,
                "lon": -75.17,
            },
            {
                "name": "Toronto",
                "country": "Canada",
                "continent": "North America",
                "lat": 43.65,
                "lon": -79.38,
            },
            {
                "name": "Montreal",
                "country": "Canada",
                "continent": "North America",
                "lat": 45.50,
                "lon": -73.57,
            },
            {
                "name": "Vancouver",
                "country": "Canada",
                "continent": "North America",
                "lat": 49.28,
                "lon": -123.12,
            },
            {
                "name": "Calgary",
                "country": "Canada",
                "continent": "North America",
                "lat": 51.04,
                "lon": -114.07,
            },
            {
                "name": "Ottawa",
                "country": "Canada",
                "continent": "North America",
                "lat": 45.42,
                "lon": -75.70,
            },
            # SOUTH AMERICA (20 Cities)
            {
                "name": "Lima",
                "country": "Peru",
                "continent": "South America",
                "lat": -12.05,
                "lon": -77.04,
            },
            {
                "name": "Santiago",
                "country": "Chile",
                "continent": "South America",
                "lat": -33.45,
                "lon": -70.67,
            },
            {
                "name": "São Paulo",
                "country": "Brazil",
                "continent": "South America",
                "lat": -23.55,
                "lon": -46.63,
            },
            {
                "name": "Rio de Janeiro",
                "country": "Brazil",
                "continent": "South America",
                "lat": -22.91,
                "lon": -43.17,
            },
            {
                "name": "Bogotá",
                "country": "Colombia",
                "continent": "South America",
                "lat": 4.71,
                "lon": -74.07,
            },
            {
                "name": "La Paz",
                "country": "Bolivia",
                "continent": "South America",
                "lat": -16.50,
                "lon": -68.15,
            },
            {
                "name": "Medellín",
                "country": "Colombia",
                "continent": "South America",
                "lat": 6.24,
                "lon": -75.58,
            },
            {
                "name": "Buenos Aires",
                "country": "Argentina",
                "continent": "South America",
                "lat": -34.61,
                "lon": -58.40,
            },
            {
                "name": "Quito",
                "country": "Ecuador",
                "continent": "South America",
                "lat": -0.18,
                "lon": -78.47,
            },
            {
                "name": "Caracas",
                "country": "Venezuela",
                "continent": "South America",
                "lat": 10.48,
                "lon": -66.90,
            },
            {
                "name": "Belo Horizonte",
                "country": "Brazil",
                "continent": "South America",
                "lat": -19.92,
                "lon": -43.93,
            },
            {
                "name": "Brasília",
                "country": "Brazil",
                "continent": "South America",
                "lat": -15.78,
                "lon": -47.93,
            },
            {
                "name": "Porto Alegre",
                "country": "Brazil",
                "continent": "South America",
                "lat": -30.03,
                "lon": -51.22,
            },
            {
                "name": "Montevideo",
                "country": "Uruguay",
                "continent": "South America",
                "lat": -34.90,
                "lon": -56.16,
            },
            {
                "name": "Asunción",
                "country": "Paraguay",
                "continent": "South America",
                "lat": -25.26,
                "lon": -57.58,
            },
            {
                "name": "Córdoba",
                "country": "Argentina",
                "continent": "South America",
                "lat": -31.42,
                "lon": -64.19,
            },
            {
                "name": "Valparaíso",
                "country": "Chile",
                "continent": "South America",
                "lat": -33.05,
                "lon": -71.61,
            },
            {
                "name": "Cali",
                "country": "Colombia",
                "continent": "South America",
                "lat": 3.45,
                "lon": -76.53,
            },
            {
                "name": "Curitiba",
                "country": "Brazil",
                "continent": "South America",
                "lat": -25.43,
                "lon": -49.27,
            },
            {
                "name": "Fortaleza",
                "country": "Brazil",
                "continent": "South America",
                "lat": -3.72,
                "lon": -38.54,
            },
        ]

        self.measurement_data = {}
        self.api_sources = {
            "openaq_v3": "https://api.openaq.org/v3",
            "purpleair": "https://api.purpleair.com/v1",
            "waqi": "https://api.waqi.info/feed",
            "eea": "https://discomap.eea.europa.eu/map/fme/AirQualityExport.fmw",
            "airnow": "https://www.airnowapi.org/aq",
        }

    def try_openaq_v3(self, city_info):
        """Try OpenAQ v3 API for real measurements"""
        try:
            # Try new OpenAQ v3 endpoint
            url = f"{self.api_sources['openaq_v3']}/locations"
            params = {
                "limit": 10,
                "coordinates": f"{city_info['lat']},{city_info['lon']}",
                "radius": 50000,  # 50km radius
                "has_geo": True,
            }

            headers = {
                "User-Agent": "ResearchProject/1.0",
                "Accept": "application/json",
            }

            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()
                locations = data.get("results", [])

                if locations:
                    location = locations[0]
                    location_id = location.get("id")

                    # Get measurements for this location
                    measurements_url = f"{self.api_sources['openaq_v3']}/measurements"
                    measurements_params = {
                        "locations_id": location_id,
                        "limit": 100,
                        "date_from": self.end_date.strftime("%Y-%m-%d"),
                        "date_to": self.start_date.strftime("%Y-%m-%d"),
                    }

                    measurements_response = requests.get(
                        measurements_url,
                        params=measurements_params,
                        headers=headers,
                        timeout=15,
                    )

                    if measurements_response.status_code == 200:
                        measurements_data = measurements_response.json()
                        measurements = measurements_data.get("results", [])

                        if measurements:
                            return {
                                "source": "OpenAQ_v3",
                                "location_id": location_id,
                                "location_name": location.get("name"),
                                "measurements_count": len(measurements),
                                "sample_measurement": measurements[0],
                                "data_type": "REAL_GROUND_MEASUREMENTS",
                            }

            return None

        except Exception as e:
            print(f"    OpenAQ v3 error for {city_info['name']}: {str(e)}")
            return None

    def try_waqi_real_measurements(self, city_info):
        """Try WAQI for real measurements (not forecasts)"""
        try:
            # WAQI typically provides real measurements, not forecasts
            url = (
                f"{self.api_sources['waqi']}/geo:{city_info['lat']};{city_info['lon']}/"
            )
            params = {"token": "demo"}  # Demo token

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    station_data = data.get("data", {})
                    iaqi = station_data.get("iaqi", {})

                    # Check if we have individual pollutant measurements
                    pollutants = {}
                    for pollutant in ["pm25", "pm10", "no2", "o3", "co", "so2"]:
                        if pollutant in iaqi:
                            pollutants[pollutant] = iaqi[pollutant].get("v")

                    if pollutants:
                        return {
                            "source": "WAQI",
                            "station_name": station_data.get("city", {}).get("name"),
                            "aqi": station_data.get("aqi"),
                            "pollutants": pollutants,
                            "measurement_time": station_data.get("time", {}).get("s"),
                            "data_type": "REAL_STATION_MEASUREMENTS",
                        }

            return None

        except Exception as e:
            print(f"    WAQI error for {city_info['name']}: {str(e)}")
            return None

    def try_purpleair_measurements(self, city_info):
        """Try PurpleAir for real PM2.5 measurements"""
        try:
            # PurpleAir provides real-time sensor measurements
            url = f"{self.api_sources['purpleair']}/sensors"
            params = {
                "fields": "name,latitude,longitude,pm2.5_atm,pm2.5_cf_1,humidity,temperature",
                "location_type": 0,  # Outside sensors
                "nwlat": city_info["lat"] + 0.1,
                "nwlng": city_info["lon"] - 0.1,
                "selat": city_info["lat"] - 0.1,
                "selng": city_info["lon"] + 0.1,
            }

            # Note: PurpleAir requires API key for production use
            headers = {
                "User-Agent": "ResearchProject/1.0",
                "Accept": "application/json",
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                sensors = data.get("data", [])

                if sensors:
                    sensor = sensors[0]
                    return {
                        "source": "PurpleAir",
                        "sensor_name": sensor[0] if sensor else "Unknown",
                        "pm25_measurement": sensor[3] if len(sensor) > 3 else None,
                        "data_type": "REAL_SENSOR_MEASUREMENTS",
                    }

            return None

        except Exception as e:
            print(f"    PurpleAir error for {city_info['name']}: {str(e)}")
            return None

    def collect_real_measurements_for_city(self, city_info):
        """Try multiple APIs to get real measurements for a city"""
        print(
            f"  Collecting real measurements for {city_info['name']}, {city_info['country']}..."
        )

        measurement_results = {}

        # Try OpenAQ v3
        openaq_result = self.try_openaq_v3(city_info)
        if openaq_result:
            measurement_results["openaq"] = openaq_result
            print(
                f"    SUCCESS OpenAQ: {openaq_result['measurements_count']} measurements"
            )

        # Try WAQI
        waqi_result = self.try_waqi_real_measurements(city_info)
        if waqi_result:
            measurement_results["waqi"] = waqi_result
            print(
                f"    SUCCESS WAQI: AQI={waqi_result['aqi']}, {len(waqi_result['pollutants'])} pollutants"
            )

        # Try PurpleAir
        purpleair_result = self.try_purpleair_measurements(city_info)
        if purpleair_result:
            measurement_results["purpleair"] = purpleair_result
            print(
                f"    SUCCESS PurpleAir: PM2.5={purpleair_result['pm25_measurement']}"
            )

        if not measurement_results:
            print(f"    FAILED: No real measurements found")
            return None

        return {
            "city_info": city_info,
            "measurement_sources": measurement_results,
            "total_sources": len(measurement_results),
            "collection_timestamp": datetime.now().isoformat(),
        }

    def collect_all_real_measurements(self):
        """Collect real measurements for all 100 cities"""
        print("REAL MEASUREMENT DATA COLLECTOR")
        print("Collecting actual ground station measurements")
        print("Target: PM2.5, PM10, NO2, O3, CO, SO2 from monitoring stations")
        print("=" * 80)

        successful_cities = 0
        failed_cities = []

        for i, city_info in enumerate(self.complete_100_cities):
            print(
                f"[{i+1:3d}/100] {city_info['name']}, {city_info['country']} ({city_info['continent']})"
            )

            city_measurements = self.collect_real_measurements_for_city(city_info)

            if city_measurements:
                self.measurement_data[f"{city_info['name']}_{city_info['country']}"] = (
                    city_measurements
                )
                successful_cities += 1
            else:
                failed_cities.append(f"{city_info['name']}, {city_info['country']}")

            # Rate limiting
            time.sleep(1)

            # Progress report every 20 cities
            if (i + 1) % 20 == 0:
                print(f"\nProgress: {successful_cities}/{i+1} cities successful")
                print("-" * 40)

        print(f"\nREAL MEASUREMENT COLLECTION COMPLETE:")
        print(f"  Successful cities: {successful_cities}/100")
        print(f"  Failed cities: {len(failed_cities)}")
        if failed_cities:
            print(
                f"  Sample failures: {', '.join(failed_cities[:5])}{'...' if len(failed_cities) > 5 else ''}"
            )

        return successful_cities

    def save_measurement_results(self, successful_cities):
        """Save the real measurement collection results"""
        timestamp_str = self.generation_timestamp.strftime("%Y%m%d_%H%M%S")

        comprehensive_results = {
            "collection_timestamp": self.generation_timestamp.isoformat(),
            "collection_type": "REAL_GROUND_MEASUREMENTS",
            "objective": "Collect actual measured pollutant data from monitoring stations",
            "target_pollutants": ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"],
            "api_sources_attempted": list(self.api_sources.keys()),
            "collection_summary": {
                "total_cities_attempted": len(self.complete_100_cities),
                "successful_cities": successful_cities,
                "success_rate": f"{successful_cities/100*100:.1f}%",
                "cities_with_measurements": len(self.measurement_data),
            },
            "measurement_data": self.measurement_data,
            "data_authenticity": {
                "measurement_type": "Real ground station measurements",
                "data_source": "Multiple APIs (OpenAQ, WAQI, PurpleAir)",
                "not_forecasts": "These are actual measured values from sensors",
                "spatial_resolution": "City-specific monitoring stations",
            },
        }

        results_file = (
            f"../final_dataset/REAL_MEASUREMENTS_collection_{timestamp_str}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                comprehensive_results, f, indent=2, default=str, ensure_ascii=False
            )

        print(f"\nResults saved: {results_file}")

        # Create summary report
        print(f"\nREAL MEASUREMENT DATA COLLECTION SUMMARY:")
        print(f"Target: 100% real measured pollutant data")
        print(
            f"Success: {successful_cities}/100 cities ({successful_cities/100*100:.1f}%)"
        )
        print(f"Sources: OpenAQ, WAQI, PurpleAir APIs")
        print(f"Data Type: Actual ground station measurements (not forecasts)")

        return results_file


def main():
    """Main execution"""
    collector = RealMeasurementCollector()
    successful_cities = collector.collect_all_real_measurements()
    results_file = collector.save_measurement_results(successful_cities)

    print(f"\n🎉 REAL MEASUREMENT COLLECTION COMPLETE!")
    print(f"Next step: Use this data for walk-forward ensemble validation")


if __name__ == "__main__":
    main()
