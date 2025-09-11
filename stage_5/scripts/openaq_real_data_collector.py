#!/usr/bin/env python3
"""
OpenAQ Real Data Collector
Collect authentic measured air pollutant data from monitoring stations
Using OpenAQ API with authentication for real ground truth data
"""
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


class OpenAQRealDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v3"  # Try v3 first
        self.headers = {
            "User-Agent": "AQF311-Research/1.0",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",  # Try Bearer token first
        }

        self.fallback_headers = [
            {
                "User-Agent": "AQF311-Research/1.0",
                "Accept": "application/json",
                "X-API-Key": api_key,
            },
            {
                "User-Agent": "AQF311-Research/1.0",
                "Accept": "application/json",
                "x-api-key": api_key,
            },
            {
                "User-Agent": "AQF311-Research/1.0",
                "Accept": "application/json",
                "api-key": api_key,
            },
        ]

        self.fallback_urls = [
            "https://api.openaq.org/v3",
            "https://api.openaq.org/v2",
            "https://api.openaq.org/v1",
        ]

        # 100 cities from the project
        self.cities = [
            {
                "name": "Delhi",
                "country": "IN",
                "lat": 28.6139,
                "lon": 77.2090,
                "continent": "Asia",
            },
            {
                "name": "Lahore",
                "country": "PK",
                "lat": 31.5497,
                "lon": 74.3436,
                "continent": "Asia",
            },
            {
                "name": "Beijing",
                "country": "CN",
                "lat": 39.9042,
                "lon": 116.4074,
                "continent": "Asia",
            },
            {
                "name": "Dhaka",
                "country": "BD",
                "lat": 23.8103,
                "lon": 90.4125,
                "continent": "Asia",
            },
            {
                "name": "Mumbai",
                "country": "IN",
                "lat": 19.0760,
                "lon": 72.8777,
                "continent": "Asia",
            },
            {
                "name": "Karachi",
                "country": "PK",
                "lat": 24.8607,
                "lon": 67.0011,
                "continent": "Asia",
            },
            {
                "name": "Shanghai",
                "country": "CN",
                "lat": 31.2304,
                "lon": 121.4737,
                "continent": "Asia",
            },
            {
                "name": "Kolkata",
                "country": "IN",
                "lat": 22.5726,
                "lon": 88.3639,
                "continent": "Asia",
            },
            {
                "name": "Bangkok",
                "country": "TH",
                "lat": 13.7563,
                "lon": 100.5018,
                "continent": "Asia",
            },
            {
                "name": "Jakarta",
                "country": "ID",
                "lat": -6.2088,
                "lon": 106.8456,
                "continent": "Asia",
            },
            {
                "name": "Manila",
                "country": "PH",
                "lat": 14.5995,
                "lon": 120.9842,
                "continent": "Asia",
            },
            {
                "name": "Ho Chi Minh City",
                "country": "VN",
                "lat": 10.8231,
                "lon": 106.6297,
                "continent": "Asia",
            },
            {
                "name": "Hanoi",
                "country": "VN",
                "lat": 21.0285,
                "lon": 105.8542,
                "continent": "Asia",
            },
            {
                "name": "Seoul",
                "country": "KR",
                "lat": 37.5665,
                "lon": 126.9780,
                "continent": "Asia",
            },
            {
                "name": "Taipei",
                "country": "TW",
                "lat": 25.0330,
                "lon": 121.5654,
                "continent": "Asia",
            },
            {
                "name": "Ulaanbaatar",
                "country": "MN",
                "lat": 47.8864,
                "lon": 106.9057,
                "continent": "Asia",
            },
            {
                "name": "Almaty",
                "country": "KZ",
                "lat": 43.2220,
                "lon": 76.8512,
                "continent": "Asia",
            },
            {
                "name": "Tashkent",
                "country": "UZ",
                "lat": 41.2995,
                "lon": 69.2401,
                "continent": "Asia",
            },
            {
                "name": "Tehran",
                "country": "IR",
                "lat": 35.6892,
                "lon": 51.3890,
                "continent": "Asia",
            },
            {
                "name": "Kabul",
                "country": "AF",
                "lat": 34.5553,
                "lon": 69.2075,
                "continent": "Asia",
            },
            # Africa
            {
                "name": "N'Djamena",
                "country": "TD",
                "lat": 12.1348,
                "lon": 15.0557,
                "continent": "Africa",
            },
            {
                "name": "Cairo",
                "country": "EG",
                "lat": 30.0444,
                "lon": 31.2357,
                "continent": "Africa",
            },
            {
                "name": "Lagos",
                "country": "NG",
                "lat": 6.5244,
                "lon": 3.3792,
                "continent": "Africa",
            },
            {
                "name": "Accra",
                "country": "GH",
                "lat": 5.6037,
                "lon": -0.1870,
                "continent": "Africa",
            },
            {
                "name": "Khartoum",
                "country": "SD",
                "lat": 15.5007,
                "lon": 32.5599,
                "continent": "Africa",
            },
            {
                "name": "Kampala",
                "country": "UG",
                "lat": 0.3476,
                "lon": 32.5825,
                "continent": "Africa",
            },
            {
                "name": "Nairobi",
                "country": "KE",
                "lat": -1.2921,
                "lon": 36.8219,
                "continent": "Africa",
            },
            {
                "name": "Abidjan",
                "country": "CI",
                "lat": 5.3600,
                "lon": -4.0083,
                "continent": "Africa",
            },
            {
                "name": "Bamako",
                "country": "ML",
                "lat": 12.6392,
                "lon": -8.0029,
                "continent": "Africa",
            },
            {
                "name": "Ouagadougou",
                "country": "BF",
                "lat": 12.3714,
                "lon": -1.5197,
                "continent": "Africa",
            },
            {
                "name": "Dakar",
                "country": "SN",
                "lat": 14.7167,
                "lon": -17.4677,
                "continent": "Africa",
            },
            {
                "name": "Kinshasa",
                "country": "CD",
                "lat": -4.4419,
                "lon": 15.2663,
                "continent": "Africa",
            },
            {
                "name": "Casablanca",
                "country": "MA",
                "lat": 33.5731,
                "lon": -7.5898,
                "continent": "Africa",
            },
            {
                "name": "Johannesburg",
                "country": "ZA",
                "lat": -26.2041,
                "lon": 28.0473,
                "continent": "Africa",
            },
            {
                "name": "Addis Ababa",
                "country": "ET",
                "lat": 9.1450,
                "lon": 38.7451,
                "continent": "Africa",
            },
            {
                "name": "Dar es Salaam",
                "country": "TZ",
                "lat": -6.7924,
                "lon": 39.2083,
                "continent": "Africa",
            },
            {
                "name": "Algiers",
                "country": "DZ",
                "lat": 36.7538,
                "lon": 3.0588,
                "continent": "Africa",
            },
            {
                "name": "Tunis",
                "country": "TN",
                "lat": 36.8065,
                "lon": 10.1815,
                "continent": "Africa",
            },
            {
                "name": "Maputo",
                "country": "MZ",
                "lat": -25.9692,
                "lon": 32.5732,
                "continent": "Africa",
            },
            {
                "name": "Cape Town",
                "country": "ZA",
                "lat": -33.9249,
                "lon": 18.4241,
                "continent": "Africa",
            },
            # Europe
            {
                "name": "Skopje",
                "country": "MK",
                "lat": 41.9973,
                "lon": 21.4280,
                "continent": "Europe",
            },
            {
                "name": "Sarajevo",
                "country": "BA",
                "lat": 43.8563,
                "lon": 18.4131,
                "continent": "Europe",
            },
            {
                "name": "Sofia",
                "country": "BG",
                "lat": 42.6977,
                "lon": 23.3219,
                "continent": "Europe",
            },
            {
                "name": "Plovdiv",
                "country": "BG",
                "lat": 42.1354,
                "lon": 24.7453,
                "continent": "Europe",
            },
            {
                "name": "Bucharest",
                "country": "RO",
                "lat": 44.4268,
                "lon": 26.1025,
                "continent": "Europe",
            },
            {
                "name": "Belgrade",
                "country": "RS",
                "lat": 44.7866,
                "lon": 20.4489,
                "continent": "Europe",
            },
            {
                "name": "Warsaw",
                "country": "PL",
                "lat": 52.2297,
                "lon": 21.0122,
                "continent": "Europe",
            },
            {
                "name": "Krakow",
                "country": "PL",
                "lat": 50.0647,
                "lon": 19.9450,
                "continent": "Europe",
            },
            {
                "name": "Prague",
                "country": "CZ",
                "lat": 50.0755,
                "lon": 14.4378,
                "continent": "Europe",
            },
            {
                "name": "Budapest",
                "country": "HU",
                "lat": 47.4979,
                "lon": 19.0402,
                "continent": "Europe",
            },
            {
                "name": "Milan",
                "country": "IT",
                "lat": 45.4642,
                "lon": 9.1900,
                "continent": "Europe",
            },
            {
                "name": "Turin",
                "country": "IT",
                "lat": 45.0703,
                "lon": 7.6869,
                "continent": "Europe",
            },
            {
                "name": "Naples",
                "country": "IT",
                "lat": 40.8518,
                "lon": 14.2681,
                "continent": "Europe",
            },
            {
                "name": "Athens",
                "country": "GR",
                "lat": 37.9838,
                "lon": 23.7275,
                "continent": "Europe",
            },
            {
                "name": "Madrid",
                "country": "ES",
                "lat": 40.4168,
                "lon": -3.7038,
                "continent": "Europe",
            },
            {
                "name": "Barcelona",
                "country": "ES",
                "lat": 41.3851,
                "lon": 2.1734,
                "continent": "Europe",
            },
            {
                "name": "Paris",
                "country": "FR",
                "lat": 48.8566,
                "lon": 2.3522,
                "continent": "Europe",
            },
            {
                "name": "London",
                "country": "GB",
                "lat": 51.5074,
                "lon": -0.1278,
                "continent": "Europe",
            },
            {
                "name": "Berlin",
                "country": "DE",
                "lat": 52.5200,
                "lon": 13.4050,
                "continent": "Europe",
            },
            {
                "name": "Amsterdam",
                "country": "NL",
                "lat": 52.3676,
                "lon": 4.9041,
                "continent": "Europe",
            },
            # North America
            {
                "name": "Mexicali",
                "country": "MX",
                "lat": 32.6245,
                "lon": -115.4523,
                "continent": "North America",
            },
            {
                "name": "Mexico City",
                "country": "MX",
                "lat": 19.4326,
                "lon": -99.1332,
                "continent": "North America",
            },
            {
                "name": "Guadalajara",
                "country": "MX",
                "lat": 20.6597,
                "lon": -103.3496,
                "continent": "North America",
            },
            {
                "name": "Tijuana",
                "country": "MX",
                "lat": 32.5149,
                "lon": -117.0382,
                "continent": "North America",
            },
            {
                "name": "Monterrey",
                "country": "MX",
                "lat": 25.6866,
                "lon": -100.3161,
                "continent": "North America",
            },
            {
                "name": "Los Angeles",
                "country": "US",
                "lat": 34.0522,
                "lon": -118.2437,
                "continent": "North America",
            },
            {
                "name": "Fresno",
                "country": "US",
                "lat": 36.7378,
                "lon": -119.7871,
                "continent": "North America",
            },
            {
                "name": "Phoenix",
                "country": "US",
                "lat": 33.4484,
                "lon": -112.0740,
                "continent": "North America",
            },
            {
                "name": "Houston",
                "country": "US",
                "lat": 29.7604,
                "lon": -95.3698,
                "continent": "North America",
            },
            {
                "name": "New York",
                "country": "US",
                "lat": 40.7128,
                "lon": -74.0060,
                "continent": "North America",
            },
            {
                "name": "Chicago",
                "country": "US",
                "lat": 41.8781,
                "lon": -87.6298,
                "continent": "North America",
            },
            {
                "name": "Denver",
                "country": "US",
                "lat": 39.7392,
                "lon": -104.9903,
                "continent": "North America",
            },
            {
                "name": "Detroit",
                "country": "US",
                "lat": 42.3314,
                "lon": -83.0458,
                "continent": "North America",
            },
            {
                "name": "Atlanta",
                "country": "US",
                "lat": 33.7490,
                "lon": -84.3880,
                "continent": "North America",
            },
            {
                "name": "Philadelphia",
                "country": "US",
                "lat": 39.9526,
                "lon": -75.1652,
                "continent": "North America",
            },
            {
                "name": "Toronto",
                "country": "CA",
                "lat": 43.6532,
                "lon": -79.3832,
                "continent": "North America",
            },
            {
                "name": "Montreal",
                "country": "CA",
                "lat": 45.5017,
                "lon": -73.5673,
                "continent": "North America",
            },
            {
                "name": "Vancouver",
                "country": "CA",
                "lat": 49.2827,
                "lon": -123.1207,
                "continent": "North America",
            },
            {
                "name": "Calgary",
                "country": "CA",
                "lat": 51.0447,
                "lon": -114.0719,
                "continent": "North America",
            },
            {
                "name": "Ottawa",
                "country": "CA",
                "lat": 45.4215,
                "lon": -75.6972,
                "continent": "North America",
            },
            # South America
            {
                "name": "Lima",
                "country": "PE",
                "lat": -12.0464,
                "lon": -77.0428,
                "continent": "South America",
            },
            {
                "name": "Santiago",
                "country": "CL",
                "lat": -33.4489,
                "lon": -70.6693,
                "continent": "South America",
            },
            {
                "name": "São Paulo",
                "country": "BR",
                "lat": -23.5505,
                "lon": -46.6333,
                "continent": "South America",
            },
            {
                "name": "Rio de Janeiro",
                "country": "BR",
                "lat": -22.9068,
                "lon": -43.1729,
                "continent": "South America",
            },
            {
                "name": "Bogotá",
                "country": "CO",
                "lat": 4.7110,
                "lon": -74.0721,
                "continent": "South America",
            },
            {
                "name": "La Paz",
                "country": "BO",
                "lat": -16.5000,
                "lon": -68.1193,
                "continent": "South America",
            },
            {
                "name": "Medellín",
                "country": "CO",
                "lat": 6.2442,
                "lon": -75.5812,
                "continent": "South America",
            },
            {
                "name": "Buenos Aires",
                "country": "AR",
                "lat": -34.6118,
                "lon": -58.3960,
                "continent": "South America",
            },
            {
                "name": "Quito",
                "country": "EC",
                "lat": -0.1807,
                "lon": -78.4678,
                "continent": "South America",
            },
            {
                "name": "Caracas",
                "country": "VE",
                "lat": 10.4806,
                "lon": -66.9036,
                "continent": "South America",
            },
            {
                "name": "Belo Horizonte",
                "country": "BR",
                "lat": -19.8157,
                "lon": -43.9542,
                "continent": "South America",
            },
            {
                "name": "Brasília",
                "country": "BR",
                "lat": -15.8267,
                "lon": -47.9218,
                "continent": "South America",
            },
            {
                "name": "Porto Alegre",
                "country": "BR",
                "lat": -30.0346,
                "lon": -51.2177,
                "continent": "South America",
            },
            {
                "name": "Montevideo",
                "country": "UY",
                "lat": -34.9011,
                "lon": -56.1645,
                "continent": "South America",
            },
            {
                "name": "Asunción",
                "country": "PY",
                "lat": -25.2637,
                "lon": -57.5759,
                "continent": "South America",
            },
            {
                "name": "Córdoba",
                "country": "AR",
                "lat": -31.4201,
                "lon": -64.1888,
                "continent": "South America",
            },
            {
                "name": "Valparaíso",
                "country": "CL",
                "lat": -33.0458,
                "lon": -71.6197,
                "continent": "South America",
            },
            {
                "name": "Cali",
                "country": "CO",
                "lat": 3.4516,
                "lon": -76.5320,
                "continent": "South America",
            },
            {
                "name": "Curitiba",
                "country": "BR",
                "lat": -25.4284,
                "lon": -49.2733,
                "continent": "South America",
            },
            {
                "name": "Fortaleza",
                "country": "BR",
                "lat": -3.7319,
                "lon": -38.5267,
                "continent": "South America",
            },
        ]

        self.results = {
            "collection_timestamp": datetime.now().isoformat(),
            "api_key_used": True,
            "data_source": "OpenAQ Real Monitoring Stations",
            "cities_data": [],
            "summary": {
                "total_cities": len(self.cities),
                "successful_cities": 0,
                "cities_with_stations": 0,
                "cities_with_data": 0,
                "total_measurements": 0,
                "unique_pollutants": set(),
                "aqi_components_found": set(),
            },
        }

    def test_api_connection(self):
        """Test API connection with key - try multiple endpoints and auth methods"""
        print("=== TESTING OPENAQ API CONNECTION ===")

        # Try different combinations of URLs and headers
        for base_url in self.fallback_urls:
            print(f"Trying {base_url}...")

            all_headers = [self.headers] + self.fallback_headers

            for headers in all_headers:
                try:
                    # Try different endpoints
                    endpoints = ["/locations", "/measurements", "/"]

                    for endpoint in endpoints:
                        url = f"{base_url}{endpoint}"
                        params = {"limit": 1}

                        response = requests.get(
                            url, headers=headers, params=params, timeout=10
                        )

                        print(f"  {endpoint}: Status {response.status_code}")

                        if response.status_code == 200:
                            try:
                                data = response.json()
                                if (
                                    "results" in data
                                    or "data" in data
                                    or "locations" in data
                                ):
                                    print(f"SUCCESS: API authenticated successfully")
                                    print(f"   Working endpoint: {url}")
                                    print(f"   Auth header: {list(headers.keys())}")

                                    # Update working configuration
                                    self.base_url = base_url
                                    self.headers = headers

                                    # Try to get count
                                    meta = data.get("meta", {})
                                    found = meta.get(
                                        "found", meta.get("count", "unknown")
                                    )
                                    print(f"   Data available: {found}")
                                    return True
                            except:
                                pass
                        elif response.status_code == 401:
                            print(f"  Authentication failed with these headers")
                        elif response.status_code == 403:
                            print(
                                f"  Access forbidden - may need different permissions"
                            )
                        elif response.status_code == 410:
                            print(f"  Endpoint deprecated/gone")

                except Exception as e:
                    print(f"  Exception: {str(e)[:50]}...")
                    continue

        print("ERROR: Could not establish API connection with any configuration")
        return False

    def find_nearest_station(self, city):
        """Find nearest monitoring station for a city"""
        print(
            f"\n--- Searching for stations near {city['name']}, {city['country']} ---"
        )

        try:
            url = f"{self.base_url}/locations"
            params = {
                "coordinates": f"{city['lat']},{city['lon']}",
                "radius": 100000,  # 100km radius
                "limit": 5,
                "sort": "lastUpdated",
                "order": "desc",
            }

            response = requests.get(
                url, headers=self.headers, params=params, timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                stations = data.get("results", [])

                if stations:
                    # Get best station (most recent data)
                    best_station = stations[0]
                    station_id = best_station.get("id")
                    station_name = best_station.get("name", "Unknown")
                    last_updated = best_station.get("lastUpdated", "Unknown")
                    parameters = [
                        p.get("parameter") for p in best_station.get("parameters", [])
                    ]

                    print(f"  FOUND: {station_name} (ID: {station_id})")
                    print(f"  Last updated: {last_updated}")
                    print(f"  Parameters: {', '.join(parameters)}")
                    print(f"  Total nearby stations: {len(stations)}")

                    return {
                        "station_found": True,
                        "station_id": station_id,
                        "station_name": station_name,
                        "parameters": parameters,
                        "last_updated": last_updated,
                        "total_nearby": len(stations),
                    }
                else:
                    print(f"  NO STATIONS found within 100km")
                    return {"station_found": False, "reason": "No stations in range"}

            else:
                print(f"  API ERROR: {response.status_code}")
                return {"station_found": False, "error": response.status_code}

        except Exception as e:
            print(f"  EXCEPTION: {str(e)}")
            return {"station_found": False, "error": str(e)}

    def collect_recent_measurements(self, station_id, city_name, days_back=7):
        """Collect recent measurements from a station"""
        print(
            f"--- Collecting measurements for {city_name} (Station: {station_id}) ---"
        )

        try:
            # Get measurements from last week
            date_to = datetime.now()
            date_from = date_to - timedelta(days=days_back)

            url = f"{self.base_url}/measurements"
            params = {
                "location_id": station_id,
                "date_from": date_from.strftime("%Y-%m-%d"),
                "date_to": date_to.strftime("%Y-%m-%d"),
                "limit": 1000,
                "sort": "datetime",
                "order": "desc",
            }

            response = requests.get(
                url, headers=self.headers, params=params, timeout=20
            )

            if response.status_code == 200:
                data = response.json()
                measurements = data.get("results", [])

                if measurements:
                    # Organize by pollutant
                    pollutant_data = {}
                    for m in measurements:
                        param = m.get("parameter")
                        value = m.get("value")
                        unit = m.get("unit")
                        date_utc = m.get("date", {}).get("utc", None)

                        if param not in pollutant_data:
                            pollutant_data[param] = {
                                "parameter": param,
                                "unit": unit,
                                "measurements": [],
                            }

                        pollutant_data[param]["measurements"].append(
                            {"value": value, "datetime": date_utc}
                        )

                    # Calculate statistics for each pollutant
                    for param, data in pollutant_data.items():
                        values = [
                            m["value"]
                            for m in data["measurements"]
                            if m["value"] is not None
                        ]
                        if values:
                            data["count"] = len(values)
                            data["mean"] = np.mean(values)
                            data["min"] = np.min(values)
                            data["max"] = np.max(values)
                            data["std"] = np.std(values)
                            data["latest_value"] = data["measurements"][0]["value"]
                            data["latest_datetime"] = data["measurements"][0][
                                "datetime"
                            ]

                    print(f"  COLLECTED: {len(measurements)} measurements")
                    print(f"  Pollutants found: {', '.join(pollutant_data.keys())}")

                    for param, data in pollutant_data.items():
                        if "count" in data:
                            print(
                                f"    {param}: {data['count']} readings, latest={data['latest_value']} {data['unit']}"
                            )

                    return {
                        "measurements_found": True,
                        "total_measurements": len(measurements),
                        "pollutant_data": pollutant_data,
                        "date_range": f"{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}",
                    }
                else:
                    print(f"  NO MEASUREMENTS available")
                    return {
                        "measurements_found": False,
                        "reason": "No recent measurements",
                    }

            else:
                print(f"  API ERROR: {response.status_code}")
                return {"measurements_found": False, "error": response.status_code}

        except Exception as e:
            print(f"  EXCEPTION: {str(e)}")
            return {"measurements_found": False, "error": str(e)}

    def collect_all_cities_data(self):
        """Collect data for all 100 cities"""
        print("OPENAQ REAL DATA COLLECTION")
        print("Collecting authentic measured pollutant data from monitoring stations")
        print("=" * 80)

        # Test API connection first
        if not self.test_api_connection():
            print("ERROR: Cannot proceed without API connection")
            return False

        print(f"\nStarting collection for {len(self.cities)} cities...")

        for i, city in enumerate(self.cities, 1):
            print(
                f"\n[{i:3}/{len(self.cities)}] Processing {city['name']}, {city['country']} ({city['continent']})..."
            )

            city_result = {
                "city_name": city["name"],
                "country": city["country"],
                "continent": city["continent"],
                "coordinates": {"lat": city["lat"], "lon": city["lon"]},
                "collection_timestamp": datetime.now().isoformat(),
            }

            # Find nearest station
            station_result = self.find_nearest_station(city)
            city_result["station_search"] = station_result

            if station_result.get("station_found"):
                self.results["summary"]["cities_with_stations"] += 1

                # Collect measurements
                measurements_result = self.collect_recent_measurements(
                    station_result["station_id"], city["name"]
                )
                city_result["measurements"] = measurements_result

                if measurements_result.get("measurements_found"):
                    self.results["summary"]["cities_with_data"] += 1
                    self.results["summary"][
                        "total_measurements"
                    ] += measurements_result.get("total_measurements", 0)

                    # Track pollutants found
                    pollutant_data = measurements_result.get("pollutant_data", {})
                    for param in pollutant_data.keys():
                        self.results["summary"]["unique_pollutants"].add(param)

                        # Check for AQI components
                        param_lower = param.lower().replace(".", "").replace("_", "")
                        if "pm25" in param_lower or "pm2.5" in param_lower:
                            self.results["summary"]["aqi_components_found"].add("PM2.5")
                        elif "pm10" in param_lower:
                            self.results["summary"]["aqi_components_found"].add("PM10")
                        elif "no2" in param_lower:
                            self.results["summary"]["aqi_components_found"].add("NO2")
                        elif "o3" in param_lower or "ozone" in param_lower:
                            self.results["summary"]["aqi_components_found"].add("O3")
                        elif "co" in param_lower and "carbon" in param_lower:
                            self.results["summary"]["aqi_components_found"].add("CO")
                        elif "so2" in param_lower:
                            self.results["summary"]["aqi_components_found"].add("SO2")

                city_result["status"] = "success"
                self.results["summary"]["successful_cities"] += 1
            else:
                city_result["status"] = "no_station"

            self.results["cities_data"].append(city_result)

            # Rate limiting
            time.sleep(0.5)

            # Progress update every 10 cities
            if i % 10 == 0:
                stations_found = self.results["summary"]["cities_with_stations"]
                data_found = self.results["summary"]["cities_with_data"]
                print(f"\nProgress: {i}/{len(self.cities)} cities processed")
                print(
                    f"Stations found: {stations_found}/{i} ({stations_found/i*100:.1f}%)"
                )
                print(f"Data collected: {data_found}/{i} ({data_found/i*100:.1f}%)")

        return True

    def generate_summary_report(self):
        """Generate comprehensive summary of collection results"""
        print("\n" + "=" * 80)
        print("OPENAQ REAL DATA COLLECTION SUMMARY")
        print("=" * 80)

        summary = self.results["summary"]

        print(f"Cities processed: {summary['total_cities']}")
        print(f"Cities with monitoring stations: {summary['cities_with_stations']}")
        print(f"Cities with measurement data: {summary['cities_with_data']}")
        print(
            f"Success rate (stations): {summary['cities_with_stations']/summary['total_cities']*100:.1f}%"
        )
        print(
            f"Success rate (data): {summary['cities_with_data']/summary['total_cities']*100:.1f}%"
        )
        print(f"Total measurements collected: {summary['total_measurements']:,}")

        print(f"\nUnique pollutants found: {len(summary['unique_pollutants'])}")
        print(f"Pollutants: {', '.join(sorted(summary['unique_pollutants']))}")

        print(f"\nAQI Components Coverage:")
        aqi_standard = {"PM2.5", "PM10", "NO2", "O3", "CO", "SO2"}
        found_components = summary["aqi_components_found"]

        for component in aqi_standard:
            status = "YES" if component in found_components else "NO"
            print(f"  {status}: {component}")

        coverage_rate = len(found_components) / len(aqi_standard) * 100
        print(
            f"\nAQI Component Coverage: {len(found_components)}/6 ({coverage_rate:.1f}%)"
        )

        # Continental breakdown
        continental_stats = {}
        for city_data in self.results["cities_data"]:
            continent = city_data["continent"]
            if continent not in continental_stats:
                continental_stats[continent] = {
                    "total": 0,
                    "with_stations": 0,
                    "with_data": 0,
                }

            continental_stats[continent]["total"] += 1
            if city_data.get("station_search", {}).get("station_found"):
                continental_stats[continent]["with_stations"] += 1
            if city_data.get("measurements", {}).get("measurements_found"):
                continental_stats[continent]["with_data"] += 1

        print(f"\nContinental Breakdown:")
        for continent, stats in continental_stats.items():
            station_rate = stats["with_stations"] / stats["total"] * 100
            data_rate = stats["with_data"] / stats["total"] * 100
            print(
                f"  {continent}: {stats['with_data']}/{stats['total']} cities with data ({data_rate:.1f}%)"
            )

        return summary

    def save_results(self):
        """Save comprehensive results"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert sets to lists for JSON serialization
        self.results["summary"]["unique_pollutants"] = list(
            self.results["summary"]["unique_pollutants"]
        )
        self.results["summary"]["aqi_components_found"] = list(
            self.results["summary"]["aqi_components_found"]
        )

        # Full results
        full_results_file = (
            f"../final_dataset/OPENAQ_real_data_collection_{timestamp_str}.json"
        )
        with open(full_results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Summary report
        summary_report = {
            "collection_timestamp": self.results["collection_timestamp"],
            "data_source": "OpenAQ Real Monitoring Stations",
            "data_authenticity": "100% Real measured data from ground monitoring stations",
            "api_authentication": "Authenticated with OpenAQ API key",
            "summary_statistics": self.results["summary"],
            "methodology": {
                "station_selection": "Nearest station within 100km radius",
                "data_collection": "Last 7 days of measurements",
                "pollutant_coverage": "All available parameters from each station",
            },
        }

        summary_file = (
            f"../final_dataset/OPENAQ_collection_summary_{timestamp_str}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2, default=str)

        print(f"\nResults saved:")
        print(f"  Full data: {full_results_file}")
        print(f"  Summary: {summary_file}")

        return full_results_file, summary_file


def main():
    """Main execution"""
    api_key = "8c71a560478a03671edd9be444571ba70afbe82d9fd3a9d9b2612e8d806287f8"

    print("OPENAQ REAL DATA COLLECTOR")
    print("Collecting authentic measured air pollutant data")
    print("Data Source: Real monitoring stations via OpenAQ API")
    print("=" * 80)

    collector = OpenAQRealDataCollector(api_key)

    # Collect data
    success = collector.collect_all_cities_data()

    if success:
        # Generate summary
        summary = collector.generate_summary_report()

        # Save results
        full_file, summary_file = collector.save_results()

        print(f"\nSUCCESS: OpenAQ real data collection complete!")
        print(f"Collected authentic measured pollutant data from monitoring stations")
        print(
            f"Cities with data: {summary['cities_with_data']}/{summary['total_cities']}"
        )
        print(f"Total measurements: {summary['total_measurements']:,}")
        print(f"AQI components found: {len(summary['aqi_components_found'])}/6")
    else:
        print(f"\nERROR: Data collection failed")


if __name__ == "__main__":
    main()
