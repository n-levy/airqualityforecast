#!/usr/bin/env python3
"""
Stage 5 Cities Configuration Loader
====================================

Loads the 100 cities from Stage 5 configuration (20 cities per continent)
for use in Stage 6 ETL scripts.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_stage5_cities() -> Dict[str, Dict[str, Any]]:
    """
    Load the 100 cities from Stage 5 configuration.
    
    Returns:
        Dictionary with city names as keys and city metadata as values
    """
    config_file = Path("stage_5/scripts/stage_5/config/cities_config.json")
    
    if not config_file.exists():
        raise FileNotFoundError(f"Stage 5 cities configuration not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        continent_cities = json.load(f)
    
    # Flatten the continental structure into a single dictionary
    cities = {}
    continent_mapping = {
        "asia": "Asia",
        "europe": "Europe", 
        "north_america": "North America",
        "africa": "Africa",
        "south_america": "South America"
    }
    
    for continent_key, city_list in continent_cities.items():
        continent_name = continent_mapping.get(continent_key, continent_key.title())
        
        for city_data in city_list:
            city_name = city_data["name"]
            
            # Enhanced city metadata with additional fields for Stage 6
            cities[city_name] = {
                "country": city_data["country"],
                "lat": city_data["lat"],
                "lon": city_data["lon"],
                "continent": continent_name,
                "aqi_standard": city_data.get("aqi_standard", "WHO"),
                # Add timezone estimation based on longitude
                "timezone": _estimate_timezone(city_data["lat"], city_data["lon"]),
                # Add population estimates (these would be enhanced with real data)
                "population": _estimate_population(city_name, city_data["country"]),
                # Add elevation estimates
                "elevation": _estimate_elevation(city_name, city_data["lat"], city_data["lon"]),
                # Add country codes
                "country_code": _get_country_code(city_data["country"])
            }
    
    return cities


def _estimate_timezone(lat: float, lon: float) -> str:
    """Estimate timezone based on coordinates."""
    # Simple timezone estimation based on longitude
    # This is a rough approximation - real implementation should use timezone libraries
    utc_offset = round(lon / 15)
    
    # Common timezone mappings based on regions
    timezone_map = {
        # Asia
        (20, 90, 60, 150): {  # Rough bounds for major Asian cities
            77: "Asia/Kolkata",    # Delhi area
            74: "Asia/Karachi",    # Lahore/Karachi
            116: "Asia/Shanghai",  # Beijing/Shanghai
            90: "Asia/Dhaka",      # Dhaka
            88: "Asia/Kolkata",    # Kolkata
            100: "Asia/Bangkok",   # Bangkok
            106: "Asia/Jakarta",   # Jakarta/Ho_Chi_Minh
            120: "Asia/Manila",    # Manila
            105: "Asia/Bangkok",   # Hanoi  
            126: "Asia/Seoul",     # Seoul
            121: "Asia/Taipei",    # Taipei
            76: "Asia/Almaty",     # Almaty
            69: "Asia/Tashkent",   # Tashkent
            51: "Asia/Tehran",     # Tehran
        },
        # Europe  
        (35, -15, 70, 40): {
            21: "Europe/Skopje",
            18: "Europe/Sarajevo", 
            23: "Europe/Sofia",
            26: "Europe/Bucharest",
            20: "Europe/Belgrade",
            21: "Europe/Warsaw",
            19: "Europe/Warsaw",
            14: "Europe/Prague",
            19: "Europe/Budapest",
            9: "Europe/Rome",
            7: "Europe/Rome",
            14: "Europe/Rome",
            23: "Europe/Athens",
            -3: "Europe/Madrid",
            2: "Europe/Paris",
            2: "Europe/Paris", 
            -0: "Europe/London",
            13: "Europe/Berlin",
            4: "Europe/Amsterdam"
        }
    }
    
    # Default timezone estimation
    if -8 <= utc_offset <= -5:
        return "America/New_York"
    elif -12 <= utc_offset <= -9:
        return "America/Los_Angeles"  
    elif -6 <= utc_offset <= -3:
        return "America/Mexico_City"
    elif 0 <= utc_offset <= 3:
        return "Europe/London"
    elif 4 <= utc_offset <= 7:
        return "Asia/Kolkata"
    elif 8 <= utc_offset <= 10:
        return "Asia/Shanghai"
    else:
        return "UTC"


def _estimate_population(city_name: str, country: str) -> int:
    """Estimate population for major cities."""
    # Population estimates for major cities (in thousands)
    population_estimates = {
        # Asia - Major cities
        "Delhi": 32900000,
        "Lahore": 11100000, 
        "Beijing": 21540000,
        "Dhaka": 22400000,
        "Mumbai": 20410000,
        "Karachi": 14910000,
        "Shanghai": 27060000,
        "Kolkata": 14850000,
        "Bangkok": 10540000,
        "Jakarta": 10770000,
        "Manila": 13480000,
        "Ho Chi Minh City": 9000000,
        "Hanoi": 8100000,
        "Seoul": 9720000,
        "Taipei": 2650000,
        "Ulaanbaatar": 1520000,
        "Almaty": 1980000,
        "Tashkent": 2570000,
        "Tehran": 9270000,
        "Kabul": 4435000,
        
        # Europe
        "Skopje": 540000,
        "Sarajevo": 400000,
        "Sofia": 1400000,
        "Plovdiv": 350000,
        "Bucharest": 1830000,
        "Belgrade": 1390000,
        "Warsaw": 1790000,
        "Krakow": 780000,
        "Prague": 1320000,
        "Budapest": 1750000,
        "Milan": 1400000,
        "Turin": 870000,
        "Naples": 970000,
        "Athens": 3150000,
        "Madrid": 6750000,
        "Barcelona": 5570000,
        "Paris": 11020000,
        "London": 9540000,
        "Berlin": 3670000,
        "Amsterdam": 1150000,
        
        # North America
        "Mexicali": 1050000,
        "Mexico City": 21580000,
        "Guadalajara": 5250000,
        "Tijuana": 1810000,
        "Monterrey": 4690000,
        "Los Angeles": 3970000,
        "Fresno": 540000,
        "Phoenix": 1690000,
        "Houston": 2320000,
        "New York": 8380000,
        "Chicago": 2710000,
        "Denver": 720000,
        "Detroit": 670000,
        "Atlanta": 510000,
        "Philadelphia": 1580000,
        "Toronto": 2930000,
        "Montreal": 1780000,
        "Vancouver": 630000,
        "Calgary": 1340000,
        "Ottawa": 1000000,
        
        # Africa
        "N'Djamena": 1605000,
        "Cairo": 20900000,
        "Lagos": 15390000,
        "Accra": 2560000,
        "Khartoum": 5900000,
        "Kampala": 3650000,
        "Nairobi": 4920000,
        "Abidjan": 5515000,
        "Bamako": 2810000,
        "Ouagadougou": 2415000,
        "Dakar": 3140000,
        "Kinshasa": 17070000,
        "Casablanca": 3750000,
        "Johannesburg": 4950000,
        "Addis Ababa": 5230000,
        "Dar es Salaam": 6370000,
        "Algiers": 3410000,
        "Tunis": 2390000,
        "Maputo": 1100000,
        "Cape Town": 4620000,
        
        # South America
        "Lima": 10720000,
        "Santiago": 6160000,
        "São Paulo": 22430000,
        "Rio de Janeiro": 6780000,
        "Bogotá": 11340000,
        "La Paz": 840000,
        "Medellín": 2570000,
        "Buenos Aires": 15370000,
        "Quito": 2780000,
        "Caracas": 2940000,
        "Belo Horizonte": 2530000,
        "Brasília": 3140000,
        "Porto Alegre": 1490000,
        "Montevideo": 1740000,
        "Asunción": 3280000,
        "Córdoba": 1490000,
        "Valparaíso": 310000,
        "Cali": 2230000,
        "Curitiba": 1960000,
        "Fortaleza": 2700000
    }
    
    return population_estimates.get(city_name, 1000000)  # Default 1M


def _estimate_elevation(city_name: str, lat: float, lon: float) -> int:
    """Estimate elevation for cities."""
    # Elevation estimates in meters
    elevation_estimates = {
        # High altitude cities
        "La Paz": 3500,
        "Mexico City": 2240,
        "Bogotá": 2640,
        "Quito": 2850,
        "Tehran": 1190,
        "Almaty": 800,
        "Ulaanbaatar": 1350,
        "Kabul": 1790,
        "Denver": 1655,
        "Madrid": 650,
        "Brasília": 1170,
        
        # Medium altitude
        "São Paulo": 760,
        "Curitiba": 935,
        "Monterrey": 540,
        "Calgary": 1045,
        "Tashkent": 455,
        "Turin": 240,
        "Krakow": 219,
        "Prague": 180,
        
        # Low altitude/coastal cities
        "Amsterdam": -2,
        "Bangkok": 1,
        "Shanghai": 4,
        "Tokyo": 6,
        "Manila": 16,
        "Mumbai": 14,
        "Rio de Janeiro": 2,
        "Lagos": 39,
        "Dar es Salaam": 56,
        "Cape Town": 42
    }
    
    return elevation_estimates.get(city_name, 100)  # Default 100m


def _get_country_code(country: str) -> str:
    """Get ISO country code for country."""
    country_codes = {
        "India": "IN",
        "Pakistan": "PK", 
        "China": "CN",
        "Bangladesh": "BD",
        "Thailand": "TH",
        "Indonesia": "ID",
        "Philippines": "PH",
        "Vietnam": "VN",
        "South Korea": "KR",
        "Taiwan": "TW",
        "Mongolia": "MN",
        "Kazakhstan": "KZ",
        "Uzbekistan": "UZ",
        "Iran": "IR",
        "Afghanistan": "AF",
        
        "North Macedonia": "MK",
        "Bosnia and Herzegovina": "BA",
        "Bulgaria": "BG",
        "Romania": "RO",
        "Serbia": "RS",
        "Poland": "PL",
        "Czech Republic": "CZ",
        "Hungary": "HU",
        "Italy": "IT",
        "Greece": "GR",
        "Spain": "ES",
        "France": "FR",
        "UK": "GB",
        "Germany": "DE",
        "Netherlands": "NL",
        
        "Mexico": "MX",
        "USA": "US",
        "Canada": "CA",
        
        "Chad": "TD",
        "Egypt": "EG",
        "Nigeria": "NG",
        "Ghana": "GH",
        "Sudan": "SD",
        "Uganda": "UG",
        "Kenya": "KE",
        "Côte d'Ivoire": "CI",
        "Mali": "ML",
        "Burkina Faso": "BF",
        "Senegal": "SN",
        "DR Congo": "CD",
        "Morocco": "MA",
        "South Africa": "ZA",
        "Ethiopia": "ET",
        "Tanzania": "TZ",
        "Algeria": "DZ",
        "Tunisia": "TN",
        "Mozambique": "MZ",
        
        "Peru": "PE",
        "Chile": "CL",
        "Brazil": "BR",
        "Colombia": "CO",
        "Bolivia": "BO",
        "Argentina": "AR",
        "Ecuador": "EC",
        "Venezuela": "VE",
        "Uruguay": "UY",
        "Paraguay": "PY"
    }
    
    return country_codes.get(country, "XX")


def get_stage5_cities_count() -> Dict[str, int]:
    """Get count of cities per continent from Stage 5."""
    cities = load_stage5_cities()
    continent_counts = {}
    
    for city_data in cities.values():
        continent = city_data["continent"]
        continent_counts[continent] = continent_counts.get(continent, 0) + 1
    
    return continent_counts


if __name__ == "__main__":
    # Test the loader
    cities = load_stage5_cities()
    print(f"Loaded {len(cities)} cities from Stage 5")
    
    continent_counts = get_stage5_cities_count()
    print("Cities per continent:")
    for continent, count in continent_counts.items():
        print(f"  {continent}: {count}")
    
    # Show first few cities
    print("\nFirst 5 cities:")
    for i, (city_name, city_data) in enumerate(cities.items()):
        if i >= 5:
            break
        print(f"  {city_name}: {city_data['country']}, {city_data['continent']}")