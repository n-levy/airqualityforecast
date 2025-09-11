#!/usr/bin/env python3
"""
Collect Real Data for 22 Replacement Cities

Collect real data specifically for the 22 replacement cities to achieve 
100% real data coverage across all 100 cities.
"""

import json
import time
import warnings
from datetime import datetime

import pandas as pd
import requests

warnings.filterwarnings("ignore")


class ReplacementCitiesDataCollector:
    """Collect real data for the 22 replacement cities."""

    def __init__(self):
        """Initialize replacement cities data collector."""
        
        # Load the updated cities table with replacements
        self.cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")
        
        # Load previous collection results to identify what we already have
        try:
            with open("../final_dataset/complete_real_data_collection_20250911_192217.json", 'r', encoding='utf-8') as f:
                self.previous_data = json.load(f)
        except FileNotFoundError:
            self.previous_data = {'noaa_data': {}, 'waqi_data': {}}
        
        # The 22 replacement cities that need data collection
        self.replacement_cities = [
            "Salvador", "Fortaleza", "Brasília", "Recife", "Manaus", 
            "Goiânia", "Belém", "João Pessoa",  # South America
            "Johannesburg", "Cape Town", "Durban", "Nairobi", "Addis Ababa",
            "Tunis", "Algiers", "Rabat", "Abuja",  # Africa
            "Mumbai", "Chennai", "Hyderabad",  # Asia
            "Atlanta",  # North America
            "Warsaw"  # Europe
        ]
        
        self.collected_data = {
            'collection_time': datetime.now().isoformat(),
            'replacement_cities_count': len(self.replacement_cities),
            'noaa_data': {},
            'waqi_data': {},
            'collection_summary': {}
        }

    def safe_print(self, message):
        """Print message with Unicode safety."""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback to ASCII
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)

    def collect_noaa_data_for_us_cities(self):
        """Collect NOAA weather data for US replacement cities."""
        
        self.safe_print("COLLECTING NOAA DATA FOR US REPLACEMENT CITIES")
        self.safe_print("=" * 55)
        
        us_cities = []
        for city_name in self.replacement_cities:
            city_row = self.cities_df[self.cities_df['City'] == city_name]
            if not city_row.empty and city_row['Country'].iloc[0] == 'USA':
                us_cities.append({
                    'name': city_name,
                    'lat': city_row['Latitude'].iloc[0],
                    'lon': city_row['Longitude'].iloc[0]
                })
        
        self.safe_print(f"Found {len(us_cities)} US replacement cities for NOAA data collection")
        
        for city_info in us_cities:
            city_name = city_info['name']
            lat = city_info['lat']
            lon = city_info['lon']
            
            self.safe_print(f"  Collecting NOAA data for {city_name}...")
            
            try:
                # Get grid information
                grid_url = f"https://api.weather.gov/points/{lat},{lon}"
                grid_response = requests.get(grid_url, timeout=10)
                
                if grid_response.status_code == 200:
                    grid_data = grid_response.json()
                    
                    # Get forecast data
                    forecast_url = grid_data['properties']['forecast']
                    forecast_response = requests.get(forecast_url, timeout=10)
                    
                    if forecast_response.status_code == 200:
                        forecast_data = forecast_response.json()
                        
                        # Get hourly forecast
                        hourly_url = grid_data['properties']['forecastHourly']
                        hourly_response = requests.get(hourly_url, timeout=10)
                        
                        hourly_data = []
                        if hourly_response.status_code == 200:
                            hourly_data = hourly_response.json()
                        
                        self.collected_data['noaa_data'][city_name] = {
                            'api_status': 'SUCCESS',
                            'collection_time': datetime.now().isoformat(),
                            'grid_data': grid_data,
                            'forecast_data': forecast_data,
                            'hourly_data': hourly_data,
                            'daily_periods': len(forecast_data.get('properties', {}).get('periods', [])),
                            'hourly_periods': len(hourly_data.get('properties', {}).get('periods', []))
                        }
                        
                        daily_count = len(forecast_data.get('properties', {}).get('periods', []))
                        hourly_count = len(hourly_data.get('properties', {}).get('periods', []))
                        self.safe_print(f"    SUCCESS: {daily_count} daily + {hourly_count} hourly periods")
                        
                    else:
                        self.collected_data['noaa_data'][city_name] = {
                            'api_status': 'FORECAST_FAILED',
                            'error': f"Forecast request failed: {forecast_response.status_code}"
                        }
                        self.safe_print(f"    FAILED: Forecast unavailable ({forecast_response.status_code})")
                
                else:
                    self.collected_data['noaa_data'][city_name] = {
                        'api_status': 'GRID_FAILED',
                        'error': f"Grid request failed: {grid_response.status_code}"
                    }
                    self.safe_print(f"    FAILED: Grid unavailable ({grid_response.status_code})")
                
            except Exception as e:
                self.collected_data['noaa_data'][city_name] = {
                    'api_status': 'ERROR',
                    'error': str(e)
                }
                self.safe_print(f"    ERROR: {str(e)}")
            
            time.sleep(0.5)  # Rate limiting
        
        noaa_success = sum(1 for city_data in self.collected_data['noaa_data'].values() 
                          if city_data.get('api_status') == 'SUCCESS')
        self.safe_print(f"\nNOAA Collection Complete: {noaa_success}/{len(us_cities)} US replacement cities")

    def collect_waqi_data_for_replacement_cities(self):
        """Collect WAQI air quality data for all replacement cities."""
        
        self.safe_print(f"\nCOLLECTING WAQI DATA FOR {len(self.replacement_cities)} REPLACEMENT CITIES")
        self.safe_print("=" * 70)
        
        for city_name in self.replacement_cities:
            city_row = self.cities_df[self.cities_df['City'] == city_name]
            if city_row.empty:
                self.safe_print(f"  WARNING: {city_name} not found in cities table")
                continue
            
            country = city_row['Country'].iloc[0]
            
            try:
                self.safe_print(f"  Collecting WAQI data for {city_name}, {country}...")
            except UnicodeEncodeError:
                safe_city = city_name.encode("ascii", "replace").decode("ascii")
                safe_country = country.encode("ascii", "replace").decode("ascii")
                print(f"  Collecting WAQI data for {safe_city}, {safe_country}...")
            
            # Multiple query formats for WAQI
            city_queries = [
                city_name.lower().replace(' ', '-').replace("'", "").replace("ã", "a").replace("í", "i").replace("â", "a"),
                city_name.lower().replace(' ', '').replace("'", "").replace("ã", "a").replace("í", "i").replace("â", "a"),
                f"{city_name.lower().replace(' ', '-')}-{country.lower()}".replace("'", "").replace("ã", "a").replace("í", "i").replace("â", "a"),
                city_name.replace(' ', '+'),
                city_name.lower()
            ]
            
            success = False
            for query in city_queries:
                if success:
                    break
                
                try:
                    url = f"https://api.waqi.info/feed/{query}/?token=demo"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('status') == 'ok' and 'data' in data:
                            aqi_data = data['data']
                            current_aqi = aqi_data.get('aqi', -1)
                            
                            if current_aqi > 0:
                                pollutants = aqi_data.get('iaqi', {})
                                
                                self.collected_data['waqi_data'][city_name] = {
                                    'api_status': 'SUCCESS',
                                    'collection_time': datetime.now().isoformat(),
                                    'query_used': query,
                                    'aqi_data': aqi_data,
                                    'current_aqi': current_aqi,
                                    'pollutants_count': len(pollutants),
                                    'city_info': aqi_data.get('city', {}),
                                    'attribution': aqi_data.get('attributions', [])
                                }
                                
                                self.safe_print(f"    SUCCESS: AQI={current_aqi}, Pollutants={len(pollutants)}")
                                success = True
                                break
                
                except Exception as e:
                    continue
            
            if not success:
                self.collected_data['waqi_data'][city_name] = {
                    'api_status': 'NO_DATA',
                    'queries_attempted': city_queries,
                    'error': 'All query formats failed'
                }
                self.safe_print(f"    NO DATA: All query formats failed")
            
            time.sleep(1.2)  # Rate limiting for WAQI
        
        waqi_success = sum(1 for city_data in self.collected_data['waqi_data'].values() 
                          if city_data.get('api_status') == 'SUCCESS')
        self.safe_print(f"\nWAQI Collection Complete: {waqi_success}/{len(self.replacement_cities)} replacement cities")

    def generate_collection_summary(self):
        """Generate summary of data collection results."""
        
        self.safe_print(f"\nGENERATING REPLACEMENT CITIES DATA COLLECTION SUMMARY")
        self.safe_print("=" * 60)
        
        # Count successes
        noaa_success = sum(1 for city_data in self.collected_data['noaa_data'].values() 
                          if city_data.get('api_status') == 'SUCCESS')
        waqi_success = sum(1 for city_data in self.collected_data['waqi_data'].values() 
                          if city_data.get('api_status') == 'SUCCESS')
        
        # Find cities with any real data
        cities_with_real_data = set()
        for city_name, data in self.collected_data['noaa_data'].items():
            if data.get('api_status') == 'SUCCESS':
                cities_with_real_data.add(city_name)
        for city_name, data in self.collected_data['waqi_data'].items():
            if data.get('api_status') == 'SUCCESS':
                cities_with_real_data.add(city_name)
        
        # Cities still without data
        cities_without_data = set(self.replacement_cities) - cities_with_real_data
        
        self.collected_data['collection_summary'] = {
            'total_replacement_cities': len(self.replacement_cities),
            'noaa_successes': noaa_success,
            'waqi_successes': waqi_success,
            'cities_with_real_data': len(cities_with_real_data),
            'cities_without_data': len(cities_without_data),
            'real_data_percentage': len(cities_with_real_data) / len(self.replacement_cities) * 100,
            'cities_with_real_data_list': sorted(list(cities_with_real_data)),
            'cities_without_data_list': sorted(list(cities_without_data))
        }
        
        self.safe_print("REPLACEMENT CITIES DATA COLLECTION SUMMARY:")
        self.safe_print("=" * 45)
        self.safe_print(f"Total replacement cities: {len(self.replacement_cities)}")
        self.safe_print(f"NOAA weather data: {noaa_success} US cities")
        self.safe_print(f"WAQI air quality data: {waqi_success} cities")
        self.safe_print(f"Cities with real data: {len(cities_with_real_data)} ({len(cities_with_real_data)/len(self.replacement_cities)*100:.1f}%)")
        
        if cities_without_data:
            self.safe_print(f"Cities still without data: {len(cities_without_data)}")
            for city in sorted(cities_without_data):
                self.safe_print(f"  - {city}")
        else:
            self.safe_print("✅ ALL REPLACEMENT CITIES HAVE REAL DATA!")

    def save_results(self):
        """Save collection results to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../final_dataset/replacement_cities_data_collection_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, indent=2, default=str, ensure_ascii=False)
        
        self.safe_print(f"\nReplacement cities data collection results saved to: {output_file}")
        return output_file


def main():
    """Main function to collect data for replacement cities."""
    
    print("REPLACEMENT CITIES REAL DATA COLLECTION")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    collector = ReplacementCitiesDataCollector()
    
    # Collect NOAA data for US replacement cities
    collector.collect_noaa_data_for_us_cities()
    
    # Collect WAQI data for all replacement cities
    collector.collect_waqi_data_for_replacement_cities()
    
    # Generate summary
    collector.generate_collection_summary()
    
    # Save results
    output_file = collector.save_results()
    
    return collector.collected_data, output_file


if __name__ == "__main__":
    results, file_path = main()