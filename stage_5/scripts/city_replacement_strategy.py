#!/usr/bin/env python3
"""
City Replacement Strategy

Replace cities without reliable real data sources with backup cities from the same 
continent that have low air quality and reliable API data availability.
"""

import json
import requests
import time
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")


class CityReplacementStrategy:
    """Identify and replace cities without reliable real data sources."""

    def __init__(self):
        """Initialize city replacement strategy."""
        
        # Current 100 worst cities
        self.current_cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")
        
        # Backup cities by continent (cities with known poor air quality)
        self.backup_cities = {
            'Asia': [
                {'name': 'Mumbai', 'country': 'India', 'lat': 19.0760, 'lon': 72.8777},
                {'name': 'Karachi', 'country': 'Pakistan', 'lat': 24.8607, 'lon': 67.0011},
                {'name': 'Chengdu', 'country': 'China', 'lat': 30.5728, 'lon': 104.0668},
                {'name': 'Jakarta', 'country': 'Indonesia', 'lat': -6.2088, 'lon': 106.8456},
                {'name': 'Manila', 'country': 'Philippines', 'lat': 14.5995, 'lon': 120.9842},
                {'name': 'Hanoi', 'country': 'Vietnam', 'lat': 21.0285, 'lon': 105.8542},
                {'name': 'Bangkok', 'country': 'Thailand', 'lat': 13.7563, 'lon': 100.5018},
                {'name': 'Seoul', 'country': 'South Korea', 'lat': 37.5665, 'lon': 126.9780},
                {'name': 'Bishkek', 'country': 'Kyrgyzstan', 'lat': 42.8746, 'lon': 74.5698},
                {'name': 'Tashkent', 'country': 'Uzbekistan', 'lat': 41.2995, 'lon': 69.2401}
            ],
            'Africa': [
                {'name': 'Johannesburg', 'country': 'South Africa', 'lat': -26.2041, 'lon': 28.0473},
                {'name': 'Nairobi', 'country': 'Kenya', 'lat': -1.2921, 'lon': 36.8219},
                {'name': 'Addis Ababa', 'country': 'Ethiopia', 'lat': 9.1450, 'lon': 40.4897},
                {'name': 'Tunis', 'country': 'Tunisia', 'lat': 36.8065, 'lon': 10.1815},
                {'name': 'Algiers', 'country': 'Algeria', 'lat': 36.7372, 'lon': 3.0863},
                {'name': 'Rabat', 'country': 'Morocco', 'lat': 34.0209, 'lon': -6.8416},
                {'name': 'Luanda', 'country': 'Angola', 'lat': -8.8390, 'lon': 13.2894},
                {'name': 'Kano', 'country': 'Nigeria', 'lat': 12.0022, 'lon': 8.5920},
                {'name': 'Abuja', 'country': 'Nigeria', 'lat': 9.0765, 'lon': 7.3986},
                {'name': 'Maputo', 'country': 'Mozambique', 'lat': -25.9692, 'lon': 32.5732}
            ],
            'Europe': [
                {'name': 'Warsaw', 'country': 'Poland', 'lat': 52.2297, 'lon': 21.0122},
                {'name': 'Prague', 'country': 'Czech Republic', 'lat': 50.0755, 'lon': 14.4378},
                {'name': 'Budapest', 'country': 'Hungary', 'lat': 47.4979, 'lon': 19.0402},
                {'name': 'Athens', 'country': 'Greece', 'lat': 37.9838, 'lon': 23.7275},
                {'name': 'Rome', 'country': 'Italy', 'lat': 41.9028, 'lon': 12.4964},
                {'name': 'Madrid', 'country': 'Spain', 'lat': 40.4168, 'lon': -3.7038},
                {'name': 'Ljubljana', 'country': 'Slovenia', 'lat': 46.0569, 'lon': 14.5058},
                {'name': 'Bratislava', 'country': 'Slovakia', 'lat': 48.1486, 'lon': 17.1077},
                {'name': 'Podgorica', 'country': 'Montenegro', 'lat': 42.4304, 'lon': 19.2594},
                {'name': 'Zagreb', 'country': 'Croatia', 'lat': 45.8150, 'lon': 15.9819}
            ],
            'North_America': [
                {'name': 'Mexico City', 'country': 'Mexico', 'lat': 19.4326, 'lon': -99.1332},
                {'name': 'Los Angeles', 'country': 'USA', 'lat': 34.0522, 'lon': -118.2437},
                {'name': 'Phoenix', 'country': 'USA', 'lat': 33.4484, 'lon': -112.0740},
                {'name': 'Houston', 'country': 'USA', 'lat': 29.7604, 'lon': -95.3698},
                {'name': 'Chicago', 'country': 'USA', 'lat': 41.8781, 'lon': -87.6298},
                {'name': 'New York', 'country': 'USA', 'lat': 40.7128, 'lon': -74.0060},
                {'name': 'Toronto', 'country': 'Canada', 'lat': 43.6532, 'lon': -79.3832},
                {'name': 'Vancouver', 'country': 'Canada', 'lat': 49.2827, 'lon': -123.1207},
                {'name': 'Montreal', 'country': 'Canada', 'lat': 45.5017, 'lon': -73.5673},
                {'name': 'Atlanta', 'country': 'USA', 'lat': 33.7490, 'lon': -84.3880}
            ],
            'South_America': [
                {'name': 'São Paulo', 'country': 'Brazil', 'lat': -23.5505, 'lon': -46.6333},
                {'name': 'Rio de Janeiro', 'country': 'Brazil', 'lat': -22.9068, 'lon': -43.1729},
                {'name': 'Lima', 'country': 'Peru', 'lat': -12.0464, 'lon': -77.0428},
                {'name': 'Bogotá', 'country': 'Colombia', 'lat': 4.7110, 'lon': -74.0721},
                {'name': 'Santiago', 'country': 'Chile', 'lat': -33.4489, 'lon': -70.6693},
                {'name': 'Buenos Aires', 'country': 'Argentina', 'lat': -34.6118, 'lon': -58.3960},
                {'name': 'Caracas', 'country': 'Venezuela', 'lat': 10.4806, 'lon': -66.9036},
                {'name': 'Montevideo', 'country': 'Uruguay', 'lat': -34.9011, 'lon': -56.1645},
                {'name': 'Asunción', 'country': 'Paraguay', 'lat': -25.2637, 'lon': -57.5759},
                {'name': 'La Paz', 'country': 'Bolivia', 'lat': -16.5000, 'lon': -68.1193}
            ]
        }
        
        self.replacement_results = {
            'cities_tested': {},
            'cities_to_replace': {},
            'recommended_replacements': {},
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'strategy': '100% real data priority with city replacement'
            }
        }

    def test_backup_city_data_availability(self, city_info, continent):
        """Test data availability for a backup city."""
        
        city_name = city_info['name']
        country = city_info['country']
        lat = city_info['lat']
        lon = city_info['lon']
        
        print(f"    Testing {city_name}, {country}...")
        
        data_sources = {
            'noaa_available': False,
            'waqi_available': False,
            'data_quality_score': 0
        }
        
        # Test NOAA if US city
        if country == 'USA':
            try:
                grid_url = f"https://api.weather.gov/points/{lat},{lon}"
                response = requests.get(grid_url, timeout=5)
                if response.status_code == 200:
                    data_sources['noaa_available'] = True
                    data_sources['data_quality_score'] += 50
                    print(f"      NOAA: Available")
                else:
                    print(f"      NOAA: Unavailable (Status {response.status_code})")
            except Exception as e:
                print(f"      NOAA: Error - {str(e)}")
            
            time.sleep(0.5)
        
        # Test WAQI
        try:
            city_queries = [
                city_name.lower().replace(' ', '-').replace("'", ""),
                city_name.lower().replace(' ', '').replace("'", ""),
                f"{city_name.lower().replace(' ', '-')}-{country.lower()}"
            ]
            
            waqi_success = False
            for query in city_queries:
                if waqi_success:
                    break
                    
                url = f"https://api.waqi.info/feed/{query}/?token=demo"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok' and 'data' in data:
                        aqi_data = data['data']
                        current_aqi = aqi_data.get('aqi', -1)
                        
                        if current_aqi > 0:
                            data_sources['waqi_available'] = True
                            data_sources['data_quality_score'] += 50
                            data_sources['current_aqi'] = current_aqi
                            waqi_success = True
                            print(f"      WAQI: Available (AQI={current_aqi})")
                        
            if not waqi_success:
                print(f"      WAQI: Unavailable")
                
        except Exception as e:
            print(f"      WAQI: Error - {str(e)}")
        
        time.sleep(1.0)  # Rate limiting
        
        return data_sources

    def analyze_collection_results(self, collection_file_path):
        """Analyze collection results to identify cities needing replacement."""
        
        print("ANALYZING COLLECTION RESULTS")
        print("=" * 35)
        
        try:
            with open(collection_file_path, 'r') as f:
                collection_data = json.load(f)
        except FileNotFoundError:
            print(f"Collection results file not found: {collection_file_path}")
            return {}
        
        cities_needing_replacement = {}
        
        # Analyze NOAA results
        noaa_data = collection_data.get('noaa_data', {})
        noaa_failed = [city for city, data in noaa_data.items() 
                      if data.get('api_status') != 'SUCCESS']
        
        # Analyze WAQI results  
        waqi_data = collection_data.get('waqi_data', {})
        waqi_failed = [city for city, data in waqi_data.items() 
                      if data.get('api_status') != 'SUCCESS']
        
        # Find cities with no real data sources
        all_cities = set(self.current_cities_df['City'].tolist())
        cities_with_data = set()
        
        # Add cities with successful NOAA data
        cities_with_data.update([city for city, data in noaa_data.items() 
                               if data.get('api_status') == 'SUCCESS'])
        
        # Add cities with successful WAQI data
        cities_with_data.update([city for city, data in waqi_data.items() 
                               if data.get('api_status') == 'SUCCESS'])
        
        # Cities needing replacement
        cities_without_data = all_cities - cities_with_data
        
        # Group by continent
        for city in cities_without_data:
            city_row = self.current_cities_df[self.current_cities_df['City'] == city].iloc[0]
            continent = city_row['Continent']
            
            if continent not in cities_needing_replacement:
                cities_needing_replacement[continent] = []
            
            cities_needing_replacement[continent].append({
                'city': city,
                'country': city_row['Country'],
                'reason': 'No reliable real data sources available'
            })
        
        print(f"Cities needing replacement:")
        for continent, cities in cities_needing_replacement.items():
            print(f"  {continent}: {len(cities)} cities")
            for city_info in cities:
                print(f"    - {city_info['city']}, {city_info['country']}")
        
        return cities_needing_replacement

    def find_replacement_cities(self, cities_needing_replacement):
        """Find replacement cities with reliable data sources."""
        
        print(f"\nFINDING REPLACEMENT CITIES")
        print("=" * 32)
        
        recommended_replacements = {}
        
        for continent, failed_cities in cities_needing_replacement.items():
            print(f"\nTesting backup cities for {continent}:")
            
            backup_cities = self.backup_cities.get(continent, [])
            tested_backups = []
            
            for backup_city in backup_cities:
                # Skip if already in current dataset
                if backup_city['name'] in self.current_cities_df['City'].values:
                    continue
                
                data_availability = self.test_backup_city_data_availability(backup_city, continent)
                
                tested_backups.append({
                    'city_info': backup_city,
                    'data_availability': data_availability
                })
            
            # Rank backup cities by data quality score
            tested_backups.sort(key=lambda x: x['data_availability']['data_quality_score'], reverse=True)
            
            # Recommend replacements
            recommended_replacements[continent] = {
                'failed_cities': failed_cities,
                'backup_options': tested_backups[:len(failed_cities) + 2],  # Extra options
                'recommended_swaps': []
            }
            
            # Create specific replacement recommendations
            for i, failed_city in enumerate(failed_cities):
                if i < len(tested_backups) and tested_backups[i]['data_availability']['data_quality_score'] > 0:
                    recommended_replacements[continent]['recommended_swaps'].append({
                        'remove': failed_city,
                        'replace_with': tested_backups[i]['city_info'],
                        'data_score': tested_backups[i]['data_availability']['data_quality_score'],
                        'data_sources': tested_backups[i]['data_availability']
                    })
        
        return recommended_replacements

    def generate_replacement_report(self, cities_needing_replacement, recommended_replacements):
        """Generate comprehensive replacement report."""
        
        print(f"\nGENERATING REPLACEMENT REPORT")
        print("=" * 35)
        
        self.replacement_results.update({
            'cities_needing_replacement': cities_needing_replacement,
            'recommended_replacements': recommended_replacements,
            'summary': {
                'total_cities_to_replace': sum(len(cities) for cities in cities_needing_replacement.values()),
                'continents_affected': list(cities_needing_replacement.keys()),
                'replacement_feasibility': {}
            }
        })
        
        # Calculate replacement feasibility
        for continent, replacements in recommended_replacements.items():
            failed_count = len(replacements['failed_cities'])
            available_count = len([r for r in replacements['backup_options'] 
                                 if r['data_availability']['data_quality_score'] > 0])
            
            self.replacement_results['summary']['replacement_feasibility'][continent] = {
                'cities_to_replace': failed_count,
                'viable_replacements': available_count,
                'feasibility_ratio': available_count / failed_count if failed_count > 0 else 1.0
            }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"../final_dataset/city_replacement_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.replacement_results, f, indent=2, default=str)
        
        print(f"Replacement analysis saved to: {report_file}")
        
        # Print summary
        total_replacements = self.replacement_results['summary']['total_cities_to_replace']
        print(f"\nREPLACEMENT ANALYSIS SUMMARY:")
        print(f"Cities requiring replacement: {total_replacements}")
        
        for continent, feasibility in self.replacement_results['summary']['replacement_feasibility'].items():
            print(f"{continent}: {feasibility['cities_to_replace']} to replace, {feasibility['viable_replacements']} viable options")
        
        return report_file, self.replacement_results


def main():
    """Main city replacement analysis."""
    
    print("CITY REPLACEMENT STRATEGY ANALYSIS")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 45)
    
    strategy = CityReplacementStrategy()
    
    # Look for existing collection results
    import glob
    collection_files = glob.glob("../final_dataset/comprehensive_real_data_collection_*.json")
    
    if collection_files:
        latest_collection = max(collection_files)
        print(f"Analyzing collection results from: {latest_collection}")
        
        # Analyze results
        cities_needing_replacement = strategy.analyze_collection_results(latest_collection)
        
        if cities_needing_replacement:
            # Find replacements
            recommended_replacements = strategy.find_replacement_cities(cities_needing_replacement)
            
            # Generate report
            report_file, results = strategy.generate_replacement_report(
                cities_needing_replacement, recommended_replacements
            )
            
            return results, report_file
        else:
            print("\nALL CITIES HAVE RELIABLE REAL DATA SOURCES!")
            print("No city replacements needed.")
            return {}, None
    else:
        print("No collection results found. Run comprehensive_real_data_collector.py first.")
        return {}, None


if __name__ == "__main__":
    results, file_path = main()