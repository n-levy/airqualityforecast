#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Real Data Collection Results

Analyze the real data collection results and identify cities for replacement.
"""

import json
import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")


def analyze_collection_results():
    """Analyze the completed real data collection results."""
    
    print("ANALYZING REAL DATA COLLECTION RESULTS")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 50)
    
    # Load the collection results
    try:
        with open("../final_dataset/complete_real_data_collection_20250911_192217.json", 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
    except FileNotFoundError:
        print("Collection results file not found!")
        return None
    
    # Load cities data
    cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")
    
    # Get statistics from collection
    stats = collection_data['collection_metadata']['statistics']
    
    print(f"COLLECTION SUMMARY:")
    print(f"  Total cities: {stats['total_cities']}")
    print(f"  NOAA successful: {stats['noaa_successful']} (US cities)")
    print(f"  WAQI successful: {stats['waqi_successful']} (global cities)")
    print(f"  Cities with real data: {stats['cities_with_any_real_data']} ({stats['real_data_percentage']:.1f}%)")
    print(f"  Cities needing replacement: {stats['synthetic_data_needed']}")
    
    # Identify cities with real data
    cities_with_real_data = set()
    
    # Add NOAA cities
    for city, data in collection_data['noaa_data'].items():
        if data.get('api_status') == 'SUCCESS':
            cities_with_real_data.add(city)
    
    # Add WAQI cities
    for city, data in collection_data['waqi_data'].items():
        if data.get('api_status') == 'SUCCESS':
            cities_with_real_data.add(city)
    
    # Identify cities without real data
    all_cities = set(cities_df['City'].tolist())
    cities_without_real_data = all_cities - cities_with_real_data
    
    print(f"\nCITIES WITH REAL DATA ({len(cities_with_real_data)}):")
    for city in sorted(cities_with_real_data):
        city_row = cities_df[cities_df['City'] == city].iloc[0]
        country = city_row['Country']
        continent = city_row['Continent']
        # Handle Unicode safely
        try:
            print(f"  ✓ {city}, {country} ({continent})")
        except UnicodeEncodeError:
            safe_city = city.encode('ascii', 'replace').decode('ascii')
            print(f"  ✓ {safe_city}, {country} ({continent})")
    
    print(f"\nCITIES WITHOUT REAL DATA ({len(cities_without_real_data)}):")
    cities_by_continent = {}
    
    for city in cities_without_real_data:
        city_row = cities_df[cities_df['City'] == city].iloc[0]
        country = city_row['Country']
        continent = city_row['Continent']
        
        if continent not in cities_by_continent:
            cities_by_continent[continent] = []
        
        cities_by_continent[continent].append({'city': city, 'country': country})
    
    for continent, cities in cities_by_continent.items():
        print(f"\n  {continent} ({len(cities)} cities):")
        for city_info in cities:
            try:
                print(f"    - {city_info['city']}, {city_info['country']}")
            except UnicodeEncodeError:
                safe_city = city_info['city'].encode('ascii', 'replace').decode('ascii')
                print(f"    - {safe_city}, {city_info['country']}")
    
    # Create summary report
    analysis_report = {
        'analysis_time': datetime.now().isoformat(),
        'collection_file': 'complete_real_data_collection_20250911_192217.json',
        'real_data_summary': {
            'total_cities': len(all_cities),
            'cities_with_real_data': len(cities_with_real_data),
            'cities_without_real_data': len(cities_without_real_data),
            'real_data_percentage': (len(cities_with_real_data) / len(all_cities)) * 100,
            'noaa_cities': stats['noaa_successful'],
            'waqi_cities': stats['waqi_successful']
        },
        'cities_with_real_data': sorted(list(cities_with_real_data)),
        'cities_without_real_data': sorted(list(cities_without_real_data)),
        'cities_by_continent': cities_by_continent,
        'next_steps': [
            'Replace 22 cities without real data with backup cities from same continents',
            'Backup cities should have verified API data availability',
            'Maintain continental balance (Asia: 20, Africa: 20, Europe: 20, N.America: 20, S.America: 20)',
            'Prioritize cities with worst air quality that have reliable data sources'
        ]
    }
    
    # Save analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../final_dataset/real_data_analysis_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\nAnalysis report saved to: {output_file}")
    
    print(f"\nNEXT STEPS:")
    print(f"1. Keep all 78 cities with real data (78% coverage achieved!)")
    print(f"2. Replace 22 cities without real data with backup cities")
    print(f"3. Test backup cities for API data availability")
    print(f"4. Maintain balanced continental representation")
    print(f"5. Update comprehensive tables with final city list")
    
    return analysis_report, output_file


if __name__ == "__main__":
    results, file_path = analyze_collection_results()