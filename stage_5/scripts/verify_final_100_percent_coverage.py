#!/usr/bin/env python3
"""
Verify Final 100% Real Data Coverage

Combine original data collection with replacement cities data to verify
if we have achieved 100% real data coverage across all 100 cities.
"""

import json
import pandas as pd
from datetime import datetime


def verify_final_coverage():
    """Verify final real data coverage after city replacements."""
    
    print("FINAL 100% REAL DATA COVERAGE VERIFICATION")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Load current cities table (with replacements)
    cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")
    
    # Load original data collection
    with open("../final_dataset/complete_real_data_collection_20250911_192217.json", 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load replacement cities data collection
    with open("../final_dataset/replacement_cities_data_collection_20250911_194712.json", 'r', encoding='utf-8') as f:
        replacement_data = json.load(f)
    
    # Combine all cities with real data
    cities_with_real_data = set()
    
    print("ORIGINAL DATA SOURCES:")
    print("=" * 25)
    
    # Add cities from original NOAA data
    original_noaa_count = 0
    for city, info in original_data['noaa_data'].items():
        if info.get('api_status') == 'SUCCESS':
            cities_with_real_data.add(city)
            original_noaa_count += 1
    print(f"Original NOAA cities: {original_noaa_count}")
    
    # Add cities from original WAQI data
    original_waqi_count = 0
    for city, info in original_data['waqi_data'].items():
        if info.get('api_status') == 'SUCCESS':
            cities_with_real_data.add(city)
            original_waqi_count += 1
    print(f"Original WAQI cities: {original_waqi_count}")
    
    print(f"Total unique cities from original collection: {len(cities_with_real_data)}")
    
    print(f"\nREPLACEMENT CITIES DATA SOURCES:")
    print("=" * 35)
    
    # Add cities from replacement NOAA data
    replacement_noaa_count = 0
    for city, info in replacement_data['noaa_data'].items():
        if info.get('api_status') == 'SUCCESS':
            cities_with_real_data.add(city)
            replacement_noaa_count += 1
    print(f"Replacement NOAA cities: {replacement_noaa_count}")
    
    # Add cities from replacement WAQI data
    replacement_waqi_count = 0
    for city, info in replacement_data['waqi_data'].items():
        if info.get('api_status') == 'SUCCESS':
            cities_with_real_data.add(city)
            replacement_waqi_count += 1
    print(f"Replacement WAQI cities: {replacement_waqi_count}")
    
    print(f"Total unique cities from replacement collection: {replacement_noaa_count + replacement_waqi_count}")
    
    # Find cities still without real data
    all_cities = set(cities_df['City'].tolist())
    cities_without_real = all_cities - cities_with_real_data
    
    print(f"\nFINAL COVERAGE RESULTS:")
    print("=" * 25)
    print(f"Total cities in dataset: {len(all_cities)}")
    print(f"Cities with real data: {len(cities_with_real_data)}")
    print(f"Cities without real data: {len(cities_without_real)}")
    print(f"Real data coverage: {len(cities_with_real_data)/len(all_cities)*100:.1f}%")
    
    print(f"\nDETAILED DATA SOURCE BREAKDOWN:")
    print("=" * 35)
    print(f"Total NOAA cities: {original_noaa_count + replacement_noaa_count}")
    print(f"Total WAQI cities: {original_waqi_count + replacement_waqi_count}")
    print(f"Total unique cities with real data: {len(cities_with_real_data)}")
    
    # Check if 100% achieved
    if len(cities_with_real_data) == 100:
        print(f"\nSUCCESS: 100% REAL DATA COVERAGE ACHIEVED!")
        print("=" * 45)
        print("All 100 cities have verified real data sources!")
        print("- 0% synthetic data required")
        print("- Success criterion met: 100 cities, 20 per continent, 100% real data")
        is_100_percent = True
    else:
        print(f"\nPARTIAL SUCCESS: {len(cities_with_real_data)}/100 cities have real data")
        print("=" * 55)
        print(f"Real data coverage: {len(cities_with_real_data)/100*100:.1f}%")
        print(f"Synthetic data still needed for: {len(cities_without_real)} cities")
        is_100_percent = False
        
        if cities_without_real:
            print(f"\nCITIES STILL REQUIRING SYNTHETIC DATA:")
            print("-" * 40)
            for i, city in enumerate(sorted(cities_without_real), 1):
                try:
                    print(f"  {i:2d}. {city}")
                except UnicodeEncodeError:
                    safe_city = city.encode('ascii', 'replace').decode('ascii')
                    print(f"  {i:2d}. {safe_city}")
    
    # Generate continental breakdown
    print(f"\nCONTINENTAL BREAKDOWN:")
    print("=" * 22)
    continent_stats = {}
    for continent in cities_df['Continent'].unique():
        continent_cities = cities_df[cities_df['Continent'] == continent]['City'].tolist()
        continent_real = len([city for city in continent_cities if city in cities_with_real_data])
        continent_stats[continent] = {
            'total': len(continent_cities),
            'real_data': continent_real,
            'percentage': continent_real / len(continent_cities) * 100
        }
        print(f"{continent}: {continent_real}/{len(continent_cities)} ({continent_real/len(continent_cities)*100:.1f}%)")
    
    # Save verification report
    verification_report = {
        'verification_time': datetime.now().isoformat(),
        'objective': 'Verify 100% real data coverage after city replacements',
        'total_cities': len(all_cities),
        'cities_with_real_data': len(cities_with_real_data),
        'cities_without_real_data': len(cities_without_real),
        'real_data_percentage': len(cities_with_real_data) / len(all_cities) * 100,
        'is_100_percent_real': is_100_percent,
        'data_sources': {
            'original_noaa': original_noaa_count,
            'original_waqi': original_waqi_count,
            'replacement_noaa': replacement_noaa_count,
            'replacement_waqi': replacement_waqi_count,
            'total_noaa': original_noaa_count + replacement_noaa_count,
            'total_waqi': original_waqi_count + replacement_waqi_count
        },
        'continental_breakdown': continent_stats,
        'cities_with_real_data_list': sorted(list(cities_with_real_data)),
        'cities_without_real_data_list': sorted(list(cities_without_real)),
        'success_criterion_met': is_100_percent and len(all_cities) == 100
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../final_dataset/final_100_percent_verification_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(verification_report, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\nFinal verification report saved to: {output_file}")
    
    return verification_report, is_100_percent


if __name__ == "__main__":
    results, achieved_100_percent = verify_final_coverage()