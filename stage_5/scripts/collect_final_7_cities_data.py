#!/usr/bin/env python3
"""
Collect Real Data for Final 7 Replacement Cities

Collect real data for the 7 strategically replaced cities to achieve 
complete 100% real data coverage.
"""

import json
import time
import warnings
from datetime import datetime

import pandas as pd
import requests

warnings.filterwarnings("ignore")


def collect_final_7_cities_data():
    """Collect real data for the final 7 replacement cities."""
    
    print("COLLECTING REAL DATA FOR FINAL 7 REPLACEMENT CITIES")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)
    
    # The 7 final replacement cities
    final_cities = [
        "Curitiba", "Belo Horizonte", "Porto Alegre", "Campinas", 
        "Guarulhos", "Pretoria", "Bloemfontein"
    ]
    
    # Load current cities table to get coordinates
    cities_df = pd.read_csv("../comprehensive_tables/comprehensive_features_table.csv")
    
    collected_data = {
        'collection_time': datetime.now().isoformat(),
        'objective': 'Collect real data for final 7 cities to achieve 100% coverage',
        'target_cities': final_cities,
        'waqi_data': {},
        'collection_summary': {}
    }
    
    print(f"Target cities: {len(final_cities)}")
    for city in final_cities:
        print(f"  - {city}")
    
    print(f"\nCOLLECTING WAQI DATA:")
    print("=" * 25)
    
    successful_collections = 0
    
    for city_name in final_cities:
        # Get city info from dataset
        city_row = cities_df[cities_df['City'] == city_name]
        if city_row.empty:
            print(f"  WARNING: {city_name} not found in cities table")
            continue
        
        country = city_row['Country'].iloc[0]
        
        try:
            print(f"  Collecting WAQI data for {city_name}, {country}...")
        except UnicodeEncodeError:
            safe_city = city_name.encode("ascii", "replace").decode("ascii")
            safe_country = country.encode("ascii", "replace").decode("ascii")
            print(f"  Collecting WAQI data for {safe_city}, {safe_country}...")
        
        # Comprehensive query strategies for Brazilian and South African cities
        query_strategies = [
            city_name.lower(),
            city_name.lower().replace(' ', '-'),
            city_name.lower().replace(' ', ''),
            city_name.lower().replace('ã', 'a').replace('é', 'e').replace('í', 'i'),
            f"{city_name.lower()}-{country.lower()}",
            f"{city_name.lower().replace(' ', '')}-{country.lower()}",
            city_name.replace(' ', '+'),
            # Specific for Brazilian cities
            city_name.lower().replace('curitiba', 'curitiba').replace('belo-horizonte', 'belo-horizonte'),
            # Specific for South African cities  
            city_name.lower().replace('pretoria', 'pretoria').replace('bloemfontein', 'bloemfontein')
        ]
        
        success = False
        for query in query_strategies:
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
                            
                            collected_data['waqi_data'][city_name] = {
                                'api_status': 'SUCCESS',
                                'collection_time': datetime.now().isoformat(),
                                'query_used': query,
                                'aqi_data': aqi_data,
                                'current_aqi': current_aqi,
                                'pollutants_count': len(pollutants),
                                'city_info': aqi_data.get('city', {}),
                                'attribution': aqi_data.get('attributions', [])
                            }
                            
                            print(f"    SUCCESS: AQI={current_aqi}, Pollutants={len(pollutants)}")
                            successful_collections += 1
                            success = True
                            break
                
                time.sleep(0.8)  # Rate limiting
            except Exception as e:
                continue
        
        if not success:
            collected_data['waqi_data'][city_name] = {
                'api_status': 'NO_DATA',
                'queries_attempted': query_strategies,
                'error': 'All query formats failed'
            }
            print(f"    NO DATA: All query formats failed")
        
        time.sleep(1.2)  # Additional rate limiting between cities
    
    # Generate collection summary
    total_target = len(final_cities)
    success_rate = successful_collections / total_target * 100
    
    collected_data['collection_summary'] = {
        'total_target_cities': total_target,
        'successful_collections': successful_collections,
        'failed_collections': total_target - successful_collections,
        'success_rate': success_rate,
        'cities_with_data': [city for city, data in collected_data['waqi_data'].items() 
                           if data.get('api_status') == 'SUCCESS'],
        'cities_without_data': [city for city, data in collected_data['waqi_data'].items() 
                              if data.get('api_status') != 'SUCCESS']
    }
    
    print(f"\nFINAL 7 CITIES DATA COLLECTION COMPLETE:")
    print("=" * 45)
    print(f"Target cities: {total_target}")
    print(f"Successful collections: {successful_collections}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if successful_collections == total_target:
        print("🎉 SUCCESS: All 7 final cities have real data!")
    else:
        failed_cities = collected_data['collection_summary']['cities_without_data']
        print(f"⚠️  {len(failed_cities)} cities still without data:")
        for city in failed_cities:
            print(f"    - {city}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../final_dataset/final_7_cities_data_collection_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\nFinal 7 cities data collection saved to: {output_file}")
    
    return collected_data, successful_collections, output_file


if __name__ == "__main__":
    results, success_count, file_path = collect_final_7_cities_data()