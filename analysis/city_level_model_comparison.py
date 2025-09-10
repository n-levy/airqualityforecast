#!/usr/bin/env python3
"""
City-Level Model Performance Comparison Analysis
===============================================

Compare Gradient Boosting Enhanced (best model) vs benchmarks across all 100 cities.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_city_level_performance():
    """Analyze city-level performance comparison between best model and benchmarks."""
    
    # Load the evaluation results
    results_path = Path("data/analysis/stage4_forecasting_evaluation/stage4_quick_evaluation_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract city-level data
    city_comparisons = []
    
    for continent, cont_data in results["continental_results"].items():
        for city_result in cont_data["city_results"]:
            city_name = city_result["city"]
            
            # Get performance for each model
            gb_perf = city_result["model_performance"]["gradient_boosting_enhanced"]
            simple_perf = city_result["model_performance"]["simple_average_ensemble"]
            weighted_perf = city_result["model_performance"]["quality_weighted_ensemble"]
            
            # Calculate improvements
            gb_vs_simple_improvement = (gb_perf["r2_score"] - simple_perf["r2_score"]) / simple_perf["r2_score"] * 100
            gb_vs_weighted_improvement = (gb_perf["r2_score"] - weighted_perf["r2_score"]) / weighted_perf["r2_score"] * 100
            
            city_comparisons.append({
                "city": city_name,
                "continent": continent,
                "gradient_boosting_r2": gb_perf["r2_score"],
                "simple_average_r2": simple_perf["r2_score"],
                "quality_weighted_r2": weighted_perf["r2_score"],
                "gb_vs_simple_improvement_pct": gb_vs_simple_improvement,
                "gb_vs_weighted_improvement_pct": gb_vs_weighted_improvement,
                "gradient_boosting_mae": gb_perf["mae"],
                "simple_average_mae": simple_perf["mae"],
                "quality_weighted_mae": weighted_perf["mae"],
                "production_ready_gb": gb_perf["r2_score"] > 0.80,
                "production_ready_simple": simple_perf["r2_score"] > 0.80,
                "production_ready_weighted": weighted_perf["r2_score"] > 0.80,
            })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(city_comparisons)
    
    # Global statistics
    print("="*80)
    print("CITY-LEVEL MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Total Cities Analyzed: {len(df)}")
    print()
    
    # Overall performance comparison
    print("GLOBAL AVERAGE PERFORMANCE:")
    print(f"• Gradient Boosting Enhanced R²: {df['gradient_boosting_r2'].mean():.3f}")
    print(f"• Simple Average Ensemble R²:   {df['simple_average_r2'].mean():.3f}")
    print(f"• Quality-Weighted Ensemble R²: {df['quality_weighted_r2'].mean():.3f}")
    print()
    
    # Improvement statistics
    print("IMPROVEMENT STATISTICS:")
    print(f"• GB vs Simple Average - Mean Improvement: {df['gb_vs_simple_improvement_pct'].mean():.1f}%")
    print(f"• GB vs Quality-Weighted - Mean Improvement: {df['gb_vs_weighted_improvement_pct'].mean():.1f}%")
    print(f"• Cities where GB > Simple Average: {(df['gb_vs_simple_improvement_pct'] > 0).sum()}/100")
    print(f"• Cities where GB > Quality-Weighted: {(df['gb_vs_weighted_improvement_pct'] > 0).sum()}/100")
    print()
    
    # Production readiness comparison
    print("PRODUCTION READINESS (R² > 0.80):")
    print(f"• Gradient Boosting Enhanced: {df['production_ready_gb'].sum()}/100 cities")
    print(f"• Simple Average Ensemble:    {df['production_ready_simple'].sum()}/100 cities")
    print(f"• Quality-Weighted Ensemble:  {df['production_ready_weighted'].sum()}/100 cities")
    print()
    
    # Continental breakdown
    print("CONTINENTAL PERFORMANCE BREAKDOWN:")
    for continent in ['europe', 'north_america', 'south_america', 'africa', 'asia']:
        cont_data = df[df['continent'] == continent]
        print(f"\n{continent.replace('_', ' ').title()}:")
        print(f"  • GB Enhanced R²:     {cont_data['gradient_boosting_r2'].mean():.3f}")
        print(f"  • Simple Average R²:  {cont_data['simple_average_r2'].mean():.3f}")  
        print(f"  • Quality-Weighted R²: {cont_data['quality_weighted_r2'].mean():.3f}")
        print(f"  • GB Production Ready: {cont_data['production_ready_gb'].sum()}/{len(cont_data)} cities")
        print(f"  • Mean GB Improvement: {cont_data['gb_vs_weighted_improvement_pct'].mean():.1f}% vs Quality-Weighted")
    
    # Best and worst performing cities
    print("\nTOP 10 BEST PERFORMING CITIES (Gradient Boosting):")
    top_cities = df.nlargest(10, 'gradient_boosting_r2')[['city', 'continent', 'gradient_boosting_r2']]
    for idx, city in top_cities.iterrows():
        print(f"  {city['city']:<25} ({city['continent']:<15}): R² = {city['gradient_boosting_r2']:.3f}")
    
    print("\nTOP 10 WORST PERFORMING CITIES (Gradient Boosting):")
    worst_cities = df.nsmallest(10, 'gradient_boosting_r2')[['city', 'continent', 'gradient_boosting_r2']]
    for idx, city in worst_cities.iterrows():
        print(f"  {city['city']:<25} ({city['continent']:<15}): R² = {city['gradient_boosting_r2']:.3f}")
    
    # Cities where benchmarks outperform GB (rare cases)
    print("\nCITIES WHERE BENCHMARKS OUTPERFORM GRADIENT BOOSTING:")
    underperform_simple = df[df['gb_vs_simple_improvement_pct'] < 0]
    underperform_weighted = df[df['gb_vs_weighted_improvement_pct'] < 0]
    
    if len(underperform_simple) > 0:
        print("Cities where Simple Average > Gradient Boosting:")
        for idx, city in underperform_simple.iterrows():
            print(f"  {city['city']}: GB={city['gradient_boosting_r2']:.3f}, Simple={city['simple_average_r2']:.3f}")
    else:
        print("• Gradient Boosting outperforms Simple Average in ALL 100 cities")
    
    if len(underperform_weighted) > 0:
        print("Cities where Quality-Weighted > Gradient Boosting:")
        for idx, city in underperform_weighted.iterrows():
            print(f"  {city['city']}: GB={city['gradient_boosting_r2']:.3f}, Weighted={city['quality_weighted_r2']:.3f}")
    else:
        print("• Gradient Boosting outperforms Quality-Weighted in ALL 100 cities")
    
    # Save detailed comparison
    output_path = Path("data/analysis/stage4_forecasting_evaluation/city_level_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed city-level comparison saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    analyze_city_level_performance()