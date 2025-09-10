#!/usr/bin/env python3
"""
Storage Requirements Optimizer
============================

Recalculate storage requirements using optimization strategies
for laptop deployment scenarios.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def calculate_optimized_storage(
    total_records: int, optimization_level: str = "laptop"
) -> Dict:
    """Calculate optimized storage requirements for different deployment scenarios."""

    # Original storage per record (from temporal scaling scripts)
    original_per_record = 0.92  # MB per record

    if optimization_level == "standard":
        # Current approach - no optimization
        return {
            "raw_data_mb": total_records * 0.58,
            "processed_data_mb": total_records * 0.34,
            "metadata_mb": 70,
            "total_mb": total_records * 0.92 + 70,
            "optimization": "None",
            "reduction_percentage": 0,
        }

    elif optimization_level == "compressed":
        # Strategy 1: Data compression only
        return {
            "raw_data_mb": total_records * 0.12,  # Binary + compression
            "processed_data_mb": total_records * 0.08,  # Parquet format
            "metadata_mb": 20,  # Compressed metadata
            "total_mb": total_records * 0.20 + 20,
            "optimization": "Compression (binary formats, gzip)",
            "reduction_percentage": 78,
        }

    elif optimization_level == "selective":
        # Strategy 2: Essential data only
        essential_bytes_per_record = 23  # bytes
        essential_mb_per_record = essential_bytes_per_record / (1024 * 1024)

        return {
            "essential_data_mb": total_records * essential_mb_per_record,
            "quality_indicators_mb": total_records * 0.002,
            "metadata_mb": 10,
            "total_mb": total_records * 0.025,
            "optimization": "Essential data only (PM2.5, PM10, NO2, O3, AQI)",
            "reduction_percentage": 97,
        }

    elif optimization_level == "laptop":
        # Strategy 3: Hybrid approach for laptop deployment
        # Compression + selective data + temporal sampling
        base_size = total_records * 0.05  # Compressed essential data
        sampling_reduction = 0.3  # Keep 30% of temporal data
        final_size = base_size * sampling_reduction

        return {
            "optimized_data_mb": final_size,
            "metadata_mb": 10,
            "cache_mb": 20,
            "total_mb": final_size + 30,
            "optimization": "Hybrid: Compression + Selective + Temporal Sampling",
            "reduction_percentage": 97,
        }

    elif optimization_level == "ultra_minimal":
        # Strategy 4: Ultra-minimal for constrained environments
        # Daily averages only, essential pollutants
        daily_records = total_records / 24  # Convert hourly to daily

        return {
            "daily_averages_mb": daily_records * 0.001,  # 1KB per daily record
            "metadata_mb": 5,
            "total_mb": daily_records * 0.001 + 5,
            "optimization": "Daily averages only, ultra-compressed",
            "reduction_percentage": 99,
        }


def calculate_system_storage_requirements() -> Dict:
    """Calculate storage requirements for different system deployment scenarios."""

    # Base data from Week 2 temporal scaling results
    cities_data = {
        "berlin": {"records": 50105, "original_gb": 40.1},
        "toronto": {"records": 50105, "original_gb": 40.1},
        "delhi": {"records": 42128, "original_gb": 40.1},
        "cairo": {"records": 43377, "original_gb": 36.9},
        "sao_paulo": {"records": 41963, "original_gb": 38.7},
    }

    # Calculate average records per city for 100-city projection
    avg_records_per_city = sum(city["records"] for city in cities_data.values()) / len(
        cities_data
    )
    total_records_100_cities = avg_records_per_city * 100

    scenarios = {}
    optimization_levels = [
        "standard",
        "compressed",
        "selective",
        "laptop",
        "ultra_minimal",
    ]

    for level in optimization_levels:
        city_storage = calculate_optimized_storage(avg_records_per_city, level)
        system_storage_gb = (city_storage["total_mb"] * 100) / 1024  # Convert to GB

        scenarios[level] = {
            "per_city_mb": city_storage["total_mb"],
            "per_city_gb": city_storage["total_mb"] / 1024,
            "system_total_gb": system_storage_gb,
            "system_total_tb": system_storage_gb / 1024,
            "optimization": city_storage["optimization"],
            "reduction_vs_original": city_storage.get("reduction_percentage", 0),
        }

    return {
        "calculated_at": datetime.now().isoformat(),
        "base_metrics": {
            "cities_analyzed": 5,
            "avg_records_per_city": int(avg_records_per_city),
            "total_cities_projected": 100,
            "total_records_projected": int(total_records_100_cities),
        },
        "storage_scenarios": scenarios,
        "recommendations": {
            "laptop_deployment": "Use 'laptop' optimization level",
            "constrained_storage": "Use 'ultra_minimal' optimization level",
            "cloud_deployment": "Use 'compressed' optimization level",
            "research_deployment": "Use 'selective' optimization level",
        },
    }


def main():
    """Generate storage optimization analysis."""

    print("Calculating optimized storage requirements...")

    # Calculate storage scenarios
    analysis = calculate_system_storage_requirements()

    # Save analysis
    output_dir = Path("data/analysis/storage_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "storage_optimization_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("STORAGE OPTIMIZATION ANALYSIS")
    print("=" * 80)

    print(f"\nBASE METRICS:")
    base = analysis["base_metrics"]
    print(f"• Cities analyzed: {base['cities_analyzed']}")
    print(f"• Average records per city: {base['avg_records_per_city']:,}")
    print(
        f"• Projected 100-city system: {base['total_records_projected']:,} total records"
    )

    print(f"\nSTORAGE SCENARIOS (100 cities):")
    for scenario, data in analysis["storage_scenarios"].items():
        print(f"\n{scenario.upper()}:")
        print(f"  • Per city: {data['per_city_gb']:.1f} GB")
        print(
            f"  • System total: {data['system_total_gb']:.1f} GB ({data['system_total_tb']:.2f} TB)"
        )
        print(f"  • Optimization: {data['optimization']}")
        print(f"  • Reduction: {data['reduction_vs_original']}%")

    print(f"\nRECOMMENDATIONS:")
    for use_case, recommendation in analysis["recommendations"].items():
        print(f"• {use_case.replace('_', ' ').title()}: {recommendation}")

    print(f"\n💡 LAPTOP DEPLOYMENT HIGHLIGHT:")
    laptop_scenario = analysis["storage_scenarios"]["laptop"]
    print(f"• Total system storage: {laptop_scenario['system_total_gb']:.1f} GB")
    print(f"• Storage per city: {laptop_scenario['per_city_gb']:.1f} GB")
    print(f"• Reduction from original: {laptop_scenario['reduction_vs_original']}%")
    print(
        f"• Feasible for laptop: {'✅ YES' if laptop_scenario['system_total_gb'] < 200 else '❌ NO'}"
    )

    print(f"\n🔥 ULTRA-MINIMAL HIGHLIGHT:")
    minimal_scenario = analysis["storage_scenarios"]["ultra_minimal"]
    print(f"• Total system storage: {minimal_scenario['system_total_gb']:.1f} GB")
    print(f"• Storage per city: {minimal_scenario['per_city_mb']:.0f} MB")
    print(f"• Reduction from original: {minimal_scenario['reduction_vs_original']}%")

    print(f"\nAnalysis saved to: {output_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
