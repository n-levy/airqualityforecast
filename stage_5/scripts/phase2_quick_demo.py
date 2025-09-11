#!/usr/bin/env python3
"""
Phase 2: Quick Demo - Continental Data Collection
===============================================

Quick demonstration of Phase 2 continental data collection.
Simulates the collection process with minimal delays for demo purposes.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from global_100city_data_collector import Global100CityCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/phase2_quick_demo.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class Phase2QuickDemo:
    """Quick demo of Phase 2 continental collection."""

    def __init__(self):
        """Initialize Phase 2 quick demo."""
        self.collector = Global100CityCollector()
        self.phase2_results = {
            "phase": "Phase 2: Continental Implementation (Quick Demo)",
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "continental_results": {},
            "overall_summary": {},
            "status": "in_progress",
        }

        # Demo with 2 continents to show the process
        self.demo_continents = ["south_america", "north_america"]

        log.info("Phase 2 Quick Demo initialized")

    def execute_quick_demo(self) -> Dict[str, Any]:
        """Execute quick demo of Phase 2."""
        log.info("=== STARTING PHASE 2 QUICK DEMO ===")

        try:
            for i, continent in enumerate(self.demo_continents):
                step_name = f"Step {i+3}: {self._get_pattern_name(continent)}"
                log.info(f"Starting {step_name}")

                # Quick collection (first 3 cities only)
                continent_results = self._quick_continental_collection(continent)

                self.phase2_results["continental_results"][
                    continent
                ] = continent_results
                self.phase2_results["steps_completed"].append(step_name)

                log.info(
                    f"Completed {step_name}: {continent_results['successful_collections']}/3 cities successful"
                )

                # Short delay between continents
                time.sleep(2)

            self._generate_demo_summary()
            self._save_demo_results()

            log.info("=== PHASE 2 QUICK DEMO COMPLETED ===")

        except Exception as e:
            log.error(f"Demo execution failed: {str(e)}")
            self.phase2_results["status"] = "failed"
            self.phase2_results["error"] = str(e)
            raise

        return self.phase2_results

    def _quick_continental_collection(self, continent: str) -> Dict[str, Any]:
        """Quick collection for demo (first 3 cities only)."""
        log.info(f"=== DEMO COLLECTION: {continent.upper()} ===")

        results = {
            "continent": continent,
            "pattern_name": self.collector.continental_patterns[continent][
                "pattern_name"
            ],
            "timestamp": datetime.now().isoformat(),
            "cities_processed": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "total_records": 0,
            "city_results": {},
            "status": "demo",
        }

        # Get first 3 cities for demo
        cities = self.collector.cities_config[continent][:3]
        data_sources = self.collector.data_sources[continent]

        for city in cities:
            city_name = city["name"]
            log.info(f"Demo collecting: {city_name}, {city['country']}")

            # Quick simulated collection
            city_data = self._quick_city_collection(city, data_sources, continent)
            results["city_results"][city_name] = city_data

            if city_data["status"] == "success":
                results["successful_collections"] += 1
                results["total_records"] += city_data.get("record_count", 0)
            else:
                results["failed_collections"] += 1

            results["cities_processed"] += 1
            time.sleep(0.5)  # Minimal delay for demo

        # Set overall status
        success_rate = results["successful_collections"] / len(cities)
        results["status"] = "success" if success_rate >= 0.7 else "partial_success"

        return results

    def _quick_city_collection(
        self, city: Dict, data_sources: Dict, continent: str
    ) -> Dict[str, Any]:
        """Quick city data collection for demo."""
        city_result = {
            "city": city["name"],
            "country": city["country"],
            "coordinates": {"lat": city["lat"], "lon": city["lon"]},
            "aqi_standard": city["aqi_standard"],
            "status": "success",
            "data_sources": {},
            "record_count": 0,
            "quality_score": 0.0,
        }

        successful_sources = 0
        total_records = 0

        # Quick collection from each source
        for source_type, source_info in data_sources.items():
            # Simulate collection success (90% success rate)
            import random

            if random.random() < 0.9:
                estimated_records = int(1825 * 0.8)  # 80% coverage
                city_result["data_sources"][source_type] = {
                    "source_name": source_info["name"],
                    "status": "success",
                    "record_count": estimated_records,
                    "data_coverage": 0.8,
                }
                successful_sources += 1
                total_records += estimated_records
            else:
                city_result["data_sources"][source_type] = {
                    "source_name": source_info["name"],
                    "status": "failed",
                    "record_count": 0,
                }

        city_result["record_count"] = total_records
        city_result["quality_score"] = successful_sources / len(data_sources)

        # Status based on successful sources
        if successful_sources >= 2:
            city_result["status"] = "success"
        elif successful_sources >= 1:
            city_result["status"] = "partial_success"
        else:
            city_result["status"] = "failed"

        return city_result

    def _get_pattern_name(self, continent: str) -> str:
        """Get pattern name for continent."""
        pattern_names = {
            "south_america": "South America - São Paulo Pattern",
            "north_america": "North America - Toronto Pattern",
        }
        return pattern_names.get(continent, f"{continent.title()} Pattern")

    def _generate_demo_summary(self):
        """Generate demo summary."""
        continental_results = self.phase2_results["continental_results"]

        total_cities = sum(
            r.get("cities_processed", 0) for r in continental_results.values()
        )
        successful_cities = sum(
            r.get("successful_collections", 0) for r in continental_results.values()
        )
        total_records = sum(
            r.get("total_records", 0) for r in continental_results.values()
        )

        success_rate = successful_cities / total_cities if total_cities > 0 else 0

        self.phase2_results["overall_summary"] = {
            "demo_mode": True,
            "continents_tested": len(continental_results),
            "total_cities": total_cities,
            "successful_cities": successful_cities,
            "total_records": total_records,
            "success_rate": round(success_rate, 3),
            "estimated_full_dataset_records": total_records * (100 / total_cities),
            "estimated_full_dataset_size_gb": round(
                (total_records * (100 / total_cities)) * 0.0001, 2
            ),
            "completion_time": datetime.now().isoformat(),
        }

        self.phase2_results["status"] = "demo_success"

    def _save_demo_results(self):
        """Save demo results."""
        results_path = Path("stage_5/logs/phase2_quick_demo_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(self.phase2_results, f, indent=2)

        log.info(f"Demo results saved to: {results_path}")


def main():
    """Main execution for quick demo."""
    log.info("Starting Phase 2 Quick Demo")

    try:
        demo = Phase2QuickDemo()
        results = demo.execute_quick_demo()

        # Print summary
        summary = results["overall_summary"]
        log.info("\n" + "=" * 50)
        log.info("PHASE 2 QUICK DEMO COMPLETED")
        log.info("=" * 50)
        log.info(f"Status: {results['status'].upper()}")
        log.info(
            f"Cities tested: {summary['successful_cities']}/{summary['total_cities']}"
        )
        log.info(f"Success rate: {summary['success_rate']:.1%}")
        log.info(f"Records collected: {summary['total_records']:,}")
        log.info(
            f"Projected full dataset: {summary['estimated_full_dataset_records']:,.0f} records"
        )
        log.info(f"Projected size: {summary['estimated_full_dataset_size_gb']} GB")
        log.info("=" * 50)

        return results

    except Exception as e:
        log.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
