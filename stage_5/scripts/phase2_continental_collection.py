#!/usr/bin/env python3
"""
Phase 2: Continental Data Collection
====================================

Executes Step 3-7 of the Global 100-City Dataset Collection plan.
Collects actual air quality data for all 100 cities using validated
continental patterns and data sources.

Priority Order (based on validation results):
1. South America (São Paulo Pattern) - 100% ready
2. North America (Toronto Pattern) - validated
3. Europe (Berlin Pattern) - validated with alternatives
4. Asia (Delhi Pattern) - validated
5. Africa (Cairo Pattern) - validated
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from global_100city_data_collector import Global100CityCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/phase2_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class Phase2ContinentalCollector:
    """Phase 2 implementation for continental data collection."""

    def __init__(self):
        """Initialize Phase 2 collector."""
        self.collector = Global100CityCollector()
        self.phase2_results = {
            "phase": "Phase 2: Continental Implementation",
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "continental_results": {},
            "overall_summary": {},
            "status": "in_progress",
        }

        # Priority order based on validation results
        self.collection_order = [
            "south_america",  # Step 4 - Highest success probability (100% ready)
            "north_america",  # Step 5 - Well validated
            "europe",  # Step 3 - Validated with alternatives
            "asia",  # Step 6 - Partial readiness
            "africa",  # Step 7 - Partial readiness
        ]

        log.info("Phase 2 Continental Collector initialized")

    def execute_phase2(self) -> Dict[str, Any]:
        """
        Execute complete Phase 2: Continental Implementation (Steps 3-7).

        Returns:
            Complete Phase 2 results
        """
        log.info("=== STARTING PHASE 2: CONTINENTAL IMPLEMENTATION ===")

        try:
            # Execute continental collection in priority order
            for i, continent in enumerate(self.collection_order, 3):
                step_name = f"Step {i}: {self._get_pattern_name(continent)}"
                log.info(f"Starting {step_name}")

                try:
                    # Collect data for the continent
                    continent_results = self.collector.collect_continental_data(
                        continent
                    )

                    # Store results
                    self.phase2_results["continental_results"][
                        continent
                    ] = continent_results
                    self.phase2_results["steps_completed"].append(step_name)

                    # Save intermediate results
                    self._save_intermediate_results(continent, continent_results)

                    # Rate limiting between continents
                    if i < len(self.collection_order) + 2:  # Not the last continent
                        log.info(f"Waiting 5 minutes before next continent...")
                        time.sleep(300)  # 5 minutes between continents

                except Exception as e:
                    log.error(f"Failed to collect data for {continent}: {str(e)}")
                    self.phase2_results["continental_results"][continent] = {
                        "continent": continent,
                        "status": "failed",
                        "error": str(e),
                    }

            # Generate overall summary
            self._generate_phase2_summary()

            # Save final results
            self._save_final_results()

            # Update progress tracking
            self._update_progress()

            log.info("=== PHASE 2 COMPLETED ===")

        except Exception as e:
            log.error(f"Phase 2 execution failed: {str(e)}")
            self.phase2_results["status"] = "failed"
            self.phase2_results["error"] = str(e)
            raise

        return self.phase2_results

    def _get_pattern_name(self, continent: str) -> str:
        """Get the pattern name for a continent."""
        pattern_names = {
            "europe": "Europe - Berlin Pattern",
            "south_america": "South America - São Paulo Pattern",
            "north_america": "North America - Toronto Pattern",
            "asia": "Asia - Delhi Pattern",
            "africa": "Africa - Cairo Pattern",
        }
        return pattern_names.get(continent, f"{continent.title()} Pattern")

    def _save_intermediate_results(self, continent: str, results: Dict[str, Any]):
        """Save intermediate results for a continent."""
        results_path = Path(
            f"stage_5/logs/step{self.collection_order.index(continent) + 3}_{continent}_results.json"
        )

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"{continent.title()} results saved to: {results_path}")

    def _generate_phase2_summary(self):
        """Generate overall Phase 2 summary."""
        continental_results = self.phase2_results["continental_results"]

        total_continents = len(continental_results)
        successful_continents = sum(
            1 for r in continental_results.values() if r.get("status") == "success"
        )
        partial_continents = sum(
            1
            for r in continental_results.values()
            if r.get("status") == "partial_success"
        )
        failed_continents = (
            total_continents - successful_continents - partial_continents
        )

        total_cities = sum(
            r.get("cities_processed", 0) for r in continental_results.values()
        )
        successful_cities = sum(
            r.get("successful_collections", 0) for r in continental_results.values()
        )
        total_records = sum(
            r.get("total_records", 0) for r in continental_results.values()
        )

        # Calculate success rate
        overall_success_rate = (
            successful_cities / total_cities if total_cities > 0 else 0
        )

        # Determine overall status
        if successful_continents >= 4:
            overall_status = "success"
        elif successful_continents + partial_continents >= 3:
            overall_status = "partial_success"
        else:
            overall_status = "needs_improvement"

        self.phase2_results["overall_summary"] = {
            "total_continents": total_continents,
            "successful_continents": successful_continents,
            "partial_continents": partial_continents,
            "failed_continents": failed_continents,
            "total_cities": total_cities,
            "successful_cities": successful_cities,
            "total_records": total_records,
            "overall_success_rate": round(overall_success_rate, 3),
            "overall_status": overall_status,
            "completion_time": datetime.now().isoformat(),
            "estimated_dataset_size_gb": round(
                total_records * 0.0001, 2
            ),  # Rough estimate
        }

        self.phase2_results["status"] = overall_status

        log.info(f"Phase 2 Summary:")
        log.info(f"  - Cities processed: {successful_cities}/{total_cities} successful")
        log.info(
            f"  - Continents: {successful_continents} success, {partial_continents} partial"
        )
        log.info(f"  - Total records: {total_records:,}")
        log.info(f"  - Overall status: {overall_status}")

    def _save_final_results(self):
        """Save final Phase 2 results."""
        results_path = Path("stage_5/logs/phase2_continental_results.json")

        with open(results_path, "w") as f:
            json.dump(self.phase2_results, f, indent=2)

        log.info(f"Phase 2 final results saved to: {results_path}")

    def _update_progress(self):
        """Update overall project progress."""
        # Load current progress
        progress_path = Path("stage_5/logs/collection_progress.json")
        try:
            with open(progress_path, "r") as f:
                progress = json.load(f)
        except FileNotFoundError:
            progress = {}

        # Update progress
        completed_steps = [
            "Step 1: Initialize Collection Framework",
            "Step 2: Validate Data Sources",
        ] + self.phase2_results["steps_completed"]

        progress.update(
            {
                "phase": "Phase 2: Continental Implementation",
                "current_step": 7,  # Completed through Step 7
                "completed_steps": completed_steps,
                "phase2_summary": self.phase2_results["overall_summary"],
                "last_updated": datetime.now().isoformat(),
            }
        )

        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        log.info("Project progress updated")


def main():
    """Main execution function for Phase 2."""
    log.info("Starting Phase 2: Continental Data Collection")

    try:
        # Initialize Phase 2 collector
        phase2_collector = Phase2ContinentalCollector()

        # Execute Phase 2
        results = phase2_collector.execute_phase2()

        # Print summary
        summary = results["overall_summary"]
        log.info("\n" + "=" * 50)
        log.info("PHASE 2 EXECUTION COMPLETED")
        log.info("=" * 50)
        log.info(f"Status: {results['status'].upper()}")
        log.info(
            f"Cities collected: {summary['successful_cities']}/{summary['total_cities']}"
        )
        log.info(f"Success rate: {summary['overall_success_rate']:.1%}")
        log.info(f"Total records: {summary['total_records']:,}")
        log.info(f"Estimated size: {summary['estimated_dataset_size_gb']} GB")
        log.info("=" * 50)

        return results

    except Exception as e:
        log.error(f"Phase 2 execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
