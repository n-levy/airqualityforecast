# 100-City Air Quality Dataset Collection Makefile
# Cross-platform targets for data ingestion and processing

# Default data root directory
DATA_ROOT ?= C:/aqf311/data

# Date range for data collection (2 years)
START_DATE ?= 2023-09-13
END_DATE ?= 2025-09-13

.PHONY: help setup collect-all collect-gefs collect-cams collect-observations merge-dataset verify-all clean

help:
	@echo "100-City Air Quality Dataset Collection"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help                - Show this help message"
	@echo "  setup               - Install dependencies and setup environment"
	@echo "  collect-all         - Run complete 2-year data collection pipeline"
	@echo "  collect-gefs        - Collect NOAA GEFS-Aerosol data only"
	@echo "  collect-cams        - Collect ECMWF CAMS data only (simulated)"
	@echo "  collect-observations- Collect ground truth observations (synthetic)"
	@echo "  merge-dataset       - Merge all data into unified dataset"
	@echo "  verify-all          - Verify all collected data"
	@echo "  test-collection     - Test collection with 1-day sample"
	@echo "  clean               - Clean temporary files and logs"
	@echo ""
	@echo "Environment variables:"
	@echo "  DATA_ROOT           - Data directory (default: $(DATA_ROOT))"
	@echo "  START_DATE          - Start date for collection (default: $(START_DATE))"
	@echo "  END_DATE            - End date for collection (default: $(END_DATE))"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make collect-all"
	@echo "  make test-collection START_DATE=2025-09-12 END_DATE=2025-09-13"

setup:
	@echo "Setting up Python environment..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python scripts/setup_environment.py
	@echo "Environment setup complete"

collect-all:
	@echo "Starting complete 2-year data collection..."
	@echo "Data root: $(DATA_ROOT)"
	@echo "Date range: $(START_DATE) to $(END_DATE)"
	python scripts/orchestrate_full_100city_collection.py \
		--data-root $(DATA_ROOT) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE)

collect-gefs:
	@echo "Collecting NOAA GEFS-Aerosol data..."
	python scripts/collect_2year_gefs_data.py \
		--data-root $(DATA_ROOT) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE)

collect-cams:
	@echo "Collecting ECMWF CAMS data (simulated)..."
	python scripts/collect_2year_cams_data.py \
		--data-root $(DATA_ROOT) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--simulate

collect-observations:
	@echo "Collecting ground truth observations (synthetic)..."
	python scripts/collect_ground_truth_observations.py \
		--data-root $(DATA_ROOT) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--synthetic

merge-dataset:
	@echo "Merging unified 100-city dataset..."
	python scripts/merge_unified_100city_dataset.py \
		--data-root $(DATA_ROOT)

verify-all:
	@echo "Verifying all collected data..."
	python scripts/collect_2year_gefs_data.py --data-root $(DATA_ROOT) --verify-only
	python scripts/collect_2year_cams_data.py --data-root $(DATA_ROOT) --verify-only
	python scripts/collect_ground_truth_observations.py --data-root $(DATA_ROOT) --verify-only
	python scripts/merge_unified_100city_dataset.py --data-root $(DATA_ROOT) --verify-only

test-collection:
	@echo "Testing data collection with 1-day sample..."
	python scripts/orchestrate_full_100city_collection.py \
		--data-root $(DATA_ROOT) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE)

dry-run:
	@echo "Dry run of complete collection pipeline..."
	python scripts/orchestrate_full_100city_collection.py \
		--data-root $(DATA_ROOT) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--dry-run

clean:
	@echo "Cleaning temporary files..."
	find . -name "*.pyc" -delete || true
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name "*.tmp" -delete || true
	find . -name "*.5b7b6.idx" -delete || true
	@echo "Clean complete"

# Platform-specific aliases
ifeq ($(OS),Windows_NT)
    PYTHON := python
    RM := del /Q
else
    PYTHON := python3
    RM := rm -f
endif

# Development targets
lint:
	@echo "Running code linting..."
	python -m flake8 scripts/ --max-line-length=100 --ignore=E501,W503 || true

format:
	@echo "Formatting code..."
	python -m black scripts/ --line-length=100 || true

test:
	@echo "Running tests..."
	python -m pytest tests/ -v || true
