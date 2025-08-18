# Makefile for Trading System with DataSanity Enforcement

.PHONY: help install test sanity falsify bench-sanity clean coverage lint lint-changed format promote wf smoke type datasanity golden bless_golden

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run all tests"
	@echo "  sanity       - Run DataSanity tests only"
	@echo "  falsify      - Run DataSanity falsification battery"
	@echo "  bench-sanity - Run DataSanity performance benchmarks"
	@echo "  coverage     - Run tests with coverage report"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up generated files"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov hypothesis pytest-randomly

# Run all tests
test:
	pytest -v

# Run DataSanity tests only
sanity:
	pytest tests/test_data_integrity.py tests/test_data_sanity_enforcement.py -v

# Run DataSanity falsification battery
falsify:
	python scripts/falsify_data_sanity.py

# Run DataSanity performance benchmarks
bench-sanity:
	pytest tests/test_data_sanity_enforcement.py::TestDataSanityEnforcement::test_performance_safety -v -s

# Run tests with coverage
coverage:
	pytest --cov=core --cov=features --cov=strategies --cov-report=html --cov-report=term-missing

# Run linting checks
lint:
	ruff check --output-format=github tools/ core/metrics/ core/io_* tests/unit/
	black --check .
	isort --check-only .

CHANGED_PY := $(shell git fetch origin main >/dev/null 2>&1; git diff --name-only --diff-filter=ACMRT origin/main...HEAD | grep -E '\.py$$' || true)

lint-changed:
	@if [ -z "$(CHANGED_PY)" ]; then echo "No changed Python files."; else ruff check --output-format=github $(CHANGED_PY); fi

# Format code
format:
	black .
	isort .

# Run walkforward (placeholder)
wf:
	pytest -q tests/walkforward -k spy --maxfail=1 || true

# Promotion gate
promote:
	python tools/promotion_gate.py --fail-on-quarantine --report

# Smoke preset (fast CI-friendly walkforward)
smoke:
	python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO

datasanity:
	pytest -q tests/walkforward/test_datasanity_*.py tests/backtest/test_datasanity_*.py

golden: tests/golden/SPY.parquet
	pytest -q tests/backtest/test_golden_regression.py

tests/golden/SPY.parquet:
	python tools/gen_golden_spy.py

bless_golden:
	python tools/gen_golden_spy.py
	python tools/bless_golden.py

# Type checking (non-blocking for now)
type:
	mypy --strict || true

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf results/falsification_results.json

# CI target - run all checks
ci: lint test sanity falsify coverage

# Development setup
dev-setup: install
	pre-commit install

# Quick validation check
quick-sanity:
	pytest tests/test_data_integrity.py::TestDataSanity::test_time_series_validation -v

# Integration presence check
check-integration:
	pytest tests/test_data_sanity_enforcement.py::test_integration_presence -v

# Guard enforcement check
check-guards:
	pytest tests/test_data_sanity_enforcement.py::test_guard_enforcement -v

# Property-based testing
property-tests:
	pytest tests/test_data_sanity_enforcement.py -k "property" -v

# Edge case testing
edge-cases:
	pytest tests/test_data_sanity_enforcement.py::TestDataSanityEnforcement::test_edge_cases -v

# Falsification scenarios
falsify-scenarios:
	pytest tests/test_data_sanity_enforcement.py::test_falsification_scenarios -v

# Performance testing
perf-test:
	pytest tests/test_data_sanity_enforcement.py::TestDataSanityEnforcement::test_performance_safety -v -s

# Generate DataSanity report
sanity-report:
	python scripts/falsify_data_sanity.py --report-only

# All DataSanity checks
all-sanity: sanity falsify bench-sanity check-integration check-guards property-tests edge-cases falsify-scenarios perf-test

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Update pre-commit hooks
update-hooks:
	pre-commit autoupdate

# Go/No-Go gate for paper/live trading
go-nogo:
	@STRUCTURED_LOGS=1 RUN_ID=$$(date +%Y%m%d-%H%M%S) \
	MAX_POSITION_PCT=0.15 MAX_GROSS_LEVERAGE=2.0 \
	DAILY_LOSS_CUT_PCT=0.03 MAX_DRAWDOWN_CUT_PCT=0.20 MAX_TURNOVER_PCT=300 \
	python scripts/go_nogo.py

# Go/No-Go gate with custom environment
go-nogo-custom:
	@python scripts/go_nogo.py

# Block live trading (remove override file)
block-live:
	@rm -f runtime/ALLOW_LIVE.txt
	@echo "Live trading blocked. Remove runtime/ALLOW_LIVE.txt to re-enable."

# Allow live trading (create override file)
allow-live:
	@echo "# Manual Override for Live Trading" > runtime/ALLOW_LIVE.txt
	@echo "# Created: $$(date)" >> runtime/ALLOW_LIVE.txt
	@echo "# Purpose: Manual override for Go/No-Go gate" >> runtime/ALLOW_LIVE.txt
	@echo "Live trading enabled. Delete runtime/ALLOW_LIVE.txt to block."
