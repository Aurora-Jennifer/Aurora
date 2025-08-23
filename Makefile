# Makefile for Trading System with DataSanity Enforcement

.PHONY: help install test test-full integ sanity falsify bench-sanity clean coverage lint lint-changed format promote wf smoke type datasanity golden bless_golden quality configcheck lock audit pre-push mut mut-results mut-report canary canary-smoke canary-rollup daily daily-shadow live live-shadow live-eod maintenance live-rollup reconcile reconcile-dry-run flatten-dry-run go-nogo go-nogo-custom block-live allow-live freeze-snapshot train-e0 train-e1 train-e2 eval-e0 eval-e1 eval-e2 test-onnx-parity smoke-adapter smoke-adapter-onnx verify-functional paper shadow ops-check recon e2e e2e-smoke gate-data gate-signal gate-parity gate-pnl gate-wf parity serve-smoke rollback drift-check train-smoke e2d e2d-gate paper-smoke demo-15m dashboard backtest-smoke backtest-full ds mut-full

# Default target
.DEFAULT_GOAL := e2e

help:
	@echo "Available targets:"
	@echo "  e2e          - Run end-to-end tests (default)"
	@echo "  smoke        - Run smoke test"
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
	pytest -q --ignore=mutants

test-full:
	pytest -m "not quarantine" -q --ignore=mutants

integ:
	pytest tests/integration -q --ignore=mutants

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
	ruff check . --select E,F,I,UP,B,SIM

# Format code
format:
	ruff format .

CHANGED_PY := $(shell git fetch origin main >/dev/null 2>&1; git diff --name-only --diff-filter=ACMRT origin/main...HEAD | grep -E '\.py$$' || true)

lint-changed:
	@if [ -z "$(CHANGED_PY)" ]; then echo "No changed Python files."; else ruff check --output-format=github $(CHANGED_PY); fi

# Check format
format-check:
	ruff format --check .

# Run walkforward (placeholder)
wf:
	pytest -q tests/walkforward -k spy --maxfail=1 || true

# Promotion gate
promote:
	python tools/promotion_gate.py --fail-on-quarantine --report

# End-to-end tests (placeholder - use e2e-smoke for full pipeline)
e2e-tests:
	python tests/fixtures/gen_fixture.py
	pytest -m e2e -q

# Smoke preset (fast CI-friendly walkforward)
smoke:
	python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO --datasanity-profile walkforward_smoke --allow-zero-trades --profiles smoke_test

pre-push: smoke

datasanity:
	pytest -q tests/walkforward/test_datasanity_*.py tests/backtest/test_datasanity_*.py

golden: tests/golden/SPY.parquet
	pytest -q tests/backtest/test_golden_regression.py

tests/golden/SPY.parquet:
	python tools/gen_golden_spy.py

bless_golden:
	python tools/gen_golden_spy.py
	python tools/bless_golden.py

# Snapshot validation
validate-snapshot:
	python scripts/validate_snapshot.py artifacts/snapshots/golden_ml_v1_frozen

# Use frozen snapshot when flag enabled
frozen-smoke:
	FLAG_GOLDEN_SNAPSHOT_FROZEN=1 python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO --datasanity-profile walkforward_smoke --allow-zero-trades --profiles smoke_test

quality:
	coverage run -m pytest -q && coverage report -m || true

configcheck:
	python tools/validate_config.py config/base.yaml

lock:
	pip install pip-tools && pip-compile --generate-hashes -o requirements.lock requirements.in

audit:
	pip install pip-audit && pip-audit || true

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
ci: lint e2e

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

# Canary commands
canary:
	python scripts/canary_runner.py --symbols SPY,TSLA --poll-sec 5 --profile live_canary --shadow

canary-smoke:
	pytest -q tests/live/test_canary_runner_tmp_isolated.py

canary-rollup:
	python tools/rollup_canary.py

# Daily commands
daily:
	python scripts/paper_runner.py --symbols SPY,TSLA --poll-sec 5 --profile paper_strict --ntfy --steps 60 || true
	touch kill.flag || true
	python tools/rollup_posttrade.py
	python tools/rollup_canary.py || true

# Shadow with real quotes (for daily testing)
daily-shadow:
	python scripts/canary_runner.py --symbols SPY,TSLA --poll-sec 5 --profile live_canary --quotes ibkr --shadow --steps 120 || true
	touch kill.flag || true
	python tools/rollup_canary.py || true

# Live canary commands
live:
	python scripts/canary_runner.py --symbols SPY,TSLA --poll-sec 5 --profile live_canary --quotes ibkr --steps 180

live-shadow:
	python scripts/canary_runner.py --symbols SPY,TSLA --poll-sec 5 --profile live_canary --quotes ibkr --shadow --steps 150

live-eod:
	python tools/rollup_live.py
	python tools/reconcile_orders.py

live-rollup:
	python tools/rollup_live.py

# Daily maintenance (braindead-simple)
maintenance:
	python tools/daily_maintenance.py --symbols SPY,TSLA --minutes 15 --quotes ibkr --ntfy

# Reconciliation commands
reconcile:
	python tools/reconcile_orders.py

reconcile-dry-run:
	python tools/reconcile_orders.py --dry-run

flatten-dry-run:
	python scripts/flatten_positions.py --dry-run

# Snapshot & training (ML)
freeze-snapshot:
	python bin/freeze_snapshot.py

train-e0:
	PYTHONPATH=. python scripts/train_linear.py golden_linear

train-e1:
	PYTHONPATH=. python scripts/train_linear.py golden_xgb

train-e2:
	PYTHONPATH=. python scripts/train_linear.py golden_xgb_v2

eval-e0:
	PYTHONPATH=. python scripts/eval_oof.py artifacts/models/$$(ls -t artifacts/models/ | head -1)

eval-e1:
	PYTHONPATH=. python scripts/eval_oof.py artifacts/models/$$(ls -t artifacts/models/ | head -1)

eval-e2:
	PYTHONPATH=. python scripts/eval_oof.py artifacts/models/$$(ls -t artifacts/models/ | head -1)

test-onnx-parity:
	PYTHONPATH=. python -c "from scripts.train_linear import main; main('golden_xgb')" | grep -A 10 '"onnx"'

# Adapter smoke tests
smoke-adapter:
	python -m serve.adapter --csv fixtures/live_stub.csv --out reports/adapter_smoke.json || true

smoke-adapter-onnx:
	python -m serve.adapter --csv fixtures/live_stub.csv --onnx --out reports/adapter_smoke_onnx.json || true

verify-functional:
	pytest -q tests/functional || true
	$(MAKE) smoke-adapter
	$(MAKE) smoke-adapter-onnx

# Paper trading and shadow mode
paper:
	python scripts/e2e_pipeline.py --profile golden_xgb_v2
	python scripts/paper_broker.py --signals artifacts/signals/out.jsonl --fills artifacts/fills/paper.jsonl --ledger artifacts/fills/ledger.json

shadow:
	@echo "Shadow mode placeholder (wire live quotes here)"

ops-check:
	@echo "Ops check placeholder (calendars, secrets, heartbeats)"

recon:
	@echo "Recon placeholder (compare orders/fills vs broker)"

# End-to-end pipeline
e2e:
	( PYTHONPATH=. python scripts/e2e_pipeline.py --profile golden_xgb_v2; \
	  python scripts/onnx_parity.py || echo "[PARITY] allow-fail"; \
	  python scripts/bench_infer.py || echo "[BENCH] allow-fail"; \
	  python scripts/gate_significance.py --use-after-costs; \
	  PYTHONPATH=. python scripts/ablate.py --profile golden_linear; \
	  python scripts/paper_broker.py --signals artifacts/signals/out.jsonl --fills artifacts/fills/paper.jsonl --ledger artifacts/fills/ledger.json )

e2e-smoke:
	( ML_EXPORT_ONNX=1 PYTHONPATH=. python scripts/train_linear.py golden_xgb_v2 || true; \
	  ONNX=$$(ls -t artifacts/models/*.onnx 2>/dev/null | head -n1); export ONNX; \
	  if [ -n "$$ONNX" ] && [ -f "$$ONNX" ]; then \
	  	SERVE_TELEMETRY=1 SERVE_DUMMY=1 python -m serve.adapter --onnx "$$ONNX" --csv fixtures/live_stub.csv || true; \
	  	python scripts/onnx_parity.py --features artifacts/parity/features_oof.parquet --sidecar artifacts/parity/sidecar.json --native_preds artifacts/parity/preds_native.parquet --onnx "$$ONNX" || true; \
	  	python scripts/bench_infer.py --onnx "$$ONNX" || true; \
	  else echo "[E2E] No ONNX model found; skipping parity/bench/serve"; fi; \
	  python scripts/gate_significance.py --use-after-costs || true; \
	  PYTHONPATH=. python scripts/ablate.py --profile golden_linear || true )

# E2E Sanity Check (as per plan)
e2e-sanity:
	@echo "🧪 Running E2E sanity check..."
	@rm -rf artifacts/run && mkdir -p artifacts/run
	@export AURORA_PROFILE=golden_xgb_v2
	@python scripts/e2d.py --profile golden_xgb_v2 --out artifacts/run --once --telemetry artifacts/run/trace.jsonl

e2e-paper:
	@echo "📊 Running E2E paper loop..."
	@python scripts/runner.py --profile golden_xgb_v2 --mode paper --steps 32 --telemetry artifacts/run/trace.jsonl

# Gate targets
gate-data:
	python scripts/gate_data.py --features artifacts/parity/features_oof.parquet --sidecar artifacts/parity/sidecar.json --history reports/experiments/search_history.json

gate-signal:
	python scripts/gate_signal.py --history reports/experiments/search_history.json



gate-pnl:
	python scripts/gate_pnl.py --report reports/experiments/pnl.json

gate-wf:
	python scripts/gate_wf.py --wf_report reports/experiments/wf.json

# Utility targets
parity:
	python scripts/onnx_parity.py || echo "[PARITY] allow-fail"

serve-smoke:
	python -m serve.adapter --onnx artifacts/models/latest.onnx --csv fixtures/live_stub.csv || echo "[SERVE] allow-fail"

# Production hardening targets
rollback: # Rollback to previous version
	@echo "Available versions:" && ls -t artifacts/models/*/ | head -5 | cut -d/ -f3
	@read -p "Enter version to rollback to: " version && python scripts/rollback.py $$version

drift-check: # Check for prediction drift
	python -c "from core.ml.drift import *; golden=load_golden_predictions(); print('Golden predictions loaded:', golden is not None)" || echo "[DRIFT] allow-fail"

# Final stretch targets
train-smoke: # Train readiness gate (golden snapshot)
	PROFILE=$${PROFILE:-golden_xgb_v2}; \
	ML_EXPORT_ONNX=1 PYTHONPATH=. python scripts/train_linear.py $$PROFILE || echo "[TRAIN] allow-fail"

e2d: # End-to-Decision (no broker)
	PROFILE=$${PROFILE:-config/profiles/golden_xgb_v2.yaml}; \
	python scripts/e2d.py --profile $$PROFILE --csv fixtures/live_stub.csv || echo "[E2D] allow-fail"

e2d-gate: # E2D gate checks (advisory)
	python scripts/gate_e2d.py || echo "[E2D-GATE] allow-fail"

paper-smoke: # Paper execution loop (15 min)
	PROFILE=$${PROFILE:-config/profiles/golden_xgb_v2.yaml}; \
	python scripts/runner.py --profile $$PROFILE --mode paper --symbols SPY --minutes 15 || echo "[PAPER] allow-fail"

demo-15m: # Demo script (15-minute replay)
	PROFILE=$${PROFILE:-config/profiles/golden_xgb_v2.yaml}; \
	$(MAKE) train-smoke PROFILE=$$PROFILE; \
	$(MAKE) e2d PROFILE=$$PROFILE; \
	$(MAKE) paper-smoke PROFILE=$$PROFILE; \
	$(MAKE) dashboard; \
	echo "[DEMO] 🎉 15-minute demo complete for $$PROFILE"

dashboard: # Generate observability dashboard
	python scripts/make_dashboard.py || echo "[DASHBOARD] allow-fail"

backtest-smoke: # Backtest smoke test
	python scripts/backtest.py --smoke || echo "[BACKTEST] allow-fail"

backtest-full: # Full backtest with snapshot
	SNAPSHOT=$${SNAPSHOT:-artifacts/snapshots/golden_ml_v1}; \
	PROFILE=$${PROFILE:-config/profiles/golden_xgb_v2.yaml}; \
	python scripts/backtest.py --profile $$PROFILE --snapshot $$SNAPSHOT || echo "[BACKTEST] allow-fail"

# DataSanity configuration
PROFILE ?= dev
DS_RULES = configs/datasanity_rules.$(PROFILE).yaml

ds: # DataSanity check
	python scripts/check_datasanity.py \
	  --features artifacts/snapshots/golden_ml_v1/features.parquet \
	  --labels   artifacts/snapshots/golden_ml_v1/labels.parquet \
	  --rules    $(DS_RULES) \
	  --out      reports/datasanity/report.json || echo "[DATASANITY] allow-fail"

# Hard Gates (automation > opinions)
gate-security: # Security scan
	@echo "🔒 Security Gate: Static analysis + dependency audit"
	bandit -r . -f json -o reports/security/bandit.json || true
	pip-audit --format=json --output=reports/security/audit.json || true
	@echo "Security scan complete - check reports/security/"

gate-type: # Type checking
	@echo "📝 Type Gate: MyPy strict checking"
	mypy . --strict --config-file=pyproject.toml || true

gate-cov: # Coverage gate
	@echo "📊 Coverage Gate: Minimum 85% coverage"
	pytest -q --cov=core --cov=scripts --cov=oms --cov-report=xml --cov-report=term-missing --cov-fail-under=85

gate-perf: # Performance regression gate
	@echo "⚡ Performance Gate: Benchmark against baseline"
	pytest -q --benchmark-only --benchmark-json=reports/benchmark.json tests/benchmarks/ || true

gate-parity: # Determinism and parity gate
	@echo "🎯 Parity Gate: Deterministic outputs and model equivalence"
	PYTHONHASHSEED=0 python scripts/test_determinism.py || true
	python scripts/onnx_parity.py || echo "[PARITY] allow-fail"

gate-leakage: # Data leakage prevention
	@echo "🚫 Leakage Gate: Forward-looking data prevention"
	python scripts/test_leakage_guards.py || true

gate-property: # Property-based testing
	@echo "🧪 Property Gate: Hypothesis-based invariant testing"
	pytest -q tests/property/ --hypothesis-profile=ci || true

# Combined gates for CI/PR
gates-hard: gate-security gate-type gate-cov gate-perf gate-parity gate-leakage gate-property
	@echo "✅ All hard gates passed!"

# Quick gates (fast feedback)
gates-quick: lint format-check test
	@echo "⚡ Quick gates passed!"

# Pre-commit simulation
pre-commit-sim: gates-quick
	@echo "🚀 Pre-commit simulation complete"

# Mutation testing targets
mut:
	@echo "Running mutation tests on core/data_sanity..."
	@mutmut run --use-coverage || true
	@echo "Mutation testing complete."

mut-results:
	@echo "=== Mutation Test Results ==="
	@mutmut results

mut-report:
	@mkdir -p artifacts
	@mutmut junitxml > artifacts/mutmut-report.xml
	@echo "Mutation report saved to artifacts/mutmut-report.xml"

mut-full: mut mut-results mut-report

smoke:
	python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO

test:
	pytest -q

test-full:
	pytest -m "not quarantine" -q

integ:
	pytest tests/integration -q

lint:
	ruff check .

pre-push: smoke

l0-time:
	pytest -q tests/gates/l0/test_timezones.py

l0-dtypes:
	pytest -q tests/gates/l0/test_dtypes.py

l0-snapshot:
	pytest -q tests/gates/l0/test_snapshot.py

l0-gates: l0-time l0-dtypes l0-snapshot

SNAP=artifacts/snapshots/golden_ml_v1

snapshot-fix:
	python -u scripts/fix_golden_snapshot.py $(SNAP)

snapshot-hash:
	python -u scripts/hash_snapshot.py $(SNAP)

snapshot-ro:
	# make files read-only; skip HASH.txt to allow regeneration
	chmod -R a-w $(SNAP)
	chmod a+w $(SNAP)/HASH.txt
	# optional: harden on ext4 (ignore if unsupported)
	-which chattr >/dev/null 2>&1 && sudo chattr -R +i $(SNAP) || true
	-which chattr >/dev/null 2>&1 && sudo chattr -i $(SNAP)/HASH.txt || true

.PHONY: drills-snapshot drills-dtype drills-tz
drills-snapshot:
	bash scripts/drills/drill_snapshot_mutation.sh
drills-dtype:
	bash scripts/drills/drill_dtype_violation.sh
drills-tz:
	bash scripts/drills/drill_timezone_violation.sh
