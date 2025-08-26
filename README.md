# Aurora Trader — Project README

<!-- Update OWNER/REPO once you push this branch -->
![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)

## Purpose
Deterministic, guardrail-first trading research and paper/live framework with DataSanity validation, walkforward orchestration, and strict CI discipline.

## Entrypoints
- `scripts/multi_walkforward_report.py` — smoke/backtest runner
- `core/data_sanity/api.py` — DataSanity validation API (engine switch + telemetry)
- `Makefile` — common targets: `smoke`, `test`, `test-full`, `integ`, `lint`

## Do-not-touch
- Contracts under `core/data_sanity/main.py` and public API of `core/data_sanity/api.py`
- Fold builder/walkforward invariants

## CI Status
Smoke: ![Smoke](https://img.shields.io/github/actions/workflow/status/Aurora-jennifer/aurora/ci.yml?branch=main&label=Smoke)

## Docs
- Summaries: `docs/summaries/` (core, data_sanity)
- Runbooks: `docs/runbooks/` (e.g., `datasanity_ops.md`)
- Audit: `docs/audit/`

## Reality Check

**Before you get too excited:** This is a retail-grade trading system. It won't make you rich.

- 📊 **What it does:** Basic technical analysis, paper trading, opportunity scoring
- 🚨 **What it doesn't:** Real-time data, institutional features, sophisticated models
- 🗺️ **Improvement roadmap:** See [docs/REALITY_CHECK.md](docs/REALITY_CHECK.md)

**Bottom line:** Good for learning, not for consistent profits.

## Quick Start
```bash
# Deterministic smoke + lineage validation
make smoke

# Focused walkforward tests (stable env)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests -k walkforward

# Smoke-marked tests only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -m smoke
```
