# Aurora Trader — Project README

## Purpose
Deterministic, guardrail-first trading research and paper/live framework with DataSanity validation, walkforward orchestration, and strict CI discipline.

## Entrypoints
- `scripts/multi_walkforward_report.py` — smoke/backtest runner
- `core/data_sanity/api.py` — DataSanity validation API (engine switch + telemetry)
- `apps/walk_cli.py` — CLI entry for walk/experiment flows
- `Makefile` — common targets: `smoke`, `test`, `test-full`, `integ`, `lint`, `canary`

## Do-not-touch
- Contracts under `core/data_sanity/main.py` and public API of `core/data_sanity/api.py`
- Fold builder/walkforward invariants

## CI Status
Smoke: ![Smoke](https://img.shields.io/github/actions/workflow/status/Aurora-jennifer/aurora/ci.yml?branch=main&label=Smoke)

## Docs
- Summaries: `docs/summaries/` (core, data_sanity)
- Runbooks: `docs/runbooks/` (e.g., `datasanity_ops.md`)
- Audit: `docs/audit/`

## Quick Start
```bash
make smoke           # deterministic smoke
pytest -q            # tests
python scripts/canary_datasanity.py --profiles walkforward_smoke -v
```
