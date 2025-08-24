# cli

## Purpose
Command-line commands for dev and ops flows. No business logic here.

## Entrypoints
- `cli/backtest.py` — run backtests with flags
- `cli/paper.py` — paper trading helpers
- `apps/walk_cli.py` — walk/experiment entry

## Do-not-touch
- No hidden state; all inputs via flags/env
- Keep command names stable; forward to `core/` and `scripts/`
