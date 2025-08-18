# Smoke Runbook

## Inputs
- Symbols: default SPY,BTC-USD (override via CLI)
- Dates: default 2020-01-01 → 2020-03-31
- Profiles: risk_balanced (internal), composer assets via config

## Outputs
- Markdown: `docs/analysis/walkforward_smoke_<ts>.md`
- JSON: `reports/smoke_run.json`

## How to run locally
```bash
make smoke
# or
python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO
```

## Common failures & fixes
- 0 folds: widen date range or reduce train/test windows
- No trades: reduce thresholds or increase test window
- DataSanity violations: fix duplicates/NaNs/non-monotonic index in source or synthetic cache
- Runtime >60s: reduce symbols / windows, use CI cache

## Synthetic cache
Generate deterministic cache (no network):
```bash
python tools/gen_smoke_cache.py
```

## CI policy
- Smoke must pass on PRs/main
- Artifacts retained 7 days
- Promote gate validates `reports/smoke_run.json`
