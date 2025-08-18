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

## Failure Codes

| Code | Meaning | Typical fix |
| ---- | ------- | ----------- |
| DUP_TS | Duplicate timestamps detected | Drop duplicates in data source; ensure strict monotonic index |
| NON_MONO_INDEX | Index is not strictly non-decreasing | Sort index; fix data emitter to write in order |
| NAN_VALUES | NaN fraction exceeds allowed limit | Fix data pipeline; fill/repair upstream; check `allow_ffill_nans` |
| INF_VALUES | Infinite values present | Replace/clip bad values at source |
| EMPTY_SERIES | No rows available for validation/run | Widen date range; ensure cache exists |
| SHORT_FOLD | Test length < step size with truncation disabled | Enable truncation or widen window |
| ZERO_TRADES | No trades executed in smoke window | Increase window slightly or lower signal threshold in config |
| BUDGET_EXCEEDED | Runtime exceeded budget | Reduce symbols or folds; increase budget if justified |
| UNEXPECTED_ERROR | Unhandled exception in smoke | Inspect `reports/smoke_run.json` and logs for traceback |

## Synthetic cache
Generate deterministic cache (no network):
```bash
python tools/gen_smoke_cache.py
```

## CI policy
- Smoke must pass on PRs/main
- Artifacts retained 7 days
- Promote gate validates `reports/smoke_run.json`
