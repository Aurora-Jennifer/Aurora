# features

## Purpose
Feature builders from validated OHLCV to model-ready columns.

## Entrypoints
- `features/feature_engine.py` — feature transforms
- `ml/features/build_daily.py` — daily feature pipeline

## Do-not-touch
- No label leakage; UTC tz-aware index; purged splits only
