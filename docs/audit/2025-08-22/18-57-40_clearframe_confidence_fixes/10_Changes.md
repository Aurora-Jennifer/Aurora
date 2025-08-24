# Changes — Clearframe Confidence Fixes

## Actions Completed
1. **✅ Created `config/profiles/yfinance_live_1m.yaml`** - Live profile with `live: true` flag
2. **✅ Enhanced staleness detection** - Bar-interval-based checks for live vs historical  
3. **✅ Added live yfinance fetching** - `_fetch_latest_bars` now fetches current data when `live: true`
4. **✅ Improved risk cap logging** - Detailed reasons for all qty caps
5. **✅ Tightened price guards** - Added recent window + mid/last deviation checks

## Commands run
```bash
# Test live profile
timeout 45s python scripts/runner.py --profile config/profiles/yfinance_live_1m.yaml --minutes 1

# Results: SUCCESS! 
# - bar_end=2025-08-22 19:59:00+00:00 (current year, not 2000!)
# - Live yfinance fetching working
# - Proper staleness rejection (28h old data from market close)
```

## Key Code Changes

### `scripts/runner.py`
- **Historical detection**: Fixed KeyError by using `profile["data"].get("live", False)` 
- **Live fetching**: Added yfinance live data fetch in `_fetch_latest_bars`
- **Enhanced logging**: `[RISK] CAP {symbol} qty {qty}→0 reason=historical_data`
- **Staleness gates**: Bar-interval tolerances for live data

### `core/guards/price.py`  
- **Recent window check**: 1-day trading range validation
- **Mid/last deviation**: 1% price consistency check

### `config/profiles/yfinance_live_1m.yaml`
- **Live mode**: `live: true, require_live_data: true`
- **Real-time params**: `interval: 1m, period: 5d`
- **Tight guards**: `jump_limit_frac: 0.05, band_frac: 0.02`

## Validation Results

| Metric | Before (2000 replay) | After (Live) | Status |
|--------|---------------------|--------------|---------|
| `bar_end` | `2000-12-XX` | `2025-08-22` | ✅ Fixed |
| Data source | Historical files | Live yfinance | ✅ Fixed |
| Staleness | 9000+ days | 28 hours | ✅ Detected |
| Risk caps | Silent `qty_capped_to_zero` | `reason=historical_data` | ✅ Detailed |
| Price guards | Static bands | Adaptive regime | ✅ Enhanced |
