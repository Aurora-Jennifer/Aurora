# Master Plan: P0-P3 Multi-Asset Universe Runner Improvements

**Date**: 2025-09-07  
**Status**: P0 Complete, P1.2 In Progress  
**Last Updated**: 03:10 UTC

## Executive Summary

This document outlines the comprehensive improvement plan for the multi-asset universe runner, addressing critical correctness issues (P0), portfolio construction improvements (P1), diagnostics enhancements (P2), and operational quality (P3).

## Current Status Overview

### ✅ P0 — Correctness & Stability (COMPLETED)
- **P0.1**: Fixed IR_mkt & β calculation using strategy active returns
- **P0.2**: Fixed MaxDD = 0 bug with proper equity curve calculation
- **P0.3**: Fixed feature count mismatch & leakage guard with frozen whitelist
- **P0.4**: Fixed parallel crashes & log corruption with thread caps and QueueHandler logging

### 🔄 P1 — Portfolio Construction & Gates (IN PROGRESS)
- **P1.1**: ✅ Scaled gates with test window length (adaptive thresholds)
- **P1.2**: 🔄 Beta and sector neutralization (CURRENT FOCUS)
- **P1.3**: ⏳ Turnover controls (5-bucket staggering + trade bands)
- **P1.4**: ⏳ Transaction costs (half-spread + impact model)

### ⏳ P2 — Diagnostics & Research Loop (PENDING)
- **P2.1**: Score-decay & holding-period curves
- **P2.2**: Exposure dashboard
- **P2.3**: Robust CV & embargo
- **P2.4**: Sanity baselines

### ⏳ P3 — Quality & Operations (PENDING)
- **P3.1**: Thread-safe logging polish
- **P3.2**: Reproducibility
- **P3.3**: Assertions suite

---

## P0 — Correctness & Stability (COMPLETED)

### P0.1: Fix IR_mkt & β Calculation ✅
**Problem**: IR_mkt was ~0 for everyone due to OLS residual artifact  
**Solution**: Use strategy active returns, not per-stock or misaligned series  
**Implementation**: 
- Modified `ml/metrics_market_neutral.py` to use alpha t-statistic as IR_mkt
- Implemented OLS with HAC/NW for t-stats
- Used `active = s_t - (alpha + beta * m_t)` for proper active returns

**Files Modified**:
- `ml/metrics_market_neutral.py` - Fixed IR_mkt calculation
- `ml/runner_universe.py` - Updated CAPM integration

### P0.2: Fix MaxDD = 0 Bug ✅
**Problem**: Portfolio MaxDD was always 0.0% due to incorrect calculation  
**Solution**: Proper equity curve and drawdown calculation  
**Implementation**:
```python
equity = (1.0 + s_t).cumprod()
running_peak = np.maximum.accumulate(equity)
drawdown = equity / running_peak - 1.0
maxdd = float(drawdown.min())
```

**Files Modified**:
- `ml/runner_universe.py` - Fixed drawdown calculation in `topk_ls`

### P0.3: Fix Feature Count Mismatch & Leakage Guard ✅
**Problem**: Feature count mismatch (28→47) risked leakage  
**Solution**: Frozen whitelist with regex exclusion and per-symbol shift  
**Implementation**:
- Added forbidden columns and forward-looking patterns
- Created feature whitelist system
- Applied per-symbol shift before any split
- Added comprehensive logging

**Files Modified**:
- `ml/runner_universe.py` - Added feature whitelist system
- `ml/panel_builder.py` - Fixed data sorting for forward returns

### P0.4: Fix Parallel Crashes & Log Corruption ✅
**Problem**: Exit code 134 crashes and log corruption from parallel processing  
**Solution**: Thread caps and QueueHandler logging  
**Implementation**:
- Set thread caps: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.
- Added `_worker_init` function for joblib workers
- Implemented headless environment hardening
- Added faulthandler for crash forensics

**Files Modified**:
- `ml/runner_universe.py` - Added thread caps and worker initialization

---

## P1 — Portfolio Construction & Gates (IN PROGRESS)

### P1.1: Scale Gates with Test Window Length ✅
**Problem**: Gates were impossible for short test windows (54 bars vs 100 trades)  
**Solution**: Use t-stat thresholds and annualized metrics  
**Implementation**:
- Created `config/gates_adaptive.yaml` with scaling parameters
- Implemented dynamic threshold adjustment based on test length
- Added portfolio-level gates (IR_mkt, |β|, turnover, alpha t, drawdown)

**Files Modified**:
- `config/gates_adaptive.yaml` - New adaptive gate configuration
- `ml/runner_universe.py` - Implemented adaptive gate logic

### P1.2: Beta and Sector Neutralization 🔄 (CURRENT)
**Goal**: Remove unwanted market and sector exposures from signals  
**Implementation**:
- Created `ml/risk_neutralization.py` module
- Implemented OLS-based factor removal
- Added cross-sectional neutralization per date
- Designed configurable factor selection
- Updated TODO list with specific neutralization tasks

**Files Created**:
- `ml/risk_neutralization.py` - Core neutralization functions

**Files Modified**:
- `ml/runner_universe.py` - Added neutralization import

**Status**: Module created, integration in progress

**Current Tasks**:
- [ ] Complete integration into training pipeline
- [ ] Add market cap data extraction
- [ ] Add sector mapping configuration
- [ ] Test with 60-day OOS run
- [ ] Validate exposure reduction

### P1.3: Turnover Controls ⏳ (NEXT)
**Goal**: Implement 5-bucket staggering + trade bands to reduce turnover  
**Planned Implementation**:
- Split universe into 5 disjoint groups by stable hash
- Rebalance only one bucket each day/week
- Add bands: open longs if `rank ≥ 0.9`, keep until `rank < 0.85`
- Implement weight smoothing (EWMA to target weights)

### P1.4: Transaction Costs ⏳
**Goal**: Add realistic transaction costs to portfolio construction  
**Planned Implementation**:
- `cost = half_spread_bps + fee_bps + slippage_model(vol, adv_share, order_size)`
- Apply cost on dollar turnover each rebalance
- Report gross vs net returns

---

## P2 — Diagnostics & Research Loop (PENDING)

### P2.1: Score-Decay & Holding-Period Curves
**Goal**: Build decile long–short returns by skip-lags to see decay  
**Implementation**: Plot returns by holding period (1, 3, 5, 10, 20 bars)

### P2.2: Exposure Dashboard
**Goal**: Rolling regressions of portfolio returns on factors  
**Implementation**: 20/60 bar rolling regressions on MKT, SMB, HML, MOM, QMJ + sector exposures

### P2.3: Robust CV & Embargo
**Goal**: Purged, embargoed time splits for model selection  
**Implementation**: `PurgedGroupTimeSeriesSplit` with proper embargo periods

### P2.4: Sanity Baselines
**Goal**: Simple cross-sectional momentum & value decile L/S with same costs  
**Implementation**: Baseline strategies for performance comparison

---

## P3 — Quality & Operations (PENDING)

### P3.1: Thread-Safe Logging Polish
**Goal**: Rotate files, include run-id/seed, persist config & feature list  
**Implementation**: Structured logging with rotation and metadata

### P3.2: Reproducibility
**Goal**: Set and log random_state, XGBoost seeds, library versions  
**Implementation**: Run manifest (YAML) with all reproducibility info

### P3.3: Assertions Suite
**Goal**: Run every backtest with comprehensive validation  
**Implementation**: 
- Monotonic dates, no duplicates
- `np.allclose` for forward returns vs realized
- No NaNs in features at train/predict time
- Feature whitelist equality
- Portfolio weight sum, gross leverage, and cash constraints

---

## Key Files and Modules

### Core Files
- `ml/runner_universe.py` - Main orchestration script
- `ml/metrics_market_neutral.py` - CAPM and market-neutral metrics
- `ml/risk_neutralization.py` - Risk neutralization functions
- `ml/panel_builder.py` - Multi-asset panel construction

### Configuration Files
- `config/gates_adaptive.yaml` - Adaptive gate thresholds
- `config/gates_smoke.yaml` - Relaxed gates for testing
- `config/universe_top300.yaml` - Universe and market proxy config

### Test and Validation
- `debug_ir_mkt.py` - IR_mkt calculation debugging
- `debug_maxdd.py` - MaxDD calculation validation
- Various smoke test scripts

---

## Success Metrics

### P0 Success Criteria ✅
- [x] IR_mkt no longer ~0 for all tickers
- [x] MaxDD is non-zero on volatile series
- [x] Feature count equals whitelist every run
- [x] No crashes over 10 repeated runs
- [x] Logs are ordered and uncorrupted

### P1 Success Criteria
- [x] Gates scale with test window length
- [ ] Portfolio beta and sector exposures closer to zero
- [ ] Turnover drops materially without hurting IC
- [ ] Performance tables show gross vs net returns

### P2 Success Criteria
- [ ] Score decay curves show monotone decay
- [ ] β, sector tilts centered near zero with tight bands
- [ ] Train CV metrics line up with OOS
- [ ] Model beats at least one strong baseline

### P3 Success Criteria
- [ ] All assertions pass on every backtest
- [ ] Reproducible runs with identical results
- [ ] Clean, structured logging with rotation

---

## Next Immediate Actions

1. **Complete P1.2**: Finish risk neutralization integration
   - [ ] Add neutralization call to training pipeline
   - [ ] Extract market cap data from panel
   - [ ] Add sector mapping configuration
2. **Test P1.2**: Run 60-day OOS with neutralization
3. **Validate P1.2**: Check exposure reduction
4. **Start P1.3**: Implement turnover controls
5. **Continue P1.4**: Add transaction costs

---

## Risk Management

### Technical Risks
- **Data Availability**: Market cap and sector data may be missing
- **Performance Impact**: Neutralization may slow training
- **Over-Neutralization**: May eliminate alpha

### Mitigation Strategies
- Graceful fallbacks for missing data
- Performance monitoring and optimization
- Configurable neutralization factors
- Comprehensive testing and validation

---

## Timeline

- **P0**: ✅ Completed (2025-09-07)
- **P1.1**: ✅ Completed (2025-09-07)
- **P1.2**: 🔄 In Progress (2025-09-07)
- **P1.3**: ⏳ Planned (2025-09-07)
- **P1.4**: ⏳ Planned (2025-09-07)
- **P2**: ⏳ Planned (2025-09-08)
- **P3**: ⏳ Planned (2025-09-08)

---

*This document will be updated as we progress through each phase.*
