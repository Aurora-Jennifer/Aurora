# TODO - Aurora Trading System

## Current Status
- **Paper Trading Readiness**: 33/52 (63%) - Updated with hard gate requirements
- **L0 Gates**: ✅ Implemented and integrated into CI
- **L1/L2 Gates**: 📋 Required for production readiness

## High Priority (Blocking Production)

### Hard Gates Implementation
- [ ] **Wire L0 gates to real E2D outputs** (currently using mocks)
- [ ] **Implement L1 Gates** (nightly CI):
  - [ ] Corporate actions normalization
  - [ ] Export parity edge cases  
  - [ ] Idempotency & crash recovery
  - [ ] Circuit breakers & kill switch
  - [ ] Mutation testing (metrics + data_sanity)
- [ ] **Implement L2 Gates** (pre-promotion):
  - [ ] Ratchet enforcement vs goldens
  - [ ] Cross-process determinism
  - [ ] Promotion discipline validation

### Critical Safety Gaps
- [ ] **Circuit breakers**: Notional limits, rate limits, price bands
- [ ] **Kill switch**: Environment variable + file-based triggers
- [ ] **Idempotency**: Crash recovery, resume from last offset
- [ ] **Rollback discipline**: Single env var disables new behavior

### Operational Discipline
- [ ] **Mutation testing**: Validate metrics math and contamination detection
- [ ] **Ratchet enforcement**: Prevent performance regressions
- [ ] **Promotion gates**: Flags default off, ADR required

## Medium Priority

### Integration & Testing
- [ ] Integration test: mock trade loop runs, logs decisions
- [ ] Backtest parity against golden snapshot
- [ ] Traces: span per stage, inputs hash, artifact paths

### Documentation & Runbooks
- [ ] Runbooks: incident + perf regression documented
- [ ] Runbook updated with hard gate procedures
- [ ] Rollback tested and documented

## Completed ✅

### Core Functionality
- [x] Data sources connected (broker API, market data)
- [x] DataSanity suite running (schema, leakage, NaNs, monotonic index)
- [x] Golden snapshot frozen (reference dataset for determinism)
- [x] E2D pipeline: data → features → model → signal → position
- [x] Paper broker: position tracking, PnL, mock fills
- [x] Risk engine: position limits, stop-loss, max exposure
- [x] Structured logs (JSON, include run_id, phase, duration_ms)
- [x] Metrics: IC, turnover, fill_rate, latency, memory_peak
- [x] Metrics contract: schema validation, golden reference, ratchet enforcement

### CI & Testing
- [x] Lint + unit tests green
- [x] Datasanity tests pass
- [x] Train-smoke runs in CI
- [x] Export parity test green
- [x] End-to-Decision smoke green
- [x] L0 Gates: timezone, dtypes, flags, snapshot (PR CI, <3min)

## Notes

### Hard Gate Strategy
- **L0 Gates**: Fast, always-on, prevent basic operational failures
- **L1 Gates**: Adversarial/mutation, catch edge cases without blocking PRs  
- **L2 Gates**: Pre-promotion, ensure quality before live deployment

### Production Readiness Criteria
- All L0/L1/L2 gates passing
- Circuit breakers and kill switch implemented
- Idempotency and crash recovery validated
- Ratchet enforcement preventing regressions
- Promotion discipline enforced

### Next Actions
1. Wire L0 gates to real E2D outputs (not mocks)
2. Implement circuit breakers and kill switch
3. Add idempotency/crash recovery
4. Implement L1 gates for nightly validation
5. Add L2 gates for promotion discipline
