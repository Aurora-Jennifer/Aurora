# Action Plan - Production Readiness

## Current Status
- **Paper Trading Readiness**: 33/52 (63%)
- **L0 Gates**: ✅ Implemented and CI-integrated
- **Critical Gaps**: 7 blocking items prevent production deployment

## Immediate Actions (This Week)

### 1. Wire L0 Gates to Real E2D (Priority 1)
**Problem**: L0 gates use mock data, not real E2D outputs
**Solution**: 
- Update `tests/gates/l0/test_timezones.py` to load actual E2D artifacts
- Update `tests/gates/l0/test_dtypes.py` to validate real feature outputs
- Update `tests/gates/l0/test_snapshot.py` to use actual snapshot data
- **Time**: 2-3 hours
- **Risk**: Low (test-only changes)

### 2. Implement Circuit Breakers (Priority 1)
**Problem**: No protection against runaway orders or system failures
**Solution**:
- Add notional limits to paper broker
- Add rate limiting to order router
- Add price band validation
- **Time**: 4-6 hours
- **Risk**: Medium (touches execution logic)

### 3. Implement Kill Switch (Priority 1)
**Problem**: No emergency stop mechanism
**Solution**:
- Add environment variable kill switch (`AURORA_KILL=1`)
- Add file-based kill switch (`kill.flag`)
- Wire to all execution components
- **Time**: 2-3 hours
- **Risk**: Low (safety feature)

### 4. Add Idempotency/Crash Recovery (Priority 2)
**Problem**: System crashes leave corrupted state
**Solution**:
- Add checkpoint/resume capability to E2D
- Implement exactly-once decisioning
- Add crash recovery tests
- **Time**: 6-8 hours
- **Risk**: Medium (state management)

## Next Week Actions

### 5. Implement L1 Gates (Nightly CI)
**Problem**: No adversarial/mutation testing
**Solution**:
- Corporate actions normalization tests
- Export parity edge case tests
- Mutation testing for metrics math
- **Time**: 8-10 hours
- **Risk**: Low (test infrastructure)

### 6. Implement L2 Gates (Pre-Promotion)
**Problem**: No promotion discipline or ratchet enforcement
**Solution**:
- Ratchet enforcement vs golden metrics
- Cross-process determinism validation
- Promotion checklist automation
- **Time**: 6-8 hours
- **Risk**: Low (validation only)

## Success Criteria

### Week 1 Goals
- [ ] L0 gates validate real E2D outputs
- [ ] Circuit breakers prevent pathological orders
- [ ] Kill switch stops system immediately
- [ ] **Progress**: 33/52 → 38/52 (73%)

### Week 2 Goals
- [ ] L1 gates catch edge cases nightly
- [ ] Idempotency handles crashes gracefully
- [ ] L2 gates enforce promotion discipline
- [ ] **Progress**: 38/52 → 45/52 (87%)

### Production Ready Criteria
- [ ] All L0/L1/L2 gates passing
- [ ] Circuit breakers and kill switch tested
- [ ] Idempotency validated under crash conditions
- [ ] Ratchet enforcement preventing regressions
- [ ] **Status**: Production Ready

## Risk Mitigation

### High Risk Items
1. **Circuit breakers**: Test thoroughly with adversarial inputs
2. **Idempotency**: Validate crash scenarios in CI
3. **Kill switch**: Test emergency stop procedures

### Rollback Plan
- All new features behind feature flags
- Single env var disables new behavior: `AURORA_ROLLBACK=1`
- Documented rollback procedures for each component

## Daily Check-ins

### Monday
- Wire L0 gates to real E2D outputs
- Start circuit breaker implementation

### Tuesday
- Complete circuit breakers
- Start kill switch implementation

### Wednesday
- Complete kill switch
- Start idempotency implementation

### Thursday
- Continue idempotency work
- Begin L1 gate planning

### Friday
- Complete idempotency
- Plan L1 gate implementation

## Blockers & Dependencies

### Current Blockers
- None identified

### Potential Blockers
- Complex state management in idempotency
- Integration issues with existing risk engine
- Performance impact of additional safety checks

### Dependencies
- E2D pipeline stability for real output testing
- CI infrastructure for L1/L2 gate integration
- Documentation for operational procedures
