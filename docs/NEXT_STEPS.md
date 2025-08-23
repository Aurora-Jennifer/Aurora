## Production Readiness: Next Steps and Hard Gates (L0 → L2)

Purpose: achieve production readiness through operational discipline and hard gate enforcement.

### Next 7 days (priority, blocking)
- **Wire L0 gates to real E2D outputs**: Update test fixtures to use actual E2D artifacts
- **Implement circuit breakers**: Add notional limits, rate limiting, price band validation
- **Implement kill switch**: Add environment variable and file-based emergency stop
- **Add idempotency/crash recovery**: Implement checkpoint/resume capability
- **Begin L1 gate implementation**: Corporate actions, export parity edge cases, mutation testing

### Production readiness criteria (target after ~2 weeks)
- **L0 gates**: All passing with real E2D outputs (not mocks)
- **L1 gates**: Nightly CI passing, mutation testing ≥80% score
- **L2 gates**: Pre-promotion validation, ratchet enforcement vs goldens
- **Safety**: Circuit breakers, kill switch, idempotency validated
- **Promotion**: Flags default off, ADR required, rollback tested

### Immediate actions (low risk)
- **Wire L0 gates**: Update test fixtures to load actual E2D artifacts
- **Add circuit breakers**: Implement notional limits and rate limiting
- **Add kill switch**: Environment variable and file-based triggers
- **CI integration**: Add L1 gates to nightly schedule, L2 gates to promotion workflow

### Hard gate implementation (prioritized)
1) **L0 gates to real E2D**
   - Update test fixtures to load actual E2D artifacts instead of mocks
   - Validate real timezone handling, dtypes, flags, snapshot integrity

2) **Circuit breakers & safety**
   - Add notional limits to paper broker
   - Add rate limiting to order router  
   - Add price band validation
   - Implement kill switch (env var + file-based)

3) **Idempotency & crash recovery**
   - Add checkpoint/resume capability to E2D
   - Implement exactly-once decisioning
   - Add crash recovery tests

4) **L1 gates (nightly CI)**
   - Corporate actions normalization tests
   - Export parity edge case tests
   - Mutation testing for metrics math

5) **L2 gates (pre-promotion)**
   - Ratchet enforcement vs golden metrics
   - Cross-process determinism validation
   - Promotion checklist automation

6) **Integration & validation**
   - Wire all gates to CI workflows
   - Add rollback testing procedures
   - Document operational runbooks

### Flags and rollback
- **Safety flags**: `AURORA_KILL=1`, `AURORA_ROLLBACK=1`, `kill.flag` file
- **Feature flags**: All new features default off, require explicit enable
- **Rollback**: Single env var disables new behavior, documented procedures


