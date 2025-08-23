# Solo PR Checklist

## Change Summary
**Type**: `feat|fix|perf|ops|docs` **(core/module)**: Brief description

**Why now?** One-line business justification.

## Solo Review Checklist

### ✅ Scope & Design
- [ ] **Single problem focus** - Change addresses exactly one issue
- [ ] **Clear "why now"** - Business justification documented
- [ ] **Backward compatibility** assessed and documented
- [ ] **Rollback plan** identified (feature flag, revert commit, config change)

### ✅ Code Quality
- [ ] **Public API documented** with examples in docstrings
- [ ] **Unit tests added** for new functionality
- [ ] **Golden tests added** for deterministic behaviors
- [ ] **Invariants & failure modes** documented in docstrings
- [ ] **Logging added** at system boundaries (ingest, model I/O, orders)

### ✅ Security & Performance
- [ ] **Security risks checked** (file I/O, eval, shell commands, secrets)
- [ ] **Performance impact measured** - benchmark results attached
- [ ] **Memory/allocation impact** considered
- [ ] **Configuration completeness** - no undeclared env vars or globals

### ✅ Quant-Specific Hygiene
- [ ] **Leakage guards verified** - no forward-looking data
- [ ] **Signal fidelity maintained** - IC/IR within expected bounds
- [ ] **Backtest realism** - costs, slippage, latency properly modeled
- [ ] **Parity tested** - same inputs produce identical outputs
- [ ] **Risk controls tested** - position limits, exposure, vetos working

### ✅ Hard Gates Passed
- [ ] **Static analysis**: `make gate-security` ✅
- [ ] **Type checking**: `make gate-type` ✅  
- [ ] **Coverage**: `make gate-cov` (≥85%) ✅
- [ ] **Performance**: `make gate-perf` (no regression) ✅
- [ ] **Parity**: `make gate-parity` ✅
- [ ] **Leakage**: `make gate-leakage` ✅

## Hostile Review (Break It)
Attempted to break with:
- [ ] **Bad inputs** - NaNs, out-of-order timestamps, duplicates, timezone shifts
- [ ] **Config errors** - deleted/renamed keys fail loudly with helpful messages
- [ ] **Network failures** - retry/backoff behavior tested
- [ ] **Resource limits** - memory/disk exhaustion handled gracefully

## Performance Impact
```bash
# Benchmark results
make gate-perf
```

**Result**: No regression | X% improvement | X% regression (justified because...)

## Security Assessment
```bash
make gate-security
```

**Findings**: No issues | Minor issues (documented) | Major issues (blocked)

## Files Changed
**Core files**: (list if any)
**Config files**: (list changes to config schema/defaults)
**Test files**: (list new test coverage)

---

**Merge criteria**: All checkboxes ✅, gates passed, performance acceptable, security clean.
