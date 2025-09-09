# System Invariants - Runtime Enforcement

## Core Invariants

### 1. Single Decision Path
- **Rule**: Exactly one decision engine active per run
- **Enforcement**: `cfg.runtime.single_decision_core == True`
- **Violation**: System refuses to start
- **Code**: `core/decision_core.py:ensure_unified_decision_core()`

### 2. No Legacy Paths
- **Rule**: Legacy decision paths forbidden in production
- **Enforcement**: `cfg.runtime.allow_legacy_paths == False`
- **Violation**: System refuses to start
- **Code**: `core/decision_core.py:ensure_unified_decision_core()`

### 3. Deterministic Execution
- **Rule**: All runs emit `code_hash`, `config_hash`, `data_snapshot_id`
- **Enforcement**: Logging system validates presence
- **Violation**: Logs marked as invalid
- **Code**: `core/logging.py:validate_log_entry()`

### 4. Reproducible Results
- **Rule**: Folds produce identical metrics across repeated runs with same seeds
- **Enforcement**: Determinism test suite
- **Violation**: Test suite fails
- **Code**: `scripts/falsification_harness.py:run_determinism_test()`

### 5. Cost Consistency
- **Rule**: Costs applied on position change only
- **Enforcement**: Runtime asserts in decision core
- **Violation**: AssertionError raised
- **Code**: `core/decision_core.py:simulate_step()`

### 6. No NaN/Inf Propagation
- **Rule**: No trade on NaN/Inf features; fail fast
- **Enforcement**: Input validation in decision core
- **Violation**: ValueError raised immediately
- **Code**: `core/decision_core.py:validate_decision_inputs()`

### 7. Position State Consistency
- **Rule**: Position changes only on BUY/SELL actions
- **Enforcement**: Runtime asserts
- **Violation**: AssertionError raised
- **Code**: `core/decision_core.py:next_position()`

### 8. τ Threshold Consistency
- **Rule**: τ, costs, and position-logic identical across eval, falsification, and live
- **Enforcement**: Shared decision core module
- **Violation**: Different results across environments
- **Code**: `core/decision_core.py:decide()`

## Runtime Guards

### Startup Validation
```python
def validate_system_startup(cfg):
    ensure_unified_decision_core(cfg)
    validate_config_hash(cfg)
    validate_code_hash()
    validate_data_snapshot()
    return True
```

### Per-Decision Validation
```python
def validate_decision_inputs(logits, advantage, cfg):
    assert torch.isfinite(logits).all()
    assert torch.isfinite(advantage).all()
    assert cfg.tau > 0
    assert cfg.temperature > 0
    return True
```

### Position Update Validation
```python
def validate_position_update(prev_pos, action, new_pos):
    if action == HOLD:
        assert new_pos == prev_pos
    elif action == BUY:
        assert new_pos == 1
    elif action == SELL:
        assert new_pos == -1
    return True
```

## Violation Handling

### Critical Violations (System Halt)
- Multiple decision paths detected
- NaN/Inf in decision inputs
- Position state inconsistency
- Cost calculation error

### Warning Violations (Log and Continue)
- Config hash mismatch
- Code hash mismatch
- Data snapshot drift

### Monitoring Violations (Alert)
- Excessive HOLD rate
- Cost model drift
- Performance degradation

## Enforcement Points

1. **System Startup**: Validate all invariants before starting
2. **Decision Time**: Validate inputs and outputs
3. **Position Update**: Validate state transitions
4. **Cost Calculation**: Validate cost application
5. **Logging**: Validate log entry completeness
6. **End of Run**: Validate final state consistency
