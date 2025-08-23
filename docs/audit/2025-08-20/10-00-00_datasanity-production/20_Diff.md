# Diff Summary

## Files Modified: 10

### config/base.yaml
- **Lines changed**: ~10
- **Key changes**: Added `datasanity` configuration section with engine selection and telemetry settings

### core/data_sanity/config.py (NEW)
- **Lines added**: ~80
- **Key features**: Centralized configuration management with dot-separated path access and profile config loading

### core/data_sanity/telemetry.py (NEW)
- **Lines added**: ~120
- **Key features**: JSONL telemetry system with validation run tracking and statistics aggregation

### core/data_sanity/metrics.py (NEW)
- **Lines added**: ~100
- **Key features**: Metrics collection with error budget tracking and export functionality

### core/data_sanity/api.py
- **Lines changed**: ~15
- **Key changes**: Added engine switch facade, telemetry imports, and updated exports

### core/data_sanity/main.py
- **Lines changed**: ~5
- **Key changes**: Updated docstring to indicate v1 implementation

### scripts/canary_datasanity.py (NEW)
- **Lines added**: ~300
- **Key features**: Canary testing script with v1/v2 comparison and regression detection

### tests/datasanity/test_engine_switch.py (NEW)
- **Lines added**: ~150
- **Key features**: Comprehensive tests for engine switch functionality and telemetry

### Makefile
- **Lines changed**: ~3
- **Key changes**: Added `canary` target for running canary tests

### .pre-commit-config.yaml
- **Lines changed**: ~10
- **Key changes**: Added DataSanity smoke test hook and updated pre-commit configuration

## Summary
- **Insertions**: ~850 lines
- **Deletions**: ~0 lines
- **Net change**: +850 lines
- **Scope**: Production rollout infrastructure with observability and safety mechanisms
