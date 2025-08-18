# DataSanity Upgrade & Testing Guardrails

## Overview

This document defines the guardrails and testing requirements for the DataSanity module and related data validation components in the trading system. These guardrails ensure data validation remains robust, performant, and reliable.

## Core Principles

1. **Functional > Performance** — Never sacrifice correctness for speed
2. **Incremental Changes** — All changes must be incremental and reversible
3. **No Full Rewrites** — Use v1/v2 or feature flags for risky upgrades
4. **Backward Compatibility** — All schema/API changes must include adapters

## Performance Thresholds

### Baselines (Based on Known-Good Runs)

| Dataset Size | Max Duration | Memory Limit |
|--------------|--------------|--------------|
| 100 rows     | ≤ 0.05s      | N/A          |
| 1,000 rows   | ≤ 0.20s      | N/A          |
| 10,000 rows  | ≤ 0.60s      | ≤ 250 MB RSS |

### Tolerance Modes

- **RELAXED (local default):** Allow up to **+30%** over thresholds
- **STRICT (CI):** Allow up to **+10%** over thresholds

### Mode Control

```bash
# Local development (relaxed)
export SANITY_PERF_MODE=RELAXED

# CI/CD (strict)
export SANITY_PERF_MODE=STRICT
```

## Required Test Workflow

When updating DataSanity or related code, follow this workflow:

### 1. Context Recap
- Problem being solved
- Affected files
- Dependencies

### 2. Implementation Plan
- 5–7 steps max
- Minimal diff
- No cascade rewrites

### 3. Modified Code Only
- Comment changes with `# NEW` or `# CHANGED`
- Preserve existing functionality

### 4. Validation Commands

```bash
# Core validation tests
pytest -m "data_sanity or property or integration" -q

# Performance tests with logging
pytest -m "perf or benchmark" -v > results/perf.log 2>&1

# Performance gate check
scripts/perf_gate.py --mode STRICT --log-file results/perf.log

# Full test suite
pytest tests/test_data_integrity.py tests/test_data_sanity_enforcement.py -v
```

### 5. Fallback Instructions
- How to revert to v1
- How to disable feature flag if issues appear

## Anti-Flake Measures

### Deterministic Testing
```bash
# Pin randomness
export PYTHONHASHSEED=0

# Hypothesis with limited examples
@settings(max_examples=50, verbosity=Verbosity.quiet)
```

### Dependency Management
- Freeze key deps (pandas, pyarrow) during migration windows
- Use golden datasets for deterministic testing
- Default network tests to **mocked** unless `RUN_NETWORK=1`

## Performance Gate Implementation

The `scripts/perf_gate.py` script:

1. **Parses** `results/perf.log` for performance test results
2. **Compares** against thresholds with mode-based tolerance
3. **Fails** if exceeded
4. **Reports** detailed failure information

### Usage Examples

```bash
# Basic usage (uses SANITY_PERF_MODE env var)
python scripts/perf_gate.py

# Explicit mode
python scripts/perf_gate.py --mode STRICT

# Custom log file
python scripts/perf_gate.py --log-file custom_perf.log

# Verbose output
python scripts/perf_gate.py --verbose
```

## Test Categories

### Core Validation Tests
- `@pytest.mark.data_sanity` - Core data sanity validation
- `@pytest.mark.validation` - Data validation logic
- `@pytest.mark.property` - Property-based testing

### Performance Tests
- `@pytest.mark.perf` - Performance tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.stress` - Stress testing

### Integration Tests
- `@pytest.mark.integration` - Integration verification
- `@pytest.mark.network` - Network resilience
- `@pytest.mark.flaky` - Flaky tests

### Edge Cases
- `@pytest.mark.edge_case` - Edge case handling
- `@pytest.mark.falsification` - Falsification scenarios

## Configuration

### Environment Variables

```bash
# Performance mode
export SANITY_PERF_MODE=RELAXED  # or STRICT

# Network testing
export RUN_NETWORK=1  # Enable real network tests

# Randomness
export PYTHONHASHSEED=0  # Deterministic testing
```

### Configuration Files

- `config/data_sanity.yaml` - DataSanity configuration
- `pytest.ini` - Pytest configuration with markers
- `scripts/perf_gate.py` - Performance gate script

## Migration Strategy

### Phase 1: Internal Standardization
- Variable and function naming
- Internal method standardization
- Documentation updates

### Phase 2: Configuration Standardization
- Configuration key standardization with aliases
- Backward compatibility preservation

### Phase 3: Public API Standardization
- Public API standardization with deprecation warnings
- Gradual migration path

### Phase 4: Cleanup
- Remove deprecated aliases (future version)
- Final cleanup

## When to Stop and Reassess

If a proposed change touches **>3 dependent modules**, stop and propose a **phased migration** instead of a single large patch.

## Troubleshooting

### Common Issues

1. **Performance Threshold Exceeded**
   - Check if change is necessary
   - Consider optimization
   - Use RELAXED mode for development

2. **Test Flakiness**
   - Set `PYTHONHASHSEED=0`
   - Use deterministic data
   - Mock network calls

3. **Memory Issues**
   - Check for memory leaks
   - Optimize data structures
   - Consider chunking large datasets

### Debugging Commands

```bash
# Run specific test categories
pytest -m "data_sanity" -v
pytest -m "perf" -v
pytest -m "property" -v

# Run with performance logging
pytest -m "perf or benchmark" -v > results/perf.log 2>&1

# Check performance gate
python scripts/perf_gate.py --verbose

# Run with deterministic settings
PYTHONHASHSEED=0 pytest tests/test_data_integrity.py -v
```

## Best Practices

1. **Always test performance impact**
2. **Use property-based testing for edge cases**
3. **Maintain backward compatibility**
4. **Document all changes**
5. **Use feature flags for risky changes**
6. **Monitor memory usage**
7. **Test with realistic data sizes**

## References

- [DataSanity Test Files](../tests/test_data_integrity.py)
- [DataSanity Enforcement Tests](../tests/test_data_sanity_enforcement.py)
- [Performance Gate Script](../scripts/perf_gate.py)
- [DataSanity Configuration](../config/data_sanity.yaml)
- [Pytest Configuration](../pytest.ini)
