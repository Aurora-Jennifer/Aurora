# Harden Data Sanity Tests (use existing files)

## Context
Tests live in:
- `tests/test_data_integrity.py`
- `tests/test_data_sanity_enforcement.py`

Validator/wrapper code under `core/` (e.g., `core/data_sanity.py`, `core/contracts.py`, `core/utils.py`)

Config likely in `config/data_sanity.yaml`

Test runner is pytest

## Task
Enhance the existing test files (no new test files) to add the coverage below. Keep current names, expand them with new parametrized tests and fixtures. Add marks in place; don't split files.

## What to implement (in-place)

### A) Pytest marks to separate flaky/network/perf

Create/extend `pytest.ini`:
```ini
[tool:pytest]
markers =
    flaky: marks tests as flaky (may fail intermittently)
    network: marks tests as requiring network access
    perf: marks tests as performance tests
    slow: marks tests as slow running
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    data_sanity: marks tests as data sanity validation tests
    property: marks tests as property-based tests
    falsification: marks tests as falsification scenarios
    edge_case: marks tests as edge case handling
    contract: marks tests as contract validation
    guard: marks tests as guard enforcement
```

### B) Parametrized test fixtures

Add fixtures for:
- Different data corruption scenarios
- Various config profiles (strict, lenient, custom)
- Different data sizes (small, medium, large)
- Different time periods
- Different symbols/instruments

### C) Property-based testing expansion

Expand hypothesis-based tests for:
- Price boundary conditions
- Volume edge cases
- OHLC relationship violations
- Time series anomalies
- Returns calculation edge cases

### D) Performance benchmarks

Add performance tests that:
- Measure validation overhead
- Test with large datasets
- Benchmark different repair modes
- Test memory usage patterns

### E) Network resilience tests

Add tests for:
- API rate limiting handling
- Network timeout scenarios
- Data source failures
- Partial data corruption

### F) Integration verification

Verify DataSanity is integrated in:
- Backtest engine
- Paper trading engine
- Data providers
- Feature engineering
- Strategy execution

### G) Falsification scenarios

Add tests that should always fail:
- Extreme data corruption
- Impossible OHLC relationships
- Future data contamination
- Invalid data types

### H) Edge case hardening

Test:
- Empty datasets
- Single-row datasets
- Mixed timezones
- MultiIndex columns
- Corporate actions
- Market holidays

### I) Contract enforcement

Test decorators and contracts:
- `@require_validated_data`
- `@require_validated_signals`
- DataFrame contracts
- Signal contracts
- Feature contracts

### J) Guard mechanisms

Test DataSanityGuard:
- Attachment verification
- Status tracking
- Enforcement blocking
- Cleanup handling

## Implementation Notes

1. **Keep existing structure**: Don't create new test files, enhance existing ones
2. **Use parametrization**: Leverage `@pytest.mark.parametrize` for comprehensive coverage
3. **Add fixtures**: Create reusable test data and config fixtures
4. **Mark appropriately**: Use pytest marks to categorize tests
5. **Maintain backward compatibility**: Don't break existing test functionality
6. **Add comprehensive assertions**: Ensure all edge cases are properly tested
7. **Document test scenarios**: Add clear docstrings explaining test purposes
8. **Handle flaky tests**: Use appropriate retry mechanisms for network-dependent tests
9. **Performance thresholds**: Set reasonable performance expectations
10. **Error message validation**: Test that error messages are informative and actionable
