# Changes

## Actions
- **pyproject.toml**: Added mutmut configuration for DataSanity mutation testing
- **Makefile**: Added mutation testing targets (mut, mut-results, mut-report, mut-full)
- **tests/util/corruptions.py**: Created data corruption utilities for metamorphic testing
- **tests/datasanity/test_mutations.py**: Implemented metamorphic tests for DataSanity validation
- **.github/workflows/ci.yml**: Added mutation-testing job (non-blocking)
- **docs/runbooks/mutation_testing.md**: Created comprehensive documentation

## Commands run
```bash
# Install mutmut
pip install mutmut coverage

# Test metamorphic tests
pytest -q tests/datasanity/test_mutations.py::test_nans_are_flagged -v
pytest -q tests/datasanity/test_mutations.py::test_duplicates_are_flagged tests/datasanity/test_mutations.py::test_negative_prices_flagged tests/datasanity/test_mutations.py::test_string_dtype_flagged -v

# Attempt mutmut run (had configuration issues)
mutmut run

# Commit changes
git add .
git commit -m "feat(testing): add mutation testing infrastructure for DataSanity validation"
```

## Key Features Implemented

### 1. Data Corruption Utilities
- `inject_nans()`: Add NaN values to OHLC columns
- `inject_lookahead()`: Create lookahead contamination by shifting columns
- `inject_duplicates()`: Duplicate timestamps
- `inject_negative_prices()`: Add negative price values
- `inject_string_dtype()`: Convert numeric columns to strings
- `inject_extreme_prices()`: Add extreme price values
- `inject_zero_volume()`: Add zero volume values

### 2. Metamorphic Tests
- Test that DataSanity correctly detects various data corruption scenarios
- Assert specific error messages and types
- Verify clean data passes validation
- Test profile differences (strict vs warn)

### 3. CI Integration
- Non-blocking mutation testing job
- Runs metamorphic tests and limited mutmut scope
- Reports results without blocking CI pipeline

### 4. Documentation
- Comprehensive guide for mutation testing approach
- Usage instructions for local testing
- Troubleshooting guide
- Future enhancement roadmap
