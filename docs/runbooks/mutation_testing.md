# Mutation Testing for DataSanity

## Overview

This document describes the mutation testing approach used to verify that DataSanity validation correctly detects data quality issues.

## Components

### 1. Metamorphic Tests (`tests/datasanity/test_mutations.py`)

These tests verify that DataSanity correctly detects various data corruption scenarios by:

- **Injecting specific corruptions**: NaN values, lookahead contamination, duplicate timestamps, etc.
- **Asserting expected failures**: Each test expects a specific `DataSanityError` with relevant error message
- **Testing clean data**: Verifying that valid data passes validation

### 2. Data Corruption Utilities (`tests/util/corruptions.py`)

Helper functions to inject specific data quality issues:

- `inject_nans()`: Add NaN values to OHLC columns
- `inject_lookahead()`: Create lookahead contamination by shifting columns
- `inject_duplicates()`: Duplicate timestamps
- `inject_negative_prices()`: Add negative price values
- `inject_string_dtype()`: Convert numeric columns to strings

### 3. Mutation Testing with mutmut

Automated mutation testing of the DataSanity codebase:

- **Scope**: `core/data_sanity/` modules
- **Runner**: pytest with DataSanity tests
- **Goal**: Ensure tests catch logic changes in validation code

## Usage

### Local Testing

```bash
# Run metamorphic tests
pytest tests/datasanity/test_mutations.py -v

# Run mutation testing (limited scope)
mutmut run --paths-to-mutate core/data_sanity/clean.py --max-children 5

# View mutation results
mutmut results
```

### CI Integration

The mutation testing job in CI:

1. Runs metamorphic tests to verify corruption detection
2. Runs limited mutation tests on DataSanity code
3. Reports results (non-blocking)

## Test Categories

### Metamorphic Tests

- **Data Corruption**: NaN, Inf, negative prices, extreme values
- **Structural Issues**: Duplicate timestamps, non-monotonic index
- **Type Issues**: String dtypes in numeric columns
- **Lookahead**: Future data contamination
- **Timezone**: Naive timestamps

### Mutation Tests

- **Code Logic**: Operators, conditionals, function calls
- **Error Handling**: Exception paths, validation logic
- **Data Processing**: DataFrame operations, type conversions

## Adding New Tests

### New Metamorphic Test

1. Add corruption function to `tests/util/corruptions.py`
2. Create test in `tests/datasanity/test_mutations.py`
3. Assert specific error message/type
4. Mark with `@pytest.mark.mutation`

### New Mutation Test

1. Add test to appropriate DataSanity test file
2. Ensure test covers specific code paths
3. Run `mutmut run` to verify coverage

## Success Criteria

- **Metamorphic tests**: All corruption scenarios detected with correct error messages
- **Mutation tests**: High kill rate (>90%) for DataSanity code mutations
- **CI**: Non-blocking but informative results

## Troubleshooting

### Common Issues

1. **Lookahead detection too sensitive**: Adjust synthetic data generation
2. **NaN values from shifts**: Use `fillna()` in corruption functions
3. **Timezone issues**: Ensure test data is properly timezone-aware

### Debugging

```bash
# Run specific test with verbose output
pytest tests/datasanity/test_mutations.py::test_nans_are_flagged -v -s

# Check DataSanity validation manually
python -c "
import pandas as pd
from core.data_sanity import DataSanityValidator
df = pd.DataFrame({'Close': [1, 2, 3]})
validator = DataSanityValidator(profile='strict')
result = validator.validate_and_repair(df, 'TEST')
print(result)
"
```

## Future Enhancements

1. **Property-based testing**: Use Hypothesis for more comprehensive data generation
2. **Performance testing**: Measure validation speed with corrupted data
3. **Integration testing**: Test DataSanity with real market data scenarios
4. **Fuzzing**: Random data corruption for edge case discovery
