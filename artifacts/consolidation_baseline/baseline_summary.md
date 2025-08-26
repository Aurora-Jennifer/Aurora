# Consolidation Baseline Report

**Date**: 2025-01-27 19:32
**Repository**: /home/Jennifer/secure/trader

## Code Size Metrics

### Overall Repository
- **Total Python files**: 544
- **Total lines of code**: 186,282 (Python only)
- **Total repository size**: 329,811 lines across all languages

### Duplication Analysis

#### Logging Setup Duplication
- **Logging hits found**: 189 lines across all Python files
- **Files with logging setup**: ~30+ files identified
- **Patterns found**:
  - `logging.basicConfig()` - Most common
  - `logging.getLogger()` - Common
  - `StreamHandler()`, `FileHandler()`, `Formatter()` - Scattered

#### Data Validation Duplication
- **Validator hits found**: 11,154 lines (very high!)
- **Patterns found**:
  - `DataSanity` references - Extensive
  - Validation functions scattered across multiple files
  - Duplicate validation logic in `fetch_yfinance.py`, `data_sanity.py`, etc.

## Test Status

### Current Test Issues
- **6 collection errors** (mostly import issues)
- **127 warnings** (mostly pytest mark warnings)
- **Main issues**:
  - Missing `_last`, `_safe_len` functions in `core.utils`
  - Duplicate pytest option `--datasanity-profile`
  - Missing `build_demo_features` function

### Test Coverage
- **Tests run**: Partial (errors during collection)
- **Need to fix**: Import issues before proceeding with consolidation

## Consolidation Opportunities

### Phase 1: Logging Consolidation (SAFE)
- **Target**: 189 logging setup lines
- **Expected reduction**: ~150-200 lines
- **Risk**: Very low (no functional changes)

### Phase 2: Data Validation Consolidation (HIGH IMPACT)
- **Target**: 11,154 validator hits
- **Expected reduction**: ~300-500 lines
- **Risk**: Medium (need to ensure validation logic preserved)

### Phase 3: Training Consolidation (HIGH IMPACT)
- **Target**: 4 large training scripts
- **Expected reduction**: ~500-1000 lines
- **Risk**: Medium (need thorough testing)

### Phase 4: Walkforward Consolidation (HIGH IMPACT)
- **Target**: Multiple walkforward implementations
- **Expected reduction**: ~400-800 lines
- **Risk**: Medium (need thorough testing)

## Next Steps

1. **Fix test collection issues** before proceeding
2. **Start with Phase 1** (logging consolidation) - safest
3. **Establish CI gates** to prevent regression
4. **Measure impact** after each phase

## Baseline Files Created

- `cloc.txt` - Code size metrics
- `py_file_count.txt` - Python file count
- `logging_hits.txt` - Logging setup locations
- `validator_hits.txt` - Data validation locations
- `pytest.txt` - Test status (with errors)

## Acceptance Criteria for Phase 1

- [ ] `logging_hits.txt` line count = 0 (or close to 0)
- [ ] Tests pass (fix collection issues first)
- [ ] No functional regression
- [ ] ~150-200 lines of code removed
- [ ] CI gate prevents logging setup regression
