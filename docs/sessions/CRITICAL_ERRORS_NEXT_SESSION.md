# Critical Errors Requiring Immediate Fix - Next Session
**Priority**: 🔴 **URGENT** - Must fix before production deployment
**Based on**: 4-hour refactoring session analysis
**Risk Level**: High - Potential production failures

## 🚨 **CRITICAL ERRORS - IMMEDIATE ACTION REQUIRED**

### 1. **Memory Leak in Composer Integration**
**Severity**: 🔴 **CRITICAL**
**Location**: `core/engine/composer_integration.py`
**Issue**: Composer objects may not be properly cleaned up between folds

```python
# PROBLEM: Composer objects accumulate in memory
def get_composer_decision(self, data, symbol, current_idx, asset_class=None):
    # Creates new composer instance each time
    temp_composer = copy.deepcopy(self.composer)  # Memory leak!
    # ... rest of function
```

**Fix Required**:
```python
# SOLUTION: Reuse composer instance or implement proper cleanup
def get_composer_decision(self, data, symbol, current_idx, asset_class=None):
    # Use existing composer instance
    result = self.composer.compose(market_state, self.strategies, self.regime_extractor)
    # ... rest of function
```

**Impact**: Memory exhaustion in long-running backtests
**Testing**: Run 5-year backtest and monitor memory usage

### 2. **Configuration File Loading Failures**
**Severity**: 🔴 **CRITICAL**
**Location**: `core/config.py`, `config/data_sanity.yaml`
**Issue**: Configuration files may not load correctly in production

```python
# PROBLEM: Missing error handling for config file loading
def load_config(config_paths: List[Union[str, Path]]) -> Dict[str, Any]:
    # No validation of file existence or YAML syntax
    # No fallback for missing files
```

**Fix Required**:
```python
# SOLUTION: Add comprehensive error handling
def load_config(config_paths: List[Union[str, Path]]) -> Dict[str, Any]:
    result = {}
    for path in config_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                result = deep_merge(result, config)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")
    return result
```

**Impact**: System startup failures in production
**Testing**: Test with missing/invalid config files

### 3. **Timezone Handling Edge Cases**
**Severity**: 🟡 **HIGH**
**Location**: `core/data_sanity.py`
**Issue**: Some data sources may have different timezone formats

```python
# PROBLEM: Assumes all data has UTC timezone
if data.index.tz != timezone.utc:
    # May fail with mixed timezone data
    data.index = data.index.tz_convert(timezone.utc)
```

**Fix Required**:
```python
# SOLUTION: Robust timezone handling
def _validate_time_series_strict(self, data: pd.DataFrame, symbol: str):
    # Handle mixed timezone data
    if data.index.tz is None:
        data.index = data.index.tz_localize(timezone.utc)
    elif data.index.tz != timezone.utc:
        try:
            data.index = data.index.tz_convert(timezone.utc)
        except Exception as e:
            # Handle mixed timezone data
            data.index = data.index.tz_localize(timezone.utc)
```

**Impact**: Data validation failures with certain data sources
**Testing**: Test with various timezone formats

### 4. **Non-Deterministic Test Behavior**
**Severity**: 🟡 **HIGH**
**Location**: Multiple test files
**Issue**: Some tests may have non-deterministic behavior

```python
# PROBLEM: Random seeds not properly set in all tests
def create_valid_ohlc_data(n_periods=100, base_price=100.0):
    np.random.seed(42)  # Only set in some functions
    # ... rest of function
```

**Fix Required**:
```python
# SOLUTION: Consistent random seed management
@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)
    random.seed(42)
    yield
```

**Impact**: Flaky tests in CI/CD pipeline
**Testing**: Run tests multiple times to ensure consistency

### 5. **Error Message Inconsistencies**
**Severity**: 🟡 **HIGH**
**Location**: `core/data_sanity.py`
**Issue**: Error messages may not match test expectations

```python
# PROBLEM: Error messages may change based on validation order
if self._detect_lookahead_contamination(clean_data):
    raise DataSanityError(f"{symbol}: Lookahead contamination detected")
# But OHLC validation might fail first
```

**Fix Required**:
```python
# SOLUTION: Consistent error message ordering
def validate_and_repair(self, data: pd.DataFrame, symbol: str = "UNKNOWN"):
    # Validate in consistent order
    # 1. Basic data validation
    # 2. OHLC validation
    # 3. Lookahead detection
    # 4. Final checks
```

**Impact**: Test failures due to changing error messages
**Testing**: Ensure error messages are consistent across runs

## 🔧 **HIGH PRIORITY FIXES**

### 6. **Missing Type Hints**
**Severity**: 🟡 **HIGH**
**Location**: New functions in refactored files
**Issue**: Some new functions lack proper type hints

**Fix Required**: Add comprehensive type hints to all new functions

### 7. **Incomplete Error Handling**
**Severity**: 🟡 **HIGH**
**Location**: `core/engine/composer_integration.py`
**Issue**: Some edge cases not handled

**Fix Required**: Add error handling for network failures, invalid configs, memory issues

### 8. **Performance Bottlenecks**
**Severity**: 🟡 **HIGH**
**Location**: Composer integration, data validation
**Issue**: Potential performance issues with large datasets

**Fix Required**: Profile and optimize slow functions

## 📋 **FIX PRIORITY MATRIX**

| Error | Severity | Impact | Effort | Priority |
|-------|----------|--------|--------|----------|
| Memory Leak | 🔴 Critical | High | Medium | 1 |
| Config Loading | 🔴 Critical | High | Low | 2 |
| Timezone Handling | 🟡 High | Medium | Medium | 3 |
| Test Determinism | 🟡 High | Medium | Low | 4 |
| Error Messages | 🟡 High | Low | Low | 5 |
| Type Hints | 🟡 High | Low | Medium | 6 |
| Error Handling | 🟡 High | Medium | High | 7 |
| Performance | 🟡 High | Medium | High | 8 |

## 🧪 **TESTING STRATEGY FOR CRITICAL FIXES**

### Memory Leak Testing
```bash
# Test memory usage over time
python -c "
import psutil
import time
from core.engine.composer_integration import ComposerIntegration
from core.config import load_config

process = psutil.Process()
initial_memory = process.memory_info().rss

config = load_config(['config/base.yaml'])
composer = ComposerIntegration(config)

for i in range(100):
    # Simulate composer calls
    composer.get_composer_decision(data, 'TEST', i)
    if i % 10 == 0:
        current_memory = process.memory_info().rss
        print(f'Iteration {i}: {current_memory - initial_memory} bytes')
"
```

### Configuration Loading Testing
```bash
# Test all configuration combinations
python -c "
from core.config import load_config
import glob

config_files = glob.glob('config/*.yaml')
for config_file in config_files:
    try:
        cfg = load_config([config_file])
        print(f'✓ {config_file} loaded successfully')
    except Exception as e:
        print(f'✗ {config_file} failed: {e}')
"
```

### Timezone Testing
```bash
# Test with various timezone formats
python -c "
import pandas as pd
from core.data_sanity import DataSanityValidator

# Test naive timezone
dates_naive = pd.date_range('2020-01-01', periods=100, freq='D')
# Test UTC timezone
dates_utc = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
# Test mixed timezone
dates_mixed = pd.date_range('2020-01-01', periods=100, freq='D', tz='America/New_York')

validator = DataSanityValidator(profile='strict')
for dates, name in [(dates_naive, 'naive'), (dates_utc, 'utc'), (dates_mixed, 'mixed')]:
    try:
        data = create_test_data(dates)
        validator.validate_and_repair(data, f'TEST_{name}')
        print(f'✓ {name} timezone handled correctly')
    except Exception as e:
        print(f'✗ {name} timezone failed: {e}')
"
```

## 🚀 **DEPLOYMENT CHECKLIST**

### Pre-Deployment (Must Complete)
- [ ] Fix memory leak in composer integration
- [ ] Validate all configuration file loading
- [ ] Test timezone handling with various formats
- [ ] Ensure test determinism
- [ ] Verify error message consistency

### Deployment Day
- [ ] Run full test suite (expect 95%+ success rate)
- [ ] Test with production-like data volumes
- [ ] Monitor memory usage during execution
- [ ] Validate all configuration combinations
- [ ] Test error handling with edge cases

### Post-Deployment
- [ ] Monitor system performance
- [ ] Check error logs for any issues
- [ ] Validate data processing accuracy
- [ ] Confirm memory usage is stable

## 📊 **SUCCESS METRICS**

### Critical Fixes Success Criteria
- [ ] Memory usage remains stable during 5-year backtest
- [ ] All configuration files load without errors
- [ ] Timezone conversion handles all common formats
- [ ] Tests run deterministically (same results each time)
- [ ] Error messages are consistent and helpful

### Performance Targets
- [ ] Memory usage <2GB for large datasets
- [ ] Execution time <30min for 5-year backtest
- [ ] No memory leaks detected
- [ ] Configuration loading <5 seconds

---

**Next Session Priority**: 🔴 **CRITICAL FIXES FIRST**
**Estimated Fix Time**: 2-3 hours for critical issues
**Risk Mitigation**: Comprehensive testing after each fix
