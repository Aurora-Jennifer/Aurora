# Fix Next List - Remaining DataSanity Issues

**Date:** 2024-12-19 15:30  
**Status:** 20 sanity tests remaining, 57 full tests remaining  
**Priority:** High - Core functionality working, edge cases need attention

---

## 🚨 **Critical Fixes (High Priority)**

### 1. **Lookahead Detection Module**
**Error:** `lookahead_returns_fail` - Missing lookahead contamination detection
**Files:** `core/data_sanity/lookahead.py` (create)
**Issue:** Returns column being flagged as lookahead when it shouldn't be
**Fix:** Implement proper lookahead detection with allowed columns list

```python
# core/data_sanity/lookahead.py
_ALLOWED = frozenset({"Returns","Label","Target","y"})

def detect_lookahead(df: pd.DataFrame, feature_cols=None) -> list[str]:
    # Implement lookahead contamination detection
    # Skip allowed columns like "Returns"
    # Check for future-referencing patterns
```

### 2. **Future Data Detection Module**
**Error:** `future_data_fail` - Missing future timestamp detection
**Files:** `core/data_sanity/future.py` (create)
**Issue:** Data with future timestamps not being caught
**Fix:** Implement future timestamp validation

```python
# core/data_sanity/future.py
def assert_no_future_timestamps(df: pd.DataFrame, profile, now=None):
    # Check if any timestamps are beyond current time
    # Strict mode: raise error
    # Lenient mode: drop future rows
```

### 3. **String/Dtype Validation Edge Cases**
**Error:** `string_prices_fail`, `mixed_dtypes_fail`
**Files:** `core/data_sanity/clean.py` (modify)
**Issue:** Non-numeric price columns not being handled properly
**Fix:** Improve dtype coercion logic for edge cases

---

## ⚠️ **Medium Priority Fixes**

### 4. **Missing Columns Edge Cases**
**Error:** Some missing column scenarios still failing
**Files:** `core/data_sanity/clean.py`, `core/data_sanity/columnmap.py`
**Issue:** Complex MultiIndex scenarios not handled
**Fix:** Enhance column mapping for edge cases

### 5. **Error Message Refinement**
**Error:** Some tests expect specific error message formats
**Files:** `core/data_sanity/codes.py`, `core/data_sanity/errors.py`
**Issue:** Error messages don't match test expectations exactly
**Fix:** Align error messages with test requirements

### 6. **Validator Integration**
**Error:** Main validator still using old pipeline in some places
**Files:** `core/data_sanity/main.py`
**Issue:** New modular approach not fully integrated
**Fix:** Complete integration of new validation pipeline

---

## 🔧 **Light Priority Fixes**

### 7. **Performance Optimization**
**Issue:** Some validation steps could be optimized
**Files:** Various validation modules
**Fix:** Optimize validation pipeline performance

### 8. **Logging Improvements**
**Issue:** Error logging could be more informative
**Files:** All validation modules
**Fix:** Add better error context and logging

### 9. **Test Coverage**
**Issue:** Some edge cases not covered by tests
**Files:** `tests/sanity/test_cases.py`
**Fix:** Add missing test cases

---

## 📊 **Error Log Summary**

### **Sanity Test Failures (20 remaining)**
```
❌ lookahead_returns_fail - Lookahead detection missing
❌ future_data_fail - Future timestamp detection missing  
❌ string_prices_fail - String dtype handling
❌ mixed_dtypes_fail - Mixed dtype scenarios
❌ missing_columns_edge - Complex MultiIndex cases
❌ error_message_mismatch - Message format issues
❌ [15 more edge cases...]
```

### **Full Test Suite Failures (57 remaining)**
```
❌ test_strict_profile.py - Lookahead contamination
❌ test_data_sanity_integration.py - Error message format
❌ test_repro_and_parallel.py - Seed reproducibility
❌ [54 more integration tests...]
```

---

## 🎯 **Quick Win Fixes (Start Here)**

### **1. Create Lookahead Module (30 min)**
```bash
# Create the missing module
touch core/data_sanity/lookahead.py
# Implement basic lookahead detection
# Test: pytest "tests/sanity/test_cases.py::test_case[lookahead_returns_fail]"
```

### **2. Create Future Data Module (20 min)**
```bash
# Create the missing module  
touch core/data_sanity/future.py
# Implement future timestamp detection
# Test: pytest "tests/sanity/test_cases.py::test_case[future_data_fail]"
```

### **3. Fix String Dtype Handling (15 min)**
```bash
# Modify clean.py to handle string prices better
# Test: pytest "tests/sanity/test_cases.py::test_case[string_prices_fail]"
```

---

## 🔍 **Debugging Commands**

### **Check Current Status**
```bash
# Run sanity tests only
pytest tests/sanity/test_cases.py -v --tb=short

# Check specific failing test
pytest "tests/sanity/test_cases.py::test_case[lookahead_returns_fail]" -v -s

# Run full suite for overall status
pytest -v --tb=no | tail -5
```

### **Debug Specific Issues**
```bash
# Test lookahead detection manually
python -c "from core.data_sanity import DataSanityValidator; from tests.factories import build_case; df = build_case('lookahead_returns'); v = DataSanityValidator(profile='strict'); v.validate_and_repair(df, 'test')"

# Test future data detection manually
python -c "from core.data_sanity import DataSanityValidator; from tests.factories import build_case; df = build_case('future_data'); v = DataSanityValidator(profile='strict'); v.validate_and_repair(df, 'test')"
```

---

## 📝 **Implementation Notes**

### **Lookahead Detection Logic**
- Skip allowed columns: `Returns`, `Label`, `Target`, `y`
- Check for future-referencing patterns in column names
- Check for high correlation with shifted data
- Use regex patterns: `(t\+1|tplus1|lead|future|next|_tp1|_tplus1)`

### **Future Data Detection Logic**
- Compare timestamps against current time
- Handle both single-level and MultiIndex timestamps
- Strict mode: raise error with count of future rows
- Lenient mode: drop future rows and continue

### **String Dtype Handling**
- Try `pd.to_numeric()` with `errors="coerce"`
- Check for non-coercible values in strict mode
- Provide helpful error messages for string data

---

## 🎯 **Success Criteria**

### **Phase 1 Complete When:**
- ✅ `lookahead_returns_fail` passes
- ✅ `future_data_fail` passes  
- ✅ `string_prices_fail` passes
- ✅ Sanity test count: 20 → 10 or fewer

### **Phase 2 Complete When:**
- ✅ All sanity tests pass (20 → 0)
- ✅ Full test count: 57 → 20 or fewer
- ✅ Core functionality fully working

### **Phase 3 Complete When:**
- ✅ All tests pass
- ✅ Performance maintained
- ✅ Documentation updated

---

**Next Session:** Start with the Quick Win Fixes (lookahead + future data modules), then tackle the remaining edge cases systematically.
