# Latest Context Error Fix - Next Steps Document

**Date:** 2024-12-19 15:30  
**Status:** Major Progress - 69/89 sanity tests fixed  
**Context:** Applied focused fix pack to resolve priority-driven validation issues

---

## 🎯 **What Was Accomplished**

### **Major Test Suite Improvements**
- **Before:** 89 failed sanity tests, 89 failed full tests
- **After:** 20 failed sanity tests, 57 failed full tests
- **Progress:** 69 sanity tests fixed, 32 full tests fixed

### **Core Issues Resolved**

#### 1. **Priority-Driven Validation Order** ✅
- **Problem:** OHLC invariants were failing before non-finite value checks
- **Fix:** Reordered validation pipeline to check fundamental issues first:
  1. Datetime canonicalization (timezone, monotonic, duplicates)
  2. Numeric coercion & non-finite checks
  3. OHLC invariants
  4. Volume validation
  5. Outlier detection (lenient only)
  6. Returns calculation
  7. Final validation

#### 2. **Strict vs Lenient Behavior** ✅
- **Problem:** `getattr(profile, "allow_repairs", True)` was always returning `True` for dict profiles
- **Fix:** Changed to `profile.get("allow_repairs", True)` for proper dict access
- **Result:** Strict mode now properly raises errors instead of doing repairs

#### 3. **Standardized Error Infrastructure** ✅
- **Created:** `core/data_sanity/codes.py` with consistent error codes
- **Created:** `core/data_sanity/errors.py` with `DataSanityError` and `estring()`
- **Fixed:** Exception import consistency across all modules

#### 4. **Timezone Handling** ✅
- **Problem:** `TypeError: data is already tz-aware US/Eastern, unable to set specified tz: UTC`
- **Fix:** Proper `tz_convert()` vs `tz_localize()` logic
- **Result:** Strict mode raises specific timezone errors, lenient mode handles conversion

#### 5. **MultiIndex Alignment** ✅
- **Problem:** "Operands are not aligned" errors in OHLC checks
- **Fix:** Use aligned Series operations with `pd.concat()` and `.align()`
- **Result:** No more alignment errors in invariant checks

---

## 📋 **Current Status**

### **✅ Working Tests (109 sanity tests)**
- Non-finite value detection (`nan_burst_fail`, `nan_scattered_fail`)
- Timezone validation (`naive_timezone_fail`, `wrong_timezone_fail`)
- Index monotonicity (`non_monotonic_fail`)
- Basic OHLC invariants
- Volume validation
- Most edge cases

### **❌ Remaining Issues (20 sanity tests)**
- **Lookahead detection** (not implemented in fix pack)
- **Future data detection** (not implemented in fix pack)
- **String/dtype validation** (edge cases)
- **Mixed data types** (complex scenarios)
- **Missing columns** (some edge cases)

---

## 🔧 **Technical Architecture**

### **New Modular Structure**
```
core/data_sanity/
├── codes.py          # Standardized error codes
├── errors.py         # DataSanityError + estring helper
├── datetime.py       # Timezone canonicalization
├── clean.py          # Numeric coercion & non-finite checks
├── invariants.py     # OHLC invariant validation
├── columnmap.py      # MultiIndex column mapping
├── group.py          # Groupwise time ordering
├── api.py            # Public API surface
├── main.py           # Core validator logic
└── __init__.py       # Package exports
```

### **Validation Pipeline Order**
1. **Datetime canonicalization** (strict raises: tz, monotonic, duplicates)
2. **Numeric coercion** (strict raises: non-numeric, non-finite)
3. **OHLC invariants** (strict raises: negative, inconsistent)
4. **Volume validation** (strict raises: negative, zero)
5. **Outlier detection** (lenient only)
6. **Returns calculation** (strict bounds)
7. **Final validation** (lookahead, future data)

---

## 🚀 **Next Steps (When Ready to Continue)**

### **Phase 1: Complete Missing Modules**
1. **Lookahead Detection**
   ```python
   # core/data_sanity/lookahead.py
   def detect_lookahead(df: pd.DataFrame, feature_cols=None) -> list[str]:
       # Implement lookahead contamination detection
   ```

2. **Future Data Detection**
   ```python
   # core/data_sanity/future.py
   def assert_no_future_timestamps(df: pd.DataFrame, profile, now=None):
       # Implement future timestamp detection
   ```

3. **Validator Integration**
   ```python
   # core/data_sanity/validator.py
   class DataSanityValidator:
       def apply(self, df):
           # Integrate all modules in priority order
   ```

### **Phase 2: Edge Case Handling**
1. **String/Dtype Validation**
   - Handle non-numeric price columns
   - Mixed data type scenarios
   - Object dtype handling

2. **Missing Columns**
   - Complex MultiIndex scenarios
   - Synonym mapping edge cases
   - Column synthesis logic

3. **Error Message Refinement**
   - Match exact test expectations
   - Improve error detail messages
   - Add context information

### **Phase 3: Integration & Testing**
1. **Update Main Validator**
   - Replace old pipeline with new modular approach
   - Maintain backward compatibility
   - Update repair tracking

2. **Comprehensive Testing**
   - Run full test suite
   - Fix remaining edge cases
   - Performance validation

3. **Documentation**
   - Update API documentation
   - Add migration guide
   - Update examples

---

## 🔍 **Key Files Modified**

### **Core Infrastructure**
- `core/data_sanity/codes.py` - **NEW** - Error codes
- `core/data_sanity/errors.py` - **NEW** - Exception classes
- `core/data_sanity/main.py` - **MODIFIED** - Import fixes, pipeline order

### **Validation Modules**
- `core/data_sanity/datetime.py` - **MODIFIED** - Strict vs lenient timezone handling
- `core/data_sanity/clean.py` - **MODIFIED** - Priority-driven numeric validation
- `core/data_sanity/invariants.py` - **NEW** - Aligned OHLC invariant checks
- `core/data_sanity/columnmap.py` - **EXISTING** - MultiIndex support
- `core/data_sanity/group.py` - **EXISTING** - Groupwise ordering

### **API Surface**
- `core/data_sanity/api.py` - **EXISTING** - Public exports
- `core/data_sanity/__init__.py` - **EXISTING** - Package structure

---

## 🎯 **Success Metrics**

### **Achieved**
- ✅ 69/89 sanity tests fixed (77% success rate)
- ✅ Priority-driven validation order implemented
- ✅ Strict vs lenient behavior working correctly
- ✅ Standardized error infrastructure in place
- ✅ Timezone handling resolved
- ✅ MultiIndex alignment fixed

### **Target**
- 🎯 89/89 sanity tests passing (100% success rate)
- 🎯 All edge cases handled
- 🎯 Performance maintained or improved
- 🎯 Backward compatibility preserved

---

## 💡 **Key Insights**

1. **Priority Order Matters**: Fundamental issues (dtype, finite, timezone) must be checked before complex validations (OHLC, lookahead)

2. **Profile Handling**: Dict-based profiles need `.get()` not `getattr()` for proper access

3. **Exception Consistency**: All modules must import `DataSanityError` from the same source

4. **Alignment Issues**: MultiIndex operations require explicit alignment to avoid "Operands not aligned" errors

5. **Modular Design**: Breaking validation into focused modules makes debugging and maintenance much easier

---

**Next Session Focus:** Complete the missing lookahead and future data detection modules, then tackle the remaining edge cases.
