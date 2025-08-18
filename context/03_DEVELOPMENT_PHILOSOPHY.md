# 🎯 Development Philosophy & Coding Standards

## **Core Principles**

### **1. No Hardcoded Runtime Knobs**
- **Rule**: Anything a user might hot-swap must come from config, not source code
- **Examples**: Risk profile, tickers, warmup bars, fold sizes, blender temps, data flags
- **Implementation**: All magic numbers moved to `config/base.yaml` and overlays
- **Validation**: Grep for TODO: hardcoded / magic numbers

### **2. Safety First - Never Index Empty Arrays**
- **Rule**: Before any `[-1]`, `.iloc[-1]`, or tail slice, check `len > 0`
- **Fallback**: Return a **HOLD** decision with a reason if empty
- **Implementation**: Use `_safe_len()` and `_last()` helper functions
- **Example**:
```python
def _safe_len(x) -> int:
    return 0 if x is None else (len(x) if hasattr(x, "__len__") else 0)

def _last(x):
    if x is None or _safe_len(x) == 0:
        return None
    if hasattr(x, "iloc"):
        return x.iloc[-1]
    return x[-1]
```

### **3. Warmup Discipline**
- **Rule**: Enforce `min_history_bars` before any feature use or decisions
- **Implementation**: Drop the first N rows after feature build
- **Code Pattern**:
```python
min_bars = cfg["composer"]["min_history_bars"]
if _safe_len(features) < min_bars:
    return {"action": "HOLD", "reason": "insufficient_history"}
```

### **4. One Log Per Cause Per Fold**
- **Rule**: No per-bar error spam. Summarize counts by cause
- **Implementation**: Use fold-level warnings with counts
- **Examples**: `insufficient_history`, `empty_strategy_output`, `nan_in_scores`
- **Debug Level**: Per-bar anomalies → `DEBUG` only

### **5. yfinance Behavior is Explicit**
- **Rule**: Configure `auto_adjust` via config; never rely on defaults
- **Implementation**: Always pass `auto_adjust = cfg["data"]["auto_adjust"]`
- **Documentation**: Log this in run header (once)

### **6. Idempotent, Minimal Diffs**
- **Rule**: Only touch files you must. Preserve public APIs unless change is justified and tested
- **Implementation**: Surgical changes with extensive context
- **Testing**: Verify changes don't break existing functionality

## **Code Quality Standards**

### **Type Safety**
- **Requirement**: Exhaustive type hints; `from __future__ import annotations`
- **Implementation**: All public functions properly typed
- **Exceptions**: Only typed exceptions from `core/errors.py`. No bare `except:`

### **Determinism**
- **Requirement**: Respect `rng_seed` where randomness exists
- **Implementation**: Consistent random seed management across tests
- **Validation**: Tests run deterministically (same results each time)

### **Pure Functions**
- **Requirement**: Composer blenders must be pure functions: inputs → weights, no side effects
- **Implementation**: No global state modifications in composer functions
- **Testing**: Verify function outputs are consistent for same inputs

## **Configuration Management**

### **Required Config Structure**
```yaml
engine:
  min_history_bars: 120
  max_na_fraction: 0.05
  rng_seed: 42
walkforward:
  fold_length: 252
  step_size: 63
  allow_truncated_final_fold: false
data:
  source: yfinance
  auto_adjust: false
  cache: true
risk:
  pos_size_method: vol_target
  vol_target: 0.15
  max_drawdown: 0.20
  daily_loss_limit: 0.03
composer:
  use_composer: true
  regime_extractor: basic_kpis
  blender: softmax_blender
  min_history_bars: 120
  hold_on_nan: true
  params:
    temperature: 1.0
    trend_bias: 1.2
    chop_bias: 1.1
    min_confidence: 0.10
tickers:
  - SPY
```

### **Configuration Loading**
- **Implementation**: Deep-merge base + overlays
- **API**: `cfg = load_config([Path("config/base.yaml"), *overlays])`
- **Helper**: `get_cfg(path, default=None) -> Any`

## **Error Handling Patterns**

### **Composer Integration Safety**
```python
# Before reading strategy outputs or features
min_bars = cfg["composer"]["min_history_bars"]
if _safe_len(features) < min_bars:
    return {"action": "HOLD", "reason": "insufficient_history"}
if not outputs or any(_safe_len(o) == 0 for o in outputs):
    return {"action": "HOLD", "reason": "empty_strategy_output"}
last_scores = [_last(o) for o in outputs]
if any(s is None or (isinstance(s, float) and math.isnan(s)) for s in last_scores):
    return {"action": "HOLD", "reason": "nan_in_scores"}
```

### **Fold Generator Contract**
```python
# When building folds
if test_len < step_size:
    if allow_truncated_final_fold: true:
        step_size = test_len  # and proceed
    else:
        logger.warning("Skipping final fold: test window too small (%d < %d)", test_len, step_size)
        # skip it
```

### **Feature Pipeline Rules**
```python
# After feature construction
features = features.dropna().iloc[cfg["engine"]["min_history_bars"]:]
# If empty after warmup: return structured empty result; log one fold-level summary
```

## **Testing Standards**

### **Required Test Coverage**
- Empty features → HOLD (no exception)
- Short tail fold handling (skip vs truncate)
- `min_history_bars` enforcement
- Config overlay precedence (base < overlay)
- yfinance `auto_adjust` propagation (mock inspected)

### **Test Execution**
```bash
# Run tests with proper output
pytest -q
# Enforce code quality
ruff + black in pre-commit
```

## **Logging Policy**

### **Structured Logging**
- **Fold Level**: One warning summarizing counts by cause
- **Debug Level**: Per-bar anomalies only
- **No Spam**: Absolutely no repeated identical error lines in tight loops
- **Implementation**: Guard with `once` flags or counters

### **Log Levels**
- **ERROR**: System failures, data corruption
- **WARNING**: Configuration issues, performance problems
- **INFO**: Fold-level summaries, system status
- **DEBUG**: Per-bar details, detailed debugging

## **Performance Standards**

### **Memory Management**
- **Target**: <2GB memory usage for large datasets
- **Monitoring**: Track memory usage during long backtests
- **Cleanup**: Explicit cleanup of large objects between folds

### **Execution Time**
- **Target**: <30min for 5-year backtest
- **Optimization**: Caching, parallel processing where possible
- **Profiling**: Regular performance profiling and optimization

## **Documentation Standards**

### **Code Documentation**
- **Docstrings**: All public functions must have comprehensive docstrings
- **Type Hints**: Exhaustive type hints for all functions
- **Examples**: Include usage examples in docstrings

### **Change Documentation**
- **Changelog**: Document all changes in `CHANGELOG.md`
- **Context**: Provide context for why changes were made
- **Testing**: Document test cases for new features

## **Deployment Standards**

### **Pre-Deployment Checklist**
- [ ] No unguarded `[-1]` usages remain
- [ ] No hardcoded knobs remain
- [ ] Tests pass locally; new behavior covered
- [ ] Logs are clean: zero repeated per-bar errors, one summary per fold
- [ ] README/CHANGELOG updated if behavior changes

### **Definition of Done**
- **Code Quality**: All new code follows established patterns
- **Testing**: Comprehensive test coverage for new features
- **Documentation**: Updated documentation reflecting changes
- **Performance**: No regression in performance metrics
- **Safety**: All safety checks implemented and tested

## **Known Hotspots to Fix**

### **Immediate Fixes Required**
- `core.engine.composer_integration`: guard all tail reads with `_last`, return HOLD on empties/NaN
- `core.utils` fold builder: enforce short-fold policy
- `core/data_sanity.py`: implement `validate_dataframe`; fix contradictory logs
- All `yf.download(...)`: add `auto_adjust=cfg["data"]["auto_adjust"]`
- After feature build: `features = features.dropna().iloc[cfg["engine"]["min_history_bars"]:]`

### **Ongoing Maintenance**
- Regular code reviews for adherence to principles
- Performance monitoring and optimization
- Test suite maintenance and expansion
- Documentation updates and improvements
