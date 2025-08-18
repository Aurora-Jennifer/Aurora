# Technical Indicators Consolidation Report

**Generated**: 2025-08-17 00:51:44.069217
**Files Analyzed**: 144

**Total Duplicate Patterns Found**: 10

## ./core/ml/profit_learner.py

### RSI calculation (1 instances)

**Line 340**:
```python
delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain /...
```

**Suggested Replacement**:
```python
from utils.indicators import rsi
rsi_val = rsi(close, window=14)
```

## ./features/ensemble.py

### RSI calculation (1 instances)

**Line 436**:
```python
delta = price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
```

**Suggested Replacement**:
```python
from utils.indicators import rsi
rsi_val = rsi(close, window=14)
```

## ./strategies/regime_aware_ensemble.py

### RSI calculation (1 instances)

**Line 481**:
```python
delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
       ...
```

**Suggested Replacement**:
```python
from utils.indicators import rsi
rsi_val = rsi(close, window=14)
```

## ./viz/ml_visualizer.py

### Rolling mean calculation (1 instances)

**Line 205**:
```python
.rolling(window=10).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

## ./scripts/generate_simple_signal_templates.py

### Rolling mean calculation (7 instances)

**Line 100**:
```python
.rolling(window=20).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

**Line 101**:
```python
.rolling(window=50).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

**Line 102**:
```python
.rolling(window=200).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

**Line 116**:
```python
.rolling(window=14).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

**Line 117**:
```python
.rolling(window=14).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

**Line 129**:
```python
.rolling(window=20).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

**Line 136**:
```python
.rolling(window=20).mean()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_mean
rolling_mean(data, window=WINDOW)
```

### Rolling standard deviation calculation (3 instances)

**Line 111**:
```python
.rolling(window=20).std()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_std
rolling_std(data, window=WINDOW)
```

**Line 112**:
```python
.rolling(window=50).std()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_std
rolling_std(data, window=WINDOW)
```

**Line 130**:
```python
.rolling(window=20).std()
```

**Suggested Replacement**:
```python
from utils.indicators import rolling_std
rolling_std(data, window=WINDOW)
```

## Migration Recommendations

1. **Start with high-impact files**: Focus on files with the most duplicates
2. **Use the centralized functions**: Import from `utils.indicators`
3. **Test thoroughly**: Ensure calculations remain identical
4. **Update imports**: Add `from utils.indicators import ...` statements

## Expected Benefits

- **Reduced code duplication**: Centralized indicator calculations
- **Improved maintainability**: Single source of truth for indicators
- **Better performance**: Optimized vectorized operations
- **Enhanced testing**: Easier to test indicator accuracy
- **Consistent behavior**: Standardized calculation methods
