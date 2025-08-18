# Composer Integration and Validation Refactoring Summary

This document summarizes the comprehensive refactoring of the composer integration and validation system according to the engineering charter requirements.

## Changes Made

### 1. Data Sanity Back-Compat Shim (`core/data_sanity.py`)

- **Added**: `validate_dataframe(self, data: pd.DataFrame, **kwargs)` method that proxies to `validate`
- **Features**:
  - Supports both positional and keyword arguments for backward compatibility
  - Handles mode-based behavior (warn/error) from profile configuration
  - Raises `DataSanityError` if mode is "error" and validation issues exist
  - Maintains existing API while adding new functionality

### 2. Composer Integration Warmup Gating (`core/engine/composer_integration.py`)

- **Added**: Warmup gating that prevents composer calls before `min_history_bars`
- **Features**:
  - All composer calls are now gated behind `bar_idx < min_history_bars`
  - Returns HOLD decisions with reason "warmup" instead of "composer_exception"
  - Improved error handling with first-failure-only logging
  - Added NaN count tracking in error messages
  - Added DEBUG logs for fold information and first composer call

### 3. Composer Registry Strategy Filtering (`core/composer/registry.py`)

- **Added**: Strategy filtering to only use registered strategies
- **Features**:
  - Filters `eligible_strategies` to only registered ones
  - Warns about missing strategies (e.g., "breakout")
  - Raises error if fewer than 2 strategies remain
  - Stores strategies in composer for validation
  - Enhanced logging with strategy names

### 4. Weight Vector Validation (`core/engine/composer_integration.py`)

- **Added**: Post-composer weight validation
- **Features**:
  - Asserts weight vector length equals `len(composer.strategies)`
  - Validates all weights are finite
  - Raises descriptive errors for validation failures

### 5. Fold Generation Short Window Handling (`core/walk/folds.py`)

- **Added**: Short final test window handling
- **Features**:
  - New parameter `allow_truncated_final_fold: bool = False`
  - Skips under-sized final test windows by default
  - Logs single INFO message about skips
  - Option to allow truncated folds with adjusted stride

### 6. Metrics Aggregation Empty Handling (`scripts/walkforward_with_composer.py`, `scripts/walkforward_framework.py`)

- **Added**: Empty equity curve and zero trades handling
- **Features**:
  - Returns zero metrics with reason "no_trades" for empty/NaN equity curves
  - Prevents NumPy warnings on empty arrays
  - Handles both empty PnL series and zero trade counts
  - Maintains consistent return structure

### 7. Configuration System (`core/config.py`, `config/*.yaml`)

- **Added**: Comprehensive configuration system
- **Features**:
  - `load_config()` with deep merging support
  - `get_cfg()` for dot-notation access
  - `validate_config()` for structure validation
  - Base configuration with all required settings
  - Risk overlay configurations (low, balanced, strict)

### 8. Enhanced Logging and Debugging

- **Added**: Comprehensive DEBUG logging
- **Features**:
  - Fold-level information (strategies, min_history_bars)
  - First composer call logging with feature size and NaN counts
  - One-time failure logging per fold with bar index
  - Compact NaN count dumps in error messages

## Configuration Files Created

### `config/base.yaml`
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

### Risk Overlay Configurations
- `config/risk_low.yaml` - Conservative settings
- `config/risk_balanced.yaml` - Default settings
- `config/risk_strict.yaml` - Aggressive settings

## Testing

Created comprehensive test suite (`tests/test_composer_refactoring.py`) covering:

1. **Warmup Gating**: Verifies composer calls are blocked before min_history_bars
2. **Strategy Filtering**: Tests registered strategy filtering and error handling
3. **Insufficient Strategies**: Validates error when < 2 strategies available
4. **Short Window Handling**: Tests fold generation with truncated final folds
5. **Empty Metrics**: Verifies zero metrics with reason for empty equity curves
6. **Config Loading**: Tests configuration loading and merging
7. **Data Sanity Back-Compat**: Validates backward compatibility

All tests pass successfully.

## Charter Compliance

✅ **Never index empty arrays/frames**: All composer calls use `_safe_len()` and `_last()` helpers
✅ **No hardcoded runtime knobs**: All settings moved to configuration files
✅ **Short test folds are illegal**: Implemented skip/truncate logic with config control
✅ **Warmup discipline**: Enforced `min_history_bars` before any composer decisions
✅ **One log per cause per fold**: Implemented first-failure-only logging with counters
✅ **yfinance behavior is explicit**: `auto_adjust` configured via config
✅ **Idempotent, minimal diffs**: Surgical changes preserving public APIs

## Usage Examples

### Basic Configuration Loading
```python
from core.config import load_config

# Load base configuration
config = load_config(['config/base.yaml'])

# Load with risk overlay
config = load_config(['config/base.yaml', 'config/risk_low.yaml'])
```

### Composer Integration
```python
from core.engine.composer_integration import ComposerIntegration

composer = ComposerIntegration(config)
signal, metadata = composer.get_composer_decision(data, 'SPY', current_idx)
```

### Fold Generation
```python
from core.walk.folds import gen_walkforward

# Skip short final folds (default)
folds = list(gen_walkforward(n, train_len, test_len, stride))

# Allow truncated final folds
folds = list(gen_walkforward(n, train_len, test_len, stride, allow_truncated_final_fold=True))
```

## Benefits

1. **Safety**: Eliminates array indexing errors and provides graceful degradation
2. **Configurability**: All runtime parameters configurable without code changes
3. **Robustness**: Handles edge cases (empty data, zero trades, short folds)
4. **Observability**: Enhanced logging for debugging and monitoring
5. **Maintainability**: Clean separation of concerns and comprehensive testing
6. **Performance**: Efficient warmup handling and error logging

The refactoring successfully implements all charter requirements while maintaining backward compatibility and adding comprehensive safety measures.
