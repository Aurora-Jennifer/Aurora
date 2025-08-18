# Performance Optimization Plan for ML Walkforward System

## Executive Summary

**Target**: ≥3× faster wall-clock runtime on 8 cores without changing results
**Current State**: Single-threaded walkforward analysis with heavy I/O and feature engineering
**Goal**: Parallelize folds, optimize I/O, and vectorize computations

## Current Performance Baseline

### Representative Workloads
- **Small**: 2 years, 4 folds, 5 assets = ~20 backtest runs
- **Medium**: 5 years, 10 folds, 5 assets = ~50 backtest runs
- **Large**: 10 years, 20 folds, 5 assets = ~100 backtest runs

### Expected Hotspots (Based on Code Analysis)
1. **Data Loading & Validation** (~40%): Repeated CSV parsing, data sanity checks
2. **Feature Engineering** (~30%): Pandas operations, rolling windows, correlations
3. **ML Training** (~20%): Model fitting, feature importance calculation
4. **I/O Operations** (~10%): Logging, checkpointing, results saving

## Optimization Strategy

### Phase 1: Profiling & Baseline (Day 1-2)

#### 1.1 Profile Collection
```bash
# Install profiling tools
pip install line_profiler memory_profiler psutil

# Collect baseline profiles
mkdir -p .perf

# Small workload profile
python -m cProfile -o .perf/small_profile.pstats scripts/ml_walkforward.py \
  --start-date 2020-01-01 --end-date 2021-12-31 \
  --fold-length 252 --step-size 126 \
  --config config/ml_backtest_unified.json \
  --output-dir results/perf_baseline_small

# Medium workload profile
python -m cProfile -o .perf/medium_profile.pstats scripts/ml_walkforward.py \
  --start-date 2018-01-01 --end-date 2022-12-31 \
  --fold-length 252 --step-size 126 \
  --config config/ml_backtest_unified.json \
  --output-dir results/perf_baseline_medium

# Analyze profiles
python -c "
import pstats
p = pstats.Stats('.perf/small_profile.pstats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### 1.2 Line Profiling
```bash
# Profile top functions
kernprof -l -v scripts/ml_walkforward.py \
  --start-date 2020-01-01 --end-date 2021-12-31 \
  --fold-length 252 --step-size 126 \
  --config config/ml_backtest_unified.json
```

### Phase 2: Parallelization (Day 3-5)

#### 2.1 Fold-Level Parallelization
**Effort**: Medium | **Risk**: Low | **Est. Speedup**: 3-4x

**Implementation**:
```python
# scripts/ml_walkforward_parallel.py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import joblib

class ParallelMLWalkforwardAnalyzer(MLWalkforwardAnalyzer):
    def run_ml_walkforward_parallel(self, start_date, end_date,
                                   fold_length, step_size, warm_start,
                                   n_jobs=-1):
        """Run walkforward analysis with parallel fold processing."""
        folds = self._generate_folds(start_date, end_date, fold_length, step_size)

        # Prepare fold arguments
        fold_args = [
            (fold, i, warm_start) for i, fold in enumerate(folds)
        ]

        # Process folds in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(self._process_fold_worker, fold_args))

        return self._aggregate_results(results)

    def _process_fold_worker(self, fold_args):
        """Worker function for processing a single fold."""
        (train_start, train_end, test_start, test_end), fold_idx, warm_start = fold_args

        # Create isolated environment for each worker
        worker_analyzer = MLWalkforwardAnalyzer(self.config_file,
                                               f"{self.output_dir}/fold_{fold_idx}")

        # Process fold
        train_results = worker_analyzer._run_training_fold(train_start, train_end)
        test_results = worker_analyzer._run_testing_fold(test_start, test_end, train_results)

        return {
            'fold_idx': fold_idx,
            'train_results': train_results,
            'test_results': test_results,
            'fold_dates': (train_start, train_end, test_start, test_end)
        }
```

**Files to Modify**:
- `scripts/ml_walkforward.py` → Add parallel execution option
- `core/engine/backtest.py` → Ensure thread-safe operations
- `core/ml/profit_learner.py` → Add model serialization for workers

#### 2.2 Asset-Level Parallelization
**Effort**: Low | **Risk**: Low | **Est. Speedup**: 1.2-1.5x

**Implementation**:
```python
# core/engine/backtest.py
def run_backtest_parallel(self, start_date, end_date, n_jobs=-1):
    """Run backtest with parallel asset processing."""
    if self.unified_model:
        # Process assets in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            asset_results = list(executor.map(
                self._process_asset_worker,
                self.assets
            ))
        return self._combine_asset_results(asset_results)
    else:
        return self.run_backtest(start_date, end_date)
```

### Phase 3: I/O Optimization (Day 6-7)

#### 3.1 Data Caching & Prefetching
**Effort**: Medium | **Risk**: Low | **Est. Speedup**: 1.5-2x

**Implementation**:
```python
# core/data/cache.py
import joblib
import hashlib
from pathlib import Path

class DataCache:
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_or_load_data(self, symbol, start_date, end_date):
        """Load data with caching."""
        cache_key = self._generate_cache_key(symbol, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load and cache
        data = self._load_raw_data(symbol, start_date, end_date)
        joblib.dump(data, cache_file)
        return data

    def _generate_cache_key(self, symbol, start_date, end_date):
        """Generate deterministic cache key."""
        content = f"{symbol}_{start_date}_{end_date}"
        return hashlib.md5(content.encode()).hexdigest()
```

#### 3.2 Optimized Data Formats
**Effort**: High | **Risk**: Medium | **Est. Speedup**: 1.3-1.8x

**Implementation**:
```python
# Convert CSV to Parquet for faster I/O
import pandas as pd

def convert_data_to_parquet():
    """Convert all CSV data to Parquet format."""
    for symbol in ['SPY', 'AAPL', 'TSLA', 'GOOG', 'BTC-USD']:
        csv_file = f"data/{symbol}.csv"
        parquet_file = f"data/{symbol}.parquet"

        if not Path(parquet_file).exists():
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)

# Update data loading
def load_data_optimized(symbol, start_date, end_date):
    """Load data from optimized format."""
    parquet_file = f"data/{symbol}.parquet"
    df = pd.read_parquet(parquet_file)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
```

### Phase 4: Computational Optimization (Day 8-10)

#### 4.1 Vectorized Feature Engineering
**Effort**: High | **Risk**: Medium | **Est. Speedup**: 1.5-2.5x

**Implementation**:
```python
# core/data/features_vectorized.py
import numpy as np
from numba import jit

@jit(nopython=True)
def calculate_technical_indicators_vectorized(prices, volumes):
    """Vectorized technical indicator calculation."""
    n = len(prices)

    # Pre-allocate arrays
    sma_20 = np.zeros(n)
    sma_50 = np.zeros(n)
    rsi = np.zeros(n)
    volatility = np.zeros(n)

    # Vectorized calculations
    for i in range(20, n):
        sma_20[i] = np.mean(prices[i-20:i])

    for i in range(50, n):
        sma_50[i] = np.mean(prices[i-50:i])

    # RSI calculation
    for i in range(1, n):
        gains = np.maximum(0, prices[i] - prices[i-1])
        losses = np.maximum(0, prices[i-1] - prices[i])

        if i >= 14:
            avg_gain = np.mean(gains[i-14:i])
            avg_loss = np.mean(losses[i-14:i])
            if avg_loss != 0:
                rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))

    return sma_20, sma_50, rsi, volatility
```

#### 4.2 Model Training Optimization
**Effort**: Medium | **Risk**: Low | **Est. Speedup**: 1.2-1.8x

**Implementation**:
```python
# core/ml/profit_learner_optimized.py
from sklearn.ensemble import RandomForestRegressor
import joblib

class OptimizedProfitLearner(ProfitLearner):
    def __init__(self, config):
        super().__init__(config)
        self.model_cache = {}

    def train_model_optimized(self, X, y):
        """Optimized model training with caching."""
        # Use more efficient algorithms
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,  # Use all cores
            random_state=42
        )

        # Train with early stopping
        model.fit(X, y)
        return model

    def predict_batch(self, X_batch):
        """Batch prediction for efficiency."""
        return self.model.predict(X_batch)
```

### Phase 5: Memory Optimization (Day 11-12)

#### 5.1 Memory Mapping
**Effort**: Low | **Risk**: Low | **Est. Speedup**: 1.1-1.3x

**Implementation**:
```python
# core/data/memory_mapped.py
import numpy as np
import mmap

class MemoryMappedData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.mmap = None

    def __enter__(self):
        with open(self.file_path, 'rb') as f:
            self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return np.frombuffer(self.mmap, dtype=np.float64)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mmap:
            self.mmap.close()
```

## Implementation Plan

### Day 1-2: Profiling & Baseline
- [ ] Set up profiling environment
- [ ] Run baseline measurements
- [ ] Identify top 5 hotspots
- [ ] Document current performance metrics

### Day 3-5: Parallelization
- [ ] Implement fold-level parallelization
- [ ] Add asset-level parallelization
- [ ] Ensure thread safety
- [ ] Test with 1, 2, 4, 8 workers

### Day 6-7: I/O Optimization
- [ ] Implement data caching
- [ ] Convert to Parquet format
- [ ] Add prefetching
- [ ] Optimize logging

### Day 8-10: Computational Optimization
- [ ] Vectorize feature engineering
- [ ] Optimize model training
- [ ] Add JIT compilation
- [ ] Implement batch processing

### Day 11-12: Memory & Final Optimization
- [ ] Add memory mapping
- [ ] Optimize memory usage
- [ ] Final performance testing
- [ ] Documentation

## Expected Results

| Optimization | Effort | Risk | Est. Speedup | Owners | PRs |
|--------------|--------|------|--------------|--------|-----|
| Fold Parallelization | Medium | Low | 3-4x | Core Team | #1 |
| I/O Optimization | Medium | Low | 1.5-2x | Data Team | #2 |
| Vectorization | High | Medium | 1.5-2.5x | ML Team | #3 |
| Memory Mapping | Low | Low | 1.1-1.3x | Core Team | #4 |
| **Total Expected** | - | - | **6-12x** | - | - |

## Validation & Testing

### Determinism Testing
```bash
# Test that parallel results match sequential
python scripts/test_determinism.py \
  --sequential-results results/baseline \
  --parallel-results results/optimized \
  --tolerance 1e-9
```

### Performance Benchmarking
```bash
# Run performance benchmarks
python scripts/benchmark_performance.py \
  --workers 1,2,4,8 \
  --workloads small,medium,large \
  --output results/performance_benchmarks.json
```

### Acceptance Criteria
- [ ] Same trades/metrics as single-core within tolerance
- [ ] Wall time improvement ≥3x on 8 cores
- [ ] No race conditions or deadlocks
- [ ] Graceful cancellation support
- [ ] Memory usage within reasonable bounds

## Risk Mitigation

1. **Backward Compatibility**: Maintain sequential execution as fallback
2. **Gradual Rollout**: Implement optimizations incrementally
3. **Extensive Testing**: Validate results at each step
4. **Monitoring**: Add performance metrics and alerts
5. **Rollback Plan**: Keep previous versions for quick rollback

## Success Metrics

- **Primary**: ≥3x wall-clock speedup on 8 cores
- **Secondary**: Same numerical results within 1e-9 tolerance
- **Tertiary**: Memory usage <2x baseline
- **Quaternary**: Zero race conditions or deadlocks

## Next Steps

1. **Immediate**: Start with profiling (Day 1-2)
2. **Short-term**: Implement parallelization (Day 3-5)
3. **Medium-term**: Optimize I/O and computation (Day 6-10)
4. **Long-term**: Memory optimization and fine-tuning (Day 11-12)

This plan provides a systematic approach to achieving the target 3x+ speedup while maintaining result accuracy and system reliability.
