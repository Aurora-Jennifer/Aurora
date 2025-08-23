# Metrics Contract - Aurora Trading System

**Version**: 1.0  
**Status**: LOCKED - Changes require explicit approval  
**Purpose**: Define exact schema, units, and methods for comprehensive metrics collection

---

## Schema Definition

All metrics must be emitted as JSON to `artifacts/<run_id>/metrics.json` and in structured logs.

### Required Fields

```json
{
  "run_id": "string",
  "timestamp": "ISO8601 string",
  "runtime_seconds": "float >= 0",
  
  "ic_spearman": {
    "value": "float in [-1,1] or null",
    "method": "scipy.stats.spearmanr on same-bar future return target",
    "horizon": "1 bar forward",
    "nan_handling": "drop pairwise",
    "min_pairs": 10,
    "pairs_used": "int >= 0"
  },
  
  "turnover": {
    "value": "float >= 0",
    "method": "0.5 * sum(|w_t - w_{t-1}|) per period",
    "denominator_includes": "all position changes",
    "units": "fraction of portfolio value"
  },
  
  "fill_rate": {
    "value": "float in [0,1] or null", 
    "method": "fills_received / orders_submitted",
    "denominator_includes": "submitted orders, excludes cancels",
    "zero_orders_convention": "null"
  },
  
  "latency_ms": {
    "avg": "float >= 0",
    "p95": "float >= avg", 
    "max": "float >= p95",
    "method": "wall-clock time including I/O",
    "computation": "numpy.percentile(..., 95)"
  },
  
  "memory_peak_mb": {
    "value": "float >= 0",
    "method": "RSS peak during run via psutil.Process().memory_info().rss",
    "units": "megabytes"
  },
  
  "trading": {
    "orders_sent": "int >= 0",
    "fills_received": "int >= 0", 
    "rejections": "int >= 0"
  }
}
```

## Validation Rules

### Type Constraints
- All floats must be finite (no `inf`, `-inf`)
- All timestamps must be valid ISO8601
- All counts must be non-negative integers

### Logical Invariants
- `0 <= fill_rate <= 1` when not null
- `latency_ms.p95 >= latency_ms.avg`
- `latency_ms.max >= latency_ms.p95`  
- `abs(ic_spearman) <= 1` when not null
- `turnover = 0` when positions are constant
- `fills_received <= orders_sent`

### Null Handling
- `ic_spearman`: null when insufficient data (<10 pairs)
- `fill_rate`: null when zero orders submitted
- All other fields must have valid values

## Golden Reproducibility

### Stability Tolerances
When comparing to golden reference on fixed snapshot + seed:

- `latency_ms.avg`: ±10%
- `latency_ms.p95`: ±10% 
- `memory_peak_mb`: ±15%
- `ic_spearman`: ±0.02 absolute
- `turnover`: ±0.05 absolute
- `fill_rate`: ±0.02 absolute

### CI Gate Requirements
1. File must exist at expected path
2. JSON must parse and validate against schema
3. All invariants must hold
4. Stability check must pass vs golden reference

## Emission Requirements

### Artifacts
- Primary: `artifacts/<run_id>/metrics.json`
- Paper trading: `artifacts/paper/<session_id>/metrics.json` 
- Logs: Single JSON line with prefix `METRICS:`

### Frequency
- E2D: Once per run completion
- Paper trading: Every 10 loops or 60 seconds
- Walkforward: Once per fold completion

## Implementation Notes

### IC Calculation
```python
# Exact method specification
def calculate_ic_spearman(predictions: np.array, returns: np.array) -> float:
    """
    Calculate Information Coefficient using Spearman rank correlation.
    
    Args:
        predictions: Model predictions (any numeric values)
        returns: Same-bar future returns (1 period forward)
    
    Returns:
        Spearman correlation coefficient or None if insufficient data
    """
    if len(predictions) < 10 or len(returns) < 10:
        return None
    
    # Drop pairs where either value is NaN
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    if np.sum(mask) < 10:
        return None
        
    corr, _ = spearmanr(predictions[mask], returns[mask])
    return float(corr) if not np.isnan(corr) else None
```

### Turnover Calculation
```python
def calculate_turnover(positions: pd.Series) -> float:
    """
    Calculate portfolio turnover.
    
    Args:
        positions: Position weights over time
        
    Returns:
        Average turnover per period
    """
    if len(positions) < 2:
        return 0.0
        
    position_changes = positions.diff().abs()
    return float(0.5 * position_changes.mean())
```

## Failure Modes

### Hard Failures (CI must fail)
- Schema validation failure
- Invariant violation 
- Stability tolerance breach
- Missing required files

### Soft Failures (log warnings)
- Null values in expected ranges
- Performance degradation within tolerance
- Memory usage approaching limits

---

**Contract Lock**: This specification is binding. All implementations must conform exactly to these requirements. Changes require explicit approval and version increment.
