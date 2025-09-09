# Decision Core - Single Source of Truth

## Purpose

The `core/decision_core.py` module defines the **single** function that maps `(state, τ, costs, prev_pos)` → `(action, position, costs_applied)`. This is the only allowed decision path in production.

## Exact Decision Logic

### τ Comparison Rule
```python
if abs(edge) >= τ:
    action = 1 if edge > 0 else -1  # BUY or SELL
else:
    action = 0  # HOLD
```

### Position Transition Table
| Action | Previous Position | New Position | Cost Applied |
|--------|------------------|--------------|--------------|
| BUY (1) | -1, 0, 1 | 1 | Yes (if position changed) |
| SELL (-1) | -1, 0, 1 | -1 | Yes (if position changed) |
| HOLD (0) | -1, 0, 1 | Same | No |

### Cost Policy
- Commission + Slippage applied **only** when `new_position != previous_position`
- Cost = `(commission_bps + slippage_bps) / 10000.0`
- No cost on HOLD actions

### Numerical Rules
- Logits clamped to finite values
- Temperature scaling: `logits / max(temperature, 1e-8)`
- Softmax applied after temperature scaling
- Advantage values must be finite
- τ threshold: 0.0001 (validated from research)

## Invariants (Runtime Asserts)

1. **Single Decision Engine**: `cfg.runtime.single_decision_core == True`
2. **No Legacy Paths**: `cfg.runtime.allow_legacy_paths == False`
3. **No NaN/Inf**: All inputs must be finite
4. **Monotonic State Updates**: Position changes only on BUY/SELL actions
5. **Cost Consistency**: Costs applied only on position changes

## Code Location
- **File**: `core/decision_core.py`
- **Main Function**: `decide(logits, advantage, cfg)`
- **Position Logic**: `next_position(prev_pos, action)`
- **Cost Logic**: `simulate_step(prev_pos, action, price, cost_bps)`

## Validation
- All decision logic tested in `scripts/test_decision_core.py`
- Wire-level asserts catch inconsistencies
- Smoke tests verify realistic trading scenarios

## Usage
```python
from core.decision_core import decide, DecisionCfg, BUY, SELL, HOLD

cfg = DecisionCfg(tau=0.0001, temperature=1.0, gate_on="adv", cost_bps=4.0)
action = decide(logits, advantage, cfg)
```

## Enforcement
- All evaluation paths must use this module
- Falsification harness uses identical logic
- Paper trading uses identical logic
- Production trading uses identical logic
- No exceptions or legacy paths allowed
