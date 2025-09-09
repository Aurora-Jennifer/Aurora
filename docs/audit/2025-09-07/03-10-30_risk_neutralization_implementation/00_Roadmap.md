# Risk Neutralization Implementation — Roadmap (2025-09-07 03:10)

**Prompt:** "Can you create a context file for this update phase"

## Context

We are in the middle of implementing **P1 — Portfolio Construction & Gates** improvements to the multi-asset universe runner. The previous P0 fixes (IR_mkt calculation, MaxDD bug, feature leakage, parallel crashes) have been successfully completed and tested.

## Current Phase: P1.2 — Risk Neutralization

### What We're Implementing

**Goal:** Implement beta-neutralization and sector-neutralization before cross-sectional ranking to remove unwanted market and sector exposures from signals.

**Key Components:**
1. **Market Beta Neutralization**: Remove market beta exposure from cross-sectional scores
2. **Sector Neutralization**: Remove sector-specific exposures using dummy variables
3. **Size Factor Neutralization**: Remove size bias (optional)
4. **Momentum Factor Neutralization**: Remove momentum bias (optional)

### Implementation Status

✅ **Completed:**
- Created `ml/risk_neutralization.py` module with comprehensive neutralization functions
- Added import to `ml/runner_universe.py`
- Designed neutralization pipeline with OLS-based factor removal
- Updated TODO list with specific neutralization tasks

🔄 **In Progress:**
- Integrating risk neutralization into cross-sectional training pipeline
- Adding market cap data extraction for size factor
- Configuring neutralization parameters

⏳ **Pending:**
- Complete integration into training pipeline
- Add market cap data extraction from panel
- Add sector mapping configuration
- Test risk neutralization with 60-day OOS run
- Validate exposure reduction (market beta and sector exposures)
- Performance comparison (neutralized vs non-neutralized)

## Technical Approach

### Neutralization Method
- **OLS-based**: Use linear regression to remove factor exposures
- **Cross-sectional**: Apply neutralization within each date
- **Robust**: Handle missing data and edge cases gracefully

### Factor Construction
- **Market Beta**: Rolling 252-day regression vs market proxy (SPY)
- **Sector Dummies**: One-hot encoding of sector classifications
- **Size Factor**: Log market cap (most recent available)
- **Momentum Factor**: 252-day cumulative return (skipping last 21 days)

### Integration Points
- **Before Training**: Neutralize cross-sectional targets
- **After Prediction**: Neutralize model predictions
- **Configurable**: Enable/disable individual factors via config

## Success Criteria

1. **Exposure Reduction**: Portfolio beta and sector exposures closer to zero
2. **Performance Preservation**: Maintain or improve risk-adjusted returns
3. **Stability**: Robust neutralization across different market regimes
4. **Configurability**: Easy to enable/disable different neutralization factors

## Next Steps

1. Complete integration into training pipeline
2. Add market cap data extraction
3. Test with 60-day OOS run
4. Validate exposure reduction
5. Move to P1.3 (turnover controls) and P1.4 (transaction costs)

## Files Modified

- `ml/risk_neutralization.py` (NEW) - Core neutralization functions
- `ml/runner_universe.py` - Integration point for neutralization
- `config/gates_adaptive.yaml` - Adaptive gate configuration (completed)

## Dependencies

- `sklearn.linear_model.LinearRegression` for OLS neutralization
- Market cap data from panel builder
- Sector mapping (to be added to universe config)
