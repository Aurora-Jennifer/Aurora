# Reproducibility Checklist

## Configuration Bundle: frozen_config_20250905_143551_6fd75c9
**Created:** 2025-09-05T14:35:51.660538
**Git SHA:** 6fd75c9fb85641addb8964f52cf5b60d0f50fb4d

## Performance Metrics
- **Mean Sharpe:** 0.000
- **Median Sharpe:** 0.000
- **Successful Folds:** 0/0

## Reproducibility Steps

### 1. Environment Setup
```bash
# Install exact requirements
pip install -r requirements.txt

# Verify Python version
python --version  # Should match manifest
```

### 2. Data Requirements
- **Symbols:** SPY, QQQ
- **Lookback:** 400 days
- **Test Window:** 120 days
- **Folds:** 3

### 3. Reproducibility Test
```bash
# Run with frozen config
python scripts/walkforward.py --config frozen_config_20250905_143551_6fd75c9/config.yaml

# Expected results should match:
# Mean Sharpe: 0.000 ± 0.1
```

### 4. Determinism Check
```bash
# Run 3 times with different seeds
python scripts/falsification_harness.py --config frozen_config_20250905_143551_6fd75c9/config.yaml
```

## Validation Gates
- [ ] Mean Sharpe ≥ 0.3
- [ ] Determinism test passes (CV < 0.2)
- [ ] Cost stress test shows graceful degradation
- [ ] Feature ablation maintains performance

## Next Steps
1. Run falsification harness
2. If passes, proceed to paper trading
3. Monitor for 30 days with frozen config
4. Only then consider live deployment
