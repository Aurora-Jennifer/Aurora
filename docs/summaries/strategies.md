# Strategies Module — Summary

Purpose: Strategy implementations and composition (base, momentum, mean-reversion, SMA crossover, ensembles, regime-aware).

Entrypoints:
- `strategies/base.py` — base classes and interfaces
- `strategies/momentum.py`, `strategies/mean_reversion.py`, `strategies/sma_crossover.py` — concrete strategies
- `strategies/ensemble_strategy.py`, `strategies/regime_aware_ensemble.py` — ensembles/regime-aware composition

Do-not-touch:
- Base interfaces in `base.py` and factory wiring in `factory.py`

```mermaid
graph TD
  A[Market Data + Features] --> B[Strategy (e.g., momentum)]
  A --> C[Strategy (mean_reversion)]
  A --> D[Strategy (sma_crossover)]
  B --> E[Ensemble]
  C --> E[Ensemble]
  D --> E[Ensemble]
  E --> F[Regime-aware Ensemble]
  F --> G[Signals/Weights]
```

### API (selected public)
- `BaseStrategy` (from `base.py`)
- `MomentumStrategy`, `MeanReversionStrategy`, `SmaCrossoverStrategy`
- `EnsembleStrategy`, `RegimeAwareEnsemble`
- `factory.create(name, **kwargs)`

