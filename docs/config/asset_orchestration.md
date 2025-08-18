# Asset-Oriented Composer Orchestration

This page explains how to drive multi-model/strategy selection per asset class using configuration only.

## Runtime assets block

Add the following to `config/base.yaml` (already added):

```yaml
assets:
  crypto:
    eligible_strategies: [mom_fast, breakout_adx, trend_follow_x]
    composer_params:
      family: momentum-v3
      temperature: 0.65
      top_k: 2
      min_conf: 0.62
      tie_break: "vol_adj"
  equity:
    eligible_strategies: [mean_rev_basic, carry_blend, momentum_balanced, seasonal_edge]
    composer_params:
      family: blend-v2
      temperature: 0.85
      top_k: 3
      min_conf: 0.45
      tie_break: "risk_first"
  etf:
    eligible_strategies: [low_vol_carry, mean_rev_basic]
    composer_params:
      family: conservative-v1
      temperature: 0.75
      top_k: 2
      min_conf: 0.50
      tie_break: "sharpe_first"
```

- The composer uses `get_asset_class(symbol)` to determine `asset_class`.
- It then narrows to `eligible_strategies` and applies `composer_params` for that class.

## Relationship to `asset_classes` overlays

- `asset_classes` is used by `core/config_loader.py` when you pass `--asset` at CLI time.
- `assets` is read at runtime by the composer; it is the primary source for orchestration.

## Logging verification

Composer logs a compact line when making decisions (INFO level):

```
composer: asset_class=<class> family=<family> elig=<list> conf=<float> signal=<float> regime=<str>
```

This confirms the class, family, and narrowed strategies used for each symbol.

## Quick verification run

Short range, small folds to keep it fast:

```bash
python scripts/multi_walkforward_report.py \
  --symbols SPY TSLA BTC-USD \
  --start 2019-01-01 --end 2019-06-30 \
  --separate-crypto --validate-data
```

- Expect SPY/TSLA to log as `asset_class=equity`, `family=blend-v2`, broader eligible set.
- Expect BTC-USD to log as `asset_class=crypto`, `family=momentum-v3`, narrower eligible set.
- Reports are written under `docs/analysis/`.

## Deployment note

- No core logic changes are required for orchestration. Only configuration and logging are involved.
