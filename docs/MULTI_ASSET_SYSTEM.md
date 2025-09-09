# Multi-Asset Universe System

The multi-asset universe system extends the single-asset research pipeline to run systematic experiments across multiple assets, producing leaderboards and enabling portfolio construction.

## Overview

The system orchestrates per-asset grid experiments using the existing single-asset infrastructure:
- **Per-asset costs**: Different transaction costs for different assets (e.g., GME: 8bps, AAPL: 3bps)
- **Market proxy**: Uses QQQ as market benchmark for excess return calculations
- **Cross-asset proxies**: Optional additional assets for feature engineering
- **Leaderboard**: Ranks assets by performance with gate-passing criteria
- **Portfolio selection**: Tools to select top-K assets for portfolio construction

## Components

### 1. Universe Configuration (`config/universe.yaml`)

Defines the universe of assets, market proxy, and per-asset costs:

```yaml
universe:
  - AAPL
  - MSFT
  - NVDA
  - AMD
  - AMZN
  - GOOGL
  - META
  - TSLA
  - AVGO
  - SMCI
  - PLTR
  - COIN
  - SOFI
  - HOOD
  - SNAP
  - SHOP
  - RIVN
  - LCID
  - NIO
  - GME

market_proxy: QQQ
cross_proxies: [TLT, IEF, DXY, USO]

costs_bps:
  default: 3
  GME: 8      # High volatility, wide spreads
  COIN: 6     # Crypto exposure
  RIVN: 6     # EV startup
  # ... etc
```

### 2. Universe Runner (`ml/runner_universe.py`)

Orchestrates per-asset grid runs:

- Loads universe and grid configurations
- Runs `run_single_model_grid()` for each asset
- Applies per-asset costs (commission + slippage)
- Creates leaderboard with performance metrics
- Saves per-asset CSV files and summary statistics

### 3. CLI Interface (`scripts/run_universe.py`)

Command-line interface for running universe experiments:

```bash
# Classical models across universe
python scripts/run_universe.py \
  --universe-cfg config/universe.yaml \
  --grid-cfg config/grid.yaml \
  --out-dir results/classical_universe

# Deep learning models across universe
python scripts/run_universe.py \
  --universe-cfg config/universe.yaml \
  --grid-cfg config/dl.yaml \
  --out-dir results/dl_universe
```

### 4. Portfolio Selection (`scripts/portfolio_from_leaderboard.py`)

Selects top-K assets from leaderboard for portfolio construction:

```bash
# Select top 5 gate-passing assets
python scripts/portfolio_from_leaderboard.py \
  --leaderboard results/dl_universe/leaderboard.csv \
  --k 5 \
  --create-config

# Select assets with minimum Sharpe of 0.5
python scripts/portfolio_from_leaderboard.py \
  --leaderboard results/dl_universe/leaderboard.csv \
  --k 5 \
  --min-sharpe 0.5 \
  --create-config
```

## Output Structure

```
results/
├── classical_universe/
│   ├── leaderboard.csv          # Main leaderboard
│   ├── summary.json             # Run statistics
│   ├── universe_run.log         # Detailed logs
│   ├── AAPL_grid.csv           # Per-asset results
│   ├── NVDA_grid.csv
│   └── ...
└── dl_universe/
    ├── leaderboard.csv
    ├── summary.json
    ├── universe_run.log
    ├── AAPL_grid.csv
    └── ...
```

## Leaderboard Format

The leaderboard CSV contains:

| Column | Description |
|--------|-------------|
| `ticker` | Asset symbol |
| `best_median_sharpe` | Best configuration's median Sharpe ratio |
| `best_vs_BH` | Improvement over Buy & Hold |
| `best_vs_rule` | Improvement over Simple Rule |
| `median_turnover` | Median turnover across folds |
| `median_trades` | Median number of trades |
| `gate_pass` | Whether asset passed gate criteria |
| `runtime_sec` | Time to run grid for this asset |
| `costs_bps` | Per-asset transaction costs |
| `num_configs` | Number of successful configurations |

## Gate Criteria

An asset **passes the gate** if:

```
median_model_sharpe > max(median_sharpe_bh, median_sharpe_rule) + threshold_delta_vs_baseline
```

Where `threshold_delta_vs_baseline` is typically 0.1 (10 basis points).

## Usage Examples

### 1. Quick Test Run

```bash
# Create a small test universe
echo "universe: [AAPL, NVDA]
market_proxy: QQQ
costs_bps: {default: 3}" > test_universe.yaml

# Run with minimal grid
python scripts/run_universe.py \
  --universe-cfg test_universe.yaml \
  --grid-cfg config/grid.yaml \
  --out-dir test_results
```

### 2. Full Universe Run

```bash
# Run classical models across all 20 assets
python scripts/run_universe.py \
  --universe-cfg config/universe.yaml \
  --grid-cfg config/grid.yaml \
  --out-dir results/classical_full

# Run deep learning models
python scripts/run_universe.py \
  --universe-cfg config/universe.yaml \
  --grid-cfg config/dl.yaml \
  --out-dir results/dl_full
```

### 3. Portfolio Construction

```bash
# Select top 5 gate-passing assets
python scripts/portfolio_from_leaderboard.py \
  --leaderboard results/dl_full/leaderboard.csv \
  --k 5 \
  --create-config

# This creates portfolio_config.yaml with selected assets
```

## Performance Considerations

- **Runtime**: ~30-60 seconds per asset for classical models, ~2-5 minutes for DL models
- **Memory**: ~1-2GB per asset during training
- **Storage**: ~1-10MB per asset CSV file
- **Parallelization**: Currently sequential; can be parallelized with joblib

## Troubleshooting

### Common Issues

1. **"Edges nearly constant"**: Usually means SPY vs SPY (zero excess returns)
   - **Fix**: Use QQQ as market proxy, or different asset pairs

2. **"No successful experiments"**: All configurations failed
   - **Check**: Data availability, feature engineering, model parameters

3. **"Import errors"**: Missing dependencies
   - **Fix**: Install required packages (torch, yfinance, etc.)

### Debug Mode

```bash
# Run with verbose logging
python scripts/run_universe.py \
  --universe-cfg config/universe.yaml \
  --grid-cfg config/grid.yaml \
  --out-dir results/debug \
  --verbose
```

## Integration with Existing Pipeline

The multi-asset system reuses all existing components:

- ✅ **Walkforward validator**: Same validation logic, costs, hysteresis
- ✅ **Feature engineering**: Same 28-feature pipeline
- ✅ **Model training**: Same classical and DL models
- ✅ **Decision making**: Same temperature calibration and hysteresis
- ✅ **Baselines**: Same Buy & Hold and Simple Rule baselines
- ✅ **Gating**: Same gate criteria and thresholds

## Future Enhancements

- **Parallel execution**: Run assets in parallel with joblib
- **Portfolio combiner**: Equal-weight portfolio with synchronized timeline
- **Risk management**: Position sizing and risk controls
- **Real-time updates**: Live universe monitoring and rebalancing
- **Cross-asset features**: Features using relationships between assets
