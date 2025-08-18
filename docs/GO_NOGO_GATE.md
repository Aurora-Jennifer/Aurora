# Go/No-Go Gate for Paper/Live Trading

## Overview

The Go/No-Go gate is a comprehensive safety system that performs essential checks before allowing paper or live trading runs. It's designed to fail fast on anything that would make a trading run meaningless or unsafe.

## What It Enforces

### 1. Data Sanity
- No extreme/negative prices
- No NaN bursts
- No dtype drift
- No duplicate timestamps
- Proper timezone handling

### 2. Leakage Checks
- No future data in features/signals
- Train/test splits respected
- Rolling stats use min_periods

### 3. Backtest Realism
- Fees enabled and configured
- Slippage enabled and configured
- Position inventory never negative
- Cash/borrow checks

### 4. Risk Rails
- Max position %/notional limits
- Per-day loss cut limits
- Max drawdown cut limits
- Turnover caps

### 5. PnL Accounting
- Positions × prices + cash + fees = equity
- No divergence > tolerance

### 6. Attribution Ready
- Alpha vs. fees vs. slippage logged
- Strategy/feature contributions recorded

### 7. Walk-Forward OOS
- Latest run has at least one OOS segment
- Positive edge required or gate blocks live

### 8. Observability
- Structured logs enabled
- Run ID/version pinned
- Config hash logged

### 9. Kill-Switch
- Circuit breaker signal reachable
- Manual override file exists

## Files

### Configuration
- **`config/go_nogo.yaml`** - Gate configuration and thresholds

### Scripts
- **`scripts/go_nogo.py`** - Main gate script

### Runtime Files
- **`runtime/ALLOW_LIVE.txt`** - Manual override file (must exist for live trading)
- **`runtime/killswitch.signal`** - Kill-switch signal file (created automatically)

### Results
- **`results/walkforward/latest_oos_summary.json`** - OOS performance summary

## Usage

### Basic Usage

```bash
# Run the gate with default settings
python scripts/go_nogo.py

# Run with proper environment variables
STRUCTURED_LOGS=1 RUN_ID=$(date +%Y%m%d-%H%M%S) \
MAX_POSITION_PCT=0.15 MAX_GROSS_LEVERAGE=2.0 \
DAILY_LOSS_CUT_PCT=0.03 MAX_DRAWDOWN_CUT_PCT=0.20 MAX_TURNOVER_PCT=300 \
python scripts/go_nogo.py
```

### Makefile Targets

```bash
# Run gate with all environment variables set
make go-nogo

# Run gate with custom environment
make go-nogo-custom

# Block live trading (remove override file)
make block-live

# Allow live trading (create override file)
make allow-live
```

### Environment Variables

Set these environment variables for proper operation:

```bash
export STRUCTURED_LOGS=1
export RUN_ID=$(date +%Y%m%d-%H%M%S)
export MAX_POSITION_PCT=0.15
export MAX_GROSS_LEVERAGE=2.0
export DAILY_LOSS_CUT_PCT=0.03
export MAX_DRAWDOWN_CUT_PCT=0.20
export MAX_TURNOVER_PCT=300
```

## Configuration

### `config/go_nogo.yaml`

```yaml
profile: strict

data:
  max_gap_minutes: 5
  allow_negative_prices: false
  max_price: 100000.0
  allow_duplicate_timestamps: false
  timezone: "UTC"

leakage:
  forbid_future_lookback: true
  min_rolling_window: 20
  check_train_test_separation: true

backtest:
  require_fees: true
  require_slippage: true
  max_slippage_bps: 50
  allow_shorting: true
  require_borrow_check: true

risk:
  max_position_pct: 0.15        # per symbol
  max_gross_leverage: 2.0
  daily_loss_cut_pct: 0.03
  max_drawdown_cut_pct: 0.20
  max_turnover_pct: 300

accounting:
  pnl_tolerance: 1e-6
  inventory_nonnegative: true   # for long-only; if shorting, enforce borrow

walkforward:
  require_oos: true
  min_oos_days: 30
  min_oos_sharpe: 0.2
  min_oos_winrate: 0.35

observability:
  require_structured_logs: true
  require_run_id: true
  require_config_hash: true

killswitch:
  require_signal_path: "runtime/killswitch.signal"
  require_manual_override_path: "runtime/ALLOW_LIVE.txt"
```

## OOS Summary Schema

The gate expects an OOS summary file with this structure:

```json
{
  "oos_days": 45,
  "oos_sharpe": 0.31,
  "oos_winrate": 0.37,
  "oos_trades": 52,
  "oos_total_pnl": 1234.56,
  "oos_fee_pnl": -78.90,
  "oos_slippage_pnl": -112.34,
  "oos_gross_alpha_pnl": 1425.80
}
```

## Integration with Your Stack

### Replace Placeholders

The gate includes placeholders that should be replaced with your real hooks:

1. **DataSanityValidator** - Already integrated with your existing DataSanity system
2. **Leakage checks** - Implement `check_for_leakage()` function
3. **Backtest realism** - Run tiny smoke backtest with fees/slippage
4. **Accounting consistency** - Implement `verify_accounting_consistency()` function
5. **Walk-forward** - Update `run_walkforward()` to write `latest_oos_summary.json`

### Example Integration

```python
# In your trading pipeline
def run_trading_cycle():
    # Run Go/No-Go gate first
    result = subprocess.run(['python', 'scripts/go_nogo.py'],
                          capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Go/No-Go gate failed")
        logger.error(result.stdout)
        return False

    # Proceed with trading
    logger.info("Go/No-Go gate passed, proceeding with trading")
    # ... rest of trading logic
```

## Kill-Switch Operation

### Manual Override

The gate requires `runtime/ALLOW_LIVE.txt` to exist for live trading:

```bash
# Block live trading
rm runtime/ALLOW_LIVE.txt

# Allow live trading
echo "# Manual Override" > runtime/ALLOW_LIVE.txt
```

### Circuit Breaker

The gate creates `runtime/killswitch.signal` automatically. You can use this file to implement circuit breakers:

```python
# Check for kill-switch signal
if os.path.exists("runtime/killswitch.signal"):
    logger.warning("Kill-switch activated")
    # Stop trading immediately
    return False
```

## Troubleshooting

### Common Issues

1. **Missing environment variables**
   ```bash
   # Set all required variables
   export STRUCTURED_LOGS=1
   export RUN_ID=$(date +%Y%m%d-%H%M%S)
   # ... other variables
   ```

2. **Missing manual override file**
   ```bash
   # Create the override file
   make allow-live
   ```

3. **Config file not found**
   - Ensure one of the expected config files exists
   - Check file permissions

4. **Data files missing**
   - Ensure recent data files are available
   - Check data directory structure

5. **OOS summary missing**
   - Run walk-forward analysis first
   - Check OOS summary file format

### Debug Mode

Run with verbose output to see detailed information:

```bash
python scripts/go_nogo.py --verbose
```

## Best Practices

1. **Always run the gate before live trading**
2. **Set up automated gate checks in CI/CD**
3. **Monitor gate failures and fix root causes**
4. **Keep kill-switch files secure**
5. **Regularly update OOS performance metrics**
6. **Document any gate bypasses**

## Security Considerations

1. **Manual override file should be secure**
2. **Kill-switch should be accessible in emergencies**
3. **Environment variables should be properly managed**
4. **Config hashes should be logged for audit trails**

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run Go/No-Go Gate
  run: |
    export STRUCTURED_LOGS=1
    export RUN_ID=${{ github.run_id }}
    export MAX_POSITION_PCT=0.15
    export MAX_GROSS_LEVERAGE=2.0
    export DAILY_LOSS_CUT_PCT=0.03
    export MAX_DRAWDOWN_CUT_PCT=0.20
    export MAX_TURNOVER_PCT=300
    python scripts/go_nogo.py
```

## Exit Codes

- **0** - All checks passed, proceed with trading
- **2** - Gate blocked, fix issues before proceeding

## Dependencies

- Python 3.8+
- pandas
- pyyaml
- Your existing trading system modules

## Support

For issues with the Go/No-Go gate:

1. Check the troubleshooting section
2. Review the configuration
3. Ensure all dependencies are available
4. Check file permissions and paths
