# Capital Scaling Guide

## Overview

The Aurora Trading System implements advanced capital scaling to maximize capital utilization while maintaining risk controls. This guide explains the capital scaling implementation and configuration.

## 🎯 Capital Scaling Features

### 2x Position Scaling
- **Capital Utilization Factor**: 2.0 (configurable)
- **Position Sizing**: All intended position values are scaled by the factor
- **Risk Preservation**: Maintains existing risk ratios while increasing absolute position sizes

### Advanced Position Sizing
- **Order Notional Cap**: 15,000 per order
- **Max Position Size**: 15% of portfolio per symbol
- **Capital Deployment**: ~$44,000 of $195,000 available capital

## 📊 Configuration

### Position Sizing Configuration
```yaml
# config/execution.yaml
position_sizing:
  capital_utilization_factor: 2.0  # Scale positions by 2x
  order_notional_cap: 15000.0      # 15k per order
  max_position_size: 0.15          # 15% per symbol
  max_total_exposure: 0.8          # 80% total exposure
  min_trade_size: 25.0             # $25 minimum
  max_trade_size: 25000.0          # $25k maximum
  signal_threshold: 0.02           # 2% signal threshold
  volatility_adjustment: true      # Volatility adjustment
  portfolio_heat: 0.02             # 2% portfolio heat
  min_shares: 1                    # Minimum 1 share
```

### Risk Management Configuration
```yaml
risk_management:
  max_pos_pct: 0.15                # 15% per symbol
  max_order_notional: 15000        # 15k per order
  max_notional_opener: 15000       # 15k for openers
  max_notional_reducer: 15000      # 15k for reducers
  min_notional_opener: 10          # $10 minimum opener
  min_notional_reducer: 10         # $10 minimum reducer
  allow_dust_close: true           # Allow small closes
```

## 🔧 Implementation Details

### Position Sizing Engine
```python
# core/execution/position_sizing.py
def compute_size(self, symbol: str, target_weight: float, price: float,
                 portfolio_value: float, min_notional: float,
                 max_order_notional: float, 
                 capital_utilization_factor: float = 1.0) -> Optional[SizeDecision]:
    
    # Apply capital scaling
    intended_val = abs(target_weight) * portfolio_value * capital_utilization_factor
    capped_val = min(intended_val, max_order_notional)
    
    # Calculate quantity
    qty = int(capped_val // price)
    
    return SizeDecision(qty=qty, ref_price=price, notional=abs(qty) * price)
```

### Execution Engine Integration
```python
# core/execution/execution_engine.py
# Get capital utilization factor from config
capital_utilization_factor = getattr(self.config, 'capital_utilization_factor', 1.0)

size_decision = self.position_sizer.compute_size(
    symbol=symbol,
    target_weight=signal,
    price=ref_price,
    portfolio_value=portfolio_metrics.total_value,
    min_notional=min_notional,
    max_order_notional=max_order_notional,
    capital_utilization_factor=capital_utilization_factor
)
```

## 📈 Performance Impact

### Before Capital Scaling (1x)
- **Net Position Value**: ~$14,000
- **Capital Utilization**: ~7% of available capital
- **Order Sizes**: 500-5,000 per order
- **Position Concentration**: Low (diversified)

### After Capital Scaling (2x)
- **Net Position Value**: ~$44,000
- **Capital Utilization**: ~22% of available capital
- **Order Sizes**: Up to 15,000 per order
- **Position Concentration**: Higher (more concentrated)

### Capital Efficiency Gains
- **3x More Capital Deployed**: From ~$14k to ~$44k
- **Larger P&L Movements**: Position adjustments are more significant
- **Better Capital Utilization**: Using more of available capital
- **Maintained Risk Ratios**: Same percentage allocations, larger absolute sizes

## 🛡️ Risk Management

### Risk Controls Maintained
- **Position Limits**: 15% per symbol maximum
- **Total Exposure**: 80% maximum portfolio exposure
- **Order Limits**: 15k per order maximum
- **Signal Thresholds**: 2% minimum signal strength

### Two-Phase Batching
- **Phase A**: Reducers first (free up capital)
- **Phase B**: Openers within budget
- **Self-Funding**: System funds itself through position reductions

### Pre-Trade Gate
- **Price Sanity**: NBBO mid or last trade validation
- **Notional Limits**: Dynamic limits based on order intent
- **Symbol Caps**: Per-symbol position limits
- **Risk Checks**: Comprehensive risk validation

## 📊 Monitoring

### Key Metrics to Watch
```bash
# Check position sizes
journalctl --user -u paper-trading-session.service -n 50 | grep "SizeDecision.*TARGET"

# Monitor capital utilization
journalctl --user -u paper-trading-session.service -n 50 | grep "net_position_value"

# Check order sizes
journalctl --user -u paper-trading-session.service -n 50 | grep "notional="
```

### Expected Log Output
```
SizeDecision: AAPL qty=-22 (TARGET), notional=4989.60, intended_val=41819.22
SizeDecision: MSFT qty=-9 (TARGET), notional=4506.89, intended_val=72363.85
SizeDecision: GOOGL qty=-20 (TARGET), notional=4787.60, intended_val=61075.15
```

## ⚙️ Configuration Tuning

### Increasing Capital Utilization
```yaml
# For more aggressive capital use
position_sizing:
  capital_utilization_factor: 3.0  # 3x scaling
  order_notional_cap: 20000.0      # 20k per order
  max_position_size: 0.20          # 20% per symbol

risk_management:
  max_pos_pct: 0.20                # 20% per symbol
  max_order_notional: 20000        # 20k per order
```

### Conservative Scaling
```yaml
# For more conservative approach
position_sizing:
  capital_utilization_factor: 1.5  # 1.5x scaling
  order_notional_cap: 10000.0      # 10k per order
  max_position_size: 0.10          # 10% per symbol
```

## 🚨 Troubleshooting

### Common Issues

#### Position Sizes Too Small
- Check `capital_utilization_factor` in config
- Verify `order_notional_cap` settings
- Ensure signal strength meets threshold

#### Orders Rejected by Gate
- Check `max_pos_pct` vs `max_position_size`
- Verify notional limits in risk management
- Check price cap constraints

#### Capital Not Being Used
- Verify `capital_utilization_factor > 1.0`
- Check signal strength and thresholds
- Ensure positions are not at target already

### Debug Commands
```bash
# Check current configuration
grep -A 20 "position_sizing:" config/execution.yaml

# Monitor position sizing decisions
journalctl --user -u paper-trading-session.service -f | grep "SizeDecision"

# Check capital utilization
journalctl --user -u paper-trading-session.service -n 100 | grep "net_position_value"
```

## 📚 Related Documentation

- **[Execution System Status](execution_system_final_status.md)** - Current execution engine status
- **[Risk Management Guide](RISK_MANAGEMENT_GUIDE.md)** - Comprehensive risk controls
- **[Position Sizing Guide](POSITION_SIZING_GUIDE.md)** - Detailed position sizing logic
- **[Systemd Automation Guide](SYSTEMD_AUTOMATION_GUIDE.md)** - Automation setup

## 🎯 Best Practices

1. **Start Conservative**: Begin with 1.5x scaling and increase gradually
2. **Monitor Closely**: Watch position sizes and capital utilization
3. **Test Changes**: Use paper trading to validate configuration changes
4. **Risk First**: Always maintain proper risk controls
5. **Document Changes**: Keep track of configuration modifications

## 🚀 Future Enhancements

- **Dynamic Scaling**: Adjust scaling based on market conditions
- **Portfolio Optimization**: Advanced portfolio-level position sizing
- **Risk-Adjusted Scaling**: Scale based on risk metrics
- **Multi-Asset Scaling**: Different scaling factors per asset class
