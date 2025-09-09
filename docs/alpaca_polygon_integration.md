# Alpaca Integration Guide (Paper Trading Ready!)

## 🎯 ALPACA-ONLY SETUP (PERFECT FOR PAPER TRADING)

**Alpaca provides everything you need for professional paper trading validation:**
- **Alpaca:** Paper trading + real-time market data + basic corporate actions (unified API)
- **Future enhancement:** Add Polygon later for advanced corporate actions if needed

## 📋 IMPLEMENTATION CHECKLIST

### 1. 🔑 API KEYS SETUP

**Alpaca Account:**
```bash
# Sign up at https://alpaca.markets/
# Get paper trading account (free)
export ALPACA_API_KEY="PKQ9ZKNTB5HV9SNQ929E"
export ALPACA_SECRET_KEY="HaZ9FkKaXJdK1HFxp6Vr3449nMXUgPWvbyZhMpPn"  
export ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2"  # Paper trading endpoint
```

**Environment Setup:**
```bash
# Set these in your shell environment
export IS_PAPER_TRADING=true
export PYTHONPATH="/home/Jennifer/secure/trader"
```

### 2. 📦 INSTALL DEPENDENCIES

```bash
pip install alpaca-trade-api pandas numpy requests
```

### 3. 🔌 DATA PROVIDER IMPLEMENTATION

**Create: `ml/alpaca_data_provider.py`**
```python
import alpaca_trade_api as tradeapi
from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta

class AlpacaPolygonProvider:
    def __init__(self):
        # Alpaca for market data + paper trading
        self.alpaca = tradeapi.REST(
            key_id=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY'],
            base_url=os.environ['ALPACA_BASE_URL']
        )
        
        # Polygon for corporate actions
        self.polygon = RESTClient(os.environ['POLYGON_API_KEY'])
    
    def get_daily_bars(self, symbols, start_date, end_date):
        # Alpaca market data API
        bars = self.alpaca.get_bars(
            symbols,
            timeframe='1Day',
            start=start_date,
            end=end_date,
            adjustment='raw'  # Handle adjustments separately
        ).df
        return bars
    
    def get_corporate_actions(self, symbol, start_date, end_date):
        # Polygon corporate actions
        splits = self.polygon.list_splits(
            ticker=symbol,
            execution_date_gte=start_date,
            execution_date_lte=end_date
        )
        
        dividends = self.polygon.list_dividends(
            ticker=symbol,
            ex_dividend_date_gte=start_date,
            ex_dividend_date_lte=end_date  
        )
        
        return {'splits': splits, 'dividends': dividends}
```

### 4. 🔄 UPDATE EXISTING FILES

**Update: `ml/panel_builder.py`**
```python
# Replace mock data creation with:
from ml.alpaca_data_provider import AlpacaPolygonProvider

def load_real_market_data(symbols, start_date, end_date):
    provider = AlpacaPolygonProvider()
    
    # Get raw market data
    raw_data = provider.get_daily_bars(symbols, start_date, end_date)
    
    # Apply corporate actions adjustments
    adjusted_data = apply_corporate_actions(raw_data, provider)
    
    return adjusted_data
```

**Update: `ops/pre_market_dry_run.py`**
```python
# Replace create_mock_market_data() with:
def get_latest_market_data(symbols):
    provider = AlpacaPolygonProvider()
    
    # Get last 60 days for feature calculation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    return provider.get_daily_bars(symbols, start_date, end_date)
```

### 5. 📊 DATA QUALITY CHECKS

**Add: `ml/data_quality.py`**
```python
def validate_market_data(data):
    checks = {
        'completeness': check_missing_dates(data),
        'price_sanity': check_price_ranges(data),
        'volume_sanity': check_volume_ranges(data),
        'corporate_actions': check_split_adjustments(data),
        'survivorship': check_delisted_symbols(data)
    }
    return checks

def check_missing_dates(data):
    # Ensure no missing trading days
    pass

def check_split_adjustments(data):
    # Verify splits are properly applied
    pass
```

### 6. 🕐 NIGHTLY CORPORATE ACTIONS JOB

**Create: `ops/nightly_corp_actions.py`**
```python
def update_corporate_actions():
    """Run nightly to update splits/dividends."""
    provider = AlpacaPolygonProvider()
    
    # Get yesterday's corporate actions
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    for symbol in get_active_universe():
        actions = provider.get_corporate_actions(symbol, yesterday, yesterday)
        
        if actions['splits']:
            apply_split_adjustments(symbol, actions['splits'])
            
        if actions['dividends']:
            apply_dividend_adjustments(symbol, actions['dividends'])
```

**Add to cron:**
```bash
# Add to crontab for nightly run (01:00 CT = 07:00 UTC)
0 7 * * 1-6 /home/Jennifer/secure/trader/ops/nightly_corp_actions.py
```

### 7. 🔍 PAPER TRADING INTEGRATION

**Create: `ml/alpaca_paper_trader.py`**
```python
class AlpacaPaperTrader:
    def __init__(self):
        self.alpaca = tradeapi.REST(...)  # Paper account
    
    def submit_orders(self, target_weights):
        """Submit paper trading orders."""
        for symbol, weight in target_weights.items():
            # Calculate position size
            equity = self.alpaca.get_account().equity
            target_dollars = float(equity) * weight
            
            # Submit paper order
            order = self.alpaca.submit_order(
                symbol=symbol,
                notional=target_dollars,
                side='buy' if weight > 0 else 'sell',
                type='market',
                time_in_force='day'
            )
    
    def get_positions(self):
        """Get current paper positions."""
        return self.alpaca.list_positions()
    
    def get_performance(self):
        """Get paper trading performance."""
        account = self.alpaca.get_account()
        return {
            'equity': float(account.equity),
            'pnl': float(account.unrealized_pl),
            'buying_power': float(account.buying_power)
        }
```

### 8. 📈 REAL-TIME MONITORING

**Create: `ops/real_time_monitor.py`**
```python
def monitor_paper_trading():
    trader = AlpacaPaperTrader()
    
    # Check positions
    positions = trader.get_positions()
    
    # Check performance
    performance = trader.get_performance()
    
    # Log to daily report
    log_performance_metrics(performance)
    
    # Check kill conditions
    daily_pnl_pct = performance['pnl'] / performance['equity']
    if daily_pnl_pct <= -0.02:  # -2% daily loss
        trigger_emergency_halt("Daily loss limit breached")
```

## 🧪 TESTING YOUR INTEGRATION

### Integration Test Script:
```bash
# Test data fetching
python -c "
from ml.alpaca_data_provider import AlpacaPolygonProvider
provider = AlpacaPolygonProvider()
data = provider.get_daily_bars(['AAPL', 'MSFT'], '2025-08-01', '2025-09-01')
print(f'Fetched {len(data)} bars')
print(data.head())
"

# Test paper trading
python -c "
from ml.alpaca_paper_trader import AlpacaPaperTrader  
trader = AlpacaPaperTrader()
account = trader.alpaca.get_account()
print(f'Paper account equity: ${account.equity}')
"
```

## 🚨 PRODUCTION DEPLOYMENT

### 1. Environment Variables:
```bash
# Add to ~/.bashrc or cron environment
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"  
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
export POLYGON_API_KEY="your_polygon_key"
export IS_PAPER_TRADING=true
```

### 2. Update Daily Script:
```bash
# Modify daily_paper_trading.sh to use real data
# Replace mock data calls with Alpaca/Polygon calls
```

### 3. Monitoring:
```bash
# Add data quality checks to preflight
# Monitor API rate limits and failures
# Alert on stale data or missing corporate actions
```

## 🎯 SUCCESS METRICS

**Data Quality Gates:**
- ✅ Data freshness < 4 hours during trading days
- ✅ No missing dates in trading calendar
- ✅ Corporate actions applied within 24h
- ✅ API uptime > 99%

**Paper Trading Validation:**
- ✅ Orders filled within 1 second
- ✅ Position reconciliation accurate
- ✅ Performance tracking aligned with backtest
- ✅ No API errors during trading hours

## 🔄 MIGRATION TO LIVE TRADING

When ready for live trading:

1. **Keep Polygon** for corporate actions (same API)
2. **Switch Alpaca endpoint** from paper to live
3. **OR migrate to IBKR** for execution while keeping Alpaca/Polygon for data

Your data infrastructure will be production-ready and easily portable! 🚀
