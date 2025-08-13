# IBKR Integration Setup Guide

This guide will help you set up Interactive Brokers (IBKR) integration for your trading system.

## 📋 Prerequisites

### 1. Interactive Brokers Account
- **Paper Trading Account**: For testing (recommended to start)
- **Live Trading Account**: For actual trading
- **TWS or IB Gateway**: Must be installed and running

### 2. Software Requirements
- **TWS (Trader Workstation)** or **IB Gateway**
- **Python 3.8+**
- **ib_insync library**

## 🚀 Installation Steps

### Step 1: Install IB_insync
```bash
pip install ib_insync>=0.9.85
```

### Step 2: Install TWS or IB Gateway

#### Option A: TWS (Trader Workstation)
1. Download TWS from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/tws.php)
2. Install and launch TWS
3. Log in with your IBKR credentials
4. Enable API connections in TWS

#### Option B: IB Gateway (Recommended)
1. Download IB Gateway from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/ib-api.php)
2. Install and launch IB Gateway
3. Log in with your IBKR credentials
4. IB Gateway is lighter and more suitable for automated trading

### Step 3: Configure TWS/IB Gateway

#### Enable API Connections
1. In TWS: Go to **File > Global Configuration > API > Settings**
2. In IB Gateway: Go to **Configure > API > Settings**
3. Enable **Enable ActiveX and Socket Clients**
4. Set **Socket port** to:
   - **7497** for Paper Trading
   - **7496** for Live Trading
5. Set **Read-Only API** to **No** (if you want to place orders)
6. Add your local IP to **Trusted IPs** if needed

#### API Settings
```
Enable ActiveX and Socket Clients: ✓
Socket port: 7497 (paper) / 7496 (live)
Read-Only API: No
Download open orders on connection: ✓
Include FX positions in portfolio: ✓
Create API message log file: ✓ (for debugging)
```

## ⚙️ Configuration

### 1. Environment Variables (Optional)
Create a `.env` file in your project root:
```bash
# IBKR Configuration
IBKR_PAPER_TRADING=true
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_TIMEOUT=20
IBKR_MAX_RETRIES=3
```

### 2. Configuration File
The system uses `config/enhanced_paper_trading_config.json`:
```json
{
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": true,
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 1,
    "timeout": 20,
    "max_retries": 3
  }
}
```

### 3. Data Configuration
```json
{
  "data_config": {
    "use_cache": true,
    "fallback_to_yfinance": true,
    "cache_dir": "data/ibkr",
    "default_duration": "1 Y",
    "default_bar_size": "1 day"
  }
}
```

## 🧪 Testing the Integration

### 1. Test Connection
```bash
python test_ibkr_integration.py
```

### 2. Test Individual Components
```bash
# Test broker connection
python -c "from brokers.ibkr_broker import test_ibkr_connection; test_ibkr_connection()"

# Test data provider
python -c "from brokers.data_provider import test_data_provider; test_data_provider()"
```

### 3. Test Enhanced System
```bash
python enhanced_paper_trading.py --test-ibkr
```

## 📊 Data Access

### Historical Data
```python
from brokers.data_provider import IBKRDataProvider

# Initialize data provider
provider = IBKRDataProvider(use_cache=True, fallback_to_yfinance=True)

# Get historical data
data = provider.get_historical_data("SPY", "1 Y", "1 day")
print(f"SPY data: {len(data)} rows")
```

### Real-time Data
```python
# Get real-time price
price = provider.get_real_time_price("SPY")
print(f"SPY price: ${price:.2f}")
```

### Multiple Symbols
```python
# Get data for multiple symbols
symbols = ["SPY", "AAPL", "NVDA"]
data_dict = provider.get_multiple_symbols_data(symbols, "1 M", "1 day")

for symbol, data in data_dict.items():
    print(f"{symbol}: {len(data)} rows")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Failed
**Error**: `Failed to connect to IBKR`
**Solution**:
- Ensure TWS/IB Gateway is running
- Check port settings (7497 for paper, 7496 for live)
- Verify API connections are enabled
- Check firewall settings

#### 2. No Data Available
**Error**: `No market data received for SYMBOL`
**Solution**:
- Ensure you have market data subscriptions
- Check symbol format (e.g., "SPY" not "SPY.US")
- Verify market hours
- Try fallback to yfinance

#### 3. Permission Denied
**Error**: `Permission denied for order placement`
**Solution**:
- Check Read-Only API setting
- Verify account permissions
- Ensure paper trading is enabled for testing

#### 4. Timeout Errors
**Error**: `Connection timeout`
**Solution**:
- Increase timeout in configuration
- Check network connectivity
- Restart TWS/IB Gateway

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### API Logs
Check TWS/IB Gateway API logs:
- TWS: **File > Global Configuration > API > Settings > Create API message log file**
- IB Gateway: **Configure > API > Settings > Create API message log file**

## 🔒 Security Considerations

### 1. Paper Trading First
- Always test with paper trading before live trading
- Set `"paper_trading": true` in configuration

### 2. API Security
- Use trusted IP addresses
- Don't share API credentials
- Use read-only mode for data-only access

### 3. Risk Management
- Set appropriate position limits
- Use stop-loss orders
- Monitor account regularly

## 📈 Performance Optimization

### 1. Caching
- Enable data caching for better performance
- Cache historical data to reduce API calls
- Set appropriate cache expiration times

### 2. Connection Management
- Reuse connections when possible
- Implement connection pooling for multiple requests
- Handle disconnections gracefully

### 3. Data Frequency
- Use appropriate bar sizes for your strategy
- Don't request unnecessary data
- Batch requests when possible

## 🚀 Production Deployment

### 1. Live Trading Setup
```json
{
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": false,
    "host": "127.0.0.1",
    "port": 7496,
    "client_id": 1,
    "timeout": 20
  }
}
```

### 2. Monitoring
- Set up alerts for connection issues
- Monitor order execution
- Track performance metrics

### 3. Backup Plans
- Keep yfinance fallback enabled
- Have alternative data sources
- Implement circuit breakers

## 📞 Support

### IBKR Support
- **TWS Help**: Help menu in TWS
- **API Documentation**: [IBKR API Guide](https://interactivebrokers.github.io/tws-api/)
- **Community**: [IBKR Community](https://community.interactivebrokers.com/)

### System Support
- Check logs in `logs/` directory
- Run test scripts for diagnostics
- Review configuration files

## ✅ Checklist

- [ ] IBKR account created
- [ ] TWS/IB Gateway installed and running
- [ ] API connections enabled
- [ ] Port configured correctly
- [ ] ib_insync installed
- [ ] Configuration files set up
- [ ] Connection test passed
- [ ] Data access working
- [ ] Paper trading tested
- [ ] Risk management configured

## 🎯 Next Steps

1. **Test with Paper Trading**: Run the system with paper trading first
2. **Validate Data**: Ensure data quality and consistency
3. **Test Strategies**: Run your trading strategies with IBKR data
4. **Monitor Performance**: Track system performance and reliability
5. **Scale Up**: Gradually increase position sizes and add more symbols

---

**Note**: Always test thoroughly with paper trading before using live trading. The system includes fallback mechanisms to yfinance, but IBKR provides more reliable and comprehensive data for live trading.
