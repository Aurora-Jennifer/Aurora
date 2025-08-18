# IBKR Gateway Setup Guide

## 🔧 **Current Status**
- ✅ IBKR Gateway is running (process found)
- ✅ Port 4002 is open and accepting connections
- ❌ API connections are not properly configured

## 📋 **Step-by-Step IBKR Gateway Configuration**

### **Step 1: Access IBKR Gateway Configuration**

1. **Open IBKR Gateway** (it should already be running)
2. **Click on "Configure"** in the main window
3. **Select "API"** from the left sidebar
4. **Click on "Settings"**

### **Step 2: Enable API Connections**

In the API Settings window, configure the following:

#### **Socket Port Settings:**
- ✅ **Enable ActiveX and Socket Clients**: Check this box
- **Socket port**: Set to `4002` (since this is the port that's open)
- ✅ **Download open orders on connection**: Check this box
- ✅ **Include FX positions in portfolio**: Check this box

#### **API Settings:**
- **Read-Only API**: Set to `No` (if you want to place orders)
- **Create API message log file**: Check this box (for debugging)
- **Log level**: Set to `DETAIL` (for detailed logging)

#### **Trusted IPs (Optional):**
- Add `127.0.0.1` to trusted IPs if needed

### **Step 3: Save and Restart**

1. **Click "OK"** to save the configuration
2. **Restart IBKR Gateway** completely
3. **Log back in** to your IBKR account

### **Step 4: Verify Configuration**

After restarting, check that:
- IBKR Gateway shows "Connected" status
- No error messages in the main window
- API settings are saved correctly

## 🔍 **Alternative Configuration Options**

### **Option 1: Use Standard Ports**
If you prefer to use standard ports, configure IBKR Gateway to use:
- **Port 7497** for Paper Trading
- **Port 7496** for Live Trading

### **Option 2: Use Alternative Ports**
If standard ports don't work, use:
- **Port 4001** for Paper Trading
- **Port 4002** for Live Trading

## 🧪 **Testing the Connection**

### **Test 1: Basic Connection**
```bash
python -c "
from brokers.ibkr_broker import IBKRConfig, IBKRBroker
config = IBKRConfig()
config.port = 4002  # or 7497 for paper trading
broker = IBKRBroker(config=config, auto_connect=True)
print(f'Connected: {broker.is_connected()}')
if broker.is_connected():
    print('✅ IBKR connection successful!')
    account_info = broker.get_account_info()
    print(f'Account Info: {account_info}')
broker.disconnect()
"
```

### **Test 2: Data Provider Test**
```bash
python -c "
from brokers.data_provider import IBKRDataProvider
from brokers.ibkr_broker import IBKRConfig
config = IBKRConfig()
config.port = 4002
provider = IBKRDataProvider(config=config, use_cache=True, fallback_to_yfinance=True)
print(f'IBKR Connected: {provider.is_connected()}')
if provider.is_connected():
    data = provider.get_historical_data('SPY', '1 M', '1 day')
    print(f'SPY Data: {len(data) if data is not None else 0} rows')
else:
    print('Using yfinance fallback')
"
```

### **Test 3: Full Integration Test**
```bash
python test_ibkr_integration.py
```

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: Connection Timeout**
**Symptoms**: Connection times out after 20 seconds
**Solutions**:
- Check that API connections are enabled in IBKR Gateway
- Verify the correct port is configured
- Restart IBKR Gateway after configuration changes
- Check firewall settings

### **Issue 2: Permission Denied**
**Symptoms**: "Permission denied" or "Access denied" errors
**Solutions**:
- Set "Read-Only API" to "No" in IBKR Gateway
- Check account permissions
- Verify you're logged into the correct account

### **Issue 3: Port Already in Use**
**Symptoms**: "Port already in use" errors
**Solutions**:
- Close other instances of IBKR Gateway or TWS
- Use a different port (4001, 4002, 7496, 7497)
- Restart your computer if needed

### **Issue 4: No Market Data**
**Symptoms**: Connected but no market data available
**Solutions**:
- Check market data subscriptions
- Verify market hours
- Ensure you have the necessary permissions

## 📊 **Configuration Examples**

### **Paper Trading Configuration**
```json
{
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": true,
    "host": "127.0.0.1",
    "port": 4002,
    "client_id": 1,
    "timeout": 20,
    "max_retries": 3
  }
}
```

### **Live Trading Configuration**
```json
{
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": false,
    "host": "127.0.0.1",
    "port": 4002,
    "client_id": 1,
    "timeout": 20,
    "max_retries": 3
  }
}
```

## 🔒 **Security Considerations**

### **API Security**
- Use trusted IP addresses when possible
- Don't share API credentials
- Use read-only mode for data-only access
- Monitor API usage regularly

### **Account Security**
- Use paper trading for testing
- Set appropriate position limits
- Monitor account activity
- Use strong passwords

## 📞 **Getting Help**

### **IBKR Support**
- **IBKR Help**: Help menu in IBKR Gateway
- **API Documentation**: https://interactivebrokers.github.io/tws-api/
- **Community**: https://community.interactivebrokers.com/

### **System Support**
- Check logs in `logs/` directory
- Run diagnostic tools: `python diagnose_ibkr.py`
- Review configuration files
- Test with simple scripts first

## ✅ **Checklist**

- [ ] IBKR Gateway is running
- [ ] API connections are enabled
- [ ] Correct port is configured (4002)
- [ ] Read-Only API is set to "No" (if placing orders)
- [ ] IBKR Gateway is restarted after configuration
- [ ] Connection test passes
- [ ] Data provider test passes
- [ ] Full integration test passes

## 🎯 **Next Steps**

1. **Configure IBKR Gateway** using the steps above
2. **Test the connection** with the provided scripts
3. **Run your enhanced trading system** with IBKR data
4. **Monitor performance** and data quality
5. **Scale up** gradually as confidence builds

---

**Note**: The diagnostic shows port 4002 is open, so your IBKR Gateway is running but needs API configuration. Follow the steps above to enable API connections and test the integration.
