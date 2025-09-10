# 🔧 Systemd Integration with Execution Infrastructure

## 📋 **Current Systemd Architecture**

Your current systemd setup provides automated daily paper trading with these services:

### **Existing Services**
```
paper-trading-preflight.service    # 08:00 CT - Pre-market checks
paper-trading-session.service      # 08:30 CT - Trading session  
paper-trading-eod.service          # 15:10 CT - End-of-day operations
```

### **Current Flow**
```
08:00 CT → Preflight checks → Validate system readiness
08:30 CT → Trading session → Generate signals & mock execution
15:10 CT → EOD operations → Reports & reconciliation
```

## 🚀 **Updated Architecture with Execution Infrastructure**

### **Enhanced Services**
The execution infrastructure integrates seamlessly with your existing systemd services:

```
paper-trading-preflight.service    # 08:00 CT - Pre-market checks + execution validation
paper-trading-session.service      # 08:30 CT - Trading session + REAL order execution
paper-trading-eod.service          # 15:10 CT - EOD operations + portfolio reconciliation
```

### **New Execution Flow**
```
08:00 CT → Preflight checks → Validate system + Alpaca connection + execution readiness
08:30 CT → Trading session → Generate signals → Position sizing → Risk checks → REAL orders
15:10 CT → EOD operations → Portfolio reconciliation + execution reports
```

## 🔄 **Integration Options**

### **Option 1: Replace Existing Script (Recommended)**
Update your existing systemd services to use the new execution-enabled script:

```bash
# Current
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading.py --mode trading

# Updated with execution
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading_with_execution.py --mode trading
```

### **Option 2: Add New Execution Services**
Create additional systemd services specifically for execution monitoring:

```bash
# New execution monitoring service
paper-trading-execution-monitor.service  # Continuous execution monitoring
paper-trading-execution-reconcile.service # Periodic order reconciliation
```

### **Option 3: Hybrid Approach**
Keep existing services for signal generation, add execution services for order management.

## 🛠️ **Implementation Plan**

### **Step 1: Update Existing Services**
Modify your current systemd services to use the execution-enabled script:

```bash
# Update the automation setup script
./ops/setup_paper_trading_automation_with_execution.sh
```

### **Step 2: Add Execution Configuration**
Ensure execution configuration is loaded in systemd services:

```bash
# Add to systemd service environment
Environment=ALPACA_API_KEY=${ALPACA_API_KEY}
Environment=ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
Environment=EXECUTION_MODE=paper
```

### **Step 3: Enhanced Monitoring**
Add execution-specific monitoring and alerting:

```bash
# Enhanced status checking
./status_paper_trading_with_execution.sh
```

## 📊 **Service Dependencies & Timing**

### **Preflight Service (08:00 CT)**
```ini
[Unit]
Description=Paper Trading Preflight + Execution Validation
After=network.target
Requires=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 ops/daily_paper_trading_with_execution.py --mode preflight
Environment=ALPACA_API_KEY=${ALPACA_API_KEY}
Environment=ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
Environment=EXECUTION_MODE=paper
```

**What it does:**
- ✅ System readiness checks
- ✅ Model loading validation  
- ✅ Alpaca connection verification
- ✅ Execution engine initialization
- ✅ Market data availability
- ✅ Risk limits validation

### **Trading Session Service (08:30 CT)**
```ini
[Unit]
Description=Paper Trading Session + Real Order Execution
After=network.target paper-trading-preflight.service
Requires=paper-trading-preflight.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 ops/daily_paper_trading_with_execution.py --mode trading
Restart=on-failure
RestartSec=30
Environment=ALPACA_API_KEY=${ALPACA_API_KEY}
Environment=ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
Environment=EXECUTION_MODE=paper
```

**What it does:**
- ✅ Generate XGBoost trading signals
- ✅ Calculate position sizes with risk adjustments
- ✅ Execute real orders on Alpaca (paper trading)
- ✅ Monitor order fills and portfolio updates
- ✅ Enforce risk limits and safety checks
- ✅ Real-time execution monitoring

### **EOD Service (15:10 CT)**
```ini
[Unit]
Description=Paper Trading EOD + Portfolio Reconciliation
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 ops/daily_paper_trading_with_execution.py --mode eod
Environment=ALPACA_API_KEY=${ALPACA_API_KEY}
Environment=ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
Environment=EXECUTION_MODE=paper
```

**What it does:**
- ✅ Final order reconciliation
- ✅ Portfolio position updates
- ✅ P&L calculation and reporting
- ✅ Risk metrics summary
- ✅ Execution performance analysis
- ✅ Daily reports generation

## 🔧 **Updated Setup Script**

Here's the updated automation setup script that integrates execution:

```bash
#!/bin/bash
# Paper Trading Automation Setup with Execution Infrastructure

# ... existing setup code ...

# Update service definitions to use execution-enabled script
ExecStart=/usr/bin/python3 $PROJECT_ROOT/ops/daily_paper_trading_with_execution.py --mode trading

# Add execution environment variables
Environment=ALPACA_API_KEY=${ALPACA_API_KEY}
Environment=ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
Environment=EXECUTION_MODE=paper
Environment=EXECUTION_CONFIG_PATH=$PROJECT_ROOT/config/execution.yaml
```

## 📈 **Enhanced Monitoring & Alerting**

### **Execution-Specific Monitoring**
```bash
# Enhanced status script
./status_paper_trading_with_execution.sh

# Shows:
# - Systemd service status
# - Execution engine health
# - Order execution statistics
# - Portfolio performance
# - Risk limit compliance
# - Alpaca connection status
```

### **Real-time Execution Monitoring**
```bash
# Monitor execution in real-time
journalctl --user -u paper-trading-session.service -f

# Filter for execution events
journalctl --user -u paper-trading-session.service | grep -E "(EXECUTION|ORDER|RISK|PORTFOLIO)"
```

## 🚨 **Safety & Emergency Procedures**

### **Emergency Stop Procedures**
```bash
# Stop all trading immediately
./stop_paper_trading.sh

# Emergency stop execution engine
systemctl --user stop paper-trading-session.service

# Cancel all pending orders (via execution engine)
python3 ops/daily_paper_trading_with_execution.py --emergency-stop
```

### **Execution Safety Features**
- ✅ **Paper trading only** (hardcoded in systemd services)
- ✅ **Risk limit enforcement** before every order
- ✅ **Emergency stop mechanisms** via systemd
- ✅ **Order reconciliation** and validation
- ✅ **Portfolio monitoring** and alerts
- ✅ **Complete audit trails** in systemd logs

## 📋 **Migration Steps**

### **1. Backup Current Setup**
```bash
# Backup existing systemd services
cp -r ~/.config/systemd/user/paper-trading-* ~/.config/systemd/user/backup/
```

### **2. Update Services**
```bash
# Run updated setup script
./ops/setup_paper_trading_automation_with_execution.sh
```

### **3. Configure Execution**
```bash
# Set up Alpaca credentials
cp config/alpaca_credentials.yaml.example config/alpaca_credentials.yaml
# Edit with your credentials

# Configure execution parameters
# Edit config/execution.yaml as needed
```

### **4. Test Integration**
```bash
# Test preflight with execution
./run_paper_trading_now.sh preflight

# Test trading session with execution
./run_paper_trading_now.sh trading
```

### **5. Start Enhanced Automation**
```bash
# Start updated automation
./start_paper_trading.sh

# Monitor execution
./status_paper_trading_with_execution.sh
```

## 🎯 **Benefits of Systemd Integration**

### **Automated Execution**
- ✅ **Scheduled order execution** at market open
- ✅ **Continuous monitoring** throughout trading day
- ✅ **Automatic reconciliation** at market close
- ✅ **Systemd restart policies** for fault tolerance

### **Production-Ready Operations**
- ✅ **Service dependencies** ensure proper startup order
- ✅ **Logging integration** with systemd journal
- ✅ **Resource management** and limits
- ✅ **User isolation** and security
- ✅ **Automatic recovery** from failures

### **Monitoring & Alerting**
- ✅ **Centralized logging** via systemd journal
- ✅ **Service health monitoring** via systemctl
- ✅ **Timer status tracking** for schedule compliance
- ✅ **Performance metrics** and reporting

## 🚀 **Ready for Production**

Your systemd integration with execution infrastructure provides:

1. **Automated daily trading** with real order execution
2. **Production-grade reliability** with systemd service management
3. **Comprehensive monitoring** and alerting
4. **Safety mechanisms** and emergency procedures
5. **Complete audit trails** and compliance reporting

**The execution infrastructure seamlessly integrates with your existing systemd automation! 🎉**
