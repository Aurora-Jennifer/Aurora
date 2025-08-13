# 🔧 End-to-End Testing & Automation Guide

## 🎯 **How to Ensure End-to-End Functionality**

### **1. Complete System Validation**

#### **Step 1: Verify All Components**
```bash
# Test IBKR connection
python test_ibkr_connection.py

# Test data provider
python -c "
from brokers.data_provider import IBKRDataProvider
from brokers.ibkr_broker import IBKRConfig
config = IBKRConfig()
provider = IBKRDataProvider(config)
data = provider.get_daily_data('SPY', '2025-08-01', '2025-08-13')
print(f'✅ Data Provider: {len(data)} rows retrieved')
"

# Test strategy generation
python -c "
from strategies.regime_aware_ensemble import RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams
import pandas as pd
import yfinance as yf

# Get test data
spy = yf.download('SPY', start='2025-07-01', end='2025-08-13')
params = RegimeAwareEnsembleParams()
strategy = RegimeAwareEnsembleStrategy(params)
signals = strategy.generate_signals(spy)
print(f'✅ Strategy: {len(signals)} signals generated')
print(f'Signal range: {signals.min():.3f} to {signals.max():.3f}')
"
```

#### **Step 2: Full System Test**
```bash
# Run complete end-to-end test
python enhanced_paper_trading.py --daily --verbose

# Check all outputs
echo "=== Trade Log ==="
cat logs/trades/trades_$(date +%Y-%m).log

echo "=== Performance Log ==="
cat logs/performance/performance_$(date +%Y-%m).log

echo "=== System Log ==="
tail -20 logs/trading_bot.log

echo "=== Performance Report ==="
cat results/performance_report.json
```

#### **Step 3: Validation Checklist**
```bash
# Create validation script
cat > validate_system.py << 'EOF'
#!/usr/bin/env python3
"""
System Validation Script
"""

import os
import json
import pandas as pd
from datetime import datetime

def validate_system():
    """Validate all system components."""
    print("🔍 System Validation Report")
    print("=" * 50)
    
    # Check IBKR connection
    try:
        from brokers.ibkr_broker import IBKRConfig, IBKRBroker
        config = IBKRConfig()
        config.port = 7497
        config.client_id = 12399
        broker = IBKRBroker(config=config, auto_connect=True)
        if broker.is_connected():
            print("✅ IBKR Connection: Working")
        else:
            print("❌ IBKR Connection: Failed")
        broker.disconnect()
    except Exception as e:
        print(f"❌ IBKR Connection: Error - {e}")
    
    # Check data provider
    try:
        from brokers.data_provider import IBKRDataProvider
        provider = IBKRDataProvider()
        data = provider.get_daily_data('SPY', '2025-08-01', '2025-08-13')
        if data is not None and len(data) > 0:
            print("✅ Data Provider: Working")
        else:
            print("❌ Data Provider: No data")
    except Exception as e:
        print(f"❌ Data Provider: Error - {e}")
    
    # Check strategy
    try:
        from strategies.regime_aware_ensemble import RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams
        import yfinance as yf
        spy = yf.download('SPY', start='2025-07-01', end='2025-08-13')
        params = RegimeAwareEnsembleParams()
        strategy = RegimeAwareEnsembleStrategy(params)
        signals = strategy.generate_signals(spy)
        if len(signals) > 0:
            print("✅ Strategy: Working")
        else:
            print("❌ Strategy: No signals")
    except Exception as e:
        print(f"❌ Strategy: Error - {e}")
    
    # Check logs
    log_files = [
        'logs/trading_bot.log',
        'logs/trades/trades_2025-08.log',
        'logs/performance/performance_2025-08.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"✅ Log File: {log_file}")
        else:
            print(f"❌ Log File: {log_file} - Missing")
    
    # Check results
    result_files = [
        'results/performance_report.json',
        'results/trade_history.csv',
        'results/daily_returns.csv'
    ]
    
    for result_file in result_files:
        if os.path.exists(result_file):
            print(f"✅ Result File: {result_file}")
        else:
            print(f"❌ Result File: {result_file} - Missing")
    
    print("=" * 50)
    print("Validation Complete!")

if __name__ == "__main__":
    validate_system()
EOF

# Run validation
python validate_system.py
```

---

## 🤖 **Setting Up Cron Automation**

### **1. Create Systemd Service (Recommended)**

#### **Create Service File**
```bash
sudo tee /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=Trading Bot Service
After=network.target

[Service]
Type=simple
User=Jennifer
WorkingDirectory=/home/Jennifer/projects/trader
Environment=PATH=/home/Jennifer/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/Jennifer/miniconda3/bin/python /home/Jennifer/projects/trader/enhanced_paper_trading.py --daily
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

#### **Enable and Start Service**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable trading-bot.service

# Start service
sudo systemctl start trading-bot.service

# Check status
sudo systemctl status trading-bot.service

# View logs
sudo journalctl -u trading-bot.service -f
```

### **2. Alternative: Traditional Cron Job**

#### **Create Cron Script**
```bash
# Create cron script
cat > /home/Jennifer/projects/trader/run_trading_cron.sh << 'EOF'
#!/bin/bash

# Set environment
export PATH="/home/Jennifer/miniconda3/bin:$PATH"
cd /home/Jennifer/projects/trader

# Run trading system
python enhanced_paper_trading.py --daily >> logs/cron.log 2>&1

# Send notification (optional)
if [ $? -eq 0 ]; then
    echo "Trading completed successfully at $(date)" >> logs/cron.log
else
    echo "Trading failed at $(date)" >> logs/cron.log
fi
EOF

# Make executable
chmod +x /home/Jennifer/projects/trader/run_trading_cron.sh

# Add to crontab (runs at 9:30 AM ET daily)
(crontab -l 2>/dev/null; echo "30 9 * * 1-5 /home/Jennifer/projects/trader/run_trading_cron.sh") | crontab -

# Check crontab
crontab -l
```

### **3. Health Check Script**

#### **Create Health Monitor**
```bash
cat > /home/Jennifer/projects/trader/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Health Check Script for Trading Bot
"""

import os
import json
import subprocess
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

def check_service_health():
    """Check if trading bot service is healthy."""
    
    # Check if service is running
    try:
        result = subprocess.run(['systemctl', 'is-active', 'trading-bot.service'], 
                              capture_output=True, text=True)
        if result.stdout.strip() == 'active':
            print("✅ Service Status: Running")
            return True
        else:
            print("❌ Service Status: Not running")
            return False
    except Exception as e:
        print(f"❌ Service Check Error: {e}")
        return False

def check_recent_logs():
    """Check for recent log activity."""
    
    log_file = f"logs/trading_bot.log"
    if not os.path.exists(log_file):
        print("❌ Log file not found")
        return False
    
    # Check if logs are recent (within last hour)
    stat = os.stat(log_file)
    last_modified = datetime.fromtimestamp(stat.st_mtime)
    if datetime.now() - last_modified < timedelta(hours=1):
        print("✅ Recent log activity found")
        return True
    else:
        print("❌ No recent log activity")
        return False

def check_performance():
    """Check recent performance."""
    
    perf_file = "results/performance_report.json"
    if not os.path.exists(perf_file):
        print("❌ Performance file not found")
        return False
    
    try:
        with open(perf_file, 'r') as f:
            data = json.load(f)
        
        total_return = data.get('total_return', 0)
        if total_return > -0.5:  # Less than 50% loss
            print(f"✅ Performance OK: {total_return:.1%}")
            return True
        else:
            print(f"❌ Poor Performance: {total_return:.1%}")
            return False
    except Exception as e:
        print(f"❌ Performance Check Error: {e}")
        return False

def send_alert(message):
    """Send alert notification."""
    # Configure your email settings here
    print(f"ALERT: {message}")
    # Add email sending logic if needed

def main():
    """Main health check."""
    print("🔍 Trading Bot Health Check")
    print("=" * 40)
    
    checks = [
        check_service_health(),
        check_recent_logs(),
        check_performance()
    ]
    
    if all(checks):
        print("✅ All health checks passed")
    else:
        print("❌ Health check failed")
        send_alert("Trading bot health check failed")
    
    print("=" * 40)

if __name__ == "__main__":
    main()
EOF

# Make executable
chmod +x /home/Jennifer/projects/trader/health_check.py

# Add health check to crontab (runs every 30 minutes)
(crontab -l 2>/dev/null; echo "*/30 * * * * /home/Jennifer/projects/trader/health_check.py") | crontab -
```

---

## 📊 **User-Friendly Monitoring Dashboard**

### **1. Create Web Dashboard**

#### **Install Dependencies**
```bash
pip install flask plotly dash pandas
```

#### **Create Dashboard**
```bash
cat > /home/Jennifer/projects/trader/dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Trading Bot Dashboard
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px

app = Flask(__name__)

def load_performance_data():
    """Load performance data."""
    try:
        with open('results/performance_report.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def load_trade_history():
    """Load trade history."""
    try:
        return pd.read_csv('results/trade_history.csv')
    except:
        return pd.DataFrame()

def load_daily_returns():
    """Load daily returns."""
    try:
        return pd.read_csv('results/daily_returns.csv')
    except:
        return pd.DataFrame()

def create_performance_chart():
    """Create performance chart."""
    returns_df = load_daily_returns()
    if returns_df.empty:
        return None
    
    # Create cumulative returns
    returns_df['cumulative_return'] = (1 + returns_df['return']).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns_df['date'],
        y=returns_df['cumulative_return'],
        mode='lines',
        name='Cumulative Return',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_trade_chart():
    """Create trade chart."""
    trades_df = load_trade_history()
    if trades_df.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=trades_df['pnl'],
        mode='markers',
        name='Trade PnL',
        marker=dict(
            color=trades_df['pnl'].apply(lambda x: 'green' if x > 0 else 'red'),
            size=8
        )
    ))
    
    fig.update_layout(
        title='Trade PnL',
        xaxis_title='Date',
        yaxis_title='PnL',
        template='plotly_white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def dashboard():
    """Main dashboard page."""
    perf_data = load_performance_data()
    trades_df = load_trade_history()
    returns_df = load_daily_returns()
    
    # Calculate metrics
    total_trades = len(trades_df) if not trades_df.empty else 0
    win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
    
    return render_template('dashboard.html',
                         performance=perf_data,
                         total_trades=total_trades,
                         win_rate=win_rate,
                         total_pnl=total_pnl)

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance data."""
    return jsonify(load_performance_data())

@app.route('/api/trades')
def api_trades():
    """API endpoint for trade data."""
    trades_df = load_trade_history()
    if not trades_df.empty:
        return jsonify(trades_df.to_dict('records'))
    return jsonify([])

@app.route('/api/charts/performance')
def api_performance_chart():
    """API endpoint for performance chart."""
    chart_json = create_performance_chart()
    return chart_json if chart_json else jsonify({})

@app.route('/api/charts/trades')
def api_trade_chart():
    """API endpoint for trade chart."""
    chart_json = create_trade_chart()
    return chart_json if chart_json else jsonify({})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF
```

#### **Create HTML Template**
```bash
mkdir -p templates
cat > templates/dashboard.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Trading Bot Dashboard</h1>
            <p>Real-time monitoring of your automated trading system</p>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="metric-value" id="total-return">Loading...</div>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="metric-value" id="sharpe-ratio">Loading...</div>
            </div>
            <div class="metric-card">
                <h3>Total Trades</h3>
                <div class="metric-value" id="total-trades">Loading...</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="metric-value" id="win-rate">Loading...</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Portfolio Performance</h2>
            <div id="performance-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Trade PnL</h2>
            <div id="trade-chart"></div>
        </div>
    </div>
    
    <script>
        function refreshData() {
            loadMetrics();
            loadCharts();
        }
        
        function loadMetrics() {
            $.get('/api/performance', function(data) {
                $('#total-return').text((data.total_return * 100).toFixed(1) + '%')
                    .removeClass('positive negative')
                    .addClass(data.total_return >= 0 ? 'positive' : 'negative');
                $('#sharpe-ratio').text(data.sharpe_ratio ? data.sharpe_ratio.toFixed(2) : 'N/A');
                $('#total-trades').text(data.total_trades || 0);
            });
            
            $.get('/api/trades', function(data) {
                if (data.length > 0) {
                    const wins = data.filter(trade => trade.pnl > 0).length;
                    const winRate = (wins / data.length * 100).toFixed(1);
                    $('#win-rate').text(winRate + '%');
                } else {
                    $('#win-rate').text('N/A');
                }
            });
        }
        
        function loadCharts() {
            $.get('/api/charts/performance', function(data) {
                if (data.data) {
                    Plotly.newPlot('performance-chart', data.data, data.layout);
                }
            });
            
            $.get('/api/charts/trades', function(data) {
                if (data.data) {
                    Plotly.newPlot('trade-chart', data.data, data.layout);
                }
            });
        }
        
        // Load data on page load
        $(document).ready(function() {
            loadMetrics();
            loadCharts();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
EOF
```

#### **Start Dashboard**
```bash
# Start dashboard
python dashboard.py &

# Access at: http://localhost:5000
```

### **2. Simple Terminal Dashboard**

#### **Create Terminal Monitor**
```bash
cat > /home/Jennifer/projects/trader/terminal_dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Terminal Dashboard for Trading Bot
"""

import os
import json
import pandas as pd
import time
from datetime import datetime
import curses

def load_data():
    """Load all trading data."""
    data = {}
    
    # Load performance
    try:
        with open('results/performance_report.json', 'r') as f:
            data['performance'] = json.load(f)
    except:
        data['performance'] = {}
    
    # Load trades
    try:
        data['trades'] = pd.read_csv('results/trade_history.csv')
    except:
        data['trades'] = pd.DataFrame()
    
    # Load recent logs
    try:
        with open('logs/trading_bot.log', 'r') as f:
            lines = f.readlines()
            data['recent_logs'] = lines[-10:]  # Last 10 lines
    except:
        data['recent_logs'] = []
    
    return data

def display_dashboard(stdscr, data):
    """Display dashboard in terminal."""
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Title
    title = "🚀 Trading Bot Dashboard"
    stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
    
    # Performance metrics
    perf = data.get('performance', {})
    row = 2
    
    stdscr.addstr(row, 0, "📊 Performance Metrics:", curses.A_BOLD)
    row += 1
    
    total_return = perf.get('total_return', 0)
    sharpe = perf.get('sharpe_ratio', 0)
    trades = perf.get('total_trades', 0)
    
    stdscr.addstr(row, 2, f"Total Return: {total_return:.1%}")
    row += 1
    stdscr.addstr(row, 2, f"Sharpe Ratio: {sharpe:.2f}")
    row += 1
    stdscr.addstr(row, 2, f"Total Trades: {trades}")
    row += 2
    
    # Recent trades
    if not data['trades'].empty:
        stdscr.addstr(row, 0, "💰 Recent Trades:", curses.A_BOLD)
        row += 1
        
        recent_trades = data['trades'].tail(5)
        for _, trade in recent_trades.iterrows():
            symbol = trade.get('symbol', 'N/A')
            action = trade.get('action', 'N/A')
            pnl = trade.get('pnl', 0)
            timestamp = trade.get('timestamp', 'N/A')
            
            trade_str = f"{symbol} {action} ${pnl:.2f} ({timestamp})"
            stdscr.addstr(row, 2, trade_str)
            row += 1
    
    row += 1
    
    # Recent logs
    stdscr.addstr(row, 0, "📝 Recent Logs:", curses.A_BOLD)
    row += 1
    
    for log in data['recent_logs']:
        if row < height - 2:
            stdscr.addstr(row, 2, log.strip()[:width-4])
            row += 1
    
    # Footer
    footer = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press 'q' to quit, 'r' to refresh"
    stdscr.addstr(height-1, 0, footer[:width-1])
    
    stdscr.refresh()

def main(stdscr):
    """Main dashboard loop."""
    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(1000)  # 1 second timeout
    
    while True:
        data = load_data()
        display_dashboard(stdscr, data)
        
        # Handle input
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('r'):
                continue  # Refresh
        except:
            pass

if __name__ == '__main__':
    curses.wrapper(main)
EOF

# Make executable
chmod +x /home/Jennifer/projects/trader/terminal_dashboard.py

# Run terminal dashboard
python terminal_dashboard.py
```

---

## ✅ **Easy Verification Commands**

### **1. Quick Health Check**
```bash
# One-liner health check
python -c "
import os, json
print('🔍 Quick Health Check')
print('=' * 30)
print(f'IBKR Config: {os.path.exists(\"config/enhanced_paper_trading_config.json\")}')
print(f'Strategy Files: {os.path.exists(\"strategies/regime_aware_ensemble.py\")}')
print(f'Log Directory: {os.path.exists(\"logs/\")}')
print(f'Results Directory: {os.path.exists(\"results/\")}')
print(f'Recent Logs: {os.path.exists(\"logs/trading_bot.log\")}')
print('=' * 30)
"
```

### **2. Test Run Verification**
```bash
# Test run with verification
python enhanced_paper_trading.py --daily

# Verify outputs
echo "=== Verification ==="
echo "Trades generated: $(wc -l < logs/trades/trades_$(date +%Y-%m).log)"
echo "Performance logged: $(wc -l < logs/performance/performance_$(date +%Y-%m).log)"
echo "Results saved: $(ls -la results/ | wc -l) files"
echo "Latest trade: $(tail -1 logs/trades/trades_$(date +%Y-%m).log)"
```

### **3. Automated Testing Script**
```bash
cat > test_end_to_end.py << 'EOF'
#!/usr/bin/env python3
"""
End-to-End Testing Script
"""

import subprocess
import time
import os
import json

def run_test():
    """Run complete end-to-end test."""
    print("🧪 Running End-to-End Test")
    print("=" * 40)
    
    # Step 1: Run trading system
    print("1. Running trading system...")
    result = subprocess.run(['python', 'enhanced_paper_trading.py', '--daily'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Trading system completed successfully")
    else:
        print("❌ Trading system failed")
        print(result.stderr)
        return False
    
    # Step 2: Check outputs
    print("\n2. Checking outputs...")
    
    checks = [
        ('Trade log exists', 'logs/trades/trades_2025-08.log'),
        ('Performance log exists', 'logs/performance/performance_2025-08.log'),
        ('Performance report exists', 'results/performance_report.json'),
        ('Trade history exists', 'results/trade_history.csv')
    ]
    
    all_passed = True
    for check_name, file_path in checks:
        if os.path.exists(file_path):
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    # Step 3: Validate data
    print("\n3. Validating data...")
    
    try:
        with open('results/performance_report.json', 'r') as f:
            perf_data = json.load(f)
        
        if 'total_trades' in perf_data and perf_data['total_trades'] > 0:
            print("✅ Trades recorded")
        else:
            print("❌ No trades recorded")
            all_passed = False
            
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        all_passed = False
    
    # Step 4: Check logs
    print("\n4. Checking logs...")
    
    try:
        with open('logs/trading_bot.log', 'r') as f:
            logs = f.readlines()
        
        recent_logs = [log for log in logs[-10:] if 'Trade executed' in log]
        if recent_logs:
            print("✅ Trade execution logged")
        else:
            print("❌ No trade execution found in logs")
            all_passed = False
            
    except Exception as e:
        print(f"❌ Log check failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 End-to-End Test PASSED!")
        return True
    else:
        print("❌ End-to-End Test FAILED!")
        return False

if __name__ == '__main__':
    success = run_test()
    exit(0 if success else 1)
EOF

# Run end-to-end test
python test_end_to_end.py
```

---

## 🚀 **Complete Setup Summary**

### **1. Verify Everything Works**
```bash
# Run validation
python validate_system.py

# Run end-to-end test
python test_end_to_end.py

# Check dashboard
python dashboard.py
```

### **2. Set Up Automation**
```bash
# Enable systemd service
sudo systemctl enable trading-bot.service
sudo systemctl start trading-bot.service

# Or set up cron
crontab -e
# Add: 30 9 * * 1-5 /home/Jennifer/projects/trader/run_trading_cron.sh
```

### **3. Monitor Performance**
```bash
# Web dashboard
python dashboard.py
# Access: http://localhost:5000

# Terminal dashboard
python terminal_dashboard.py

# Health monitoring
python health_check.py
```

### **4. Success Indicators**
- ✅ **Service running**: `systemctl status trading-bot.service`
- ✅ **Recent logs**: `tail -f logs/trading_bot.log`
- ✅ **Trades executing**: `tail -f logs/trades/trades_*.log`
- ✅ **Performance tracking**: `cat results/performance_report.json`
- ✅ **Dashboard accessible**: http://localhost:5000

**Your trading system is now fully automated and monitored!** 🎉
