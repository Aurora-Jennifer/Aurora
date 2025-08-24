# Simple Usage Guide - How to Use the Trading System

**For Non-Technical Users**  
**Date**: 2025-08-24

## 🎯 **Quick Start (3 Steps)**

### Step 1: Start Paper Trading (Easiest)
```bash
# Just run this one command:
python scripts/easy_trade.py
```

### Step 2: Check if it's working
Look for these messages:
- ✅ "Paper trading started"
- ✅ "Processing SPY data"
- ✅ "Paper trading completed"

### Step 3: Stop if needed
Press `Ctrl+C` to stop anytime.

---

## 🚀 **Common Tasks (Copy & Paste)**

### 1. **Paper Trade SPY for 5 minutes**
```bash
python scripts/easy_trade.py
# Then choose option 1
```

### 2. **Paper Trade multiple stocks**
```bash
python scripts/easy_trade.py
# Then choose option 2
```

### 3. **Paper Trade crypto (weekends)**
```bash
python scripts/easy_trade.py
# Then choose option 3
```

### 4. **Run a quick test**
```bash
make smoke
```

---

## 🔧 **Troubleshooting (If Something Goes Wrong)**

### **Problem**: "Command not found"
**Solution**: Make sure you're in the right folder:
```bash
cd /home/Jennifer/secure/aurora_plain/trader
```

### **Problem**: "Module not found"
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

### **Problem**: "Permission denied"
**Solution**: Make scripts executable:
```bash
chmod +x scripts/easy_trade.py
```

### **Problem**: "Model not found"
**Solution**: Check if models exist:
```bash
ls models/
```

---

## 📊 **What Each Command Does**

| Command | What it does | When to use |
|---------|-------------|-------------|
| `python scripts/easy_trade.py` | Opens menu for trading options | Testing the system |
| `python scripts/paper_runner.py --symbols SPY --poll-sec 5` | Paper trades SPY continuously | Direct trading |
| `python scripts/paper_runner.py --symbols SPY,QQQ --poll-sec 5` | Paper trades multiple stocks | Longer testing |
| `python scripts/paper_runner.py --symbols BTCUSDT --poll-sec 5` | Paper trades crypto | Weekend testing |
| `make smoke` | Runs quick tests | Check if system is working |
| `python tools/test_asset_routing.py` | Tests new features | Only if you're curious |

---

## 🎮 **Simple Control Panel**

### **Start Trading**
```bash
# Choose one:
python scripts/easy_trade.py    # Menu-based (recommended)
python scripts/paper_runner.py --symbols SPY --poll-sec 5    # Direct (advanced)
```

### **Check Status**
```bash
make smoke  # Quick health check
```

### **Stop Everything**
```bash
# Press Ctrl+C in any running terminal
# OR
echo "1" > kill.flag  # Emergency stop
```

---

## 📱 **One-Button Operations**

### **"I want to test the system"**
```bash
python scripts/easy_trade.py
# Then choose option 1
```

### **"I want to see if it's working"**
```bash
make smoke
```

### **"I want to stop everything"**
```bash
# Press Ctrl+C
```

---

## 🆘 **Emergency Contacts**

### **If nothing works:**
1. Press `Ctrl+C` to stop everything
2. Run: `make smoke`
3. If that fails, restart your computer

### **If you're confused:**
- Just use: `python scripts/easy_trade.py`
- That's the safest command

### **If you want to learn more:**
- Read: `docs/PAPER_TRADING_STATUS.md`
- But you don't need to - the simple commands above work fine

---

## 🎯 **The Only Commands You Need**

**For daily use, just remember these 3:**

1. **Test**: `python scripts/easy_trade.py`
2. **Check**: `make smoke`
3. **Stop**: `Ctrl+C`

**That's it! Everything else is optional.**

---

**Remember**: This is paper trading - no real money is at risk. You can experiment safely!
