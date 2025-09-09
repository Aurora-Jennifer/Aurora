# Simple Trading System - Easy to Use

**For Non-Technical Users**

## 🎯 **Quick Start (Just 2 Steps)**

### Step 1: Open Terminal
```bash
cd /home/Jennifer/secure/aurora_plain/trader
```

### Step 2: Run Easy Trading
```bash
python scripts/easy_trade.py
```

**That's it!** You'll see a simple menu where you can choose what to do.

---

## 🚀 **What You Can Do**

### **Option 1: Quick Test (5 minutes)**
- Tests SPY stock
- Takes 5 minutes
- Good for checking if system works

### **Option 2: Longer Test (10 minutes)**
- Tests SPY and QQQ stocks
- Takes 10 minutes
- More thorough test

### **Option 3: Crypto Test (30 minutes)**
- Tests Bitcoin and Ethereum
- Takes 30 minutes
- Works on weekends

### **Option 4: System Check**
- Quick health check
- Takes 1 minute
- Shows if everything is working

---

## 🆘 **If Something Goes Wrong**

### **"Command not found"**
```bash
cd /home/Jennifer/secure/aurora_plain/trader
```

### **"Module not found"**
```bash
pip install -r requirements.txt
```

### **"Permission denied"**
```bash
chmod +x scripts/easy_trade.py
```

### **"Nothing works"**
1. Press `Ctrl+C` to stop everything
2. Restart your computer
3. Try again

---

## 📱 **Even Simpler (One Command)**

If you just want to test SPY for 5 minutes:
```bash
python scripts/paper_runner.py --symbols SPY --poll-sec 5
```
*(Press Ctrl+C to stop)*

---

## 🎮 **What Happens When You Run It**

1. **System loads** - You'll see some messages
2. **Data downloads** - Gets current stock prices
3. **AI makes decisions** - Computer decides what to do
4. **Paper trading** - Simulates buying/selling (no real money)
5. **Results shown** - You see what happened

---

## 🔒 **Safety**

- ✅ **Paper trading only** - No real money at risk
- ✅ **Can stop anytime** - Press `Ctrl+C`
- ✅ **Safe to experiment** - Won't break anything

---

## 📞 **Need Help?**

1. **Read**: `docs/SIMPLE_USAGE_GUIDE.md`
2. **Try**: `make smoke` (quick health check)
3. **Ask**: If you're still confused

---

**Remember**: This is just a test system. You can't lose money because it's not real trading!
