# Super Simple Guide - How to Use the Trading System

**For Non-Technical Users**  
**Date**: 2025-08-24

## 🎯 **The Only Thing You Need to Know**

### **To use the system, just run this:**
```bash
python scripts/easy_trade.py
```

**That's it!** You'll see a menu where you can choose what to do.

---

## 🚀 **What Each Option Does**

### **Option 1: Quick Test (5 minutes)**
- Tests SPY stock
- Takes 5 minutes
- **Use this to check if the system works**

### **Option 2: Longer Test (10 minutes)**
- Tests SPY and QQQ stocks
- Takes 10 minutes
- **Use this for a more thorough test**

### **Option 3: Crypto Test (30 minutes)**
- Tests Bitcoin and Ethereum
- Takes 30 minutes
- **Use this on weekends (when stock market is closed)**

### **Option 4: System Check**
- Quick health check
- Takes 1 minute
- **Use this to see if everything is working**

### **Option 5: Exit**
- Closes the program
- **Use this when you're done**

---

## 🔧 **If Something Goes Wrong**

### **"Command not found"**
Make sure you're in the right folder:
```bash
cd /home/Jennifer/secure/aurora_plain/trader
```

### **"Module not found"**
Install the required software:
```bash
pip install -r requirements.txt
```

### **"Permission denied"**
Make the script executable:
```bash
chmod +x scripts/easy_trade.py
```

### **"Nothing works"**
1. Press `Ctrl+C` to stop everything
2. Restart your computer
3. Try again

---

## 📱 **Even Simpler (One Command)**

If you just want to test SPY for 5 minutes without the menu:
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

1. **Read**: `docs/SIMPLE_USAGE_GUIDE.md` (more detailed)
2. **Try**: `make smoke` (quick health check)
3. **Ask**: If you're still confused

---

## 🎯 **The Only Commands You Need to Remember**

**For daily use:**
1. **Start**: `python scripts/easy_trade.py`
2. **Stop**: `Ctrl+C`
3. **Check**: `make smoke`

**That's it! Everything else is optional.**

---

**Remember**: This is just a test system. You can't lose money because it's not real trading!
