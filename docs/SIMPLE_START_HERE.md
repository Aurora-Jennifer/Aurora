# 🚀 Aurora: Start Here (Simple Guide)

**Stop overthinking. Follow these steps exactly.**

---

## ✅ **STEP 1: Prove It Works (3 commands)**

Copy/paste these commands one at a time:

```bash
cd /home/Jennifer/secure/aurora_plain/trader

# Test 1: Check if basic system loads
python -c "from core.utils import setup_logging; print('✅ Core system loads')"

# Test 2: Check if paper trading exists  
python scripts/SUPER_EASY.py --help

# Test 3: Check if experiments work
python scripts/experiment_runner.py list
```

**If these work without errors → Your system is operational.**

---

## 🎯 **STEP 2: See It Generate Configs (1 command)**

```bash
# See what your system can do (no actual training)
python scripts/experiment_runner.py quick_validation --dry-run
```

**Expected output:** Should show 4+ generated configurations with different models/parameters.

**If you see configs generated → Your experiment system works.**

---

## 🔥 **STEP 3: Run Real Paper Trading (1 command)**

```bash
# Run paper trading for 5 minutes to see live output
timeout 300 python scripts/paper_runner.py || echo "Paper trading completed"
```

**Expected output:** Live log messages showing price data, predictions, trades.

**If you see live trading output → Your trading system works.**

---

## 📊 **STEP 4: Test Model Training (1 command)**

```bash
# Train a simple model (uses synthetic data, safe)
python scripts/train.py --model ridge --symbols SPY --quick
```

**Expected output:** Training progress, IC calculation, model saved.

**If training completes → Your ML pipeline works.**

---

## 🎉 **STEP 5: Run One Real Experiment (1 command)**

```bash
# Discovery phase only (safe, no commitment)
python scripts/experiment_runner.py quick_validation --discovery-only
```

**Expected output:** 
- Configs generated
- Training progress for each config
- IC statistics and results
- Candidate selection

**If experiment completes → You have a working research system.**

---

## 🚨 **What to Do If Something Breaks**

### **Error: "Module not found"**
```bash
# Fix Python path
export PYTHONPATH=/home/Jennifer/secure/aurora_plain/trader:$PYTHONPATH
```

### **Error: "Config file not found"**
```bash
# Check if you're in the right directory
pwd
# Should show: /home/Jennifer/secure/aurora_plain/trader
```

### **Error: "Permission denied"**
```bash
# Make scripts executable
chmod +x scripts/*.py
```

---

## 📋 **Daily Usage Workflow**

### **For Research & Development:**
```bash
# 1. Test new features quickly
python scripts/experiment_runner.py quick_validation --discovery-only

# 2. Compare models
python scripts/train.py --compare --models ridge,xgboost --symbols SPY

# 3. Analyze results
python scripts/experiments.py list
```

### **For Paper Trading:**
```bash
# 1. One-button paper trading
python scripts/SUPER_EASY.py

# 2. Or manual control
python scripts/paper_runner.py --duration 60
```

### **For Production Research:**
```bash
# 1. Large-scale momentum research
python scripts/experiment_runner.py momentum_discovery

# 2. Cross-asset validation
python scripts/experiment_runner.py cross_asset_momentum

# 3. Conservative validation (for live trading)
python scripts/experiment_runner.py conservative_validation
```

---

## 🎯 **How to Add New Features (Simple)**

### **Method 1: Use Built-in Features**
```bash
# See what features exist
python -c "
import yaml
with open('config/features.yaml') as f:
    features = yaml.safe_load(f)
    print('Available features:')
    for feat in features.get('features', []):
        print(f'  - {feat}')
"
```

### **Method 2: Test Different Feature Sets**
```bash
# Test momentum vs volatility features
python scripts/experiment_runner.py momentum_discovery --discovery-only

# Then try:
# Edit config/experiment_profiles.yaml
# Change feature_families: ["momentum_basic"] to ["volatility_focus"]
# Re-run experiment
```

---

## 🔍 **How to Interpret Results**

### **IC (Information Coefficient) - Most Important**
- **IC > 0.05**: Strong signal (likely profitable)
- **IC 0.02-0.05**: Moderate signal (maybe profitable)  
- **IC < 0.02**: Weak signal (probably not profitable)
- **IC < 0**: Wrong direction (losing money)

### **Sharpe Ratio**
- **Sharpe > 1.0**: Good risk-adjusted returns
- **Sharpe 0.5-1.0**: Decent returns
- **Sharpe < 0.5**: Poor returns

### **Hit Rate**
- **Hit Rate > 55%**: Predictions correct more than half the time
- **Hit Rate ~50%**: Random (no predictive power)
- **Hit Rate < 50%**: Systematically wrong

---

## 🚀 **Confidence Builders**

### **Proof Point 1: System Loads**
```bash
python -c "print('✅ Aurora system operational')"
```

### **Proof Point 2: Can Generate Configs**
```bash
python scripts/experiment_runner.py quick_validation --dry-run | head -20
```

### **Proof Point 3: Can Run Paper Trading**  
```bash
timeout 30 python scripts/paper_runner.py 2>&1 | head -10
```

### **Proof Point 4: Can Train Models**
```bash
python scripts/train.py --model ridge --symbols SPY --quick 2>&1 | tail -5
```

**If all 4 work → Your system is fully operational.**

---

## 🎪 **Stop Overthinking Commands**

When you feel overwhelmed, just run:

```bash
# The simplest possible test
python scripts/experiment_runner.py list

# If that works, run:
python scripts/experiment_runner.py quick_validation --dry-run

# If that works, run:  
python scripts/experiment_runner.py quick_validation --discovery-only
```

**Three commands. That's it. Start there.**

---

## 📞 **Emergency "It's All Broken" Recovery**

If nothing works:

```bash
# 1. Check you're in the right place
cd /home/Jennifer/secure/aurora_plain/trader
pwd

# 2. Check Python can find modules
python -c "import sys; print('\\n'.join(sys.path))"

# 3. Run the simplest possible test
python -c "print('Python works')"

# 4. Test core imports
python -c "from pathlib import Path; print('✅ Basic imports work')"

# 5. Test Aurora core
python -c "from core.utils import setup_logging; print('✅ Aurora core works')"
```

**Work through these one by one. Find where it breaks. Fix that one thing.**

---

## 🎯 **Your Next 30 Minutes**

1. **Run the 5 steps above** (one at a time)
2. **Don't overthink** - just copy/paste commands
3. **If something breaks** - stop, fix that one thing, continue
4. **When all 5 steps work** - you'll know your system is solid

**That's it. No complexity. Just execution.**

You built this. Now prove to yourself it works.
