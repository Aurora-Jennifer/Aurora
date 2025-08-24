# Market Open Validation Runbook

**Purpose**: Monday market open validation for multi-symbol realtime infrastructure  
**Date**: 2025-08-25 (Monday)  
**Market Open**: 9:30 AM EST  
**Duration**: 30 minutes active monitoring  

## 🎯 **Success Criteria**

### Primary Objectives
- [ ] **No staleness breaches** (< 5s feed lag)
- [ ] **Deterministic decisions logged** (same symbol → same output)
- [ ] **No CI regression** (smoke tests remain green)
- [ ] **Kill switches functional** (FLAG_TRADING_HALTED stops orders)
- [ ] **Multi-symbol coordination** (no cross-contamination)

### Secondary Objectives  
- [ ] **Latency within budget** (feed → decision < 150ms)
- [ ] **Memory stability** (no leaks during 30min run)
- [ ] **Error recovery** (graceful WebSocket reconnection)

---

## 📋 **Pre-Market Checklist (9:00-9:25 AM)**

### 1. Environment Setup
```bash
cd /path/to/trader
git checkout feat/multi-symbol-validation
export FLAG_MULTI_SYMBOL=1
export FLAG_REALTIME=1
export STAGING_MODE=1
```

### 2. Baseline Validation
```bash
# Confirm single-symbol still works
make smoke
# Expected: PASS in <60s

# Check multi-symbol smoke test
pytest tests/smoke/test_multi_symbol_determinism.py -v
# Expected: PASS with fixture data
```

### 3. Staging Configuration
```bash
# Copy production config with staging overrides
cp config/live_canary.yaml config/staging_multi.yaml
# Edit: symbols: [SPY, QQQ] (start with 2 symbols)
# Edit: max_positions: 1 (conservative)
# Edit: log_level: DEBUG (verbose for validation)
```

---

## ⏰ **Market Open Protocol (9:30-10:00 AM)**

### T+0 (9:30:00) - Launch
```bash
# Start staging run with logging
python scripts/realtime_runner.py \
  --config config/staging_multi.yaml \
  --symbols SPY,QQQ \
  --duration 30min \
  --log-file logs/market_open_validation.log \
  --trace-file logs/validation_trace.jsonl
```

### T+1 (9:31:00) - Initial Health Check
Monitor these log patterns:
```bash
tail -f logs/market_open_validation.log | grep -E "(HEARTBEAT|STALENESS|DECISION)"
```

**Expected patterns:**
```
[09:31:00] HEARTBEAT_OK: SPY last_update=1.2s
[09:31:00] HEARTBEAT_OK: QQQ last_update=0.8s  
[09:31:01] DECISION_LOGGED: symbol=SPY confidence=0.65 latency=89ms
[09:31:02] DECISION_LOGGED: symbol=QQQ confidence=0.72 latency=95ms
```

**🚨 ABORT CONDITIONS:**
- Any `STALENESS_BREACH` (> 5s)
- Any `KILL_SWITCH_TRIGGERED`
- Any `WEBSOCKET_DISCONNECT` without `RECONNECT_SUCCESS`

### T+5 (9:35:00) - Determinism Check
```bash
# Check decision consistency
python tools/validate_determinism.py \
  --trace logs/validation_trace.jsonl \
  --symbols SPY,QQQ \
  --window 5min
```

**Expected output:**
```
✅ SPY: 5 decisions, 0 inconsistencies  
✅ QQQ: 5 decisions, 0 inconsistencies
✅ Cross-symbol independence: PASS
```

### T+10 (9:40:00) - Load Test
```bash
# Add third symbol to test scaling
echo "Adding TSLA to feed..." 
# Send SIGUSR1 to add symbol dynamically (if implemented)
kill -USR1 $RUNNER_PID
```

### T+15 (9:45:00) - Kill Switch Test
```bash
# Test graceful shutdown
touch FLAG_TRADING_HALTED
# Monitor logs for graceful halt within 10s
```

**Expected behavior:**
```
[09:45:XX] KILL_SWITCH_DETECTED: FLAG_TRADING_HALTED exists
[09:45:XX] TRADING_HALTED: no new orders accepted
[09:45:XX] POSITIONS_HELD: current positions maintained
[09:45:XX] FEEDS_ACTIVE: continue monitoring (no trading)
```

---

## 📊 **Success Validation**

### Immediate Checks (During Run)
```bash
# Staleness monitoring
grep "STALENESS" logs/market_open_validation.log | tail -10

# Decision latency  
grep "latency=" logs/validation_trace.jsonl | \
  jq '.latency_ms' | \
  awk '{sum+=$1; count++} END {print "Avg latency:", sum/count "ms"}'
# Expected: < 150ms average

# Error rate
grep "ERROR\|EXCEPTION" logs/market_open_validation.log | wc -l
# Expected: 0
```

### Post-Run Analysis (10:00+ AM)
```bash
# Generate validation report
python tools/generate_validation_report.py \
  --log logs/market_open_validation.log \
  --trace logs/validation_trace.jsonl \
  --output reports/market_open_validation.json

# Check CI regression
make smoke
# Expected: Still PASS (no regression)
```

---

## 🎯 **Go/No-Go Decision Matrix**

### ✅ **GO (Production Ready)**
- [x] Zero staleness breaches (< 5s lag)
- [x] All decisions deterministic (same inputs → same outputs)  
- [x] Latency budget met (< 150ms average)
- [x] Kill switch functional (graceful halt in <10s)
- [x] No CI regression (smoke tests still green)
- [x] Zero errors/exceptions during 30min run
- [x] Memory stable (< 5% growth over 30min)

### ❌ **NO-GO (Needs Fixes)**
- [ ] Any staleness breach > 5s
- [ ] Non-deterministic decisions (flaky outputs)
- [ ] Latency degradation (> 150ms average)
- [ ] Kill switch failure (orders continue after halt)
- [ ] CI regression (smoke tests fail)
- [ ] Errors/exceptions in logs
- [ ] Memory leak detected

---

## 🚨 **Emergency Procedures**

### If Staleness Detected
```bash
# Check network/feed health
curl -s "https://api.binance.us/api/v3/ping" | jq .
# Fallback to alternative feed if needed
```

### If Memory Leak Detected  
```bash
# Monitor process memory
ps aux | grep realtime_runner
# Expected: RSS should be stable, not growing

# If growing > 5% in 30min:
kill -TERM $RUNNER_PID  # Graceful shutdown
```

### If Kill Switch Fails
```bash
# Force halt (emergency)
kill -KILL $RUNNER_PID
# Investigate logs immediately
```

---

## 📝 **Post-Validation Actions**

### If GO Decision
1. **Tag Release**: `git tag v2.1.0-multi-symbol`
2. **Update Docs**: Mark multi-symbol as production-ready
3. **Enable Feature**: Set `FLAG_MULTI_SYMBOL=1` in production config
4. **Monitor**: 24h production monitoring with alerts

### If NO-GO Decision  
1. **Log Issues**: Document specific failures in audit trail
2. **Disable Feature**: Ensure `FLAG_MULTI_SYMBOL=0` 
3. **Fix Plan**: Create targeted fix plan for next iteration
4. **Re-test**: Schedule follow-up validation

---

## 🔗 **Quick Reference**

**Log Locations:**
- Main log: `logs/market_open_validation.log`
- Trace: `logs/validation_trace.jsonl`  
- Report: `reports/market_open_validation.json`

**Key Commands:**
- Start: `python scripts/realtime_runner.py --config config/staging_multi.yaml`
- Monitor: `tail -f logs/market_open_validation.log | grep -E "(HEARTBEAT|DECISION)"`
- Validate: `python tools/validate_determinism.py --trace logs/validation_trace.jsonl`
- Emergency Stop: `touch FLAG_TRADING_HALTED`

**Contact Info:**
- Emergency: [Your contact info]
- Escalation: [Team lead contact]

---

**Duration**: 30 minutes active monitoring  
**Risk Level**: Low (staging environment, feature-flagged)  
**Rollback**: `FLAG_MULTI_SYMBOL=0` (immediate)
