# Live Trading Runbook

## Daily Maintenance Workflow (Braindead-Simple)

**One command runs everything**: preflight checks, sessions, rollups, validation, alerts, and housekeeping.

### Quick Start
```bash
# Run complete daily maintenance (15 min sessions)
make maintenance

# Or customize:
python tools/daily_maintenance.py --symbols SPY,TSLA --minutes 15 --quotes ibkr --ntfy
```

### What It Does
1. **Preflight Checks**: IBKR port, session locks, environment
2. **Paper Session**: Bounded by `--minutes`, with rollup
3. **Canary Shadow**: Real quotes, validation, rollup
4. **Housekeeping**: Rotate old logs/reports (14 days)
5. **Alerts**: NTFY summary + GitHub issue on anomalies

### Cron Setup (Optional)
Run twice per weekday at 10:30a and 2:30p America/Chicago:

```bash
# crontab -e   (cron uses system local time)
30 10,14 * * 1-5 cd /path/to/Aurora && NTFY_URL="https://ntfy.sh/aurora_live_123" \
/usr/bin/env python3 tools/daily_maintenance.py --symbols SPY,TSLA --minutes 15 --quotes ibkr --ntfy >> logs/maintenance.log 2>&1
```

### Maintenance Options
```bash
# Skip paper session (canary only)
python tools/daily_maintenance.py --skip-paper --minutes 10

# Skip canary session (paper only)  
python tools/daily_maintenance.py --skip-canary --minutes 10

# Custom retention (7 days)
python tools/daily_maintenance.py --keep-days 7

# Test with dummy quotes
python tools/daily_maintenance.py --quotes dummy
```

### Exit Codes
- `0`: All sessions successful, no anomalies
- `2`: Preflight failure (IBKR port, locks)
- `3`: Anomalies detected (auto-issue filed)

## Tiny-Live Canary Checklist

### Pre-Open Setup (Once)
```bash
# Confirm branch protections require: Smoke / smoke, Smoke / Promote gate, Datasanity, model_eval
# Set environment variables (never commit)
export BROKER=ibkr
export IB_HOST=127.0.0.1
export IB_PORT=7497        # paper
export IB_CLIENT_ID=777
export IB_ACCOUNT=UXXXXXXX
export NTFY_URL="https://ntfy.sh/aurora_live_123"
rm -f kill.flag
```

### Warmup (Shadow on Real Quotes, 10-15 min)
```bash
make live-shadow
python tools/reconcile_orders.py --dry-run
```
**Green if**: `anomalies=[]`, `fallbacks=0`, reconcile OK

### Flip to Live (Same Caps, 1-2% Size)
```bash
make live
# Emergency stop: touch kill.flag
# Rehearsal: python scripts/flatten_positions.py --dry-run
```

### End-of-Session
```bash
make live-eod
```
**If any breach/fallback/anomaly**: **flatten** (no dry-run) and auto-issue filed

## SLOs to Watch (Day 1-3)
- p95 loop latency < target (from `phase_times_ms`)
- no quote heartbeat losses
- per-trade notional ≤ cap; daily turnover ≤ cap
- realized slippage ≈ config; p95 reasonable
- **0** fallbacks; **0** anomalies; reconcile clean

## Daily Shadow Workflow (This Week)

**Goal**: Prove reliability with real quotes before tiny-live canary.

### Daily Commands
```bash
# Run daily shadow with real quotes (5-15 min)
make daily-shadow

# Check results
python tools/rollup_canary.py
python tools/reconcile_orders.py --dry-run
```

### Green Criteria (per session)
- `model_fallbacks == 0`
- `anomalies == []` (no stale quotes, wide spreads, market closed)
- Reconcile passes (no position/order drift)
- Drift PSI < fail threshold
- All CI green: Smoke, Promote gate, Datasanity, model_eval

### Tuning After 3-5 Sessions
- Adjust `risk.slippage_bps` to match rollup median
- Adjust `risk.fee_bps` to match rollup p95
- Fine-tune spread tripwire (currently 50 bps) per symbol

### After 1 Week of Green
- Ready for tiny-live canary (1-2% size)
- Same runner, strict caps, real quotes

## Pre-Flight Checklist

### T-1 Day Setup
- [ ] Branch protection: required checks = Smoke / smoke, Smoke / Promote gate, Datasanity, **model_eval**
- [ ] Profiles locked: `profiles/live_canary.yaml` caps sane (per_trade_notional_pct, notional_daily_cap_pct, daily_loss_limit, max_drawdown)
- [ ] Secrets set (in Actions + locally): broker keys, account id, venue; **never** in code
- [ ] Last week's sessions: 0 fallbacks / 0 anomalies, PSI < fail, Smoke < 60s, Promote green

### T-0 (pre-market, one time)
- [ ] Time sync: drift < 2s; UTC everywhere
- [ ] Kill-switch works: `touch kill.flag` → runner halts + ntfy
- [ ] Broker dry-run: positions() & cash() succeed; positions flat
- [ ] Rollback plan open:
  - `touch kill.flag` (stop)
  - switch to paper or `profiles/paper_strict.yaml`
  - `git checkout -p config/profiles/live_canary.yaml` (revert tweaks)
  - models off quickly: set `models.enable: false`

## Broker Adapter Setup

### IBKR-Specific Hardening Checklist

#### 1. TWS/Gateway Setup
- [ ] **API enabled** in TWS/Gateway settings
- [ ] **Read-only off** (for order routing)
- [ ] **Correct port**: 7497 (paper) or 7496 (live)
- [ ] **Market data entitlements** for symbols you trade
- [ ] **Fractional shares enabled** in account (if using)

#### 2. Order Management
- [ ] **Client order IDs**: `orderRef` set for idempotency
- [ ] **Permanent IDs**: Track `permId` in addition to `orderId`
- [ ] **Outside RTH**: Explicit RTH vs outside RTH behavior
- [ ] **Pacing limits**: ≥100ms between API calls
- [ ] **Status mapping**: Normalize IB statuses to your taxonomy

#### 3. Risk Management
- [ ] **Short sale marking**: Avoid shorts until locate logic added
- [ ] **Position limits**: Enforce per-symbol and gross exposure caps
- [ ] **Reconciliation**: Run `tools/reconcile_orders.py` after sessions
- [ ] **Auto-reconnect**: Handle disconnects gracefully

#### 4. Quote Provider
- [ ] **Real quotes**: Use `--quotes ibkr` for live bid/ask
- [ ] **Shadow mode**: Test with real quotes before live
- [ ] **Spread monitoring**: Alert on wide spreads
- [ ] **Market hours**: Respect RTH vs pre/post market

#### 5. Pre-Trade Tripwires
- [ ] **Stale quotes**: HOLD if quote age > 1.5s
- [ ] **Wide spreads**: HOLD if spread > 50 bps (0.5%)
- [ ] **Market state**: HOLD if null/zero quotes (halt/closed)
- [ ] **Error handling**: Graceful fallback on quote failures

#### 6. Session Safety
- [ ] **Circuit breaker**: Session lockout on first anomaly/fallback
- [ ] **Quote heartbeat**: Alert after 5 consecutive missing quotes
- [ ] **Kill-switch**: Immediate stop via `kill.flag`

## Tiny-Live Canary Commands

### Start Tiny-Live Canary
```bash
# Export broker env locally (or use a .env you DO NOT commit)
export BROKER=ibkr  # or your_venue

# For IBKR: ensure TWS/Gateway is running and API enabled
# For other venues: export BROKER_KEY=...  BROKER_SECRET=...  BROKER_ACCOUNT_ID=...

# Start tiny-live (same runner, no --shadow) with strict profile
python scripts/canary_runner.py --symbols SPY,TSLA --poll-sec 5 --profile live_canary --steps 120

# For shadow mode with real quotes:
python scripts/canary_runner.py --symbols SPY,TSLA --poll-sec 5 --profile live_canary --quotes ibkr --shadow --steps 60

# Daily shadow workflow (run this week):
make daily-shadow
```

**Green to continue intraday**
- Orders acknowledged; fills recorded
- `model_fallbacks == 0`, anomalies == []
- Notional within per-trade & daily caps
- Live rollup shows sane slippage vs config

## Post-Session Checklist
- [ ] Run `python tools/rollup_live.py`; glance at median/p95 slippage and turnover
- [ ] If daily_loss_limit or max_drawdown breached → flatten, file auto-issue (you already wired `gh_issue.py`), and pause live until root-caused
- [ ] If any fallback/tripwire triggers → treat as red; fix before next session

## Emergency Procedures

### Emergency Stop
```bash
# Immediate stop
touch kill.flag

# Emergency flatten all positions
python scripts/flatten_positions.py

# Check reconciliation
python tools/reconcile_orders.py
```

### Rollback Commands
```bash
# Revert profile changes
git checkout -p config/profiles/live_canary.yaml

# Switch to paper mode
python scripts/canary_runner.py --profile paper_strict --shadow

# Disable models
# Edit config/base.yaml: models.enable: false
```

## Observability & Operations

### Daily Rollup
```bash
# Paper trading rollup
python tools/rollup_posttrade.py

# Canary shadow rollup  
python tools/rollup_canary.py

# Live trading rollup
python tools/rollup_live.py
```

### Alerts & Monitoring
- **End-of-run**: if fallbacks/anomalies > 0 → ntfy + GitHub issue (you already do this in paper/canary; reuse)
- **SLA**: alert if loop p95 latency > target, or if no trades for N bars (possible data/feed issue)

## Governance Knobs
- **Phase 1**: canary-smoke non-blocking (current)
- **Phase 2**: Flip canary-smoke to blocking once stable
- **Phase 3**: Flip golden to blocking after a week of greens
- **Phase 4**: Make security jobs block on high severity


