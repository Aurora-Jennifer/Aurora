# Risks & Assumptions - Realtime Infrastructure

## Assumptions
- **Network connectivity**: Assumes stable connection to Binance testnet/mainnet for production use
- **WebSocket stability**: Relies on Binance WebSocket API availability and message format consistency  
- **Feature flag discipline**: Assumes operators will properly set `FLAG_REALTIME=1` for production use
- **Single symbol limitation**: Current implementation handles one symbol per realtime feed (scalable design)
- **Testnet equivalence**: Assumes testnet WebSocket behavior matches production for validation

## Risk Assessment

### 🔴 **High Risk**
- **WebSocket outages**: Complete trading halt if feed fails (MITIGATED: kill switch + graceful degradation)
- **Message format changes**: Binance API changes could break parsing (MITIGATED: comprehensive error handling)

### 🟡 **Medium Risk**  
- **Latency spikes**: High latency could impact trading performance (MITIGATED: latency tracking + monitoring)
- **Memory leaks**: Long-running WebSocket connections (MITIGATED: proper cleanup + timeouts)
- **Race conditions**: Async message handling (MITIGATED: sequential processing design)

### 🟢 **Low Risk**
- **Feature flag confusion**: Forgetting to enable `FLAG_REALTIME=1` (MITIGATED: clear logging + validation)
- **Test environment differences**: Testnet vs production behavior (MITIGATED: configuration-driven URLs)

## Rollback Plan

### Immediate Rollback
```bash
# Disable realtime mode globally
export FLAG_REALTIME=0

# Revert to static mode
python scripts/runner.py --source csv --symbols SPY,QQQ --minutes 15
```

### Code Rollback  
```bash
# Selective file revert
git checkout HEAD~1 -- scripts/runner.py

# Full rollback to previous commit
git revert <commit_hash>
```

### Emergency Procedures
1. **Kill switch activation**: Set `FLAG_TRADING_HALTED=1` immediately stops all trading
2. **Fallback to static**: Change `--source csv` to use historical data
3. **Monitor rollback**: Verify trading resumes with static data sources

## Monitoring Requirements
- **Feed staleness**: Alert if no messages > 10s
- **Trading halts**: Immediate alerts on kill switch activation  
- **Latency degradation**: Alert if total latency > 1000ms
- **Connection failures**: Alert on WebSocket disconnections
