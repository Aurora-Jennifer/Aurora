# TODO / Follow-ups

- [x] **Investigate paper trading 0 bars issue**: Fixed yfinance data path missing from runner.py (owner: me)
- [x] **Fix timestamp adjustment bug**: Removed problematic timestamp shifting causing DataSanity failures (owner: me)
- [ ] **Refine split detection logic**: Current 5% threshold may be too sensitive for some stocks (owner: me)
- [ ] **Add dividend gap detection**: Implement logic to detect ex-dividend date price gaps (owner: me) 
- [ ] **Automated corporate actions refresh**: Schedule periodic updates of corporate actions cache (owner: me)
- [ ] **Historical split adjustment repair**: Add repair logic to automatically adjust historical prices for detected splits (owner: me)

**Next Clearframe Rungs**
- [ ] **Rung 5**: Options data integration (owner: me) 
- [ ] **Rung 6**: Alternative data sources (economic indicators, sentiment) (owner: me)

**Recent Fixes**
- ✅ Added yfinance dataset path to `_fetch_latest_bars` in runner.py
- ✅ Removed problematic timestamp adjustment logic that was causing invalid dates
- ✅ Paper trading system now successfully loads 200 bars of AAPL data
- ✅ Corporate actions validation working correctly (detected potential splits in AAPL)
- ✅ Idempotency working (skips reprocessing same timestamps)

(Links: see ../17-16-36_corporate_actions_rung4/)