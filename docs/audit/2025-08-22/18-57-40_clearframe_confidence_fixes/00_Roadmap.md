# Clearframe Confidence Fixes — Roadmap (2025-08-22 18:57)

**Prompt:** "I think the confidence scores are broken"

## Context
Clearframe identified critical issues:
1. **Historical Data Replay**: System replaying year-2000 data (`bar_end=2000-12-...`) instead of live
2. **Risk Caps Correct**: Risk layer correctly capping qty to 0 for historical data (`qty_capped_to_zero`)
3. **Price Guards Too Permissive**: $132 SPY passing validation when should be more restrictive
4. **Poor Logging**: Caps logged without detailed reasons why

## Plan (now)
1. **✅ Create live profile** - `yfinance_live_1m.yaml` for true live 1-minute feed
2. **✅ Implement staleness gate** - Bar-interval-based staleness with caps to zero for historical
3. **✅ Tighten price guards** - Add dual checks: recent window band + mid/last deviation  
4. **✅ Improve cap logging** - Detailed reason logging for all risk caps
5. **🔄 Test live mode** - Verify `bar_end≈2025-08-22` and `historical=False`
6. **⏳ Align bar close** - Compute only at bar close to prevent same-bar churn

## Success criteria
- Live profile shows current dates (`bar_end≈2025-08-22`)
- Historical data properly caps to zero with clear reasons
- Price guards reject more aggressively with detailed messages
- All caps log specific reasons (historical_data, max_qty_per_symbol, etc.)
- Confidence scores work properly with live data
