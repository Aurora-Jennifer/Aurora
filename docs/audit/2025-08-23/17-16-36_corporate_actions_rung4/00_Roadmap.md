# Corporate Actions Rung 4 Implementation — Roadmap (2025-08-23 17:16)
**Prompt:** "Implement Rung 4: Corporate actions handling (splits/dividends)"

## Context
- Part of Clearframe playbook implementation for data quality
- Corporate actions (splits/dividends) can create fake PnL in backtests if not properly accounted for
- yfinance data should be adjusted, but we need validation to detect potential issues

## Plan (completed)
1) Create `scripts/fetch_corporate_actions.py` to download corporate action metadata from yfinance
2) Implement `CorporateActionsRule` in DataSanity pipeline
3) Integrate rule into `config/data_sanity.yaml` stages 
4) Test validation on AAPL (known to have 5 splits) and SPY (cleaner ETF data)
5) Validate integration with paper trading system

## Success criteria
- ✅ Corporate actions data fetched and cached for 11 symbols
- ✅ DataSanity rule detects potential unadjusted splits (4 found in AAPL)
- ✅ Rule integrated into validation pipeline without breaking existing functionality
- ✅ Paper trading system runs with corporate actions validation enabled
