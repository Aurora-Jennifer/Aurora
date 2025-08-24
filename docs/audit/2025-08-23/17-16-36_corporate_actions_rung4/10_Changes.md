# Changes

## Actions
- `scripts/fetch_corporate_actions.py`: Create new script to fetch corporate actions metadata from yfinance
- `core/data_sanity/rules/corporate_actions.py`: Create new validation rule for corporate actions
- `core/data_sanity/registry.py`: Register corporate actions rule in RULES dictionary  
- `config/data_sanity.yaml`: Add corporate_actions rule to ingest and post_adjust stages

## Commands run
```bash
# Fetch corporate actions for expanded symbol universe
python scripts/fetch_corporate_actions.py --symbols "AAPL,SPY,QQQ,IWM,TLT,GLD,UUP,VIX,EEM,FXI,EWJ" --start-date 2000-01-01

# Test individual rule
python -c "from core.data_sanity.rules.corporate_actions import create_corporate_actions_rule; ..."

# Test DataSanity integration
python -c "from core.data_sanity import DataSanityValidator; validator.validate_and_repair(aapl_data, symbol='AAPL')"

# Test paper trading integration
python scripts/runner.py --profile config/profiles/yfinance_multi_regime.yaml --minutes 1 --symbols "AAPL" --source csv
```

## Key findings
- AAPL shows 4 potential unadjusted splits (2000, 2005, 2014, 2020) - validation working correctly
- SPY (ETF) has clean data with no corporate action issues  
- Corporate actions cached for 11 symbols: splits_count and dividends_count per symbol
- Integration successful - warnings logged but don't break validation pipeline
