# Diff Summary

**Files touched**
- ml/runner_universe.py (+200/-10)
- ml/metrics_market_neutral.py (+80/-0)

**Key Changes**

### 1. Electron Crash Fixes
- Added kaleido engine for plotly (no Electron)
- Set headless matplotlib backend before imports
- Disabled plotting in parallel workers
- Added environment variable controls

### 2. Thread Oversubscription Fixes  
- Set thread limits for all BLAS libraries
- Used loky backend with inner_max_num_threads=1
- Proper joblib configuration

### 3. Gate Logic Improvements
- Added turnover_pct calculation with actual values
- Made gate reasons self-explanatory
- Added turnover_pct to output dictionary

### 4. Portfolio Validation
- Implemented topk_ls() function (75 lines)
- Daily portfolio construction with costs
- Performance metrics calculation
- Integration into main flow

### 5. Crash Forensics
- Added faulthandler import and enable
- Crash dumps to faulthandler.log

### 6. Enhanced Outputs
- Added per_ticker_summary.csv
- Portfolio stats in metadata
- Enhanced performance reporting

### 7. Market-Neutral Gates
- Created metrics_market_neutral.py module (80 lines)
- CAPM metrics with Newey-West standard errors
- Replaced vs_BH gate with IR, alpha t-stat, beta cap
- Market returns alignment and configurable thresholds
- Enhanced per-ticker and portfolio-level validation

**Notes**
- All changes maintain backward compatibility
- No breaking changes to existing functionality
- Enhanced debugging and analysis capabilities
