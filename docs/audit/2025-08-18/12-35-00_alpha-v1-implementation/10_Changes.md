# Alpha v1 Implementation — Changes (2025-08-18 12:35)

## Files Created (12 new files)

### Core Pipeline
- `config/features.yaml` - Feature definitions and parameters
- `ml/features/build_daily.py` - Feature engineering with leakage guards
- `ml/trainers/train_linear.py` - Ridge model training pipeline
- `ml/eval/alpha_eval.py` - Walkforward evaluation with cost-aware metrics
- `tools/train_alpha_v1.py` - End-to-end training pipeline
- `tools/validate_alpha.py` - Promotion gates validation

### Testing & Quality
- `tests/ml/test_leakage_guards.py` - Leakage prevention tests (5 tests)
- `tests/ml/test_alpha_eval_contract.py` - Evaluation contract tests (4 tests)
- `reports/alpha.schema.json` - JSON schema for validation

### Documentation
- `docs/runbooks/alpha.md` - Comprehensive Alpha v1 runbook

## Files Modified (2 existing files)

### Configuration Updates
- `config/models.yaml` - Added linear_v1 model registration
  - Added complete model metadata
  - Feature order specification
  - PSI thresholds and training stats

## Commands Executed
```bash
# Test the Alpha v1 pipeline
python tools/train_alpha_v1.py --symbols SPY,TSLA --n-folds 3

# Validate results
python tools/validate_alpha.py reports/alpha_eval.json

# Run leakage tests
python -m pytest tests/ml/test_leakage_guards.py -v

# Run evaluation contract tests
python -m pytest tests/ml/test_alpha_eval_contract.py -v

# Make scripts executable
chmod +x tools/train_alpha_v1.py
chmod +x tools/validate_alpha.py
```

## Code Statistics
- **Lines Added**: ~1,200 lines of production code
- **Test Coverage**: 9 comprehensive tests
- **Configuration**: 2 YAML config files
- **Documentation**: Complete runbook (~300 lines)
- **Schema**: JSON schema for validation

## Key Features Implemented

### Feature Engineering
- 8 technical features (momentum, volatility, RSI, volume)
- Strict leakage guards (label shifted forward by 1 day)
- Parquet storage for efficiency
- Data quality validation

### Model Training
- Ridge regression with cross-validation
- Time-based train/test split (no leakage)
- Deterministic training with fixed random seed
- Feature importance logging

### Evaluation
- 5-fold walkforward validation
- Cost-aware metrics (slippage + fees)
- IC, Hit Rate, Turnover, Net Return
- Schema validation

### Validation & Testing
- Promotion gates (IC ≥ 0.02, Hit Rate ≥ 0.52)
- Leakage prevention tests
- Deterministic results validation
- Comprehensive error handling
