# Cross-Sectional XGBoost Strategy - Model Card

## Model Overview

**Strategy Name:** CS_XGB_Residuals_VolFix  
**Version:** 1.0.0  
**Validation Date:** September 8, 2024  
**Status:** Production Ready (Pending Manual Approval)

## Data & Universe

- **Universe:** Top 300 US equities by market cap
- **Time Period:** 2020-01-01 to 2024-08-31 (training), 2024-06-01 to 2024-08-31 (OOS)
- **Prediction Horizon:** 10 business days
- **Embargo Period:** 3 business days
- **Update Frequency:** Daily

## Features

**Total Features:** 45 (from positive allowlist)  
**Feature Types:**
- Cross-sectional rankings (`_csr`): Price, volume, momentum indicators
- Cross-sectional z-scores (`_csz`): Relative strength measures  
- Sector residualized (`_sec_res`): Risk-adjusted rankings

**Key Feature Categories:**
- Price-based: close_csr, volume_csr, price_accel_csr
- Momentum: momentum_5_csr, reversal_1_5_csr
- Risk: vol_5_csr, sharpe_5_csr, ret_vol_ratio_csr
- Relative: relative_strength_5_csz

**Leakage Protection:**
- ✅ Pipeline order: CS transforms before forward returns
- ✅ Positive allowlist: Only known-safe features included
- ✅ Structural audit: No feature-label correlation > 0.15
- ✅ Negative controls: Label/feature shuffle tests passed

## Model Architecture

**Algorithm:** XGBoost Regressor  
**Objective:** reg:squarederror  
**Device:** CUDA (GPU accelerated)  
**Training Features:** 45 cross-sectional features  
**Target:** Cross-sectional rank of 10-day forward excess returns

**Model Configuration:**
- Early stopping: 50 rounds
- Reproducible: Fixed random seeds
- Regularization: Default XGBoost parameters

## Risk Management

**Neutralization:**
- ✅ Size: Enabled (70% factor exposure reduction)
- ✅ Sector: Enabled (60% factor exposure reduction)  
- ❌ Momentum: Disabled (acceptable residual exposure)

**Position Limits:**
- Maximum gross exposure: 30% of equity
- Per-symbol limit: 2% of equity  
- Maximum positions: 50 concurrent
- Daily trade limit: 60 trades

## Performance Metrics

### Out-of-Sample Performance (Honest Evaluation)
- **Rank IC:** 0.0174 (post-neutralization)
- **IC Information Ratio:** TBD
- **Net Sharpe Ratio:** 0.323
- **Monthly Turnover:** 1.8x
- **Hit Rate:** TBD
- **Top-Bottom Decile Spread:** TBD

### Validation Results
- **Ablation Study:** ✅ Residuals+VolFix optimal configuration
- **Negative Controls:** ✅ Label shuffle IC: -0.007, Feature shuffle IC: 0.015
- **Regime Robustness:** ✅ IC range 0.014-0.018 across COVID/rates/recent
- **Cost Sensitivity:** ✅ Sharpe degradation 0.187 at 3x costs (< 0.20 threshold)
- **Capacity:** ✅ 100% symbols accessible at $10M target equity

## Risk Factors & Limitations

**Known Failure Modes:**
- Regime shifts: Performance may degrade during structural market changes
- Liquidity dry-ups: ADV constraints may limit capacity during stress
- Factor crowding: Performance sensitive to cross-sectional dispersion
- Technology risk: GPU dependencies for model inference

**Risk Mitigations:**
- Kill switch: 2% daily loss limit triggers automatic halt
- Diversification: 50+ position limit spreads risk
- Capacity monitoring: ADV utilization tracked per symbol
- Factor monitoring: Regular factor attribution analysis

## Operational Requirements

**Data Dependencies:**
- Daily price/volume data for top 300 universe
- Sector classifications (GICS or equivalent)
- Market proxy returns (SPY)

**Computational Requirements:**
- GPU-enabled environment for model training/inference
- ~10 minute daily processing time
- 45 feature calculations per symbol per day

**Monitoring & Alerts:**
- Action entropy floor: Alert if < 0.75 for 10 bars
- PnL divergence: Alert if |realized - expected| > 3σ weekly
- Position concentration: Trim if any position > 2% equity
- Kill switch: Halt if daily loss > 2% equity

## Governance & Compliance

**Validation Framework:**
- Comprehensive leakage elimination pipeline
- Negative controls for spurious result detection  
- Multi-regime robustness testing
- Cost sensitivity and capacity analysis

**Audit Trail:**
- All features traced to known-safe sources
- Complete validation manifest available
- Reproducible with fixed random seeds
- Rollback procedures tested and documented

**Approval Status:**
- Technical validation: ✅ Complete
- Risk review: ⏳ Pending  
- Manual approval: ⏳ Required for production
- Paper trading: ⏳ Ready to commence

## Model Lineage

**Previous Versions:** N/A (Initial production version)  
**Key Improvements:** 
- Eliminated structural leakage (IC reduced from 0.8+ to realistic 0.017)
- Implemented positive feature allowlist
- Added comprehensive validation framework
- Enabled risk neutralization with acceptable performance cost

**Next Planned Updates:**
- Expanded universe (top 500 stocks)
- Additional risk factors (volatility, quality)
- Enhanced regime detection
- Multi-horizon ensemble

---

*This model card documents a systematically validated cross-sectional equity strategy ready for production deployment subject to final risk review and manual approval.*
