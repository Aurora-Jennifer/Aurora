# Risks & Assumptions

**Assumptions**
- yfinance provides accurate corporate actions metadata
- Adjusted prices from yfinance are properly split-adjusted 
- 5% price jump tolerance is appropriate for detecting unadjusted splits
- Volume spike detection (5x normal) identifies potential corporate action dates

**Risks**
- False positives: Normal price movements might trigger split warnings
- False negatives: Some splits might not cause detectable price jumps  
- Performance: Loading corporate actions for each validation adds overhead
- Data staleness: Corporate actions cache not automatically refreshed

**Mitigation**
- Rule configured as warning-only, doesn't fail validation pipeline
- Corporate actions directory can be updated independently
- Volume and price change thresholds are configurable
- Symbol extraction fallback prevents validation failures

**Rollback**
```bash
# Remove corporate actions rule from config
git checkout HEAD -- config/data_sanity.yaml

# Remove rule files
rm core/data_sanity/rules/corporate_actions.py
git checkout HEAD -- core/data_sanity/registry.py
```
