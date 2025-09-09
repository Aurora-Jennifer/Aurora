# TODO / Follow-ups

## High Priority
- [ ] **Test the fixes** with a small universe run to validate stability
- [ ] **Install kaleido** if not already available: `pip install kaleido`
- [ ] **Monitor memory usage** in large universe runs for potential OOM issues

## Medium Priority  
- [ ] **Tune thread limits** if memory issues persist (may need chunking)
- [ ] **Validate cost model** against actual trading costs
- [ ] **Consider market-neutral gates** (IR, alpha t-stat, beta cap) as suggested

## Low Priority
- [ ] **Add more portfolio strategies** beyond top-K long-short
- [ ] **Enhance cost model** with more sophisticated transaction cost modeling
- [ ] **Add portfolio visualization** for equity curves and drawdowns

## Testing
- [ ] **Small universe test** (10-20 tickers) to validate fixes
- [ ] **Medium universe test** (100+ tickers) to check memory usage
- [ ] **Large universe test** (500+ tickers) to validate scalability

## Documentation
- [ ] **Update README** with new portfolio validation features
- [ ] **Add troubleshooting guide** for common parallel processing issues
- [ ] **Document cost model assumptions** and tuning parameters

(Links: see ../00-32-59_universe_runner_fixes/)
