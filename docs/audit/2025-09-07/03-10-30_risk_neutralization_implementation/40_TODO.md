# TODO / Follow-ups - Risk Neutralization Implementation

## Immediate Tasks (P1.2 Completion)

### 1. Complete Integration into Training Pipeline
- [ ] Add neutralization call after cross-sectional target creation (line ~582)
- [ ] Extract market cap data from panel during preprocessing
- [ ] Add neutralization parameters to gates config
- [ ] Test integration with 60-day OOS run

### 2. Market Cap Data Integration
- [ ] Identify market cap column in panel data
- [ ] Add market cap extraction to panel builder
- [ ] Handle missing market cap data gracefully
- [ ] Add market cap to universe configuration

### 3. Sector Mapping Configuration
- [ ] Add sector classification to universe config
- [ ] Map tickers to sectors (GICS, SIC, or custom)
- [ ] Handle sector changes over time
- [ ] Add sector neutralization to config options

### 4. TODO List Management
- [x] Updated TODO list with specific neutralization tasks
- [x] Added granular tasks for integration, testing, and validation
- [x] Tracked current status of P1.2 implementation

## Testing & Validation

### 4. Neutralization Testing
- [ ] Unit tests for neutralization functions
- [ ] Integration tests with real data
- [ ] Performance benchmarks (neutralized vs non-neutralized)
- [ ] Edge case testing (missing data, single sector, etc.)

### 5. Exposure Validation
- [ ] Verify market beta reduction after neutralization
- [ ] Check sector exposure reduction
- [ ] Monitor factor stability over time
- [ ] Compare pre/post neutralization factor exposures

## Configuration & Documentation

### 6. Configuration Updates
- [ ] Add neutralization section to gates config
- [ ] Create neutralization parameter documentation
- [ ] Add neutralization examples to config files
- [ ] Update universe config with sector mappings

### 7. Documentation
- [ ] Document neutralization methodology
- [ ] Add usage examples and best practices
- [ ] Create troubleshooting guide
- [ ] Update API documentation

## Next Phase Preparation (P1.3 & P1.4)

### 8. Turnover Controls (P1.3)
- [ ] Implement 5-bucket staggering system
- [ ] Add trade bands (hysteresis) for position management
- [ ] Create turnover monitoring and reporting
- [ ] Test turnover reduction effectiveness

### 9. Transaction Costs (P1.4)
- [ ] Add half-spread cost model
- [ ] Implement impact model for large orders
- [ ] Add cost reporting to portfolio stats
- [ ] Test cost impact on net returns

## Quality & Monitoring

### 10. Performance Monitoring
- [ ] Add neutralization timing to logs
- [ ] Monitor memory usage during neutralization
- [ ] Track factor exposure metrics over time
- [ ] Create neutralization quality dashboard

### 11. Error Handling
- [ ] Add comprehensive error handling for neutralization failures
- [ ] Create fallback mechanisms for missing data
- [ ] Add validation checks for factor construction
- [ ] Implement graceful degradation

## Long-term Improvements

### 12. Advanced Neutralization
- [ ] Consider non-linear neutralization methods
- [ ] Add dynamic factor updates
- [ ] Implement regime-aware neutralization
- [ ] Add factor timing and selection

### 13. Research & Analysis
- [ ] Analyze neutralization impact on different market regimes
- [ ] Study factor stability and persistence
- [ ] Research optimal neutralization frequency
- [ ] Compare different neutralization methodologies

## Dependencies & Blockers

- **Market Cap Data**: Need to identify source and format
- **Sector Mapping**: Need sector classification system
- **Configuration**: Need to define neutralization parameters
- **Testing**: Need sample data for validation

## Success Metrics

- [ ] Market beta exposure reduced by >50%
- [ ] Sector exposures closer to zero
- [ ] No significant performance degradation
- [ ] Robust handling of edge cases
- [ ] Configurable and maintainable system
