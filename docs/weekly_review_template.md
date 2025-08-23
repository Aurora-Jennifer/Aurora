# Weekly Owner's Review Template

**Date**: [YYYY-MM-DD]  
**Reviewer**: [Your Name]  
**Duration**: [60-90 minutes]  

## 📊 Metrics Audit

### Performance Metrics
- **IC/IR**: Current vs last week
  - IC: [current] vs [last_week] (Δ: [change])
  - IR: [current] vs [last_week] (Δ: [change])
  - **Action**: [Improve/Investigate/Accept]

- **Turnover**: Current vs last week
  - Turnover: [current] vs [last_week] (Δ: [change])
  - **Action**: [Optimize/Investigate/Accept]

- **Hit Rate**: Current vs last week
  - Hit Rate: [current] vs [last_week] (Δ: [change])
  - **Action**: [Improve/Investigate/Accept]

- **Drawdown**: Current vs last week
  - Max Drawdown: [current] vs [last_week] (Δ: [change])
  - **Action**: [Risk Management/Investigate/Accept]

### Operational Metrics
- **Slippage Error vs Assumption**: [actual] vs [assumed]
- **Latency p95**: [current] vs [target] (≤150ms)
- **Coverage**: [current] vs [target] (≥85%)
- **Mutation Score**: [current] vs [target] (≥30%)

## 🚨 Error Budget Review

### Incidents This Week
| Date | Class | Description | Impact | Root Cause | Fix Status |
|------|-------|-------------|--------|------------|------------|
| [Date] | [Data/Model/Broker/Infra] | [Description] | [High/Med/Low] | [Root Cause] | [Open/In Progress/Closed] |

### Error Budget Allocation
- **Data Incidents**: [count] (Budget: [X] per month)
- **Model Incidents**: [count] (Budget: [X] per month)  
- **Broker Incidents**: [count] (Budget: [X] per month)
- **Infra Incidents**: [count] (Budget: [X] per month)

### One Fix Per Class (This Week)
- [ ] **Data**: [Specific fix with owner and deadline]
- [ ] **Model**: [Specific fix with owner and deadline]
- [ ] **Broker**: [Specific fix with owner and deadline]
- [ ] **Infra**: [Specific fix with owner and deadline]

## 🏗️ Tech Debt Ledger

### Top 3 Debt Items (Priority Order)
1. **[High Priority]**: [Description] - [Impact] - [Effort] - [Owner] - [Deadline]
2. **[Medium Priority]**: [Description] - [Impact] - [Effort] - [Owner] - [Deadline]
3. **[Low Priority]**: [Description] - [Impact] - [Effort] - [Owner] - [Deadline]

### Debt Reduction This Week
- [ ] **Committed**: [Specific debt reduction with measurable outcome]
- [ ] **Progress**: [Status update on ongoing debt reduction]

### New Debt Identified
- [ ] [New debt item with justification and priority]

## 📚 Documentation Drift

### Pages Updated This Week
- [ ] **README Architecture Diagram**: [Update description]
- [ ] **"How to Add a New Asset"**: [Update description]
- [ ] **"Backtest Assumptions"**: [Update description]
- [ ] **Other**: [Update description]

### Documentation Gaps Identified
- [ ] [Missing documentation with priority]

## 🧪 Chaos Engineering

### Failure Injection Test
**Test**: [Description of injected failure]
- **Injection**: [What was broken]
- **Expected**: [Expected graceful degradation]
- **Actual**: [What actually happened]
- **Result**: [Pass/Fail with details]

### Graceful Degradation Verification
- [ ] **Stale Data**: System handles gracefully
- [ ] **Broker 500**: System handles gracefully  
- [ ] **Time Skew**: System handles gracefully
- [ ] **Network Partition**: System handles gracefully

## 🔧 Error Resolution Tracking

### Errors Resolved This Week
| Error Type | Count | Resolution Time | Pattern | Action |
|------------|-------|-----------------|---------|--------|
| [Linting/CI/Data/Perf] | [X] | [Avg time] | [Pattern] | [Action] |

### Error-Checking Prompt Usage
- [ ] **Applied Master Prompt**: [Count] times
- [ ] **Success Rate**: [X/Y] successful resolutions
- [ ] **Average Resolution Time**: [X minutes]
- [ ] **Rollback Rate**: [X%] requiring rollback

### Error Patterns Identified
- [ ] **Recurring Issues**: [List patterns]
- [ ] **Root Cause Analysis**: [Common causes]
- [ ] **Prevention Measures**: [Proactive fixes]

## 🔍 Gate Performance Review

### Hard Gates Status
| Gate | Status | Performance | Issues | Action |
|------|--------|-------------|--------|--------|
| Security | [Pass/Fail] | [Time] | [Issues] | [Action] |
| Type | [Pass/Fail] | [Time] | [Issues] | [Action] |
| Coverage | [Pass/Fail] | [%] | [Issues] | [Action] |
| Performance | [Pass/Fail] | [Regressions] | [Issues] | [Action] |
| Parity | [Pass/Fail] | [Drift] | [Issues] | [Action] |
| Leakage | [Pass/Fail] | [Violations] | [Issues] | [Action] |
| Property | [Pass/Fail] | [Properties] | [Issues] | [Action] |

### Gate Improvements Needed
- [ ] [Specific gate improvement with owner and deadline]

## 📈 External Validation

### Public Artifacts Status
- [ ] **Architecture Diagram**: [Current/Needs Update]
- [ ] **Redacted Config Profile**: [Current/Needs Update]
- [ ] **Sample Trace Log**: [Current/Needs Update]
- [ ] **SLO Metrics Screenshot**: [Current/Needs Update]
- [ ] **Test Matrix**: [Current/Needs Update]
- [ ] **CI Badge**: [Current/Needs Update]
- [ ] **Mutation Score Trend**: [Current/Needs Update]
- [ ] **Coverage Trend**: [Current/Needs Update]
- [ ] **Parity Test Output**: [Current/Needs Update]

### Open Source Opportunities
- [ ] **DataSanity Library**: [Status - Ready/In Progress/Blocked]
- [ ] **Walk-Forward Harness**: [Status - Ready/In Progress/Blocked]
- [ ] **Other Utilities**: [Status - Ready/In Progress/Blocked]

## 🎯 Next Week's Focus

### Priority 1: [High priority item with measurable outcome]
### Priority 2: [Medium priority item with measurable outcome]  
### Priority 3: [Low priority item with measurable outcome]

## ✅ Review Completion

- [ ] **Metrics reviewed and trends identified**
- [ ] **Error budget allocated and fixes assigned**
- [ ] **Tech debt prioritized and one reduction committed**
- [ ] **Documentation updated**
- [ ] **Chaos test executed and results documented**
- [ ] **Gates performance reviewed**
- [ ] **External validation status updated**
- [ ] **Next week's priorities set**

---

**Review Completed**: [Time]  
**Next Review**: [Date]  
**Action Items**: [Count]  
**Follow-up Required**: [Yes/No]
