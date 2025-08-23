# Workflow Quick Reference
ADOI-aware workflow for Aurora development.

## 🚀 Quick Start

### For Every Change:
1. **Copy task card template** from `docs/todo.md`
2. **Fill SPEC section** before coding
3. **Implement behind flag** (default=0)
4. **Test thoroughly** (unit + integration + golden smoke)
5. **Update CI** if needed
6. **Document rollback** path

## 📋 Task Card Structure

```
### TITLE
One-line, imperative

**Status:** [ ] todo / [ ] 🚧 in progress / [ ] ✅ done / [ ] ❌ blocked  
**Flag:** FLAG_<name>=0  
**Owner:** name • **ETA:** YYYY-MM-DD • **WIP slot:** Today/This Week

#### 1) SPEC
- Problem: What's broken?
- Impact: Who benefits?
- Success Metric: Measurable outcome
- Guardrails: Which invariants touched?

#### 2) CONTRACT GATES
- Schema changes
- API changes  
- Perf budgets

#### 3) IMPLEMENT
- Minimal surface area
- Behind flag
- No rewrites

#### 4) TEST PLAN
- Unit cases
- Property/metamorphic
- Parity/determinism
- Golden smoke

#### 5) CI & ARTIFACTS
- CI steps
- Artifacts/manifest

#### 6) RISK & ROLLBACK
- Risk assessment
- Rollback command

#### 7) RELEASE NOTE
- What changed
- User impact
```

## 🎯 Key Principles

### Always:
- ✅ **Ship behind flags** (default=0)
- ✅ **Write SPEC first**
- ✅ **Test deterministically** (fixed seeds)
- ✅ **Document rollback**
- ✅ **Keep changes small**

### Never:
- ❌ **Ship without tests**
- ❌ **Break existing contracts**
- ❌ **Introduce nondeterminism**
- ❌ **Skip CI gates**
- ❌ **Forget rollback path**

## 🔧 Common Patterns

### Feature Flag Pattern:
```python
# config/feature_flags.yaml
feature_flags:
  new_validation_logic: false  # default off

# code
if config.get("feature_flags", {}).get("new_validation_logic", False):
    # new logic
else:
    # old logic
```

### Rollback Pattern:
```bash
# Rollback command
export FLAG_NEW_FEATURE=0
make test  # verify rollback worked
```

### Test Pattern:
```python
# Deterministic test
@pytest.fixture
def deterministic_seed():
    np.random.seed(42)
    return 42

def test_feature_deterministic(deterministic_seed):
    # test with fixed seed
    assert result1 == result2  # should be identical
```

## 📊 Success Metrics

### Performance Budgets:
- **Unit tests:** ≤1s each
- **Train smoke:** ≤60s total
- **Memory:** ≤2GB peak
- **Parity tolerance:** ≤1e-5 abs error

### Quality Gates:
- **CI green:** All checks pass
- **Golden smoke:** Identical outputs
- **No regressions:** Existing behavior preserved
- **Rollback verified:** Can disable feature

## 🚨 Emergency Procedures

### If CI Breaks:
1. **Check recent changes** (last 3 commits)
2. **Run locally** with same seed
3. **Rollback flag** if needed
4. **Document issue** in todo.md

### If Feature Flag Issues:
1. **Disable flag** immediately
2. **Verify rollback** works
3. **Debug in isolation**
4. **Re-enable** only when fixed

## 📚 References

- **Full workflow:** `docs/claude.md`
- **Task tracking:** `docs/todo.md`
- **Example cards:** `docs/task_cards.md`
- **Aurora rules:** `docs/ENGINEERING_CHARTER.md`
