# DataSanity Upgrade & Testing Guardrails for Cursor

When working on the **DataSanity** module, validators, or any critical data validation layer in this trading bot project, follow these rules:

---

## Core Principles

1. **Functional > performance** — never sacrifice correctness for speed.
2. Changes must be **incremental and reversible** (use v1/v2 or feature flags for risky upgrades).
3. No full rewrites unless explicitly approved after v2 passes in production.
4. All schema/API changes must include **adapters** so old data/tests pass until deprecation.

---

## Test & Perf Threshold Policy

### Performance Baselines

Based on current known-good runs:

* **100 rows:** ≤ **0.05s** (with 5× headroom for local runs)
* **1,000 rows:** ≤ **0.20s**
* **10,000 rows:** ≤ **0.60s**
* Memory for 10k rows: ≤ **250 MB RSS peak**

### Tolerance Modes

* **RELAXED (local default):** Allow up to **+30%** over thresholds.
* **STRICT (CI):** Allow up to **+10%** over thresholds.

### Mode Control

Use `SANITY_PERF_MODE=RELAXED` for local work and `SANITY_PERF_MODE=STRICT` in CI.

---

## Required Test Workflow in Changes

When updating DataSanity or related code, output should include:

1. **Context Recap** — problem being solved, affected files, dependencies.
2. **Implementation Plan** — 5–7 steps max, minimal diff, no cascade rewrites.
3. **Modified Code Only** — comment changes with `# NEW` or `# CHANGED`.
4. **Validation Commands** — include:

   ```bash
   pytest -m "data_sanity or property or integration" -q
   pytest -m "perf or benchmark" -v > results/perf.log 2>&1
   scripts/perf_gate.py
   ```
5. **Fallback Instructions** — how to revert to v1 or disable feature flag if issues appear.

---

## Anti-Flake Measures

* Pin randomness: `PYTHONHASHSEED=0` and Hypothesis profile with ≤50 examples.
* Freeze key deps (pandas, pyarrow) during migration windows.
* Golden datasets for deterministic testing.
* Default network tests to **mocked** unless `RUN_NETWORK=1`.

---

## Perf Gate Implementation

Maintain a `scripts/perf_gate.py` that:

* Parses `results/perf.log` for `test_performance_validation[...]` durations.
* Compares against thresholds with mode-based tolerance.
* Fails if exceeded.

**Usage:**
```bash
# Basic usage (uses SANITY_PERF_MODE env var)
python scripts/perf_gate.py

# Explicit mode
python scripts/perf_gate.py --mode STRICT

# Custom log file
python scripts/perf_gate.py --log-file custom_perf.log

# Verbose output
python scripts/perf_gate.py --verbose
```

---

## When in Doubt

If a proposed change touches >3 dependent modules, stop and propose a **phased migration** instead of a single large patch.

---

## File Locations

* **Tests:** `tests/test_data_integrity.py`, `tests/test_data_sanity_enforcement.py`
* **Config:** `config/data_sanity.yaml`
* **Perf Gate:** `scripts/perf_gate.py`
* **Pytest Config:** `pytest.ini`
* **Documentation:** `docs/DATASANITY_GUARDRAILS.md`

---

## Test Categories & Markers

### Core Validation
- `@pytest.mark.data_sanity` - Core data sanity validation
- `@pytest.mark.validation` - Data validation logic
- `@pytest.mark.property` - Property-based testing

### Performance
- `@pytest.mark.perf` - Performance tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.stress` - Stress testing

### Integration
- `@pytest.mark.integration` - Integration verification
- `@pytest.mark.network` - Network resilience
- `@pytest.mark.flaky` - Flaky tests

### Edge Cases
- `@pytest.mark.edge_case` - Edge case handling
- `@pytest.mark.falsification` - Falsification scenarios

---

## Quick Reference Commands

```bash
# Run core validation tests
pytest -m "data_sanity or property or integration" -q

# Run performance tests with logging
pytest -m "perf or benchmark" -v > results/perf.log 2>&1

# Check performance gate
python scripts/perf_gate.py --mode STRICT

# Run with deterministic settings
PYTHONHASHSEED=0 pytest tests/test_data_integrity.py -v

# Full test suite
pytest tests/test_data_integrity.py tests/test_data_sanity_enforcement.py -v
```

---

## Configuration Updates

When modifying DataSanity, update these files:

1. **`config/data_sanity.yaml`** - Add performance monitoring settings
2. **`pytest.ini`** - Add new test markers if needed
3. **`scripts/perf_gate.py`** - Update thresholds if baselines change
4. **`docs/DATASANITY_GUARDRAILS.md`** - Update documentation

---

## Performance Gate Code (Self-Healing)

If `scripts/perf_gate.py` is lost, regenerate it with this structure:

```python
#!/usr/bin/env python3
"""
Performance Gate for DataSanity Tests
Enforces performance thresholds for data validation operations.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Performance baselines (in seconds)
PERFORMANCE_BASELINES = {
    100: 0.05,      # 100 rows: ≤ 0.05s
    1000: 0.20,     # 1,000 rows: ≤ 0.20s
    10000: 0.60,    # 10,000 rows: ≤ 0.60s
}

# Memory baseline (in MB)
MEMORY_BASELINE = 250  # 10k rows: ≤ 250 MB RSS peak

# Tolerance modes
TOLERANCE_MODES = {
    "RELAXED": 0.30,  # Allow up to +30% over thresholds
    "STRICT": 0.10,   # Allow up to +10% over thresholds
}

# Test patterns to monitor
PERFORMANCE_TEST_PATTERNS = [
    r"test_performance_validation\[(\d+)\]",
    r"test_performance_guard_enforcement\[(\d+)\]",
    r"test_performance_contract_validation",
    r"test_stress_large_dataset",
    r"test_memory_usage_patterns\[(\d+)\]",
]

# ... rest of implementation with parse_perf_log, check_performance_thresholds, etc.
```

---

**Remember:** Always prioritize correctness over performance, use incremental changes, and maintain backward compatibility!
