# Test Traceability

test_context | target_files (count)
--- | ---
test_dtype_drift.test_mixed_dtypes_rejected_strict | core/data_sanity/clean.py, core/data_sanity/columnmap.py, core/data_sanity/datetime.py, core/data_sanity/errors.py, core/data_sanity/main.py (+1 more) (6)
test_duplicates_timestamps.test_duplicate_timestamps_flagged | core/data_sanity/datetime.py, core/data_sanity/errors.py, core/data_sanity/main.py, tests/unit/test_duplicates_timestamps.py (4)
test_lookahead.test_lookahead_contamination_rejected_strict | core/data_sanity/clean.py, core/data_sanity/columnmap.py, core/data_sanity/datetime.py, core/data_sanity/invariants.py, core/data_sanity/main.py (+1 more) (6)
test_nonfinite.test_nan_inf_rejected_strict | core/data_sanity/clean.py, core/data_sanity/columnmap.py, core/data_sanity/datetime.py, core/data_sanity/errors.py, core/data_sanity/main.py (+1 more) (6)
test_price_bounds.test_extreme_price_rejected_strict | core/data_sanity/clean.py, core/data_sanity/columnmap.py, core/data_sanity/datetime.py, core/data_sanity/errors.py, core/data_sanity/invariants.py (+2 more) (7)
test_price_bounds.test_negative_price_rejected_strict | core/data_sanity/clean.py, core/data_sanity/columnmap.py, core/data_sanity/datetime.py, core/data_sanity/errors.py, core/data_sanity/invariants.py (+2 more) (7)
test_returns_properties.test_config_default_percent | tests/unit/test_returns_properties.py (1)
test_returns_properties.test_percent_returns_additive_shift_correlation_high | core/metrics/returns.py, tests/unit/test_returns_properties.py (2)
test_returns_properties.test_percent_returns_scale_invariant_strict | core/metrics/returns.py, tests/unit/test_returns_properties.py (2)
test_timezone_enforcement.test_naive_index_rejected_strict | core/data_sanity/datetime.py, core/data_sanity/errors.py, core/data_sanity/main.py, tests/unit/test_timezone_enforcement.py (4)
