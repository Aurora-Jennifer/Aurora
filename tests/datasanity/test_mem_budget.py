import os
import tracemalloc

import pytest

BUDGET_MB = float(os.getenv("DS_MEM_BUDGET_MB_SMALL", "64"))


@pytest.mark.sanity
def test_validate_memory_small(load_clean_df, datasanity):
    df = load_clean_df("SPY", size="small")
    tracemalloc.start()
    datasanity.validate(df)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < BUDGET_MB


