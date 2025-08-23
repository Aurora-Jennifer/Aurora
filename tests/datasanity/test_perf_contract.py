import os

import pytest

BUDGET_SMALL_MS = float(os.getenv("DS_PERF_BUDGET_MS_SMALL", "20.0"))


@pytest.mark.sanity
def test_validate_perf_small(benchmark, load_clean_df, datasanity):
    df = load_clean_df("SPY", size="small")

    def run():
        datasanity.validate(df)

    res = benchmark(run)
    assert (res.stats.mean * 1000.0) < BUDGET_SMALL_MS


