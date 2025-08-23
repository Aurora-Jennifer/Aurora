import numpy as np
import pandas as pd
import pytest

from core.data_sanity.main import DataSanityValidator


@pytest.mark.rule_NON_MONO_INDEX
def test_non_monotonic_index_fails(load_clean_df):
    df = load_clean_df("SPY", size="tiny").copy()
    if len(df) > 3:
        # Make index non-monotonic by swapping two rows
        idx = df.index.to_list()
        idx[1], idx[2] = idx[2], idx[1]
        df2 = df.copy()
        df2.index = pd.DatetimeIndex(idx)
    else:
        pytest.skip("insufficient rows to permute index")
    v = DataSanityValidator(profile="strict")
    r = v.validate_dataframe_fast(df2, profile="strict")
    assert not r.ok


@pytest.mark.rule_INF_VALUES
def test_infinite_values_fail(load_clean_df):
    df = load_clean_df("SPY", size="tiny").copy()
    if "Close" not in df.columns:
        pytest.skip("no Close column to corrupt")
    df.loc[df.index[:1], "Close"] = np.inf
    v = DataSanityValidator(profile="strict")
    r = v.validate_dataframe_fast(df, profile="strict")
    assert not r.ok


@pytest.mark.rule_NAN_VALUES
def test_nan_values_fail(load_clean_df):
    df = load_clean_df("SPY", size="tiny").copy()
    if "Close" not in df.columns:
        pytest.skip("no Close column to corrupt")
    df.loc[df.index[: max(1, int(len(df) * 0.1))], "Close"] = np.nan
    v = DataSanityValidator(profile="strict")
    r = v.validate_dataframe_fast(df, profile="strict")
    assert not r.ok


