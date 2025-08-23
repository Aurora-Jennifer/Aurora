import pandas as pd
import pytest

from core.data_sanity.main import DataSanityValidator
from core.data_sanity.trace import Trace


@pytest.mark.sanity
def test_idempotence(load_clean_df):
    df = load_clean_df("SPY", size="tiny")
    v = DataSanityValidator(profile="default")
    t1 = Trace()
    v.validate_dataframe_fast(df.copy(), profile="default", trace=t1)
    t2 = Trace()
    v.validate_dataframe_fast(df.copy(), profile="default", trace=t2)
    assert t1.summary() == t2.summary()


@pytest.mark.sanity
@pytest.mark.rule_DUP_TS
def test_duplicate_increase_monotonic(load_clean_df):
    df = load_clean_df("SPY", size="small")
    # Introduce duplicates progressively
    df1 = df.copy()
    if len(df1) > 10:
        dup_idx = df1.index[:int(len(df1)*0.01)]
        df1 = pd.concat([df1, df1.loc[dup_idx]]).sort_index()

    df2 = df.copy()
    if len(df2) > 10:
        dup_idx2 = df2.index[:int(len(df2)*0.02)]
        df2 = pd.concat([df2, df2.loc[dup_idx2]]).sort_index()

    v = DataSanityValidator(profile="strict")
    t1 = Trace()
    r1 = v.validate_dataframe_fast(df1, profile="strict", trace=t1)
    t2 = Trace()
    r2 = v.validate_dataframe_fast(df2, profile="strict", trace=t2)
    # Both should fail on duplicates
    assert not r1.ok and not r2.ok


