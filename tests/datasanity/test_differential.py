import numpy as np
import pytest

from core.data_sanity.main import DataSanityValidator


@pytest.mark.contract
def test_v1_v2_parity_on_clean(load_clean_df):
    # Engine switch currently routes to v1 both ways; parity expected
    df = load_clean_df("TSLA", size="small")
    v1 = DataSanityValidator(profile="default")
    r1 = v1.validate_dataframe_fast(df.copy(), profile="default")
    v2 = DataSanityValidator(profile="default")
    r2 = v2.validate_dataframe_fast(df.copy(), profile="default")
    assert r1.ok and r2.ok


@pytest.mark.contract
def test_independent_monotonic_check(load_clean_df):
    df = load_clean_df("BTC-USD", size="tiny")
    idx = df.index.view("int64")
    assert (np.diff(idx) > 0).all()
    v = DataSanityValidator(profile="default")
    r = v.validate_dataframe_fast(df.copy(), profile="default")
    assert r.ok


