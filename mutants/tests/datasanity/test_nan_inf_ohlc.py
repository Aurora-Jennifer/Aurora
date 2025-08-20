import pytest

from tests.datasanity._mutate import inject_nan_inf


@pytest.mark.sanity
def test_nan_inf_ohlc_strict_fails(load_clean_df, datasanity):
    # force strict behavior for this test
    from core.data_sanity import DataSanityValidator, DataSanityError

    df = inject_nan_inf(load_clean_df("TSLA", size="tiny"))
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "TSLA")


