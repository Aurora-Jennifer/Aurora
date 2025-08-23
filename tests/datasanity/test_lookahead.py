import pytest

from tests.datasanity._mutate import inject_lookahead


@pytest.mark.sanity
def test_lookahead_detected_in_strict(load_clean_df):
    from core.data_sanity import DataSanityError, DataSanityValidator

    df = inject_lookahead(load_clean_df("BTC-USD", size="tiny"))
    df.index = df.index.tz_localize("UTC")
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "BTC-USD")


