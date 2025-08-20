import pytest

from tests.datasanity._mutate import inject_duplicates, inject_non_monotonic
from tests.datasanity._golden import EXPECTED


@pytest.mark.sanity
@pytest.mark.contract
def test_duplicate_timestamps_fail(load_clean_df):
    from core.data_sanity import DataSanityValidator, DataSanityError

    df = inject_duplicates(load_clean_df("SPY", size="small"))
    df.index = df.index.tz_localize("UTC")
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "SPY")


@pytest.mark.sanity
def test_non_monotonic_fail(load_clean_df):
    from core.data_sanity import DataSanityValidator, DataSanityError

    df = inject_non_monotonic(load_clean_df("SPY", size="tiny"))
    df.index = df.index.tz_localize("UTC")
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "SPY")


