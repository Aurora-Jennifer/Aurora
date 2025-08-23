from pathlib import Path

import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator


def _load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")
    return df


def test_extreme_price_rejected_strict():
    v = DataSanityValidator(profile="strict")
    df = _load_csv(Path("tests/golden/violations/extreme_prices.csv"))
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "EXTREME_PRICE_TEST")


def test_negative_price_rejected_strict():
    v = DataSanityValidator(profile="strict")
    df = _load_csv(Path("tests/golden/violations/negative_prices.csv"))
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "NEG_PRICE_TEST")


