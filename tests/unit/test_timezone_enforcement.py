import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator


def test_naive_index_rejected_strict():
    # Build OHLCV with naive timestamps
    dates = pd.date_range("2023-01-01", periods=5, freq="D")  # naive
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 102, 5),
            "High": np.linspace(101, 103, 5),
            "Low": np.linspace(99, 101, 5),
            "Close": np.linspace(100.5, 102.5, 5),
            "Volume": np.full(5, 1_000_000),
        },
        index=dates,
    )
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "NAIVE_TZ_TEST")


