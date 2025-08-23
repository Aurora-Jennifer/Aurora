import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator


def test_mixed_dtypes_rejected_strict():
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 101, 5),
            "High": np.linspace(101, 102, 5),
            "Low": np.linspace(99, 100, 5),
            "Close": ["100.5", "100.7", "100.9", "101.1", "101.3"],  # strings
            "Volume": np.full(5, 1_000_000),
        },
        index=dates,
    )
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "DTYPE_DRIFT_TEST")


