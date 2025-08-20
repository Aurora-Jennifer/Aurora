import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityValidator, DataSanityError


def test_nan_inf_rejected_strict():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 101, 10),
            "High": np.linspace(101, 102, 10),
            "Low": np.linspace(99, 100, 10),
            "Close": np.linspace(100, 101, 10),
            "Volume": np.linspace(1_000_000, 1_100_000, 10),
        },
        index=dates,
    )
    df.loc[dates[3], "Close"] = np.nan
    df.loc[dates[5], "Volume"] = np.inf
    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "NONFINITE_TEST")


