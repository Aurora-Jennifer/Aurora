import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator


def test_lookahead_contamination_rejected_strict():
    dates = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    df = pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.randint(1000000, 2000000, len(dates)),
        },
        index=dates,
    )
    # Inject lookahead contamination in a derived column
    df["Returns"] = pd.Series(prices).pct_change().to_numpy()
    df.loc[dates[10], "Returns"] = df.loc[dates[11], "Returns"]  # future leak

    v = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        v.validate_and_repair(df, "LOOKAHEAD_TEST")


