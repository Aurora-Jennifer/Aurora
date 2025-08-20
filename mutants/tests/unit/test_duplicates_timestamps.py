import pandas as pd
from pathlib import Path

from core.data_sanity import DataSanityValidator, DataSanityError


def test_duplicate_timestamps_flagged():
    p = Path("tests/golden/violations/dup_timestamps.csv")
    df = pd.read_csv(p, parse_dates=["timestamp"])  # UTC in data
    df = df.set_index("timestamp")
    v = DataSanityValidator(profile="strict")
    try:
        v.validate_and_repair(df, "DUP_TS_TEST")
        raised = False
    except DataSanityError as e:
        raised = True
        msg = str(e)
        assert "duplicate" in msg.lower()
    assert raised, "Expected DataSanityError for duplicate timestamps"


