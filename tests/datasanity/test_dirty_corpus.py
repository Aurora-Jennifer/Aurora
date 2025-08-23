import json
import pathlib

import pandas as pd
import pytest

DIR = pathlib.Path("corpora/dirty")
CASES = []
for p in DIR.glob("*.json"):
    with open(p) as f:
        CASES.append(json.load(f))


@pytest.mark.integration
@pytest.mark.quarantine
@pytest.mark.parametrize("meta", CASES, ids=lambda m: m["name"])
def test_dirty_case(meta, datasanity):
    df = pd.read_csv(meta["csv"], parse_dates=["Timestamp"]).set_index("Timestamp")
    if meta["expect"] == "pass":
        datasanity.validate(df)
    else:
        with pytest.raises(datasanity.Error) as e:
            datasanity.validate(df)
        msg = str(e.value)
        for code in meta["codes"]:
            assert code in msg


