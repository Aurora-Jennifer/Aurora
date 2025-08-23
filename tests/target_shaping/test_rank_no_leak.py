import numpy as np
import pandas as pd

from core.ml.shape_labels import shape_rank_labels


def test_rank_uses_no_future():
    s = pd.Series(np.arange(300, dtype=float))
    raw1, r1 = shape_rank_labels(s, {"rank_window": 50, "options": {"rank_type": "time_series", "rank_method": "percentile"}})
    s2 = s.copy()
    s2.iloc[-1] += 999
    raw2, r2 = shape_rank_labels(s2, {"rank_window": 50, "options": {"rank_type": "time_series", "rank_method": "percentile"}})
    pd.testing.assert_series_equal(r1.iloc[:-1], r2.iloc[:-1], check_names=False)


