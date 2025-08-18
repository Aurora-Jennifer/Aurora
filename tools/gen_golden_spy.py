import numpy as np
import pandas as pd
from pathlib import Path


def main():
    rng = np.random.default_rng(42)
    n = 180
    r = rng.normal(0.0002, 0.01, n)
    px = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=n, tz="UTC")
    df = pd.DataFrame({"Close": px}, index=idx)
    Path("tests/golden").mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path("tests/golden/SPY.parquet"))
    print("wrote tests/golden/SPY.parquet")


if __name__ == "__main__":
    main()


