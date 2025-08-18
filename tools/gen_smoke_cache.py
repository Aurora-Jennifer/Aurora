#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path


def synth(series_len: int = 80, start: float = 100.0, drift: float = -0.0005, vol: float = 0.01, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, series_len)
    px = start * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=series_len, tz="UTC")
    return pd.DataFrame({"Close": px}, index=idx)


def main() -> int:
    out_dir = Path("data/smoke_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in ["SPY", "TSLA", "BTC-USD"]:
        df = synth()
        df.to_parquet(out_dir / f"{s}.parquet")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


