#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    })
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="SPY,QQQ")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2020-12-31")
    args = p.parse_args()

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        df = fetch_symbol(sym, args.start, args.end)
        path = outdir / f"{sym}.parquet"
        df.to_parquet(path)
        print(f"saved: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()


