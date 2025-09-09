#!/usr/bin/env python3
import argparse, pathlib as p
import yfinance as yf, yaml
import numpy as np, pandas as pd

ETF_LIKE = {"SPY","VOO","IVV","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB",
            "XME","XOP","SOXX","SMH","HYG","LQD","EEM","EFA","TLT","IEF","SHY","GLD","SLV","GDX","GDXJ",
            "VXX","UVXY","SVXY","UUP","VIXY"}

def read_seed(path):
    with open(path) as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    out, seen = [], set()
    for line in raw:
        # Split on whitespace to get individual tickers
        tickers = line.split()
        for t in tickers:
            t = t.upper()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out

def looks_like_etf(t): return t.startswith("^") or t.endswith("=F") or t in ETF_LIKE

def main(a):
    tickers = [t for t in read_seed(a.seed) if not looks_like_etf(t)]
    if not tickers:
        raise SystemExit("Seed list empty after ETF filtering")

    rows = []
    for t in tickers:
        try:
            df = yf.download(t, period=f"{a.lookback+7}d", interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty: continue
            df = df.tail(a.lookback)
            if df.empty or "Close" not in df or "Volume" not in df: continue
            last_close = float(df["Close"].iloc[-1])
            if last_close < a.min_price: continue
            adv = float((df["Close"] * df["Volume"]).mean())
            if adv < a.min_dollar_vol: continue
            rows.append((t, adv, last_close))
        except Exception:
            continue

    rows.sort(key=lambda x: x[1], reverse=True)
    top = [t for t,_,_ in rows[:a.n]]

    out_cfg = {
        "name": f"top_traded_{a.n}",
        "market_proxy": a.market_proxy,
        "cross_proxies": a.cross_proxies,
        "universe": top
    }
    p.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print(f"Saved {len(top)} tickers to {a.out}")
    if rows:
        mx = max(rows, key=lambda r: r[1])[1]
        mn = min(rows, key=lambda r: r[1])[1]
        print(f"$ADV range (selected): ~${mn:,.0f} .. ${mx:,.0f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--min-price", type=float, default=3.0)
    ap.add_argument("--min-dollar-vol", type=float, default=10_000_000.0)
    ap.add_argument("--market-proxy", default="QQQ")
    ap.add_argument("--cross-proxies", nargs="*", default=["QQQ","TLT","UUP","VIXY"])
    ap.add_argument("--out", default="config/universe_top300.yaml")
    main(ap.parse_args())