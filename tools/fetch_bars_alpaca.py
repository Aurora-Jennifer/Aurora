#!/usr/bin/env python3
"""
Fetch historical bars from Alpaca and save to parquet.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
from datetime import datetime, timedelta
import requests
import json

# Symbol normalization map for Alpaca API
NORMALIZE_MAP = {
    "BRK-B": "BRK.B",
    "BF-B": "BF.B", 
    "ANTM": "ELV",   # renamed
    "HCP": "PEAK",   # renamed
    "FB": "META",    # renamed long ago
}

def normalize_symbol(s: str) -> str:
    """Normalize symbol names for Alpaca API compatibility."""
    s = s.strip().upper()
    if "-" in s and s not in NORMALIZE_MAP:
        # common: convert hyphen to dot if no explicit mapping
        return s.replace("-", ".")
    return NORMALIZE_MAP.get(s, s)

def chunks(lst, n): 
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n): 
        yield lst[i:i+n]

def fetch_alpaca_bars(symbols, start_date, end_date, timeframe="1Day", feed="iex", allow_mock=False):
    """Fetch bars from Alpaca API with batching."""
    
    # Set allow_mock flag for the function
    fetch_alpaca_bars.allow_mock = allow_mock
    
    api_key = os.environ.get('APCA_API_KEY_ID')
    secret_key = os.environ.get('APCA_API_SECRET_KEY')
    
    if not api_key or not secret_key:
        if not allow_mock:
            raise RuntimeError("Alpaca credentials not found and mock data not allowed.")
        print("⚠️  Alpaca credentials not found, creating mock data...")
        return create_mock_bars(symbols, start_date, end_date)
    
    print(f"📡 Fetching {len(symbols)} symbols from Alpaca...")
    
    # Alpaca API endpoint
    url = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key
    }
    
    all_bars = []
    
    # Batch symbols to avoid URL limits (max 50 per request)
    for chunk in chunks(symbols, 50):
        print(f"   📦 Fetching chunk: {len(chunk)} symbols")
        
        params = {
            'symbols': ','.join(chunk),
            'timeframe': timeframe,
            'start': start_date,
            'end': end_date,
            'feed': feed,
            'asof': None,
            'page_token': None
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for symbol, bars in data.get('bars', {}).items():
                for bar in bars:
                    all_bars.append({
                        'date': pd.to_datetime(bar['t']),
                        'symbol': symbol,
                        'open': float(bar['o']),
                        'high': float(bar['h']),
                        'low': float(bar['l']),
                        'close': float(bar['c']),
                        'volume': int(bar['v'])
                    })
                    
        except Exception as e:
            print(f"⚠️  Alpaca API error for chunk {chunk[:3]}...: {e}")
            if not getattr(fetch_alpaca_bars, 'allow_mock', False):
                raise RuntimeError(f"Fetch failed for symbols {chunk} (no mock allowed).")
            print("📊 Creating mock data for this chunk...")
            chunk_mock = create_mock_bars(chunk, start_date, end_date)
            all_bars.extend(chunk_mock.to_dict('records'))
    
    # Process all chunks before returning
    if all_bars:
        df = pd.DataFrame(all_bars)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
        return df.sort_values(['date', 'symbol']).reset_index(drop=True)
    else:
        if not allow_mock:
            raise RuntimeError(f"No data returned from Alpaca for symbols {symbols} (no mock allowed).")
        print("⚠️  No data returned from Alpaca, creating mock data...")
        return create_mock_bars(symbols, start_date, end_date)

def create_mock_bars(symbols, start_date, end_date):
    """Create mock OHLCV data for development."""
    import numpy as np, pandas as pd
    dates = pd.bdate_range(start_date, end_date)  # business days only
    if len(dates) == 0:
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date","symbol"])
    df = idx.to_frame(index=False)
    rng = np.random.default_rng(42)
    base = rng.uniform(10, 200, size=len(df))
    df["open"] = base
    df["high"] = df["open"] * (1 + rng.normal(0.002, 0.005, len(df)))
    df["low"]  = df["open"] * (1 - np.abs(rng.normal(0.002, 0.005, len(df))))
    df["close"]= df["open"] * (1 + rng.normal(0.000, 0.01,  len(df)))
    df["volume"] = rng.integers(100_000, 5_000_000, len(df))
    return df.sort_values(["date","symbol"]).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description='Fetch bars from Alpaca')
    parser.add_argument('--symbols-file', help='File with symbol list')
    parser.add_argument('--symbols', nargs='+', help='Symbols to fetch')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1Day', help='Timeframe (1Day, 1Hour, etc)')
    parser.add_argument('--feed', default='iex', help='Data feed (iex, sip)')
    parser.add_argument('--out', required=True, help='Output parquet file')
    parser.add_argument('--allow-mock', action='store_true', help='Allow mock data on API failure')
    
    args = parser.parse_args()
    
    # Auto-fill dates if not provided, using last trading day
    try:
        from ops.date_helpers import last_trading_day
        if not args.start or not args.end:
            d = last_trading_day()
            args.start = args.start or d
            args.end = args.end or d
    except ImportError:
        # Fallback if date_helpers not available
        if not args.start or not args.end:
            from datetime import date, timedelta
            today = date.today()
            # Go back to last weekday
            while today.weekday() > 4:  # Saturday=5, Sunday=6
                today -= timedelta(days=1)
            args.start = args.start or str(today)
            args.end = args.end or str(today)
    
    print("📊 FETCHING HISTORICAL BARS")
    print("=" * 30)
    
    # Get symbols
    if args.symbols_file and Path(args.symbols_file).exists():
        with open(args.symbols_file) as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default mock universe
        symbols = [f"SYM{i:03d}" for i in range(300)]
        print(f"📝 Using default mock symbols: {len(symbols)} symbols")
    
    # Normalize symbols for Alpaca API compatibility
    original_count = len(symbols)
    symbols = [normalize_symbol(s) for s in symbols]
    if len(symbols) != original_count:
        print(f"📝 Normalized {original_count} symbols for Alpaca API compatibility")
    
    print(f"📈 Fetching {len(symbols)} symbols")
    print(f"📅 Date range: {args.start} to {args.end}")
    print(f"⏰ Timeframe: {args.timeframe}")
    
    # Fetch data
    df = fetch_alpaca_bars(symbols, args.start, args.end, args.timeframe, args.feed, args.allow_mock)
    
    print(f"✅ Fetched {len(df):,} bars")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    
    # Sanity check: expected vs actual rows
    n_syms = len(set(symbols))
    n_days = len(pd.bdate_range(args.start, args.end))
    expected = n_syms * n_days
    actual = len(df)
    if actual != expected:
        missing = expected - actual
        print(f"⚠️ Missing {missing} rows ({expected=}, {actual=}). "
              "Check invalid symbols or API misses.", flush=True)
    
    # Save to parquet
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    
    # Sample data
    print(f"\n📋 Sample data:")
    print(df.head().to_string(index=False))
    
    print("\n✅ BARS FETCH COMPLETE")

if __name__ == "__main__":
    main()
