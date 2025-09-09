#!/usr/bin/env python3
"""
Create a comprehensive seed list of US stocks from major indices
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        return sp500_table['Symbol'].tolist()
    except Exception as e:
        print(f"Failed to get S&P 500 from Wikipedia: {e}")
        return []

def get_nasdaq_tickers():
    """Get NASDAQ 100 tickers"""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        nasdaq_table = tables[4]  # The table with tickers
        return nasdaq_table['Ticker'].tolist()
    except Exception as e:
        print(f"Failed to get NASDAQ 100 from Wikipedia: {e}")
        return []

def get_dow_tickers():
    """Get Dow Jones Industrial Average tickers"""
    try:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tables = pd.read_html(url)
        dow_table = tables[1]  # The table with current components
        return dow_table['Symbol'].tolist()
    except Exception as e:
        print(f"Failed to get Dow Jones from Wikipedia: {e}")
        return []

def main():
    print("📊 Creating comprehensive US stock seed list...")
    
    # Get tickers from major indices
    sp500 = get_sp500_tickers()
    nasdaq = get_nasdaq_tickers()
    dow = get_dow_tickers()
    
    # Combine and deduplicate
    all_tickers = set()
    all_tickers.update(sp500)
    all_tickers.update(nasdaq)
    all_tickers.update(dow)
    
    # Add some additional popular stocks
    additional = [
        "TSLA", "NVDA", "AMD", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "CSCO", "ORCL",
        "IBM", "QCOM", "AVGO", "TXN", "AMAT", "LRCX", "KLAC", "MCHP", "ADI", "MRVL",
        "COIN", "ROKU", "ZM", "DOCU", "SNOW", "PLTR", "CRWD", "OKTA", "NET", "DDOG",
        "SQ", "SHOP", "TWLO", "ESTC", "MDB", "WDAY", "VEEV", "NOW", "SPLK", "TEAM"
    ]
    all_tickers.update(additional)
    
    # Remove any None values and sort
    tickers = sorted([t for t in all_tickers if t and isinstance(t, str)])
    
    print(f"Found {len(tickers)} unique tickers")
    
    # Save to file
    seed_file = Path("config/seeds/usa_candidates.txt")
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(seed_file, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
    
    print(f"✅ Saved {len(tickers)} tickers to {seed_file}")
    print(f"First 20 tickers: {tickers[:20]}")
    
    return len(tickers)

if __name__ == "__main__":
    main()
