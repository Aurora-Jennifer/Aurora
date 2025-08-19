"""
Generate deterministic test fixtures for Aurora trading system.

Creates realistic price data for testing without external dependencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json


def generate_test_data(
    symbols: List[str] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42
) -> None:
    """
    Generate deterministic test data for specified symbols.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date for data generation
        end_date: End date for data generation
        seed: Random seed for reproducibility
    """
    if symbols is None:
        symbols = ["SPY", "TSLA"]
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Remove weekends (simple approach)
    date_range = date_range[date_range.dayofweek < 5]
    
    # Generate data for each symbol
    all_data = []
    
    for symbol in symbols:
        symbol_data = generate_symbol_data(symbol, date_range, seed)
        all_data.append(symbol_data)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp and symbol
    combined_df = combined_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    # Save to parquet
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)
    
    output_path = fixtures_dir / "quotes.parquet"
    combined_df.to_parquet(output_path, index=False)
    
    # Also save as CSV for compatibility
    csv_path = fixtures_dir / "quotes.csv"
    combined_df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "seed": seed,
        "n_days": len(date_range),
        "n_records": len(combined_df),
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = fixtures_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Generated test data: {output_path}")
    print(f"   Records: {len(combined_df)}")
    print(f"   Symbols: {symbols}")
    print(f"   Date range: {start_date} to {end_date}")


def generate_symbol_data(symbol: str, date_range: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """
    Generate realistic price data for a single symbol.
    
    Args:
        symbol: Symbol name
        date_range: Date range for data
        seed: Random seed (modified per symbol)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Create symbol-specific seed
    symbol_seed = seed + hash(symbol) % 1000
    np.random.seed(symbol_seed)
    
    n_days = len(date_range)
    
    # Base prices (realistic starting points)
    base_prices = {
        "SPY": 400.0,
        "TSLA": 200.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    
    # Generate daily returns with realistic characteristics
    # SPY: lower volatility, more stable
    # TSLA: higher volatility, more erratic
    
    if symbol == "SPY":
        daily_returns = np.random.normal(0.0005, 0.015, n_days)  # ~15% annual vol
    elif symbol == "TSLA":
        daily_returns = np.random.normal(0.001, 0.035, n_days)   # ~35% annual vol
    else:
        daily_returns = np.random.normal(0.0008, 0.025, n_days)  # ~25% annual vol
    
    # Add some autocorrelation and clustering
    for i in range(1, n_days):
        daily_returns[i] += 0.1 * daily_returns[i-1]  # Some momentum
    
    # Generate price series
    prices = [base_price]
    for ret in daily_returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    prices = np.array(prices)
    
    # Generate OHLC data
    data = []
    
    for i, (date, close) in enumerate(zip(date_range, prices)):
        # Generate realistic OHLC from close
        volatility = 0.02 if symbol == "SPY" else 0.04
        
        # High and low relative to close
        high_offset = np.random.uniform(0, volatility * close)
        low_offset = np.random.uniform(0, volatility * close)
        
        high = close + high_offset
        low = max(close - low_offset, close * 0.95)  # Ensure low <= close
        
        # Open relative to previous close
        if i == 0:
            open_price = close * np.random.uniform(0.98, 1.02)
        else:
            prev_close = prices[i-1]
            open_price = prev_close * np.random.uniform(0.99, 1.01)
        
        # Volume (realistic ranges)
        base_volume = 1000000 if symbol == "SPY" else 50000000
        volume = int(base_volume * np.random.uniform(0.5, 2.0))
        
        data.append({
            'timestamp': date,
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


def create_minimal_fixture() -> None:
    """
    Create a minimal fixture for quick testing.
    """
    generate_test_data(
        symbols=["SPY"],
        start_date="2023-01-01",
        end_date="2023-03-31",  # ~3 months
        seed=42
    )


def create_comprehensive_fixture() -> None:
    """
    Create a comprehensive fixture for full testing.
    """
    generate_test_data(
        symbols=["SPY", "TSLA", "AAPL", "GOOGL"],
        start_date="2022-01-01",
        end_date="2023-12-31",  # 2 years
        seed=42
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "minimal":
            create_minimal_fixture()
        elif command == "comprehensive":
            create_comprehensive_fixture()
        else:
            print("Usage: python gen_fixture.py [minimal|comprehensive]")
    else:
        # Default: generate standard fixture
        generate_test_data()
