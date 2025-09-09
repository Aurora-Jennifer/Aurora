#!/usr/bin/env python3
"""
Build fundamentals snapshot with schema-correct empty data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def create_empty_fundamentals_schema():
    """Create an empty but schema-correct fundamentals dataset."""
    
    # Define expected fundamentals schema
    columns = [
        'date', 'symbol',
        'market_cap', 'enterprise_value', 'pe_ratio', 'pb_ratio',
        'debt_to_equity', 'current_ratio', 'roe', 'roa',
        'revenue_growth', 'earnings_growth', 'gross_margin', 'operating_margin',
        'dividend_yield', 'payout_ratio', 'shares_outstanding',
        'book_value_per_share', 'cash_per_share', 'revenue_per_share'
    ]
    
    # Create empty DataFrame with correct dtypes
    df = pd.DataFrame(columns=columns)
    
    # Set proper dtypes
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = df['symbol'].astype('string')
    
    for col in columns[2:]:  # All numeric columns
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_mock_fundamentals(symbols, start_date="2020-01-01", end_date="2025-09-06"):
    """Create mock fundamentals data for development."""
    
    # Quarterly reporting dates
    dates = pd.date_range(start_date, end_date, freq='Q')
    
    all_fundamentals = []
    np.random.seed(42)  # For reproducibility
    
    for symbol in symbols[:50]:  # Limit to first 50 for demo
        base_market_cap = np.random.uniform(1e9, 100e9)  # $1B to $100B
        
        for date in dates:
            # Generate correlated but noisy fundamentals
            market_cap = base_market_cap * np.random.uniform(0.5, 2.0)
            pe_ratio = np.random.uniform(5, 50)
            pb_ratio = np.random.uniform(0.5, 5.0)
            
            all_fundamentals.append({
                'date': date.date(),
                'symbol': symbol,
                'market_cap': market_cap,
                'enterprise_value': market_cap * np.random.uniform(0.8, 1.3),
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'debt_to_equity': np.random.uniform(0.1, 2.0),
                'current_ratio': np.random.uniform(0.8, 3.0),
                'roe': np.random.uniform(0.05, 0.25),
                'roa': np.random.uniform(0.02, 0.15),
                'revenue_growth': np.random.uniform(-0.2, 0.3),
                'earnings_growth': np.random.uniform(-0.5, 0.5),
                'gross_margin': np.random.uniform(0.2, 0.6),
                'operating_margin': np.random.uniform(0.05, 0.3),
                'dividend_yield': np.random.uniform(0.0, 0.06),
                'payout_ratio': np.random.uniform(0.0, 0.8),
                'shares_outstanding': market_cap / np.random.uniform(20, 200),
                'book_value_per_share': np.random.uniform(10, 100),
                'cash_per_share': np.random.uniform(1, 50),
                'revenue_per_share': np.random.uniform(50, 500)
            })
    
    df = pd.DataFrame(all_fundamentals)
    return df

def main():
    parser = argparse.ArgumentParser(description='Build fundamentals snapshot')
    parser.add_argument('--out', required=True, help='Output parquet file')
    parser.add_argument('--mock-data', action='store_true', help='Include mock data')
    parser.add_argument('--universe-size', type=int, default=300, help='Number of symbols for mock data')
    
    args = parser.parse_args()
    
    print("📊 BUILDING FUNDAMENTALS SNAPSHOT")
    print("=" * 38)
    
    if args.mock_data:
        # Create mock symbols
        symbols = [f"SYM{i:03d}" for i in range(args.universe_size)]
        print(f"📈 Creating mock fundamentals for {len(symbols)} symbols...")
        df = create_mock_fundamentals(symbols)
        print(f"✅ Generated {len(df):,} fundamental records")
    else:
        # Create empty schema-correct DataFrame
        print("📋 Creating empty schema-correct fundamentals...")
        df = create_empty_fundamentals_schema()
        print("✅ Empty fundamentals schema created")
    
    # Save to parquet
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    
    # Show schema
    print(f"\n📋 Schema:")
    print(f"   Columns: {len(df.columns)}")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"   {col:<25} {dtype}")
    
    if len(df) > 0:
        print(f"\n📊 Sample data:")
        print(df.head(3).to_string(index=False))
    
    print("\n✅ FUNDAMENTALS SNAPSHOT COMPLETE")

if __name__ == "__main__":
    main()
