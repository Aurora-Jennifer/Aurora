#!/usr/bin/env python3
"""
Process Kaggle ETF/Stocks dataset for Aurora trading system.
Convert text files to parquet format with proper timezone handling.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_symbol_data(file_path: Path) -> Tuple[str, pd.DataFrame]:
    """Load and clean data for a single symbol."""
    symbol = file_path.stem.replace('.us', '').upper()
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Clean up column names and data
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Add timezone (assume US market data is in US/Eastern, convert to UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('US/Eastern', ambiguous='infer').tz_convert('UTC')
        
        # Clean data: remove rows with NaN or zero prices
        df = df.dropna()
        df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
        df = df[df['Volume'] > 0]
        
        # Basic sanity checks
        df = df[df['High'] >= df['Low']]
        df = df[df['High'] >= df['Open']]
        df = df[df['High'] >= df['Close']]
        df = df[df['Low'] <= df['Open']]
        df = df[df['Low'] <= df['Close']]
        
        # Keep only OHLCV
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return symbol, df
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def filter_quality_symbols(data_dir: Path, min_days: int = 252, min_volume: float = 100000) -> List[Path]:
    """Filter for high-quality symbols with sufficient history and volume."""
    quality_symbols = []
    
    for category in ['ETFs', 'Stocks']:
        category_dir = data_dir / category
        if not category_dir.exists():
            continue
            
        for file_path in category_dir.glob('*.txt'):
            symbol, df = load_symbol_data(file_path)
            if df is None or len(df) < min_days:
                continue
                
            # Check average volume
            avg_volume = df['Volume'].mean()
            if avg_volume < min_volume:
                continue
                
            # Check recent data (within last 2 years)
            latest_date = df.index.max()
            if latest_date < pd.Timestamp('2017-01-01', tz='UTC'):
                continue
                
            quality_symbols.append(file_path)
            
    return quality_symbols

def main():
    """Process Kaggle dataset and create parquet files for Aurora system."""
    
    # Paths
    kaggle_dir = Path("data/kaggle/Data")
    output_dir = Path("data/training/kaggle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not kaggle_dir.exists():
        print(f"❌ Kaggle data directory not found: {kaggle_dir}")
        return 1
        
    print(f"🔍 Scanning {kaggle_dir} for quality symbols...")
    
    # Filter for quality symbols
    quality_files = filter_quality_symbols(kaggle_dir, min_days=252, min_volume=50000)
    print(f"📊 Found {len(quality_files)} quality symbols")
    
    # Process symbols
    processed_count = 0
    failed_count = 0
    
    # Create manifest file
    manifest = []
    
    for file_path in quality_files[:100]:  # Limit to first 100 for initial processing
        symbol, df = load_symbol_data(file_path)
        
        if df is None or len(df) < 252:
            failed_count += 1
            continue
            
        # Save as parquet
        output_file = output_dir / f"{symbol}.parquet"
        df.to_parquet(output_file)
        
        # Add to manifest
        manifest.append({
            'symbol': symbol,
            'file': f"{symbol}.parquet",
            'start_date': df.index.min().isoformat(),
            'end_date': df.index.max().isoformat(),
            'bars': len(df),
            'avg_volume': int(df['Volume'].mean()),
            'category': 'ETF' if 'ETFs' in str(file_path) else 'Stock'
        })
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"✅ Processed {processed_count} symbols...")
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    
    print(f"🎉 Processing complete!")
    print(f"   ✅ Processed: {processed_count} symbols")
    print(f"   ❌ Failed: {failed_count} symbols")
    print(f"   📁 Output: {output_dir}")
    print(f"   📋 Manifest: {output_dir}/manifest.csv")
    
    # Show sample of what we processed
    print(f"\n📈 Sample symbols:")
    for i, row in manifest_df.head(10).iterrows():
        print(f"   {row['symbol']:6s} ({row['category']}) - {row['bars']:4d} bars, avg vol: {row['avg_volume']:>10,}")
    
    return 0

if __name__ == "__main__":
    exit(main())
