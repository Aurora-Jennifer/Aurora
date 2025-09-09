#!/usr/bin/env python3
"""
Build sector snapshot for consistent residualization.
Creates a per-date sector mapping that's frozen for historical consistency.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timedelta

def create_mock_sector_snapshot(universe_size=300, start_date="2020-01-01", end_date="2025-09-06"):
    """Create a mock sector snapshot for development."""
    
    # Mock sectors (GICS-like)
    sectors = [
        'Technology', 'Healthcare', 'Financials', 'Consumer_Discretionary',
        'Communication_Services', 'Industrials', 'Consumer_Staples', 
        'Energy', 'Utilities', 'Real_Estate', 'Materials'
    ]
    
    # Generate mock symbols
    symbols = [f"SYM{i:03d}" for i in range(universe_size)]
    
    # Create date range
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Assign sectors randomly but consistently
    np.random.seed(42)  # For reproducibility
    symbol_sectors = {}
    for symbol in symbols:
        # Most symbols stay in same sector, but allow some changes
        base_sector = np.random.choice(sectors)
        symbol_sectors[symbol] = base_sector
    
    # Create the snapshot
    snapshot_data = []
    for date in dates:
        for symbol in symbols:
            # 99.5% chance to keep same sector, 0.5% chance to change
            if np.random.random() < 0.995:
                sector = symbol_sectors[symbol]
            else:
                sector = np.random.choice(sectors)
                symbol_sectors[symbol] = sector  # Update for consistency
            
            snapshot_data.append({
                'date': date,
                'symbol': symbol,
                'sector': sector
            })
    
    df = pd.DataFrame(snapshot_data)
    return df

def generate_snapshot_hash(df):
    """Generate a content hash for the snapshot."""
    content = df.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(content).hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser(description='Build sector snapshot')
    parser.add_argument('--universe', default='config/universe_top300.yaml', help='Universe file')
    parser.add_argument('--out', required=True, help='Output parquet file')
    parser.add_argument('--universe-size', type=int, default=300, help='Number of symbols')
    
    args = parser.parse_args()
    
    print("🏗️  BUILDING SECTOR SNAPSHOT")
    print("=" * 35)
    
    # Create mock snapshot for development
    print(f"📊 Creating snapshot for {args.universe_size} symbols...")
    df = create_mock_sector_snapshot(universe_size=args.universe_size)
    
    print(f"✅ Generated {len(df):,} sector mappings")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    print(f"   Sectors: {df['sector'].nunique()}")
    
    # Generate hash
    content_hash = generate_snapshot_hash(df)
    print(f"🔐 Content hash: {content_hash}")
    
    # Save to parquet
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'content_hash': content_hash,
        'num_symbols': df['symbol'].nunique(),
        'num_sectors': df['sector'].nunique(),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        },
        'sectors': sorted(df['sector'].unique().tolist())
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Metadata saved to: {metadata_path}")
    
    print("\n✅ SECTOR SNAPSHOT BUILD COMPLETE")
    
    return content_hash

if __name__ == "__main__":
    main()
