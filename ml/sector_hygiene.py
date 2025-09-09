"""
Sector snapshot hygiene for production trading.

Ensures sector mappings are frozen per-date and prevents historical drift.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib
import json
import os
from pathlib import Path
import warnings
from datetime import datetime


class SectorSnapshotManager:
    """
    Manage sector mappings with per-date snapshots and integrity checks.
    """
    
    def __init__(self, snapshot_dir: str = "snapshots"):
        """
        Initialize sector snapshot manager.
        
        Args:
            snapshot_dir: Directory to store sector snapshots
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.sector_cache = {}
        
    def create_sector_snapshot(self, 
                              data: pd.DataFrame,
                              date_col: str = 'date',
                              symbol_col: str = 'symbol', 
                              sector_col: str = 'sector') -> str:
        """
        Create sector snapshot from data.
        
        Args:
            data: DataFrame with sector mappings
            date_col: Date column name
            symbol_col: Symbol column name  
            sector_col: Sector column name
            
        Returns:
            Content hash of the snapshot
        """
        # Create per-date sector mapping
        sector_snapshot = data[[date_col, symbol_col, sector_col]].copy()
        sector_snapshot = sector_snapshot.drop_duplicates()
        
        # Sort for deterministic output
        sector_snapshot = sector_snapshot.sort_values([date_col, symbol_col])
        
        # Save snapshot
        snapshot_path = self.snapshot_dir / "sector_map.parquet"
        sector_snapshot.to_parquet(snapshot_path, index=False)
        
        # Compute content hash
        content_hash = self._compute_snapshot_hash(sector_snapshot)
        
        # Save hash
        hash_path = self.snapshot_dir / "sector_map.hash"
        with open(hash_path, 'w') as f:
            f.write(content_hash)
            
        print(f"✅ Sector snapshot created: {len(sector_snapshot)} mappings")
        print(f"   Snapshot: {snapshot_path}")
        print(f"   Hash: {content_hash[:12]}...")
        
        return content_hash
    
    def load_sector_snapshot(self) -> Tuple[pd.DataFrame, str]:
        """
        Load sector snapshot with integrity check.
        
        Returns:
            Tuple of (sector_data, content_hash)
            
        Raises:
            FileNotFoundError: If snapshot doesn't exist
            ValueError: If integrity check fails
        """
        snapshot_path = self.snapshot_dir / "sector_map.parquet"
        hash_path = self.snapshot_dir / "sector_map.hash"
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Sector snapshot not found: {snapshot_path}")
            
        # Load snapshot
        sector_data = pd.read_parquet(snapshot_path)
        
        # Load expected hash
        if hash_path.exists():
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
        else:
            expected_hash = None
            
        # Verify integrity
        current_hash = self._compute_snapshot_hash(sector_data)
        
        if expected_hash and current_hash != expected_hash:
            raise ValueError(f"Sector snapshot integrity check failed: "
                           f"{current_hash[:12]}... != {expected_hash[:12]}...")
        
        return sector_data, current_hash
    
    def validate_historical_consistency(self, 
                                      new_data: pd.DataFrame,
                                      date_col: str = 'date',
                                      symbol_col: str = 'symbol',
                                      sector_col: str = 'sector') -> Dict:
        """
        Validate that historical sector mappings haven't changed.
        
        Args:
            new_data: New data to validate
            date_col: Date column name
            symbol_col: Symbol column name
            sector_col: Sector column name
            
        Returns:
            Dict with validation results
        """
        try:
            snapshot_data, snapshot_hash = self.load_sector_snapshot()
        except FileNotFoundError:
            return {
                'status': 'no_baseline',
                'message': 'No sector snapshot found for comparison'
            }
        
        # Prepare data for comparison
        new_sectors = new_data[[date_col, symbol_col, sector_col]].drop_duplicates()
        snapshot_sectors = snapshot_data.copy()
        
        # Find overlapping date-symbol pairs
        new_keys = set(zip(new_sectors[date_col], new_sectors[symbol_col]))
        snapshot_keys = set(zip(snapshot_sectors['date'], snapshot_sectors['symbol']))
        
        overlapping_keys = new_keys & snapshot_keys
        
        if not overlapping_keys:
            return {
                'status': 'no_overlap',
                'message': 'No overlapping date-symbol pairs to validate'
            }
        
        # Check for changes in overlapping mappings
        changes = []
        
        for date, symbol in overlapping_keys:
            # Get sector from snapshot
            snapshot_sector = snapshot_sectors[
                (snapshot_sectors['date'] == date) & 
                (snapshot_sectors['symbol'] == symbol)
            ]['sector'].iloc[0]
            
            # Get sector from new data
            new_sector = new_sectors[
                (new_sectors[date_col] == date) & 
                (new_sectors[symbol_col] == symbol)
            ][sector_col].iloc[0]
            
            if snapshot_sector != new_sector:
                changes.append({
                    'date': date,
                    'symbol': symbol,
                    'snapshot_sector': snapshot_sector,
                    'new_sector': new_sector
                })
        
        # Return validation results
        if changes:
            return {
                'status': 'changes_detected',
                'changes': changes,
                'overlapping_pairs': len(overlapping_keys),
                'changed_pairs': len(changes),
                'message': f'{len(changes)} historical sector changes detected'
            }
        else:
            return {
                'status': 'consistent',
                'overlapping_pairs': len(overlapping_keys),
                'message': 'All historical sector mappings consistent'
            }
    
    def _compute_snapshot_hash(self, data: pd.DataFrame) -> str:
        """Compute SHA256 hash of sector snapshot."""
        # Sort for deterministic hashing
        sorted_data = data.sort_values(['date', 'symbol'])
        
        # Create string representation
        content = sorted_data.to_csv(index=False)
        
        # Compute hash
        return hashlib.sha256(content.encode()).hexdigest()


def enforce_sector_hygiene(data: pd.DataFrame, 
                          snapshot_manager: SectorSnapshotManager,
                          strict_mode: bool = True) -> pd.DataFrame:
    """
    Enforce sector hygiene by validating against frozen snapshot.
    
    Args:
        data: DataFrame with sector data
        snapshot_manager: Sector snapshot manager instance
        strict_mode: If True, fail on any changes; if False, warn only
        
    Returns:
        Data with validated sector mappings
        
    Raises:
        ValueError: If strict_mode=True and changes detected
    """
    # Validate historical consistency
    validation_result = snapshot_manager.validate_historical_consistency(data)
    
    if validation_result['status'] == 'changes_detected':
        message = f"SECTOR HYGIENE VIOLATION: {validation_result['message']}"
        
        # Log details of changes
        for change in validation_result['changes'][:5]:  # Show first 5
            print(f"   {change['date']} {change['symbol']}: "
                  f"{change['snapshot_sector']} → {change['new_sector']}")
        
        if len(validation_result['changes']) > 5:
            print(f"   ... and {len(validation_result['changes']) - 5} more changes")
        
        if strict_mode:
            raise ValueError(message)
        else:
            warnings.warn(message)
    
    elif validation_result['status'] == 'consistent':
        print(f"✅ Sector hygiene check passed: {validation_result['overlapping_pairs']} mappings validated")
    
    return data


def create_mock_sector_data(symbols: List[str], 
                           start_date: str = '2024-01-01',
                           end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Create mock sector data for testing.
    
    Args:
        symbols: List of symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with mock sector mappings
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Define realistic sectors
    sectors = [
        'Technology', 'Healthcare', 'Financials', 'Consumer_Discretionary',
        'Communication_Services', 'Industrials', 'Consumer_Staples',
        'Energy', 'Utilities', 'Real_Estate', 'Materials'
    ]
    
    data = []
    np.random.seed(42)
    
    for symbol in symbols:
        # Assign sector (stays constant for each symbol)
        sector = np.random.choice(sectors)
        
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'sector': sector
            })
    
    return pd.DataFrame(data)


def run_sector_hygiene_test():
    """Test sector hygiene functionality."""
    print("🧪 TESTING SECTOR SNAPSHOT HYGIENE")
    print("="*50)
    
    # Create test data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    sector_data = create_mock_sector_data(symbols, '2024-01-01', '2024-01-10')
    
    print(f"✅ Created mock sector data: {len(sector_data)} mappings")
    
    # Initialize manager
    manager = SectorSnapshotManager("test_snapshots")
    
    # Create snapshot
    snapshot_hash = manager.create_sector_snapshot(sector_data)
    
    # Test validation with same data (should pass)
    print(f"\n🧪 TESTING CONSISTENCY VALIDATION:")
    validation_result = manager.validate_historical_consistency(sector_data)
    print(f"   Status: {validation_result['status']}")
    print(f"   Message: {validation_result['message']}")
    
    # Test with modified data (should detect changes)
    print(f"\n🧪 TESTING CHANGE DETECTION:")
    modified_data = sector_data.copy()
    # Change AAPL's sector for one date
    mask = (modified_data['symbol'] == 'AAPL') & (modified_data['date'] == pd.Timestamp('2024-01-05'))
    modified_data.loc[mask, 'sector'] = 'Modified_Sector'
    
    try:
        enforce_sector_hygiene(modified_data, manager, strict_mode=True)
        print(f"   ❌ Should have failed on sector change")
    except ValueError as e:
        print(f"   ✅ Correctly detected sector change: {str(e)[:50]}...")
    
    # Test with strict_mode=False (should warn)
    print(f"\n🧪 TESTING WARNING MODE:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enforce_sector_hygiene(modified_data, manager, strict_mode=False)
        if w:
            print(f"   ✅ Warning issued: {w[0].message}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_snapshots"):
        shutil.rmtree("test_snapshots")
    
    print(f"\n✅ Sector hygiene test completed")


if __name__ == "__main__":
    run_sector_hygiene_test()
