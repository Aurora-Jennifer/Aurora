"""
Sector residualization for cross-sectional features.
Matches the exact method used in training.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def load_sector_map(sector_file: str = "data/sectors/sector_map.parquet") -> pd.DataFrame:
    """
    Load sector classifications for symbols.
    
    Args:
        sector_file: Path to sector mapping file
        
    Returns:
        DataFrame with symbol, date, sector columns
    """
    try:
        from pathlib import Path
        
        if Path(sector_file).exists():
            df = pd.read_parquet(sector_file)
            logger.info(f"✅ Loaded sector map: {len(df)} records")
            return df
        else:
            logger.warning(f"Sector file not found: {sector_file}")
            return create_mock_sector_map()
            
    except Exception as e:
        logger.warning(f"Error loading sector map: {e}")
        return create_mock_sector_map()

def create_mock_sector_map() -> pd.DataFrame:
    """
    Create a mock sector map for testing.
    In production, this should be replaced with real GICS/ICB classifications.
    """
    logger.warning("Creating mock sector map - replace with real GICS data")
    
    # Mock sectors for common symbols
    sector_map = {
        'AAPL': 'Technology',
        'MSFT': 'Technology', 
        'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'TSLA': 'Consumer Discretionary',
        'META': 'Technology',
        'NVDA': 'Technology',
        'NFLX': 'Communication Services',
        'AMD': 'Technology',
        'INTC': 'Technology',
        'JPM': 'Financials',
        'JNJ': 'Healthcare',
        'V': 'Financials',
        'PG': 'Consumer Staples',
        'UNH': 'Healthcare'
    }
    
    # Create DataFrame with all symbols mapped to sectors
    data = []
    for symbol, sector in sector_map.items():
        data.append({
            'symbol': symbol,
            'sector': sector,
            'date': pd.Timestamp('2024-01-01')  # Single date for simplicity
        })
    
    return pd.DataFrame(data)

def sector_residualize(df: pd.DataFrame, feature_cols: List[str], 
                      sector_map: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Apply sector residualization to cross-sectional features.
    
    This matches the exact method used in training:
    - For each date, group by sector
    - Calculate sector mean for each feature
    - Subtract sector mean to get residuals
    - Add _sec_res suffix to residualized features
    
    Args:
        df: DataFrame with features to residualize
        feature_cols: List of feature column names
        sector_map: DataFrame with symbol, sector, date mappings
        date_col: Name of date column
        
    Returns:
        DataFrame with original features + residualized features
    """
    df = df.copy()
    
    # Ensure timezone consistency for merging
    if date_col in df.columns and date_col in sector_map.columns:
        # Convert both to the same timezone
        if df[date_col].dt.tz is not None and sector_map[date_col].dt.tz is None:
            sector_map[date_col] = sector_map[date_col].dt.tz_localize('UTC')
        elif df[date_col].dt.tz is None and sector_map[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_localize('UTC')
    
    # Merge with sector map
    df_with_sectors = df.merge(sector_map, on=['symbol', date_col], how='left')
    
    # Fill missing sectors with 'Unknown'
    df_with_sectors['sector'] = df_with_sectors['sector'].fillna('Unknown')
    
    logger.info(f"Residualizing {len(feature_cols)} features across {df_with_sectors['sector'].nunique()} sectors")
    
    # Apply sector residualization for each feature
    for feature in feature_cols:
        if feature in df_with_sectors.columns:
            # Calculate sector means by date
            sector_means = df_with_sectors.groupby([date_col, 'sector'])[feature].transform('mean')
            
            # Calculate residuals (feature - sector_mean)
            residuals = df_with_sectors[feature] - sector_means
            
            # Add residualized feature with _sec_res suffix
            df_with_sectors[f"{feature}_sec_res"] = residuals
            
            logger.debug(f"   ✅ {feature} → {feature}_sec_res")
        else:
            logger.warning(f"   ⚠️ Feature {feature} not found in DataFrame")
    
    # Drop the sector column (keep original structure)
    df_with_sectors = df_with_sectors.drop('sector', axis=1)
    
    return df_with_sectors

def get_residualized_features(df: pd.DataFrame, base_features: List[str]) -> List[str]:
    """
    Get the list of residualized feature names.
    
    Args:
        df: DataFrame that may contain residualized features
        base_features: List of base feature names
        
    Returns:
        List of residualized feature names that exist in the DataFrame
    """
    residualized = []
    for feature in base_features:
        residualized_name = f"{feature}_sec_res"
        if residualized_name in df.columns:
            residualized.append(residualized_name)
    
    return residualized
