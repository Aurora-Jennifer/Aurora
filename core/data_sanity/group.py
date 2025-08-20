import pandas as pd


def enforce_groupwise_time_order(df, symbol_col=None):
    """
    Enforce groupwise time order, especially for symbol-grouped data.
    
    Args:
        df: DataFrame to sort
        symbol_col: Optional symbol column name for grouped data
        
    Returns:
        DataFrame: Sorted and deduplicated DataFrame
    """
    if symbol_col and symbol_col in df.columns:
        df = df.sort_values([symbol_col, df.index.name])
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df
