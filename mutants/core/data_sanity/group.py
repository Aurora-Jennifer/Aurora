import pandas as pd
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


def x_enforce_groupwise_time_order__mutmut_orig(df, symbol_col=None):
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


def x_enforce_groupwise_time_order__mutmut_1(df, symbol_col=None):
    """
    Enforce groupwise time order, especially for symbol-grouped data.
    
    Args:
        df: DataFrame to sort
        symbol_col: Optional symbol column name for grouped data
        
    Returns:
        DataFrame: Sorted and deduplicated DataFrame
    """
    if symbol_col or symbol_col in df.columns:
        df = df.sort_values([symbol_col, df.index.name])
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_2(df, symbol_col=None):
    """
    Enforce groupwise time order, especially for symbol-grouped data.
    
    Args:
        df: DataFrame to sort
        symbol_col: Optional symbol column name for grouped data
        
    Returns:
        DataFrame: Sorted and deduplicated DataFrame
    """
    if symbol_col and symbol_col not in df.columns:
        df = df.sort_values([symbol_col, df.index.name])
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_3(df, symbol_col=None):
    """
    Enforce groupwise time order, especially for symbol-grouped data.
    
    Args:
        df: DataFrame to sort
        symbol_col: Optional symbol column name for grouped data
        
    Returns:
        DataFrame: Sorted and deduplicated DataFrame
    """
    if symbol_col and symbol_col in df.columns:
        df = None
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_4(df, symbol_col=None):
    """
    Enforce groupwise time order, especially for symbol-grouped data.
    
    Args:
        df: DataFrame to sort
        symbol_col: Optional symbol column name for grouped data
        
    Returns:
        DataFrame: Sorted and deduplicated DataFrame
    """
    if symbol_col and symbol_col in df.columns:
        df = df.sort_values(None)
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_5(df, symbol_col=None):
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
        df = None
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_6(df, symbol_col=None):
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
        df = df[df.duplicated(subset=[symbol_col, df.index.name], keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_7(df, symbol_col=None):
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
        df = df[~df.duplicated(subset=None, keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_8(df, symbol_col=None):
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
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep=None)]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_9(df, symbol_col=None):
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
        df = df[~df.duplicated(keep="first")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_10(df, symbol_col=None):
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
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], )]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_11(df, symbol_col=None):
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
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="XXfirstXX")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_12(df, symbol_col=None):
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
        df = df[~df.duplicated(subset=[symbol_col, df.index.name], keep="FIRST")]
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_13(df, symbol_col=None):
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
    elif isinstance(df.index, pd.MultiIndex) or "timestamp" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_14(df, symbol_col=None):
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
    elif isinstance(df.index, pd.MultiIndex) and "XXtimestampXX" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_15(df, symbol_col=None):
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
    elif isinstance(df.index, pd.MultiIndex) and "TIMESTAMP" in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_16(df, symbol_col=None):
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
    elif isinstance(df.index, pd.MultiIndex) and "timestamp" not in df.index.names:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_17(df, symbol_col=None):
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
        df = None
        df = df[~df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_18(df, symbol_col=None):
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
        df = None
    return df


def x_enforce_groupwise_time_order__mutmut_19(df, symbol_col=None):
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
        df = df[df.index.duplicated(keep="first")]
    return df


def x_enforce_groupwise_time_order__mutmut_20(df, symbol_col=None):
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
        df = df[~df.index.duplicated(keep=None)]
    return df


def x_enforce_groupwise_time_order__mutmut_21(df, symbol_col=None):
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
        df = df[~df.index.duplicated(keep="XXfirstXX")]
    return df


def x_enforce_groupwise_time_order__mutmut_22(df, symbol_col=None):
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
        df = df[~df.index.duplicated(keep="FIRST")]
    return df

x_enforce_groupwise_time_order__mutmut_mutants : ClassVar[MutantDict] = {
'x_enforce_groupwise_time_order__mutmut_1': x_enforce_groupwise_time_order__mutmut_1, 
    'x_enforce_groupwise_time_order__mutmut_2': x_enforce_groupwise_time_order__mutmut_2, 
    'x_enforce_groupwise_time_order__mutmut_3': x_enforce_groupwise_time_order__mutmut_3, 
    'x_enforce_groupwise_time_order__mutmut_4': x_enforce_groupwise_time_order__mutmut_4, 
    'x_enforce_groupwise_time_order__mutmut_5': x_enforce_groupwise_time_order__mutmut_5, 
    'x_enforce_groupwise_time_order__mutmut_6': x_enforce_groupwise_time_order__mutmut_6, 
    'x_enforce_groupwise_time_order__mutmut_7': x_enforce_groupwise_time_order__mutmut_7, 
    'x_enforce_groupwise_time_order__mutmut_8': x_enforce_groupwise_time_order__mutmut_8, 
    'x_enforce_groupwise_time_order__mutmut_9': x_enforce_groupwise_time_order__mutmut_9, 
    'x_enforce_groupwise_time_order__mutmut_10': x_enforce_groupwise_time_order__mutmut_10, 
    'x_enforce_groupwise_time_order__mutmut_11': x_enforce_groupwise_time_order__mutmut_11, 
    'x_enforce_groupwise_time_order__mutmut_12': x_enforce_groupwise_time_order__mutmut_12, 
    'x_enforce_groupwise_time_order__mutmut_13': x_enforce_groupwise_time_order__mutmut_13, 
    'x_enforce_groupwise_time_order__mutmut_14': x_enforce_groupwise_time_order__mutmut_14, 
    'x_enforce_groupwise_time_order__mutmut_15': x_enforce_groupwise_time_order__mutmut_15, 
    'x_enforce_groupwise_time_order__mutmut_16': x_enforce_groupwise_time_order__mutmut_16, 
    'x_enforce_groupwise_time_order__mutmut_17': x_enforce_groupwise_time_order__mutmut_17, 
    'x_enforce_groupwise_time_order__mutmut_18': x_enforce_groupwise_time_order__mutmut_18, 
    'x_enforce_groupwise_time_order__mutmut_19': x_enforce_groupwise_time_order__mutmut_19, 
    'x_enforce_groupwise_time_order__mutmut_20': x_enforce_groupwise_time_order__mutmut_20, 
    'x_enforce_groupwise_time_order__mutmut_21': x_enforce_groupwise_time_order__mutmut_21, 
    'x_enforce_groupwise_time_order__mutmut_22': x_enforce_groupwise_time_order__mutmut_22
}

def enforce_groupwise_time_order(*args, **kwargs):
    result = _mutmut_trampoline(x_enforce_groupwise_time_order__mutmut_orig, x_enforce_groupwise_time_order__mutmut_mutants, args, kwargs)
    return result 

enforce_groupwise_time_order.__signature__ = _mutmut_signature(x_enforce_groupwise_time_order__mutmut_orig)
x_enforce_groupwise_time_order__mutmut_orig.__name__ = 'x_enforce_groupwise_time_order'
