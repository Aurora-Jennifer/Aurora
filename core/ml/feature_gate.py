"""
Hard feature gate for XGBoost model contracts.
Enforces exact feature matching and fails fast on violations.
"""
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def prepare_X_for_xgb(df: pd.DataFrame, whitelist: list[str]) -> pd.DataFrame:
    """
    Prepare features for XGBoost prediction with strict contract enforcement.
    
    Args:
        df: Live feature DataFrame
        whitelist: Expected feature names from training
        
    Returns:
        Cleaned DataFrame with exact feature match
        
    Raises:
        SystemExit: If feature contract is violated
    """
    # Normalize column names
    cols = pd.Index(map(str, df.columns)).str.strip()
    expected = pd.Index(whitelist)
    
    # Find mismatches
    missing = [c for c in expected if c not in cols]
    extra = [c for c in cols if c not in expected]
    
    if missing:
        # Write diff for investigation
        Path("artifacts").mkdir(parents=True, exist_ok=True)
        diff_content = f"MISSING ({len(missing)}):\n" + "\n".join(missing) + "\n\nEXTRA ({len(extra)}):\n" + "\n".join(extra[:20])
        Path("artifacts/feature_diff.txt").write_text(diff_content)
        
        logger.critical(f"Feature contract violated. Missing: {missing[:8]}...")
        logger.critical(f"Full diff written to artifacts/feature_diff.txt")
        raise SystemExit(f"[ABORT] Feature contract violated. Missing: {missing[:8]}...")
    
    # Select and validate features
    X = df.loc[:, expected].astype("float32")
    
    # Check for NaN/Inf
    if not X.replace([float("inf"), -float("inf")], pd.NA).notna().all().all():
        logger.critical("NaN/Inf detected in selected features")
        raise SystemExit("[ABORT] NaN/Inf in selected features.")
    
    logger.info(f"✅ Feature contract satisfied: {len(expected)}/{len(expected)} features matched")
    return X

def validate_feature_contract(df: pd.DataFrame, whitelist: list[str]) -> bool:
    """
    Validate feature contract without raising exceptions.
    
    Returns:
        True if contract satisfied, False otherwise
    """
    try:
        prepare_X_for_xgb(df, whitelist)
        return True
    except SystemExit:
        return False
