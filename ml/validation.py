# ml/validation.py
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Dict, Any, Callable
import warnings

def time_blocks(dates: np.ndarray, n_splits=5, embargo=10) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_mask, test_mask) using contiguous time folds with embargo."""
    uniq = np.asarray(np.sort(np.unique(dates)))
    folds = np.array_split(uniq, n_splits)
    
    for k in range(n_splits):
        test_dates = folds[k]
        lo = test_dates[0] - np.timedelta64(embargo, "D")
        hi = test_dates[-1] + np.timedelta64(embargo, "D")
        
        train_mask = (dates < lo) | (dates > hi)
        test_mask = (dates >= test_dates[0]) & (dates <= test_dates[-1])
        
        yield train_mask, test_mask

def cv_rank_ic(panel: pd.DataFrame, feat_cols, target_col="cs_target",
               n_splits=5, embargo=10, make_model=None) -> Dict[str, Any]:
    """Return per-fold Rank-IC and summary. make_model() must return a fresh, unfitted model."""
    if make_model is None:
        raise ValueError("make_model function is required")
    
    dates = panel["date"].values
    X = panel[feat_cols].values
    y = panel[target_col].values
    
    fold_results = []
    
    for fold_idx, (train_mask, test_mask) in enumerate(time_blocks(dates, n_splits, embargo)):
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            print(f"⚠️  Fold {fold_idx}: insufficient data (train: {train_mask.sum()}, test: {test_mask.sum()})")
            continue
            
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train model
        model = make_model()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Compute IC
        ic = np.corrcoef(y_pred, y_test)[0, 1] if len(y_pred) > 1 else np.nan
        rank_ic = pd.Series(y_pred).corr(pd.Series(y_test), method='spearman') if len(y_pred) > 1 else np.nan
        
        fold_results.append({
            'fold': fold_idx,
            'train_size': train_mask.sum(),
            'test_size': test_mask.sum(),
            'ic': ic,
            'rank_ic': rank_ic,
            'hit_rate': (rank_ic > 0) if not np.isnan(rank_ic) else np.nan
        })
    
    if not fold_results:
        return {'error': 'No valid folds'}
    
    df_folds = pd.DataFrame(fold_results)
    
    return {
        'n_folds': len(fold_results),
        'mean_ic': df_folds['ic'].mean(),
        'std_ic': df_folds['ic'].std(),
        'mean_rank_ic': df_folds['rank_ic'].mean(),
        'std_rank_ic': df_folds['rank_ic'].std(),
        'mean_hit_rate': df_folds['hit_rate'].mean(),
        'fold_details': fold_results
    }

def walk_forward_validation(panel: pd.DataFrame, feat_cols, target_col="cs_target",
                           train_window=252, test_window=63, step_size=21) -> Dict[str, Any]:
    """Walk-forward validation with expanding training window."""
    dates = np.sort(panel["date"].unique())
    
    if len(dates) < train_window + test_window:
        return {'error': f'Insufficient data: need {train_window + test_window} days, have {len(dates)}'}
    
    results = []
    
    for i in range(train_window, len(dates) - test_window + 1, step_size):
        train_end = dates[i]
        test_start = dates[i]
        test_end = dates[min(i + test_window - 1, len(dates) - 1)]
        
        train_mask = panel["date"] <= train_end
        test_mask = (panel["date"] >= test_start) & (panel["date"] <= test_end)
        
        if train_mask.sum() < 100 or test_mask.sum() < 20:
            continue
            
        train_data = panel[train_mask]
        test_data = panel[test_mask]
        
        # For now, just compute correlation on existing predictions
        if 'prediction' in test_data.columns:
            ic = test_data['prediction'].corr(test_data[target_col])
            rank_ic = test_data['prediction'].corr(test_data[target_col], method='spearman')
            
            results.append({
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': train_mask.sum(),
                'test_size': test_mask.sum(),
                'ic': ic,
                'rank_ic': rank_ic,
                'hit_rate': rank_ic > 0 if not np.isnan(rank_ic) else np.nan
            })
    
    if not results:
        return {'error': 'No valid walk-forward periods'}
    
    df_results = pd.DataFrame(results)
    
    return {
        'n_periods': len(results),
        'mean_ic': df_results['ic'].mean(),
        'std_ic': df_results['ic'].std(),
        'mean_rank_ic': df_results['rank_ic'].mean(),
        'std_rank_ic': df_results['rank_ic'].std(),
        'mean_hit_rate': df_results['hit_rate'].mean(),
        'period_details': results
    }

def feature_leakage_audit(panel: pd.DataFrame, feat_cols, target_col="cs_target") -> Dict[str, Any]:
    """Audit features for potential leakage by checking correlations with future returns."""
    results = {}
    
    for feat in feat_cols:
        # Check correlation with target
        corr = panel[feat].corr(panel[target_col])
        
        # Check if feature has suspiciously high correlation
        results[feat] = {
            'correlation_with_target': corr,
            'suspicious': abs(corr) > 0.8,  # Flag high correlations
            'std': panel[feat].std(),
            'null_pct': panel[feat].isnull().mean()
        }
    
    suspicious_features = [f for f, r in results.items() if r['suspicious']]
    
    return {
        'suspicious_features': suspicious_features,
        'feature_details': results,
        'n_suspicious': len(suspicious_features)
    }

def lag_more_test(panel: pd.DataFrame, pred_col="prediction", target_col="cs_target", 
                  max_lags=5) -> Dict[str, Any]:
    """Test IC degradation with increasing lag to detect subtle leakage."""
    results = []
    
    for lag in range(max_lags + 1):
        if lag == 0:
            test_panel = panel.copy()
        else:
            test_panel = panel.copy()
            test_panel[pred_col] = test_panel.groupby('symbol')[pred_col].shift(lag)
            test_panel = test_panel.dropna()
        
        if len(test_panel) < 100:
            continue
            
        # Compute daily IC
        daily_ic = []
        for date in test_panel['date'].unique():
            date_data = test_panel[test_panel['date'] == date]
            if len(date_data) > 1:
                ic = date_data[pred_col].corr(date_data[target_col])
                if not np.isnan(ic):
                    daily_ic.append(ic)
        
        if daily_ic:
            mean_ic = np.mean(daily_ic)
            rank_ic = pd.Series(test_panel[pred_col]).corr(pd.Series(test_panel[target_col]), method='spearman')
            
            results.append({
                'lag': lag,
                'n_obs': len(test_panel),
                'mean_ic': mean_ic,
                'rank_ic': rank_ic,
                'hit_rate': (rank_ic > 0) if not np.isnan(rank_ic) else np.nan
            })
    
    return {
        'lag_results': results,
        'ic_degradation': results[0]['mean_ic'] - results[-1]['mean_ic'] if len(results) > 1 else 0
    }

def comprehensive_ic_validation(panel: pd.DataFrame, feat_cols, target_col="cs_target",
                               make_model=None) -> Dict[str, Any]:
    """Run comprehensive IC validation suite."""
    print("🔍 Running comprehensive IC validation...")
    
    results = {}
    
    # 1. Cross-validation
    print("  📊 Cross-validation...")
    try:
        cv_results = cv_rank_ic(panel, feat_cols, target_col, make_model=make_model)
        results['cross_validation'] = cv_results
    except Exception as e:
        results['cross_validation'] = {'error': str(e)}
    
    # 2. Walk-forward validation
    print("  📈 Walk-forward validation...")
    try:
        wf_results = walk_forward_validation(panel, feat_cols, target_col)
        results['walk_forward'] = wf_results
    except Exception as e:
        results['walk_forward'] = {'error': str(e)}
    
    # 3. Feature leakage audit
    print("  🔍 Feature leakage audit...")
    try:
        leak_results = feature_leakage_audit(panel, feat_cols, target_col)
        results['leakage_audit'] = leak_results
    except Exception as e:
        results['leakage_audit'] = {'error': str(e)}
    
    # 4. Lag-more test
    print("  ⏰ Lag-more test...")
    try:
        lag_results = lag_more_test(panel)
        results['lag_test'] = lag_results
    except Exception as e:
        results['lag_test'] = {'error': str(e)}
    
    return results