"""
Run manifest and model card generation for tracking lineage and performance.
"""
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess
import hashlib


def get_git_info() -> Dict[str, str]:
    """Get current git commit and status information."""
    try:
        # Get commit hash
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.').decode().strip()
        short_commit = commit[:8]
        
        # Get branch
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd='.').decode().strip()
        
        # Get status
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd='.').decode().strip()
        clean = len(status) == 0
        
        return {
            'commit': commit,
            'short_commit': short_commit,
            'branch': branch,
            'clean': clean,
            'status': 'clean' if clean else 'dirty'
        }
    except subprocess.CalledProcessError:
        return {
            'commit': 'unknown',
            'short_commit': 'unknown',
            'branch': 'unknown',
            'clean': False,
            'status': 'unknown'
        }


def compute_data_hash(df: pd.DataFrame, columns: List[str] = None) -> str:
    """Compute hash of data for reproducibility tracking."""
    if columns:
        df_subset = df[columns]
    else:
        df_subset = df
    
    # Create deterministic hash from data
    data_str = df_subset.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(data_str).hexdigest()[:16]


def generate_run_manifest(
    out_dir: str,
    config: Dict[str, Any],
    data_info: Dict[str, Any],
    feature_info: Dict[str, Any],
    model_info: Dict[str, Any],
    performance_metrics: Dict[str, Any] = None,
    guard_results: Dict[str, Any] = None
) -> str:
    """
    Generate comprehensive run manifest for lineage tracking.
    
    Args:
        out_dir: Output directory
        config: Configuration dictionary
        data_info: Data window and universe information
        feature_info: Feature engineering details
        model_info: Model configuration and training details
        performance_metrics: Performance results
        guard_results: Results from feature guards and assertions
        
    Returns:
        Path to generated manifest file
    """
    timestamp = datetime.now().isoformat()
    git_info = get_git_info()
    
    manifest = {
        'metadata': {
            'timestamp': timestamp,
            'version': '1.0',
            'run_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{git_info['short_commit']}"
        },
        'git': git_info,
        'data': data_info,
        'features': feature_info,
        'training': model_info,
        'config': {
            'universe_config': config.get('universe_cfg', {}),
            'grid_config': config.get('grid_cfg', {}),
            'parameters': config.get('parameters', {})
        },
        'guards': guard_results or {},
        'performance': performance_metrics or {}
    }
    
    # Write manifest
    manifest_path = os.path.join(out_dir, 'manifest.json')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Run manifest generated: {manifest_path}")
    return manifest_path


def generate_model_card(
    manifest_path: str,
    out_dir: str,
    additional_info: Dict[str, Any] = None
) -> str:
    """
    Generate model card from manifest information.
    
    Args:
        manifest_path: Path to run manifest
        out_dir: Output directory
        additional_info: Additional information for model card
        
    Returns:
        Path to generated model card
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Extract key information
    data_info = manifest.get('data', {})
    feature_info = manifest.get('features', {})
    model_info = manifest.get('training', {})
    performance = manifest.get('performance', {})
    git_info = manifest.get('git', {})
    
    # Generate model card content
    model_card = f"""# Model Card: Cross-Sectional Alpha Model

## Model Information
- **Model Type**: {model_info.get('model_type', 'XGBoost Regressor')}
- **Objective**: {model_info.get('objective', 'reg:squarederror')}
- **Device**: {model_info.get('device', 'unknown')}
- **Training Date**: {manifest['metadata']['timestamp'][:10]}
- **Git Commit**: {git_info.get('short_commit', 'unknown')} ({git_info.get('branch', 'unknown')})

## Data Information
- **Universe**: {data_info.get('universe', 'unknown')}
- **Date Range**: {data_info.get('start_date', 'unknown')} to {data_info.get('end_date', 'unknown')}
- **Training Samples**: {data_info.get('train_samples', 'unknown'):,}
- **Test Samples**: {data_info.get('test_samples', 'unknown'):,}
- **Data Hash**: {data_info.get('data_hash', 'unknown')}

## Feature Engineering
- **Total Features**: {feature_info.get('total_features', 'unknown')}
- **Cross-sectional Ranks**: {feature_info.get('cs_ranks', 'unknown')}
- **Cross-sectional Z-scores**: {feature_info.get('cs_zscores', 'unknown')}
- **Residuals**: {feature_info.get('residuals', 'unknown')}
- **Market Residualization**: {'✅' if feature_info.get('market_residualization') else '❌'}
- **Sector Residualization**: {'✅' if feature_info.get('sector_residualization') else '❌'}

## Model Configuration
- **Estimators**: {model_info.get('n_estimators', 'unknown')}
- **Max Depth**: {model_info.get('max_depth', 'unknown')}
- **Learning Rate**: {model_info.get('learning_rate', 'unknown')}
- **Early Stopping**: {model_info.get('early_stopping_rounds', 'unknown')} rounds

## Performance Metrics
"""

    if performance:
        for metric, value in performance.items():
            if isinstance(value, (int, float)):
                model_card += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
            else:
                model_card += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
    else:
        model_card += "- *Performance metrics will be added after evaluation*\n"

    model_card += f"""
## Quality Guards
- **Dispersion Check**: {'✅ Passed' if manifest.get('guards', {}).get('dispersion_passed') else '❌ Failed/Unknown'}
- **Feature Variance**: {manifest.get('guards', {}).get('avg_zero_std_features', 'unknown')} avg flat features per date
- **Leakage Check**: {'✅ Passed' if manifest.get('guards', {}).get('leakage_passed') else '❌ Failed/Unknown'}

## Known Limitations
- Model trained on historical data; performance may vary in live trading
- Cross-sectional features assume consistent universe composition
- Residualization methods may remove signal in highly correlated markets

## Usage
This model generates cross-sectional predictions for universe ranking.
Use predictions for portfolio construction with appropriate risk controls.

## Reproducibility
- **Git Commit**: {git_info.get('commit', 'unknown')}
- **Config Hash**: {manifest.get('config', {}).get('hash', 'unknown')}
- **Random Seed**: {model_info.get('random_state', 'unknown')}

---
*Generated automatically by run_manifest.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Write model card
    model_card_path = os.path.join(out_dir, 'MODEL_CARD.md')
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    
    print(f"✅ Model card generated: {model_card_path}")
    return model_card_path


def collect_feature_info(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Collect feature engineering information for manifest."""
    cs_ranks = [col for col in feature_cols if col.endswith('_csr')]
    cs_zscores = [col for col in feature_cols if col.endswith('_csz')]
    residuals = [col for col in feature_cols if col.endswith('_res')]
    
    return {
        'total_features': len(feature_cols),
        'cs_ranks': len(cs_ranks),
        'cs_zscores': len(cs_zscores),
        'residuals': len(residuals),
        'market_residualization': any('mkt_res' in col for col in feature_cols),
        'sector_residualization': any('sec_res' in col for col in feature_cols),
        'feature_columns': feature_cols,
        'data_hash': compute_data_hash(df, feature_cols)
    }


def collect_data_info(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     universe_config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect data information for manifest."""
    all_data = pd.concat([train_df, test_df])
    
    return {
        'universe': universe_config.get('name', 'unknown'),
        'start_date': str(all_data['date'].min()),
        'end_date': str(all_data['date'].max()),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'unique_symbols': all_data['symbol'].nunique(),
        'total_dates': all_data['date'].nunique(),
        'data_hash': compute_data_hash(all_data, ['date', 'symbol'])
    }


def collect_model_info(model, config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Collect model training information for manifest."""
    model_params = config.get('model', {})
    
    info = {
        'model_type': type(model).__name__,
        'device': device,
        **{k: v for k, v in model_params.items() if not callable(v)}
    }
    
    # Add model-specific info if available
    try:
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            info['n_boosted_rounds'] = booster.num_boosted_rounds()
    except:
        pass
    
    return info


def track_performance_metrics(pred_df: pd.DataFrame, actual_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Track basic performance metrics for manifest."""
    metrics = {}
    
    # Basic prediction statistics
    if 'prediction' in pred_df.columns:
        pred_stats = pred_df['prediction'].describe()
        metrics.update({
            'prediction_mean': pred_stats['mean'],
            'prediction_std': pred_stats['std'],
            'prediction_min': pred_stats['min'],
            'prediction_max': pred_stats['max']
        })
        
        # Cross-sectional variance check
        pred_variance_by_date = pred_df.groupby('date')['prediction'].std()
        metrics['avg_daily_prediction_std'] = pred_variance_by_date.mean()
        metrics['flat_prediction_days'] = (pred_variance_by_date < 1e-12).sum()
    
    # If actual returns available, compute correlation metrics
    if actual_df is not None and 'actual_return' in actual_df.columns:
        # This would be computed in the evaluation phase
        pass
    
    return metrics
