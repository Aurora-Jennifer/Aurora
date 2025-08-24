#!/usr/bin/env python3
"""
Crypto Model Evaluation Metrics

Specialized metrics for evaluating crypto trading models, focusing on
Information Coefficient (IC), hit rates, and other trading-relevant measures.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


def information_coefficient(
    predictions: Union[np.ndarray, pd.Series],
    actuals: Union[np.ndarray, pd.Series],
    method: str = 'spearman'
) -> Dict[str, float]:
    """
    Calculate Information Coefficient between predictions and actual returns.
    
    IC measures the correlation between predictions and future returns.
    Higher absolute IC indicates better predictive power.
    
    Args:
        predictions: Model predictions
        actuals: Actual returns/targets
        method: Correlation method ('spearman', 'pearson')
    
    Returns:
        Dictionary with IC, p-value, and confidence metrics
    """
    # Convert to numpy arrays
    preds = np.asarray(predictions).flatten()
    acts = np.asarray(actuals).flatten()
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(preds) | np.isnan(acts))
    preds_clean = preds[valid_mask]
    acts_clean = acts[valid_mask]
    
    if len(preds_clean) < 3:
        logger.warning(f"Too few valid samples for IC: {len(preds_clean)}")
        return {
            'ic': 0.0,
            'ic_abs': 0.0,
            'p_value': 1.0,
            'n_samples': len(preds_clean),
            'is_significant': False,
            'method': method
        }
    
    # Calculate correlation
    if method == 'spearman':
        ic, p_value = stats.spearmanr(preds_clean, acts_clean)
    elif method == 'pearson':
        ic, p_value = stats.pearsonr(preds_clean, acts_clean)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Handle NaN result
    if np.isnan(ic):
        ic = 0.0
        p_value = 1.0
    
    return {
        'ic': float(ic),
        'ic_abs': float(abs(ic)),
        'p_value': float(p_value),
        'n_samples': len(preds_clean),
        'is_significant': p_value < 0.05,
        'method': method
    }


def hit_rate(
    predictions: Union[np.ndarray, pd.Series],
    actuals: Union[np.ndarray, pd.Series],
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calculate hit rate (directional accuracy).
    
    Hit rate measures the percentage of times the model correctly
    predicts the direction of the move.
    
    Args:
        predictions: Model predictions
        actuals: Actual returns/targets
        threshold: Threshold for considering a prediction/actual as positive
    
    Returns:
        Dictionary with hit rate and related metrics
    """
    # Convert to numpy arrays
    preds = np.asarray(predictions).flatten()
    acts = np.asarray(actuals).flatten()
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(preds) | np.isnan(acts))
    preds_clean = preds[valid_mask]
    acts_clean = acts[valid_mask]
    
    if len(preds_clean) == 0:
        logger.warning("No valid samples for hit rate calculation")
        return {
            'hit_rate': 0.0,
            'correct_predictions': 0,
            'total_predictions': 0,
            'baseline_accuracy': 0.5,
            'lift_over_baseline': 0.0
        }
    
    # Convert to directional signals
    pred_directions = preds_clean > threshold
    actual_directions = acts_clean > threshold
    
    # Calculate hit rate
    correct = pred_directions == actual_directions
    hit_rate_value = np.mean(correct)
    
    # Calculate baseline (random chance)
    pos_actual_rate = np.mean(actual_directions)
    baseline = max(pos_actual_rate, 1 - pos_actual_rate)  # Better of always pos/neg
    
    return {
        'hit_rate': float(hit_rate_value),
        'correct_predictions': int(np.sum(correct)),
        'total_predictions': len(preds_clean),
        'baseline_accuracy': float(baseline),
        'lift_over_baseline': float(hit_rate_value - 0.5),  # vs random
        'positive_prediction_rate': float(np.mean(pred_directions)),
        'positive_actual_rate': float(pos_actual_rate)
    }


def crypto_specific_metrics(
    predictions: Union[np.ndarray, pd.Series],
    actuals: Union[np.ndarray, pd.Series],
    timestamps: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive crypto-specific evaluation metrics.
    
    Args:
        predictions: Model predictions
        actuals: Actual returns/targets
        timestamps: Optional timestamps for time-based analysis
        symbol: Optional symbol name for logging
    
    Returns:
        Comprehensive metrics dictionary
    """
    preds = np.asarray(predictions).flatten()
    acts = np.asarray(actuals).flatten()
    
    # Basic validation
    if len(preds) != len(acts):
        raise ValueError(f"Length mismatch: predictions {len(preds)} vs actuals {len(acts)}")
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(preds) | np.isnan(acts))
    n_valid = np.sum(valid_mask)
    n_total = len(preds)
    
    if n_valid < 3:
        logger.warning(f"Insufficient valid data for {symbol}: {n_valid}/{n_total}")
        return _create_empty_metrics(symbol, n_valid, n_total)
    
    preds_clean = preds[valid_mask]
    acts_clean = acts[valid_mask]
    
    # Core metrics
    ic_results = information_coefficient(preds_clean, acts_clean, method='spearman')
    hit_results = hit_rate(preds_clean, acts_clean)
    
    # Statistical measures
    pred_std = np.std(preds_clean)
    actual_std = np.std(acts_clean)
    pred_mean = np.mean(preds_clean)
    actual_mean = np.mean(acts_clean)
    
    # Risk-adjusted metrics
    if actual_std > 0:
        sharpe_like = ic_results['ic'] * np.sqrt(n_valid)  # Annualized-like IC
        information_ratio = (pred_mean - actual_mean) / actual_std if actual_std > 0 else 0.0
    else:
        sharpe_like = 0.0
        information_ratio = 0.0
    
    # Prediction quality
    pred_range = np.max(preds_clean) - np.min(preds_clean)
    actual_range = np.max(acts_clean) - np.min(acts_clean)
    
    # Time-based analysis (if timestamps provided)
    time_metrics = {}
    if timestamps is not None and len(timestamps) == len(valid_mask):
        clean_timestamps = np.asarray(timestamps)[valid_mask]
        if len(clean_timestamps) == len(preds_clean):
            time_metrics = _analyze_time_patterns(
                preds_clean, acts_clean, clean_timestamps
            )
    
    # Compile comprehensive results
    metrics = {
        'symbol': symbol,
        'data_quality': {
            'n_total_samples': n_total,
            'n_valid_samples': n_valid,
            'data_completeness': n_valid / n_total if n_total > 0 else 0.0,
            'prediction_range': float(pred_range),
            'actual_range': float(actual_range)
        },
        'information_coefficient': ic_results,
        'hit_rate': hit_results,
        'statistical_measures': {
            'prediction_mean': float(pred_mean),
            'prediction_std': float(pred_std),
            'actual_mean': float(actual_mean),
            'actual_std': float(actual_std),
            'rmse': float(np.sqrt(np.mean((preds_clean - acts_clean)**2))),
            'mae': float(np.mean(np.abs(preds_clean - acts_clean)))
        },
        'risk_adjusted': {
            'sharpe_like_ic': float(sharpe_like),
            'information_ratio': float(information_ratio)
        },
        'time_analysis': time_metrics
    }
    
    # Overall quality score (0-100)
    quality_score = _calculate_quality_score(metrics)
    metrics['overall_quality_score'] = quality_score
    
    # Log summary
    logger.info(f"Crypto metrics for {symbol or 'unknown'}:")
    logger.info(f"  IC: {ic_results['ic']:.4f} (p={ic_results['p_value']:.4f})")
    logger.info(f"  Hit Rate: {hit_results['hit_rate']:.3f}")
    logger.info(f"  Quality Score: {quality_score:.1f}/100")
    logger.info(f"  Valid samples: {n_valid}/{n_total}")
    
    return metrics


def _create_empty_metrics(symbol: Optional[str], n_valid: int, n_total: int) -> Dict[str, Any]:
    """Create empty metrics structure for insufficient data cases."""
    return {
        'symbol': symbol,
        'data_quality': {
            'n_total_samples': n_total,
            'n_valid_samples': n_valid,
            'data_completeness': n_valid / n_total if n_total > 0 else 0.0,
            'prediction_range': 0.0,
            'actual_range': 0.0
        },
        'information_coefficient': {
            'ic': 0.0, 'ic_abs': 0.0, 'p_value': 1.0,
            'n_samples': n_valid, 'is_significant': False, 'method': 'spearman'
        },
        'hit_rate': {
            'hit_rate': 0.0, 'correct_predictions': 0, 'total_predictions': 0,
            'baseline_accuracy': 0.5, 'lift_over_baseline': 0.0,
            'positive_prediction_rate': 0.0, 'positive_actual_rate': 0.0
        },
        'statistical_measures': {
            'prediction_mean': 0.0, 'prediction_std': 0.0,
            'actual_mean': 0.0, 'actual_std': 0.0,
            'rmse': 0.0, 'mae': 0.0
        },
        'risk_adjusted': {
            'sharpe_like_ic': 0.0, 'information_ratio': 0.0
        },
        'time_analysis': {},
        'overall_quality_score': 0.0
    }


def _analyze_time_patterns(
    predictions: np.ndarray,
    actuals: np.ndarray,
    timestamps: np.ndarray
) -> Dict[str, Any]:
    """Analyze time-based patterns in predictions vs actuals."""
    try:
        # Convert to pandas for easier time analysis
        df = pd.DataFrame({
            'predictions': predictions,
            'actuals': actuals,
            'timestamp': pd.to_datetime(timestamps)
        }).set_index('timestamp')
        
        # Monthly IC if enough data
        time_metrics = {}
        if len(df) >= 60:  # At least 2 months of daily data
            monthly_ic = df.resample('M').apply(
                lambda x: information_coefficient(x['predictions'], x['actuals'])['ic']
                if len(x) >= 10 else np.nan
            )
            
            valid_monthly = monthly_ic.dropna()
            if len(valid_monthly) >= 2:
                time_metrics.update({
                    'monthly_ic_mean': float(valid_monthly.mean()),
                    'monthly_ic_std': float(valid_monthly.std()),
                    'monthly_ic_consistency': float(np.mean(valid_monthly > 0))
                })
        
        # Recent vs historical performance
        if len(df) >= 30:
            midpoint = len(df) // 2
            recent_ic = information_coefficient(
                df.iloc[midpoint:]['predictions'], 
                df.iloc[midpoint:]['actuals']
            )['ic']
            historical_ic = information_coefficient(
                df.iloc[:midpoint]['predictions'], 
                df.iloc[:midpoint]['actuals']
            )['ic']
            
            time_metrics.update({
                'recent_ic': float(recent_ic),
                'historical_ic': float(historical_ic),
                'ic_trend': float(recent_ic - historical_ic)
            })
        
        return time_metrics
        
    except Exception as e:
        logger.warning(f"Time pattern analysis failed: {e}")
        return {}


def _calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate overall quality score (0-100) based on multiple factors.
    
    Scoring factors:
    - IC absolute value (0-40 points)
    - Hit rate vs baseline (0-30 points) 
    - Statistical significance (0-20 points)
    - Data completeness (0-10 points)
    """
    score = 0.0
    
    # IC component (0-40 points)
    ic_abs = metrics['information_coefficient']['ic_abs']
    ic_points = min(40, ic_abs * 200)  # 0.2 IC = 40 points
    score += ic_points
    
    # Hit rate component (0-30 points)
    hit_rate = metrics['hit_rate']['hit_rate']
    baseline = metrics['hit_rate']['baseline_accuracy']
    hit_lift = max(0, hit_rate - baseline)
    hit_points = min(30, hit_lift * 300)  # 0.1 lift = 30 points
    score += hit_points
    
    # Significance component (0-20 points)
    if metrics['information_coefficient']['is_significant']:
        p_value = metrics['information_coefficient']['p_value']
        sig_points = max(0, 20 * (1 - p_value / 0.05))  # Stronger sig = more points
        score += sig_points
    
    # Data quality component (0-10 points)
    completeness = metrics['data_quality']['data_completeness']
    n_samples = metrics['data_quality']['n_valid_samples']
    
    # Completeness points
    score += completeness * 5
    
    # Sample size points (logarithmic)
    if n_samples >= 100:
        sample_points = min(5, np.log10(n_samples / 100) * 2.5 + 5)
        score += sample_points
    else:
        score += n_samples / 100 * 5
    
    return min(100.0, max(0.0, score))


def batch_evaluate_crypto_models(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Batch evaluation of multiple crypto models/symbols.
    
    Args:
        results_dict: Dict[symbol][metric] = array of values
                     Expected keys: 'predictions', 'actuals', 'timestamps' (optional)
        output_path: Optional path to save results JSON
    
    Returns:
        Dictionary of metrics per symbol
    """
    all_metrics = {}
    
    for symbol, data in results_dict.items():
        try:
            if 'predictions' not in data or 'actuals' not in data:
                logger.warning(f"Missing required data for {symbol}")
                continue
                
            timestamps = data.get('timestamps', None)
            
            metrics = crypto_specific_metrics(
                predictions=data['predictions'],
                actuals=data['actuals'],
                timestamps=timestamps,
                symbol=symbol
            )
            
            all_metrics[symbol] = metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate {symbol}: {e}")
            all_metrics[symbol] = _create_empty_metrics(symbol, 0, 0)
    
    # Create summary
    if all_metrics:
        summary = _create_evaluation_summary(all_metrics)
        all_metrics['_summary'] = summary
    
    # Save if requested
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info(f"Saved batch evaluation results to {output_path}")
    
    return all_metrics


def _create_evaluation_summary(all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics across all evaluated symbols."""
    symbols = [k for k in all_metrics.keys() if not k.startswith('_')]
    
    if not symbols:
        return {'n_symbols': 0}
    
    # Collect key metrics
    ics = [all_metrics[s]['information_coefficient']['ic'] for s in symbols]
    hit_rates = [all_metrics[s]['hit_rate']['hit_rate'] for s in symbols]
    quality_scores = [all_metrics[s]['overall_quality_score'] for s in symbols]
    
    # Calculate percentiles
    ic_percentiles = np.percentile(ics, [25, 50, 75]) if ics else [0, 0, 0]
    hit_percentiles = np.percentile(hit_rates, [25, 50, 75]) if hit_rates else [0, 0, 0]
    
    return {
        'n_symbols': len(symbols),
        'ic_statistics': {
            'mean': float(np.mean(ics)),
            'std': float(np.std(ics)),
            'median': float(np.median(ics)),
            'q25': float(ic_percentiles[0]),
            'q75': float(ic_percentiles[2]),
            'positive_rate': float(np.mean(np.array(ics) > 0))
        },
        'hit_rate_statistics': {
            'mean': float(np.mean(hit_rates)),
            'std': float(np.std(hit_rates)),
            'median': float(np.median(hit_rates)),
            'q25': float(hit_percentiles[0]),
            'q75': float(hit_percentiles[2]),
            'above_50_rate': float(np.mean(np.array(hit_rates) > 0.5))
        },
        'overall_quality': {
            'mean_score': float(np.mean(quality_scores)),
            'best_symbol': symbols[np.argmax(quality_scores)],
            'worst_symbol': symbols[np.argmin(quality_scores)],
            'high_quality_rate': float(np.mean(np.array(quality_scores) > 60))
        }
    }


def create_crypto_metrics_report(
    metrics: Dict[str, Any], 
    output_path: str,
    include_plots: bool = False
) -> None:
    """
    Generate a comprehensive metrics report.
    
    Args:
        metrics: Output from crypto_specific_metrics or batch_evaluate_crypto_models
        output_path: Path for the markdown report
        include_plots: Whether to generate and include plots
    """
    import json
    from datetime import datetime
    
    # Determine if single symbol or batch
    is_batch = '_summary' in metrics
    
    report_lines = [
        "# Crypto Model Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    if is_batch:
        # Batch report
        summary = metrics['_summary']
        symbols = [k for k in metrics.keys() if not k.startswith('_')]
        
        report_lines.extend([
            f"## Summary ({summary['n_symbols']} symbols)",
            "",
            "### Information Coefficient",
            f"- Mean IC: {summary['ic_statistics']['mean']:.4f}",
            f"- Median IC: {summary['ic_statistics']['median']:.4f}",
            f"- Positive IC Rate: {summary['ic_statistics']['positive_rate']:.1%}",
            "",
            "### Hit Rate", 
            f"- Mean Hit Rate: {summary['hit_rate_statistics']['mean']:.3f}",
            f"- Above 50% Rate: {summary['hit_rate_statistics']['above_50_rate']:.1%}",
            "",
            "### Quality Scores",
            f"- Mean Quality: {summary['overall_quality']['mean_score']:.1f}/100",
            f"- Best Symbol: {summary['overall_quality']['best_symbol']}",
            f"- High Quality Rate: {summary['overall_quality']['high_quality_rate']:.1%}",
            "",
            "## Individual Symbol Results",
            ""
        ])
        
        # Add table of results
        report_lines.extend([
            "| Symbol | IC | Hit Rate | Quality | Samples |",
            "|--------|----|---------:|--------:|--------:|"
        ])
        
        for symbol in sorted(symbols):
            m = metrics[symbol]
            ic = m['information_coefficient']['ic']
            hit = m['hit_rate']['hit_rate'] 
            quality = m['overall_quality_score']
            samples = m['data_quality']['n_valid_samples']
            
            report_lines.append(
                f"| {symbol} | {ic:+.4f} | {hit:.3f} | {quality:.1f} | {samples} |"
            )
    
    else:
        # Single symbol report
        symbol = metrics.get('symbol', 'Unknown')
        ic_data = metrics['information_coefficient']
        hit_data = metrics['hit_rate']
        quality = metrics['overall_quality_score']
        
        report_lines.extend([
            f"## {symbol} Model Evaluation",
            "",
            "### Information Coefficient",
            f"- IC: {ic_data['ic']:+.4f}",
            f"- P-value: {ic_data['p_value']:.4f}",
            f"- Significant: {'Yes' if ic_data['is_significant'] else 'No'}",
            f"- Method: {ic_data['method']}",
            "",
            "### Hit Rate",
            f"- Hit Rate: {hit_data['hit_rate']:.3f}",
            f"- Baseline: {hit_data['baseline_accuracy']:.3f}",
            f"- Lift: {hit_data['lift_over_baseline']:+.3f}",
            "",
            f"### Overall Quality Score: {quality:.1f}/100",
            ""
        ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Generated metrics report: {output_path}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    true_signal = np.random.randn(n_samples) * 0.02
    predictions = true_signal + np.random.randn(n_samples) * 0.01
    actuals = true_signal + np.random.randn(n_samples) * 0.015
    
    # Calculate metrics
    metrics = crypto_specific_metrics(predictions, actuals, symbol="BTCUSDT")
    
    print("Example Crypto Metrics:")
    print(f"IC: {metrics['information_coefficient']['ic']:.4f}")
    print(f"Hit Rate: {metrics['hit_rate']['hit_rate']:.3f}")
    print(f"Quality Score: {metrics['overall_quality_score']:.1f}/100")