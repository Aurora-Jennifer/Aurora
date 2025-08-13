"""
Performance Metrics Calculation
Computes accurate performance metrics including win rate and profit factor.
"""

import math
from typing import List, Dict, Union
from .trade_logger import TradeRecord


def calculate_trade_metrics(closed_trades: List[TradeRecord]) -> Dict[str, Union[float, str]]:
    """
    Calculate comprehensive trade metrics.
    
    Args:
        closed_trades: List of closed trade records
        
    Returns:
        Dictionary with performance metrics
    """
    if not closed_trades:
        return {
            "win_rate": 0.0,
            "profit_factor": "N/A",
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "total_fees": 0.0
        }
    
    pnls = [t.realized_pnl for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    total_trades = len(closed_trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    
    sum_pos = sum(wins)
    sum_neg = abs(sum(losses))
    
    if sum_neg == 0:
        profit_factor = "N/A"
    else:
        profit_factor = sum_pos / sum_neg
    
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "avg_win": (sum_pos / len(wins)) if wins else 0.0,
        "avg_loss": (-sum(losses) / len(losses)) if losses else 0.0,
        "total_trades": total_trades,
        "total_pnl": sum(pnls),
        "total_fees": sum(t.cum_fees for t in closed_trades)
    }


def calculate_portfolio_metrics(equity_curve: List[Dict]) -> Dict[str, float]:
    """
    Calculate portfolio-level performance metrics.
    
    Args:
        equity_curve: List of daily equity values
        
    Returns:
        Dictionary with portfolio metrics
    """
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0
        }
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(equity_curve)):
        prev_equity = equity_curve[i-1]["equity"]
        curr_equity = equity_curve[i]["equity"]
        if prev_equity > 0:
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)
    
    if not returns:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0
        }
    
    # Total return
    initial_equity = equity_curve[0]["equity"]
    final_equity = equity_curve[-1]["equity"]
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
    
    # Annualized return (assuming 252 trading days)
    days = len(equity_curve)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0.0
    
    # Volatility (annualized)
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    volatility = math.sqrt(variance * 252) if variance > 0 else 0.0
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
    
    # Maximum drawdown
    peak = initial_equity
    max_drawdown = 0.0
    
    for point in equity_curve:
        equity = point["equity"]
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio
    }


def validate_daily_returns(equity_curve: List[Dict], max_outlier_threshold: float = 0.5) -> List[Dict]:
    """
    Validate and cap outlier daily returns.
    
    Args:
        equity_curve: List of daily equity values
        max_outlier_threshold: Maximum allowed daily return (50% default)
        
    Returns:
        List of validated equity curve points
    """
    if len(equity_curve) < 2:
        return equity_curve
    
    validated_curve = [equity_curve[0]]  # First point is always valid
    
    for i in range(1, len(equity_curve)):
        prev_point = validated_curve[-1]
        curr_point = equity_curve[i]
        
        prev_equity = prev_point["equity"]
        curr_equity = curr_point["equity"]
        
        if prev_equity > 0:
            daily_return = (curr_equity - prev_equity) / prev_equity
            
            # Cap outlier returns
            if abs(daily_return) > max_outlier_threshold:
                # Adjust current equity to cap the return
                if daily_return > max_outlier_threshold:
                    curr_equity = prev_equity * (1 + max_outlier_threshold)
                else:
                    curr_equity = prev_equity * (1 - max_outlier_threshold)
                
                # Update the point
                adjusted_point = curr_point.copy()
                adjusted_point["equity"] = curr_equity
                adjusted_point["outlier_capped"] = True
                validated_curve.append(adjusted_point)
            else:
                validated_curve.append(curr_point)
        else:
            validated_curve.append(curr_point)
    
    return validated_curve


def generate_performance_report(
    trade_metrics: Dict,
    portfolio_metrics: Dict,
    equity_curve: List[Dict],
    backtest_period: Dict
) -> str:
    """
    Generate a formatted performance report.
    
    Args:
        trade_metrics: Trade-level metrics
        portfolio_metrics: Portfolio-level metrics
        equity_curve: Daily equity curve
        backtest_period: Backtest period information
        
    Returns:
        Formatted performance report string
    """
    # Calculate equity curve values
    initial_equity = equity_curve[0]['equity'] if equity_curve else 0.0
    final_equity = equity_curve[-1]['equity'] if equity_curve else 0.0
    peak_equity = max(point['equity'] for point in equity_curve) if equity_curve else 0.0
    
    report = f"""
📊 PERFORMANCE REPORT
{'=' * 60}

📅 BACKTEST PERIOD
  Start Date: {backtest_period.get('start_date', 'N/A')}
  End Date: {backtest_period.get('end_date', 'N/A')}
  Trading Days: {backtest_period.get('trading_days', 0)}

💰 PORTFOLIO METRICS
  Total Return: {portfolio_metrics.get('total_return', 0):.2%}
  Annualized Return: {portfolio_metrics.get('annualized_return', 0):.2%}
  Volatility: {portfolio_metrics.get('volatility', 0):.2%}
  Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}
  Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}
  Calmar Ratio: {portfolio_metrics.get('calmar_ratio', 0):.2f}

📈 TRADING METRICS
  Total Trades: {trade_metrics.get('total_trades', 0)}
  Win Rate: {trade_metrics.get('win_rate', 0):.2%}
  Profit Factor: {trade_metrics.get('profit_factor', 'N/A')}
  Total PnL: ${trade_metrics.get('total_pnl', 0):,.2f}
  Total Fees: ${trade_metrics.get('total_fees', 0):,.2f}
  Largest Win: ${trade_metrics.get('largest_win', 0):,.2f}
  Largest Loss: ${trade_metrics.get('largest_loss', 0):,.2f}
  Average Win: ${trade_metrics.get('avg_win', 0):,.2f}
  Average Loss: ${trade_metrics.get('avg_loss', 0):,.2f}

📊 EQUITY CURVE
  Initial Equity: ${initial_equity:,.2f}
  Final Equity: ${final_equity:,.2f}
  Peak Equity: ${peak_equity:,.2f}

{'=' * 60}
"""
    
    # Add note if profit factor is N/A
    if trade_metrics.get('profit_factor') == 'N/A':
        report += "\n⚠️  NOTE: No losing trades recorded; verify accounting.\n"
    
    return report
