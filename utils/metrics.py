"""
Performance and Risk Metrics Utilities

Pure functions for computing trading performance and risk metrics.
All functions return structured dataclasses for consistent output.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class PerformanceMetrics:
    """Structured output for performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    avg_gain: float
    avg_loss: float
    gain_loss_ratio: float
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int


@dataclass
class RiskMetrics:
    """Structured output for risk metrics."""

    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    downside_deviation: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float


@dataclass
class DailyMetrics:
    """Structured output for daily metrics."""

    avg_daily_return: float
    daily_return_std: float
    daily_var_95: float
    best_day: float
    worst_day: float
    profitable_days: float
    consecutive_losses: int
    avg_daily_turnover: float


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Compute Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan

    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return np.nan

    return (excess_returns.mean() * periods_per_year) / (
        excess_returns.std() * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Compute Sortino ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return np.nan

    excess_returns = returns - (risk_free_rate / periods_per_year)
    negative_returns = excess_returns[excess_returns < 0]

    if len(negative_returns) == 0:
        return np.inf

    downside_std = negative_returns.std()
    if downside_std == 0:
        return np.inf

    return (excess_returns.mean() * periods_per_year) / (downside_std * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute Calmar ratio.

    Args:
        returns: Return series
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return np.nan

    max_dd = max_drawdown(returns)
    if max_dd == 0:
        return np.inf

    annualized_return = returns.mean() * periods_per_year
    return annualized_return / abs(max_dd)


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown.

    Args:
        returns: Return series

    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return np.nan

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Compute Value at Risk (VaR).

    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value
    """
    if len(returns) == 0:
        return np.nan

    return np.percentile(returns, (1 - confidence_level) * 100)


def cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (CVaR).

    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return np.nan

    var_threshold = var(returns, confidence_level)
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return var_threshold

    return tail_returns.mean()


def turnover(positions: pd.Series, method: str = "absolute") -> float:
    """
    Compute portfolio turnover.

    Args:
        positions: Position series
        method: Turnover method ('absolute' or 'relative')

    Returns:
        Turnover value
    """
    if len(positions) < 2:
        return 0.0

    position_changes = positions.diff().abs()

    if method == "absolute":
        return position_changes.mean()
    elif method == "relative":
        total_positions = positions.abs().mean()
        if total_positions == 0:
            return 0.0
        return position_changes.mean() / total_positions
    else:
        raise ValueError(f"Unknown turnover method: {method}")


def hit_rate(returns: pd.Series) -> float:
    """
    Compute hit rate (percentage of positive returns).

    Args:
        returns: Return series

    Returns:
        Hit rate (0.0 to 1.0)
    """
    if len(returns) == 0:
        return np.nan

    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    """
    Compute profit factor.

    Args:
        returns: Return series

    Returns:
        Profit factor
    """
    if len(returns) == 0:
        return np.nan

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


def gain_loss_ratio(returns: pd.Series) -> float:
    """
    Compute gain/loss ratio.

    Args:
        returns: Return series

    Returns:
        Gain/loss ratio
    """
    if len(returns) == 0:
        return np.nan

    avg_gain = returns[returns > 0].mean()
    avg_loss = abs(returns[returns < 0].mean())

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return np.nan

    if avg_loss == 0:
        return np.inf

    return avg_gain / avg_loss


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Compute beta relative to benchmark.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series

    Returns:
        Beta value
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")

    if len(returns) == 0:
        return np.nan

    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)

    if benchmark_variance == 0:
        return np.nan

    return covariance / benchmark_variance


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute alpha relative to benchmark.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Alpha value (annualized)
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")

    if len(returns) == 0:
        return np.nan

    beta_val = beta(returns, benchmark_returns)
    if pd.isna(beta_val):
        return np.nan

    portfolio_return = returns.mean() * periods_per_year
    benchmark_return = benchmark_returns.mean() * periods_per_year
    rf_return = risk_free_rate

    return portfolio_return - (rf_return + beta_val * (benchmark_return - rf_return))


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Compute information ratio.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series

    Returns:
        Information ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")

    if len(returns) == 0:
        return np.nan

    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        return np.nan

    return excess_returns.mean() / tracking_error


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Treynor ratio.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Treynor ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")

    if len(returns) == 0:
        return np.nan

    beta_val = beta(returns, benchmark_returns)
    if pd.isna(beta_val) or beta_val == 0:
        return np.nan

    excess_return = (returns.mean() * periods_per_year) - risk_free_rate
    return excess_return / beta_val


def compute_performance_metrics(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        PerformanceMetrics dataclass
    """
    if len(returns) == 0:
        return PerformanceMetrics(
            total_return=np.nan,
            annualized_return=np.nan,
            volatility=np.nan,
            sharpe_ratio=np.nan,
            sortino_ratio=np.nan,
            calmar_ratio=np.nan,
            max_drawdown=np.nan,
            var_95=np.nan,
            cvar_95=np.nan,
            win_rate=np.nan,
            profit_factor=np.nan,
            avg_gain=np.nan,
            avg_loss=np.nan,
            gain_loss_ratio=np.nan,
            num_trades=0,
            num_winning_trades=0,
            num_losing_trades=0,
        )

    total_return = (1 + returns).prod() - 1
    annualized_return = returns.mean() * periods_per_year
    volatility = returns.std() * np.sqrt(periods_per_year)

    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calmar_ratio(returns, periods_per_year)
    max_dd = max_drawdown(returns)
    var_95_val = var(returns, 0.95)
    cvar_95_val = cvar(returns, 0.95)

    win_rate_val = hit_rate(returns)
    profit_factor_val = profit_factor(returns)
    gain_loss_ratio_val = gain_loss_ratio(returns)

    avg_gain_val = returns[returns > 0].mean()
    avg_loss_val = abs(returns[returns < 0].mean())

    num_trades = len(returns)
    num_winning_trades = (returns > 0).sum()
    num_losing_trades = (returns < 0).sum()

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        var_95=var_95_val,
        cvar_95=cvar_95_val,
        win_rate=win_rate_val,
        profit_factor=profit_factor_val,
        avg_gain=avg_gain_val,
        avg_loss=avg_loss_val,
        gain_loss_ratio=gain_loss_ratio_val,
        num_trades=num_trades,
        num_winning_trades=num_winning_trades,
        num_losing_trades=num_losing_trades,
    )


def compute_risk_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> RiskMetrics:
    """
    Compute comprehensive risk metrics.

    Args:
        returns: Return series
        benchmark_returns: Optional benchmark return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        RiskMetrics dataclass
    """
    if len(returns) == 0:
        return RiskMetrics(
            volatility=np.nan,
            var_95=np.nan,
            cvar_95=np.nan,
            max_drawdown=np.nan,
            downside_deviation=np.nan,
            beta=np.nan,
            alpha=np.nan,
            information_ratio=np.nan,
            treynor_ratio=np.nan,
        )

    volatility_val = returns.std() * np.sqrt(periods_per_year)
    var_95_val = var(returns, 0.95)
    cvar_95_val = cvar(returns, 0.95)
    max_dd = max_drawdown(returns)

    # Downside deviation
    negative_returns = returns[returns < 0]
    downside_deviation = (
        negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0.0
    )

    # Benchmark-relative metrics
    if benchmark_returns is not None:
        beta_val = beta(returns, benchmark_returns)
        alpha_val = alpha(returns, benchmark_returns, risk_free_rate, periods_per_year)
        info_ratio = information_ratio(returns, benchmark_returns)
        treynor = treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year)
    else:
        beta_val = alpha_val = info_ratio = treynor = np.nan

    return RiskMetrics(
        volatility=volatility_val,
        var_95=var_95_val,
        cvar_95=cvar_95_val,
        max_drawdown=max_dd,
        downside_deviation=downside_deviation,
        beta=beta_val,
        alpha=alpha_val,
        information_ratio=info_ratio,
        treynor_ratio=treynor,
    )


def compute_daily_metrics(
    daily_returns: pd.Series, daily_positions: pd.Series | None = None
) -> DailyMetrics:
    """
    Compute daily performance metrics.

    Args:
        daily_returns: Daily return series
        daily_positions: Optional daily position series for turnover

    Returns:
        DailyMetrics dataclass
    """
    if len(daily_returns) == 0:
        return DailyMetrics(
            avg_daily_return=np.nan,
            daily_return_std=np.nan,
            daily_var_95=np.nan,
            best_day=np.nan,
            worst_day=np.nan,
            profitable_days=np.nan,
            consecutive_losses=0,
            avg_daily_turnover=np.nan,
        )

    avg_daily_return = daily_returns.mean()
    daily_return_std = daily_returns.std()
    daily_var_95 = var(daily_returns, 0.95)
    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    profitable_days = hit_rate(daily_returns)

    # Consecutive losses
    losses = daily_returns < 0
    max_consecutive = 0
    current_consecutive = 0

    for loss in losses:
        if loss:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    # Daily turnover
    if daily_positions is not None:
        avg_daily_turnover = turnover(daily_positions, method="absolute")
    else:
        avg_daily_turnover = np.nan

    return DailyMetrics(
        avg_daily_return=avg_daily_return,
        daily_return_std=daily_return_std,
        daily_var_95=daily_var_95,
        best_day=best_day,
        worst_day=worst_day,
        profitable_days=profitable_days,
        consecutive_losses=max_consecutive,
        avg_daily_turnover=avg_daily_turnover,
    )


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    import pandas as pd

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)

    # Test metrics
    print("Testing performance metrics...")

    # Performance metrics
    perf_metrics = compute_performance_metrics(returns)
    print(f"Sharpe Ratio: {perf_metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {perf_metrics.max_drawdown:.1%}")
    print(f"Win Rate: {perf_metrics.win_rate:.1%}")

    # Risk metrics
    risk_metrics = compute_risk_metrics(returns, benchmark_returns)
    print(f"Beta: {risk_metrics.beta:.3f}")
    print(f"Alpha: {risk_metrics.alpha:.3f}")

    # Daily metrics
    daily_metrics = compute_daily_metrics(returns)
    print(f"Best Day: {daily_metrics.best_day:.1%}")
    print(f"Worst Day: {daily_metrics.worst_day:.1%}")

    print("All metrics computed successfully!")
