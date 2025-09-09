import numpy as np


def newey_west_var(x: np.ndarray, lags: int = 5) -> float:
    x = x - x.mean()
    T = x.shape[0]
    if T <= 1:
        return 0.0
    gamma0 = (x @ x) / T
    s = gamma0
    for L in range(1, min(lags, T - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        cov = (x[L:] @ x[:-L]) / T
        s += 2.0 * w * cov
    return max(s, 1e-12)


def sharpe_newey_west(ret: np.ndarray, lags: int = 5, annualize: int = 252) -> float:
    if ret.size == 0:
        return 0.0
    mu = ret.mean()
    var = newey_west_var(ret, lags)
    if var <= 0:
        return 0.0
    return (mu * annualize) / (np.sqrt(var) * np.sqrt(annualize))


def sortino(ret: np.ndarray, annualize: int = 252) -> float:
    if ret.size == 0:
        return 0.0
    down = ret[ret < 0]
    if down.size == 0:
        return float("inf") if ret.mean() > 0 else 0.0
    down_vol = down.std() * np.sqrt(annualize)
    if down_vol == 0:
        return 0.0
    return (ret.mean() * annualize) / down_vol


def max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    peak = np.maximum.accumulate(pnl)
    peak[peak == 0] = np.nan
    dd = (pnl - peak) / peak
    dd = np.nan_to_num(dd, nan=0.0)
    return float(dd.min())


def win_rate(wins: int, losses: int) -> float:
    n = wins + losses
    return (wins / n) if n > 0 else 0.0


def daily_turnover(signal: np.ndarray) -> float:
    # proxy: average absolute day-over-day change in position
    if signal.size <= 1:
        return 0.0
    chg = np.abs(np.diff(signal))
    return float(chg.mean())


def psr(sharpe_ann: float, T: int, var: float = 1.0) -> float:
    # Simplified PSR proxy: prob(Sharpe > 0)
    from math import erf, sqrt

    if T <= 0:
        return 0.0
    z = sharpe_ann * sqrt(T) / (sqrt(var) + 1e-12)
    return 0.5 * (1.0 + erf(z / np.sqrt(2)))
