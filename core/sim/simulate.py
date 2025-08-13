import numpy as np
from numba import njit


@njit
def simulate_orders_numba(
    close: np.ndarray, signal: np.ndarray, fee_bps: float = 0.5, slip_bps: float = 1.0
):
    """
    close: price array aligned to signal
    signal: -1/0/+1 desired position (not trade intent; simple target pos)
    Returns (pnl_series, trades_count, wins, losses, median_hold_bars)
    """
    n = close.shape[0]
    pos = 0
    entry_px = 0.0
    holds = np.empty(n, dtype=np.int32)
    hix = 0
    pnl = np.zeros(n, dtype=np.float64)
    wins = 0
    losses = 0
    bars_held = 0

    for i in range(1, n):
        target = signal[i - 1]
        # rebalance if target != pos
        if target != pos:
            # exit current pos if any
            if pos != 0:
                gross = (close[i] - entry_px) * pos
                cost = (fee_bps + slip_bps) * 1e-4 * abs(pos) * close[i]
                pnl[i] += gross - cost
                if gross > 0:
                    wins += 1
                elif gross < 0:
                    losses += 1
                holds[hix] = bars_held
                hix += 1
                bars_held = 0
            # enter new
            if target != 0:
                entry_px = close[i]
        pos = target
        if pos != 0:
            bars_held += 1
        pnl[i] += pnl[i - 1]

    median_hold = 0
    if hix > 0:
        # simple selection sort median for numba
        for a in range(hix):
            m = a
            for b in range(a + 1, hix):
                if holds[b] < holds[m]:
                    m = b
            tmp = holds[a]
            holds[a] = holds[m]
            holds[m] = tmp
        mid = hix // 2
        median_hold = (
            holds[mid] if (hix % 2 == 1) else (holds[mid - 1] + holds[mid]) // 2
        )

    return pnl, wins + losses, wins, losses, median_hold


def simulate_safe(close, signal, **kw):
    """Safe wrapper for numba simulation with input validation."""
    # Ensure contiguous arrays with correct dtypes
    close = np.ascontiguousarray(close, dtype=np.float64)
    signal = np.ascontiguousarray(signal, dtype=np.int8)

    # Validate inputs
    n = min(close.size, signal.size)
    if n == 0:
        return np.zeros(0), 0, 0, 0, 0

    # Check for invalid values
    if np.any(np.isnan(close)) or np.any(np.isinf(close)):
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        signal = np.nan_to_num(signal, nan=0, posinf=0, neginf=0).astype(np.int8)

    # Ensure signal is in valid range
    signal = np.clip(signal, -1, 1).astype(np.int8)

    return simulate_orders_numba(close[:n], signal[:n], **kw)
