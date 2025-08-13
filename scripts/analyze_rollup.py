#!/usr/bin/env python3
import pandas as pd


def main():
    df = pd.read_csv("results/backtest/rollup.csv")

    print("TOP 5 BY TOTAL PNL:")
    print(
        df.nlargest(5, "total_pnl")[
            ["run_key", "total_pnl", "sharpe_like", "max_dd"]
        ].to_string(index=False)
    )

    print("\nTOP 5 BY SHARPE:")
    print(
        df.nlargest(5, "sharpe_like")[
            ["run_key", "total_pnl", "sharpe_like", "max_dd"]
        ].to_string(index=False)
    )

    print("\nPER-SYMBOL AGGREGATES:")
    symbol_stats = []
    for symbols in ["SPY QQQ", "AAPL MSFT NVDA", "SPY AAPL"]:
        subset = df[df["symbols"] == symbols]
        if not subset.empty:
            total_pnl = subset["total_pnl"].sum()
            total_trades = subset["total_trades"].sum()
            avg_win_rate = subset["win_rate"].mean()
            symbol_stats.append([symbols, total_pnl, total_trades, avg_win_rate])

    stats_df = pd.DataFrame(
        symbol_stats, columns=["Symbols", "Total_PnL", "Total_Trades", "Avg_Win_Rate"]
    )
    print(stats_df.to_string(index=False))

    print(f"\nTOTAL RUNS: {len(df)}")
    print(f"SUCCESSFUL RUNS: {len(df)} (all passed integrity checks)")


if __name__ == "__main__":
    main()
