#!/usr/bin/env python3
"""Simple Prometheus metrics server using FastAPI."""

from __future__ import annotations

import os

from fastapi import FastAPI, Response

app = FastAPI()


def _gauge(name: str, value: float, help_text: str) -> str:
    return f"# HELP {name} {help_text}\n# TYPE {name} gauge\n{name} {value}\n"


@app.get("/metrics")
def metrics() -> Response:
    # Minimal stub; real implementation should pull from shared state/files
    lines = []
    lines.append(
        _gauge(
            "trader_daily_ret",
            float(os.getenv("TRADER_DAILY_RET", 0.0)),
            "Latest daily return",
        )
    )
    lines.append(
        _gauge(
            "trader_sharpe_rolling",
            float(os.getenv("TRADER_SHARPE", 0.0)),
            "Rolling Sharpe",
        )
    )
    lines.append(
        _gauge(
            "trader_sortino_rolling",
            float(os.getenv("TRADER_SORTINO", 0.0)),
            "Rolling Sortino",
        )
    )
    lines.append(_gauge("trader_mdd", float(os.getenv("TRADER_MDD", 0.0)), "Max drawdown"))
    lines.append(
        _gauge(
            "trader_gross_exposure",
            float(os.getenv("TRADER_GROSS_EXPOSURE", 0.0)),
            "Gross exposure",
        )
    )
    lines.append(
        _gauge(
            "trader_net_leverage",
            float(os.getenv("TRADER_NET_LEVERAGE", 1.0)),
            "Net leverage",
        )
    )
    lines.append(
        _gauge(
            "trader_slippage_bps",
            float(os.getenv("TRADER_SLIPPAGE_BPS", 0.0)),
            "Average slippage bps",
        )
    )
    lines.append(
        _gauge(
            "trader_failure_counts",
            float(os.getenv("TRADER_FAILURES", 0.0)),
            "Failure counts",
        )
    )
    body = "".join(lines)
    return Response(content=body, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
