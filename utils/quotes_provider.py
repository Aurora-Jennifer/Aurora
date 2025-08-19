from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol


class QuoteProvider(Protocol):
    def quote(self, symbol: str) -> dict[str, Any]:
        """Return a quote dict with keys: bid, ask, mid, ts (ISO UTC)."""
        ...


class DummyMidProvider:
    def __init__(self, last_px_fn: Callable[[str], float]) -> None:
        self._last = last_px_fn

    def quote(self, symbol: str) -> dict[str, Any]:
        mid = float(self._last(symbol))
        return {
            "bid": mid,
            "ask": mid,
            "mid": mid,
            "ts": datetime.now(UTC).isoformat(),
        }


def get_quote_provider(provider_type: str = "dummy") -> QuoteProvider:
    """Factory function to get quote provider.

    Args:
        provider_type: Provider type ("dummy", "ibkr", etc.)

    Returns:
        QuoteProvider instance
    """
    if provider_type == "dummy":
        return DummyMidProvider(lambda s: 100.0)  # Default dummy provider
    elif provider_type == "ibkr":
        from pathlib import Path

        import yaml

        from .quotes_provider_ibkr import IBKRQuoteConfig, IBKRQuoteProvider

        # Load IBKR config from overlay
        ibkr_cfg = yaml.safe_load(Path("config/brokers/ibkr.yaml").read_text())

        # Create config
        config = IBKRQuoteConfig(
            host=ibkr_cfg.get("ibkr", {}).get("host", "127.0.0.1"),
            port=ibkr_cfg.get("ibkr", {}).get("port", 7497),
            client_id=ibkr_cfg.get("ibkr", {}).get("client_id", 321),
            route=ibkr_cfg.get("ibkr", {}).get("route", "SMART"),
            currency=ibkr_cfg.get("ibkr", {}).get("currency", "USD"),
            snap_ms=ibkr_cfg.get("ibkr", {}).get("snap_ms", 150),
        )
        return IBKRQuoteProvider(config)
    else:
        raise ValueError(f"Unknown quote provider: {provider_type}")
