"""Optional alternative data interface (stubs degrade gracefully)."""

from __future__ import annotations

from typing import Any

import numpy as np


class AltDataProvider:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("alt_data", {}).get("enabled", False))
        self.providers = config.get("alt_data", {}).get("providers", [])

    def fetch_features(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        if not self.enabled or not self.providers:
            # Return NaN features that downstream can handle
            return {
                s: {
                    "news_intensity": np.nan,
                    "options_skew": np.nan,
                    "etf_flows": np.nan,
                    "social_buzz": np.nan,
                }
                for s in symbols
            }
        # Placeholder: extend per provider
        return {
            s: {
                "news_intensity": np.nan,
                "options_skew": np.nan,
                "etf_flows": np.nan,
                "social_buzz": np.nan,
            }
            for s in symbols
        }
