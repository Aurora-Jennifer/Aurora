"""
Compatibility exports for legacy imports:
from scripts.walkforward_framework_compat import LeakageProofPipeline, ...
This module re-exports implementations from scripts.walk_core.
"""
from __future__ import annotations

import importlib

# Import from walk_core instead of the deprecated walkforward_framework
_IMPL = importlib.import_module("scripts.walkforward.walk_core")


# Map legacy names -> actual attribute names in the implementation module
ALIASES = {
    "LeakageProofPipeline": "LeakageProofPipeline",
    "Fold": "Fold",
    "build_feature_table": "build_feature_table",
    "gen_walkforward": "gen_walkforward",
    "walkforward_run": "walkforward_run",
}


def __getattr__(name: str):  # pragma: no cover - thin adapter
    target = ALIASES.get(name, name)
    if hasattr(_IMPL, target):
        return getattr(_IMPL, target)
    raise AttributeError(
        f"[compat] `{name}` not found. Update ALIASES or migrate imports away from scripts.walkforward_framework_compat."
    )

# Materialize attributes so `from ... import Name` works without __all__
for _legacy, _target in list(ALIASES.items()):
    if hasattr(_IMPL, _target):
        globals()[_legacy] = getattr(_IMPL, _target)


