"""
Public API for core.data_sanity.

This package exposes the symbols from api.py. Internal submodules must use
relative imports and MUST NOT import from this __init__ to avoid cycles.
"""
from .api import *  # noqa: F401,F403
