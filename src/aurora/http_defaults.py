"""
Centralized HTTP defaults (import-only; no side effects).
Runtime wiring is deferred to a separate PR to avoid behavior changes.
"""

# NEW: conservative default timeout for outbound HTTP in seconds.
DEFAULT_TIMEOUT = 15

# NEW: allowlist placeholder for future enforcement.
ALLOWED_DOMAINS = {
    "api.github.com",
    # "internal.service.local",
}

