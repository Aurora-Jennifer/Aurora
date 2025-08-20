# API Throttles

Purpose: Handle upstream rate limiting.

Entrypoints:  risk.rate_limit, caller modules.

Do-not-touch: core simulation/engine.

- Reduce concurrency; backoff with jitter.
- Enforce max_requests_per_minute from config; abort gracefully on exceed.
