# Changes

- Added decision tracing (core/data_sanity/trace.py) and test-only fault toggle (core/data_sanity/_faults.py).
- Threaded optional trace through validate_dataframe_fast without breaking public API.
- Added sentry script bin/ds_sentry.py to verify rule-targeted tests fail when rules are disabled.
- Added metamorphic and differential tests under tests/datasanity/.
