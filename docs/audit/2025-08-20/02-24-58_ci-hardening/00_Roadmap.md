# CI hardening — Roadmap

Prompt: Harden CI determinism, add pre-push smoke, badge, pytest timeout, and lockfile rename.

## Plan
1) Add deterministic env to smoke job
2) Make lint allow-fail
3) Add pre-push smoke script and Makefile target
4) Add README Smoke badge
5) Add pytest timeout
6) Rename requirements.lock.txt to requirements.lock

## Success
- CI runs deterministic smoke; lint/tests-full allow-fail
- Pre-push script in place
- Badge visible in README
