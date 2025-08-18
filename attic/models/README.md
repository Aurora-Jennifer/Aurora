# Deprecated Models Archive

This directory contains deprecated model artifacts moved out of the active runtime.

- Purpose: retain historical models for reference and rollback only
- Location format: `attic/models/YYYY-MM/`
- Do not point any runtime loaders here.

For each archived set, capture:
- Reason for deprecation and replacement pointer
- Last-known metrics (Sharpe, MaxDD, trades)
- Training config snapshot and data hash (if available)
- Source code git SHA used for training

Current contents were auto-moved from `state/`:
- `performance_history.pkl`
- `models.pkl`
- (If present) `selector.pkl`

Source: AI suggestion (reviewed by me)
