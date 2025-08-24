# Diff Summary

**Files touched**
- `scripts/fetch_corporate_actions.py` (+258/−0) — NEW: Corporate actions fetcher
- `core/data_sanity/rules/corporate_actions.py` (+258/−0) — NEW: Corporate actions validation rule  
- `core/data_sanity/registry.py` (+2/−0) — Register new rule
- `config/data_sanity.yaml` (+2/−0) — Add rule to validation stages

**Data generated**
- `data/corporate_actions/` — 11 JSON files with splits/dividends metadata
- `data/corporate_actions/manifest.json` — Summary of fetched data

**Notes**
- Corporate actions rule follows same pattern as other DataSanity rules
- Graceful degradation: logs warnings for potential issues but doesn't fail validation
- Symbol extraction logic handles cases where symbol not explicitly provided to rule
