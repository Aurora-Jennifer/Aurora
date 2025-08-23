# Diff Summary

**Files touched**
- pyproject.toml (+15 lines) - Added mutmut configuration
- Makefile (+20 lines) - Added mutation testing targets
- tests/util/corruptions.py (+150 lines) - NEW: Data corruption utilities
- tests/datasanity/test_mutations.py (+250 lines) - NEW: Metamorphic tests
- .github/workflows/ci.yml (+25 lines) - Added mutation-testing job
- docs/runbooks/mutation_testing.md (+150 lines) - NEW: Documentation

**Notes**
- 370 files changed total (mostly mutmut-generated mutant files)
- Core changes focused on 6 key files
- All new files are test infrastructure and documentation
- No changes to production DataSanity code
- CI job is non-blocking to avoid pipeline disruption

**Key additions**
- Data corruption utilities for metamorphic testing
- Comprehensive metamorphic test suite
- CI integration for mutation testing
- Complete documentation and troubleshooting guide
