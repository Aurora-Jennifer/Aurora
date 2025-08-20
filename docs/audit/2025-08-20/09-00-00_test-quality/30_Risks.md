# Risks & Rollback
- Risk: Property tests may be sensitive to DataSanity changes
- Rollback: git checkout HEAD~1 tests/datasanity/test_property_invariants.py tests/golden/
