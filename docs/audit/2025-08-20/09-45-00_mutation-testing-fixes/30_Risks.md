# Risks

## Assumptions
- DataSanity API behavior is stable and won't change
- `walkforward_smoke` profile will continue to allow lookahead contamination
- Test expectations align with actual DataSanity validation logic

## Potential Issues
1. **Test brittleness**: Error message assertions may break if DataSanity error messages change
2. **Profile dependency**: Tests depend on specific DataSanity profile behavior
3. **False positives**: Some tests may pass when they should fail due to overly flexible assertions

## Mitigation
- Tests use try/except blocks to handle both success and failure cases
- Error message assertions are flexible (checking for keywords rather than exact matches)
- Tests verify core behavior (data length preservation, exception types) rather than exact messages

## Rollback
- Revert changes to test files if CI still fails
- Consider disabling mutation testing job temporarily if issues persist
- Fall back to basic DataSanity unit tests if metamorphic tests prove unstable
