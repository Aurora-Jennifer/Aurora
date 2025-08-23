# Risks & Assumptions

## Assumptions
- DataSanity validation logic is stable and well-tested
- Metamorphic tests accurately represent real data corruption scenarios
- mutmut configuration issues can be resolved in future iterations
- CI mutation testing job won't significantly impact pipeline performance

## Risks
- **mutmut configuration complexity**: Current setup has TOML parsing issues
- **Test sensitivity**: Some metamorphic tests may be too sensitive to synthetic data patterns
- **CI performance**: Mutation testing could slow down CI pipeline
- **False positives**: Metamorphic tests might fail due to legitimate validation changes

## Rollback
```bash
# Remove mutation testing infrastructure
git checkout HEAD~1 -- pyproject.toml Makefile .github/workflows/ci.yml
rm -rf tests/util/corruptions.py tests/datasanity/test_mutations.py docs/runbooks/mutation_testing.md
rm -rf mutants/  # Remove mutmut-generated files
```

## Mitigation
- CI job is non-blocking to avoid pipeline disruption
- Metamorphic tests use synthetic data to avoid flakiness
- Documentation provides troubleshooting guidance
- Limited mutmut scope to avoid performance issues
