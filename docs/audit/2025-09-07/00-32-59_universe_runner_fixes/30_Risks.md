# Risks & Assumptions

## Assumptions
- **Kaleido engine available**: Assumes plotly with kaleido is installed (graceful fallback to matplotlib)
- **Thread limits sufficient**: Assumes setting all BLAS threads to 1 prevents OOM (may need tuning)
- **Portfolio strategy valid**: Assumes top-K long-short is appropriate for cross-sectional validation
- **Cost model accurate**: Assumes 50% turnover and 5bps costs are realistic

## Risks
- **Memory still tight**: Even with thread limits, large universes may still cause OOM
- **Kaleido dependency**: If kaleido not available, falls back to matplotlib (may be slower)
- **Portfolio assumptions**: Top-K strategy may not reflect actual trading implementation
- **Cost model oversimplified**: Real transaction costs may be more complex

## Rollback
```bash
# Revert to previous version
git checkout HEAD~1 -- ml/runner_universe.py

# Or disable specific features via config
# Set make_plots=False in all parallel calls
# Disable portfolio validation by commenting out topk_ls() call
```

## Mitigation
- **Memory monitoring**: Watch for OOM in large runs, may need chunking
- **Dependency check**: Ensure kaleido is installed: `pip install kaleido`
- **Portfolio validation**: Can disable portfolio stats if not needed
- **Cost tuning**: Adjust cost_bps parameter based on actual trading costs
