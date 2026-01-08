# Portfolio Benchmark Alignment - Phase 1B.1

## Overview

Phase 1B.1 establishes **portfolio composite benchmarks** to align portfolio-level metrics (snapshot, attribution, and beta) with wave-level dynamic benchmarks. This ensures consistency across the entire system by building portfolio benchmarks as weighted combinations of individual wave benchmarks.

## Rationale

### The Problem

Prior to Phase 1B.1:
- **Wave-level benchmarks**: Each wave had its own dynamic benchmark (introduced in Phase 1B) that accurately reflected its investment strategy
- **Portfolio-level benchmarks**: The portfolio used a single static benchmark (SPY) for all aggregated metrics

This created an inconsistency:
- A portfolio's alpha would be computed against SPY
- But each wave's alpha was computed against its own specific benchmark (e.g., QQQ for tech waves, PAVE for infrastructure waves)
- The portfolio alpha didn't accurately represent the sum of wave alphas

### The Solution

Phase 1B.1 introduces **portfolio composite benchmarks**:
- The portfolio benchmark is a weighted combination of all wave benchmarks
- By default, uses equal weights across waves (consistent with equal-weight portfolio construction)
- Ensures that portfolio-level metrics reflect the same benchmark logic as wave-level metrics

### Benefits

1. **Consistency**: Portfolio metrics now use the same benchmark philosophy as wave metrics
2. **Accuracy**: Portfolio alpha accurately represents aggregate performance vs. appropriate benchmarks
3. **Interpretability**: Beta and attribution metrics become more meaningful
4. **Composability**: Portfolio metrics properly compose from wave metrics

## Implementation

### Core Function

The system provides `build_portfolio_composite_benchmark_returns()` in `waves_engine.py`:

```python
def build_portfolio_composite_benchmark_returns(
    wave_results: dict,
    wave_weights: dict | None = None
) -> pd.Series:
    """
    Build portfolio composite benchmark returns from wave benchmark returns.
    
    Args:
        wave_results: Dictionary mapping wave names to compute_history_nav results
        wave_weights: Optional dictionary mapping wave names to weights
                     (defaults to equal weights if None)
    
    Returns:
        pd.Series: Daily return series for the portfolio composite benchmark
    """
```

### Rules and Constraints

1. **Wave Weights**:
   - Defaults to equal weights if `wave_weights` is None
   - Provided weights are automatically normalized to sum to 1.0
   - Only includes waves that have valid benchmark data

2. **Data Requirements**:
   - Removes all-NaN dates from the composite
   - Requires minimum 60 trading days of history
   - Aligns dates across all wave benchmarks

3. **Return Type**:
   - Returns daily return series (not cumulative)
   - Empty series if insufficient data or errors

### Integration Points

The composite benchmark is integrated into:

1. **Portfolio Snapshot** (`helpers/wave_performance.py::compute_portfolio_snapshot`):
   - Computes portfolio returns vs. composite benchmark
   - Calculates alpha as: `portfolio_return - composite_benchmark_return`

2. **Portfolio Attribution** (`helpers/wave_performance.py::compute_portfolio_alpha_attribution`):
   - Uses composite benchmark for attribution analysis
   - Verifies identity: `Cumulative Alpha ≈ Cumulative Realized - Cumulative Benchmark`

3. **Beta Calculations**:
   - Portfolio beta computed vs. composite benchmark
   - Handles missing/inadequate benchmark data gracefully

## Equal-Weight Benchmarks (Current Phase)

### Definition

An **equal-weight composite benchmark** assigns the same weight to each wave's benchmark:

```
weight_i = 1 / N
```

where N is the number of waves with valid benchmark data.

### Rationale

Equal-weight benchmarks align with:
- Equal-weight portfolio construction (current implementation)
- Simplicity and transparency
- Fair representation of all strategies

### Example

With 3 waves:
- S&P 500 Wave (benchmark: 100% SPY)
- Growth Wave (benchmark: 50% QQQ + 50% SMH)
- Infrastructure Wave (benchmark: 60% PAVE + 40% XLI)

Equal-weight composite:
- Weight: 1/3 for each wave's benchmark
- Daily composite return = (SPY_ret + Growth_bm_ret + Infra_bm_ret) / 3

## Wave-Level Benchmark Preservation

### S&P 500 Wave

The S&P 500 Wave maintains its static SPY benchmark at the wave level (as specified in Phase 1B):
- Wave-level benchmark: 100% SPY (static, not dynamic)
- Excluded from dynamic benchmark system
- Contributes SPY returns to portfolio composite benchmark

### Dynamic Benchmark Waves

All other equity waves use their Phase 1B dynamic benchmarks:
- Benchmarks computed from ETF components (e.g., QQQ, SMH, PAVE, etc.)
- Defined in `data/benchmarks/equity_benchmarks.json`
- Each wave contributes its own benchmark returns to the composite

## Future Extensions

### Capital-Weighted Benchmarks

Future phases may introduce capital-weighted composite benchmarks:

```
weight_i = capital_allocation_i / total_capital
```

This would align benchmarks with actual capital deployment rather than equal allocation.

### Custom Weight Schemes

The API supports custom weighting schemes via the `wave_weights` parameter:

```python
# Example: 60% growth waves, 40% value waves
custom_weights = {
    'Growth Wave': 0.30,
    'Tech Wave': 0.30,
    'Value Wave': 0.40
}

composite = build_portfolio_composite_benchmark_returns(
    wave_results=wave_results,
    wave_weights=custom_weights
)
```

### Rebalancing Logic

Future enhancements may support:
- Time-varying weights (rebalancing schedules)
- Momentum-based weight adjustments
- Risk-parity weighting schemes

## Validation

### Validation Script

Run `validate_portfolio_composite_benchmark.py` to verify:
- Portfolio composite benchmark builds without errors
- Benchmark is not empty/all-NaN
- At least 60 rows of data
- Reports benchmark range and statistics

```bash
python validate_portfolio_composite_benchmark.py
```

### Integration Tests

Run `test_portfolio_composite_benchmark_integration.py` to verify:
- Portfolio composite benchmark computation
- Alignment with portfolio snapshot window
- Alpha validation with 0.5% tolerance
- S&P 500 Wave benchmark remains SPY

```bash
pytest test_portfolio_composite_benchmark_integration.py -v
```

### CI Workflow

The `.github/workflows/validate_portfolio_composite_benchmark.yml` workflow:
- Triggers on changes to relevant files
- Runs validation script
- Executes integration tests
- Reports results in GitHub Actions summary

## Governance

### Data Sources

- **Wave benchmarks**: Defined in `data/benchmarks/equity_benchmarks.json` (Phase 1B)
- **Composite logic**: Implemented in `waves_engine.py::build_portfolio_composite_benchmark_returns`
- **Integration**: Portfolio functions in `helpers/wave_performance.py`

### Version History

- **v1.0 (Phase 1B.1)**: Initial implementation with equal-weight composites
- **Future**: Capital-weighted composites, custom schemes, rebalancing logic

### Maintenance

When modifying:
1. **Wave benchmarks**: Update `data/benchmarks/equity_benchmarks.json` (Phase 1B process)
2. **Composite logic**: Modify `build_portfolio_composite_benchmark_returns()` function
3. **Integration**: Update portfolio snapshot/attribution functions as needed
4. **Validation**: Run validation script and tests before committing
5. **CI**: Ensure CI workflow passes before merging

## FAQ

### Why not use a single static benchmark (e.g., SPY) for the portfolio?

A single static benchmark doesn't reflect the diverse strategies in the portfolio. For example:
- A portfolio with 50% tech exposure should have some QQQ in its benchmark
- A portfolio with infrastructure exposure should have some PAVE in its benchmark
- Using only SPY would make alpha calculations misleading

### How does this relate to Phase 1B?

- **Phase 1B**: Introduced dynamic benchmarks at the wave level
- **Phase 1B.1**: Extends dynamic benchmarks to the portfolio level via composites
- Both phases ensure benchmark alignment and consistency

### What if a wave's benchmark data is missing?

The composite benchmark gracefully handles missing data:
- Waves without valid benchmark data are excluded from the composite
- Remaining wave weights are renormalized
- Minimum 60 days of history required for the composite

### Can I use different weights for different waves?

Yes, via the `wave_weights` parameter:

```python
composite = build_portfolio_composite_benchmark_returns(
    wave_results=wave_results,
    wave_weights={'Wave A': 0.6, 'Wave B': 0.4}
)
```

However, the default (and recommended for equal-weight portfolios) is equal weights.

### Does this change wave-level metrics?

No. Wave-level metrics (returns, benchmarks, alpha) are unchanged. This only affects portfolio-level aggregation.

## References

- [Phase 1B: Dynamic Benchmarks](DYNAMIC_BENCHMARKS.md)
- [Wave Performance Module](helpers/wave_performance.py)
- [Waves Engine](waves_engine.py)
- [Validation Script](validate_portfolio_composite_benchmark.py)
- [Integration Tests](test_portfolio_composite_benchmark_integration.py)
