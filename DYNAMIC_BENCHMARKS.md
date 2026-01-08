# Dynamic Benchmarks - Phase 1B

## Overview

Phase 1B introduces **dynamic benchmarks** for 10 equity waves in a focused, additive, and reversible manner. This implementation preserves platform stability while enabling more sophisticated benchmark tracking for equity strategies.

## Key Principles

1. **Equity Wave Focus**: Dynamic benchmarks apply to equity-categorized waves in this phase. The Infinity Multi-Asset Growth Wave is a special case that includes crypto (BTC-USD) as it is designed as a multi-asset strategy. Pure crypto and income waves retain their existing benchmark logic.

2. **Additive & Reversible**: All changes are additive to the existing codebase. Pre-existing benchmark logic remains intact as a fallback.

3. **Cached Prices Only**: The system uses only cached price data with no network dependencies, ensuring reliability and performance.

4. **S&P 500 Exception**: The S&P 500 Wave explicitly remains static, using SPY as its sole benchmark (excluded from dynamic benchmarks).

5. **Governance & Transparency**: All benchmark definitions are version-controlled in a JSON configuration file with clear component weights and metadata.

## Covered Waves

The following 10 equity waves have dynamic benchmarks:

1. **Clean Transit / Infrastructure** (clean_transit_infrastructure_wave)
   - Benchmark: 60% PAVE + 40% XLI
   
2. **Demas Fund (Value)** (demas_fund_wave)
   - Benchmark: 60% SPY + 40% VTV
   
3. **EV & Infrastructure** (ev_infrastructure_wave)
   - Benchmark: 60% PAVE + 40% XLI
   
4. **Future Power & Energy** (future_power_energy_wave)
   - Benchmark: 50% ICLN + 50% XLE
   
5. **Infinity (Multi-Asset Growth)** (infinity_multi_asset_growth_wave)
   - Benchmark: 40% SPY + 40% QQQ + 20% BTC-USD
   - Note: Multi-asset wave includes crypto component in benchmark to match strategy allocation
   
6. **Next-Gen Compute & Semiconductors** (next_gen_compute_semis_wave)
   - Benchmark: 50% QQQ + 50% SMH
   
7. **Quantum Computing** (quantum_computing_wave)
   - Benchmark: 60% QQQ + 40% VGT
   
8. **Small to Mid Cap Growth** (small_to_mid_cap_growth_wave)
   - Benchmark: 50% IWP + 50% IWO
   
9. **US MegaCap Core** (us_megacap_core_wave)
   - Benchmark: 100% SPY
   
10. **AI & Cloud MegaCap** (ai_cloud_megacap_wave)
    - Benchmark: 60% QQQ + 25% SMH + 15% IGV

## Excluded Waves

- **S&P 500 Wave** (sp500_wave): Remains static with 100% SPY benchmark

## Configuration File

Benchmarks are defined in: `data/benchmarks/equity_benchmarks.json`

### Structure

```json
{
  "version": "v1.0",
  "description": "Phase 1B: Dynamic Benchmark Definitions for Equity Waves",
  "rebalance_rule": "static_config",
  "last_updated": "2026-01-08",
  "benchmarks": {
    "wave_id": {
      "wave_id": "wave_id",
      "benchmark_name": "Human-readable name",
      "components": [
        {
          "ticker": "TICKER",
          "weight": 0.5,
          "name": "Component name"
        }
      ],
      "notes": "Description"
    }
  }
}
```

### Editing Benchmarks

To modify a benchmark:

1. **Edit the JSON file**: Update component tickers and/or weights in `data/benchmarks/equity_benchmarks.json`

2. **Ensure weights sum to 1.0**: All component weights for a wave must sum to 1.0 (within 0.01 tolerance)

3. **Validate locally**: Run the validation script:
   ```bash
   python validate_dynamic_benchmarks.py
   ```

4. **Run tests**: Execute the integration tests:
   ```bash
   pytest test_dynamic_benchmarks_integration.py -v
   ```

5. **Version the change**: Update the `last_updated` field and consider incrementing `version` for major changes

6. **Commit and push**: CI will automatically validate the changes

### Versioning Best Practices

- **Patch changes** (v1.0 → v1.0.1): Minor weight adjustments within existing components
- **Minor changes** (v1.0 → v1.1): Adding/removing components, significant weight changes
- **Major changes** (v1.0 → v2.0): Complete benchmark methodology overhaul

## Integration with Engine

The `waves_engine.py` has been enhanced with:

1. **`load_dynamic_benchmark_specs(path)`**: Loads benchmark config JSON
   - Returns dict with benchmark definitions
   - Falls back gracefully if file missing or invalid

2. **`build_benchmark_series_from_components(price_df, components)`**: Computes benchmark returns
   - Takes price DataFrame and component list
   - Returns weighted benchmark return series
   - Handles missing tickers with proportional reweighting
   - Ensures no trailing NaNs

3. **Dynamic benchmark integration in `_compute_core()`**:
   - Checks if wave has dynamic benchmark spec
   - Excludes S&P 500 Wave from dynamic logic
   - Falls back to traditional benchmarks if dynamic unavailable
   - Adds diagnostics to result.attrs["coverage"]["dynamic_benchmark"]

## Diagnostics

When a wave uses a dynamic benchmark, the result includes diagnostic information in the `attrs["coverage"]` dictionary:

```python
result.attrs["coverage"]["dynamic_benchmark"] = {
    "enabled": True,
    "benchmark_name": "Tech & Semiconductors Composite",
    "version": "v1.0",
    "components": [
        {
            "ticker": "QQQ",
            "weight": 0.5,
            "available": True
        },
        {
            "ticker": "SMH",
            "weight": 0.5,
            "available": True
        }
    ]
}
```

For the S&P 500 Wave or waves without dynamic benchmarks:

```python
result.attrs["coverage"]["dynamic_benchmark"] = {
    "enabled": False,
    "reason": "S&P 500 Wave excluded"  # or "no_dynamic_spec_found"
}
```

## Validation

### Local Validation

Run the validation script before committing changes:

```bash
python validate_dynamic_benchmarks.py
```

This checks:
- ✓ Config file exists and parses correctly
- ✓ All 10 equity waves have valid benchmark definitions
- ✓ Weights sum to 1.0 (±0.01 tolerance)
- ✓ All tickers exist in cached price data
- ✓ Benchmarks have sufficient history (>60 days)
- ✓ S&P 500 Wave is excluded

### Automated Tests

Run integration tests:

```bash
pytest test_dynamic_benchmarks_integration.py -v
```

Tests verify:
- ✓ All waves load dynamic benchmarks successfully
- ✓ Benchmark series compute without error
- ✓ Benchmark series align with wave returns (same last date)
- ✓ No benchmarks are all-NaN
- ✓ S&P 500 Wave remains static

### CI Pipeline

GitHub Actions automatically runs both validation and tests on every push:

```yaml
- name: Validate dynamic benchmarks
  run: python validate_dynamic_benchmarks.py

- name: Test dynamic benchmarks
  run: pytest test_dynamic_benchmarks_integration.py -v
```

## Why S&P 500 Benchmark Remains Static

The S&P 500 Wave uses a **static SPY-only benchmark** for these reasons:

1. **Simplicity**: SPY is the canonical broad market benchmark
2. **Stability**: No need for weighted composite when wave itself is pure SPY
3. **Consistency**: Maintains historical continuity for the flagship wave
4. **Clarity**: Avoids confusion about what "the market" means

This exception is hardcoded in the engine:

```python
# S&P 500 Wave always uses static SPY benchmark (excluded from dynamic benchmarks)
if not freeze_benchmark and wave_id != "sp500_wave":
    # Load dynamic benchmark specs...
```

## Fallback Behavior

The system includes multiple layers of graceful degradation:

1. **Missing config file**: Falls back to traditional `BENCHMARK_WEIGHTS_STATIC`
2. **Missing wave spec**: Uses `get_auto_benchmark_holdings()` or static weights
3. **Missing component tickers**: Reweights proportionally using available components
4. **All components missing**: Sets benchmark returns to NaN with logged warning

This ensures the system never crashes due to benchmark issues.

## Future Enhancements

Phase 1B is intentionally limited to equity waves. Future phases may include:

- **Phase 1C**: Dynamic benchmarks for crypto waves
- **Phase 1D**: Dynamic benchmarks for income waves
- **Phase 2**: Time-varying benchmark weights (rebalance rules)
- **Phase 3**: Factor-based benchmark construction

## Troubleshooting

### Validation fails: "weights sum to X, expected 1.0"

Check that component weights in the JSON sum to exactly 1.0:

```python
# Example fix:
components = [
    {"ticker": "QQQ", "weight": 0.6},
    {"ticker": "SMH", "weight": 0.4}  # 0.6 + 0.4 = 1.0 ✓
]
```

### Tests fail: "ticker not in cache"

Ensure all benchmark component tickers exist in `data/cache/prices_cache.parquet`. Update the cache if needed:

```bash
python build_price_cache.py
```

### Benchmark returns are all NaN

Check diagnostics to see which components are unavailable:

```python
result.attrs["coverage"]["dynamic_benchmark"]["components"]
```

Verify ticker symbols are correct and available in the cache.

## Support

For questions or issues with dynamic benchmarks:

1. Check this documentation
2. Run validation: `python validate_dynamic_benchmarks.py`
3. Review test output: `pytest test_dynamic_benchmarks_integration.py -v`
4. Check engine diagnostics: `result.attrs["coverage"]["dynamic_benchmark"]`

## Notes

- The **Infinity Multi-Asset Growth Wave** includes BTC-USD (20%) in its benchmark to match its multi-asset strategy allocation. This is the only equity-categorized wave with a crypto component in Phase 1B.
- Pure crypto waves (e.g., Crypto L1 Growth Wave, Crypto DeFi Growth Wave) are excluded from Phase 1B and retain their existing benchmark logic.

---

**Last Updated**: 2026-01-08  
**Version**: 1.0  
**Phase**: 1B - Equity Waves with Dynamic Benchmarks
