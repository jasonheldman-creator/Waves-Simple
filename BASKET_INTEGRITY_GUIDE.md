# Basket Integrity Validation System

## Overview

The Basket Integrity Validation System ensures that all Waves reference the correct canonical universal basket and corresponding weights files. This system was implemented to eliminate mismatched references causing errors and ensure consistent Wave activation across the platform.

## Canonical Files

### 1. Universal Basket: `universal_universe.csv`
- **Location**: Root directory
- **Purpose**: Single source of truth for all allowed assets
- **Contents**: 143 tickers including:
  - Russell 3000 components (broad US equity)
  - S&P 500 components (large-cap US equity)
  - Russell 2000 components (small-cap US equity)
  - Top cryptocurrencies (BTC, ETH, and 100+ altcoins)
  - Income ETFs (dividend, bond, muni)
  - Safe assets (treasury bills, money market)
  - Sector ETFs (energy, technology, infrastructure)

### 2. Wave Weights: `wave_weights.csv`
- **Location**: Root directory
- **Purpose**: Defines portfolio allocations for each Wave
- **Structure**: 
  - `wave`: Wave display name
  - `ticker`: Asset ticker symbol
  - **weight**: Allocation weight (0.0 to 1.0)

**Important**: Weights may sum to less than 1.0 due to SmartSafe gating. The remainder is dynamically allocated to safe assets based on market regime.

### 3. Wave Configuration: `wave_config.csv`
- **Location**: Root directory
- **Purpose**: Defines benchmark and mode for each Wave
- **Structure**:
  - `Wave`: Wave display name
  - `Benchmark`: Benchmark ticker (must exist in universal basket)
  - `Mode`: Strategy mode (Standard, Alpha-Minus-Beta, Private Logic)
  - `Beta_Target`: Target beta relative to benchmark
  - `enabled`: Boolean flag for Wave activation

## Wave Registry

The canonical Wave registry is defined in `waves_engine.py`:

```python
WAVE_ID_REGISTRY: Dict[str, str] = {
    "sp500_wave": "S&P 500 Wave",
    "russell_3000_wave": "Russell 3000 Wave",
    # ... 28 total waves
}
```

All 28 waves are:
1. S&P 500 Wave
2. Russell 3000 Wave
3. US MegaCap Core Wave
4. AI & Cloud MegaCap Wave
5. Next-Gen Compute & Semis Wave
6. Future Energy & EV Wave
7. EV & Infrastructure Wave
8. US Small-Cap Disruptors Wave
9. US Mid/Small Growth & Semis Wave
10. Small Cap Growth Wave
11. Small to Mid Cap Growth Wave
12. Future Power & Energy Wave
13. Quantum Computing Wave
14. Clean Transit-Infrastructure Wave
15. Income Wave
16. Demas Fund Wave
17. Crypto L1 Growth Wave
18. Crypto L2 Growth Wave
19. Crypto DeFi Growth Wave
20. Crypto AI Growth Wave
21. Crypto Broad Growth Wave
22. Crypto Income Wave
23. SmartSafe Treasury Cash Wave
24. SmartSafe Tax-Free Money Market Wave
25. Gold Wave
26. Infinity Multi-Asset Growth Wave
27. Vector Treasury Ladder Wave
28. Vector Muni Ladder Wave

## Validation Module

### `helpers/basket_integrity.py`

This module provides comprehensive validation of the basket integrity:

#### Functions:

**`validate_basket_integrity() -> BasketIntegrityReport`**
- Runs all validation checks
- Returns detailed report with issues categorized by severity

**`print_basket_integrity_report(verbose: bool = False)`**
- Prints basket integrity report to console
- Use `--verbose` flag for detailed output

#### Validation Checks:

1. **Wave Registry Completeness**
   - Ensures all 28 waves from WAVE_ID_REGISTRY have weight definitions
   - Detects extra waves not in registry

2. **Ticker Existence**
   - Validates all tickers in wave_weights.csv exist in universal_universe.csv
   - Reports missing tickers as critical issues

3. **Benchmark Validation**
   - Ensures all benchmark tickers exist in universal basket
   - Exempts benchmarks where appropriate

4. **Weight Sum Validation**
   - Checks that weight sums are valid (0.0 to 1.01)
   - Allows weights < 1.0 for SmartSafe gating
   - Reports invalid sums as critical issues

5. **Configuration Completeness**
   - Ensures all waves have configuration entries
   - Warns about missing configs (non-critical)

#### Severity Levels:

- **Critical**: Blocks proper Wave functionality, must be fixed
- **Warning**: Should be addressed but doesn't block operation
- **Info**: Informational messages about system state

## Integration with Startup Validation

The basket integrity check is integrated into `helpers/startup_validation.py`:

```python
checks = [
    ReadinessCheck("Data Files", check_data_files, critical=True),
    ReadinessCheck("Universal Universe", check_universal_universe, critical=True),
    ReadinessCheck("Python Packages", check_imports, critical=True),
    ReadinessCheck("Helper Modules", check_helpers_available, critical=True),
    ReadinessCheck("Waves Engine", check_waves_engine, critical=True),
    ReadinessCheck("Basket Integrity", check_basket_integrity, critical=False),
    ReadinessCheck("Resilience Features", check_resilience_features, critical=False),
]
```

**Note**: Basket integrity is non-critical to allow the app to start with warnings. Issues are logged for review but don't prevent Wave loading.

## Testing

### Run Basket Integrity Test Suite

```bash
python test_basket_integrity.py
```

This runs comprehensive tests to verify:
- All 28 waves have weight definitions
- All tickers exist in universal basket
- All benchmarks are valid
- Weight sums are within acceptable ranges

### Run Basket Integrity Validation

```bash
python helpers/basket_integrity.py --verbose
```

This prints a detailed report of all validation checks.

## SmartSafe Gating

Several waves use **SmartSafe gating**, where the defined weights sum to less than 1.0. The remainder is dynamically allocated to safe assets (treasury bills, money market funds) based on market regime.

Waves using SmartSafe gating:
- AI & Cloud MegaCap Wave (78%)
- Future Energy & EV Wave (90%)
- Next-Gen Compute & Semis Wave (84%)
- US MegaCap Core Wave (45%)

The gating fraction is determined by:
- Market regime (panic, downtrend, neutral, uptrend)
- Wave mode (Standard, Alpha-Minus-Beta, Private Logic)
- VIX levels (for equity waves)
- BTC volatility (for crypto waves)

## Maintenance

### Adding a New Ticker

1. Add ticker to `universal_universe.csv`:
```csv
ticker,name,asset_class,index_membership,sector,market_cap_bucket,status,validated,validation_error
NEWTICK,New Ticker Inc,equity,WAVE_SOME_WAVE,Technology,Large,active,not_checked,
```

2. Run validation:
```bash
python test_basket_integrity.py
```

### Adding a New Wave

1. Define wave in `waves_engine.py`:
```python
WAVE_WEIGHTS: Dict[str, List[Holding]] = {
    "New Wave Name": [
        Holding("TICK1", 0.50, "Ticker 1"),
        Holding("TICK2", 0.50, "Ticker 2"),
    ],
}

WAVE_ID_REGISTRY: Dict[str, str] = {
    "new_wave": "New Wave Name",
}
```

2. Add weights to `wave_weights.csv`:
```csv
New Wave Name,TICK1,0.50
New Wave Name,TICK2,0.50
```

3. Add configuration to `wave_config.csv`:
```csv
New Wave Name,SPY,Standard,0.90,True
```

4. Run validation:
```bash
python test_basket_integrity.py
```

## Troubleshooting

### Issue: "Tickers missing from universal basket"
**Solution**: Add missing tickers to `universal_universe.csv`

### Issue: "Waves missing from wave_config.csv"
**Solution**: Add wave configuration entries for all waves

### Issue: "Weight sum != 1.0"
**Analysis**: Check if this is intentional SmartSafe gating (weights < 1.0) or an error (weights > 1.01)

### Issue: "Benchmark ticker not in universe"
**Solution**: Add benchmark ticker to `universal_universe.csv` with `BENCHMARK` in `index_membership`

## Future Enhancements

1. **Price Fetch Validation**: Add runtime validation to check which tickers fail price fetches
2. **Historical Data Validation**: Verify minimum historical data availability for each ticker
3. **Correlation Analysis**: Validate Wave diversification by checking ticker correlations
4. **Auto-Repair**: Automatically fix minor issues (e.g., add missing tickers)
5. **Performance Monitoring**: Track validation performance over time

## References

- `waves_engine.py`: Wave definitions and registry
- `helpers/basket_integrity.py`: Validation module
- `helpers/startup_validation.py`: Startup checks
- `test_basket_integrity.py`: Test suite
