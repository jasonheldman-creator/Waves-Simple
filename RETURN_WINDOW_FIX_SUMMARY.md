# Return Window Calculation Fix - Implementation Summary

## Overview

This implementation fixes return window calculations to use proper trading-day windows instead of calendar positions, and ensures wave and benchmark returns are computed over aligned date ranges for accurate alpha calculations.

## Problem Statement

The original implementation had two key issues:

1. **Calendar vs Trading Days**: Return calculations used array positions (e.g., `iloc[-(period+1)]`) which treated the period as a count of array positions rather than trading days. This meant "365D" would look back 365 positions in the data, not 252 trading days (the standard market year).

2. **Benchmark Alignment**: Wave and benchmark returns could be computed over different date ranges, leading to inaccurate alpha calculations.

## Solution

### 1. Canonical Helper Function (`helpers/period_returns.py`)

Created a single source of truth for period return calculations:

```python
compute_period_return(price_series, trading_days) -> float
```

**Key Features:**
- Uses adjusted close/close series from cached price dataframe
- Drops NaN values and sorts index ascending
- Requires at least `trading_days + 1` data points
- Returns 0.0 for insufficient data (or None with flag)
- Proper error handling for edge cases

**Trading Day Mapping:**
```python
TRADING_DAYS_MAP = {
    '1D': 1,      # 1 trading day
    '30D': 30,    # 30 trading days
    '60D': 60,    # 60 trading days
    '365D': 252,  # 252 trading days (market standard)
}
```

**Additional Functions:**
- `align_series_for_alpha(wave_prices, benchmark_prices)` - Aligns series to common date range
- `compute_alpha(wave_prices, benchmark_prices, trading_days)` - Computes alpha with aligned series

### 2. Integration with Wave Performance Module

Updated `helpers/wave_performance.py` to use the canonical helper in all return calculation paths:

**Modified Functions:**
1. `compute_wave_returns()` - Wave-level returns
2. `compute_portfolio_snapshot()` - Portfolio-level returns and alpha
3. `compute_portfolio_snapshot_with_overlay()` - Overlay portfolio returns
4. `compute_portfolio_alpha_ledger()` - Alpha ledger with attribution

**Implementation Pattern:**
```python
# Map display period to trading days
if period == 365:
    trading_days = TRADING_DAYS_MAP['365D']  # 252
elif period == 60:
    trading_days = TRADING_DAYS_MAP['60D']   # 60
elif period == 30:
    trading_days = TRADING_DAYS_MAP['30D']   # 30
else:
    trading_days = period

# Use canonical helper
ret = compute_period_return(portfolio_values, trading_days)
```

### 3. Benchmark Alignment

The `align_series_for_alpha()` function ensures both wave and benchmark use the same date range:

```python
common_index = wave_prices.index.intersection(bench_prices.index)
aligned_wave = wave_prices.loc[common_index].sort_index()
aligned_benchmark = benchmark_prices.loc[common_index].sort_index()
```

This prevents scenarios where wave and benchmark are computed over different time periods, which would produce invalid alpha values.

## Testing

### Unit Tests (`test_period_returns.py`)

10 comprehensive tests with synthetic data:

1. ✅ Basic return computation with known values
2. ✅ 30 trading day window (1% daily return → 34.78% cumulative)
3. ✅ 60 trading day window (0.5% daily return → 34.89% cumulative)
4. ✅ 252 trading day window (0.05% daily return → 13.42% cumulative)
5. ✅ Insufficient data handling (returns 0.0 or None)
6. ✅ Edge cases (NaN, zero prices, empty series)
7. ✅ Benchmark alignment (different date ranges)
8. ✅ Alpha computation with known relationship
9. ✅ Alpha with misaligned dates
10. ✅ Trading days map constants verification

**All tests pass** ✓

### Integration Tests (`test_period_returns_integration.py`)

Validates end-to-end functionality with actual price data:

```
Wave returns (S&P 500 Wave):
  1D: +0.00% (1 trading day)
  30D: +1.01% (30 trading days)
  60D: +1.92% (60 trading days)
  365D: +23.53% (252 trading days) ✓

Portfolio returns (27 waves):
  1D: +0.00%
  30D: -0.81%
  60D: -3.10%
  365D: +13.04%
```

**Integration test passes** ✓

### Existing Tests

Ran existing wave performance tests to ensure no regressions:

```bash
python test_wave_performance.py
```

**All existing tests pass** ✓

## Security Verification

Ran CodeQL security scanner:

```
Analysis Result: Found 0 alerts
```

**No security vulnerabilities** ✓

## Files Modified

### New Files
1. `helpers/period_returns.py` (285 lines) - Canonical helper functions
2. `test_period_returns.py` (477 lines) - Comprehensive unit tests  
3. `test_period_returns_integration.py` (143 lines) - Integration test

### Modified Files
1. `helpers/wave_performance.py` - Updated all period return calculations to use canonical helper with proper trading day mapping

## Constraints Met

✅ **No GitHub Actions workflow changes** - Workflows untouched  
✅ **No cache build logic changes** - Cache building logic unchanged  
✅ **Only calculation logic modified** - Changes limited to return calculations  
✅ **UI layout preserved** - No changes to UI labels or structure  
✅ **Fast tests without network** - All tests use synthetic or cached data  

## Impact

### Before
- "365D" used 365 array positions (incorrect for trading days)
- Wave and benchmark could use different date ranges
- Alpha calculations could be inaccurate
- Inconsistent behavior across different calculation paths

### After
- "365D" correctly uses 252 trading sessions (market standard)
- Wave and benchmark always use aligned date ranges
- Alpha calculations are accurate and consistent
- All calculation paths use the same canonical helper

## Example Output

```
Trading days map: {'1D': 1, '30D': 30, '60D': 60, '365D': 252}

Returns computed with trading day mapping:
  1D (1 trading days): 0.00%
  30D (30 trading days): 1.01%
  60D (60 trading days): 1.92%
  365D (252 trading days): 23.53%
  
✓ Verified: 365D uses 252 trading days
```

## Conclusion

This implementation successfully addresses the problem statement by:

1. Creating a single canonical helper for period returns
2. Using proper trading-day windows (252 for "365D")
3. Ensuring wave and benchmark alignment
4. Providing comprehensive test coverage
5. Maintaining backward compatibility
6. Following all specified constraints

All tests pass and the implementation is ready for production use.
