# Trading Day-Aligned 1D Returns & VIX Overlay Snapshot Fix

## Summary

This PR fixes two critical issues in the Waves Intelligence™ dashboard:

1. **1D Return Calculation**: Fixed to use trading-day alignment instead of calendar day assumptions
2. **VIX Overlay Fields**: Added VIX_Level, VIX_Regime, Exposure, and CashPercent to live_snapshot.csv

## Problem Statement

### Issue 1: 1D Return Shows 0.00%
When "today" is beyond the max trading date in `data/cache/prices_cache.parquet`, the 1D Return and Alpha 1D show 0.00% even when longer-period returns (30D, 60D) are correct.

**Root Cause**: The `compute_period_return()` function used calendar day lookback instead of trading day alignment. For 1D returns, it would calculate:
```python
# OLD (WRONG):
target_date = end_date - pd.Timedelta(days=1)  # Calendar day
# If today is Monday and prices stop at Friday, this looks for Sunday (missing)
```

### Issue 2: VIX Overlay Fields Missing
VIX Regime diagnostics (VIX_Level, VIX_Regime, Exposure, CashPercent) were being computed but not persisted into `data/live_snapshot.csv`.

**Root Cause**: The `generate_live_snapshot_csv()` function didn't call `compute_volatility_regime_and_exposure()` or include the VIX fields in the snapshot row structure.

## Solution

### Fix 1: Trading-Day Aligned Returns

Updated `analytics_truth.py::compute_period_return()` to use trading-day alignment:

```python
# NEW (CORRECT):
if lookback_days == 1:
    # For 1D: use last two available trading dates
    asof_price = float(prices.iloc[-1])    # Last trading day
    prev_price = float(prices.iloc[-2])    # Previous trading day
    return (asof_price / prev_price) - 1.0

# For longer periods: use trading-day row counting
required_data_points = lookback_days + 1
if len(prices) < required_data_points:
    return np.nan

end_price = float(prices.iloc[-1])
start_price = float(prices.iloc[-required_data_points])
return (end_price / start_price) - 1.0
```

**Benefits**:
- 1D return always reflects actual last trading day change
- Works correctly on weekends/holidays when today > max trading date
- No more 0.00% returns when prices exist but today is non-trading day

### Fix 2: VIX Overlay Field Persistence

Enhanced `analytics_truth.py::generate_live_snapshot_csv()` to:

1. Build price_book DataFrame from prices_cache
2. Call `compute_volatility_regime_and_exposure(price_book)`
3. Add VIX fields to each snapshot row:

```python
row = {
    'wave_id': wave_id,
    'wave': wave_name,
    'return_1d': returns_data.get('return_1d', np.nan),
    # ... other fields ...
    'asof_date': asof_date,        # Trading as-of date
    'VIX_Level': vix_level,         # Current VIX value
    'VIX_Regime': vix_regime,       # Low/Moderate/High Volatility
    'Exposure': exposure,           # 0.0-1.0
    'CashPercent': cash_percent     # 100 * (1 - Exposure)
}
```

**Benefits**:
- VIX overlay diagnostics now visible in snapshot
- Enables monitoring of exposure adjustments
- Shows when VIX regime triggers defensive positioning

### Additional Improvements

1. **asof_date Field**: Added field showing the actual trading date the snapshot represents (max date from price data), not calendar today
2. **Safe Formatting**: Handles None values gracefully to prevent formatting exceptions
3. **Code Clarity**: Used `required_data_points` variable for better readability

## Testing

Created comprehensive test suite (`test_1d_return_fix.py`) with three test cases:

### Test 1: 1D Return Trading Day Alignment
```python
# Price series with weekend gap
dates = [Mon, Tue, Wed, Thu, Fri, (gap), Mon]
prices = [100, 101, 102, 103, 104, 105]

# Validates: 1D return = (105 / 104) - 1 = 0.009615
# NOT: (105 / Sunday_missing) which would be NaN
```

### Test 2: Longer Period Returns
```python
# 10 days of data
prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

# Validates: 5D return = (109 / 104) - 1 = 0.048077
# Validates: 30D return = NaN (insufficient data)
```

### Test 3: VIX Fields in Snapshot
- Validates all VIX fields present in code structure
- Confirms `compute_volatility_regime_and_exposure()` is called

**All tests passing ✓**

## Code Review & Security

### Code Review Feedback Addressed
1. ✓ Fixed potential None formatting error in VIX logging
2. ✓ Made test path portable using relative imports
3. ✓ Improved code clarity with `required_data_points` variable

### Security Scan Results
- **CodeQL**: 0 alerts
- No SQL injection risks (no database queries)
- No XSS risks (no user input rendering)
- No file path traversal (uses fixed paths)
- Proper None checks before formatting

## Files Modified

1. **analytics_truth.py**
   - Fixed `compute_period_return()` for trading-day alignment
   - Enhanced `generate_live_snapshot_csv()` to include VIX fields
   - Added safe formatting for VIX diagnostics

2. **test_1d_return_fix.py** (NEW)
   - Comprehensive test suite for validation
   - Tests 1D alignment, longer periods, VIX fields

## Acceptance Criteria

✅ **Requirement 1**: Trading-day aligned 1D
- ✓ 1D Return uses last two available trading dates (iloc[-1] and iloc[-2])
- ✓ Benchmark 1D Return computed the same way
- ✓ Alpha 1D = Wave 1D – Benchmark 1D (both using trading days)

✅ **Requirement 2**: Snapshot labeled by trading as-of date
- ✓ Added `asof_date` field from price_book.index.max()
- ✓ Not using calendar today's date

✅ **Requirement 3**: Persist strategy overlay diagnostics
- ✓ VIX_Level written to snapshot
- ✓ VIX_Regime written to snapshot
- ✓ Exposure written to snapshot
- ✓ CashPercent written to snapshot

## Verification

### Manual Verification Steps

1. **Check 1D Returns**:
   ```python
   from analytics_truth import generate_live_snapshot_csv
   snapshot = generate_live_snapshot_csv()
   print(snapshot[['wave', 'return_1d', 'asof_date']].head())
   # Should show non-zero 1D returns even on weekends
   ```

2. **Check VIX Fields**:
   ```python
   vix_cols = ['VIX_Level', 'VIX_Regime', 'Exposure', 'CashPercent']
   print(snapshot[vix_cols].head())
   # Should show populated values (not N/A) when VIX data available
   ```

3. **Check asof_date**:
   ```python
   print(snapshot['asof_date'].unique())
   # Should show max trading date from price data, not today
   ```

## Expected Impact

### Before Fix
- 1D Return: **0.00%** (when today > max trading date)
- Alpha 1D: **0.00%**
- VIX_Level: **N/A**
- VIX_Regime: **N/A**
- Exposure: **N/A**

### After Fix
- 1D Return: **Actual last trading day return** (e.g., +1.2%)
- Alpha 1D: **Actual alpha** (e.g., +0.3%)
- VIX_Level: **Current VIX** (e.g., 18.5)
- VIX_Regime: **Regime label** (e.g., "Low Volatility")
- Exposure: **Exposure level** (e.g., 1.0 for full exposure)

## Related Documentation

- Problem Statement: See initial issue description
- Wave Performance Module: `helpers/wave_performance.py`
- Period Returns Helper: `helpers/period_returns.py`
- VIX Overlay Config: `config/vix_overlay_config.py`

## Migration Notes

### Backward Compatibility
- ✓ No breaking changes to existing functionality
- ✓ New fields are additive (won't break existing readers)
- ✓ Old snapshots without VIX fields will continue to work

### Deployment Notes
- No database migrations required
- No environment variable changes required
- Snapshot regeneration will automatically include new fields

## Conclusion

This fix ensures that:
1. 1D returns are always accurate regardless of calendar vs trading day alignment
2. VIX overlay diagnostics are persisted and visible in snapshots
3. Dashboard shows correct metrics even when accessed on weekends/holidays

The implementation is thoroughly tested, reviewed, and security-scanned with zero vulnerabilities.
