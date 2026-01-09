# Portfolio Snapshot N/A Metrics Fix

## Date
2026-01-09

## Problem Statement
Portfolio Snapshot card was showing N/A for all metrics (1D, 30D, 60D, 365D returns, alphas, and attribution) despite data being loaded in the system.

## Root Cause Analysis

### Primary Issue: Failed Ledger Caching
The Portfolio Snapshot card uses `compute_portfolio_alpha_ledger()` to calculate metrics and caches the result in `st.session_state['portfolio_alpha_ledger']` for performance.

**The bug**: The original code cached the ledger result **regardless of whether the computation succeeded or failed**.

```python
# OLD CODE (BUGGY)
ledger = compute_portfolio_alpha_ledger(...)
# Store in session state for reuse
st.session_state['portfolio_alpha_ledger'] = ledger  # ❌ Caches even if failed!
```

**Impact**: When the app first loaded and the PRICE_BOOK was temporarily unavailable (e.g., due to missing dependencies), a failed ledger with `success: False` would be cached. On subsequent renders, this failed cached ledger would be reused forever, causing all metrics to display as N/A even after data became available.

### Secondary Issue: Missing pyarrow Dependency
The `prices_cache_v2.parquet` file requires `pyarrow` to be read, but it was missing from `requirements.txt`. This likely caused the initial PRICE_BOOK load failure that triggered the caching bug.

## Solution Implemented

### 1. Conditional Caching (app.py line 10176)
Modified the session state caching logic to only cache successful ledger computations:

```python
# NEW CODE (FIXED)
ledger = compute_portfolio_alpha_ledger(...)
# Only cache successful ledger computations to allow retry on failure
if ledger['success']:
    st.session_state['portfolio_alpha_ledger'] = ledger
```

**Benefit**: If computation fails (e.g., PRICE_BOOK is empty), the ledger will NOT be cached, allowing automatic retry on the next render when data might be available.

### 2. Added pyarrow Dependency (requirements.txt)
Added `pyarrow>=14.0.0` to requirements.txt to ensure the parquet cache file can be read.

**Benefit**: Prevents PRICE_BOOK load failures due to missing dependency.

## Files Modified

1. **app.py** (line 10176)
   - Changed: Conditional caching of portfolio_alpha_ledger
   - Lines changed: 3 lines (added `if ledger['success']:` condition)

2. **requirements.txt**
   - Added: `pyarrow>=14.0.0` after numpy
   - Lines changed: 1 line

## Verification

### Test Created
Created `/tmp/test_portfolio_snapshot_fix.py` that verifies:
1. ✅ Failed ledger computations are NOT cached
2. ✅ Successful ledger computations ARE cached
3. ✅ Metrics are available in successful ledger

### Test Results
```
================================================================================
Portfolio Snapshot Session State Caching Fix Test
================================================================================

Test 1: Failed ledger computation should NOT be cached
✓ PASS: Failed ledger was NOT cached in session state

Test 2: Successful ledger computation should be cached
✓ PASS: Successful ledger was cached in session state

Test 3: Verify successful ledger contains valid metrics
✓ 1D metrics available: Cum realized: 0.0044, Total alpha: 0.0081
✓ 30D metrics available: Cum realized: 0.0581, Total alpha: 0.0471
✓ 60D metrics available: Cum realized: 0.0467, Total alpha: 0.0191

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

## Expected Behavior After Fix

### Before Fix
1. User loads app → PRICE_BOOK fails to load (missing pyarrow)
2. Failed ledger cached in session state
3. User installs pyarrow or cache becomes available
4. Portfolio Snapshot **still shows N/A** (using cached failed ledger)
5. User must manually clear session state or force ledger recompute

### After Fix
1. User loads app → PRICE_BOOK fails to load (missing pyarrow)
2. Failed ledger NOT cached (fix #1)
3. User installs pyarrow or cache becomes available (fix #2)
4. Next render → Portfolio Snapshot **automatically retries** and shows metrics
5. Successful ledger is cached for performance

## Deployment Checklist

- [x] Code changes committed
- [x] Added missing dependency to requirements.txt
- [x] Test created and verified
- [ ] Deploy to Streamlit Cloud (will auto-install pyarrow)
- [ ] Verify Portfolio Snapshot shows metrics after deployment
- [ ] Clear session state if needed (users can use Force Ledger Recompute button)

## Notes

### Why This Wasn't Caught Earlier
- The issue only manifests in specific deployment scenarios where dependencies are missing temporarily
- Local development likely had pyarrow installed globally
- The caching behavior made the issue persist even after the underlying problem was resolved

### Related Code Locations
- Line 6320 in app.py: Similar caching logic but already had success check (correct)
- Force Ledger Recompute button: Clears `portfolio_alpha_ledger` from session state as a manual workaround
