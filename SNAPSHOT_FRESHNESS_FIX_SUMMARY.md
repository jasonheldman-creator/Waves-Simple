# Snapshot Freshness Fix - Implementation Summary

## Problem Statement

The Waves-Simple project had a critical issue where portfolio and wave snapshot values (like `Return_1D`) remained stale even when SPY price data was updated. This occurred because:

1. Snapshot dates used `datetime.now()` instead of the actual SPY price date from the price cache
2. No explicit check triggered snapshot rebuilds when price data advanced
3. Cached snapshots were silently reused even when newer price data was available
4. No visibility into snapshot freshness decisions

This led to incorrect metrics on trading days with market movement, giving the false impression of no market changes.

## Solution Overview

The fix ensures snapshots always use the authoritative SPY price date and automatically rebuild when price data advances. Key changes:

### 1. Snapshot Date from SPY Price Data (Not datetime.now())

**Before:**
```python
current_date = datetime.now().strftime("%Y-%m-%d")
```

**After:**
```python
# Priority 1: Get from prices_cache_meta.json
spy_max_date = cache_meta.get("spy_max_date")

# Priority 2: Extract from SPY column in prices_cache
if 'SPY' in prices_cache:
    spy_max_date = prices_cache['SPY'].index.max()

# Error if unavailable (no datetime.now() fallback)
if snapshot_date_str is None:
    raise RuntimeError("Unable to determine SPY-based snapshot date")
```

### 2. Explicit Rebuild Triggering

**Before:**
```python
# Only checked age
if age_hours < MAX_SNAPSHOT_AGE_HOURS:
    return cached_snapshot
```

**After:**
```python
# Check if price data is newer
if prices_cache_max_date > snapshot_date:
    print("Price data updated - rebuilding snapshot")
    return generate_snapshot(force_refresh=True, generation_reason='newer_price_data')

# Then check age
if age_hours < MAX_SNAPSHOT_AGE_HOURS:
    return cached_snapshot
```

### 3. Streamlit-Visible Logging

Added comprehensive logging for every snapshot freshness decision:

```
================================================================================
📊 SNAPSHOT FRESHNESS CHECK
================================================================================
Max SPY Price Date:    2026-01-09
Current Snapshot Date: 2026-01-08
Snapshot Rebuild:      YES
Rebuild Justification: Price data updated (SPY max: 2026-01-09, Snapshot: 2026-01-08)
================================================================================
```

## Files Modified

### snapshot_ledger.py

1. **`_get_snapshot_date(price_df)`**
   - Prioritizes `spy_max_date` from `data/cache/prices_cache_meta.json`
   - Falls back to extracting from price_df or prices_cache.parquet
   - Raises `RuntimeError` if no SPY date available (no datetime.now() fallback)

2. **`load_snapshot(force_refresh)`**
   - Added comparison: `prices_cache_max_date > snapshot_date`
   - Forces rebuild when price data is newer
   - Logs detailed freshness check results

3. **`generate_snapshot(...)`**
   - Checks price data freshness before using cached snapshot
   - Enhanced logging for cache invalidation reasons

### analytics_truth.py

1. **`generate_live_snapshot_csv(...)`**
   - Extracts SPY max date from prices_cache
   - Falls back to prices_cache_meta.json
   - Uses SPY date for all snapshot date fields
   - Raises error if SPY date unavailable

## Testing

### Automated Verification

Run the verification script:
```bash
python verify_snapshot_fix.py
```

This validates that all key fixes are in place:
- ✓ _get_snapshot_date() raises error when no price data
- ✓ _get_snapshot_date() reads spy_max_date from metadata
- ✓ load_snapshot() compares price cache date vs snapshot date
- ✓ load_snapshot() has detailed freshness logging
- ✓ generate_snapshot() checks if price data is newer
- ✓ All analytics_truth.py fixes in place

### Manual Testing

To verify the fix works end-to-end:

1. **Check current snapshot date:**
   ```bash
   head -2 data/live_snapshot.csv
   ```
   Note the `Date` column value.

2. **Check SPY max date in cache metadata:**
   ```bash
   cat data/cache/prices_cache_meta.json | grep spy_max_date
   ```

3. **Verify they match:**
   - If `spy_max_date > snapshot_date`: Snapshot should auto-rebuild on next load
   - If they match: Snapshot is current

4. **Trigger rebuild with updated price data:**
   ```bash
   # Update price cache (advances spy_max_date)
   python build_price_cache.py
   
   # Load snapshot (should detect newer price data and rebuild)
   python -c "from snapshot_ledger import load_snapshot; load_snapshot()"
   
   # Check logs for freshness decision
   ```

5. **Verify in Streamlit app:**
   - Start the app
   - Check for "SNAPSHOT FRESHNESS CHECK" logs in console
   - Verify snapshot date matches SPY max date
   - Confirm Return_1D values reflect latest market data

### Expected Behavior

**Scenario 1: Price data advances (e.g., new trading day)**
```
Max SPY Price Date:    2026-01-10  (NEW)
Current Snapshot Date: 2026-01-09  (OLD)
Snapshot Rebuild:      YES
Rebuild Justification: Price data updated
```

**Scenario 2: Snapshot is current**
```
Max SPY Price Date:    2026-01-09
Current Snapshot Date: 2026-01-09
Snapshot Rebuild:      NO
Rebuild Justification: Snapshot is current (dates match and age < 24 hours)
```

**Scenario 3: No price data available**
```
CRITICAL ERROR: Unable to determine SPY-based snapshot date.
Both prices_cache_meta.json and prices_cache.parquet are unavailable or invalid.
Snapshot generation cannot proceed without authoritative price data.
```

## Benefits

1. **Accurate Metrics**: Return_1D and other metrics now reflect actual market data as of the latest SPY trading day
2. **Automatic Updates**: Snapshots automatically rebuild when price data advances (no manual intervention)
3. **Clear Visibility**: Detailed logging shows exactly why snapshots are/aren't rebuilt
4. **Data Integrity**: Snapshots can never use datetime.now() for dates - always SPY-based
5. **Error Detection**: Fails fast if price data is unavailable instead of using stale dates

## Deployment Notes

### Requirements
- `data/cache/prices_cache_meta.json` must contain `spy_max_date`
- Price cache workflows must update `spy_max_date` on each run
- Existing snapshot file will be invalidated if price data is newer

### Rollout
1. Deploy code changes
2. Ensure price cache is up-to-date (`python build_price_cache.py`)
3. Snapshot will auto-rebuild on first load if needed
4. Monitor logs for "SNAPSHOT FRESHNESS CHECK" messages

### Monitoring
- Check snapshot Date matches spy_max_date
- Verify Return_1D changes on trading days with market movement
- Monitor for "CRITICAL ERROR" messages (indicates missing price data)

## Related Issues

This fix addresses the core problem described in the issue:
- Portfolio snapshot values remaining stale
- Return_1D showing 0% on days with market movement
- Disconnect between price cache updates and snapshot dates

## Code Review Checklist

- [x] Removed all datetime.now() usage for snapshot dates
- [x] Added SPY-based date extraction from price cache
- [x] Implemented explicit rebuild triggering on price data updates
- [x] Added comprehensive logging for freshness decisions
- [x] Verified guaranteed file writes on rebuild
- [x] Validated with verification script
- [x] Compile checks pass
- [x] Changes are minimal and focused on snapshot freshness only

## Next Steps

After merging, verify:
1. Snapshots update correctly when price cache advances
2. Return_1D values reflect actual market changes
3. No false "stale data" warnings appear
4. All logging shows correct dates and rebuild decisions
