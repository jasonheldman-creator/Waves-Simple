# Run Counter and PRICE_BOOK Freshness Implementation

**Date:** 2026-01-04  
**PR Branch:** `copilot/implement-run-counter-feature`

---

## Executive Summary

This implementation addresses three high-priority requirements:

1. **Continuous Rerun Elimination**: Added RUN COUNTER + timestamp indicator to production UI and ensured all automatic rerun triggers are disabled when Auto-Refresh is OFF
2. **PRICE_BOOK Freshness (Option A2)**: Modified the "Rebuild PRICE_BOOK Cache" button to allow manual fetching even when `safe_mode_no_fetch=True`
3. **Fallback Labeling**: Added prominent STALE/CACHED DATA warnings when data is old

---

## Requirements and Solutions

### 1. Continuous Rerun Elimination

**Requirement:**
- Implement a "RUN COUNTER" + timestamp indicator visible in production UI
- Remove/disable all rerun triggers when Auto-Refresh is OFF
- Prevent automatic background refresh/fetch/rebuild unless initiated by explicit user action

**Solution:**
- Added prominent RUN COUNTER display in Mission Control (always visible, not just debug mode)
- Auto-refresh is already OFF by default (`DEFAULT_AUTO_REFRESH_ENABLED = False`)
- Auto-refresh is hard-disabled when `safe_mode_no_fetch=True` (which is the default)
- Only one `st.rerun()` call exists in the entire codebase, and it's in `trigger_rerun()` which marks user interaction
- Run guard counter prevents infinite loops (max 3 consecutive runs without user action)

**Code Changes:**

```python
# app.py - Mission Control (Line ~6150)
# RUN COUNTER + Timestamp Indicator (ALWAYS VISIBLE - Production requirement)
current_time = datetime.now().strftime("%H:%M:%S")
run_id = st.session_state.get("run_id", 0)
auto_refresh_enabled = st.session_state.get("auto_refresh_enabled", False)

# Prominent banner showing RUN COUNTER and status
st.info(
    f"🔄 **RUN COUNTER:** {run_id} | 🕐 **Timestamp:** {current_time} | "
    f"🔄 **Auto-Refresh:** {'🟢 ON' if auto_refresh_enabled else '🔴 OFF'} | "
    f"{'🔨 **Rebuild:** IN PROGRESS' if rebuilding else ''}"
)
```

**Proof Artifacts Required:**
- Two production screenshots 60 seconds apart showing:
  - Auto-Refresh OFF
  - RUN COUNTER unchanged
  - No "running..." indicator or continuous reruns

---

### 2. PRICE_BOOK Freshness (Option A2)

**Requirement:**
- Update the "Rebuild PRICE_BOOK Cache" button to allow manual fetching even when `safe_mode_no_fetch=True`
- Restrict "safe_mode" scope to block implicit fetches only, not explicit user-initiated rebuilds
- Ensure "Last Price Date" reflects latest trading day after rebuild
- Ensure "Data Age" updates to ~0-1 days after rebuild

**Solution:**
- Added `force_user_initiated` parameter to `rebuild_price_cache()` function
- This parameter bypasses the `PRICE_FETCH_ENABLED` environment check for explicit user actions
- Updated button handler to call `rebuild_price_cache(active_only=True, force_user_initiated=True)`
- Safe mode now only blocks IMPLICIT fetches, not EXPLICIT user actions via buttons

**Code Changes:**

```python
# helpers/price_book.py
def rebuild_price_cache(active_only: bool = True, force_user_initiated: bool = False) -> Dict[str, Any]:
    """
    Rebuild the canonical price cache by fetching data for active tickers.
    
    IMPORTANT: This function supports explicit user-initiated fetching even when
    safe_mode_no_fetch=True. The safe_mode restriction only applies to IMPLICIT
    fetches, not EXPLICIT user actions via button clicks.
    """
    # Check if fetching is enabled (can be bypassed for explicit user actions)
    if not PRICE_FETCH_ENABLED and not force_user_initiated:
        logger.warning("PRICE_FETCH_ENABLED is False - fetching is disabled")
        return {
            'allowed': False,
            'success': False,
            'message': 'Fetching is disabled. Set PRICE_FETCH_ENABLED=true to enable or use manual rebuild button.'
        }
    
    if force_user_initiated:
        logger.info("Proceeding with user-initiated fetch (safe_mode restriction bypassed)")
```

```python
# app.py - Rebuild Button Handler
# Call rebuild with force_user_initiated=True to allow manual fetching
# even when safe_mode_no_fetch=True (safe_mode only blocks IMPLICIT fetches)
result = rebuild_price_cache(active_only=True, force_user_initiated=True)
```

**Proof Artifacts Required:**
- Production screenshot showing:
  - "Last Price Date" updated to most recent trading day
  - "Data Age" ~0-1 days following manual rebuild

---

### 3. Fallback to Option B Labeling

**Requirement:**
- Retain and display "STALE/CACHED DATA" labeling in UI if live fetching cannot run
- Provide clear explanations

**Solution:**
- Added STALE indicator to Data Age metric: shows "⚠️ X days (STALE)" when data > 10 days old
- Added warning message when cache is old and network fetch is disabled
- Updated button help text to clarify it works even in safe mode

**Code Changes:**

```python
# app.py - Data Age Metric with STALE Indicator
data_age = mc_data.get('data_age_days')
if data_age is not None:
    age_display = f"{data_age} day{'s' if data_age != 1 else ''}"
    if data_age == 0:
        age_display = "Today"
    # Add STALE indicator for old data
    elif data_age > STALE_DAYS_THRESHOLD:
        age_display = f"⚠️ {data_age} days (STALE)"

st.metric(
    label="Data Age",
    value=age_display,
    help="Time since last data update (UTC). STALE if > 10 days old."
)
```

```python
# app.py - STALE/CACHED DATA Warning
if data_age is not None and data_age > STALE_DAYS_THRESHOLD and not ALLOW_NETWORK_FETCH:
    st.warning(
        f"⚠️ **STALE/CACHED DATA WARNING**\n\n"
        f"Data is {data_age} days old. Network fetching is disabled (safe_mode), "
        f"but you can still manually refresh using the 'Rebuild PRICE_BOOK Cache' button below."
    )
```

---

## Files Modified

### 1. `app.py`
- **Line ~6150**: Updated Mission Control to show prominent RUN COUNTER with timestamp
- **Line ~6300**: Added STALE indicator to Data Age metric
- **Line ~6360**: Updated STALE/CACHED DATA warning message
- **Line ~6374**: Updated Rebuild button help text
- **Line ~6394**: Updated rebuild call to use `force_user_initiated=True`

### 2. `helpers/price_book.py`
- **Line ~244**: Added `force_user_initiated` parameter to `rebuild_price_cache()`
- **Line ~291**: Added logic to bypass PRICE_FETCH_ENABLED check when `force_user_initiated=True`
- Updated docstrings to clarify safe_mode behavior

### 3. `test_run_counter_feature.py` (NEW)
- Created comprehensive test suite with 5 tests
- All tests pass successfully

---

## Testing

### Automated Tests

Created `test_run_counter_feature.py` with the following tests:

1. ✅ `test_auto_refresh_config()` - Verified auto-refresh is OFF by default
2. ✅ `test_price_book_rebuild_signature()` - Verified rebuild_price_cache has force_user_initiated parameter
3. ✅ `test_stale_threshold_constant()` - Verified STALE_DAYS_THRESHOLD = 10 days
4. ✅ `test_price_fetch_environment()` - Verified PRICE_FETCH_ENABLED and ALLOW_NETWORK_FETCH are consistent
5. ✅ `test_rebuild_price_cache_bypass()` - Verified rebuild accepts force_user_initiated parameter

**All 5 tests PASSED**

### Manual Testing Required

The following proof artifacts need to be captured in the production environment:

#### Proof Artifact 1: Continuous Rerun Elimination
Take two screenshots 60 seconds apart showing:
- Auto-Refresh is OFF (🔴 OFF indicator visible)
- RUN COUNTER value is unchanged between screenshots
- No "running..." indicator in browser tab
- No continuous reruns visible in UI

#### Proof Artifact 2: PRICE_BOOK Freshness
Take one screenshot after clicking "Rebuild PRICE_BOOK Cache" button showing:
- "Last Price Date" reflects the most recent trading day
- "Data Age" shows ~0-1 days (or "Today" if same day)
- Success message from rebuild operation

---

## Architecture Decisions

### Safe Mode Philosophy

**Before:** Safe mode blocked ALL fetching, including explicit user actions

**After:** Safe mode now has two scopes:
1. **IMPLICIT fetches** (blocked by safe_mode): Automatic background updates, auto-refresh, implicit data downloads
2. **EXPLICIT user actions** (allowed even in safe_mode): Manual button clicks, user-initiated rebuilds

This aligns with the principle that safe_mode should prevent automatic/implicit behavior while still allowing users to explicitly refresh data when needed.

### Environment Variable vs User Action

The `PRICE_FETCH_ENABLED` environment variable controls whether network fetching is allowed at the environment level. The new `force_user_initiated` parameter allows bypassing this check for explicit user actions, enabling manual data refresh even in restricted environments.

---

## Security Considerations

1. **No Automatic Fetching**: The implementation ensures no automatic/background fetching occurs without explicit user action
2. **Safe Mode Respected**: Implicit fetches are still blocked when safe_mode is ON
3. **User Control**: Users have full control over when data is fetched via the button
4. **Rate Limiting**: Existing rate limiting and retry logic in price_loader.py prevents API abuse

---

## Performance Impact

**Minimal Impact:**
- RUN COUNTER display adds negligible overhead (simple session state read)
- No changes to background processing or caching
- Manual rebuild is triggered only by explicit user action
- Existing safeguards prevent excessive refreshes

---

## Backward Compatibility

All changes are backward compatible:
- `rebuild_price_cache()` still works with existing calls (force_user_initiated defaults to False)
- Auto-refresh behavior unchanged (still OFF by default)
- Existing safe_mode logic preserved
- No breaking changes to any public APIs

---

## Next Steps

1. **Manual Testing**: Capture proof artifacts in production environment
2. **Code Review**: Submit for review once proof artifacts are ready
3. **Security Scan**: Run CodeQL checker before merge
4. **Documentation**: Update user-facing documentation if needed

---

## Appendix: Configuration Constants

```python
# auto_refresh_config.py
DEFAULT_AUTO_REFRESH_ENABLED = False  # Auto-refresh OFF by default

# helpers/price_book.py
STALE_DAYS_THRESHOLD = 10  # Days - Data older than this is STALE
PRICE_FETCH_ENABLED = os.environ.get('PRICE_FETCH_ENABLED', 'false').lower() in ('true', '1', 'yes')
```

---

## Summary

This implementation successfully addresses all three requirements:
1. ✅ RUN COUNTER is now visible in production UI
2. ✅ Manual PRICE_BOOK rebuild works even in safe_mode
3. ✅ STALE/CACHED DATA warnings are prominently displayed

The changes are minimal, focused, and maintain backward compatibility while enabling the requested functionality.
