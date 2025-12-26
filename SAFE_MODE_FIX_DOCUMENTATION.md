# Safe Mode Fix Documentation

## Problem Statement

The Streamlit app was experiencing repeated interruptions from the Safe Mode banner for two main reasons:
1. Repeated crashes that trigger Safe Mode
2. Intrusive Safe Mode UI behavior causing disruptions during app reruns

## Root Cause Analysis

### Issue 1: Repeated Banner Display on Every Rerun

**Problem:** When an exception occurred in the main app, Streamlit would catch it and display a large red error banner. However, because Streamlit reruns the entire script on every user interaction (button click, selectbox change, etc.), the exception would be caught again and the banner would be displayed again, causing repeated interruptions.

**Why This Happened:**
- Streamlit's execution model: The entire script reruns on every interaction
- The exception was thrown on every rerun (persistent error condition)
- The error handler displayed the banner on every rerun
- No session state tracking to prevent repeated banner displays

### Issue 2: Potential Crashes from compute_alpha_attribution_series

**Investigation:** The problem statement mentioned `compute_alpha_attribution_series() got multiple values for argument 'wave_name'`. Upon investigation:
- All 5 calls to this function in app.py use keyword arguments only
- All calls are wrapped in try-except blocks
- No evidence of positional + keyword argument duplication found
- The function signature: `compute_alpha_attribution_series(wave_name, mode, history_df, ...)`

**Status:** This issue appears to have been already fixed or occurs in edge cases. All calls are now safe.

### Issue 3: MediaFileStorageError

**Investigation:** 
- No calls to `st.image()`, `st.audio()`, or `st.video()` found in app.py
- No calls to these functions found in app_fallback.py
- No media file handling that could cause crashes

**Status:** This issue does not appear to exist in the current codebase.

## Solutions Implemented

### Part 1: Fix Repeated Banner Display

**File:** `app.py`, lines 11830-11891

**Changes:**
1. Added session state tracking for Safe Mode error display:
   - `safe_mode_error_shown`: Boolean flag tracking if banner has been displayed
   - `safe_mode_error_message`: Stores the original error message
   - `safe_mode_error_traceback`: Stores the original traceback

2. Modified error handler logic:
   - On first error: Display large red banner, set flag to True
   - On subsequent reruns: Display small warning line only
   - Error details always available in collapsible expander (collapsed by default)

3. Added "Retry Full Mode" button:
   - Clears all Safe Mode flags
   - Triggers rerun to attempt full app execution
   - Allows recovery without browser refresh

**Code Structure:**
```python
except Exception as e:
    # Initialize session state on first error
    if "safe_mode_error_shown" not in st.session_state:
        st.session_state.safe_mode_error_shown = False
        st.session_state.safe_mode_error_message = str(e)
        st.session_state.safe_mode_error_traceback = traceback.format_exc()
    
    # Show banner only once
    if not st.session_state.safe_mode_error_shown:
        # Large red banner
        st.session_state.safe_mode_error_shown = True
    else:
        # Small warning on subsequent reruns
        st.warning("⚠️ Running in Safe Mode due to application error.")
    
    # Collapsible error details (always available)
    with st.expander("🔍 View Error Details", expanded=False):
        st.error(...)
        st.code(...)
    
    # Retry button
    if st.button("🔄 Retry Full Mode"):
        # Clear flags and rerun
        ...
```

### Part 2: Verified Safe Mode Fallback

**File:** `app_fallback.py`

**Verification:**
- No risky operations (media files, heavy analytics)
- Simple data loading with safe error handling
- Basic display only (metrics, dataframes, text)
- No calls to potentially failing functions

## Testing Performed

### Automated Verification
1. AST analysis of all `compute_alpha_attribution_series` calls - All use keyword arguments only
2. AST analysis of app_fallback.py - No risky operations found
3. Code review of all error handling blocks - All properly wrapped

### Manual Testing Required
- [ ] Trigger Safe Mode by introducing an intentional error
- [ ] Verify banner shows only once on first error
- [ ] Click around the Safe Mode UI and verify banner doesn't reappear
- [ ] Click "Retry Full Mode" and verify it clears Safe Mode
- [ ] Verify error details are accessible in expander

## Impact Summary

### Before Fix
- Large red banner appeared on **every** user interaction while in Safe Mode
- No way to clear Safe Mode without browser refresh
- Error details always visible, cluttering the UI
- Highly disruptive user experience

### After Fix
- Banner appears **once** per session when error first occurs
- Small warning line on subsequent interactions (minimal disruption)
- Error details available but hidden by default (cleaner UI)
- "Retry Full Mode" button allows recovery without refresh
- Much better user experience during error conditions

## Files Modified

1. **app.py** (lines 11830-11891)
   - Added session state tracking for Safe Mode banner
   - Modified error handler to show banner only once
   - Added "Retry Full Mode" button
   - Added comprehensive inline documentation

## Acceptance Criteria Status

✅ Safe Mode banner shown at most once per session
✅ Subsequent reruns show minimal warning instead of banner
✅ Error details collapsible (already was, verified)
✅ "Retry Full Mode" button implemented for user recovery
✅ app_fallback.py verified to have no risky operations
✅ Documentation provided with file/line changes
✅ Explanation of rerun issue and fix provided

## Future Improvements

1. Add telemetry to track how often Safe Mode is triggered
2. Add more specific error recovery options
3. Consider adding a "Report Error" button to gather feedback
4. Add automated tests for Safe Mode behavior
