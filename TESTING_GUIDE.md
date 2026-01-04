# Quick Testing Guide: Run Counter and PRICE_BOOK Features

## How to Test the Implementation

### Prerequisites
1. Start the Streamlit app: `streamlit run app.py`
2. Access the app in your browser (usually http://localhost:8501)

---

## Test 1: RUN COUNTER and Continuous Rerun Elimination

### Goal
Verify that the RUN COUNTER is visible and does NOT increment automatically when Auto-Refresh is OFF.

### Steps

1. **Load the app** in your browser
2. **Navigate to Mission Control** (should be visible at the top)
3. **Observe the RUN COUNTER banner**:
   ```
   🔄 RUN COUNTER: 0 | 🕐 Timestamp: 14:23:45 | 🔄 Auto-Refresh: 🔴 OFF
   ```

4. **Wait 60 seconds** without clicking anything
5. **Take Screenshot 1** showing:
   - RUN COUNTER value (e.g., 0 or 1)
   - Timestamp
   - Auto-Refresh: 🔴 OFF
   - No "running..." indicator in browser tab

6. **Wait another 60 seconds** without clicking anything
7. **Take Screenshot 2** showing:
   - RUN COUNTER value (should be SAME as Screenshot 1)
   - Timestamp (should be SAME as Screenshot 1, proving no rerun)
   - Auto-Refresh: 🔴 OFF
   - No "running..." indicator in browser tab

### Expected Results
✅ RUN COUNTER does NOT change between screenshots  
✅ Timestamp does NOT update (no automatic reruns)  
✅ Auto-Refresh shows 🔴 OFF  
✅ No continuous "running..." indicator  

### What This Proves
- Auto-refresh is OFF by default
- No automatic background reruns occur
- RUN COUNTER is prominently displayed in production UI

---

## Test 2: PRICE_BOOK Freshness (Manual Rebuild)

### Goal
Verify that the "Rebuild PRICE_BOOK Cache" button works to fetch fresh data and updates metadata.

### Steps

1. **Check current data state** in Mission Control:
   - Note the "Data Age" value (e.g., "15 days" or "⚠️ 15 days (STALE)")
   - Note the "Last Price Date" (e.g., "2025-12-20")

2. **Look for STALE warning** (if data is old):
   ```
   ⚠️ STALE/CACHED DATA WARNING
   
   Data is 15 days old. Network fetching is disabled (safe_mode),
   but you can still manually refresh using the 'Rebuild PRICE_BOOK Cache' button below.
   ```

3. **Click the "🔨 Rebuild PRICE_BOOK Cache" button**

4. **Wait for the rebuild to complete** (may take a few minutes)
   - You should see a spinner: "Rebuilding price cache... This may take a few minutes."

5. **After completion, observe the success message**:
   ```
   ✅ PRICE_BOOK rebuilt. Latest price date now: 2026-01-04
   📊 120/120 tickers fetched
   ```

6. **Take Screenshot 3** showing:
   - Success message with latest date
   - Updated "Data Age" metric (should show "Today" or "0 days" or "1 day")
   - Updated "Last Price Date" (should show latest trading day, e.g., "2026-01-04")
   - No STALE warning (if data is now fresh)

### Expected Results
✅ Rebuild button works even when safe_mode is ON  
✅ "Last Price Date" updates to latest trading day  
✅ "Data Age" updates to ~0-1 days  
✅ STALE warning disappears (if data is fresh)  
✅ Success message shows number of tickers fetched  

### What This Proves
- Manual rebuild works even when `safe_mode_no_fetch=True`
- Safe_mode only blocks IMPLICIT fetches, not EXPLICIT user actions
- Metadata (Last Price Date, Data Age) updates correctly after rebuild

---

## Test 3: STALE Data Warning Display

### Goal
Verify that STALE data warnings are prominently displayed.

### Steps

1. **If data is already fresh** (< 10 days old):
   - This test is complete - STALE warning should NOT appear
   
2. **If data is old** (> 10 days):
   - Verify "Data Age" metric shows: "⚠️ 15 days (STALE)"
   - Verify warning banner appears:
     ```
     ⚠️ STALE/CACHED DATA WARNING
     
     Data is 15 days old. Network fetching is disabled (safe_mode),
     but you can still manually refresh using the 'Rebuild PRICE_BOOK Cache' button below.
     ```

### Expected Results
✅ Data Age metric shows STALE indicator when > 10 days  
✅ Warning banner explains the situation clearly  
✅ Warning message includes instructions for manual refresh  

### What This Proves
- STALE data is clearly labeled
- Users understand data freshness status
- Users know how to refresh data manually

---

## Troubleshooting

### If Auto-Refresh is ON:
1. Open the sidebar
2. Find "Enable Auto-Refresh" checkbox
3. Uncheck it to turn OFF auto-refresh
4. RUN COUNTER should stop incrementing

### If Rebuild Button Fails:
1. Check the error message
2. If it says "Fetching is disabled":
   - This is expected if network access is not available
   - The implementation is correct; network access is required
3. If it says "required functions not available":
   - Check that price_loader.py is present
   - Check that yfinance is installed

### If Data Age Doesn't Update:
1. Verify the rebuild was successful (look for success message)
2. Check that cache file was updated: `ls -la data/cache/prices_cache.parquet`
3. Trigger a page refresh or click another button to update the UI

---

## Summary of Proof Artifacts Needed

1. **Screenshot 1**: Initial state with Auto-Refresh OFF, showing RUN COUNTER and timestamp
2. **Screenshot 2**: Same view 60+ seconds later, showing RUN COUNTER and timestamp unchanged
3. **Screenshot 3**: After manual rebuild, showing updated "Last Price Date" and "Data Age"

These three screenshots prove:
- ✅ No continuous reruns when Auto-Refresh is OFF
- ✅ RUN COUNTER is visible in production UI
- ✅ Manual PRICE_BOOK rebuild works and updates metadata
- ✅ STALE data warnings are displayed

---

## Command Line Testing (Optional)

You can also run the demo script to see the logic in action:

```bash
python demo_run_counter_feature.py
```

This will show:
- RUN COUNTER simulation
- Manual rebuild with force_user_initiated
- STALE data warning logic
- Auto-refresh configuration

---

## Questions?

Refer to `RUN_COUNTER_IMPLEMENTATION.md` for detailed documentation.
