# Manual Testing Checklist

## Pre-Deployment Testing

Before merging this PR, please verify the following manually:

### 1. Wave Selection Updates Page Context ✓

**Test Steps:**
1. Start the application
2. Verify it loads in "Portfolio / All Waves" view by default
3. Select a specific wave from the sidebar dropdown
4. **Verify:** Page updates to show wave-specific context
5. **Verify:** Wave banner shows the selected wave name
6. **Verify:** Only ONE rerun occurs (not multiple)

**Expected Result:**
- Wave selection changes the view from Portfolio to Wave-specific
- Page re-renders with correct wave context
- No infinite loop or multiple reruns

### 2. Page Refresh Maintains Wave Selection ✓

**Test Steps:**
1. Select a specific wave from the sidebar (e.g., "Growth Wave")
2. Wait for page to update
3. Manually refresh the browser (F5 or Ctrl+R)
4. **Verify:** Selected wave is still active after refresh
5. **Verify:** Session state preserved

**Expected Result:**
- Wave selection persists across page refreshes
- No reset to Portfolio view

### 3. No Infinite Rerun Loops ✓

**Test Steps:**
1. Start the application
2. Monitor for continuous "Running..." indicator
3. Change wave selection multiple times
4. **Verify:** No continuous rerun loop
5. **Verify:** App stabilizes after each selection

**Test in Debug Mode:**
1. Enable debug mode (if available)
2. Check run counter in UI
3. **Verify:** Run count doesn't continuously increase
4. **Verify:** No "LOOP DETECTED" error appears

**Expected Result:**
- No infinite reruns
- App remains responsive
- Loop detection works correctly (max 3 runs without user interaction)

### 4. Clear Cache & Restart Button Works ✓

**Test Steps:**
1. Navigate to Diagnostics tab
2. Scroll to "Maintenance Actions" section
3. Find "🗑️ Clear Cache & Restart" button
4. Click the button
5. **Verify:** Success message appears: "✅ All caches cleared. Restarting..."
6. **Verify:** App restarts
7. **Verify:** All caches are cleared (st.cache_data, st.cache_resource, session_state)
8. **Verify:** App returns to default state

**Expected Result:**
- Button clears all caches
- App restarts successfully
- No errors or issues
- State resets to defaults

### 5. Auto-Refresh is OFF by Default ✓

**Test Steps:**
1. Start the application fresh
2. Check auto-refresh status in Diagnostics tab
3. **Verify:** Auto-refresh shows as "🔴 Off"
4. **Verify:** No automatic page refreshes occur
5. **Verify:** Page only reruns on user interaction

**Expected Result:**
- Auto-refresh disabled by default
- No automatic reruns
- User must explicitly enable auto-refresh

### 6. No Unrelated Changes ✓

**Code Review:**
1. Review the PR diff
2. **Verify:** Only 3 files modified (app.py, test, docs)
3. **Verify:** Changes are minimal (~20 lines in app.py)
4. **Verify:** No UI rearrangement
5. **Verify:** No refactoring
6. **Verify:** No changes to unrelated functionality

**Expected Result:**
- Only targeted fixes implemented
- No side effects
- No breaking changes

## Automated Tests

Run automated tests to verify:

```bash
cd /home/runner/work/Waves-Simple/Waves-Simple
python test_rerun_loops.py
```

**Expected Output:**
```
============================================================
Testing for Infinite Rerun Loops
============================================================
✓ Auto-refresh is disabled by default
✓ Limited st.rerun() calls found: 2
✓ Exception handlers checked - no obvious reruns found (heuristic check)
✓ Wave selection initialization is conditional
✓ Clear Cache button has been enhanced
✓ Clear Cache button includes all cache clearing operations
✓ trigger_rerun marks user interaction
============================================================
All tests completed!
============================================================
```

## Security Verification

CodeQL scan results:
- **python**: 0 alerts found ✓

## Sign-Off

Once all manual tests pass and automated tests complete successfully:

- [ ] Wave selection updates page correctly
- [ ] Page refresh maintains wave selection
- [ ] No infinite rerun loops occur
- [ ] Clear Cache & Restart button works as expected
- [ ] Auto-refresh is off by default
- [ ] No unrelated changes made

**Tested by:** _______________  
**Date:** _______________  
**Status:** PASS / FAIL  

## Notes

Add any observations or issues found during testing:

_______________________________________________________________________________

_______________________________________________________________________________

_______________________________________________________________________________
