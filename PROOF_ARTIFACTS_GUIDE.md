# Proof Artifacts Guide for PR #352

## Overview

This guide documents the proof artifacts required to validate the implementation of PR #352, which addresses price cache freshness and run counter tracking.

## Required Proof Artifacts

### 1. Continuous Rerun Elimination Proof

**Objective**: Prove that the application does not continuously rerun when Auto-Refresh is OFF.

**Required Screenshots**: Two screenshots taken 60 seconds apart

#### Screenshot 1 (Initial State)
- **Timestamp**: T+0 seconds
- **Must Show**:
  - RUN COUNTER value (e.g., "RUN COUNTER: 42")
  - Current timestamp in format "HH:MM:SS"
  - Auto-Refresh status showing "🔴 OFF"
  - No "running..." indicator in browser tab
  - Mission Control section fully visible

#### Screenshot 2 (60 Seconds Later)
- **Timestamp**: T+60 seconds  
- **Must Show**:
  - RUN COUNTER value (MUST be same as Screenshot 1)
  - Timestamp (MUST be same as Screenshot 1, proving no rerun occurred)
  - Auto-Refresh status showing "🔴 OFF"
  - No "running..." indicator in browser tab
  - Mission Control section fully visible

**Success Criteria**:
- ✅ RUN COUNTER unchanged between screenshots
- ✅ Timestamp unchanged between screenshots
- ✅ Auto-Refresh shows "🔴 OFF" in both screenshots
- ✅ No automatic reruns visible

---

### 2. PRICE_BOOK Freshness Proof (Option A2 - Manual Rebuild)

**Objective**: Prove that manual PRICE_BOOK rebuild updates the cache to fresh data.

**Required Screenshot**: One screenshot after clicking "Rebuild PRICE_BOOK Cache" button

#### Screenshot 3 (After Manual Rebuild)
- **Timestamp**: After rebuild completion
- **Must Show**:
  - Success message: "✅ PRICE_BOOK rebuilt. Latest price date now: [YYYY-MM-DD]"
  - "Last Price Date" metric showing latest trading day (within 0-3 days of current date)
  - "Data Age" metric showing "Today" or "1 day" or "2 days" (not STALE)
  - No STALE warning banner
  - Number of tickers fetched (e.g., "📊 120/120 tickers fetched")

**Success Criteria**:
- ✅ Rebuild succeeded with success message
- ✅ "Last Price Date" reflects most recent trading day
- ✅ "Data Age" shows ~0-3 days (fresh data)
- ✅ STALE warning disappeared (if it was present before)
- ✅ Ticker count shows successful fetch

---

### 3. STALE Data Warning Proof (Option B Fallback)

**Objective**: Prove that STALE data warnings are prominently displayed when data is old.

**Conditions**: Only applicable if data is > 10 days old

#### Screenshot 4 (STALE Data State) - If Applicable
- **Timestamp**: When data is > 10 days old
- **Must Show**:
  - "Data Age" metric displaying "⚠️ [X] days (STALE)"
  - Warning banner: "⚠️ STALE/CACHED DATA WARNING"
  - Clear explanation in warning about manual refresh option
  - Help text on rebuild button mentioning safe_mode compatibility

**Success Criteria**:
- ✅ Data Age shows "⚠️" icon and "(STALE)" label
- ✅ Warning banner is prominent and visible
- ✅ Warning explains manual refresh is available
- ✅ User understands how to resolve the issue

---

## Option A1 Automation Proof (GitHub Actions)

**Objective**: Prove that scheduled GitHub Actions workflow successfully updates the price cache.

### Required Evidence

#### Workflow Run Screenshot
- **Location**: GitHub Actions tab → "Update Price Cache (Option A1 Automation)" workflow
- **Must Show**:
  - Successful workflow run (green checkmark)
  - Workflow run timestamp
  - All steps completed successfully
  - "Update PRICE_BOOK Cache" job summary

#### Commit History Screenshot
- **Location**: GitHub repository commits page
- **Must Show**:
  - Automated commit by "github-actions[bot]"
  - Commit message: "Update price cache - [TIMESTAMP]"
  - Files changed: `data/cache/prices_cache.parquet`, `data/cache/failed_tickers.csv`
  - Commit timestamp matching workflow execution

**Success Criteria**:
- ✅ Workflow runs successfully on schedule
- ✅ Cache files are committed to repository
- ✅ Commit attributed to github-actions[bot]
- ✅ Workflow can be manually triggered via workflow_dispatch

---

## Validation Checklist

### Pre-Deployment Validation

- [ ] All required Python dependencies installed
- [ ] `helpers/price_book.py` contains `force_user_initiated` parameter
- [ ] `auto_refresh_config.py` has `DEFAULT_AUTO_REFRESH_ENABLED = False`
- [ ] `STALE_DAYS_THRESHOLD = 10` constant defined
- [ ] App.py displays RUN COUNTER in Mission Control
- [ ] App.py shows STALE indicator when data > 10 days old

### Manual Testing Validation

- [ ] Screenshot 1: Initial state captured
- [ ] Screenshot 2: 60 seconds later, RUN COUNTER unchanged
- [ ] Screenshot 3: After rebuild, fresh data confirmed
- [ ] Screenshot 4: STALE warning displayed (if applicable)

### Automation Testing Validation

- [ ] GitHub Actions workflow file created: `.github/workflows/update_price_cache.yml`
- [ ] Workflow runs successfully on schedule (cron: '0 6 * * *')
- [ ] Workflow can be manually triggered
- [ ] Cache files committed after successful run
- [ ] Workflow artifacts uploaded for review

---

## Artifact Storage and Submission

### File Naming Convention

Use this naming format for proof artifact files:

```
proof_pr352_rerun_elimination_t0.png          # Screenshot 1
proof_pr352_rerun_elimination_t60.png         # Screenshot 2
proof_pr352_fresh_data_after_rebuild.png      # Screenshot 3
proof_pr352_stale_warning_display.png         # Screenshot 4 (if applicable)
proof_pr352_workflow_run_success.png          # Workflow run
proof_pr352_automated_commit.png              # Automated commit
```

### Storage Location

Store proof artifacts in the repository under:
```
/proof_artifacts/pr352/
```

Or attach to PR #352 as comments with clear labels.

---

## Acceptance Gates

**Do not mark PR #352 as complete until**:

1. ✅ All required screenshots are captured and validated
2. ✅ All success criteria are met
3. ✅ Automated tests pass (`python test_run_counter_feature.py`)
4. ✅ Demo script runs successfully (`python demo_run_counter_feature.py`)
5. ✅ Code review completed
6. ✅ Security scan (CodeQL) passes
7. ✅ GitHub Actions workflow tested (either scheduled or manual trigger)

---

## Troubleshooting

### Issue: RUN COUNTER keeps incrementing

**Solution**: Check that `DEFAULT_AUTO_REFRESH_ENABLED = False` in `auto_refresh_config.py`

### Issue: Rebuild button doesn't work

**Solution**: Verify that `force_user_initiated=True` is passed to `rebuild_price_cache()` call

### Issue: STALE warning doesn't appear

**Solution**: Check that data is actually > 10 days old and `STALE_DAYS_THRESHOLD = 10`

### Issue: Workflow fails to commit

**Solution**: 
- Verify workflow has proper write permissions
- Check git config is set correctly
- Ensure cache files actually changed

---

## References

- PR #352: Implementation of Run Counter and PRICE_BOOK Freshness
- `RUN_COUNTER_IMPLEMENTATION.md`: Detailed implementation guide
- `TESTING_GUIDE.md`: Step-by-step manual testing instructions
- `PRICE_BOOK_OPTION_A1_IMPLEMENTATION.md`: GitHub Actions automation guide
