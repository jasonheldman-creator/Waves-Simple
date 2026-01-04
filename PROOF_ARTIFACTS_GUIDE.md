# PROOF_ARTIFACTS_GUIDE.md

## PR #352: PRICE_BOOK Freshness Option A1 Implementation

This guide describes the required proof artifacts to validate the successful implementation of PRICE_BOOK Freshness Option A1.

---

## Required Screenshot Evidence

### 1. Auto-Refresh Stability Check (2 screenshots, 60 seconds apart)

**Purpose:** Confirm that Auto-Refresh is OFF and the app is stable without infinite reruns.

**Requirements:**
- **Screenshot 1 (T=0):**
  - Capture the Streamlit app sidebar showing:
    - Auto-Refresh toggle in OFF position (🔴 OFF)
    - RUN COUNTER value (e.g., "RUN COUNTER: 5")
  - Timestamp visible or note the time
  
- **Screenshot 2 (T=60s):**
  - Capture exactly 60 seconds later
  - Show the same sidebar elements:
    - Auto-Refresh still OFF (🔴 OFF)
    - RUN COUNTER value UNCHANGED (still "RUN COUNTER: 5")
  - Timestamp visible or note the time
  
**Success Criteria:**
- RUN COUNTER value is identical in both screenshots
- Auto-Refresh shows OFF in both screenshots
- Confirms no automatic reruns or infinite loops

---

### 2. Fresh Data Validation (1 screenshot)

**Purpose:** Confirm that the GitHub Action successfully updated the price cache and the app displays fresh data.

**Requirements:**
- Take this screenshot AFTER a successful GitHub Actions workflow run
- Capture the app showing:
  - **Last Price Date:** Should show recent date (e.g., today or previous trading day)
  - **Data Age:** Should show ~0-1 days for a fresh cache
  
**Where to find this information:**
- Look for "Data Health" or "Price Book" section in the app
- May be displayed in sidebar metrics or main dashboard
- Common labels: "Last Price Date", "Cache Date", "Data Age", "Freshness"

**Success Criteria:**
- Last Price Date is recent (within last 1-2 business days)
- Data Age is 0-1 days
- No "STALE DATA" warnings visible

---

### 3. GitHub Actions Workflow Run Evidence (1 screenshot)

**Purpose:** Prove that the automated workflow executed successfully.

**Requirements:**
- Navigate to: GitHub repository → Actions tab → "Update Price Cache" workflow
- Select a recent successful run (green checkmark)
- Capture the workflow summary showing:
  - ✅ All steps completed successfully
  - Workflow summary panel with cache statistics:
    - Last Price Date
    - Dimensions (rows × columns)
    - Tickers Count
    - File Size
    - Data Age

**Success Criteria:**
- Workflow status is "Success" (green)
- Summary shows expected statistics
- Timestamp shows workflow ran at scheduled time or manual trigger

---

## Screenshot Checklist

Before submitting the PR, ensure you have:

- [ ] Screenshot 1: Auto-Refresh OFF + RUN COUNTER at T=0
- [ ] Screenshot 2: Auto-Refresh OFF + Same RUN COUNTER at T=60s
- [ ] Screenshot 3: App showing fresh Last Price Date and Data Age ~0-1 days
- [ ] Screenshot 4: GitHub Actions successful workflow run with summary

---

## File Naming Convention

Use descriptive names for clarity:

```
proof_pr352_autorefresh_off_t0.png
proof_pr352_autorefresh_off_t60.png
proof_pr352_fresh_data_validation.png
proof_pr352_github_workflow_success.png
```

---

## Notes

- Screenshots must be **genuine** from production/staging Streamlit deployment
- Do not use mock data or edited screenshots
- Timestamps should be visible when possible
- GitHub Actions workflow runs must be from actual executions (not test runs)
- If a screenshot cannot be captured (e.g., deployment issues), document the reason in PR comments

---

## Validation Script

Use `validate_pr352_implementation.py` to automatically verify:
- Workflow file exists and is properly configured
- Expected schedule and triggers are present
- Cache output path is correct

This does NOT validate screenshots - those must be manually reviewed.
