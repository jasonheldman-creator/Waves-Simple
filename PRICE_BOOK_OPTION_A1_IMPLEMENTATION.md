# PRICE_BOOK Option A1 Implementation Guide

## Overview

**Option A1: Scheduled GitHub Actions Automation**

This implementation provides automated price cache updates via GitHub Actions, running on a daily schedule to ensure fresh market data without requiring manual intervention or runtime UI changes.

**Key Benefits**:
- ✅ No runtime code changes to app.py or helpers
- ✅ Automated daily updates at market close
- ✅ No UI/navigation modifications required
- ✅ Standalone, testable, and maintainable
- ✅ Works independently of safe_mode settings

---

## Architecture

### Option A1 vs Option A2

| Feature | Option A1 (This Implementation) | Option A2 (PR #352) |
|---------|----------------------------------|---------------------|
| **Trigger** | Scheduled GitHub Actions | Manual button click |
| **Runtime Changes** | None | Modified app.py + helpers |
| **UI Changes** | None | Added RUN COUNTER, STALE warnings |
| **Dependencies** | GitHub Actions only | Streamlit session state |
| **Safe Mode** | N/A (runs with PRICE_FETCH_ENABLED=true) | Bypasses with force_user_initiated |
| **Automation** | Fully automated | Requires user action |

**Decision**: Option A1 is preferred for production as it:
1. Maintains clean separation of concerns
2. Avoids runtime UI complexity
3. Provides reliable scheduled updates
4. Easier to test and validate

---

## Implementation Components

### 1. GitHub Actions Workflow

**File**: `.github/workflows/update_price_cache.yml`

**Schedule**: Daily at 6:00 AM UTC (after market close, before pre-market)

**Workflow Steps**:
1. **Checkout**: Clone repository with full history
2. **Setup Python**: Install Python 3.11 with pip cache
3. **Install Dependencies**: Install from requirements.txt
4. **Configure Git**: Set up github-actions[bot] as committer
5. **Run Price Update**: Execute price cache rebuild with `PRICE_FETCH_ENABLED=true`
6. **Check Changes**: Detect if cache files were modified
7. **Commit & Push**: Commit updated cache files to main branch
8. **Upload Artifacts**: Archive cache files for review (30-day retention)
9. **Report Summary**: Create workflow summary in GitHub UI

**Triggers**:
- **Scheduled**: Runs daily via cron (`0 6 * * *`)
- **Manual**: Can be triggered via workflow_dispatch with optional parameters

---

### 2. Price Cache Update Logic

**Core Function**: `rebuild_price_cache(active_only=True)`

**Location**: `helpers/price_book.py` (existing function, no modifications required)

**Parameters**:
- `active_only`: If True, fetch only tickers from active waves (default: True)

**Environment Variables** (set in workflow):
```yaml
env:
  PRICE_FETCH_ENABLED: 'true'
  ALLOW_NETWORK_FETCH: 'true'
```

**Note**: The workflow sets `PRICE_FETCH_ENABLED=true` in its environment, so it can call the standard `rebuild_price_cache()` function without any code modifications.

**Return Value** (dict):
```python
{
    'allowed': bool,          # Whether fetch was permitted
    'success': bool,          # Whether fetch succeeded
    'tickers_requested': int, # Number of tickers requested
    'tickers_fetched': int,   # Number successfully fetched
    'date_max': str,          # Latest price date (YYYY-MM-DD)
    'failures': dict,         # Failed tickers with reasons
    'message': str            # Status/error message
}
```

---

### 3. Updated Cache Files

**Files Committed**:
- `data/cache/prices_cache.parquet`: Main price cache (Parquet format)
- `data/cache/failed_tickers.csv`: Log of tickers that failed to fetch

**Commit Message Format**:
```
Update price cache - YYYY-MM-DD HH:MM:SS UTC

Automated update via GitHub Actions (Option A1)
- Schedule: Daily at 6:00 AM UTC
- Triggered by: [schedule|workflow_dispatch]
```

---

## Configuration

### Workflow Schedule

Default schedule: **Daily at 6:00 AM UTC**

```yaml
schedule:
  - cron: '0 6 * * *'
```

**Why 6:00 AM UTC?**
- US markets close at 4:00 PM ET (9:00 PM UTC)
- After-hours trading ends around 8:00 PM ET (1:00 AM UTC)
- 6:00 AM UTC = 1:00 AM ET (after all US market activity)
- Before pre-market opens at 4:00 AM ET (9:00 AM UTC)

**Customization**: Edit cron expression in workflow file to change schedule.

### Manual Trigger Parameters

**Input**: `active_only`
- **Description**: Fetch only active wave tickers
- **Type**: Choice (true/false)
- **Default**: true

**Usage**: Go to Actions tab → "Update Price Cache (Option A1 Automation)" → Run workflow

---

## Validation & Testing

### Pre-Deployment Testing

1. **Validate Workflow Syntax**:
   ```bash
   # Install act (GitHub Actions local runner)
   # https://github.com/nektos/act
   
   # Dry-run workflow locally
   act workflow_dispatch -n
   ```

2. **Test Price Cache Update**:
   ```bash
   python validate_pr352_implementation.py
   ```

3. **Verify Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -c "from helpers.price_book import rebuild_price_cache; print('OK')"
   ```

### Post-Deployment Monitoring

**Check Workflow Runs**:
- Navigate to: GitHub → Actions → "Update Price Cache (Option A1 Automation)"
- Verify: Green checkmark on latest run
- Review: Workflow summary and logs

**Check Automated Commits**:
- Navigate to: GitHub → Commits
- Look for: commits by "github-actions[bot]"
- Verify: Files changed include `data/cache/prices_cache.parquet`

**Download Artifacts**:
- Go to: Workflow run → Artifacts section
- Download: `price-cache-[run_number].zip`
- Inspect: Cache files for completeness

---

## Standalone Validation Script

**File**: `validate_pr352_implementation.py`

**Purpose**: Validate that Option A1 implementation components are correctly configured.

**Usage**:
```bash
python validate_pr352_implementation.py
```

**Checks**:
- ✅ Workflow file exists and is valid YAML
- ✅ Price cache function signature includes `force_user_initiated`
- ✅ Cache directory exists
- ✅ Python dependencies are installed
- ✅ Workflow schedule is correctly configured
- ✅ Git configuration is valid

**Exit Codes**:
- `0`: All checks passed
- `1`: One or more checks failed

---

## Security & Permissions

### GitHub Actions Permissions

**Required Permissions** (set in repository settings):
```yaml
permissions:
  contents: write  # Needed to commit and push cache updates
```

**Workflow GITHUB_TOKEN**: Automatically provided by GitHub Actions with write access when `permissions: contents: write` is set.

### Safe Mode Compatibility

**Option A1 runs independently of safe_mode**:
- Sets `PRICE_FETCH_ENABLED=true` explicitly in workflow
- Runs in isolated GitHub Actions environment
- Does not affect runtime app behavior or safe_mode settings

---

## Troubleshooting

### Issue: Workflow fails to push commits

**Cause**: Insufficient permissions or git config issues

**Solution**:
1. Verify repository settings: Settings → Actions → General → Workflow permissions → "Read and write permissions"
2. Check workflow logs for git errors
3. Ensure git config step completed successfully

### Issue: Price fetch returns 0 tickers

**Cause**: Network issues, API rate limits, or invalid tickers

**Solution**:
1. Check workflow logs for specific errors
2. Review `failed_tickers.csv` for error messages
3. Verify yfinance is working: `python -c "import yfinance; print(yfinance.__version__)"`
4. Check Yahoo Finance API status

### Issue: Workflow doesn't run on schedule

**Cause**: Repository inactivity or GitHub Actions disabled

**Solution**:
1. Verify Actions are enabled: Settings → Actions → "Allow all actions"
2. Manually trigger workflow to test: Actions → Run workflow
3. Check GitHub status page for service issues
4. Note: Scheduled workflows may be delayed if repository is inactive

### Issue: Cache files not committed

**Cause**: No changes detected or git add failed

**Solution**:
1. Check "Check for cache file changes" step output
2. Verify cache files were actually modified
3. Review git status in workflow logs

---

## Monitoring & Alerting

### Workflow Failure Notifications

**GitHub will notify** (via email or GitHub UI) when:
- Workflow run fails
- Workflow is disabled due to errors

**Optional**: Set up custom alerts:
- Use GitHub Actions status badge in README
- Integrate with Slack/Discord via webhook
- Use third-party monitoring (e.g., Better Uptime)

### Data Freshness Monitoring

**Check Data Age**:
```python
from helpers.price_book import get_canonical_price_book

price_book = get_canonical_price_book()
meta = get_price_book_meta(price_book)
print(f"Latest price date: {meta['date_max']}")
print(f"Data age: {meta['age_days']} days")
```

**Alert if**:
- Data age > 3 days (workflow may have failed)
- No commits from github-actions[bot] in last 48 hours

---

## Maintenance

### Updating the Workflow

**To change schedule**:
1. Edit `.github/workflows/update_price_cache.yml`
2. Modify cron expression under `on.schedule`
3. Commit and push changes

**To change Python version**:
1. Edit `python-version` in workflow file
2. Update `runtime.txt` if deploying to Vercel

**To change retry logic or timeout**:
1. Edit workflow `timeout-minutes`
2. Modify price_loader.py retry configuration

### Disabling Automation

**Temporary disable**:
- Go to: Actions → "Update Price Cache (Option A1 Automation)" → "..." → Disable workflow

**Permanent disable**:
- Delete or rename `.github/workflows/update_price_cache.yml`
- Or comment out the `schedule` trigger

---

## Comparison with PR #352 Changes

### What PR #352 Included (Option A2):

**Runtime Code Changes**:
- `app.py`: Added RUN COUNTER display, STALE warnings, updated rebuild button
- `helpers/price_book.py`: Added `force_user_initiated` parameter
- `auto_refresh_config.py`: Configured AUTO_REFRESH defaults

**Documentation Files**:
- `RUN_COUNTER_IMPLEMENTATION.md`
- `TESTING_GUIDE.md`
- `SUMMARY.md`
- `CHANGES_VISUAL_GUIDE.md`

**Test Files**:
- `test_run_counter_feature.py`
- `demo_run_counter_feature.py`

### What Option A1 Provides Instead:

**Automation Infrastructure**:
- `.github/workflows/update_price_cache.yml`: Scheduled updates

**Documentation**:
- `PRICE_BOOK_OPTION_A1_IMPLEMENTATION.md`: This file
- `PROOF_ARTIFACTS_GUIDE.md`: Validation guide
- `README_PR352.md`: Overview
- `PR_352_FINAL_SUMMARY.md`: Summary

**Validation**:
- `validate_pr352_implementation.py`: Standalone validation script

**Key Difference**: Option A1 achieves price freshness without any runtime UI changes.

---

## Migration from Option A2 to Option A1

**If PR #352 was already merged**:

1. **Revert runtime changes** (optional if keeping manual rebuild):
   ```bash
   git revert [commit-sha-of-pr-352]
   ```

2. **Deploy Option A1 workflow**:
   ```bash
   git checkout main
   git pull
   # Workflow file should be present from this PR
   ```

3. **Test manual trigger**:
   - Go to Actions → "Update Price Cache (Option A1 Automation)"
   - Click "Run workflow"
   - Verify successful execution

4. **Wait for scheduled run**:
   - Next run will be at 6:00 AM UTC
   - Check Actions tab for automatic execution

**If keeping both options**:
- Option A1 (automated) ensures data is always fresh
- Option A2 (manual button) allows on-demand updates
- No conflicts between the two approaches

---

## Summary

**Option A1 Implementation Status**: ✅ Complete

**Deployed Components**:
- ✅ GitHub Actions workflow file
- ✅ Implementation documentation
- ✅ Validation script
- ✅ Proof artifacts guide

**Next Steps**:
1. Enable workflow in GitHub Actions
2. Test manual trigger
3. Wait for first scheduled run
4. Monitor workflow execution
5. Verify automated commits

**Support**: For issues or questions, refer to troubleshooting section or open an issue in the repository.

---

**Last Updated**: 2026-01-04  
**Version**: 1.0  
**Status**: Production Ready
