# PRICE_BOOK Cache Update & Verification Pipeline

## Overview

This document describes the end-to-end solution for reliably updating and verifying the WAVES Simple PRICE_BOOK / price cache. The system ensures that the Streamlit app displays accurate, up-to-date cache information immediately after the GitHub Actions workflow completes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions Workflow                      │
│  (.github/workflows/update_price_cache.yml)                     │
│                                                                  │
│  1. Checkout repository                                         │
│  2. Install dependencies                                        │
│  3. Run build_price_cache.py --force                           │
│  4. Validate cache file (exists & non-empty)                   │
│  5. Commit & push (only if changes detected)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    build_price_cache.py                         │
│                                                                  │
│  1. Collect all tickers from waves                             │
│  2. Load existing cache (if exists)                            │
│  3. Fetch missing/stale tickers from yfinance                  │
│  4. Merge new data with cache                                  │
│  5. Save cache: data/cache/prices_cache.parquet               │
│  6. Save metadata: data/cache/prices_cache_meta.json          │
│  7. Exit with code 0 (success) or 1 (failure)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Data Files on Disk                          │
│                                                                  │
│  • data/cache/prices_cache.parquet                             │
│    - Parquet file with price data                              │
│    - Index: DatetimeIndex (trading days)                       │
│    - Columns: Ticker symbols                                   │
│    - Values: Close prices                                      │
│                                                                  │
│  • data/cache/prices_cache_meta.json                           │
│    - generated_at_utc: ISO timestamp                           │
│    - success_rate: 0.0 to 1.0                                  │
│    - min_success_rate: Threshold (default 0.90)                │
│    - tickers_total: Number requested                           │
│    - tickers_successful: Successfully downloaded               │
│    - tickers_failed: Failed downloads                          │
│    - max_price_date: Latest date in cache (YYYY-MM-DD)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit App (app.py)                        │
│                                                                  │
│  Overview Tab → PRICE_BOOK Status Panel                        │
│                                                                  │
│  Displays:                                                      │
│  • Current git commit SHA                                       │
│  • Cache file mtime (age)                                       │
│  • Last price date (with staleness indicator)                  │
│  • Force Reload button                                          │
│  • Detailed status with metadata                                │
│  • Troubleshooting guide                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. GitHub Actions Workflow

**File:** `.github/workflows/update_price_cache.yml`

**Schedule:** Daily at 2:00 AM UTC (via cron: `"0 2 * * *"`)

**Triggers:**
- Scheduled (cron)
- Manual dispatch (workflow_dispatch)

**Key Features:**
- **Permissions:** `contents: write` to allow commits to main branch
- **Persistent credentials:** `persist-credentials: true` for checkout
- **Cache validation:** Ensures cache file exists and is non-empty before committing
- **Conditional commit:** Only commits if cache file has changes
- **Metadata display:** Shows metadata JSON in workflow logs
- **Restricted pushes:** Only pushes to main on schedule/workflow_dispatch events

**Exit Behavior:**
- Job fails if `build_price_cache.py` exits with code 1
- Job fails if cache file doesn't exist or is empty after build

### 2. Cache Builder Script

**File:** `build_price_cache.py`

**Usage:**
```bash
python build_price_cache.py [--force] [--years 5]
```

**Arguments:**
- `--force`: Force rebuild even if cache exists
- `--years`: Number of years of history to keep (default: 5)

**Environment Variables:**
- `MIN_SUCCESS_RATE`: Success rate threshold (default: 0.90, clamped to [0.0, 1.0])

**Key Features:**

1. **Success Rate Threshold:**
   - Parses `MIN_SUCCESS_RATE` from environment (default: 0.90)
   - Clamps to valid range [0.0, 1.0]
   - Falls back to default on invalid values

2. **Strict Exit Codes:**
   - Exit 0: Success rate ≥ MIN_SUCCESS_RATE AND cache file exists
   - Exit 1: Success rate < MIN_SUCCESS_RATE OR cache file missing/empty

3. **Metadata File Generation:**
   - Always writes metadata to `data/cache/prices_cache_meta.json`
   - Includes all diagnostic information
   - Written even on failure

4. **Enhanced Logging:**
   - Total tickers requested
   - Successful/failed ticker counts
   - Success rate vs. threshold
   - Latest date in cache
   - Detailed failure reasons per ticker

**Metadata File Schema:**
```json
{
  "generated_at_utc": "2025-01-05T02:00:00.000000Z",
  "success_rate": 0.96,
  "min_success_rate": 0.90,
  "tickers_total": 100,
  "tickers_successful": 96,
  "tickers_failed": 4,
  "max_price_date": "2025-01-04",
  "cache_file": "data/cache/prices_cache.parquet"
}
```

### 3. Price Loader with Cache Keys

**File:** `helpers/price_loader.py`

**Key Features:**

1. **Unique Cache Keys:**
   - Cache key based on file mtime and size: `price_cache_{mtime}_{size}`
   - Ensures Streamlit reloads cache when file changes
   - Prevents stale data issues

2. **Streamlit Integration:**
   - Uses `@st.cache_data` with unique keys when Streamlit is available
   - Falls back to direct loading without Streamlit

3. **Automatic Invalidation:**
   - Cache automatically invalidates when file is updated
   - No manual cache clearing needed in normal operation

### 4. Streamlit App - PRICE_BOOK Status Panel

**File:** `app.py` (Overview Tab)

**Location:** Between "Wave Registry Validator" and "TruthFrame" sections

**Display Components:**

1. **Header Metrics (4 columns):**
   - **Git Commit:** Current commit SHA (short form)
   - **Cache Modified:** Time since last modification (minutes/hours/days)
   - **Last Price Date:** Latest date in cache with staleness indicator
     - 🟢 Green: ≤3 days old (fresh)
     - 🟡 Yellow: 4-7 days old (acceptable)
     - 🔴 Red: >7 days old (stale)
   - **Force Reload Button:** Clears caches and triggers rerun

2. **Detailed Status (Expander):**
   - Cache file path
   - File size in MB
   - Modification timestamp
   - Full metadata JSON
   - **Mismatch Detection:** Warns if metadata doesn't match actual cache data
   - **Troubleshooting Guide:** Step-by-step instructions if cache is missing

**Force Reload Functionality:**
- Clears `st.cache_data` and `st.cache_resource`
- Removes `global_price_df` from session state
- Triggers `st.rerun()` to refresh the app
- New cache keys ensure fresh data is loaded

## Data Age Calculation

**Consistent Approach:**
1. Load cache data directly (not from metadata)
2. Extract latest date from index: `cache_df.index[-1]`
3. Calculate age: `(datetime.now() - latest_date).days`
4. Display with color coding based on age

**Age Thresholds:**
- **OK (🟢):** ≤3 days
- **Acceptable (🟡):** 4-7 days
- **Stale (🔴):** >7 days

**Mismatch Detection:**
- Compares `metadata["max_price_date"]` with actual cache data
- Warns user if dates don't match
- Suggests regenerating cache

## Safety Measures

### 1. Missing Cache Files

**Problem:** App crashes if cache files don't exist

**Solution:**
- Graceful error handling with try/except blocks
- Display error message instead of crashing
- Show troubleshooting guide with actionable steps

**Troubleshooting Steps:**
1. Run `python build_price_cache.py --force` locally
2. Check that `data/cache/prices_cache.parquet` exists
3. Verify GitHub Actions workflow is running successfully
4. Check workflow logs for errors

### 2. Stale Data

**Problem:** Users see old data even after workflow runs

**Solution:**
- Unique cache keys based on file mtime and size
- Force Reload button to manually clear caches
- Automatic cache invalidation on file changes
- Visual staleness indicators

### 3. Metadata Mismatches

**Problem:** Metadata doesn't reflect actual cache contents

**Solution:**
- Validate metadata against actual cache data
- Display warning if mismatch detected
- Suggest regenerating cache

## Testing

### Unit Tests

**File:** `test_build_price_cache_threshold.py`

**Test Coverage:**
1. **Success Rate Calculation:** Various scenarios (100%, 95%, 94%, etc.)
2. **Threshold Logic:** Above, at, and below threshold
3. **Environment Variable Parsing:** Valid values, clamping, fallback
4. **Exit Code Logic:** Exit 0 vs. 1 based on success rate
5. **Metadata File Generation:** File creation, required fields, validation
6. **Cache Key Integrity:** Unique keys based on file attributes

**Run Tests:**
```bash
python test_build_price_cache_threshold.py
```

**Expected Output:**
```
================================================================================
RUNNING ALL BUILD_PRICE_CACHE THRESHOLD TESTS
================================================================================
...
================================================================================
TEST RESULTS: 6 passed, 0 failed
================================================================================
```

### Manual Testing Checklist

1. **Build Cache Locally:**
   ```bash
   python build_price_cache.py --force
   ```
   - Verify exit code 0 on success
   - Check `data/cache/prices_cache.parquet` exists
   - Check `data/cache/prices_cache_meta.json` exists
   - Verify metadata fields are populated

2. **Test Threshold Failure:**
   ```bash
   MIN_SUCCESS_RATE=0.99 python build_price_cache.py --force
   ```
   - Should exit with code 1 if real success rate < 99%
   - Metadata should still be written

3. **Test Streamlit UI:**
   ```bash
   streamlit run app.py
   ```
   - Navigate to Overview tab
   - Verify PRICE_BOOK Status panel displays correctly
   - Check git commit SHA
   - Check cache mtime
   - Check last price date with color indicator
   - Click "Force Reload" button and verify cache clears
   - Expand "Detailed Cache Status" and verify metadata
   - Verify mismatch detection if metadata is outdated

4. **Test GitHub Actions Workflow:**
   - Trigger workflow manually via GitHub UI
   - Check workflow logs for:
     - Successful cache build
     - Cache validation passing
     - Metadata display in logs
     - Commit and push (if changes detected)

## Troubleshooting

### Problem: Workflow succeeds but app shows stale data

**Diagnosis:**
- Check PRICE_BOOK Status panel in app
- Compare "Cache Modified" time with "Last Price Date"
- Look for metadata mismatch warning

**Solutions:**
1. Click "Force Reload" button in app
2. Check if GitHub Actions actually pushed changes (look at commit history)
3. Verify workflow ran on main branch (not PR)
4. Check workflow logs for "No changes to cache files" message

### Problem: Workflow fails with exit code 1

**Diagnosis:**
- Check workflow logs for success rate
- Look for failed ticker list

**Solutions:**
1. Check if `MIN_SUCCESS_RATE` is too high (default: 0.90)
2. Verify yfinance is working (some tickers may be delisted)
3. Check network connectivity in GitHub Actions
4. Review failed tickers and determine if they should be removed

### Problem: Metadata file missing or outdated

**Diagnosis:**
- Expand "Detailed Cache Status" in app
- Look for "Metadata file not found" warning

**Solutions:**
1. Run `python build_price_cache.py --force` locally
2. Commit and push the generated metadata file
3. Verify workflow includes metadata in commit

### Problem: Cache file is empty or corrupted

**Diagnosis:**
- Check PRICE_BOOK Status panel shows "Cache file does not exist"
- Workflow logs show "ERROR: Cache file is empty"

**Solutions:**
1. Delete existing cache: `rm data/cache/prices_cache.parquet`
2. Rebuild: `python build_price_cache.py --force`
3. Check for disk space issues
4. Verify pandas and pyarrow are installed correctly

## Best Practices

1. **Monitor Success Rate:**
   - Keep `MIN_SUCCESS_RATE` at 0.90 (90%)
   - Review failed tickers regularly
   - Remove delisted tickers from wave configurations

2. **Regular Updates:**
   - Let scheduled workflow run daily (2 AM UTC)
   - Trigger manually after adding new tickers
   - Force rebuild after significant changes

3. **Version Control:**
   - Always commit both parquet and metadata files together
   - Review changes before pushing (check file sizes)
   - Don't manually edit parquet files

4. **Cache Management:**
   - Use Force Reload button if data seems stale
   - Don't modify cache files directly
   - Keep cache files in gitignore if size becomes too large

5. **Debugging:**
   - Check PRICE_BOOK Status panel first
   - Review workflow logs for errors
   - Use metadata file for diagnostics
   - Run unit tests after code changes

## Summary

This end-to-end solution provides:

✅ **Reliable Updates:** GitHub Actions workflow with validation  
✅ **Robust Building:** Threshold-based success/failure with metadata  
✅ **Fresh Data:** Unique cache keys prevent stale data issues  
✅ **Clear Status:** Visual indicators and detailed diagnostics  
✅ **Easy Debugging:** Comprehensive logging and troubleshooting  
✅ **Comprehensive Testing:** Unit tests and manual test procedures  

Users should **no longer encounter stale or inconsistent cache data** when using the Streamlit app.
