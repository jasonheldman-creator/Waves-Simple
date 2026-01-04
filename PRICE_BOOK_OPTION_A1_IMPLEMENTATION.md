# PRICE_BOOK_OPTION_A1_IMPLEMENTATION.md

## PRICE_BOOK Freshness Option A1: GitHub Actions Daily Cache Update

**Status:** ✅ Implemented  
**PR:** #352  
**Implementation Date:** 2026-01-04

---

## Overview

This implementation provides automated daily updates to the canonical price cache (`data/cache/prices_cache.parquet`) using GitHub Actions, ensuring fresh price data without requiring manual intervention or runtime fetching.

**Option A1** means:
- **Automated**: GitHub Actions workflow runs on schedule
- **Daily**: Executes after market close (9 PM ET / 2 AM UTC)
- **Canonical Path**: Updates `data/cache/prices_cache.parquet`
- **No Runtime Fetching**: App remains read-only (no network price fetches during user sessions)

---

## Architecture

```
┌─────────────────────────────────────┐
│   GitHub Actions (Scheduled)        │
│   Trigger: 2 AM UTC Tue-Sat         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   build_price_cache.py              │
│   - Fetch latest prices (yfinance)  │
│   - Build parquet file              │
│   - Validate data quality           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   data/cache/prices_cache.parquet   │
│   (Canonical PRICE_BOOK source)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Streamlit App (app.py)            │
│   - Read-only cache access          │
│   - No runtime fetching             │
│   - Display fresh data to users     │
└─────────────────────────────────────┘
```

---

## Workflow Configuration

### File Location
`.github/workflows/update_price_cache.yml`

### Triggers

#### 1. Scheduled Execution
- **Cron Schedule:** `0 2 * * 2-6`
- **Frequency:** Daily
- **Days:** Tuesday through Saturday (covers Mon-Fri market closes)
- **Time:** 2 AM UTC (9 PM ET previous day)
- **Rationale:** 
  - US markets close 4 PM ET Mon-Fri
  - 9 PM ET provides 5-hour buffer for end-of-day data availability
  - 2 AM UTC = 9 PM ET (accounting for timezone difference)

#### 2. Manual Trigger
- **Event:** `workflow_dispatch`
- **Input Parameter:** `days` (optional)
  - Description: Days of historical data to fetch
  - Default: `365`
  - Type: `string`
  - Use Case: Ad-hoc refreshes, backfills, or extended history builds

### Execution Steps

1. **Checkout Repository**
   - Uses: `actions/checkout@v4`
   - Ensures latest code and config files are available

2. **Set Up Python**
   - Uses: `actions/setup-python@v5`
   - Version: `3.11`
   - Matches production runtime environment

3. **Install Dependencies**
   - Command: `pip install -r requirements.txt`
   - Installs: pandas, yfinance, numpy, etc.
   - Ensures all price fetching dependencies are available

4. **Determine Days Parameter**
   - For scheduled runs: defaults to `365` days
   - For manual runs: uses user-provided input or default
   - Converts days to years for script parameter

5. **Run Price Cache Builder**
   - Script: `build_price_cache.py`
   - Arguments: `--force --years <calculated>`
   - Fetches price data from Yahoo Finance
   - Builds consolidated parquet file at `data/cache/prices_cache.parquet`
   - **Failure Mode:** Exits with error code if cache file not created

6. **Extract Cache Statistics**
   - Reads the newly created parquet file
   - Extracts metadata:
     - Last Price Date
     - First Price Date
     - Number of rows (date entries)
     - Number of columns (tickers)
     - File size (MB)
     - Data age (days since last price)
   - **Failure Mode:** Exits if cache file cannot be read or statistics extraction fails

7. **Check for Changes**
   - Uses: `git diff --quiet`
   - Detects if parquet file was modified
   - Prevents unnecessary commits when data unchanged

8. **Commit and Push Changes**
   - **Conditional:** Only runs if changes detected
   - Commits: `data/cache/prices_cache.parquet`
   - Also commits: `data/cache/failed_tickers.csv` (if exists)
   - Commit Message: `Update price cache - <last_date> [auto]`
   - User: `waves-bot`

9. **Workflow Summary**
   - Always runs (even on failure)
   - Outputs structured summary to GitHub Actions UI:
     - Status (Success/Failed)
     - Cache statistics (if successful)
     - Troubleshooting guidance (if failed)

---

## Output File Specification

### Canonical Cache Path
`data/cache/prices_cache.parquet`

### File Format
- **Type:** Apache Parquet (columnar format)
- **Compression:** Snappy (default)
- **Index:** DatetimeIndex (trading dates)
- **Columns:** Ticker symbols (e.g., 'SPY', 'QQQ', 'NVDA')
- **Values:** Adjusted close prices (float64)

### Data Structure
```
DatetimeIndex: ['2025-01-01', '2025-01-02', ..., '2026-01-03']
Columns: ['SPY', 'QQQ', 'NVDA', 'AAPL', ...]
Values: Adjusted close prices
```

### Expected Statistics
- **Rows:** ~252-260 per year (trading days)
- **Columns:** ~100-500 tickers (varies by wave configuration)
- **File Size:** ~1-20 MB (depends on tickers and history)
- **Data Age:** 0-1 days (if workflow runs successfully)

---

## Failure Modes & Handling

### 1. Network Connectivity Issues
**Symptom:** Cannot reach Yahoo Finance API  
**Error Message:** "Network error fetching prices"  
**Impact:** Workflow fails, no commit occurs, cache remains unchanged  
**Resolution:** Retry workflow manually; check GitHub Actions network status

### 2. Invalid Ticker Symbols
**Symptom:** Some tickers fail to fetch  
**Error Message:** Individual ticker failures logged  
**Impact:** Partial success; cache updated with available tickers  
**Output:** `data/cache/failed_tickers.csv` lists failed tickers  
**Resolution:** Review wave configuration for invalid tickers

### 3. Insufficient Data
**Symptom:** Ticker returns < 5 rows  
**Error Message:** "Insufficient data: X rows"  
**Impact:** Ticker excluded from cache  
**Resolution:** Verify ticker is active and has trading history

### 4. Rate Limiting
**Symptom:** Too many requests to Yahoo Finance  
**Error Message:** "429 Too Many Requests" or similar  
**Impact:** Some tickers may fail  
**Resolution:** Workflow implements batching and delays; retry after cooldown

### 5. Cache File Not Created
**Symptom:** build_price_cache.py completes but no parquet file  
**Error Message:** "ERROR: Cache file was not created"  
**Impact:** Workflow exits with error code 1  
**Resolution:** Check build_price_cache.py logs for underlying issue

---

## Validation & Verification

### Automated Validation
Run `validate_pr352_implementation.py` to check:
- ✅ Workflow file exists at `.github/workflows/update_price_cache.yml`
- ✅ Schedule trigger configured: `0 2 * * 2-6`
- ✅ Manual trigger (`workflow_dispatch`) present
- ✅ Cache output path: `data/cache/prices_cache.parquet`

### Manual Verification Steps

1. **Trigger Manual Workflow Run**
   - Navigate to: GitHub → Actions → "Update Price Cache"
   - Click "Run workflow"
   - Accept defaults or specify custom `days`
   - Wait for completion (typically 5-15 minutes)

2. **Check Workflow Summary**
   - Verify status is "Success" (green checkmark)
   - Review summary panel:
     - Last Price Date should be recent (0-1 days old)
     - Tickers count should match expected wave holdings
     - File size should be reasonable (1-20 MB)

3. **Verify Cache File Update**
   - Navigate to: Repository → `data/cache/prices_cache.parquet`
   - Check commit history: should show recent auto-commit
   - Verify commit message: `Update price cache - YYYY-MM-DD [auto]`

4. **Test App Data Freshness**
   - Deploy or run Streamlit app
   - Check "Data Health" or "Price Book" section
   - Verify:
     - Last Price Date is recent
     - Data Age is 0-1 days
     - No "STALE DATA" warnings

---

## Maintenance & Operations

### Monitoring
- **GitHub Actions Tab:** Review workflow runs for failures
- **Email Notifications:** GitHub sends alerts for failed workflows
- **App Health Dashboard:** Monitor data age and freshness in app UI

### Troubleshooting Commands

```bash
# Check cache file statistics
python -c "import pandas as pd; df = pd.read_parquet('data/cache/prices_cache.parquet'); print(f'Rows: {len(df)}, Cols: {len(df.columns)}, Last Date: {df.index.max()}')"

# Manually trigger cache rebuild (local)
python build_price_cache.py --force --years 2

# View failed tickers
cat data/cache/failed_tickers.csv
```

### Configuration Updates

**Change Schedule:**
Edit `.github/workflows/update_price_cache.yml`:
```yaml
schedule:
  - cron: "0 2 * * 2-6"  # Modify this line
```

**Change Default Days:**
Edit default input in `workflow_dispatch`:
```yaml
default: '365'  # Change this value
```

**Change Cache Path:**
Update `CANONICAL_CACHE_PATH` in `helpers/price_book.py` (not recommended)

---

## Constraints & Design Decisions

### No app.py Modifications
- **Constraint:** No changes to app.py navigation, tab initialization, or structure
- **Implementation:** All changes are in workflow, docs, and validation script only
- **Verification:** PR files changed list excludes app.py

### No Runtime Fetching
- **Constraint:** App must not fetch prices during user sessions
- **Implementation:** `PRICE_FETCH_ENABLED=false` in production
- **Benefit:** Prevents timeout issues, rate limiting, and unpredictable behavior

### Canonical Path Enforcement
- **Constraint:** All code must reference `data/cache/prices_cache.parquet`
- **Implementation:** Workflow writes to this exact path
- **Benefit:** Single source of truth for price data

### Safe Commit Strategy
- **Design:** Only commit if file changed (git diff check)
- **Benefit:** Avoids empty commits, keeps git history clean
- **Trade-off:** Requires git to be configured in workflow

---

## Related Documentation

- **PROOF_ARTIFACTS_GUIDE.md** - Screenshot requirements for validation
- **validate_pr352_implementation.py** - Automated validation script
- **helpers/price_book.py** - Canonical price data loader module
- **build_price_cache.py** - Price cache builder script

---

## Future Enhancements (Out of Scope for A1)

- Option A2: Multi-region deployment with regional cache updates
- Option B1: Real-time price streaming integration
- Option C1: Alternative data source failover (e.g., Alpha Vantage, IEX)
- Advanced monitoring: Slack/email notifications on failures
- Cache validation: Data quality checks before commit

---

## Change Log

| Date       | Version | Changes                                      |
|------------|---------|----------------------------------------------|
| 2026-01-04 | 1.0     | Initial implementation (PR #352)             |
