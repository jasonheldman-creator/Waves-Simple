# Live Snapshot Implementation - Quick Start Guide

## Overview

This implementation resolves the "3 waves with data" issue by adding comprehensive live market data fetching to generate snapshots with all 28 waves properly populated.

## What Changed

### Core Functionality

**File:** `analytics_truth.py`

New function: `generate_live_snapshot_csv()`

This is the main function that:
1. Fetches live market data for 120 tickers (86 equity via yfinance, 34 crypto via CoinGecko)
2. Computes weighted returns for each of 28 waves
3. Handles ticker failures gracefully (partial data still works)
4. Writes exactly 28 rows to `data/live_snapshot.csv`

**File:** `minimal_app.py`

New UI feature: "Rebuild Live Snapshot Now" button in sidebar

This button:
1. Calls `generate_live_snapshot_csv()` to fetch fresh data
2. Shows success message with statistics
3. Updates session state to track rebuild time
4. Asks user to click "Refresh Now" to see updated data

## How to Use

### Option 1: Via UI (Streamlit App)

1. Run the Streamlit app:
   ```bash
   streamlit run minimal_app.py
   ```

2. In the sidebar, click "🔨 Rebuild Live Snapshot Now"

3. Wait for the rebuild to complete (takes ~30-60 seconds for all tickers)

4. Success message will show:
   - Timestamp
   - Total Waves: 28
   - Waves with Data: X
   - Waves with NO DATA: Y

5. Click "🔄 Refresh Now" to reload the app with new data

6. Go to "Diagnostics" tab to see detailed wave status

### Option 2: Via Python Script

```python
from analytics_truth import generate_live_snapshot_csv

# Generate snapshot with live market data
df = generate_live_snapshot_csv()

# View results
print(f"Total waves: {len(df)}")
print(f"Waves with OK status: {(df['status'] == 'OK').sum()}")
print(f"Waves with NO DATA: {(df['status'] == 'NO DATA').sum()}")

# Check average coverage
print(f"Average coverage: {df['coverage_pct'].mean():.1f}%")

# View snapshot
print(df[['wave_id', 'Wave', 'status', 'coverage_pct', 'Return_30D']])
```

### Option 3: Auto-Rebuild (Optional)

In the Streamlit app sidebar:

1. Check "Enable Auto-Rebuild (every 5 min)"
2. Snapshot will automatically rebuild every 5 minutes
3. Only works when circuit breaker is closed
4. 300-second minimum cooldown prevents runaway behavior

## Expected Results

### Successful Snapshot Generation

When network access is available, you should see:

```
Total Waves: 28
Waves with Data: 28 (or close to 28)
Average Coverage: 95%+ (most tickers succeed)
```

### Output File

`data/live_snapshot.csv` will contain:

- **Exactly 28 rows** (one per wave)
- **Required columns:**
  - `wave_id` - Slugified identifier
  - `Wave` - Human-readable name
  - `Return_1D`, `Return_30D`, `Return_60D`, `Return_365D` - Returns
  - `status` - "OK" or "NO DATA"
  - `coverage_pct` - % of successful tickers (0-100)
  - `missing_tickers` - Comma-separated failed tickers
  - `tickers_ok` - Count of successful tickers
  - `tickers_total` - Total tickers for wave
  - `asof_utc` - Timestamp

### Example Output

```csv
wave_id,Wave,Return_1D,Return_30D,Return_60D,Return_365D,status,coverage_pct,missing_tickers,tickers_ok,tickers_total,asof_utc
sp500_wave,S&P 500 Wave,0.012,0.045,0.089,0.234,OK,100.0,,1,1,2026-01-02T09:00:00
russell_3000_wave,Russell 3000 Wave,0.010,0.041,0.085,0.225,OK,100.0,,1,1,2026-01-02T09:00:00
...
```

## Troubleshooting

### Issue: All waves show "NO DATA"

**Cause:** Network access blocked or API failures

**Solution:**
- Check network connectivity
- Verify yfinance can reach Yahoo Finance
- Verify CoinGecko API is accessible
- Check for rate limiting

### Issue: Some waves have partial data

**Cause:** Some tickers failed to fetch

**Expected:** This is normal! The system is designed to handle partial failures.

**Result:** Waves with at least 1 successful ticker will show status="OK" with computed returns. The `missing_tickers` field shows which tickers failed.

### Issue: Crypto waves all fail

**Cause:** CoinGecko API rate limiting or network issues

**Solution:**
- Wait a few minutes and try again
- Check CoinGecko API status
- Verify ticker mappings in `CRYPTO_COINGECKO_MAP`

## Testing

Run the test suite:

```bash
python3 test_new_snapshot_functions.py
```

Expected output:
```
================================================================================
TEST SUMMARY
================================================================================
Total tests: 6
Passed: 6
Failed: 0
================================================================================
```

Run the demo (shows structure without network):

```bash
python3 demo_snapshot_generation.py
```

## Key Features

✅ **Exactly 28 rows guaranteed** - Hard assertion ensures correct count
✅ **Live market data** - No mock data, fetches from yfinance and CoinGecko
✅ **Robust error handling** - Partial failures don't break the system
✅ **No infinite loops** - Circuit breaker + 300s cooldown + user-triggered refresh
✅ **Comprehensive tracking** - Know exactly which tickers succeeded/failed
✅ **Zero new dependencies** - Uses only existing packages

## Performance Notes

- **Full rebuild time:** ~30-60 seconds for all 120 tickers
- **Sequential fetching:** Tickers fetched one at a time (prevents rate limits)
- **Recommended frequency:** Every 5-15 minutes in production
- **Network required:** Must have access to Yahoo Finance and CoinGecko

## Security

✅ **No secrets in code** - APIs are public, no auth required
✅ **No SQL injection** - Pure pandas operations
✅ **No XSS vulnerabilities** - Streamlit handles escaping
✅ **No arbitrary code execution** - Fixed ticker list only
✅ **CodeQL scan passed** - Zero security alerts

## Support

For issues or questions:

1. Check the test suite passes: `python3 test_new_snapshot_functions.py`
2. Run the demo: `python3 demo_snapshot_generation.py`
3. Review logs for specific ticker failures
4. Check `missing_tickers` column in output for failed tickers
5. Verify network connectivity to Yahoo Finance and CoinGecko

## Files Modified

1. **analytics_truth.py** - Core snapshot generation functions
2. **minimal_app.py** - UI rebuild button
3. **test_new_snapshot_functions.py** - Test suite (6 tests)
4. **demo_snapshot_generation.py** - Demo script
5. **LIVE_SNAPSHOT_IMPLEMENTATION.md** - Detailed documentation
6. **LIVE_SNAPSHOT_QUICKSTART.md** - This file

All changes are backward compatible. Old code continues to work.
