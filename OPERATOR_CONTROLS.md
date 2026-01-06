# Operator Controls Documentation

This document describes the operator controls and diagnostic visibility features implemented in the Waves-Simple app to facilitate fast debugging and reduce reliance on Streamlit reboots.

## A) Proof Banner (Top of App)

Located directly after `st.set_page_config()` in `app.py`, the proof banner displays diagnostic information:

- **FILE**: Basename of the current Python file (e.g., `app.py`)
- **GIT SHA**: Short Git commit hash (best-effort)
  - First attempts to read from `GIT_SHA` or `BUILD_ID` environment variables
  - Falls back to executing `git rev-parse --short HEAD`
  - Displays "SHA unavailable" if both methods fail
  - **Note**: Git SHA may be unavailable in Streamlit Cloud if the `.git` directory is not present or accessible
- **UTC TIMESTAMP**: Current UTC timestamp in `YYYY-MM-DD HH:MM:SS UTC` format
- **RUN COUNTER**: Session-based counter that increments on each app rerun

The banner is designed with robust error handling to **never crash Streamlit Cloud**, even if Git commands fail or environment variables are missing.

## B) Sidebar Operator Controls

Located in the sidebar under the "🛠 Operator Controls" section, these controls are always visible and provide quick access to common debugging operations:

### 1. Clear Cache Button (🗑️)
- **Purpose**: Clears Streamlit's cache to force fresh computations
- **Action**: 
  - Calls `st.cache_data.clear()` to clear data caches
  - Calls `st.cache_resource.clear()` (if available) to clear resource caches
  - Logs the action with a UTC timestamp
- **Use Case**: When you suspect stale cached data is causing issues

### 2. Force Recompute Button (♻️)
- **Purpose**: Deletes specific session state keys to trigger clean computations
- **Action**: 
  - Safely deletes the following session state keys (if present):
    - `portfolio_alpha_ledger`
    - `portfolio_snapshot_debug`
    - `portfolio_exposure_series`
    - `wave_data_cache`
    - `price_book_cache`
    - `compute_lock`
  - Logs the action with a UTC timestamp and count of cleared keys
- **Use Case**: When you want to force recomputation of portfolio metrics without clearing all caches

### 3. Hard Rerun Button (🔄)
- **Purpose**: Immediately triggers a full app rerun
- **Action**: 
  - Logs the action with a UTC timestamp
  - Calls `st.rerun()` to force an immediate app rerun
- **Use Case**: When you need to restart the app execution flow

### Last Operator Action Feedback
After pressing any button, a caption line displays:
```
Last operator action: [Button Name] at [UTC Timestamp]
```

This provides immediate feedback confirming the action was executed.

## C) PRICE_BOOK Truth Panel

Located in the Overview tab, this panel displays comprehensive information about the canonical price data source:

### Data Shape
- **Shape**: Displays rows × columns (e.g., "1200 × 150")
- **Index Min Date**: Earliest date in the price index
- **Latest Price Date**: Most recent date in the price index
- **Data Age**: Days since the latest price date, with color-coded status:
  - 🟢 Green: ≤3 days old
  - 🟡 Yellow: 4-7 days old
  - 🔴 Red: >7 days old

### Ticker Presence
Displays the presence of critical tickers:
- **SPY** (required): S&P 500 benchmark
- **QQQ** (required): Nasdaq-100 benchmark
- **IWM** (required): Russell 2000 benchmark
- **VIX Proxy**: One of `^VIX`, `VIXY`, or `VXX` (displays which one is available, or "none")
- **Safe Asset**: One of `BIL` or `SHY` (displays which one is available, or "none")

Each ticker shows ✅ if present or ❌ if missing.

### Missing Tickers
- Lists the first 10 missing tickers (if any)
- Shows total count of missing tickers
- Example: `Missing tickers (5): AAPL, TSLA, NVDA, ... (2 more)`

### Error Handling
If `price_book` fails to load, the panel displays:
- Clear error message: "❌ Failed to load PRICE_BOOK"
- Reason for failure (e.g., "Failed to load price data from cache")

## D) Proof Label Above Portfolio Snapshot Box

Located directly above the blue "Portfolio Snapshot (All Waves)" box, this caption provides renderer metadata:

```
Renderer: Ledger | Source: compute_portfolio_alpha_ledger | Price max date: YYYY-MM-DD | Rows: N | Cols: M
```

- **Renderer**: Identifies the rendering component ("Ledger")
- **Source**: Names the data computation function (`compute_portfolio_alpha_ledger`)
- **Price max date**: Latest date from the reused `price_book`
- **Rows**: Number of rows in the `price_book`
- **Cols**: Number of columns in the `price_book`

This label helps verify data freshness and source consistency.

## E) VIX/Exposure Status Line

Located below the proof label and above the Portfolio Snapshot box, this line shows:

```
VIX Proxy: [ticker or "none found"] | Exposure Mode: [computed or fallback 1.0] | Exposure min/max (60D): [min] - [max]
```

- **VIX Proxy**: Displays which VIX ticker is being used (`^VIX`, `VIXY`, `VXX`, or "none found")
- **Exposure Mode**:
  - `computed`: Exposure is calculated from VIX overlay
  - `fallback 1.0`: No VIX data available, using full exposure (100%)
- **Exposure min/max (60D)**: If exposure data is available, shows the minimum and maximum exposure values over the last 60 rows (e.g., "0.60 - 1.00")

This line provides transparency into the risk management overlay logic.

## Implementation Notes

### Error Handling Philosophy
All operator controls and diagnostic panels are designed with defensive error handling:
- Try/except blocks wrap all operations
- Graceful fallbacks for missing data
- Clear error messages without stack traces (unless in debug mode)
- **Never crash the app** - always display "N/A" or error message instead

### Logging
All operator button actions are logged using Python's `logging` module with UTC timestamps for audit trails.

### Git SHA Availability in Streamlit Cloud
The Git SHA may display "SHA unavailable" in Streamlit Cloud deployments because:
1. The `.git` directory may not be deployed
2. Git binary may not be available in the runtime environment
3. Environment variables (`GIT_SHA`, `BUILD_ID`) may not be set

This is expected behavior and does not indicate an error. The fallback message ensures the app continues functioning normally.

## Usage Tips

1. **Before investigating an issue**: Press "Clear Cache" to ensure fresh data
2. **After data updates**: Press "Force Recompute" to rebuild metrics
3. **To reset the app state**: Press "Hard Rerun"
4. **To verify data freshness**: Check the PRICE_BOOK Truth Panel for data age and ticker presence
5. **To trace computation sources**: Review the proof label and VIX/Exposure status line

## Related Files

- `app.py`: Main application file containing all implementations
- `helpers/price_book.py`: Canonical price data loader
- `helpers/wave_performance.py`: Portfolio metrics computation (including `compute_portfolio_alpha_ledger`)
