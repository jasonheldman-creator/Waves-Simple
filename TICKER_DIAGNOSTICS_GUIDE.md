# Ticker Failure Diagnostics Guide

## Overview

This guide explains the ticker failure diagnostics and normalization system implemented to address price retrieval failures that were causing 25 out of 28 waves to be marked as "Unavailable".

## Key Features

### 1. Ticker Normalization

The system automatically normalizes ticker symbols to ensure compatibility with yfinance:

#### Common Stock Symbol Normalization
- `BRK.B` → `BRK-B` (Berkshire Hathaway Class B)
- `BF.B` → `BF-B` (Brown-Forman Class B)
- Any ticker with `.` → Replace with `-`

#### Crypto Ticker Normalization
All major crypto symbols are automatically appended with `-USD` for yfinance compatibility:

- `BTC` → `BTC-USD`
- `ETH` → `ETH-USD`
- `SOL` → `SOL-USD`
- `ADA` → `ADA-USD`
- `DOT` → `DOT-USD`
- And 35+ more crypto tickers...

#### Implementation

The normalization is handled by:
1. **TICKER_ALIASES dictionary** in `waves_engine.py` - Contains 40+ known ticker variants
2. **_normalize_ticker() function** - Applies normalization rules before price fetch

Example usage:
```python
from waves_engine import _normalize_ticker

# Normalize various ticker formats
normalized = _normalize_ticker("BRK.B")  # Returns "BRK-B"
normalized = _normalize_ticker("btc")     # Returns "BTC-USD"
normalized = _normalize_ticker("AAPL")    # Returns "AAPL" (unchanged)
```

### 2. Failure Tracking & Diagnostics

The system tracks all ticker failures with detailed categorization:

#### Failure Types
- **SYMBOL_INVALID**: Invalid or delisted ticker
- **SYMBOL_NEEDS_NORMALIZATION**: Formatting issues that need correction
- **RATE_LIMIT**: API rate limit exceeded
- **NETWORK_TIMEOUT**: Network/connection timeout
- **PROVIDER_EMPTY**: Empty response from data provider
- **INSUFFICIENT_HISTORY**: Not enough historical data
- **UNKNOWN_ERROR**: Uncategorized error

#### Diagnostic Tracking
Located in `helpers/ticker_diagnostics.py`:

```python
from helpers.ticker_diagnostics import get_diagnostics_tracker

# Get the global tracker
tracker = get_diagnostics_tracker()

# Get all failures
failures = tracker.get_all_failures()

# Get summary statistics
stats = tracker.get_summary_stats()

# Export to CSV
csv_path = tracker.export_to_csv()
```

### 3. CSV Export

Every time the analytics pipeline runs, a comprehensive failure report is generated:

**File Location**: `reports/failed_tickers_report.csv`

**Columns**:
- `ticker_original`: Original ticker symbol as specified
- `ticker_normalized`: Normalized ticker symbol used for fetching
- `wave_id`: Wave identifier (e.g., "crypto_broad_growth_wave")
- `wave_name`: Human-readable wave name (e.g., "Crypto Broad Growth Wave")
- `source`: Data source used (e.g., "yfinance")
- `failure_type`: Categorized failure type (see Failure Types above)
- `error_message`: Detailed error message
- `first_seen`: Timestamp when failure was first recorded
- `last_seen`: Timestamp when failure was last seen
- `is_fatal`: Boolean indicating if the failure is permanent
- `suggested_fix`: Actionable recommendation to resolve the issue

Example CSV content:
```csv
ticker_original,ticker_normalized,wave_id,wave_name,source,failure_type,error_message,first_seen,last_seen,is_fatal,suggested_fix
BTC-USD,BTC-USD,crypto_broad_growth_wave,Crypto Broad Growth Wave,yfinance,PROVIDER_EMPTY,Empty data returned,2025-12-28T12:10:20,2025-12-28T12:10:21,True,Verify ticker symbol is valid and not delisted. Check if ticker is available on the data provider.
```

### 4. UI Diagnostics Panel

A collapsible diagnostics panel is available in the app's Overview page showing:

#### Summary Section
- Total ticker attempts across all waves
- Total failed ticker attempts
- Overall failure rate

#### Detailed Ticker Table (Top 50)
- Ticker symbol (original and normalized)
- Asset type (Crypto, ETF, Index, Equity)
- Failure reason (categorized)
- Example waves affected (up to 3, with "+N more" if applicable)
- Fatal flag (❌ for fatal, ⚠️ for non-fatal)

#### Wave-Level Analysis Table
- Wave name
- Total tickers in wave
- Number of failed tickers
- Failure rate (percentage)
- Current readiness status
- Coverage percentage

#### Accessing the Panel

1. Navigate to the app's Overview page
2. Expand the **🔍 Ticker Failure Root Cause Analysis** section
3. Review the diagnostics and download the CSV report for detailed analysis

### 5. Refined Readiness Classification

The readiness classification has been updated to be more lenient and better support partial data:

#### Readiness Levels

**Full Readiness**:
- 90%+ ticker coverage (reduced from 95%)
- 365+ days of historical data
- Benchmark and NAV data available
- All analytics available

**Partial Readiness**:
- 70%+ ticker coverage (reduced from 90%)
- 7+ days of historical data (reduced from 30)
- Basic analytics available
- Some advanced features limited

**Operational**:
- 50%+ ticker coverage (reduced from 80%)
- 1+ day of historical data (reduced from 7)
- Current pricing available
- Simple returns calculable

**Unavailable**:
- Below 50% ticker coverage OR
- No price data files exist OR
- Less than 1 day of data

#### Benefits
- Waves with partial data are now usable instead of completely unavailable
- More gradual degradation of features as data quality decreases
- Better visibility into what analytics are available for each wave

### 6. Retry & Fallback Mechanisms

The system includes robust retry and fallback logic:

#### Batch Download with Fallback
1. Attempt batch download of all tickers for a wave
2. If batch fails, fall back to individual ticker fetching
3. Track which tickers failed in batch mode

#### Individual Ticker Retry
1. Retry each ticker up to 3 times with exponential backoff
2. Implement delays between batches to avoid rate limits
3. Categorize and record failures with detailed diagnostics

#### Rate Limit Handling
- Detect rate limit errors (429, "too many requests")
- Implement longer delays (5+ seconds) before retrying
- Track rate-limited tickers separately for future optimization

## Running the Analytics Pipeline

### Standard Mode (Real Data)
```bash
python -c "
from analytics_pipeline import run_daily_analytics_pipeline
result = run_daily_analytics_pipeline(all_waves=True, lookback_days=365)
"
```

### Specific Wave
```bash
python -c "
from analytics_pipeline import run_daily_analytics_pipeline
result = run_daily_analytics_pipeline(
    all_waves=False, 
    wave_ids=['crypto_broad_growth_wave'], 
    lookback_days=30
)
"
```

### Dummy Data Mode (Testing)
```bash
python -c "
from analytics_pipeline import run_daily_analytics_pipeline
result = run_daily_analytics_pipeline(
    all_waves=True, 
    lookback_days=7, 
    use_dummy_data=True
)
"
```

## Adding New Ticker Aliases

To add new ticker normalizations, edit `waves_engine.py`:

```python
# Around line 111
TICKER_ALIASES: Dict[str, str] = {
    # Existing aliases...
    
    # Add your new alias
    "YOUR_TICKER": "NORMALIZED_TICKER",
}
```

Example:
```python
TICKER_ALIASES: Dict[str, str] = {
    # ... existing aliases ...
    
    # Add new special ticker
    "GM.PR.B": "GM-B",  # General Motors Preferred Class B
}
```

## Troubleshooting

### Wave Shows 0% Coverage

**Possible Causes**:
1. No price data files generated yet
2. Network connectivity issues preventing data fetch
3. All tickers failed to fetch

**Solutions**:
1. Run the analytics pipeline for the wave
2. Check `reports/failed_tickers_report.csv` for specific ticker failures
3. Verify ticker symbols are correct in wave definition
4. Check if tickers need normalization aliases

### High Failure Rate

**Check**:
1. Network connectivity
2. yfinance API availability
3. Rate limiting issues (check logs for 429 errors)
4. Ticker symbols validity (delisted tickers, incorrect symbols)

**Actions**:
1. Review CSV report for failure types
2. Add aliases for tickers needing normalization
3. Implement backoff delays if rate-limited
4. Remove or replace delisted tickers

### Diagnostics Panel Not Showing

**Ensure**:
1. You're on the Overview page
2. The expander is expanded
3. Analytics pipeline has been run at least once
4. `helpers/ticker_diagnostics.py` is available

## Technical Architecture

### Data Flow

```
Wave Definition (WAVE_WEIGHTS)
    ↓
Ticker Resolution (resolve_wave_tickers)
    ↓
Ticker Normalization (_normalize_ticker with TICKER_ALIASES)
    ↓
Price Fetch (yfinance batch download)
    ↓ (on failure)
Individual Ticker Fetch (with retry)
    ↓
Diagnostics Tracking (TickerDiagnosticsTracker)
    ↓
CSV Export (failed_tickers_report.csv)
    ↓
UI Display (render_ticker_failure_diagnostics_panel)
```

### Key Files

- `waves_engine.py`: Ticker normalization, TICKER_ALIASES, price fetching
- `analytics_pipeline.py`: Pipeline orchestration, CSV export, readiness thresholds
- `helpers/ticker_diagnostics.py`: Failure tracking, categorization, CSV export
- `app.py`: UI diagnostics panel rendering
- `reports/failed_tickers_report.csv`: Generated failure report

## Best Practices

1. **Always check the CSV report** after running the analytics pipeline
2. **Review failure types** to identify systematic issues (e.g., all crypto failing due to normalization)
3. **Add aliases proactively** for known special tickers before running the pipeline
4. **Monitor rate limits** and adjust batch sizes if needed
5. **Use dummy data mode** for testing without hitting API limits
6. **Keep thresholds balanced** - too strict = unnecessary unavailable status, too lenient = misleading readiness

## Future Enhancements

Potential improvements for consideration:

1. **Alternative data sources**: Add fallback to other providers (Alpha Vantage, Polygon, etc.)
2. **Automatic alias learning**: Detect normalization patterns and suggest aliases
3. **Smart rate limiting**: Dynamic backoff based on provider responses
4. **Ticker validation**: Pre-validate ticker symbols before attempting fetch
5. **Historical failure tracking**: Maintain persistent database of ticker failures over time
6. **Email/Slack notifications**: Alert on high failure rates or new failure types
7. **Auto-retry scheduling**: Automatically retry failed tickers on a schedule

## Support

For issues or questions:
1. Check `reports/failed_tickers_report.csv` for detailed diagnostics
2. Review logs for error messages
3. Consult this guide for troubleshooting steps
4. Examine the ticker diagnostics panel in the UI for visual analysis
