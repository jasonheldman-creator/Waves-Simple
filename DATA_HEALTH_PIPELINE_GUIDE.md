# Data Health Pipeline Improvements

## Overview

This implementation addresses the "Data: Degraded" issue caused by 135 tickers failing in the readiness pipeline. The solution provides comprehensive diagnostics, error categorization, retry logic, and improved data visibility.

## Key Features

### 1. Failed Ticker Diagnostics (`helpers/ticker_diagnostics.py`)

A new diagnostics module that tracks and categorizes ticker failures with structured reporting:

#### Failure Types
- **SYMBOL_INVALID**: Invalid or delisted tickers
- **SYMBOL_NEEDS_NORMALIZATION**: Formatting issues (e.g., BRK.B → BRK-B)
- **RATE_LIMIT**: API rate limit exceeded
- **NETWORK_TIMEOUT**: Network/connection timeout
- **PROVIDER_EMPTY**: Empty response from data provider
- **INSUFFICIENT_HISTORY**: Not enough historical data
- **UNKNOWN_ERROR**: Uncategorized errors

#### FailedTickerReport Structure
Each failure is tracked with:
- `ticker_original`: Original ticker symbol
- `ticker_normalized`: Normalized ticker symbol
- `wave_id`: Associated wave identifier
- `wave_name`: Wave display name
- `source`: Data source (e.g., yfinance)
- `failure_type`: Categorized failure type
- `error_message`: Detailed error message
- `first_seen`: Timestamp of first failure
- `last_seen`: Timestamp of most recent failure
- `is_fatal`: Whether the error is fatal
- `suggested_fix`: Remediation guidance

### 2. Enhanced Error Categorization

The system automatically categorizes errors based on patterns:
```python
from helpers.ticker_diagnostics import categorize_error

failure_type, suggested_fix = categorize_error(
    error_message="rate limit exceeded",
    ticker="AAPL"
)
# Returns: (FailureType.RATE_LIMIT, "Implement exponential backoff...")
```

### 3. Ticker Symbol Normalization

Built-in normalization for common ticker format issues:
- `BRK.B` → `BRK-B`
- `BF.B` → `BF-B`
- Preserves crypto format: `BTC-USD` → `BTC-USD`

### 4. Retry Logic with Exponential Backoff

Automatic retry for transient errors:
- **Maximum retries**: 3 attempts
- **Initial delay**: 1 second
- **Backoff strategy**: Exponential (1s, 2s, 4s)
- **Retryable errors**: Rate limits, timeouts, network issues

### 5. Batch Processing with Delays

To reduce API stress and avoid rate limits:
- **Delay between tickers**: 0.5 seconds
- Applied after each individual ticker fetch
- Prevents burst requests to the API

### 6. CSV Report Generation

Failed ticker reports can be exported to CSV:
```csv
ticker_original,ticker_normalized,wave_id,wave_name,source,failure_type,error_message,first_seen,last_seen,is_fatal,suggested_fix
BRK.B,BRK-B,sp500_wave,S&P 500 Wave,yfinance,SYMBOL_NEEDS_NORMALIZATION,Empty data returned,2025-12-27T19:47:29.867841,2025-12-27T19:47:29.867843,True,Normalize ticker: BRK.B -> BRK-B
```

Reports are saved to: `./reports/failed_tickers_report_{timestamp}.csv`

### 7. Enhanced Data Health Panel

The Data Health panel now includes:
- **Ticker Diagnostics Section**: Shows failure statistics
  - Total failures
  - Unique tickers
  - Fatal vs non-fatal counts
  - Breakdown by failure type
- **Export Button**: Download failed ticker report as CSV
- **Recent Failures View**: Expandable list of recent failures with details

### 8. Admin Tools Integration

Enhanced "Force Build Data for All Waves" button:
- Clears diagnostics tracker before rebuild
- Runs data fetch with full diagnostics
- Automatically generates report if failures occur
- Shows summary of successes/failures in sidebar

## Usage

### Running the Analytics Pipeline

```python
from analytics_pipeline import fetch_prices
from datetime import datetime, timedelta

# Fetch with diagnostics tracking
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

prices_df, failures = fetch_prices(
    tickers=["AAPL", "MSFT", "BRK.B"],
    start_date=start_date,
    end_date=end_date,
    wave_id="sp500_wave",
    wave_name="S&P 500 Wave"
)
```

### Accessing Diagnostics

```python
from helpers.ticker_diagnostics import get_diagnostics_tracker

# Get the global tracker
tracker = get_diagnostics_tracker()

# Get statistics
stats = tracker.get_summary_stats()
print(f"Total failures: {stats['total_failures']}")
print(f"By type: {stats['by_type']}")

# Get all failures
failures = tracker.get_all_failures()
for failure in failures:
    print(f"{failure.ticker_original}: {failure.failure_type.value}")
    print(f"  Fix: {failure.suggested_fix}")

# Export to CSV
csv_path = tracker.export_to_csv()
print(f"Report saved to: {csv_path}")
```

### UI Integration

1. **View Data Health**:
   - Navigate to sidebar → "📊 Data Health Status"
   - Expand the section to see full diagnostics

2. **Export Failed Ticker Report**:
   - Click "Export Failed Tickers Report" button
   - Download the CSV file

3. **Force Rebuild**:
   - Click "🔨 Force Build Data for All Waves" in sidebar
   - Wait for completion
   - Check diagnostics for any failures

## Testing

### Unit Tests
```bash
python test_ticker_diagnostics.py
```

Tests:
- Error categorization logic
- FailedTickerReport dataclass
- TickerDiagnosticsTracker functionality
- CSV export

### Integration Tests
```bash
python test_analytics_integration.py
```

Tests:
- Ticker normalization
- fetch_prices with diagnostics
- End-to-end integration

## Architecture

### Data Flow

```
fetch_prices()
    ├─> Normalize tickers
    ├─> Try batch download with retry
    │   ├─> Success → Return prices
    │   └─> Failure → Try individual fetches
    │       └─> For each ticker:
    │           ├─> Retry with backoff (max 3 attempts)
    │           ├─> Track failures in diagnostics
    │           └─> Add delay between tickers
    └─> Return (prices_df, failures_dict)

Diagnostics Tracker (singleton)
    ├─> Records all failures
    ├─> Categorizes error types
    ├─> Generates remediation suggestions
    └─> Exports to CSV reports
```

### Key Components

1. **ticker_diagnostics.py**: Core diagnostics module
   - `FailureType` enum
   - `FailedTickerReport` dataclass
   - `TickerDiagnosticsTracker` class
   - `categorize_error()` function

2. **analytics_pipeline.py**: Enhanced with diagnostics
   - `_retry_with_backoff()`: Retry helper
   - `fetch_prices()`: Main fetch with diagnostics
   - `_fetch_prices_individually()`: Individual ticker fetch

3. **data_health_panel.py**: UI for diagnostics
   - Displays ticker failure statistics
   - Shows recent failures
   - Provides CSV export

4. **app.py**: Admin integration
   - Enhanced "Force Build" button
   - Clears tracker before rebuild
   - Auto-generates reports

## Best Practices

### For Developers

1. **Always pass wave_id and wave_name** when calling fetch_prices():
   ```python
   prices, failures = fetch_prices(
       tickers, start_date, end_date,
       wave_id=wave_id,  # Important for diagnostics
       wave_name=wave_name
   )
   ```

2. **Check diagnostics after bulk operations**:
   ```python
   tracker = get_diagnostics_tracker()
   if tracker.get_summary_stats()['total_failures'] > 0:
       report_path = tracker.export_to_csv()
       logger.warning(f"Failures detected. Report: {report_path}")
   ```

3. **Clear tracker before major rebuilds**:
   ```python
   tracker = get_diagnostics_tracker()
   tracker.clear()  # Start fresh
   ```

### For Operators

1. **Regular monitoring**: Check Data Health panel regularly
2. **Review reports**: Download and review CSV reports when failures occur
3. **Act on suggestions**: Follow suggested fixes in the reports
4. **Force rebuild when needed**: Use admin button to refresh all data

## Troubleshooting

### Issue: Rate limit errors
**Solution**: Implemented automatically via retry logic with exponential backoff

### Issue: Symbol normalization (BRK.B fails)
**Solution**: Automatic normalization to BRK-B format

### Issue: Need to see detailed failure info
**Solution**: 
1. Open Data Health panel
2. View "Recent Failures" section
3. Export CSV report for analysis

### Issue: Old failures cluttering diagnostics
**Solution**: Click "Force Build Data for All Waves" - clears tracker and rebuilds

## Performance Considerations

1. **Batch delays**: 0.5s between tickers adds overhead but prevents rate limits
2. **Retry logic**: Up to 3 retries can increase fetch time for failing tickers
3. **Diagnostics tracking**: Minimal overhead (~1ms per failure)
4. **CSV export**: Fast, even with 100+ failures

## Future Enhancements

Potential improvements for future iterations:
1. **Configurable retry parameters**: Allow customization of max retries and delays
2. **Email alerts**: Send alerts when failure threshold exceeded
3. **Historical trending**: Track failure rates over time
4. **Auto-remediation**: Automatically fix common issues (e.g., normalize on first failure)
5. **Integration with monitoring**: Export metrics to monitoring systems

## References

- Problem statement: Data degradation with 135 failing tickers
- Related files:
  - `helpers/ticker_diagnostics.py` - Core diagnostics
  - `analytics_pipeline.py` - Enhanced fetch logic
  - `helpers/data_health_panel.py` - UI integration
  - Tests: `test_ticker_diagnostics.py`, `test_analytics_integration.py`
