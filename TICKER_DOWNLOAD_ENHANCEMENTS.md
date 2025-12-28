# Enhanced Ticker Download with Retry Logic and Diagnostics

## Overview

This document describes the enhancements made to the ticker download functions in `waves_engine.py` to address the 135 failed tickers issue and improve overall data reliability.

## Problem Statement

The Waves-Simple repository was experiencing 135 ticker failures during data fetching, leading to:
- Incomplete data for waves
- No visibility into the root causes of failures
- No automated retry mechanism for transient failures
- No handling for common ticker formatting issues (e.g., BRK.B vs BRK-B)
- Limited ability to distinguish permanent failures (delisted tickers) from temporary ones

## Solution

### 1. Enhanced Retry Logic

**Implementation**: `_retry_with_backoff()` function

- **Exponential backoff**: Initial delay of 1 second, doubling with each retry
- **Max retries**: 3 attempts per operation
- **Rate limit detection**: Automatically increases delay for rate limit errors
- **Smart error handling**: Distinguishes between transient and permanent failures

**Benefits**:
- Handles temporary network issues automatically
- Reduces API throttling errors
- Improves overall success rate without manual intervention

### 2. Ticker Normalization

**Implementation**: `_normalize_ticker()` function

- **Automatic conversion**: Replaces dots with hyphens (e.g., BRK.B → BRK-B)
- **Case normalization**: Converts to uppercase
- **Whitespace handling**: Strips leading/trailing spaces

**Benefits**:
- Fixes common ticker formatting issues automatically
- Improves success rate for tickers with special characters
- Aligns with yfinance API expectations

### 3. Comprehensive Diagnostics Tracking

**Implementation**: `_log_diagnostics_to_json()` function + integration with `helpers.ticker_diagnostics`

Features:
- **JSON-based logging**: Structured logs in `logs/diagnostics/failed_tickers_YYYYMMDD.json`
- **Failure categorization**: 7 distinct failure types (RATE_LIMIT, SYMBOL_NEEDS_NORMALIZATION, etc.)
- **Timestamp tracking**: First seen and last seen timestamps for each failure
- **Suggested fixes**: Automatic recommendations for each failure type
- **Wave context**: Tracks which wave each failure occurred in

**Log Structure**:
```json
{
  "timestamp": "2025-12-28T10:30:00.000000",
  "wave_id": "sp500_wave",
  "wave_name": "S&P 500 Wave",
  "failures": [
    {
      "ticker_original": "BRK.B",
      "ticker_normalized": "BRK-B",
      "error_message": "Empty data returned",
      "failure_type": "SYMBOL_NEEDS_NORMALIZATION",
      "suggested_fix": "Try normalizing ticker symbol: replace '.' with '-'",
      "is_fatal": false
    }
  ]
}
```

### 4. Batch Processing with Delays

**Implementation**: Enhanced `_download_history_individually()`

- **Batch size**: 10 tickers per batch
- **Batch delay**: 0.5 seconds between batches
- **Rate limit protection**: Prevents overwhelming the API

**Benefits**:
- Reduces 429 (Too Many Requests) errors
- Maintains good API citizenship
- Improves long-term reliability

## Modified Functions

### `_download_history(tickers, days, wave_id=None, wave_name=None)`

**New Parameters**:
- `wave_id`: Optional wave identifier for diagnostics tracking
- `wave_name`: Optional wave display name for diagnostics tracking

**Enhanced Features**:
- Ticker normalization before batch download
- Retry logic with exponential backoff
- Diagnostics tracking for all failures
- JSON logging of failures
- Graceful degradation to individual ticker fetching

### `_download_history_individually(tickers, start, end, wave_id=None, wave_name=None)`

**New Parameters**:
- `wave_id`: Optional wave identifier for diagnostics tracking
- `wave_name`: Optional wave display name for diagnostics tracking

**Enhanced Features**:
- Retry logic for each ticker (3 attempts with backoff)
- Batch processing with delays to prevent rate limiting
- Comprehensive diagnostics tracking
- Categorization of permanent vs transient failures
- JSON logging of all failures

## Failure Type Classification

The system now categorizes failures into 7 types:

1. **SYMBOL_INVALID**: Ticker is invalid or delisted (permanent)
2. **SYMBOL_NEEDS_NORMALIZATION**: Formatting issues (e.g., BRK.B) (fixable)
3. **RATE_LIMIT**: API rate limit exceeded (transient)
4. **NETWORK_TIMEOUT**: Network/connection timeout (transient)
5. **PROVIDER_EMPTY**: Empty response from provider (possibly delisted)
6. **INSUFFICIENT_HISTORY**: Not enough historical data (temporary for new listings)
7. **UNKNOWN_ERROR**: Uncategorized error (needs investigation)

## Backward Compatibility

All changes are **backward compatible**:
- New parameters (`wave_id`, `wave_name`) are optional with default values of `None`
- Existing callers continue to work without modification
- Return signatures unchanged (still returns `Tuple[pd.DataFrame, Dict[str, str]]`)

## Usage Examples

### Basic Usage (backward compatible)
```python
from waves_engine import _download_history

# Old style still works
prices_df, failures = _download_history(["AAPL", "MSFT"], days=365)
```

### Enhanced Usage with Diagnostics
```python
from waves_engine import _download_history

# New style with diagnostics tracking
prices_df, failures = _download_history(
    tickers=["AAPL", "MSFT", "BRK.B"],
    days=365,
    wave_id="sp500_wave",
    wave_name="S&P 500 Wave"
)

# Failures are automatically logged to JSON file
# BRK.B will be normalized to BRK-B and retried
# Failed tickers tracked in diagnostics system
```

### Viewing Diagnostics Logs
```python
import json
from datetime import datetime

# Load today's diagnostics log
date_str = datetime.now().strftime('%Y%m%d')
log_file = f"logs/diagnostics/failed_tickers_{date_str}.json"

with open(log_file, 'r') as f:
    diagnostics = json.load(f)

# Each entry contains timestamp, wave info, and failures
for entry in diagnostics:
    print(f"Wave: {entry['wave_name']}")
    print(f"Failures: {len(entry['failures'])}")
    for failure in entry['failures']:
        print(f"  - {failure['ticker_original']}: {failure['failure_type']}")
        print(f"    Fix: {failure['suggested_fix']}")
```

## Testing

### Test Suite: `test_enhanced_ticker_download.py`

Four test categories:
1. **Ticker Normalization**: Validates conversion logic
2. **Retry Logic**: Tests exponential backoff behavior
3. **Diagnostics Logging**: Validates JSON log structure and appending
4. **Function Signatures**: Ensures backward compatibility

Run tests:
```bash
python test_enhanced_ticker_download.py
```

Expected output:
```
======================================================================
Enhanced Ticker Download Tests
======================================================================

Testing ticker normalization...
✓ Ticker normalization tests passed

Testing retry logic...
✓ Retry logic tests passed

Testing diagnostics logging...
✓ Diagnostics logging tests passed

Testing _download_history signature...
✓ Signature tests passed

======================================================================
All tests passed! ✓
======================================================================
```

## Integration with Existing Systems

The enhanced download functions integrate seamlessly with:

1. **analytics_pipeline.py**: Already uses similar retry and diagnostics logic
2. **helpers/ticker_diagnostics.py**: Shares the same diagnostics tracking system
3. **helpers/data_health_panel.py**: Can display diagnostics from JSON logs
4. **Graceful degradation tests**: All existing tests continue to pass

## Performance Impact

- **Minimal overhead**: Normalization adds negligible processing time
- **Improved success rate**: Retry logic increases first-time success rate
- **Better API citizenship**: Batch delays reduce rate limiting
- **Diagnostic value**: JSON logging provides actionable insights

## Future Enhancements

Potential improvements for future iterations:

1. **Persistent retry state**: Remember permanently failed tickers to avoid retrying
2. **Adaptive batch sizing**: Adjust batch size based on API response times
3. **Ticker alias mapping**: Maintain a map of known ticker aliases
4. **Dashboard integration**: Real-time diagnostics viewing in the UI
5. **Automated remediation**: Auto-fix common issues without manual intervention

## Summary

This enhancement addresses the 135 failed ticker issue by:
- ✅ Adding retry logic with exponential backoff
- ✅ Normalizing ticker symbols before download
- ✅ Tracking all failures with detailed diagnostics
- ✅ Logging failures to structured JSON files
- ✅ Categorizing failures into actionable types
- ✅ Providing suggested fixes for each failure
- ✅ Maintaining backward compatibility
- ✅ Including comprehensive tests

The system now provides a robust, self-healing approach to data fetching with full visibility into any issues that arise.
