# Graceful Degradation & Data Resilience Implementation

## Overview

This document describes the comprehensive fix implemented to ensure all 28 Waves in the Waves-Simple repository are data-ready and render correctly, even when data issues occur.

## Problem Statement

The application previously had several issues that could cause waves to fail silently or crash:
- Single bad ticker could break an entire wave or the application
- Binary "Ready/Not Ready" states hid partially usable waves
- No consolidated view of ticker failures
- Infinite retry loops possible in cache warming
- Missing `live_snapshot.csv` API endpoint
- Incomplete `wave_weights.csv` (8 waves instead of 28)

## Solution Components

### 1. Graceful Ticker Failure Handling

**Files Modified**: `waves_engine.py`, `analytics_pipeline.py`

#### Changes in `waves_engine.py`
- Updated `_download_history()` to return `Tuple[pd.DataFrame, Dict[str, str]]`
  - DataFrame: Price data for successful tickers
  - Dict: Failed tickers mapped to error reasons
- Updated `_download_history_individually()` with same return signature
- Modified `_compute_core()` to handle new signature and continue with partial data

#### Benefits
- Single bad ticker never breaks a wave
- Batch download failures automatically retry tickers individually
- Structured error reporting for diagnostics
- Waves with valid tickers still compute partial NAV/returns

#### Example Usage
```python
from waves_engine import _download_history

# Download prices for multiple tickers
tickers = ["AAPL", "MSFT", "INVALID_TICKER"]
prices_df, failures = _download_history(tickers, days=365)

# prices_df contains data for AAPL and MSFT
# failures = {"INVALID_TICKER": "Empty data returned"}
```

### 2. Graded Readiness Model

**Files Modified**: `analytics_pipeline.py` (already implemented, validated)

#### Readiness Levels
- **Full**: ≥95% coverage, ≥365 days - all analytics available
- **Partial**: ≥90% coverage, ≥30 days - basic analytics available
- **Operational**: ≥80% coverage, ≥7 days - current state display available
- **Unavailable**: Below operational threshold - needs attention

#### Loop Prevention
- Max retries set to 3 in `_retry_with_backoff()`
- LRU caches bounded by `maxsize` parameter
- Empty DataFrames returned on computation errors instead of raising exceptions

#### Example Usage
```python
from analytics_pipeline import compute_data_ready_status

status = compute_data_ready_status("sp500_wave")
print(f"Readiness: {status['readiness_status']}")
print(f"Coverage: {status['coverage_pct']:.1f}%")
print(f"Allowed analytics: {status['allowed_analytics']}")
```

### 3. Live Snapshot API Endpoint

**Files Modified**: `analytics_pipeline.py`

#### Functions Added
- `generate_live_snapshot(output_path)`: Creates comprehensive snapshot CSV
- `load_live_snapshot(path, fallback)`: Loads snapshot with fallback to placeholder

#### Snapshot Contents
For each wave:
- Wave ID and display name
- Readiness status and coverage percentage
- Wave returns (1D, 30D, 60D, 365D)
- Benchmark returns (1D, 30D, 60D, 365D)
- Alpha (1D, 30D, 60D, 365D)
- Exposure and cash percentage
- Data regime tag

#### Example Usage
```python
from analytics_pipeline import generate_live_snapshot, load_live_snapshot

# Generate snapshot
df = generate_live_snapshot("live_snapshot.csv")

# Load with fallback
snapshot = load_live_snapshot("live_snapshot.csv", fallback=True)
print(f"Loaded {len(snapshot)} waves")
```

### 4. Wave Weights Validation

**Files Modified**: `wave_weights.csv`

#### Changes
- Updated from 8 waves to all 28 waves
- Missing waves use actual holdings from `WAVE_WEIGHTS` in `waves_engine.py`
- Validated engine registry count matches CSV count

#### Validation
```python
from waves_engine import WAVE_WEIGHTS, WAVE_ID_REGISTRY
import pandas as pd

assert len(WAVE_WEIGHTS) == 28
assert len(WAVE_ID_REGISTRY) == 28

df = pd.read_csv('wave_weights.csv')
assert df['wave'].nunique() == 28
```

### 5. Diagnostics & UI Improvements

**Files Modified**: `app.py`

#### Broken Tickers Diagnostic Panel
Added new collapsible section to Overview tab showing:
- Total broken tickers across all waves
- Number of waves affected
- Most problematic ticker (failing in most waves)
- Top 20 failing tickers by wave count
- Breakdown by wave with failed ticker counts
- Suggested actions for addressing failures

#### Last Refresh Timestamp
Added timestamp to Wave Readiness Report showing when diagnostics were last computed.

#### Example Output
```
Total Broken Tickers: 15
Waves Affected: 12
Most Problematic Ticker: IONQ (5 waves)

Top Failing Tickers:
  IONQ - 5 waves
  RGTI - 3 waves
  ...
```

### 6. Testing

**Files Created**: `test_graceful_degradation.py`

#### Test Cases (All Passing)
1. `test_download_history_returns_failures()` - Validates tuple return signature
2. `test_live_snapshot_fallback()` - Tests fallback mechanism
3. `test_broken_tickers_report()` - Validates report structure
4. `test_wave_weights_completeness()` - Verifies 28 waves in CSV
5. `test_graded_readiness_levels()` - Tests readiness model
6. `test_compute_core_handles_failures()` - Validates graceful failure handling

#### Running Tests
```bash
python test_graceful_degradation.py
```

## Configuration

### Graded Readiness Thresholds

Defined in `analytics_pipeline.py`:
```python
MIN_DAYS_OPERATIONAL = 7      # Minimum for current state display
MIN_DAYS_PARTIAL = 30         # Minimum for basic analytics
MIN_DAYS_FULL = 365           # Required for full multi-window analytics

MIN_COVERAGE_OPERATIONAL = 0.80  # 80% coverage for operational
MIN_COVERAGE_PARTIAL = 0.90      # 90% coverage for partial
MIN_COVERAGE_FULL = 0.95         # 95% coverage for full
```

### Retry Limits

Defined in `analytics_pipeline.py`:
```python
max_retries = 3               # Maximum retry attempts
initial_delay = 1.0           # Initial delay in seconds
# Exponential backoff: delay *= 2 after each retry
```

## API Reference

### `waves_engine.py`

#### `_download_history(tickers: list[str], days: int) -> Tuple[pd.DataFrame, Dict[str, str]]`
Downloads historical price data with per-ticker isolation.

**Returns**:
- `prices_df`: DataFrame with dates as index and tickers as columns
- `failures`: Dict mapping failed tickers to error reasons

#### `_download_history_individually(tickers: list[str], start, end) -> Tuple[pd.DataFrame, Dict[str, str]]`
Downloads price data one ticker at a time for maximum resilience.

**Returns**:
- `prices_df`: DataFrame with dates as index and tickers as columns
- `failures`: Dict mapping failed tickers to error reasons

### `analytics_pipeline.py`

#### `generate_live_snapshot(output_path: str = "live_snapshot.csv") -> pd.DataFrame`
Generates comprehensive snapshot of all waves with returns, alpha, and diagnostics.

**Args**:
- `output_path`: Path to save the snapshot CSV

**Returns**: DataFrame with snapshot data

#### `load_live_snapshot(path: str = "live_snapshot.csv", fallback: bool = True) -> pd.DataFrame`
Loads live snapshot CSV with fallback to placeholder data.

**Args**:
- `path`: Path to snapshot CSV file
- `fallback`: If True, return placeholder data if file doesn't exist

**Returns**: DataFrame with snapshot data

#### `get_broken_tickers_report() -> Dict[str, Any]`
Gets comprehensive report of all broken/failed tickers across all waves.

**Returns**:
- `total_broken`: Total count of unique broken tickers
- `broken_by_wave`: Dict mapping wave_id to failed tickers
- `ticker_failure_counts`: Dict of ticker to failure count
- `most_common_failures`: List of (ticker, count) tuples
- `total_waves_with_failures`: Number of waves with at least one failure

## Migration Guide

### For Existing Code Using `_download_history()`

**Before**:
```python
prices_df = _download_history(tickers, days=365)
```

**After**:
```python
prices_df, failures = _download_history(tickers, days=365)
# Check failures if needed
if failures:
    print(f"Failed tickers: {failures}")
```

### For Code Checking Wave Readiness

**Before**:
```python
status = compute_data_ready_status(wave_id)
if status['is_ready']:
    # Compute analytics
```

**After**:
```python
status = compute_data_ready_status(wave_id)
readiness = status['readiness_status']

if readiness == 'full':
    # All analytics available
elif readiness in ['partial', 'operational']:
    # Limited analytics based on allowed_analytics dict
else:
    # Show diagnostic info
```

## Monitoring & Observability

### Key Metrics to Monitor

1. **Readiness Distribution**
   - Count of waves in each readiness level
   - Target: >80% usable (operational or better)

2. **Ticker Failure Rate**
   - Total broken tickers vs total tickers
   - Tickers failing in multiple waves (systematic issues)

3. **Data Freshness**
   - Days since last data refresh
   - Target: <5 days (configurable via `MAX_DAYS_STALE`)

4. **Coverage Percentage**
   - Average coverage across all waves
   - Target: >90%

### Logs to Watch

All logs include structured data with wave_id and reason codes:
```
[DATA_READY] wave_id=sp500_wave ready=true reasons=['READY']
[DATA_READY] wave_id=quantum_computing_wave ready=false reasons=['MISSING_PRICES'] missing_tickers=['IONQ', 'RGTI']
```

## Troubleshooting

### Wave Shows as Unavailable

1. Check `wave_weights.csv` has entry for the wave
2. Run `compute_data_ready_status(wave_id)` to see specific issues
3. Check `blocking_issues` and `suggested_actions` in status
4. Verify tickers are valid and not delisted

### Ticker Consistently Fails

1. Check if ticker is delisted or renamed
2. Verify ticker format (e.g., `BRK-B` not `BRK.B`)
3. Check yfinance API status for rate limits
4. Review ticker in Broken Tickers Diagnostic panel

### Snapshot File Missing

The `load_live_snapshot()` function automatically falls back to placeholder data. To generate a fresh snapshot:
```python
from analytics_pipeline import generate_live_snapshot
generate_live_snapshot("live_snapshot.csv")
```

## Future Enhancements

1. **Market Regime Integration**: Populate `data_regime` field with actual market regime (VIX, trend, etc.) instead of using readiness status
2. **Ticker Normalization Database**: Maintain mapping of common ticker aliases and historical names
3. **Automatic Ticker Retirement**: Flag tickers that consistently fail and suggest replacements
4. **Real-time Streaming**: Extend snapshot to support real-time updates via websocket
5. **Alert System**: Notify when waves drop below operational threshold

## Security Considerations

- All user inputs are validated and sanitized
- No secrets or credentials stored in code or CSV files
- CodeQL security scan passed with 0 alerts
- Retry logic includes exponential backoff to prevent rate limit issues
- Bounded caches prevent memory exhaustion

## Performance Considerations

- LRU caches reduce redundant API calls
- Batch downloads tried first, individual only on failure
- Max 3 retries with exponential backoff
- Partial data accepted to avoid blocking
- Empty DataFrames returned instead of exceptions for faster recovery

## Acceptance Criteria Checklist

- ✅ Overview consistently shows all 28 waves
- ✅ Graded readiness statuses displayed (full/partial/operational/unavailable)
- ✅ UI remains operational without blocking
- ✅ No infinite loading spinners (max 3 retries)
- ✅ Diagnostics Summary Panel shows:
  - ✅ Total waves count
  - ✅ Full/partial/operational/unavailable counts
  - ✅ Count and list of failed tickers
  - ✅ Last refresh timestamp
- ✅ wave_weights.csv has all 28 waves
- ✅ Comprehensive test suite (6 tests, all passing)
- ✅ Code review completed and feedback addressed
- ✅ CodeQL security scan passed (0 alerts)
