# Single Authoritative Price Source - Implementation Guide

## Overview

This implementation establishes `data/cache/prices_cache.parquet` as the **single source of truth** for all price data in the WAVES Intelligence™ system, eliminating inconsistencies caused by multiple price sources and preventing spurious background downloads.

## Problem Statement

### Before Implementation

The system had several critical issues:

1. **Multiple Price Sources**: Price data scattered across:
   - `data/prices.csv`
   - `data/waves/**/prices.csv` (per-wave files)
   - `data/cache/prices_cache.parquet`
   - Session state caches
   - Global price cache

2. **Implicit Background Fetching**: 
   - Price data automatically fetched on Streamlit reruns
   - Background downloads on startup
   - Unpredictable network activity

3. **Inconsistent Diagnostics**:
   - Different components using different price sources
   - Diagnostics reporting different data than execution logic
   - Extra tickers causing false failures

4. **Excessive Required Tickers**:
   - Universe tickers (e.g., top 200 crypto) treated as required
   - Inactive wave tickers still being fetched
   - SmartSafe cash waves requiring price data unnecessarily

### After Implementation

- ✅ **Single canonical price cache**: `data/cache/prices_cache.parquet`
- ✅ **No implicit fetching**: Downloads only on explicit user action
- ✅ **Consistent diagnostics**: All components use the same data source
- ✅ **Refined required tickers**: Only active wave tickers included
- ✅ **Proper categorization**: Missing vs extra vs failed tickers

## Architecture

### 1. Canonical Price Getter

**Function**: `get_canonical_prices(tickers, start, end)`

**Purpose**: Load price data from the canonical cache ONLY. Never fetches from network.

**Features**:
- Loads exclusively from `data/cache/prices_cache.parquet`
- Never performs downloads under any circumstances
- Returns what is available in cache or NaN for missing tickers
- Provides metadata (date range, last updated, num columns)

**Usage**:
```python
from helpers.price_loader import get_canonical_prices

# Load specific tickers
prices = get_canonical_prices(['AAPL', 'MSFT'])

# Load all cached tickers
all_prices = get_canonical_prices()

# Load with date filtering
prices = get_canonical_prices(
    tickers=['SPY', 'QQQ'],
    start='2024-01-01',
    end='2024-12-31'
)
```

### 2. No Implicit Fetching

**Updated Function**: `load_or_fetch_prices(tickers, start, end, force_fetch=False)`

**Changes**:
- **Default behavior**: Loads ONLY from cache (same as `get_canonical_prices`)
- **Explicit fetch**: Only downloads if `force_fetch=True`
- **No automatic downloads**: On Streamlit reruns, missing tickers, or stale data

**Fetching Rules**:

Price data is ONLY fetched when:

1. ✅ User clicks "💰 Refresh Prices Cache" button
2. ✅ `FORCE_CACHE_REFRESH=1` environment variable is set
3. ✅ `refresh_price_cache()` is called explicitly
4. ✅ `load_or_fetch_prices(..., force_fetch=True)` is called

Price data is NEVER fetched:

1. ❌ On Streamlit page loads or reruns
2. ❌ When loading prices for calculations
3. ❌ In diagnostics or readiness checks
4. ❌ On app startup (removed automatic prefetching)

**Usage**:
```python
from helpers.price_loader import load_or_fetch_prices

# Default: Load from cache only (NO download)
prices = load_or_fetch_prices(['AAPL', 'MSFT'])

# Explicit fetch: Download if missing/stale
prices = load_or_fetch_prices(['AAPL', 'MSFT'], force_fetch=True)
```

### 3. Refined Required Tickers

**Function**: `collect_required_tickers(active_only=True)`

**Purpose**: Collect ONLY the tickers actually needed for active waves.

**Inclusion Criteria**:
- ✅ Tickers in holdings of **active waves only**
- ✅ Benchmark tickers for **active waves only**
- ✅ Essential market indicators (SPY, ^VIX, BTC-USD)

**Exclusion Criteria**:
- ❌ Universe tickers (unless explicitly in an active wave)
- ❌ Optional watchlist tickers
- ❌ Inactive wave tickers
- ❌ SmartSafe cash wave tickers (exempt from price requirements)
- ❌ Unnecessary safe sleeve tickers

**Example**:

Before (149 required tickers):
```
SPY, QQQ, AAPL, MSFT, ..., [plus 100+ crypto universe tickers], 
SGOV, BIL, SHY, IEF, TLT, USDC-USD, USDT-USD, DAI-USD, ...
```

After (3 required tickers for minimal active setup):
```
SPY, ^VIX, BTC-USD
```

**Usage**:
```python
from helpers.price_loader import collect_required_tickers

# Get tickers for active waves only (default)
tickers = collect_required_tickers(active_only=True)

# Get tickers for all waves
tickers = collect_required_tickers(active_only=False)
```

### 4. Enhanced Diagnostics

**Function**: `check_cache_readiness(min_trading_days, max_stale_days, active_only)`

**Purpose**: Provide a single source of truth for system readiness diagnostics.

**Returns**:
```python
{
    'ready': bool,              # Overall readiness status
    'exists': bool,             # Cache file exists
    'num_days': int,            # Number of trading days in cache
    'num_tickers': int,         # Number of tickers in cache
    'max_date': str,            # Most recent date in cache
    'days_stale': int,          # Days since most recent data
    'required_tickers': int,    # Number of tickers required for active waves
    'missing_tickers': list,    # Required tickers NOT in cache (critical)
    'extra_tickers': list,      # Tickers in cache but not required (harmless)
    'failed_tickers': list,     # Tickers that failed to download
    'status': str,              # Human-readable status message
    'status_code': str          # Status code (READY, MISSING, STALE, etc.)
}
```

**Status Codes**:
- `READY`: All criteria met, system is ready
- `MISSING`: Cache file does not exist
- `EMPTY`: Cache file is empty
- `INSUFFICIENT`: Not enough trading days
- `STALE`: Data is too old
- `DEGRADED`: Missing required tickers

**Ticker Categorization**:

1. **Required Tickers**: Tickers needed for active waves
   - Missing these → system is NOT ready
   - Shown in diagnostics as critical

2. **Extra Tickers**: Tickers in cache but not required
   - Having these → harmless, does not affect readiness
   - Shown in diagnostics as informational

3. **Failed Tickers**: Required tickers that failed to download
   - Subset of missing tickers
   - Tracked in `data/cache/failed_tickers.csv`
   - Shown in diagnostics with failure reasons

**Usage**:
```python
from helpers.price_loader import check_cache_readiness

# Check readiness
readiness = check_cache_readiness(active_only=True)

if readiness['ready']:
    print(f"✅ System is ready: {readiness['status']}")
else:
    print(f"❌ System not ready: {readiness['status']}")
    
    if readiness['missing_tickers']:
        print(f"Missing: {readiness['missing_tickers']}")
    
    if readiness['extra_tickers']:
        print(f"Extra (harmless): {len(readiness['extra_tickers'])} tickers")
```

### 5. Cache Metadata

**Function**: `get_cache_info()`

**Purpose**: Get comprehensive metadata about the canonical price cache.

**Returns**:
```python
{
    'exists': bool,           # Cache file exists
    'path': str,              # Canonical cache file path
    'size_mb': float,         # File size in MB
    'num_tickers': int,       # Number of tickers in cache
    'num_days': int,          # Number of trading days in cache
    'date_range': tuple,      # (start_date, end_date) as strings
    'last_updated': str,      # Most recent date in cache
    'tickers': list,          # All ticker symbols in cache
    'is_stale': bool,         # Whether cache is older than MAX_STALE_DAYS
    'days_stale': int         # Days since last update
}
```

**Usage**:
```python
from helpers.price_loader import get_cache_info

info = get_cache_info()

print(f"Cache: {info['path']}")
print(f"Size: {info['size_mb']:.2f} MB")
print(f"Tickers: {info['num_tickers']}")
print(f"Date range: {info['date_range'][0]} to {info['date_range'][1]}")
print(f"Last updated: {info['last_updated']}")
print(f"Stale: {info['is_stale']} ({info['days_stale']} days)")
```

## User Interface

### Refresh Prices Cache Button

**Location**: Sidebar

**Label**: "💰 Refresh Prices Cache"

**Action**: Explicitly fetches latest prices for all required tickers and updates the canonical cache.

**Workflow**:
1. User clicks button
2. System collects required tickers (active waves only)
3. System fetches prices from yfinance
4. System updates `data/cache/prices_cache.parquet`
5. System shows results (success/failure counts)
6. System clears Streamlit caches
7. System triggers rerun to reflect new data

**User Feedback**:
```
✅ Price cache refreshed!
📊 120/123 tickers fetched

⚠️ 3 tickers failed
See data/cache/failed_tickers.csv for details
```

### Environment Variable

**Variable**: `FORCE_CACHE_REFRESH=1`

**Purpose**: Trigger automatic cache refresh on startup (useful for scheduled deployments)

**Usage**:
```bash
# Refresh cache on startup
FORCE_CACHE_REFRESH=1 streamlit run app.py
```

## Migration Guide

### For Developers

If you have code that uses the old price loading pattern:

**Before**:
```python
# Old pattern - implicit fetching
from helpers.price_loader import load_or_fetch_prices

prices = load_or_fetch_prices(['AAPL', 'MSFT'])  # Would download if missing
```

**After**:
```python
# New pattern - explicit caching
from helpers.price_loader import get_canonical_prices

# Load from cache only (recommended for most use cases)
prices = get_canonical_prices(['AAPL', 'MSFT'])

# Or use load_or_fetch_prices (same behavior by default)
prices = load_or_fetch_prices(['AAPL', 'MSFT'])  # NO download by default

# Only use force_fetch for explicit refresh operations
prices = load_or_fetch_prices(['AAPL', 'MSFT'], force_fetch=True)
```

### For Users

**Old Workflow**:
1. Open app → automatic background download (unpredictable)
2. Missing data → automatic download (slow)
3. Stale data → automatic refresh (unexpected network activity)

**New Workflow**:
1. Open app → loads from cache (fast, predictable)
2. Missing data → NaN values, user sees warning
3. Click "💰 Refresh Prices Cache" → explicit download (controlled)

## Benefits

### 1. Performance

- ✅ **Fast startup**: No automatic downloads on app load
- ✅ **Predictable**: No spurious background fetches
- ✅ **Efficient**: Single cache file read vs multiple CSV reads

### 2. Reliability

- ✅ **Single source of truth**: No data inconsistencies
- ✅ **Deterministic**: Same data across all components
- ✅ **Fail-safe**: Missing tickers return NaN vs crash

### 3. User Experience

- ✅ **Explicit control**: User decides when to refresh
- ✅ **Clear feedback**: Shows exactly what succeeded/failed
- ✅ **Informative diagnostics**: Distinguishes missing vs extra vs failed

### 4. Maintainability

- ✅ **Centralized logic**: Single module for all price loading
- ✅ **Consistent behavior**: All components use same functions
- ✅ **Easy debugging**: One place to track price data flow

## Testing

### Automated Test Suite

Run the comprehensive test suite:

```bash
python test_canonical_price_source.py
```

**Tests Include**:
- ✅ Canonical getter loads only from cache
- ✅ No implicit fetching occurs
- ✅ Required tickers properly refined
- ✅ Diagnostics align with execution
- ✅ Cache metadata is accurate

### Manual Verification

1. **Verify no implicit fetching**:
   - Start app
   - Observe no network activity on startup
   - Check logs for "Implicit fetching disabled"

2. **Verify explicit refresh works**:
   - Click "💰 Refresh Prices Cache"
   - Observe download progress
   - See success/failure counts
   - Verify cache file updated

3. **Verify diagnostics accuracy**:
   - Go to "System Health" tab
   - Check "Price Source Stamp"
   - Verify numbers match cache file
   - Check missing vs extra tickers

## Troubleshooting

### Cache is MISSING

**Symptom**: Diagnostics shows "MISSING - Cache file does not exist"

**Solution**:
1. Click "💰 Refresh Prices Cache" button
2. Or run: `python build_price_cache.py`
3. Or set: `FORCE_CACHE_REFRESH=1` and restart app

### Cache is STALE

**Symptom**: Diagnostics shows "STALE - Data is X days old"

**Solution**:
1. Click "💰 Refresh Prices Cache" button
2. System will fetch latest prices
3. Cache will be updated with fresh data

### Cache is DEGRADED

**Symptom**: Diagnostics shows "DEGRADED - Missing X required tickers"

**Solution**:
1. Check which tickers are missing
2. Click "💰 Refresh Prices Cache" button
3. If tickers still fail, check `data/cache/failed_tickers.csv` for reasons
4. Verify ticker symbols are valid and not delisted

### Too Many Extra Tickers

**Symptom**: Diagnostics shows many extra (harmless) tickers

**Solution**:
- This is normal and harmless
- Extra tickers do not affect system performance
- They are simply tickers that were once required but no longer are
- To clean up: Delete cache and rebuild with current active waves only

## Configuration

### Tunable Parameters

Located in `helpers/price_loader.py`:

```python
# Cache configuration
DEFAULT_CACHE_YEARS = 5           # Keep last 5 years of data
MAX_FORWARD_FILL_DAYS = 3         # Maximum gap to forward-fill
MIN_REQUIRED_DAYS = 60            # Minimum trading days for readiness
MAX_STALE_DAYS = 5                # Data older than this is stale

# Download configuration
BATCH_SIZE = 50                   # Maximum tickers per batch
RETRY_ATTEMPTS = 1                # Number of retry attempts
REQUEST_TIMEOUT = 15              # Timeout for yfinance requests
```

### Environment Variables

```bash
# Force cache refresh on startup
export FORCE_CACHE_REFRESH=1
```

## Files and Locations

### Primary Files

- **Canonical Cache**: `data/cache/prices_cache.parquet`
  - Single source of truth for all price data
  - Parquet format for efficient storage and loading

- **Failed Tickers Log**: `data/cache/failed_tickers.csv`
  - Records all ticker download failures
  - Includes timestamps and error reasons
  - Used for diagnostics and debugging

### Code Files

- **Price Loader Module**: `helpers/price_loader.py`
  - Main price loading and caching logic
  - Canonical getter functions
  - Diagnostics and metadata functions

- **Test Suite**: `test_canonical_price_source.py`
  - Comprehensive automated tests
  - Validates all key behaviors

### Legacy Files (Deprecated)

The following files are no longer used and can be safely ignored:

- ❌ `data/prices.csv` - Old global price file
- ❌ `data/waves/**/prices.csv` - Per-wave price files
- ❌ `data/cache_prices.parquet` - Old cache file

## Summary

This implementation creates a stable, predictable price data system with:

1. **Single Source of Truth**: `data/cache/prices_cache.parquet`
2. **No Implicit Fetching**: Downloads only on explicit user action
3. **Refined Required Tickers**: Only active wave tickers included
4. **Aligned Diagnostics**: Same data used everywhere
5. **Clear Categorization**: Missing vs extra vs failed tickers

The result is a more reliable, maintainable, and user-friendly system for managing price data in the WAVES Intelligence™ application.
