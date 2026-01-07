# Price Cache Validation

This document describes the comprehensive validation system for the price cache pipeline.

## Overview

The price cache validation system ensures:
1. **Trading-day freshness** - Cache is up-to-date with the latest trading day
2. **Required symbols** - All necessary symbols are present (with ALL/ANY group semantics)
3. **Cache integrity** - File exists, is non-empty, and contains valid data
4. **No-change logic** - Proper handling of fresh/stale + changed/unchanged scenarios

## Components

### 1. Cache Validation Module (`helpers/cache_validation.py`)

Core validation functions:

- `fetch_spy_trading_days(calendar_days=10)` - Fetch SPY prices to determine trading days
- `get_cache_max_date(cache_path)` - Get the maximum date from cache
- `validate_trading_day_freshness(cache_path, max_market_feed_gap_days=5)` - Validate cache freshness
- `validate_required_symbols(cache_path)` - Validate required symbols with ALL/ANY semantics
- `validate_cache_integrity(cache_path)` - Validate file exists, size > 0, has symbols
- `validate_no_change_logic(cache_freshness_valid, has_changes)` - Determine commit/success logic

### 2. Required Symbol Groups

The validation uses three symbol groups:

**ALL Group** (all must be present):
- `SPY` - S&P 500 ETF
- `QQQ` - NASDAQ-100 ETF
- `IWM` - Russell 2000 ETF

**VIX ANY Group** (at least one must be present):
- `^VIX` - VIX volatility index
- `VIXY` - ProShares VIX Short-Term Futures ETF
- `VXX` - iPath Series B S&P 500 VIX Short-Term Futures ETN

**T-bill ANY Group** (at least one must be present):
- `BIL` - SPDR Bloomberg 1-3 Month T-Bill ETF
- `SHY` - iShares 1-3 Year Treasury Bond ETF

## Trading-Day Freshness Validation

### Process

1. Fetch SPY prices for the last 10 calendar days
2. Calculate `last_trading_day = max(date_index_of_SPY)`
3. Compute `cache_max_date` from the parquet file
4. Compare dates:
   - **PASS** if `cache_max_date == last_trading_day`
   - **FAIL** if `cache_max_date != last_trading_day`

### Sanity Checks

- If `today - last_trading_day > 5 days`, fail with: "Market data feed likely broken"
- This prevents false positives during extended market closures

### Debug Logging

The validation logs:
- `today` - Current date
- `last_trading_day` - Most recent SPY trading day
- `cache_max_date` - Most recent date in cache
- `delta_days` - Difference between cache_max_date and last_trading_day
- `market_feed_gap_days` - Days between today and last_trading_day

## No-Change Logic

The pipeline handles four scenarios:

| Cache State | Has Changes | Action | Commit? | Succeed? |
|------------|-------------|--------|---------|----------|
| Fresh | No | Skip commit | ❌ | ✅ |
| Fresh | Yes | Commit changes | ✅ | ✅ |
| Stale | No | Fail workflow | ❌ | ❌ |
| Stale | Yes | Commit changes | ✅ | ✅ |

### Messages

- **Fresh + unchanged**: "Fresh but unchanged — no commit needed"
- **Fresh + changed**: "Fresh and changed — committing updates"
- **Stale + unchanged**: "Stale + unchanged"
- **Stale + changed**: "Stale but changed — committing updates"

## Usage

### Command Line Validation

```bash
# Basic validation
python validate_cache.py

# With git change detection
python validate_cache.py --check-git

# Custom cache path
python validate_cache.py --cache-path /path/to/cache.parquet

# Custom market gap threshold
python validate_cache.py --max-market-gap 7
```

### In Python Code

```python
from helpers.cache_validation import (
    validate_trading_day_freshness,
    validate_required_symbols,
    validate_cache_integrity
)

# Validate cache integrity
result = validate_cache_integrity('data/cache/prices_cache.parquet')
if not result['valid']:
    print(f"Error: {result['error']}")

# Validate required symbols
result = validate_required_symbols('data/cache/prices_cache.parquet')
if not result['valid']:
    print(f"Missing symbols: {result['error']}")

# Validate trading-day freshness
result = validate_trading_day_freshness('data/cache/prices_cache.parquet')
if not result['valid']:
    print(f"Cache is stale: {result['error']}")
```

### In GitHub Actions

The workflow automatically runs all validations:

```yaml
- name: Run comprehensive validations
  run: |
    python -c "
    from helpers.cache_validation import (
        validate_trading_day_freshness,
        validate_required_symbols,
        validate_cache_integrity
    )
    
    # Run validations...
    "
```

## Testing

### Unit Tests

Run comprehensive unit tests:

```bash
# All cache validation tests
python test_cache_validation.py

# Price pipeline stabilization tests (includes validation tests)
python test_price_pipeline_stabilization.py
```

### Test Coverage

The tests cover:
- ✅ Trading-day freshness calculation
- ✅ SPY data fetching
- ✅ Cache max date extraction
- ✅ Fresh vs stale cache detection
- ✅ ALL group symbol validation
- ✅ VIX ANY group symbol validation
- ✅ T-bill ANY group symbol validation
- ✅ Cache integrity (file exists, size > 0, has symbols)
- ✅ No-change logic (all 4 scenarios)
- ✅ Git change detection

## Workflow Integration

### Build Price Cache

The `build_price_cache.py` script now includes validation:

```bash
# Build with validation (default)
python build_price_cache.py --force

# Build without validation (for testing)
python build_price_cache.py --force --skip-validation
```

### GitHub Actions Workflow

The `update_price_cache.yml` workflow:

1. Builds the cache
2. Validates cache integrity
3. Validates required symbols
4. Validates trading-day freshness
5. Checks for git changes
6. Applies no-change logic
7. Commits only if necessary

## Error Messages

### Cache Integrity Errors

- `"Cache file does not exist: {path}"`
- `"Cache file is empty (0 bytes)"`
- `"Cache has no symbols"`

### Required Symbol Errors

- `"ALL group validation failed - missing symbols: {missing}"`
- `"VIX ANY group validation failed - none of {symbols} present"`
- `"T-bill ANY group validation failed - none of {symbols} present"`

### Trading-Day Freshness Errors

- `"Failed to fetch SPY trading days"`
- `"Market data feed likely broken: {gap} days since last trading day (>{threshold} days)"`
- `"Failed to read cache max date"`
- `"Cache max date ({cache_date}) does not equal last trading day ({trading_date})"`

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Validation Results

Each validation function returns a detailed dictionary:

```python
result = validate_trading_day_freshness('data/cache/prices_cache.parquet')
print(f"Valid: {result['valid']}")
print(f"Today: {result['today']}")
print(f"Last trading day: {result['last_trading_day']}")
print(f"Cache max date: {result['cache_max_date']}")
print(f"Delta days: {result['delta_days']}")
print(f"Market feed gap: {result['market_feed_gap_days']}")
print(f"Error: {result['error']}")
```

## Best Practices

1. **Always validate after building cache** - Use `build_price_cache.py` which includes validation by default
2. **Run validations before deployment** - Use `validate_cache.py --check-git` to ensure cache is ready
3. **Monitor workflow logs** - Check GitHub Actions logs for validation details
4. **Handle network failures gracefully** - Trading-day validation may fail in restricted environments
5. **Review failed symbols** - Check `data/cache/failed_tickers.csv` for download failures

## Future Enhancements

Potential improvements:
- Support for multiple market indices (not just SPY)
- Configurable symbol groups via environment variables
- Integration with alerting systems for validation failures
- Historical validation reports
- Automated cache repair on validation failure
