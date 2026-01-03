# PRICE_BOOK Quick Reference

## Overview

The PRICE_BOOK is the **single source of truth** for all price data in the Waves application. This document explains how to use it correctly.

## Key Principle

**ONE TRUTH, NO FETCHING**

All price data comes from the canonical cache at `data/cache/prices_cache.parquet`. No component should fetch prices directly from yfinance or other sources. All fetching is controlled and explicit.

---

## Using PRICE_BOOK in Your Code

### Load Prices (Cache Only)

```python
from helpers.price_book import get_price_book, get_price_book_meta

# Load all cached prices
prices = get_price_book()

# Load specific tickers
prices = get_price_book(active_tickers=['SPY', 'QQQ', 'NVDA'])

# Load with date filters
prices = get_price_book(
    active_tickers=['SPY', 'QQQ'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Get metadata
meta = get_price_book_meta(prices)
print(f"Date range: {meta['date_min']} to {meta['date_max']}")
print(f"Shape: {meta['rows']} rows × {meta['cols']} columns")
```

**Important:** `get_price_book()` NEVER fetches from network. It only reads from cache.

### Get Active Required Tickers

```python
from helpers.price_book import get_active_required_tickers

# Get list of tickers needed for active waves
required = get_active_required_tickers()
print(f"Active waves need {len(required)} tickers")
```

### Check Missing/Extra Tickers

```python
from helpers.price_book import compute_missing_and_extra_tickers

prices = get_price_book()
analysis = compute_missing_and_extra_tickers(prices)

print(f"Missing: {analysis['missing_count']} tickers")
print(f"Extra: {analysis['extra_count']} tickers")
print(f"Missing list: {analysis['missing_tickers']}")
```

---

## Rebuilding the Cache

### Via UI (Recommended)

1. Go to sidebar
2. Click "💰 Rebuild Price Cache (Active Tickers Only)"
3. If `PRICE_FETCH_ENABLED=false`, you'll see a warning
4. If enabled, rebuild proceeds and shows summary

### Via Code (Advanced)

```python
from helpers.price_book import rebuild_price_cache

# Rebuild cache for active tickers only
result = rebuild_price_cache(active_only=True)

if result['allowed']:
    print(f"Fetched: {result['tickers_fetched']}/{result['tickers_requested']}")
    print(f"Failed: {result['tickers_failed']}")
    print(f"Latest date: {result['date_max']}")
else:
    print(f"Fetch disabled: {result['message']}")
```

---

## Environment Variables

### PRICE_FETCH_ENABLED

Controls whether price fetching is allowed.

**Default:** `false` (safe for production)

**To enable:**
```bash
# In Streamlit Cloud secrets or environment
PRICE_FETCH_ENABLED=true
```

**When to enable:**
- When you need to update the price cache
- Only enable temporarily, then disable again
- Never leave enabled in production (prevents infinite loops)

---

## Reality Panel

The Reality Panel at the top of the app shows the current PRICE_BOOK status:

- **Cache Path** - Location of canonical cache file
- **Shape** - Rows (days) × Columns (tickers)
- **Date Range** - Min and max dates in cache
- **Ticker Coverage** - Required vs cached ticker counts
- **Missing/Extra** - Diagnostic information
- **System Info** - Git commit, UTC time, fetch status

**Key:** This panel shows the ACTUAL data used by execution, not a parallel computation.

---

## Migration Guide

### OLD WAY (Don't Do This)

```python
# ❌ Don't fetch directly
import yfinance as yf
data = yf.download(['SPY', 'QQQ'])

# ❌ Don't use parallel price loading
from some_module import load_prices_separately
prices = load_prices_separately(['SPY'])

# ❌ Don't compute readiness separately
def my_readiness():
    # loads prices independently
    return readiness
```

### NEW WAY (Do This)

```python
# ✅ Use PRICE_BOOK
from helpers.price_book import get_price_book

prices = get_price_book(active_tickers=['SPY', 'QQQ'])

# ✅ Use canonical readiness
from helpers.price_loader import check_cache_readiness

readiness = check_cache_readiness(active_only=True)
if readiness['ready']:
    # proceed with computation using PRICE_BOOK
    pass
```

---

## Troubleshooting

### "PRICE_BOOK is EMPTY"

**Cause:** Cache file doesn't exist or is empty

**Solution:**
1. Set `PRICE_FETCH_ENABLED=true`
2. Click "Rebuild Price Cache" button
3. Wait for rebuild to complete
4. Set `PRICE_FETCH_ENABLED=false` again

### "Missing tickers" warning

**Cause:** Cache doesn't have all required tickers for active waves

**Solution:**
1. Check which tickers are missing (Reality Panel shows list)
2. Rebuild cache with fetching enabled
3. If some tickers still fail, check `data/cache/failed_tickers.csv` for reasons

### "System Health shows STALE"

**Cause:** PRICE_BOOK max date is more than 5 days old

**Solution:**
1. Rebuild cache to get latest prices
2. System Health will update to reflect new max date

### App runs forever after deploy

**Cause:** Implicit fetching or auto-refresh enabled

**Solution:**
1. Verify `PRICE_FETCH_ENABLED=false` (default)
2. Verify Safe Mode is ON (default)
3. Verify auto-refresh is OFF (default)
4. Check logs for any yfinance calls (should be none)

---

## Best Practices

1. **Never fetch implicitly** - All fetching must be explicit via rebuild button
2. **Check readiness first** - Use `check_cache_readiness()` before computations
3. **Use active_only=True** - Only load tickers for active waves
4. **Trust the PRICE_BOOK** - Don't create parallel price loading paths
5. **Monitor Reality Panel** - Keep eye on missing/extra tickers
6. **Disable fetch in prod** - Only enable `PRICE_FETCH_ENABLED` when updating cache

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PRICE_BOOK                            │
│            (Single Source of Truth)                      │
│                                                          │
│         data/cache/prices_cache.parquet                  │
│                                                          │
│  Date Index  │  SPY  │  QQQ  │  NVDA  │  ...           │
│  2024-01-01  │ 475.2 │ 410.5 │  485.1 │  ...           │
│  2024-01-02  │ 476.8 │ 412.3 │  487.3 │  ...           │
│      ...     │  ...  │  ...  │   ...  │  ...           │
└─────────────────────────────────────────────────────────┘
                        ▲
                        │
                        │ get_price_book()
                        │
        ┌───────────────┴────────────────┐
        │                                │
        │                                │
   ┌────▼─────┐                    ┌────▼─────┐
   │ Readiness│                    │  System  │
   │   Check  │                    │  Health  │
   └──────────┘                    └──────────┘
        │                                │
        │                                │
   ┌────▼─────┐                    ┌────▼─────┐
   │ Analytics│                    │Diagnostics│
   │ Pipeline │                    │  Panel   │
   └──────────┘                    └──────────┘

        ALL COMPONENTS READ FROM SAME SOURCE
              (No Parallel Loading)
```

---

## Summary

- **One Truth:** `data/cache/prices_cache.parquet`
- **Load Only:** `get_price_book()` never fetches
- **Rebuild:** Via button only, with `PRICE_FETCH_ENABLED` check
- **Monitor:** Reality Panel shows actual PRICE_BOOK status
- **Safe:** No implicit fetching, no infinite loops

Follow these principles and the app will remain stable and consistent.
