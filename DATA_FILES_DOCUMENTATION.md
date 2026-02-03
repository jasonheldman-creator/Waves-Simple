# Streamlit Cloud Deployment Fix - Data File Documentation

## Overview
This document describes the canonical data files required for the WAVES Intelligence Console to run on Streamlit Cloud.

## Required Data Files

### 1. prices.csv
**Location:** `data/prices.csv`

**Purpose:** Historical price data for all securities in the portfolio universe.

**Schema:**
- `date` (string): Date in YYYY-MM-DD format
- `ticker` (string): Security ticker symbol
- `close` (float): Closing price for the security on the given date

**Example:**
```csv
date,ticker,close
2016-01-14,AAPL,22.438663482666016
2016-01-15,AAPL,21.899795532226562
2016-01-19,AAPL,21.79383087158203
```

**File Details:**
- Size: ~9.35 MB
- Records: Thousands of daily price points across multiple securities
- Used by: Price loading, attribution calculations, performance analytics

### 2. master_universe.csv
**Location:** `data/master_universe.csv`

**Purpose:** Authoritative universe definition containing all securities tracked by the system.

**Schema:**
- `Ticker` (string): Security ticker symbol
- `Company` (string): Full company name
- `Weight` (float): Portfolio weight/allocation
- `Sector` (string): Sector classification (e.g., "Equity")
- `MarketValue` (float): Market capitalization in USD
- `Price` (float): Current/reference price

**Example:**
```csv
Ticker,Company,Weight,Sector,MarketValue,Price
NVDA,Nvidia,0.01486210128,Equity,4456134118652,183.38
AAPL,Apple Inc.,0.01388338473,Equity,4417762165237,280.84
MSFT,Microsoft,0.01291882866,Equity,3579387843601,480.74
```

**File Details:**
- Size: ~12.35 KB
- Records: Complete security universe
- Used by: Wave registry, attribution, benchmarking, universe validation

## Integration Points

### In app_min.py
These files are loaded by various data loading functions and referenced throughout the attribution and analytics pipeline:

1. `prices.csv` - Loaded by price loading utilities for:
   - Historical return calculations
   - Performance attribution
   - Benchmark comparisons
   - Wave analytics

2. `master_universe.csv` - Used by:
   - Wave Registry for universe definition
   - Attribution engine for security metadata
   - Validation and governance checks
   - Portfolio snapshot generation

## Deployment Notes

### For Streamlit Cloud:
- Both files must be committed to the repository
- Files are located in the `data/` directory
- No external API calls required for basic data loading
- Files support offline/air-gapped operation

### File Updates:
- `prices.csv` is updated via GitHub Actions workflows
- `master_universe.csv` is maintained as canonical source
- Updates are version-controlled through Git

## Related Files

Additional data files in the `data/` directory that support the console:
- `live_snapshot.csv` - Current portfolio snapshot
- `wave_history.csv` - Historical wave performance
- `wave_weights.csv` - Wave allocation weights
- `alpha_attribution_summary.csv` - Attribution results
- `wave_registry.csv` - Wave metadata

## Validation

To validate data files are present and properly formatted:

```python
import pandas as pd

# Validate prices.csv
prices_df = pd.read_csv('data/prices.csv')
assert set(prices_df.columns) == {'date', 'ticker', 'close'}
print(f"✓ prices.csv: {len(prices_df)} records")

# Validate master_universe.csv  
universe_df = pd.read_csv('data/master_universe.csv')
assert 'Ticker' in universe_df.columns
assert 'Company' in universe_df.columns
print(f"✓ master_universe.csv: {len(universe_df)} securities")
```

## Troubleshooting

### Missing File Errors
If you see errors about missing data files:
1. Verify files exist in `data/` directory
2. Check file permissions (should be readable)
3. Ensure files are committed to Git repository
4. Verify Streamlit Cloud has access to the repository

### Schema Errors
If you see column/schema errors:
1. Verify column names match exactly (case-sensitive)
2. Check for extra whitespace in headers
3. Validate CSV format (UTF-8, comma-separated)
4. Ensure no trailing commas or empty columns

---

**Last Updated:** 2026-02-03
**Status:** Both files present and validated ✓
