# Data Enablement Scripts

Scripts for enabling full data readiness (28/28 waves operational) via multiple paths.

## Scripts

### `enable_full_data.py`

**Main data enablement script** - Fetches historical price data for all tickers.

**Features:**
- Detects environment capabilities (live fetch, API keys)
- Fetches 365 days of data for all 143 tickers
- Writes to canonical `data/prices.csv`
- Generates diagnostic files when offline
- Supports multiple data providers (Yahoo, Polygon, IEX)

**Usage:**

```bash
# Basic usage (auto-detect best path)
python scripts/enable_full_data.py

# With API key
export POLYGON_API_KEY=your_key
python scripts/enable_full_data.py
```

**Output:**
- `data/prices.csv` - Historical price data (if successful)
- `data/missing_tickers.csv` - List of tickers without data
- `data/stale_tickers.csv` - List of outdated tickers
- `data/data_coverage_summary.csv` - Coverage metrics

**Exit Codes:**
- `0` - Success (data fetched)
- `1` - Failure (no data sources available)

---

### `analyze_data_readiness.py`

**Data readiness analysis tool** - Analyzes existing data and generates reports.

**Features:**
- Coverage analysis by ticker and wave
- Stale data detection (> 7 days old)
- Wave operational status (X/28 operational)
- Missing ticker identification

**Usage:**

```bash
python scripts/analyze_data_readiness.py
```

**Output:**

```
============================================================
📊 COVERAGE SUMMARY
============================================================

Total tickers expected:   143
Tickers with data:        63
Coverage:                 44.1%

============================================================
🌊 WAVE-LEVEL READINESS
============================================================

✅ Fully operational waves (1):
  🟢 S&P 500 Wave: 1/1 (100%)

⚠️  Waves with incomplete data (27):
  🔴 Crypto AI Growth Wave: 0/6 (0%)
  ...

============================================================
🔴 STATUS: LIMITED OPERATIONAL
============================================================
```

---

### `bootstrap_wave_history.py`

**Existing script** - Bootstraps wave history from price data.

---

## Data Paths

### Path 1: Live Fetch (No API Key)

**When:** Outbound HTTP to Yahoo Finance is allowed  
**Provider:** Yahoo Finance (yfinance)  
**Cost:** Free  
**Coverage:** Stocks, ETFs, major crypto

```bash
python scripts/enable_full_data.py
```

### Path 2: Alternate Provider (API Key)

**When:** API key is available  
**Providers:** Polygon.io, IEX Cloud, Alpha Vantage  
**Cost:** Varies by provider  
**Coverage:** Comprehensive

```bash
# Polygon.io
export POLYGON_API_KEY=your_key
python scripts/enable_full_data.py

# IEX Cloud (future)
export IEX_TOKEN=your_token
python scripts/enable_full_data.py
```

### Path 3: Offline CSV Refresh

**When:** No live fetch or API keys available  
**Method:** Manual upload  
**Cost:** Free

1. Generate `prices.csv` offline with required format:
   ```
   date,ticker,close
   2024-01-01,AAPL,150.23
   2024-01-02,AAPL,151.45
   ```

2. Upload to `data/prices.csv`

3. Verify with:
   ```bash
   python scripts/analyze_data_readiness.py
   ```

## Environment Detection

The `enable_full_data.py` script automatically detects:

- **LIVE_FETCH_AVAILABLE**: Tests outbound HTTP to `https://query1.finance.yahoo.com`
- **POLYGON_API_KEY**: Polygon.io API key
- **IEX_TOKEN**: IEX Cloud token
- **ALPHAVANTAGE_KEY**: Alpha Vantage API key

It selects the best available path automatically.

## Diagnostic Files

When live fetch is unavailable, diagnostic files are generated:

### `data/missing_tickers.csv`

```csv
ticker,status
AAPL,missing
MSFT,missing
```

### `data/stale_tickers.csv`

```csv
ticker,latest_date,days_old
AAPL,2024-12-15,14
```

### `data/data_coverage_summary.csv`

```csv
metric,value
Total Tickers,143
Tickers with Data,63
Coverage Percentage,44.1%
Total Data Points,31500
Date Range,2024-08-08 to 2025-12-20
```

## Integration with UI

The data readiness panel is integrated into the app:

```python
from helpers.data_health_panel import render_data_readiness_panel

# In your Streamlit app
render_data_readiness_panel()
```

Shows:
- Coverage percentage
- Wave operational status (X/28 operational)
- Missing tickers
- Refresh buttons

## Workflow

### Initial Setup

```bash
# 1. Fetch data
python scripts/enable_full_data.py

# 2. Verify coverage
python scripts/analyze_data_readiness.py

# 3. Run app
streamlit run app.py
```

### Regular Updates

```bash
# Refresh data daily/weekly
python scripts/enable_full_data.py

# Check for stale data
python scripts/analyze_data_readiness.py
```

### Troubleshooting

**No network access:**
```bash
python scripts/enable_full_data.py
# Follow instructions to upload prices.csv manually
```

**API rate limits:**
```bash
# Set delays in provider or use alternate provider
export POLYGON_API_KEY=your_key
python scripts/enable_full_data.py
```

**Missing tickers:**
```bash
# Check diagnostic files
cat data/missing_tickers.csv

# Update universal_universe.csv if needed
```

## Requirements

```
pandas>=2.0.0
yfinance>=0.2.36
requests>=2.31.0
```

Install:
```bash
pip install -r requirements.txt
```

## Support

For issues:
1. Run `python scripts/analyze_data_readiness.py`
2. Check diagnostic files in `data/`
3. Review error messages in console output
4. Verify network connectivity and API keys

## License

Part of WAVES Intelligence™ platform.
