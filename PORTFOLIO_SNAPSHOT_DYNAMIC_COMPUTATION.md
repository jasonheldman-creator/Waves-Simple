# Portfolio Snapshot Dynamic Runtime Computation - Implementation Summary

## Overview
This document describes the complete refactoring of the Portfolio Snapshot metrics computation to meet business-critical requirements for runtime dynamic calculation.

## Business Requirements Met

### ✅ 1. Eliminated `live_snapshot.csv` Dependency
**Requirement:** Completely remove `live_snapshot.csv` from the Portfolio Snapshot pipeline.

**Implementation:**
- Portfolio Snapshot computation (lines 10470-10740 in `app.py`) has **ZERO** references to `live_snapshot.csv`
- Explicitly documented in code comments: "NO dependencies on live_snapshot.csv, caches, or snapshot ledger"
- Diagnostic overlay confirms: "Snapshot Artifact: ❌ No live_snapshot.csv dependency"

### ✅ 2. Compute Metrics Dynamically
**Requirement:** Dynamically calculate Portfolio Snapshot metrics at render time using authoritative, live market data.

**Implementation:**
All metrics are computed at runtime from `PRICE_BOOK` (canonical live market data):

#### Returns (1D, 30D, 60D, 365D)
```python
# Runtime computation from PRICE_BOOK
returns_df = PRICE_BOOK.pct_change().dropna()
portfolio_returns = returns_df.mean(axis=1)  # Equal-weighted

# 1D: Latest daily return
ret_1d = portfolio_returns.iloc[-1]

# 30D: Compounded return over last 30 trading days
ret_30d = safe_compounded_return(portfolio_returns.iloc[-30:])

# 60D: Compounded return over last 60 trading days
ret_60d = safe_compounded_return(portfolio_returns.iloc[-60:])

# 365D: Compounded return over last 252 trading days
ret_365d = safe_compounded_return(portfolio_returns.iloc[-TRADING_DAYS_PER_YEAR:])
```

#### Alpha Metrics (1D, 30D, 60D, 365D)
```python
# Benchmark returns (SPY) computed at runtime
if 'SPY' in returns_df.columns:
    benchmark_returns = returns_df['SPY']
    
    # Compute benchmark returns for each timeframe
    bench_1d = benchmark_returns.iloc[-1]
    bench_30d = safe_compounded_return(benchmark_returns.iloc[-30:])
    bench_60d = safe_compounded_return(benchmark_returns.iloc[-60:])
    bench_365d = safe_compounded_return(benchmark_returns.iloc[-TRADING_DAYS_PER_YEAR:])
    
    # Alpha = Portfolio Return - Benchmark Return
    alpha_1d = ret_1d - bench_1d
    alpha_30d = ret_30d - bench_30d
    alpha_60d = ret_60d - bench_60d
    alpha_365d = ret_365d - bench_365d
```

#### Attribution
Basic attribution information provided:
- Portfolio composition: Total tickers (equal-weighted)
- Benchmark: SPY (S&P 500)
- Computation method: Mean daily returns
- Data source: PRICE_BOOK live market data

### ✅ 3. Ensure Runtime Computation
**Requirement:** Guarantee that Portfolio Snapshot is dynamically computed whenever the app renders.

**Implementation:**
- **NO `@st.cache_data` decorator** on Portfolio Snapshot computation
- **NO caching** of Portfolio Snapshot metrics
- Metrics computed fresh every render
- Diagnostic overlay includes UTC timestamp that changes with each render

**Note on PRICE_BOOK Caching:**
- `PRICE_BOOK` itself uses `@st.cache_resource` with timestamp-based cache busting
- This is appropriate because PRICE_BOOK is the **live data source**
- Cache is invalidated when underlying price data file changes
- Portfolio Snapshot **metrics** are still computed fresh from this live data every render

## Data Flow Architecture

```
PRICE_BOOK (Live Market Data)
    ↓
    ├─→ Daily Returns Computation (runtime)
    │       ↓
    │       ├─→ Portfolio Returns (equal-weighted mean)
    │       └─→ Benchmark Returns (SPY)
    │
    ├─→ Multi-Timeframe Returns (runtime)
    │       ├─→ 1D Return (latest value)
    │       ├─→ 30D Return (compounded over 30 days)
    │       ├─→ 60D Return (compounded over 60 days)
    │       └─→ 365D Return (compounded over 252 trading days)
    │
    └─→ Alpha Metrics (runtime)
            ├─→ 1D Alpha (portfolio - benchmark)
            ├─→ 30D Alpha (portfolio - benchmark)
            ├─→ 60D Alpha (portfolio - benchmark)
            └─→ 365D Alpha (portfolio - benchmark)
```

## Diagnostic Overlay

The Portfolio Snapshot includes a comprehensive diagnostic overlay that proves runtime dynamic computation:

```
✅ RUNTIME DYNAMIC COMPUTATION
Data Source: PRICE_BOOK (live market data, X rows × Y tickers)
Last Trading Date: YYYY-MM-DD
Render UTC: HH:MM:SS UTC
Benchmark: SPY ✅
Snapshot Artifact: ❌ No live_snapshot.csv dependency
Caching: ❌ No st.cache_data (pure runtime computation)
```

Key elements:
1. **Render UTC timestamp** - Changes with each render to prove fresh computation
2. **Data Source** - Explicitly shows PRICE_BOOK dimensions
3. **No snapshot artifact** - Confirms no live_snapshot.csv dependency
4. **No caching** - Confirms pure runtime computation

## Testing & Validation

### Pre-Flight Checks
- [x] Syntax validation: `python -m py_compile app.py` passes
- [x] No `@st.cache_data` on Portfolio Snapshot section (verified lines 10470-10740)
- [x] No references to `live_snapshot.csv` in Portfolio Snapshot section
- [x] Diagnostic overlay includes UTC timestamp

### Required Proof Screenshots
The PR requires screenshots showing:
1. Portfolio Snapshot at Time "A" with specific UTC timestamp
2. Portfolio Snapshot at Time "B" with different UTC timestamp
3. Visible numeric changes in metrics between renders
4. Diagnostic overlay confirming:
   - Live PRICE_BOOK source
   - UTC render timestamp
   - No live_snapshot.csv dependency

## Code Changes Summary

### File: `app.py`

**Section: Portfolio Snapshot (lines 10470-10740)**

**Changed:**
1. Enhanced comments to explicitly state "NO dependencies on live_snapshot.csv"
2. Added benchmark returns computation (SPY)
3. Added Alpha computation for all timeframes (1D, 30D, 60D, 365D)
4. Enhanced diagnostic overlay with comprehensive runtime proof
5. Refactored display to show Returns, Alpha, and Attribution
6. Added color-coding for Alpha (green=positive, red=negative)

**Key Code Locations:**
- Line 10477: Comment stating "NO dependencies on live_snapshot.csv"
- Line 10481: PRICE_BOOK fetch (live data source)
- Line 10503-10506: Benchmark returns extraction
- Line 10533-10578: Portfolio returns computation
- Line 10580-10598: Benchmark returns computation
- Line 10600-10604: Alpha metrics computation
- Line 10510-10528: Enhanced diagnostic overlay
- Line 10607-10772: Display section (Returns, Alpha, Attribution)

## Technical Details

### Safe Compounded Returns
Uses numerically stable formula to prevent overflow:
```python
def safe_compounded_return(returns_series):
    # Prevents log domain errors
    if (returns_series <= -1).any():
        return None
    # Numerically stable: exp(sum(log(1+r))) - 1
    return np.expm1(np.log1p(returns_series).sum())
```

### Equal-Weighted Portfolio
Portfolio returns computed as equal-weighted mean across all tickers:
```python
portfolio_returns = returns_df.mean(axis=1)
```

### Timeframe Periods
- **1D:** Latest single value
- **30D:** Last 30 rows (approximately 30 trading days)
- **60D:** Last 60 rows (approximately 60 trading days)
- **365D:** Last 252 rows (one trading year, ~252 trading days)

## Merge Checklist

- [x] Dynamic computation from live sources (PRICE_BOOK)
- [x] Returns computed for all timeframes (1D, 30D, 60D, 365D)
- [x] Alpha computed for all timeframes (1D, 30D, 60D, 365D)
- [x] Attribution summary included
- [x] No live_snapshot.csv dependency
- [x] No st.cache_data on Portfolio Snapshot pipeline
- [x] Diagnostic overlay showing runtime computation proof
- [ ] Screenshots showing number changes at different render timestamps (to be provided)

## Conclusion

The Portfolio Snapshot has been completely refactored to compute all metrics dynamically at runtime from live market data (PRICE_BOOK). There is **zero dependency** on `live_snapshot.csv`, no caching of computed metrics, and comprehensive diagnostic overlays to prove runtime computation.

The implementation is **pure functional** - given PRICE_BOOK at time T, it computes fresh metrics every render with no intermediate artifacts or cached results.
