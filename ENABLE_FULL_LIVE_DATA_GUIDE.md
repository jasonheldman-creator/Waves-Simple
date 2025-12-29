# How to Enable Full Live Data for All 28 Waves

## Current Status

Based on the diagnostic analysis, here's what's preventing full analytics:

| Issue | Waves Affected | Impact |
|-------|---------------|---------|
| **MISSING_PRICES** | 10 waves | No data files - crypto tickers not in cache |
| **STALE_DATA** | 7 waves | Data >7 days old - needs refresh |
| **LOW_COVERAGE** | 6 waves | Incomplete ticker data - some tickers missing |
| **INSUFFICIENT_HISTORY** | All waves | Need 365 days for "full" status (currently max 500 days but stale) |

## What You Need to Do

### Option 1: Enable Live Data Fetching (Recommended)

The analytics pipeline uses yfinance to fetch live market data, but it's currently blocked in the deployment environment.

#### Steps:

1. **Unblock yfinance API Access**
   - Check your network/firewall settings
   - Ensure the deployment environment can access Yahoo Finance API endpoints
   - Test with: `python3 -c "import yfinance as yf; print(yf.download('SPY', period='5d'))"`

2. **Run the Analytics Pipeline**
   ```bash
   # This will fetch fresh data for all 28 waves with 1 year of history
   python analytics_pipeline.py --all --lookback=365
   ```

3. **Verify Results**
   ```bash
   # Check that all waves now have full data
   python wave_readiness_diagnostics.py
   ```

**Expected Outcome**: All 28 waves should reach "Full Ready" status with 365+ days of fresh data.

---

### Option 2: Update Cached Data Manually

If live data fetching isn't possible, you can update the `prices.csv` file manually.

#### Steps:

1. **Add Missing Crypto Tickers to prices.csv**
   
   The following crypto tickers are missing (needed for 10 waves):
   ```
   TAO-USD, RENDER-USD, FET-USD, ICP-USD, OCEAN-USD, AGIX-USD
   AAVE-USD, UNI-USD, INJ-USD, MKR-USD, CRV-USD, SNX-USD, COMP-USD, LINK-USD
   ETH-USD, stETH-USD, LDO-USD, CAKE-USD
   SOL-USD, AVAX-USD, ADA-USD, DOT-USD, ATOM-USD, APT-USD, NEAR-USD
   MATIC-USD, ARB-USD, OP-USD, IMX-USD, MNT-USD, STX-USD
   ```

2. **Add Missing Traditional Tickers**
   
   For waves with LOW_COVERAGE, add these tickers:
   ```
   RIVN, LCID, CHPT, NIO, VMC, MLM, BP, PLUG, TAN, BE, SEDG
   BRK-B, KO, JNJ, PG, VTV, WMT, UNH
   IWV, IJH, VTWO, VBK, MDY, IWP, SMH, IWO
   ARKK, NET, DDOG, ZS
   SUB, SHM, MUB, HYD, TFI
   BIL, SGOV, SHY, IEF, TLT, LQD
   ```

3. **Update prices.csv Format**
   
   The file should have this format:
   ```csv
   date,ticker,close
   2024-08-08,AAPL,135.23
   2024-08-09,AAPL,135.94
   ...
   ```

4. **Extend Historical Data**
   
   Ensure you have at least 365 days of data for each ticker (not 500 days of stale data).

5. **Run Offline Data Loader**
   ```bash
   # Populate all waves from the updated prices.csv
   python offline_data_loader.py --overwrite
   ```

6. **Verify Results**
   ```bash
   python wave_readiness_diagnostics.py
   ```

---

### Option 3: Use Alternative Data Provider

If yfinance is permanently blocked, integrate an alternative data provider.

#### Recommended Providers:
- **Alpha Vantage** - Free tier available
- **Polygon.io** - Good for stocks and crypto
- **IEX Cloud** - Real-time and historical data
- **CoinGecko/CoinMarketCap** - For crypto data

#### Implementation Steps:

1. **Choose a Provider** and sign up for API access

2. **Modify analytics_pipeline.py** to use the new provider:
   ```python
   # Example: Add Alpha Vantage support
   import requests
   
   def fetch_prices_alphavantage(ticker, days=365):
       api_key = "YOUR_API_KEY"
       url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
       response = requests.get(url)
       data = response.json()
       # Parse and return DataFrame
   ```

3. **Add Fallback Logic**:
   ```python
   # Try yfinance first, fall back to alternative
   try:
       df = yf.download(ticker, period=f"{days}d")
   except:
       df = fetch_prices_alphavantage(ticker, days)
   ```

4. **Run the Pipeline**
   ```bash
   python analytics_pipeline.py --all --lookback=365
   ```

---

## Verification Checklist

After implementing any option above, verify the results:

```bash
# 1. Run diagnostics
python wave_readiness_diagnostics.py

# 2. Check that all waves are "Full Ready"
# You should see:
#   Full Ready: 28 (100.0%)
#   Partial Ready: 0 (0.0%)
#   Operational: 0 (0.0%)
#   Unavailable: 0 (0.0%)

# 3. Verify data freshness
# Check that "Primary Issue" is "OK" for all waves

# 4. Check coverage
# All waves should show 90%+ coverage

# 5. Check history length
# All waves should show 365+ days
```

---

## Expected Timeline

| Approach | Setup Time | Data Collection | Total |
|----------|-----------|-----------------|-------|
| **Option 1** (yfinance) | 5 min | 10-15 min | ~20 min |
| **Option 2** (manual) | N/A | 2-4 hours | 2-4 hours |
| **Option 3** (new provider) | 30-60 min | 15-30 min | 45-90 min |

---

## Troubleshooting

### If Analytics Pipeline Fails:

1. **Check logs**:
   ```bash
   python analytics_pipeline.py --all --lookback=365 2>&1 | tee pipeline.log
   ```

2. **Review failed tickers**:
   ```bash
   # Look for "Warning: No data returned from yfinance"
   grep "Warning:" pipeline.log
   ```

3. **Run diagnostics**:
   ```bash
   python wave_readiness_diagnostics.py json > diagnostics.json
   # Review diagnostics.json for specific failures
   ```

### If Some Waves Still Show Unavailable:

1. Check if tickers are valid (not delisted)
2. Verify ticker symbols are correct (e.g., `BRK-B` not `BRK.B`)
3. For crypto, ensure `-USD` suffix (e.g., `ETH-USD`)
4. Check API rate limits if using a third-party provider

---

## Summary

**Quickest Path to Full Data**: 
1. Unblock yfinance in your deployment environment
2. Run `python analytics_pipeline.py --all --lookback=365`
3. Verify with `python wave_readiness_diagnostics.py`

**If that's not possible**:
- Manually update `prices.csv` with missing tickers and fresh data
- Run `python offline_data_loader.py --overwrite`

**Long-term solution**:
- Integrate a reliable paid data provider
- Set up automated daily data refresh
- Implement monitoring alerts for stale data

---

Need help with a specific step? Let me know which option you'd like to pursue.
