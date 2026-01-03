# UI Changes - Performance Overview & Readiness Diagnostics

## Performance Overview Table (System Health Tab)

### Before:
```
### 📋 28 Waves Performance Overview
Data Source: No data available - showing placeholders

| Wave                    | 1D Return | 30D  | 60D  | 365D | Status/Confidence |
|------------------------|-----------|------|------|------|-------------------|
| S&P 500 Wave           | N/A       | N/A  | N/A  | N/A  | Unavailable       |
| AI & Cloud MegaCap Wave| N/A       | N/A  | N/A  | N/A  | Unavailable       |
| ... (all N/A)          | N/A       | N/A  | N/A  | N/A  | Unavailable       |
```

### After:
```
### 📋 28 Waves Performance Overview
**Data Source: PRICE_BOOK (prices_cache.parquet)** - Live computation from canonical price cache

| Wave                    | 1D Return | 30D   | 60D   | 365D  | Status/Confidence |
|------------------------|-----------|-------|-------|-------|-------------------|
| S&P 500 Wave           | -2%       | -45%  | -43%  | -41%  | Full              |
| AI & Cloud MegaCap Wave| +0%       | -20%  | -18%  | -15%  | Full              |
| Crypto AI Growth Wave  | +0%       | +17%  | +25%  | +45%  | Full              |
| Clean Transit Wave     | +0%       | +6%   | +8%   | +12%  | Full              |
| ... (28 total waves)   | ...       | ...   | ...   | ...   | ...               |

⚠️ Waves with Issues (0 waves)  [collapsed by default]
```

### Key Differences:
- ✅ Shows actual returns computed from PRICE_BOOK
- ✅ Clear data source label indicating PRICE_BOOK
- ✅ Status/Confidence based on ticker coverage
- ✅ Expandable section for waves with issues (if any)

---

## Wave Data Readiness Diagnostics (System Health Tab)

### Before:
```
## Wave Data Readiness Diagnostics

⚠️ Could not find data_coverage_summary.csv (and no in-memory coverage DF was found).
Please verify the file is being generated and saved to the app's working/data directory.
```

### After:
```
## Wave Data Readiness Diagnostics
**Live evaluation against PRICE_BOOK** - Real-time ticker coverage and data quality assessment

### 📊 PRICE_BOOK Truth Diagnostics

┌─────────────────────┬──────────────────────┬────────────────────────┬─────────────────────┐
│ Cache File          │ Shape (Days×Tickers) │ Date Range             │ Waves Status        │
│ prices_cache.parquet│ 505 × 149            │ 2024-08-08 to          │ 28/28 returning data│
│ Path: data/cache/...│ Total: 505 days,     │ 2025-12-26             │ 0 waves with issues │
│                     │ 149 tickers          │ Data is 8 days old     │                     │
└─────────────────────┴──────────────────────┴────────────────────────┴─────────────────────┘

✅ All waves returning data successfully

---

### 📋 Wave-by-Wave Readiness Assessment

☑️ Show only NOT data-ready

| wave_name               | data_ready | reason | coverage_pct | history_days | total_tickers | missing_tickers |
|------------------------|------------|--------|--------------|--------------|---------------|-----------------|
| (No waves shown - all are data-ready)                                                                        |

**Summary:** 28/28 waves are data-ready (100.0% readiness)
```

### Key Differences:
- ✅ PRICE_BOOK metadata panel shows cache status
- ✅ Real-time wave status with failure reason grouping
- ✅ Live readiness computation from PRICE_BOOK
- ✅ No dependency on CSV files
- ✅ Clear summary of data readiness

---

## PRICE_BOOK Truth Diagnostics Panel (Expanded View)

When some waves have issues, the panel expands to show details:

```
### 📊 PRICE_BOOK Truth Diagnostics

[Same metrics as above]

⚠️ 3 waves with N/A data:

❌ No tickers found in PRICE_BOOK (2 waves)  [expandable]
    Crypto Income Wave, Vector Muni Ladder Wave

❌ Insufficient price history (need at least 2 days) (1 wave)  [expandable]
    Some Test Wave
```

This provides immediate visibility into:
1. What's in PRICE_BOOK (shape, dates, staleness)
2. How many waves are working vs. failing
3. Why specific waves are failing (grouped by reason)

---

## Expected User Experience

### First Load (System Health Tab):
1. User navigates to "System Health" tab
2. Scrolls down to "Wave Data Readiness Diagnostics"
3. Sees PRICE_BOOK metadata panel at top (cache status, dates, staleness)
4. Sees wave status summary (28/28 returning data)
5. If issues exist, sees grouped failure reasons in expanders
6. Can toggle "Show only NOT data-ready" checkbox
7. Scrolls down to "28 Waves Performance Overview"
8. Sees live performance table with actual returns
9. If failures exist, can expand "Waves with Issues" to see details

### Data Quality:
- Clear indication when data is stale (e.g., "Data is 8 days old")
- Explicit failure reasons (not just "N/A")
- Coverage percentages show partial data situations
- Status/Confidence indicators show data quality levels

### No Stale Data:
- Always computed from current PRICE_BOOK
- No dependency on snapshot generation timing
- No confusing "could not find CSV" messages
- Single source of truth for all metrics

---

## Color Coding / Visual Indicators

### Status/Confidence Levels:
- **Full** (95-100% coverage): 🟢 Green indicator (implicit)
- **Operational** (75-95% coverage): 🟡 Yellow indicator (implicit)
- **Partial** (50-75% coverage): 🟠 Orange indicator (implicit)
- **Degraded** (<50% coverage): 🔴 Red indicator (implicit)
- **Unavailable**: ❌ Red X

### Readiness Indicators:
- **data_ready = True**: ✅ Checkmark (green)
- **data_ready = False**: ❌ X mark (red)

### PRICE_BOOK Health:
- **Fresh data** (<5 days): ✅ Success message
- **Slightly stale** (5-10 days): ⚠️ Warning
- **Very stale** (>10 days): ❌ Error message

---

## Accessibility

### Clear Labeling:
- "Data Source: PRICE_BOOK (prices_cache.parquet)" - explicit source
- "Live computation from canonical price cache" - explains freshness
- "Real-time ticker coverage and data quality assessment" - explains methodology

### Progressive Disclosure:
- Summary metrics visible immediately
- Details in expandable sections (avoid overwhelming users)
- "Show only NOT data-ready" filter to focus on problems

### Actionable Information:
- Failure reasons explain what's wrong
- Coverage percentages show partial data situations
- Staleness indicators prompt data refresh if needed

---

## Performance Expectations

### Load Time:
- PRICE_BOOK is cached in memory after first load
- Performance computation: ~0.5-1 second for all 28 waves
- Readiness computation: ~0.5-1 second for all 28 waves
- UI render: Instant (Streamlit dataframes are fast)

### Memory:
- PRICE_BOOK: ~500KB in memory (505 days × 149 tickers)
- Results cache: Minimal (<100KB)

### Scalability:
- Works with any number of waves (currently 28)
- Works with any number of periods (1D, 30D, 60D, 365D, etc.)
- Works with any PRICE_BOOK size (tested with 505 days × 149 tickers)

---

## Error Handling

### PRICE_BOOK Empty:
```
❌ Error computing wave performance: PRICE_BOOK is empty

Please build the price cache using:
- Set PRICE_FETCH_ENABLED=true
- Click "Rebuild Price Cache" in sidebar
```

### Missing Tickers:
```
⚠️ Waves with Issues (5 waves)

Wave X: No tickers found in PRICE_BOOK (coverage: 0.0%)
Wave Y: Insufficient coverage (65.0%) - Missing: TICKER1, TICKER2
```

### Computation Errors:
```
❌ Error computing wave performance: [error message]
[Stack trace in code block for debugging]
```

All errors are caught and displayed gracefully without breaking the UI.

---

## Summary of Benefits

1. **Always Current**: Data reflects PRICE_BOOK state at page load
2. **Transparent**: Clear data source and failure reason labeling
3. **Actionable**: Coverage percentages and missing ticker lists help debugging
4. **Fast**: Direct parquet cache access, no CSV parsing
5. **Reliable**: No dependency on snapshot generation timing
6. **Maintainable**: Single source of truth eliminates "two truths" problem
