# Wave Readiness Report - User Guide

## Overview
The Wave Readiness Report provides comprehensive diagnostics showing which waves are ready for display and detailed explanations for any waves that are not ready.

## Accessing the Report

### 1. Server Logs
When the application starts, the readiness report is automatically printed to the server logs:

```
====================================================================================================
WAVE READINESS REPORT
====================================================================================================
Generated: 2025-12-27 17:20:39
Coverage Threshold: 95%

SUMMARY:
  Total Waves: 28
  Ready: 3 (10.7%)
  Not Ready: 25 (89.3%)

FAILURE BREAKDOWN:
  MISSING_PRICES: 25

READY WAVES (3):
  ✓ gold_wave: Gold Wave
    Coverage: 100.0% | Window: 25 days
  ✓ income_wave: Income Wave
    Coverage: 100.0% | Window: 25 days
  ✓ sp500_wave: S&P 500 Wave
    Coverage: 100.0% | Window: 25 days
...
```

### 2. Application UI (Overview Tab)
The readiness report is available in the Overview tab as a collapsible section:

**Location**: Overview Tab → 🔍 Wave Readiness Report (Expandable)

**Components**:
- **Summary Metrics**: 4 metric cards showing:
  - Total Waves (28)
  - Ready count with percentage
  - Not Ready count or Partial count
  - Average Coverage percentage

- **Full Report Table**: Interactive DataFrame with sortable columns:
  - Wave ID
  - Wave Name
  - Status (READY/NOT_READY)
  - Reason Category
  - Failing Tickers
  - Coverage %
  - Required Days
  - Available Days
  - Start Date
  - End Date
  - Suggested Fix

- **Download Button**: Export report as CSV for offline analysis

- **Failure Breakdown**: Summary table showing count by failure reason

## Report Fields Explained

### wave_id
Canonical identifier for the wave (e.g., `sp500_wave`)

### wave_name
Human-readable display name (e.g., `S&P 500 Wave`)

### readiness
Wave status:
- **READY**: All data requirements met, wave can be displayed
- **NOT_READY**: Critical data missing, wave cannot be displayed
- **NOT_READY (Partial)**: Meets coverage threshold but has some missing data

### reason_category
Primary failure reason:
- **READY**: Wave is ready
- **MISSING_PRICES**: Price data not available
- **INVALID_TICKER**: Ticker is delisted or unsupported
- **SHORT_HISTORY**: Insufficient historical data
- **STALE_DATA**: Data is outdated (>5 days old)
- **MISSING_BENCHMARK**: Benchmark price data not available
- **REGISTRY_MISMATCH**: Wave not properly registered
- **PROVIDER_UNSUPPORTED**: Data provider API failure
- **OTHER**: Other unclassified issues

### failing_tickers
Comma-separated list of ticker symbols causing readiness failure

Example: `AAPL, MSFT, GOOGL, AMZN, META`

### coverage_pct
Percentage of required tickers with available data (0-100)
- **100%**: All tickers have data
- **95%+**: Meets default coverage threshold
- **<95%**: Below threshold, wave not ready

### required_window_days
Policy-defined minimum days of historical data required (default: 365)

### available_window_days
Actual number of days of data available for this wave

### start_date / end_date
Date range of available data window

### suggested_fix
Human-readable action to resolve the readiness failure

Examples:
- `"Fetch missing price data for: AAPL, MSFT, GOOGL (+7 more). Run analytics pipeline to populate data."`
- `"Remove invalid/delisted tickers: XYZ, ABC"`
- `"Fetch 200 more days of history. Run analytics pipeline with longer lookback."`
- `"Data is stale. Re-run analytics pipeline to fetch fresh prices."`

## Coverage-Based Readiness Policy

The system uses a flexible coverage-based policy instead of an all-or-nothing approach:

### Default Threshold: 95%
- Waves with ≥95% coverage can be marked as ready (with warnings)
- Problematic tickers can be dropped with a warning instead of failing the entire wave

### Configurable
The coverage threshold can be adjusted:
```python
from analytics_pipeline import generate_wave_readiness_report

# Use 90% threshold instead of default 95%
report = generate_wave_readiness_report(coverage_threshold=0.90)
```

### Benefits
1. **Graceful Degradation**: Waves with minor issues can still be displayed
2. **Actionable Diagnostics**: Clear identification of which tickers are problematic
3. **Flexible Policy**: Threshold can be adjusted based on business needs

## Ticker Normalization

The system automatically normalizes ticker symbols to match data provider conventions:

### Examples
- `BRK.B` → `BRK-B` (Berkshire Hathaway Class B)
- `BF.B` → `BF-B` (Brown-Forman Class B)
- Crypto symbols already use correct format (`BTC-USD`, `ETH-USD`)

This ensures maximum compatibility with yfinance and reduces readiness failures due to ticker formatting issues.

## Populating Missing Data

To improve wave readiness:

### 1. Run Analytics Pipeline
```bash
cd /home/runner/work/Waves-Simple/Waves-Simple
python3 analytics_pipeline.py --all --lookback=400
```

This will:
- Fetch price data for all waves
- Generate benchmark price data
- Create position snapshots
- Calculate NAV history
- Validate all data

### 2. Check Readiness Report
After running the pipeline, check the readiness report to verify improvements:
```python
from analytics_pipeline import print_readiness_report
print_readiness_report()
```

### 3. Debug Individual Waves
For specific waves with issues:
```python
from analytics_pipeline import compute_data_ready_status

diagnostics = compute_data_ready_status('ai_cloud_megacap_wave')
print(diagnostics)
```

## Best Practices

### 1. Regular Monitoring
- Check readiness report daily
- Monitor coverage percentage trends
- Address failing tickers promptly

### 2. Data Freshness
- Run analytics pipeline at least daily
- Monitor for stale data warnings (>5 days old)
- Set up automated pipeline runs

### 3. Ticker Maintenance
- Remove delisted tickers from wave holdings
- Update ticker symbols as needed
- Verify ticker availability with data provider

### 4. Coverage Threshold
- Start with default 95% threshold
- Adjust based on business requirements
- Document threshold changes

## Troubleshooting

### Issue: All Waves Show "MISSING_PRICES"
**Solution**: Run the analytics pipeline to fetch and populate price data

### Issue: Specific Tickers Always Fail
**Solution**: Verify ticker symbols are correct and not delisted

### Issue: Data is Stale
**Solution**: Re-run analytics pipeline to fetch fresh data

### Issue: Coverage Below Threshold
**Solution**: 
1. Check failing tickers
2. Remove delisted/invalid tickers from wave holdings
3. Verify data provider API access
4. Consider lowering coverage threshold if appropriate

## Report Export

The readiness report can be exported as CSV from the UI or programmatically:

```python
from analytics_pipeline import generate_wave_readiness_report

df = generate_wave_readiness_report()
df.to_csv('wave_readiness_report.csv', index=False)
```

The exported CSV includes all diagnostic fields and can be used for:
- Offline analysis
- Sharing with team members
- Historical tracking
- Integration with monitoring systems

## Summary

The Wave Readiness Report provides:
- ✅ Deterministic diagnostics for all 28 waves
- ✅ Detailed failure reasons with specific tickers
- ✅ Actionable fix suggestions
- ✅ Coverage-based flexible policy
- ✅ Ticker normalization for compatibility
- ✅ Both UI and log accessibility
- ✅ CSV export capability

This comprehensive system ensures operators can quickly identify and resolve wave readiness issues, maintaining high system availability.
