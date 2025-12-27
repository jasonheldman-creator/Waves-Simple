# Wave Readiness Report - User Guide (Graded Model)

## Overview
The Wave Readiness Report provides comprehensive diagnostics showing the graded readiness of all 28 waves using a transparent, multi-level model that eliminates silent exclusions and provides actionable insights.

## Graded Readiness Model

### Philosophy
**All 28 waves are always visible** - no wave is silently excluded. Instead of binary "ready/not ready", we use a graded model that acknowledges different levels of data availability and analytics capabilities.

### Readiness Levels

#### 🟢 Full Readiness
- **Requirements**: ≥95% ticker coverage, ≥365 days of history, fresh data (<5 days old), benchmark and NAV data available
- **Capabilities**: All analytics available including:
  - Current pricing ✓
  - Simple returns ✓
  - Multi-window returns ✓
  - Volatility metrics ✓
  - Correlation analysis ✓
  - Alpha attribution ✓
  - Advanced analytics ✓
- **Use Cases**: Complete historical analysis, multi-window comparisons, full attribution

#### 🟡 Partial Readiness
- **Requirements**: ≥90% ticker coverage, ≥30 days of history
- **Capabilities**: Basic analytics available:
  - Current pricing ✓
  - Simple returns ✓
  - Volatility metrics ✓
  - Correlation analysis (if benchmark available) ✓
  - Multi-window returns ✗ (insufficient history)
  - Alpha attribution ✗ (insufficient history)
  - Advanced analytics ✗ (insufficient history)
- **Use Cases**: Current state monitoring, short-term performance tracking, basic volatility analysis
- **Limitations**: Cannot perform deep historical analysis or advanced attribution

#### 🟠 Operational Readiness
- **Requirements**: ≥80% ticker coverage, ≥7 days of history
- **Capabilities**: Minimal analytics:
  - Current pricing ✓
  - Simple returns ✓ (limited window)
  - Volatility metrics ✗
  - Multi-window returns ✗
  - Alpha attribution ✗
  - Advanced analytics ✗
- **Use Cases**: Current state display, live ticker monitoring
- **Limitations**: Very limited analytics, primarily for current state visibility

#### 🔴 Unavailable
- **Requirements**: Below operational threshold
- **Capabilities**: None - data insufficient for reliable analytics
- **Visibility**: Wave is still visible with clear diagnostics explaining what's missing
- **Use Cases**: Diagnostic information, action planning
- **Next Steps**: Follow suggested actions to improve readiness

## Accessing the Report

### 1. Server Logs
When the application starts, the graded readiness report is automatically printed to the server logs:

```
====================================================================================================
WAVE READINESS REPORT - GRADED MODEL
====================================================================================================
Generated: 2025-12-27 19:15:00
Coverage Threshold (Full): 95%

SUMMARY:
  Total Waves: 28
  Full (All Analytics): 3 (10.7%)
  Partial (Basic Analytics): 5 (17.9%)
  Operational (Current State): 8 (28.6%)
  Unavailable: 12 (42.9%)

  USABLE WAVES (operational or better): 16 (57.1%)
```

### 2. Application UI (Overview Tab)
The graded readiness report is available in the Overview tab as an expandable section:

**Location**: Overview Tab → 🔍 Wave Readiness Report (Expandable)

**Components**:
- **Summary Metrics**: 5 metric cards showing:
  - Total Waves (28)
  - Full readiness count
  - Partial readiness count
  - Operational readiness count
  - Usable waves (operational or better)

- **Graded Model Explanation**: Expandable section explaining the four readiness levels

- **Full Report Table**: Interactive DataFrame with sortable columns:
  - Wave ID
  - Wave Name
  - Readiness Status (full/partial/operational/unavailable)
  - Readiness Summary
  - Allowed Analytics
  - Blocking Issues
  - Informational Issues (Limitations)
  - Failing Tickers
  - Coverage %
  - Available Days
  - Suggested Actions

- **Download Button**: Export graded readiness report as CSV

- **Status Distribution**: Table showing count by readiness level

- **Blocking Issues Breakdown**: For unavailable waves, shows frequency of different blocking issues

## Report Fields Explained

### readiness_status
Wave's graded readiness level:
- **full**: Complete data, all analytics available
- **partial**: Good data, basic analytics available
- **operational**: Minimal data, current state display only
- **unavailable**: Insufficient data for analytics

### readiness_summary
Human-readable summary explaining the wave's capabilities and limitations

### allowed_analytics
Summary of which analytics are available for this wave based on its readiness level.
Examples:
- `"Pricing, Returns, Volatility, Multi-Window, Alpha"` (full)
- `"Pricing, Returns, Volatility"` (partial)
- `"Pricing, Returns"` (operational)
- `"None"` (unavailable)

### blocking_issues
Issues preventing the wave from reaching operational status. Examples:
- `MISSING_PRICES`: Price data file not found
- `LOW_COVERAGE`: Less than 80% ticker coverage
- `INSUFFICIENT_HISTORY`: Less than 7 days of data

### informational_issues (Limitations)
Non-blocking issues that limit advanced analytics but don't prevent operational use. Examples:
- `MISSING_BENCHMARK`: Limits relative performance metrics
- `MISSING_NAV`: Limits historical performance tracking
- `INSUFFICIENT_FULL_HISTORY`: Prevents multi-window analysis
- `STALE_DATA`: Data may not reflect current market state

### coverage_pct
Percentage of required tickers with available data (0-100)
- **100%**: All tickers have data
- **95%+**: Meets full readiness threshold
- **90-94%**: Meets partial readiness threshold
- **80-89%**: Meets operational readiness threshold
- **<80%**: Below operational threshold

### available_window_days
Actual number of days of historical data available

### suggested_actions
Actionable recommendations to improve readiness. Examples:
- `"Run analytics pipeline to fetch price data: python analytics_pipeline.py --wave ai_cloud_megacap_wave"`
- `"Run analytics pipeline with --lookback=365 for full analytics"`
- `"Re-run analytics pipeline to fetch fresh data"`
- `"Review and remove 3 delisted tickers"`

## Coverage-Based Graded Policy

The system uses a flexible, graded coverage policy instead of an all-or-nothing approach:

### Thresholds
- **Full**: ≥95% coverage (default configurable)
- **Partial**: ≥90% coverage
- **Operational**: ≥80% coverage

### Benefits
1. **Graceful Degradation**: Waves with minor issues can still be used
2. **Transparent Limitations**: Clear explanation of what each wave can/cannot do
3. **Actionable Diagnostics**: Specific guidance on improving readiness
4. **No Silent Exclusions**: All waves visible regardless of status
5. **Flexible Policy**: Thresholds can be adjusted based on business needs

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

### 2. Check Graded Readiness Report
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
print(f"Status: {diagnostics['readiness_status']}")
print(f"Allowed Analytics: {diagnostics['allowed_analytics']}")
print(f"Blocking Issues: {diagnostics['blocking_issues']}")
print(f"Suggested Actions: {diagnostics['suggested_actions']}")
```

## Best Practices

### 1. Regular Monitoring
- Check graded readiness report daily
- Monitor usable waves count (operational or better)
- Address blocking issues for unavailable waves
- Track coverage percentage trends

### 2. Data Freshness
- Run analytics pipeline at least daily
- Monitor for stale data warnings (>5 days old)
- Set up automated pipeline runs

### 3. Ticker Maintenance
- Remove delisted tickers from wave holdings
- Update ticker symbols as needed
- Verify ticker availability with data provider

### 4. Readiness Improvement
- Prioritize fixing blocking issues over informational issues
- Use suggested actions to guide improvements
- Target partial → full upgrades for critical waves
- Accept operational status for monitoring-only waves

## Troubleshooting

### Issue: Wave shows as Unavailable
**Diagnosis**: Check `blocking_issues` field
**Solutions**: 
1. If `MISSING_PRICES`: Run analytics pipeline to fetch data
2. If `LOW_COVERAGE`: Review failing tickers, remove delisted ones
3. If `INSUFFICIENT_HISTORY`: Run pipeline with longer lookback

### Issue: Wave stuck at Operational
**Diagnosis**: Check `informational_issues` - likely insufficient history
**Solution**: Run analytics pipeline with `--lookback=365` for full history

### Issue: Wave stuck at Partial
**Diagnosis**: Check `informational_issues` - likely missing benchmark or NAV
**Solution**: Run analytics pipeline to generate missing data files

### Issue: All Waves Show Low Coverage
**Solution**: Run the analytics pipeline to populate price data for all waves

## Report Export

The graded readiness report can be exported as CSV from the UI or programmatically:

```python
from analytics_pipeline import generate_wave_readiness_report

df = generate_wave_readiness_report()
df.to_csv('wave_readiness_graded_report.csv', index=False)
```

The exported CSV includes all diagnostic fields and can be used for:
- Offline analysis
- Sharing with team members
- Historical tracking of readiness improvements
- Integration with monitoring systems
- Capacity planning

## Summary

The Graded Wave Readiness Report provides:
- ✅ Four-level readiness model (full/partial/operational/unavailable)
- ✅ No silent exclusions - all 28 waves always visible
- ✅ Transparent diagnostics with blocking vs informational issues
- ✅ Analytics gating based on data availability
- ✅ Actionable recommendations for improvement
- ✅ Graceful degradation of features
- ✅ Coverage-based flexible policy
- ✅ Both UI and log accessibility
- ✅ CSV export capability

This comprehensive system ensures operators can quickly understand wave capabilities, identify improvement opportunities, and make informed decisions about wave usage without misleading binary ready/not-ready signals.

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
