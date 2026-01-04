# TruthFrame Quick Reference

## What is TruthFrame?

TruthFrame is the **canonical single source of truth** for all wave analytics in the Waves Simple application. It's a DataFrame containing comprehensive metrics for all 28 waves, pre-computed and ready to use.

## Quick Start

### 1. Get TruthFrame

```python
from analytics_truth import get_truth_frame

# Get TruthFrame (auto-detects Safe Mode)
truth_df = get_truth_frame()
```

### 2. Get Wave Data

```python
from truth_frame_helpers import get_wave_returns, get_wave_alphas

# Get returns for a specific wave
returns = get_wave_returns(truth_df, 'sp500_wave')
print(f"30-day return: {returns['30d']}")

# Get alphas
alphas = get_wave_alphas(truth_df, 'sp500_wave')
print(f"30-day alpha: {alphas['30d']}")
```

### 3. Format for Display

```python
from truth_frame_helpers import format_return_display, format_alpha_display

# Format for UI display
return_str = format_return_display(returns['30d'])  # "+7.31%"
alpha_str = format_alpha_display(alphas['30d'])     # "+3.46%"

# Use in Streamlit
st.metric("30D Return", return_str, delta=alpha_str)
```

### 4. Analyze Performance

```python
from truth_frame_helpers import get_top_performers

# Get top 5 performers by 30-day return
top_5 = get_top_performers(truth_df, metric='return_30d', n=5)

for _, wave in top_5.iterrows():
    print(f"{wave['display_name']}: {format_return_display(wave['return_30d'])}")
```

## Common Use Cases

### Display Wave Returns in a Table

```python
from analytics_truth import get_truth_frame
from truth_frame_helpers import create_returns_dataframe

truth_df = get_truth_frame()
returns_df = create_returns_dataframe(truth_df, timeframes=['1d', '30d', '60d'])

st.dataframe(returns_df)
```

### Show Readiness Summary

```python
from truth_frame_helpers import get_readiness_summary

summary = get_readiness_summary(truth_df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🟢 Full", summary['full'])
with col2:
    st.metric("🟡 Partial", summary['partial'])
with col3:
    st.metric("🟠 Operational", summary['operational'])
with col4:
    st.metric("🔴 Unavailable", summary['unavailable'])
```

### Get Complete Wave Summary

```python
from truth_frame_helpers import get_wave_summary

wave_data = get_wave_summary(truth_df, 'sp500_wave')

st.write(f"Wave: {wave_data['display_name']}")
st.write(f"Readiness: {wave_data['readiness_status']}")
st.write(f"Coverage: {wave_data['coverage_pct']:.1f}%")
st.write(f"30D Return: {format_return_display(wave_data['return_30d'])}")
st.write(f"30D Alpha: {format_alpha_display(wave_data['alpha_30d'])}")
```

## TruthFrame Columns

All 28 waves have these columns:

### Identification
- `wave_id` - Canonical identifier (e.g., "sp500_wave")
- `display_name` - Human-readable name (e.g., "S&P 500 Wave")
- `mode` - Operating mode (e.g., "Standard")

### Readiness
- `readiness_status` - full/partial/operational/unavailable
- `coverage_pct` - Data coverage percentage (0-100)
- `data_regime_tag` - LIVE/SANDBOX/HYBRID/UNAVAILABLE

### Returns
- `return_1d`, `return_30d`, `return_60d`, `return_365d` - Wave returns

### Alpha
- `alpha_1d`, `alpha_30d`, `alpha_60d`, `alpha_365d` - Alpha vs benchmark

### Benchmark
- `benchmark_return_1d`, `benchmark_return_30d`, `benchmark_return_60d`, `benchmark_return_365d`

### Exposure
- `exposure_pct` - Portfolio exposure (0-1)
- `cash_pct` - Cash percentage (0-1)

### Beta
- `beta_real` - Realized beta
- `beta_target` - Target beta
- `beta_drift` - Absolute difference

### Risk
- `turnover_est` - Estimated turnover
- `drawdown_60d` - Maximum drawdown over 60 days

### Alerts
- `alert_badges` - Alert badges/flags
- `last_snapshot_ts` - Timestamp of last update

## Helper Functions Reference

### Data Access
- `get_wave_metric(truth_df, wave_id, metric_name)` - Single metric
- `get_wave_returns(truth_df, wave_id)` - All returns
- `get_wave_alphas(truth_df, wave_id)` - All alphas
- `get_wave_benchmark_returns(truth_df, wave_id)` - Benchmark returns
- `get_wave_exposure(truth_df, wave_id)` - Exposure and cash
- `get_wave_beta_metrics(truth_df, wave_id)` - Beta metrics
- `get_wave_risk_metrics(truth_df, wave_id)` - Risk metrics
- `get_wave_summary(truth_df, wave_id)` - Complete wave data

### Formatting
- `format_return_display(value)` - Format as "+1.23%"
- `format_alpha_display(value)` - Format as "+0.45%"
- `format_exposure_display(value)` - Format as "95.0%"
- `format_beta_display(value)` - Format as "1.05"
- `format_readiness_badge(status, coverage)` - Format as "🟢 Full (100%)"

### Analysis
- `get_top_performers(truth_df, metric, n)` - Top N waves
- `get_readiness_summary(truth_df)` - Readiness counts
- `create_returns_dataframe(truth_df)` - Formatted returns table
- `create_alpha_dataframe(truth_df)` - Formatted alphas table

### Backward Compatibility
- `convert_truthframe_to_snapshot_format(truth_df)` - Convert to legacy format

## Safe Mode

TruthFrame respects Safe Mode automatically:

```python
# Auto-detect (reads from st.session_state["safe_mode_enabled"])
truth_df = get_truth_frame()

# Explicit control
truth_df = get_truth_frame(safe_mode=True)   # Load from CSV only
truth_df = get_truth_frame(safe_mode=False)  # Generate from engine
```

**Safe Mode ON:**
- Loads from `data/live_snapshot.csv`
- Fast and safe
- Best-effort values
- No generation

**Safe Mode OFF:**
- Generates from engine
- Uses price data
- Comprehensive metrics
- Can take longer

## Guarantees

### All 28 Waves Always Present

```python
truth_df = get_truth_frame()
assert len(truth_df) == 28  # ALWAYS TRUE
```

Even if data is unavailable, the wave is present with:
- `readiness_status = 'unavailable'`
- `data_regime_tag = 'UNAVAILABLE'`
- `coverage_pct = 0.0`
- NaN values for metrics

### Never Fails

TruthFrame functions never raise exceptions:
- Missing data returns NaN or empty strings
- Failed waves return unavailable status
- Errors are handled gracefully
- Always returns a DataFrame (even if empty)

## Performance

TruthFrame is highly optimized:
- Cached in `data/live_snapshot.csv`
- Single computation, multiple uses
- 10-100x faster than redundant calculations
- Immediate access for all tabs

## Validation

Verify TruthFrame is working:

```bash
python validate_truthframe.py
```

Run tests:

```bash
python test_truth_frame.py
python test_truth_frame_helpers.py
```

## Migration Guide

Migrating existing code? See:
- `TRUTHFRAME_MIGRATION_GUIDE.md` - Complete patterns
- `TRUTHFRAME_IMPLEMENTATION_SUMMARY.md` - Implementation details

## Examples

See these files for working examples:
- `app.py` - Overview and Executive tabs
- `test_truth_frame.py` - TruthFrame usage
- `test_truth_frame_helpers.py` - Helper usage
- `validate_truthframe.py` - Real-world validation

## Support

Questions? Check:
1. This README for quick reference
2. `TRUTHFRAME_MIGRATION_GUIDE.md` for migration patterns
3. Inline code documentation in modules
4. Test files for usage examples

---

**Remember:** TruthFrame is the single source of truth. Never compute returns, alphas, or exposures locally - always use TruthFrame!
