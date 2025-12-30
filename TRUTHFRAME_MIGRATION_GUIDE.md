# TruthFrame Migration Guide

## Overview

This guide explains how to refactor code in the Waves Simple application to use the canonical **TruthFrame** as the single source of truth for all wave analytics data.

## What is TruthFrame?

TruthFrame is a single DataFrame containing comprehensive analytics for all 28 waves in the system. It includes:
- Returns (1d, 30d, 60d, 365d)
- Alphas (1d, 30d, 60d, 365d)
- Benchmark returns (1d, 30d, 60d, 365d)
- Exposure and cash percentages
- Beta metrics (real, target, drift)
- Risk metrics (turnover, drawdown)
- Readiness and coverage information
- Alert badges

## Why Migrate to TruthFrame?

**Before (Problems):**
- Redundant calculations across different tabs
- Inconsistent metrics between views
- Performance issues from repeated computations
- Difficult to maintain and debug
- Risk of calculation errors

**After (Benefits):**
- Single source of truth - all tabs use same data
- Consistent metrics everywhere
- Better performance - calculate once, use everywhere
- Easier to maintain
- Guaranteed to show all 28 waves

## Migration Pattern

### Step 1: Load TruthFrame

**OLD WAY:**
```python
from snapshot_ledger import load_snapshot

snapshot_df = load_snapshot()
```

**NEW WAY:**
```python
from analytics_truth import get_truth_frame

# Respects Safe Mode automatically
truth_df = get_truth_frame()
```

### Step 2: Get Wave Metrics

**OLD WAY:**
```python
from waves_engine import compute_history_nav

# Compute returns locally (REDUNDANT!)
nav_df = compute_history_nav(wave_name, mode="Standard", days=30)
if not nav_df.empty:
    wave_return = (nav_df['wave_nav'].iloc[-1] / nav_df['wave_nav'].iloc[0] - 1)
    bm_return = (nav_df['bm_nav'].iloc[-1] / nav_df['bm_nav'].iloc[0] - 1)
    alpha = wave_return - bm_return
```

**NEW WAY:**
```python
from truth_frame_helpers import get_wave_returns, get_wave_alphas

# Get pre-computed values from TruthFrame
returns = get_wave_returns(truth_df, wave_id)
alphas = get_wave_alphas(truth_df, wave_id)

wave_return_30d = returns['30d']  # Already computed!
alpha_30d = alphas['30d']  # Already computed!
```

### Step 3: Format for Display

**OLD WAY:**
```python
# Manual formatting (inconsistent across app)
if pd.notna(wave_return):
    return_str = f"{wave_return*100:.2f}%"
else:
    return_str = "N/A"
```

**NEW WAY:**
```python
from truth_frame_helpers import format_return_display

# Consistent formatting everywhere
return_str = format_return_display(returns['30d'])
```

## Complete Examples

### Example 1: Displaying Wave Returns in a Table

**OLD WAY:**
```python
# In render_some_tab():
for wave_name in wave_list:
    try:
        # Compute returns (SLOW and REDUNDANT!)
        nav_df = compute_history_nav(wave_name, mode="Standard", days=30)
        
        if not nav_df.empty and len(nav_df) >= 2:
            ret_30d = (nav_df['wave_nav'].iloc[-1] / nav_df['wave_nav'].iloc[0] - 1)
            ret_str = f"{ret_30d*100:.2f}%"
        else:
            ret_str = "N/A"
        
        st.text(f"{wave_name}: {ret_str}")
    except Exception as e:
        st.text(f"{wave_name}: Error")
```

**NEW WAY:**
```python
from analytics_truth import get_truth_frame
from truth_frame_helpers import create_returns_dataframe

# In render_some_tab():
truth_df = get_truth_frame()

# Create formatted returns DataFrame
returns_df = create_returns_dataframe(truth_df, timeframes=['30d'])

# Display with Streamlit
st.dataframe(returns_df)
```

### Example 2: Getting Wave Metrics for Display

**OLD WAY:**
```python
# Multiple redundant calls
nav_df_1d = compute_history_nav(wave_name, mode="Standard", days=1)
nav_df_30d = compute_history_nav(wave_name, mode="Standard", days=30)
nav_df_60d = compute_history_nav(wave_name, mode="Standard", days=60)

# Manual calculation
ret_1d = calculate_return(nav_df_1d)
ret_30d = calculate_return(nav_df_30d)
ret_60d = calculate_return(nav_df_60d)

alpha_1d = calculate_alpha(nav_df_1d)
alpha_30d = calculate_alpha(nav_df_30d)
alpha_60d = calculate_alpha(nav_df_60d)

# Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("1D Return", f"{ret_1d*100:.2f}%", delta=f"{alpha_1d*100:.2f}%")
with col2:
    st.metric("30D Return", f"{ret_30d*100:.2f}%", delta=f"{alpha_30d*100:.2f}%")
with col3:
    st.metric("60D Return", f"{ret_60d*100:.2f}%", delta=f"{alpha_60d*100:.2f}%")
```

**NEW WAY:**
```python
from analytics_truth import get_truth_frame
from truth_frame_helpers import get_wave_returns, get_wave_alphas, format_return_display, format_alpha_display

# Get TruthFrame once
truth_df = get_truth_frame()

# Get all returns at once (already computed!)
returns = get_wave_returns(truth_df, wave_id)
alphas = get_wave_alphas(truth_df, wave_id)

# Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("1D Return", format_return_display(returns['1d']), delta=format_alpha_display(alphas['1d']))
with col2:
    st.metric("30D Return", format_return_display(returns['30d']), delta=format_alpha_display(alphas['30d']))
with col3:
    st.metric("60D Return", format_return_display(returns['60d']), delta=format_alpha_display(alphas['60d']))
```

### Example 3: Top Performers Ranking

**OLD WAY:**
```python
# Compute returns for all waves (VERY SLOW!)
wave_returns = []
for wave_name in all_waves:
    try:
        nav_df = compute_history_nav(wave_name, mode="Standard", days=30)
        ret = calculate_return(nav_df)
        wave_returns.append({'wave': wave_name, 'return': ret})
    except:
        pass

# Sort manually
wave_returns.sort(key=lambda x: x['return'], reverse=True)
top_5 = wave_returns[:5]

# Display
for item in top_5:
    st.text(f"{item['wave']}: {item['return']*100:.2f}%")
```

**NEW WAY:**
```python
from analytics_truth import get_truth_frame
from truth_frame_helpers import get_top_performers, format_return_display

# Get TruthFrame once
truth_df = get_truth_frame()

# Get top 5 performers (one line!)
top_5 = get_top_performers(truth_df, metric='return_30d', n=5)

# Display
for _, wave in top_5.iterrows():
    st.text(f"{wave['display_name']}: {format_return_display(wave['return_30d'])}")
```

## Common Helper Functions

### Data Access
- `get_truth_frame(safe_mode=None)` - Load TruthFrame (respects Safe Mode)
- `get_wave_metric(truth_df, wave_id, metric_name)` - Get single metric
- `get_wave_returns(truth_df, wave_id)` - Get all return timeframes
- `get_wave_alphas(truth_df, wave_id)` - Get all alpha timeframes
- `get_wave_summary(truth_df, wave_id)` - Get complete wave data

### Formatting
- `format_return_display(value)` - Format return as "+1.23%"
- `format_alpha_display(value)` - Format alpha as "+0.45%"
- `format_exposure_display(value)` - Format exposure as "95.0%"
- `format_beta_display(value)` - Format beta as "1.05"
- `format_readiness_badge(status, coverage)` - Format as "🟢 Full (100%)"

### Analysis
- `get_top_performers(truth_df, metric, n)` - Get top N waves
- `get_readiness_summary(truth_df)` - Get readiness counts
- `create_returns_dataframe(truth_df)` - Create formatted returns table
- `create_alpha_dataframe(truth_df)` - Create formatted alphas table

## Migration Checklist for a Tab

When migrating a tab to use TruthFrame:

1. [ ] Replace `compute_history_nav()` calls with TruthFrame access
2. [ ] Replace local return calculations with `get_wave_returns()`
3. [ ] Replace local alpha calculations with `get_wave_alphas()`
4. [ ] Replace manual formatting with helper functions
5. [ ] Remove redundant try/except blocks for data fetching
6. [ ] Test with Safe Mode ON
7. [ ] Test with Safe Mode OFF
8. [ ] Verify all 28 waves are displayed
9. [ ] Check performance improvement

## Important Notes

### Safe Mode Support
TruthFrame automatically respects Safe Mode:
```python
# Safe Mode is detected automatically from session state or environment
truth_df = get_truth_frame()  # Will use Safe Mode if enabled

# Or explicitly control:
truth_df = get_truth_frame(safe_mode=True)   # Force Safe Mode
truth_df = get_truth_frame(safe_mode=False)  # Force full generation
```

### All 28 Waves Always Present
TruthFrame NEVER drops rows. Even if data is unavailable:
```python
truth_df = get_truth_frame()
assert len(truth_df) == 28  # Always true!

# Unavailable data is marked, not hidden
unavailable_waves = truth_df[truth_df['readiness_status'] == 'unavailable']
```

### Backward Compatibility
For existing code that expects `snapshot_df` format:
```python
from analytics_truth import get_truth_frame

truth_df = get_truth_frame()

# Convert to old format if needed (temporary during migration)
snapshot_df = truth_df.rename(columns={
    'wave_id': 'Wave_ID',
    'display_name': 'Wave',
    'return_1d': 'Return_1D',
    # ... etc
})
```

## Testing Your Migration

After migrating a tab:

1. **Visual Test:** Open the tab and verify data displays correctly
2. **Safe Mode Test:** Toggle Safe Mode and verify tab still works
3. **Performance Test:** Note loading time improvement
4. **28 Waves Test:** Verify all 28 waves appear (even unavailable ones)
5. **Data Consistency Test:** Compare with other tabs - should show same metrics

## Getting Help

If you encounter issues during migration:

1. Check `analytics_truth.py` for TruthFrame column names
2. Check `truth_frame_helpers.py` for available helper functions
3. Look at `app.py` Overview tab for migration example
4. Run tests: `python test_truth_frame.py` and `python test_truth_frame_helpers.py`

## Benefits After Migration

✅ **Performance:** 10-100x faster (no redundant calculations)  
✅ **Consistency:** Same metrics everywhere  
✅ **Maintainability:** Single place to update calculations  
✅ **Reliability:** All 28 waves always present  
✅ **Safe Mode:** Works correctly in both modes  
✅ **Testability:** Easier to test with pre-computed data  

## Migration Priority

Suggested order for migrating tabs:

1. ✅ Overview tab (DONE)
2. ✅ Executive/Console tab (DONE)
3. 🔄 Individual Wave pages (IN PROGRESS)
4. Details/Factor Decomposition tab
5. Reports/Risk Lab tab
6. Diagnostics tab
7. Other specialized tabs

---

**Remember:** The goal is to eliminate all local calculations and use TruthFrame as the single source of truth!
