# Production app.py - Technical Documentation

## Overview

The production `app.py` is the authoritative entry point for the WAVES Intelligence™ Institutional Console. It integrates comprehensive UI components with real backend engines to provide a complete, governance-ready analytics platform.

## Architecture

### Component Stack

```
app.py (Streamlit UI)
    ├── waves_engine.py (Core computation)
    ├── vector_truth.py (Attribution & governance)
    ├── decision_engine.py (Intelligence layer)
    └── alpha_attribution.py (Alpha decomposition)
```

### Tab Structure

The UI is organized into 6 tabs, each serving a distinct function:

#### 1. Overview Tab
**Purpose**: Primary dashboard for wave selection and key metrics

**Components**:
- Sidebar controls (wave, mode, time window)
- 4-column metrics display (returns, alpha, vol, risk)
- Holdings table with normalized weights
- Benchmark composition table

**Key Metrics**:
- 365D/60D Returns and Alpha
- Annualized Volatility
- Information Ratio
- Max Drawdown
- Tracking Error

#### 2. Performance Tab
**Purpose**: Detailed performance analytics

**Components**:
- Interactive NAV chart (Plotly)
- Returns summary table (1D/30D/60D/365D)
- Risk metrics table

**Visualization**:
- Dual-line chart: Wave vs Benchmark NAV
- Hover-enabled unified crosshair
- Base NAV = 1.0 at start of window

#### 3. Rankings Tab
**Purpose**: Multi-wave comparison and WaveScore rankings

**Components**:
- Parallel computation across waves
- Sortable comparison table
- WaveScore methodology expander

**Ranking Criteria**:
- 365D Alpha (primary sort)
- Information Ratio
- Volatility
- Max Drawdown

**Note**: Limited to top 10 waves for performance

#### 4. Attribution Tab
**Purpose**: Governance-grade alpha attribution

**Components**:
- Vector Truth Layer display
- Alpha Reliability Panel
- Attribution confidence gating
- Detailed attribution breakdown (expandable)
- Component breakdown table

**Attribution Flow**:
1. Compute alpha reliability metrics
2. Display reliability panel
3. Show Vector Truth report (conditional on confidence)
4. Expand detailed attribution if High confidence
5. Show attribution components table

**Key Concepts**:
- **Attribution Confidence**: High/Medium/Low based on data quality
- **Vector Truth**: Governance-native alpha decomposition
- **Reliability Gates**: Suppresses detailed views under Low confidence

#### 5. Diagnostics Tab
**Purpose**: System-level diagnostics and operational transparency

**Components**:
- VIX & Regime summary metrics
- Recent activity table (last 10 days)
- Full diagnostics CSV download
- Strategy attribution (expandable, optional)

**Diagnostic Metrics**:
- Average VIX level
- Average exposure %
- Average safe allocation %
- Regime distribution
- Risk state tracking

#### 6. Decision Intelligence Tab
**Purpose**: Actionable intelligence and decision support

**Components**:
- Daily Wave Activity headline
- What Changed / Why / Results / Checks sections
- Actions / Watch / Notes columns

**Decision Engine Integration**:
- Parses wave performance and risk context
- Generates explanatory narratives
- Provides governance-aligned recommendations
- No predictions - pure diagnostics

## Backend Engine Integration

### waves_engine.py

**Functions Used**:
- `get_all_waves()` - Wave list for selectors
- `get_modes()` - Operating mode options
- `compute_history_nav()` - Core NAV/return computation
- `get_wave_holdings()` - Holdings table
- `get_benchmark_mix_table()` - Benchmark composition
- `get_vix_regime_diagnostics()` - Diagnostics data
- `get_strategy_attribution()` - Strategy-level insights (optional)

**Key Parameters**:
- `wave_name`: Selected wave
- `mode`: Standard / Alpha-Minus-Beta / Private Logic
- `days`: 90 / 180 / 365
- `include_diagnostics`: Enable diagnostics attrs

### vector_truth.py

**Functions Used**:
- `build_vector_truth_report()` - Main attribution report
- `compute_alpha_reliability_metrics()` - Reliability scoring
- `render_alpha_reliability_panel()` - Markdown panel
- `format_vector_truth_markdown()` - Attribution display
- `render_vector_truth_alpha_attribution()` - Detailed breakdown
- `extract_alpha_attribution_breakdown()` - Component extraction

**Governance Features**:
- Benchmark snapshot stability tracking
- Regime coverage analysis
- Attribution confidence scoring
- Alpha inflation risk assessment

### decision_engine.py

**Functions Used**:
- `build_daily_wave_activity()` - Daily narrative
- `generate_decisions()` - Actions/Watch/Notes

**Context Parameters**:
- Returns (r1d, r30, r60, r365)
- Alpha (a1d, a30, a60, a365)
- Risk (vol, te, ir, mdd)
- Wave name and mode

## Error Handling Strategy

### Graceful Degradation

The app handles missing data gracefully:

1. **No network / yfinance failures**: Empty DataFrames, NaN metrics
2. **Insufficient data**: Display "N/A" or info messages
3. **Engine errors**: Catch and display with exception details
4. **Optional features**: Wrapped in try/except with fallback messages

### Exception Hierarchy

```python
try:
    # Core logic
except (KeyError, ValueError, RuntimeError) as e:
    # Expected errors - display user-friendly message
except Exception as e:
    # Unexpected errors - display with full traceback
```

## Utility Functions

### Format Helpers

- `format_pct(value, decimals)` - Decimal to percentage string
- `format_num(value, decimals)` - Number formatting with NaN handling

### Metrics Computation

- `compute_metrics(nav_df)` - Extract all performance metrics from NAV
  - Returns: r1d/r30/r60/r365 for wave and benchmark
  - Alpha: a1d/a30/a60/a365
  - Risk: vol, te, ir, mdd

### Visualization

- `create_nav_chart(nav_df, wave_name)` - Plotly NAV chart
  - Dual-line with legend
  - Unified hover mode
  - Professional styling

## Sidebar Configuration

**Controls**:
- Wave selector: Dropdown of all 20 waves
- Mode selector: Standard / Alpha-Minus-Beta / Private Logic
- Window selector: 90D / 180D / 365D

**Default Values**:
- Wave: "US MegaCap Core Wave" (if available)
- Mode: "Standard" (index 0)
- Window: 365D (index 2)

## Testing Strategy

### Validation Levels

1. **Syntax**: `python3 -m py_compile app.py` ✓
2. **Imports**: Backend engine imports ✓
3. **Logic**: Core functions without UI ✓
4. **Integration**: With real data (requires network)

### Test Coverage

- ✓ Wave selection and holdings
- ✓ NAV computation (empty data case)
- ✓ Metrics computation
- ✓ Benchmark composition
- ✓ Vector Truth report generation
- ✓ Decision Engine integration
- ✓ Diagnostics (empty data case)
- ✓ Error handling paths

## Deployment Checklist

- [ ] Ensure internet connectivity for yfinance
- [ ] Verify Streamlit >= 1.32.0
- [ ] Install all requirements.txt dependencies
- [ ] Run `streamlit run app.py`
- [ ] Test with at least 3 different waves
- [ ] Verify all tabs load without errors
- [ ] Check download functionality (Diagnostics CSV)
- [ ] Validate charts render correctly
- [ ] Test error scenarios (no network, etc.)

## Performance Considerations

### Optimization Points

1. **Rankings Tab**: Limited to 10 waves to reduce computation time
2. **Diagnostics**: Shadow simulation cached in attrs
3. **Charts**: Single chart per tab (not per metric)
4. **Caching**: LRU cache in waves_engine for ticker metadata

### Known Bottlenecks

- yfinance downloads (~5-20s per wave)
- Multi-wave rankings (10x downloads)
- Strategy attribution (shadow simulation)

### Recommendations

- Use shorter windows (90D) for faster iteration
- Cache NAV DataFrames in session_state (future enhancement)
- Consider pre-computed wave_history.csv for offline mode

## Security Audit Results

✅ **CodeQL Scan**: 0 alerts
✅ **Code Review**: All issues resolved
✅ **Exception Handling**: Specific exception types
✅ **Input Validation**: Streamlit widgets provide safe input
✅ **No Credentials**: No hardcoded secrets or API keys

## Future Enhancements

Potential improvements (not in scope for this PR):

1. Session state caching for NAV data
2. Async/parallel wave computation
3. Real-time data refresh button
4. Export all tabs to PDF
5. User preferences persistence
6. Custom wave creation UI
7. What-If scenario builder
8. Historical backtest mode
9. Alert thresholds and notifications
10. Multi-wave portfolio construction

## Migration Notes

### From sandbox_app.py

**Removed**:
- ❌ Mock data generators
- ❌ Random seed controls
- ❌ Sandbox labels and warnings
- ❌ Placeholder performance metrics

**Added**:
- ✅ Real engine integration
- ✅ Production metrics
- ✅ Vector Truth attribution
- ✅ Decision Intelligence
- ✅ Comprehensive error handling
- ✅ Downloadable diagnostics

### Backward Compatibility

This is a **complete replacement** of app.py, not an incremental change.

Previous placeholder app.py is **not compatible** - all functionality rewritten from scratch using production engines.

## Support and Maintenance

**Primary Contact**: WAVES Engineering Team
**Documentation**: See inline docstrings and this file
**Issues**: Use GitHub Issues for bugs or feature requests
**Version**: v17.1 (matches waves_engine.py version)

---

**Last Updated**: 2025-12-21  
**Status**: Production Ready ✅
