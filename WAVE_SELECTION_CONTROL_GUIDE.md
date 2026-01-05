# Wave Selection Control - Implementation Guide

## Overview
This document describes the wave selection control UI feature that allows users to switch between Portfolio (All Waves) view and individual wave contexts.

## Problem Solved
Previously, the application was permanently locked in portfolio context with no UI control to select individual waves. Users could not access wave-specific metrics or analytics.

## Solution
Added a wave selector dropdown at the top of the sidebar that allows users to:
1. View portfolio-level metrics (all waves aggregated)
2. Switch to any individual wave to see wave-specific metrics
3. Easily switch back to portfolio view

## User Interface

### Sidebar Location
The wave selector is located at the **top of the sidebar**, before the Safe Mode controls. This ensures it's always visible and accessible.

```
┌─────────────────────────────────┐
│  Sidebar                        │
├─────────────────────────────────┤
│  🌊 Wave Selection              │  ← NEW: Always visible
│  ┌──────────────────────────┐  │
│  │ Portfolio Snapshot (All  │  │  ← Default option
│  │ Waves)                   │  │
│  └──────────────────────────┘  │
│                                 │
│  🏛️ Portfolio View Active      │  ← Visual feedback
│                                 │
├─────────────────────────────────┤
│  🛡️ Safe Mode                   │
│  ...                            │
└─────────────────────────────────┘
```

### Wave Selector Options
The dropdown contains:
1. **First option**: "Portfolio Snapshot (All Waves)" (default)
2. **Remaining options**: All active waves from the registry, sorted alphabetically

Example options:
- Portfolio Snapshot (All Waves)
- AI & Cloud MegaCap Wave
- Clean Transit-Infrastructure Wave
- Crypto AI Growth Wave
- Gold Wave
- Income Wave
- ... (all active waves)

### Visual Feedback
After making a selection, the UI shows:
- **Portfolio mode**: `🏛️ Portfolio View Active`
- **Wave mode**: `🌊 Wave View: [Wave Name]`

## Behavior

### Default State
- On first load: Portfolio mode (no wave selected)
- `st.session_state.selected_wave = None`
- Shows portfolio-level aggregated metrics

### Selecting an Individual Wave
1. User clicks dropdown
2. User selects a wave (e.g., "Gold Wave")
3. `st.session_state.selected_wave = "Gold Wave"`
4. Banner updates to show wave name
5. Wave-specific metrics appear:
   - Beta
   - VIX Regime
   - Exposure %
   - Cash %

### Switching Back to Portfolio
1. User clicks dropdown
2. User selects "Portfolio Snapshot (All Waves)"
3. `st.session_state.selected_wave = None`
4. Banner updates to show "Portfolio Snapshot"
5. Wave-specific metrics are hidden

### Persistence
- Selection persists across reruns (stored in session state)
- If selected wave becomes inactive, defaults back to portfolio

## Technical Implementation

### Code Location
File: `/home/runner/work/Waves-Simple/Waves-Simple/app.py`
Function: `render_sidebar_info()` (lines 6952-7012)

### Key Components

#### 1. Get Wave List
```python
wave_universe_version = st.session_state.get("wave_universe_version", 1)
universe = get_canonical_wave_universe(force_reload=False, _wave_universe_version=wave_universe_version)
all_waves = universe.get("waves", [])
```

#### 2. Build Options
```python
wave_options = [PORTFOLIO_VIEW_TITLE] + sorted(all_waves)
```

#### 3. Determine Default Index
```python
current_selection = st.session_state.get("selected_wave")

if current_selection is None or current_selection == PORTFOLIO_VIEW_PLACEHOLDER:
    default_index = 0  # Portfolio
elif current_selection in wave_options:
    default_index = wave_options.index(current_selection)
else:
    default_index = 0  # Fallback to portfolio
```

#### 4. Render Selector
```python
selected_option = st.sidebar.selectbox(
    "Select Context",
    options=wave_options,
    index=default_index,
    key="wave_selector",
    help="Choose Portfolio for all-waves view, or select an individual wave for wave-specific metrics"
)
```

#### 5. Update Session State
```python
if selected_option == PORTFOLIO_VIEW_TITLE:
    st.session_state.selected_wave = None
else:
    st.session_state.selected_wave = selected_option
```

### Constants Used
```python
PORTFOLIO_VIEW_PLACEHOLDER = "NONE"
PORTFOLIO_VIEW_TITLE = "Portfolio Snapshot (All Waves)"
PORTFOLIO_VIEW_ICON = "🏛️"
WAVE_VIEW_ICON = "🌊"
```

### Context Detection
The existing helper function determines whether we're in portfolio mode:
```python
def is_portfolio_context(selected_wave: str) -> bool:
    return selected_wave is None or selected_wave == PORTFOLIO_VIEW_PLACEHOLDER
```

## Integration with Existing Code

### Banner Functions
The existing banner rendering functions already handle both contexts:
- `render_selected_wave_banner_enhanced()` - Enhanced banner with metrics
- `render_selected_wave_banner_simple()` - Simple banner

Both functions:
1. Call `is_portfolio_context(selected_wave)` to check mode
2. Only load wave data if NOT in portfolio mode
3. Only show wave-specific metrics if NOT in portfolio mode

### No Changes Required To
- Calculations or analytics pipelines
- Data processing functions
- Snapshot generation
- Existing metrics display logic

## Error Handling

### If Wave List Loading Fails
```python
except Exception as e:
    st.sidebar.warning("⚠️ Could not load wave list")
    if st.session_state.get("debug_mode", False):
        st.sidebar.error(f"Error: {str(e)}")
```

### If Selected Wave No Longer Exists
The code automatically falls back to portfolio mode (default_index = 0)

## Testing

### Test Coverage
File: `test_wave_selection_control.py`

Seven comprehensive tests:
1. ✅ Default to Portfolio Mode
2. ✅ Select Individual Wave
3. ✅ Switch Back to Portfolio
4. ✅ is_portfolio_context Helper
5. ✅ Wave-Specific Metrics Visibility
6. ✅ Wave Selector Options
7. ✅ Selection Persistence

All tests passing ✅

### Security
- CodeQL scan: 0 alerts
- No vulnerabilities introduced
- Proper error handling

## Acceptance Criteria

All criteria from the problem statement are met:

✅ **Users can switch between Portfolio and any individual Wave**
   - Dropdown provides easy switching

✅ **Default selection is "Portfolio (All Waves)"**
   - `selected_wave = None` on first load

✅ **Selecting a wave switches context to wave-level view**
   - Updates session state and triggers rerender

✅ **Wave-specific metrics re-enabled**
   - Beta, VIX Regime, Exposure, Cash shown only for waves

✅ **Portfolio mode remains default**
   - Always first option, index 0

✅ **Portfolio mode does NOT suppress ability to select a wave**
   - Selector always visible and functional

✅ **No modifications to calculations, analytics, or data pipelines**
   - Strictly UI/context switching

✅ **Selector remains available even in Executive/Institutional Readiness mode**
   - Placed at top of sidebar, independent of mode settings

## Future Enhancements

Possible improvements for future iterations:
1. Add keyboard shortcuts for quick wave switching
2. Add "Recent Waves" quick access
3. Add wave comparison mode (side-by-side)
4. Add wave search/filter for large wave lists
5. Add wave favorites/bookmarks

## Conclusion

This implementation provides a clean, minimal solution to enable wave selection without modifying any core analytics or calculation logic. The UI is intuitive, always accessible, and integrates seamlessly with existing code patterns.
