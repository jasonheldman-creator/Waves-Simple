# Portfolio Snapshot and Attribution Window Alignment Implementation

## Overview

This implementation fixes a contract violation where Portfolio Snapshot tiles (30D/60D/365D) and Portfolio-Level Alpha Attribution metrics had inconsistent UI labels and diagnostic start dates that didn't accurately reflect the true rolling windows.

## Changes Made

### 1. Helper Function: `slice_windowed_series()`

**Location**: `helpers/wave_performance.py`

**Purpose**: Provides a single, reusable function to slice aligned daily return series into strict rolling windows.

**Specifications**:
- **Input**: 
  - `series`: Aligned daily return series (pandas Series with DatetimeIndex)
  - `window_days`: Requested rolling window size in trading days
  
- **Output**:
  - `windowed_series`: Last N trading rows (or empty if unavailable)
  - `start_date`: First date of window (YYYY-MM-DD) or None
  - `end_date`: Last date of window (YYYY-MM-DD) or None
  - `rows_used`: Number of rows actually used
  - `available`: True if rows_used == window_days

**Key Behavior**:
- Returns exactly the last N trading rows when sufficient data exists
- Returns `available=False` when insufficient data (no fallback to all data)
- Provides diagnostic info for troubleshooting

### 2. Updated `compute_portfolio_alpha_attribution()`

**Location**: `helpers/wave_performance.py`

**Changes**:
- Uses `slice_windowed_series()` helper to slice realized, unoverlay, and benchmark returns
- Computes cumulative returns **only** when `available=True`
- Updates diagnostics for rolling windows:
  - `start_date` and `end_date` from the windowed slice
  - `rows_used` == requested period days when available
  - If unavailable, cumulative returns are `None` with `reason="insufficient_aligned_rows"`

**Period Summary Structure** (updated):
```python
{
    'period': int,              # Period in days (e.g., 60)
    'available': bool,          # Whether period has sufficient data
    'reason': str or None,      # "insufficient_aligned_rows" if unavailable
    'requested_period_days': int,
    'rows_used': int,          # Actual rows used
    'start_date': str,         # First date of window (YYYY-MM-DD)
    'end_date': str,           # Last date of window (YYYY-MM-DD)
    'cum_real': float or None,
    'cum_sel': float or None,
    'cum_bm': float or None,
    'total_alpha': float or None,
    'selection_alpha': float or None,
    'overlay_alpha': float or None,
    'residual': float or None
}
```

### 3. Updated `compute_portfolio_snapshot()`

**Location**: `helpers/wave_performance.py`

**Changes**:
- Uses `slice_windowed_series()` helper for strict rolling windows
- Returns `None` for tiles when insufficient data
- Computes cumulative returns using compounded math from windowed series:
  ```python
  cum_return = (1 + windowed_series).prod() - 1
  ```

### 4. Attribution Diagnostics UI Updates

**Location**: `app.py` (lines ~6892-6940)

**Changes**:
- Added display of `requested_period_days` and `rows_used`
- Shows warning when `rows_used < requested_period_days`
- Updated `compute_alpha_source_breakdown()` to extract diagnostic fields from period summary
- Added captions:
  - "All cumulative returns computed using compounded math: (1 + daily_returns).prod() - 1"
  - "Window is strictly sliced to last N trading days when available."

**Diagnostic Fields Displayed**:
- Period Used (e.g., "60D" or "since_inception")
- Requested Period Days
- Rows Used
- Start Date
- End Date
- Using Fallback Exposure
- Exposure Series Found
- Exposure Min/Max
- Cumulative Returns (Realized, Unoverlay, Benchmark)

### 5. Portfolio Snapshot Tile Updates

**Location**: `app.py` (lines ~1036-1100, ~1335-1370)

**Changes**:
- Display "N/A" instead of "—" for unavailable periods
- Added tooltip support with HTML `title` attribute
- Tooltip shows: "Insufficient aligned rows (X/Y)" where X = rows available, Y = rows requested
- Example tooltips:
  - "Insufficient aligned rows (45/60)" for 60D when only 45 days available
  - "Insufficient aligned rows (200/365)" for 365D when only 200 days available

### 6. Tests

**New Test Files**:

1. **test_windowed_series_helper.py**
   - Tests the `slice_windowed_series()` helper function
   - Validates sufficient data handling
   - Validates insufficient data handling
   - Tests exact boundary conditions
   - Tests empty series handling
   - **Result**: 4/4 tests pass ✅

2. **test_window_alignment.py**
   - Validates attribution window alignment
   - Checks that `start_date` reflects last N days of aligned data
   - Validates diagnostic accuracy
   - Tests both available and unavailable scenarios
   - Validates Portfolio Snapshot strict windowing

**Updated Test Files**:
- `test_portfolio_alpha_attribution.py` already includes new keys (start_date, end_date)

## Examples

### Example 1: Sufficient Data (60D)

When 100 days of data are available:

**Attribution Period Summary (60D)**:
```python
{
    'period': 60,
    'available': True,
    'reason': None,
    'requested_period_days': 60,
    'rows_used': 60,
    'start_date': '2024-01-15',  # Exactly 60 days before end
    'end_date': '2024-04-09',
    'cum_real': 0.0523,
    'cum_sel': 0.0498,
    'cum_bm': 0.0412,
    'total_alpha': 0.0111,
    'selection_alpha': 0.0086,
    'overlay_alpha': 0.0025,
    'residual': 0.0000
}
```

**UI Display**:
- Tile: "60D Return: +5.23%"
- Tile: "Alpha 60D: +1.11%"
- No tooltip (data available)

**Diagnostics Display**:
```
Period Used: 60D
Requested Period Days: 60
Rows Used: 60
Start Date: 2024-01-15
End Date: 2024-04-09
```

### Example 2: Insufficient Data (365D)

When only 180 days of data are available:

**Attribution Period Summary (365D)**:
```python
{
    'period': 365,
    'available': False,
    'reason': 'insufficient_aligned_rows',
    'requested_period_days': 365,
    'rows_used': 180,
    'start_date': None,
    'end_date': None,
    'cum_real': None,
    'cum_sel': None,
    'cum_bm': None,
    'total_alpha': None,
    'selection_alpha': None,
    'overlay_alpha': None,
    'residual': None
}
```

**UI Display**:
- Tile: "365D Return: N/A" (with tooltip)
- Tooltip: "Insufficient aligned rows (180/365)"
- Tile: "Alpha 365D: N/A" (with tooltip)
- Tooltip: "Insufficient aligned rows (180/365)"

**Diagnostics Display**:
```
Period Used: since_inception  (fallback)
Requested Period Days: N/A
Rows Used: 180
⚠️ Insufficient aligned rows (180/365)
Start Date: N/A
End Date: N/A
```

## Validation

### Unit Tests
- ✅ `test_windowed_series_helper.py`: 4/4 tests pass
- ✅ `test_window_alignment.py`: Validates diagnostic accuracy
- ✅ `test_portfolio_alpha_attribution.py`: Existing tests still pass with new fields

### Integration
- ✅ Helper function correctly slices series to last N rows
- ✅ Period summaries include accurate start_date and end_date
- ✅ UI displays diagnostic info correctly
- ✅ Tooltips show when data is unavailable

## Benefits

1. **Consistency**: Portfolio Snapshot tiles and Attribution Diagnostics now use the same strict windowing logic
2. **Transparency**: Users can see exactly which date range was used for each period
3. **Accuracy**: No silent fallbacks to using all available data when insufficient
4. **Diagnostics**: Clear messaging when periods are unavailable with specific row counts
5. **Maintainability**: Single helper function eliminates code duplication

## Breaking Changes

None. The changes are additive and maintain backward compatibility:
- Existing period summary keys are preserved
- New keys (`start_date`, `end_date`) are added but optional
- UI gracefully handles missing diagnostic info
