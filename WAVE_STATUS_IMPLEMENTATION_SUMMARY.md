# Wave Status Field Implementation Summary

## Overview

This document describes the implementation of a wave status field to distinguish between ACTIVE and STAGING waves in the Waves-Simple application.

## Problem Statement Requirements

1. Add a `status` field to the wave registry; default value is `ACTIVE` if the field is missing or unspecified.
2. Update wave initialization: Mark all waves without completed dynamic benchmark + volatility overlay data as `STAGING`.
3. Modify aggregations in summary view components: Apply logic to exclude all waves with `status != ACTIVE` by default.
4. Add a UI toggle: UI control should toggle `Include Staging Waves` (default: off).
5. Ensure per-wave detail pages (single-wave analysis modules) remain accessible for STAGING waves.
6. Update `live_snapshot.csv` behavior: Ensure rows representing STAGING waves **do not** affect computed aggregate statistics.

## Implementation Details

### 1. Wave Registry Enhancement (`data/wave_registry.csv` and `helpers/wave_registry.py`)

**Changes:**
- Added `status` column to `data/wave_registry.csv` with default value `ACTIVE`
- Updated `get_wave_registry()` to handle missing status field with fallback to `ACTIVE`
- Added `get_active_status_waves(include_staging=False)` function to filter waves by status
- Added `check_wave_data_readiness(wave_id)` to verify wave has complete benchmark + volatility data
- Added `update_wave_status_based_on_readiness()` to automatically mark waves as STAGING/ACTIVE

**Data Readiness Criteria:**
A wave is considered ready (ACTIVE) if it has:
1. Valid benchmark specification
2. At least 30 days of history in `wave_history.csv`
3. Non-null volatility overlay data (vix_level and vix_regime columns)

**Current Status Distribution:**
- ACTIVE: 25 waves
- STAGING: 3 waves
  - Russell 3000 Wave
  - SmartSafe Tax-Free Money Market Wave
  - SmartSafe Treasury Cash Wave

### 2. Snapshot Generation Updates (`analytics_truth.py`)

**Changes:**
- Modified `_compute_wave_metrics_from_tickers()` to accept `wave_status` parameter
- Updated `generate_live_snapshot_csv()` to load wave registry and propagate `wave_status` to each row
- Updated `generate_snapshot_with_full_coverage()` to include `wave_status` from registry
- Added `wave_status` column to all NO DATA fallback rows

**Behavior:**
- Snapshot CSV now includes `wave_status` column for each wave
- Wave status is synchronized with wave registry on snapshot generation

### 3. UI Filtering (`app.py`)

**Changes in Overview Tab (`render_overview_tab`):**
- Added "Include Staging Waves" checkbox (default: off)
- Filter `snapshot_df` by `wave_status` before displaying data
- Show info message about filtered waves
- Added `wave_status` column to display table

**Changes in Executive Brief Tab (`render_executive_brief_tab`):**
- Added "Include Staging Waves" checkbox (default: off)
- Filter `snapshot_df` by `wave_status` before computing aggregations
- Show caption about filtered waves

**Filtering Logic:**
```python
if not include_staging:
    snapshot_df = snapshot_df[snapshot_df['wave_status'] == 'ACTIVE'].copy()
else:
    # Include both ACTIVE and STAGING
    pass  # No filtering
```

### 4. Per-Wave Detail Pages

**No changes required:**
- Detail pages use `get_wave_universe()` which is NOT filtered by status
- STAGING waves remain fully accessible through wave selectors
- Individual wave analysis works identically for STAGING and ACTIVE waves

### 5. Wave Status Initialization Script

**New File: `update_wave_status.py`**
- Standalone script to update wave status based on data readiness
- Can be run manually or as part of deployment pipeline
- Reports status changes and current distribution

**Usage:**
```bash
python3 update_wave_status.py
```

### 6. Testing

**New File: `test_wave_status_filtering.py`**

Test coverage includes:
1. Wave registry has status field
2. Status-based filtering works correctly
3. Live snapshot includes wave_status column
4. Snapshot filtering by wave_status
5. Wave data readiness check function

**All tests pass successfully.**

## Files Modified

1. `data/wave_registry.csv` - Added status column
2. `data/live_snapshot.csv` - Added wave_status column
3. `helpers/wave_registry.py` - Added status handling and filtering functions
4. `analytics_truth.py` - Updated snapshot generation to include wave_status
5. `app.py` - Added UI toggles and filtering in overview/executive tabs

## Files Created

1. `update_wave_status.py` - Script to update wave status based on data readiness
2. `test_wave_status_filtering.py` - Comprehensive test suite

## Usage Guide

### For End Users

**Viewing Aggregate Metrics (Overview/Executive Tabs):**
- By default, only ACTIVE waves are included in aggregate statistics
- Enable "Include Staging Waves" checkbox to see all waves
- STAGING waves are clearly marked in the wave_status column

**Viewing Individual Waves:**
- All waves (ACTIVE and STAGING) are available in wave selectors
- No filtering applied to individual wave detail views
- STAGING waves work identically to ACTIVE waves

### For Administrators

**Marking Waves as STAGING:**
```bash
python3 update_wave_status.py
```

**Manual Status Update:**
Edit `data/wave_registry.csv` and change the `status` column for specific waves.

**Regenerating Snapshot:**
The snapshot will automatically include wave_status from the registry on next generation.

## Design Decisions

1. **Default to ACTIVE:** If status field is missing, waves default to ACTIVE to maintain backward compatibility.

2. **No Filtering in Detail Views:** Individual wave pages don't filter by status to ensure all waves remain accessible for analysis.

3. **UI Toggle Default OFF:** Aggregate views exclude STAGING waves by default to show only production-ready data.

4. **Data Readiness Criteria:** Waves need 30+ days of history with volatility data to be considered ACTIVE.

5. **Minimal Changes:** Implementation touches only the necessary files to add filtering without disrupting existing functionality.

## Testing Performed

1. ✅ Unit tests for all new functions
2. ✅ Status field handling and defaults
3. ✅ Filtering logic (ACTIVE only vs ACTIVE+STAGING)
4. ✅ Snapshot generation with wave_status
5. ✅ UI toggle functionality
6. ✅ Detail page accessibility for STAGING waves
7. ✅ Code review (3 issues found and fixed)
8. ✅ Security scan (0 vulnerabilities found)

## Backward Compatibility

- Existing code continues to work without modification
- If `status` field is missing, defaults to `ACTIVE`
- If `wave_status` column is missing in snapshot, filtering is skipped with a warning
- No breaking changes to existing APIs or data structures

## Future Enhancements

Potential improvements for future iterations:

1. Add status filter to other aggregate views (heatmaps, leaderboards)
2. Add automatic promotion of STAGING → ACTIVE when data becomes ready
3. Add UI indicator for STAGING waves in wave selectors
4. Add admin UI for manually changing wave status
5. Add status change history/audit log
