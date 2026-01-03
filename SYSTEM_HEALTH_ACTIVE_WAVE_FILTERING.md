# System Health Enhancement: Active Wave Filtering

## Overview

This document describes the enhancement to the System Health logic to determine 'Stale/Degraded' state based only on tickers from active waves, excluding unused tickers from inactive waves and optional universes.

## Problem Statement

Previously, the system health status could be marked as 'degraded' or 'stale' due to ticker failures or stale data from:
- Inactive waves (e.g., Russell 3000 Wave which is marked as `active=False`)
- Tickers from optional universes (benchmarks, safe assets) that aren't actually used by any active wave

This led to false positives in system health monitoring, where the system appeared unhealthy even though all actively used waves had fresh, valid data.

## Solution

### Key Changes

1. **Added `active_only` parameter to `collect_all_required_tickers()`** (`data_cache.py`)
   - When `active_only=True`, reads `data/wave_registry.csv` to get list of active wave_ids
   - Filters the wave_registry dictionary to only include active waves
   - Excludes tickers that belong only to inactive waves

2. **Added `active_waves_only` parameter to `get_wave_holdings_tickers()`** (`helpers/ticker_sources.py`)
   - When `active_waves_only=True`, filters tickers from `universal_universe.csv` based on wave membership
   - Checks if wave is active in `data/wave_registry.csv` before including its tickers
   - Defaults to `True` to prevent inactive wave tickers from affecting health by default

3. **Enhanced `get_ticker_health_status()`** (`helpers/ticker_sources.py`)
   - Now tracks count of tickers from active waves only
   - Uses `collect_all_required_tickers()` with `active_only=True` 
   - Excludes optional benchmarks and safe assets from health checks
   - Returns `active_wave_ticker_count` in health status dict

4. **Updated Data Health Panel UI** (`helpers/data_health_panel.py`)
   - Displays count of active wave tickers being monitored
   - Shows informational message that health is based on active waves only
   - Prevents confusion about which tickers affect system health

### Wave Registry Structure

The `data/wave_registry.csv` file contains an `active` column (Boolean) that indicates whether a wave is currently active:

```csv
wave_id,wave_name,active,...
ai_cloud_megacap_wave,AI & Cloud MegaCap Wave,True,...
russell_3000_wave,Russell 3000 Wave,False,...
```

### Example: Russell 3000 Wave

The Russell 3000 Wave (`wave_id: russell_3000_wave`) is currently inactive (`active=False`). Its ticker `IWV` will:
- ✅ **Be excluded** when `active_only=True` or `active_waves_only=True`
- ❌ **Not affect** system health status
- ✅ **Still be included** in default data fetches for backward compatibility

### Backward Compatibility

All parameters default to maintain existing behavior:
- `collect_all_required_tickers(active_only=False)` - includes all waves by default
- `get_wave_holdings_tickers(active_waves_only=True)` - filters by default (safer for health)
- `get_global_price_cache(active_only=False)` - includes all waves by default

Only system health monitoring code should use `active_only=True` to filter.

## Testing

New test file: `test_active_wave_filtering.py`

Tests verify:
1. ✅ Wave registry CSV has active/inactive waves
2. ✅ `collect_all_required_tickers(active_only=True)` excludes inactive wave tickers
3. ✅ `get_wave_holdings_tickers(active_waves_only=True)` filters correctly
4. ✅ IWV (from inactive russell_3000_wave) is properly excluded

Test results:
```
Active waves: 27
Inactive waves: 1
With active_only=True: 80 tickers (vs 100 without filtering)
✓ IWV correctly excluded when active filtering enabled
```

## Usage Examples

### For System Health Monitoring

```python
from data_cache import collect_all_required_tickers
from waves_engine import WAVE_WEIGHTS

# Get tickers from active waves only for health checks
active_tickers = collect_all_required_tickers(
    WAVE_WEIGHTS,
    include_benchmarks=False,  # Don't include optional benchmarks
    include_safe_assets=False,  # Don't include optional safe assets
    active_only=True  # Only active waves
)

# Check health based on these tickers only
# Failures/staleness of tickers NOT in this list won't affect health
```

### For Data Fetching (Backward Compatible)

```python
from data_cache import get_global_price_cache
from waves_engine import WAVE_WEIGHTS

# Default: fetch all tickers (active + inactive) for full coverage
cache = get_global_price_cache(
    wave_registry=WAVE_WEIGHTS,
    days=365
)
```

## Benefits

1. **Accurate Health Monitoring**: System health reflects actual operational status, not inactive wave issues
2. **Reduced False Positives**: Inactive wave ticker failures don't trigger degraded status
3. **Clear Separation**: Active vs inactive waves clearly differentiated
4. **Backward Compatible**: Existing code continues to work without changes
5. **Configurable**: Can enable/disable filtering per use case

## Future Enhancements

Potential improvements:
1. Add staleness checking directly in `get_ticker_health_status()` for active wave tickers
2. Track health metrics per wave (not just overall)
3. Support for wave categories (crypto, equity, etc.) in filtering
4. Dashboard showing active vs inactive wave status

## Implementation Files Modified

- `data_cache.py` - Added `active_only` parameter to ticker collection
- `helpers/ticker_sources.py` - Added `active_waves_only` parameter and enhanced health status
- `helpers/data_health_panel.py` - Updated UI to show active wave ticker count
- `test_active_wave_filtering.py` - New comprehensive tests

## References

- Wave Registry: `data/wave_registry.csv`
- Universal Universe: `universal_universe.csv`
- System Health Panel: Sidebar → "📊 Data Health Status"
