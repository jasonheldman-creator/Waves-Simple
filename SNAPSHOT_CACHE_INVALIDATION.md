# Portfolio Snapshot Cache Invalidation

## Overview

This document describes the snapshot cache invalidation mechanism implemented to ensure portfolio-level results are recalculated when engine logic changes.

## Problem Addressed

Prior to this enhancement, the S&P 500 Wave was successfully promoted to use the full strategy pipeline (including VIX overlay, momentum logic, and exposure adjustments). However, portfolio-level results in `data/live_snapshot.csv` were cached and not automatically recalculated, leading to stale alpha and return metrics.

## Solution

The system now tracks the engine version and automatically invalidates cached snapshots when the engine logic changes. This ensures that strategy updates (like the S&P 500 Wave promotion) immediately trigger recalculation of portfolio metrics.

## Key Components

### 1. Engine Version Tracking (`waves_engine.py`)

**New constant:**
```python
ENGINE_VERSION = "17.3"
```

**New function:**
```python
def get_engine_version() -> str:
    """
    Get the current engine version.
    
    This version is used for snapshot cache invalidation. When the engine
    logic changes (e.g., S&P 500 Wave promotion to full strategy pipeline),
    incrementing this version will force recalculation of cached snapshots.
    
    Returns:
        Engine version string (e.g., "17.3")
    """
    return ENGINE_VERSION
```

### 2. Metadata Tracking (`governance_metadata.py`)

The `create_snapshot_metadata()` function now includes `engine_version` in the metadata:

```python
metadata = {
    # ... other fields ...
    'engine_version': engine_version,  # NEW: Track engine version for cache invalidation
    # ... other fields ...
}
```

### 3. Cache Invalidation (`snapshot_ledger.py`)

The `generate_snapshot()` function now checks the engine version before using cached snapshots:

```python
# Get engine version from cached metadata
cached_engine_version = metadata.get('engine_version', 'unknown')
current_engine_version = get_engine_version()

# Cache is valid if:
# 1. Age is within threshold AND
# 2. Engine version matches current version
if age_hours < MAX_SNAPSHOT_AGE_HOURS and cached_engine_version == current_engine_version:
    cache_valid = True
    return cached_df
elif cached_engine_version != current_engine_version:
    print(f"⚠ Cache invalidated: engine version changed from {cached_engine_version} to {current_engine_version}")
```

## Usage

### Automatic Invalidation

When you update engine logic that affects portfolio calculations:

1. **Increment the ENGINE_VERSION** in `waves_engine.py`:
   ```python
   ENGINE_VERSION = "17.4"  # Increment version
   ```

2. **Next snapshot generation will automatically invalidate cache** and recalculate all metrics

### Manual Force Rebuild

#### Option 1: Environment Variable
```bash
export FORCE_SNAPSHOT_REBUILD=1
python app.py
```

#### Option 2: UI Operator Controls
- Navigate to the **Executive Brief** tab
- Click the **"🔄 Force Refresh"** button
- Snapshot will be regenerated with `generation_reason='manual'`

#### Option 3: Mission Control (Plan B Diagnostics)
- Navigate to **Mission Control** → **Plan B Diagnostics**
- Click **"🔄 Generate Snapshot Now"**
- Snapshot will be regenerated with `generation_reason='manual'`

#### Option 4: Programmatic API
```python
from snapshot_ledger import generate_snapshot

# Force regeneration
snapshot_df = generate_snapshot(
    force_refresh=True,
    generation_reason='manual'
)
```

## Generation Reasons

The system tracks why each snapshot was generated:

| Reason | Description |
|--------|-------------|
| `auto` | Automatic generation (cache miss or stale) |
| `manual` | User-initiated via UI button |
| `version_change` | Engine version changed, cache invalidated |
| `env_force_rebuild` | FORCE_SNAPSHOT_REBUILD environment variable set |
| `fallback` | Generated as fallback after errors |

## Metadata Schema

The snapshot metadata file (`data/snapshot_metadata.json`) now includes:

```json
{
  "snapshot_id": "uuid-...",
  "snapshot_hash": "abc123...",
  "timestamp": "2026-01-12T08:30:00.000000",
  "generation_reason": "version_change",
  
  "software_version": "git-commit-hash",
  "git_branch": "main",
  "registry_version": "v20260112_083000",
  "benchmark_version": "v20260112_083000",
  "engine_version": "17.3",  // NEW: Engine version for cache invalidation
  
  "data_regime": "LIVE",
  "safe_mode": false,
  "wave_count": 28,
  "degraded_wave_count": 0,
  "broken_ticker_count": 0
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_snapshot_cache_invalidation.py
```

Test coverage includes:
- ✅ Engine version tracking
- ✅ Metadata includes engine version
- ✅ Cache invalidation on version change
- ✅ FORCE_SNAPSHOT_REBUILD environment variable
- ✅ Generation reason tracking

## Examples

### Example 1: S&P 500 Wave Strategy Update

**Before:**
- S&P 500 Wave uses shortcut computation
- Cached snapshot shows old metrics
- ENGINE_VERSION = "17.2"

**After:**
- S&P 500 Wave promoted to full strategy pipeline
- Increment ENGINE_VERSION to "17.3"
- Next load detects version mismatch
- Cache invalidated automatically
- Snapshot regenerated with new strategy logic
- New alpha and returns calculated correctly

### Example 2: VIX Overlay Parameter Update

**Steps:**
1. Update VIX overlay parameters in `config/vix_overlay_config.py`
2. Increment `ENGINE_VERSION = "17.4"` in `waves_engine.py`
3. Restart application or regenerate snapshot
4. System detects version change: `17.3 → 17.4`
5. Cache invalidated with message: `⚠ Cache invalidated: engine version changed from 17.3 to 17.4`
6. All waves recalculated with new VIX overlay logic

## Files Modified

| File | Changes |
|------|---------|
| `waves_engine.py` | Added ENGINE_VERSION constant and get_engine_version() function |
| `snapshot_ledger.py` | Added version checking in generate_snapshot(), FORCE_SNAPSHOT_REBUILD env var support |
| `governance_metadata.py` | Added engine_version to metadata tracking |
| `test_snapshot_cache_invalidation.py` | Comprehensive test suite for cache invalidation |

## Backward Compatibility

✅ **Fully backward compatible**

- Old snapshots without `engine_version` in metadata still work
- Falls back to time-based caching only (24-hour TTL)
- No breaking changes to existing APIs
- Graceful handling of missing metadata files

## Best Practices

1. **Always increment ENGINE_VERSION** when making changes that affect:
   - Strategy pipeline logic
   - Return calculations
   - Alpha computations
   - Exposure adjustments
   - VIX overlay parameters
   - Benchmark configurations

2. **Use semantic versioning** for ENGINE_VERSION:
   - Major version: Breaking changes to snapshot schema
   - Minor version: New features or strategy updates
   - Patch version: Bug fixes that don't affect calculations

3. **Document version changes** in the engine header comments

4. **Test before deployment** using `test_snapshot_cache_invalidation.py`

5. **Monitor invalidation** in application logs:
   ```
   ⚠ Cache invalidated: engine version changed from 17.2 to 17.3
   ⏳ TruthFrame: Generating new snapshot from engine...
   ✓ TruthFrame: Generated 28 waves from engine
   ```

## Troubleshooting

### Issue: Snapshot not regenerating despite version change

**Solution:**
1. Check that `ENGINE_VERSION` was actually incremented
2. Verify `data/snapshot_metadata.json` exists and contains `engine_version`
3. Try manual force rebuild: `FORCE_SNAPSHOT_REBUILD=1 python app.py`

### Issue: "Cache invalidated" message but metrics still old

**Solution:**
1. Check if Safe Mode is enabled (prevents regeneration)
2. Verify `generate_snapshot()` is completing successfully
3. Check application logs for errors during snapshot generation

### Issue: Snapshot regenerating every time

**Solution:**
1. Check that ENGINE_VERSION is constant (not computed dynamically)
2. Verify metadata file is being saved correctly to `data/snapshot_metadata.json`
3. Check file permissions on data directory

## Related Documentation

- [SP500_WAVE_PROMOTION_SUMMARY.md](SP500_WAVE_PROMOTION_SUMMARY.md) - S&P 500 Wave strategy update that motivated this feature
- [WAVE_SNAPSHOT_LEDGER_DOCUMENTATION.md](WAVE_SNAPSHOT_LEDGER_DOCUMENTATION.md) - Snapshot ledger architecture
- [GOVERNANCE_METADATA.md](GOVERNANCE_METADATA.md) - Metadata and governance tracking

---

**Version:** 1.0  
**Date:** 2026-01-12  
**Status:** ✅ Implemented and Tested
