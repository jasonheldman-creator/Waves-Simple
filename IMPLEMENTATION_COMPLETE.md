# Wave ID Validation Implementation - COMPLETE ✅

## Problem Statement Summary

The task was to update `analytics_truth.py` → `generate_live_snapshot_csv()` with three key improvements:

### A) Dynamic Expected Wave IDs
Remove hardcoded expected counts (e.g., 28) and derive them dynamically from `wave_weights.csv`

### B) Normalize wave_id Before Validation
Treat None, NaN, blank/whitespace as invalid and normalize the wave_id column

### C) Single Validation Point
Add comprehensive validation with detailed diagnostics before the return statement

## Implementation Summary

### Files Modified
1. **analytics_truth.py** (2 functions modified)
   - `expected_waves()` - Removed hardcoded count validation
   - `generate_live_snapshot_csv()` - Added dynamic derivation and comprehensive validation
   - `_convert_wave_name_to_id()` - Fixed to never return None

2. **New Test Files**
   - `test_wave_id_validation.py` - 4 unit tests
   - `test_integration_validation.py` - Integration test
   - `WAVE_ID_VALIDATION_SUMMARY.md` - Technical documentation

### Key Changes

#### 1. Dynamic Expected Wave IDs (Requirement A)
**Before:**
```python
def expected_waves(weights_df: pd.DataFrame) -> List[str]:
    wave_names = sorted(weights_df['wave'].unique().tolist())
    if len(wave_names) != 28:  # ❌ HARDCODED
        raise ValueError(f"Expected exactly 28 waves...")
    return wave_names
```

**After:**
```python
def expected_waves(weights_df: pd.DataFrame) -> List[str]:
    """Dynamically determined - no hardcoded count"""
    wave_names = sorted(weights_df['wave'].unique().tolist())
    return wave_names  # ✅ Works with any count

# In generate_live_snapshot_csv():
expected_wave_ids = sorted(set([_convert_wave_name_to_id(w) for w in waves]))
expected_count = len(expected_wave_ids)  # ✅ Derived dynamically
```

#### 2. Wave ID Normalization (Requirement B)
**Before:**
```python
def _convert_wave_name_to_id(wave_name: str) -> str:
    # Could return None from waves_engine
    return get_wave_id_from_display_name(wave_name)  # ❌ May be None
```

**After:**
```python
def _convert_wave_name_to_id(wave_name: str) -> str:
    """Never returns None - guaranteed non-None result"""
    if not wave_name:
        return 'unknown_wave'
    
    # Try waves_engine first
    if WAVES_ENGINE_AVAILABLE:
        result = get_wave_id_from_display_name(wave_name)
        if result:  # ✅ Only use if not None
            return result
    
    # ✅ Fallback always returns a valid string
    wave_id = wave_name.lower()
    wave_id = wave_id.replace(' & ', '_')
    # ... more replacements
    return wave_id
```

**Normalization in DataFrame:**
```python
# Get wave_id and strip whitespace
wave_id = _convert_wave_name_to_id(wave_name)
if isinstance(wave_id, str):
    wave_id = wave_id.strip()

# Additional normalization after DataFrame creation
df['wave_id'] = df['wave_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
```

#### 3. Single Validation Point (Requirement C)
**Before:**
```python
# Multiple scattered assertions
if len(df) != 28:
    raise AssertionError(f"Expected exactly 28 rows...")
    
if df['wave_id'].nunique() != 28:
    raise AssertionError(f"Expected 28 unique wave_ids...")
```

**After:**
```python
# === SINGLE VALIDATION POINT ===
# Helper function to avoid duplication
def is_blank_wave_id(x):
    return isinstance(x, str) and not x.strip()

# Compute all metrics
nunique_with_na = df['wave_id'].nunique(dropna=False)
nunique_without_na = df['wave_id'].nunique(dropna=True)
isna_sum = df['wave_id'].isna().sum()
blank_sum = sum(1 for x in df['wave_id'] if is_blank_wave_id(x))
duplicates = df['wave_id'].value_counts()[lambda x: x > 1]

# Validate all conditions
validation_passed = True
error_messages = []

# ✅ Check 1: nunique(dropna=False) == expected_count
if nunique_with_na != expected_count:
    error_messages.append(...)

# ✅ Check 2: No null wave_ids
if isna_sum > 0:
    error_messages.append(...)

# ✅ Check 3: No blank wave_ids
if blank_sum > 0:
    error_messages.append(...)

# ✅ Check 4: No duplicates
if len(duplicates) > 0:
    error_messages.append(...)

# ✅ Raise with comprehensive diagnostics
if not validation_passed:
    diagnostics = [
        "=" * 80,
        "WAVE_ID VALIDATION FAILED",
        "=" * 80,
        f"Expected count: {expected_count}",
        f"Unique (dropna=True): {nunique_without_na}",
        f"Unique (dropna=False): {nunique_with_na}",
        f"Null count: {isna_sum}",
        f"Blank count: {blank_sum}",
        # ... example rows for debugging
    ]
    raise AssertionError("\n".join(diagnostics))
```

### Validation Diagnostics
When validation fails, the error message includes:
- ✅ Expected count (from wave_weights.csv)
- ✅ Actual counts (with and without dropna)
- ✅ Null counts with example rows
- ✅ Blank counts with example rows
- ✅ Duplicate wave_ids with corresponding wave names

### Test Coverage

#### Unit Tests (test_wave_id_validation.py)
1. ✅ `test_dynamic_wave_ids()` - Verifies dynamic derivation
2. ✅ `test_wave_id_normalization()` - Verifies normalization rules
3. ✅ `test_validation_metrics()` - Verifies metric calculations
4. ✅ `test_expected_waves_no_hardcoded_count()` - Verifies no hardcoding

#### Integration Test (test_integration_validation.py)
1. ✅ `test_validation_logic_simulation()` - Full validation flow

#### Test Results
```
================================================================================
TEST SUMMARY
================================================================================
Total tests: 5 (4 unit + 1 integration)
Passed: 5 ✅
Failed: 0
================================================================================
```

### Code Quality
- ✅ No hardcoded counts (28 removed)
- ✅ No infinite recursion risk
- ✅ No code duplication (helper function extracted)
- ✅ Clean, readable code
- ✅ Comprehensive error messages
- ✅ All code review feedback addressed

### Backward Compatibility
- ✅ Existing tests still pass
- ✅ Function signature unchanged
- ✅ Return type unchanged
- ✅ Works with current wave_weights.csv (28 waves)
- ✅ Will work with any number of waves

## Verification Checklist

- [x] Requirement A: Dynamic expected wave IDs
  - [x] No hardcoded counts
  - [x] Derived from wave_weights.csv
  - [x] Uses sorted(set(...))
  - [x] Calculates expected_count dynamically

- [x] Requirement B: Normalize wave_id
  - [x] Never returns None
  - [x] Strips whitespace
  - [x] Handles blank/None with fallback
  - [x] Deterministic slugification

- [x] Requirement C: Single validation point
  - [x] Single validation block before return
  - [x] Checks nunique(dropna=False) == expected_count
  - [x] Checks isna().sum() == 0
  - [x] Checks blank count == 0
  - [x] Checks no duplicates
  - [x] Comprehensive diagnostics on failure

- [x] Code Quality
  - [x] All tests passing
  - [x] Code review feedback addressed
  - [x] No infinite recursion
  - [x] No code duplication
  - [x] Clean, maintainable code

## Conclusion

All requirements have been successfully implemented and tested. The code is:
- ✅ Correct (all tests pass)
- ✅ Maintainable (clean, well-documented)
- ✅ Robust (handles edge cases)
- ✅ Flexible (works with any wave count)
- ✅ Debuggable (comprehensive error messages)

**Implementation Status: COMPLETE** 🎉
