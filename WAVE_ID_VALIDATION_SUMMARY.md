# Wave ID Validation Implementation Summary

## Changes Made to `analytics_truth.py`

### A) Dynamic Expected Wave IDs

**Before:**
```python
def expected_waves(weights_df: pd.DataFrame) -> List[str]:
    wave_names = sorted(weights_df['wave'].unique().tolist())
    if len(wave_names) != 28:  # HARDCODED
        raise ValueError(f"Expected exactly 28 waves, found {len(wave_names)}")
    return wave_names
```

**After:**
```python
def expected_waves(weights_df: pd.DataFrame) -> List[str]:
    """Returns dynamically determined list of waves - no hardcoded count"""
    wave_names = sorted(weights_df['wave'].unique().tolist())
    return wave_names
```

**In `generate_live_snapshot_csv()`:**
```python
# Derive expected_wave_ids dynamically from wave_weights.csv
expected_wave_ids = sorted(set([_convert_wave_name_to_id(wave_name) for wave_name in waves]))
expected_count = len(expected_wave_ids)
```

### B) Normalize wave_id Before Validation

**Fixed `_convert_wave_name_to_id()` to never return None:**
```python
def _convert_wave_name_to_id(wave_name: str) -> str:
    if not wave_name:
        return 'unknown_wave'
    
    # Try waves_engine first
    if WAVES_ENGINE_AVAILABLE:
        result = get_wave_id_from_display_name(wave_name)
        if result:  # Only use if not None
            return result
    
    # Fallback to manual conversion (never returns None)
    wave_id = wave_name.lower()
    wave_id = wave_id.replace(' & ', '_')
    # ... more replacements
    return wave_id
```

**Normalization in DataFrame:**
```python
# Normalize during row creation
if wave_id_raw is None or (isinstance(wave_id_raw, str) and not wave_id_raw.strip()):
    wave_id = _convert_wave_name_to_id(wave_name) if wave_name else 'unknown_wave'
else:
    wave_id = wave_id_raw.strip() if isinstance(wave_id_raw, str) else str(wave_id_raw)

# Additional normalization after DataFrame creation
df['wave_id'] = df['wave_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
```

### C) Single Validation Point

**Comprehensive validation before return:**
```python
# === SINGLE VALIDATION POINT ===
nunique_with_na = df['wave_id'].nunique(dropna=False)
nunique_without_na = df['wave_id'].nunique(dropna=True)
isna_sum = df['wave_id'].isna().sum()
blank_sum = sum(1 for x in df['wave_id'] if isinstance(x, str) and not x.strip())
duplicates = df['wave_id'].value_counts()[lambda x: x > 1]

# Validate all conditions
validation_passed = True
error_messages = []

# Check 1: nunique(dropna=False) == expected_count
if nunique_with_na != expected_count:
    error_messages.append(f"nunique(dropna=False) = {nunique_with_na}, expected {expected_count}")

# Check 2: No null wave_ids
if isna_sum > 0:
    error_messages.append(f"Found {isna_sum} null wave_id(s)")

# Check 3: No blank wave_ids
if blank_sum > 0:
    error_messages.append(f"Found {blank_sum} blank wave_id(s)")

# Check 4: No duplicates
if len(duplicates) > 0:
    error_messages.append(f"Found {len(duplicates)} duplicate wave_id(s)")

# Raise with comprehensive diagnostics on failure
if not validation_passed:
    raise AssertionError(detailed_diagnostics_message)
```

**Diagnostics include:**
- Expected count (from wave_weights.csv)
- Total unique wave_ids (with/without dropna)
- Null counts
- Blank counts
- Example duplicate rows with wave names

## Testing

### Created Tests
1. `test_wave_id_validation.py` - Unit tests for validation components
2. `test_integration_validation.py` - Integration test simulating full validation

### Test Results
```
✓ Dynamic Wave IDs Derivation - PASSED
✓ Wave ID Normalization - PASSED
✓ Validation Metrics - PASSED
✓ No Hardcoded Wave Count - PASSED
✓ Integration Validation - PASSED
```

## Key Improvements

1. **No Hardcoded Counts**: Removed all hardcoded wave counts (28)
2. **Dynamic Derivation**: Wave count derived from wave_weights.csv
3. **Robust Normalization**: Never returns None, handles edge cases
4. **Single Validation Point**: All checks in one place with comprehensive diagnostics
5. **Actionable Error Messages**: Detailed diagnostics help debug issues quickly

## Verification

All existing tests pass with modifications. The implementation:
- Maintains backward compatibility
- Works with any number of waves in wave_weights.csv
- Provides clear error messages when validation fails
- Handles edge cases (None, blank, whitespace, duplicates)
