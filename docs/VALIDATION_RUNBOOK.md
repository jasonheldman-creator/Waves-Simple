# Validation Runbook: Institutional Readiness Tab

## Overview

This runbook provides step-by-step instructions for running the Institutional Readiness validation script, interpreting results, and troubleshooting common issues.

**Script Location**: `validation/validate_institutional_readiness.py`
**Purpose**: Verify Institutional Readiness tab implementation without running Streamlit
**Dependencies**: Python 3.7+ (standard library only)

---

## Quick Start

### Running the Validation

```bash
# From repository root
cd /path/to/Waves-Simple

# Run validation script
python3 validation/validate_institutional_readiness.py
```

### Expected Success Output

```
======================================================================
INSTITUTIONAL READINESS TAB VALIDATION
======================================================================

This validation script verifies the Institutional Readiness tab
implementation without running Streamlit or loading UI dependencies.

Validation Scope:
  - Tab label consistency
  - Function structure
  - Tab ordering (first position)
  - Executive language patterns
  - Diagnostics collapse state
  - Threshold constants

[1/6] Validating tab label consistency...
  ✓ Tab label 'Institutional Readiness' found 6 times

[2/6] Validating function existence and structure...
  ✓ Function 'render_overview_clean_tab()' found
  ✓ Function docstring contains 'Institutional Readiness - Tab 1'

[3/6] Validating tab ordering...
  ✓ 'Institutional Readiness' is first tab in 3 layouts

[4/6] Validating executive language patterns...
  ✓ Executive Header section found
  ✓ System Status section found
  ✓ Executive Summary section found
  ✓ Platform Signals section found
  ✓ AI Recommendations section found
  ✓ Top Performers section found
  ✓ Market Context section found

[5/6] Validating diagnostics collapse state...
  ✓ System Diagnostics in collapsed expander (expanded=False)

[6/6] Validating threshold constants...
  ✓ Constant 'CONFIDENCE_HIGH_COVERAGE_PCT' defined
  ✓ Constant 'CONFIDENCE_MODERATE_COVERAGE_PCT' defined
  ✓ Constant 'RISK_REGIME_VIX_LOW' defined
  ✓ Constant 'RISK_REGIME_VIX_HIGH' defined
  ✓ Constant 'RISK_REGIME_PERF_RISK_ON' defined
  ✓ Constant 'RISK_REGIME_PERF_RISK_OFF' defined
  ✓ Constant 'ALPHA_QUALITY_STRONG_RETURN' defined
  ✓ Constant 'ALPHA_QUALITY_STRONG_RATIO' defined
  ✓ Constant 'ALPHA_QUALITY_MIXED_RATIO' defined
  ✓ Constant 'POSTURE_STRONG_POSITIVE' defined
  ✓ Constant 'POSTURE_WEAK_NEGATIVE' defined
  ✓ Constant 'DISPERSION_HIGH' defined
  ✓ Constant 'DISPERSION_LOW' defined
  ✓ Constant 'DATA_INTEGRITY_VERIFIED_COVERAGE' defined
  ✓ Constant 'DATA_INTEGRITY_DEGRADED_COVERAGE' defined
  ✓ Constant 'DEFAULT_ALPHA_QUALITY' defined
  ✓ Constant 'DEFAULT_RISK_REGIME' defined
  ✓ Constant 'DEFAULT_DATA_INTEGRITY' defined
  ✓ Constant 'DEFAULT_CONFIDENCE' defined

======================================================================
VALIDATION SUMMARY
======================================================================
✓ PASS     Tab Label Consistency
✓ PASS     Function Structure
✓ PASS     Tab Ordering
✓ PASS     Executive Language
✓ PASS     Diagnostics Collapse
✓ PASS     Threshold Constants

Results: 6/6 checks passed

✓ All validations passed successfully!
```

**Exit Code**: `0` (success)

---

## Validation Checks Explained

### Check 1: Tab Label Consistency

**What it validates**: The string "Institutional Readiness" appears consistently as a tab label

**How it works**: Searches for `"Institutional Readiness"` in st.tabs() calls

**Success criteria**: Found at least 3 times (one per layout configuration)

**Why it matters**: Ensures users see the correct tab name across all app modes

### Check 2: Function Structure

**What it validates**: 
- Function `render_overview_clean_tab()` exists
- Function has correct docstring

**How it works**: Regex search for function definition and docstring content

**Success criteria**:
- Pattern `def render_overview_clean_tab():` found
- Docstring contains "Institutional Readiness - Tab 1"

**Why it matters**: Confirms the core rendering function is properly defined

### Check 3: Tab Ordering

**What it validates**: "Institutional Readiness" is the first tab in all layouts

**How it works**: Extracts all st.tabs() array definitions and checks first element

**Success criteria**: First position in at least 3 layout configurations

**Why it matters**: Executives see the Institutional Readiness view immediately upon app load

### Check 4: Executive Language Patterns

**What it validates**: All required executive-friendly sections are present

**How it works**: Pattern matching for section headers within function body

**Success criteria**: All 7 sections found:
1. Executive Header (🏛️ Institutional Readiness)
2. System Status (Composite System Control Status)
3. Executive Summary (Executive Intelligence Summary)
4. Platform Signals (Platform Intelligence Signals)
5. AI Recommendations (AI Recommendations)
6. Top Performers (Top Performing Strategies)
7. Market Context (Market Context)

**Why it matters**: Ensures complete executive interface with all decision-support sections

### Check 5: Diagnostics Collapse State

**What it validates**: System diagnostics are in a collapsed expander

**How it works**: Searches for `st.expander()` call with `expanded=False`

**Success criteria**: Found expander with "System Diagnostics" and `expanded=False`

**Why it matters**: Technical details shouldn't clutter the executive view by default

### Check 6: Threshold Constants

**What it validates**: All 19 AI briefing threshold constants are defined

**How it works**: Regex search for each constant definition

**Success criteria**: All constants found in app.py

**Why it matters**: Ensures AI briefing logic has proper configuration values

---

## Exit Codes

| Code | Meaning | Action Required |
|------|---------|-----------------|
| 0 | All validations passed | None - proceed with deployment |
| 1 | One or more validations failed | Review failure details and fix issues |

---

## Common Issues and Troubleshooting

### Issue: "Function not found"

**Symptom**:
```
[2/6] Validating function existence and structure...
  ✗ Function 'render_overview_clean_tab()' not found
```

**Possible Causes**:
1. Function was renamed or deleted
2. app.py file is corrupted or missing
3. Function signature changed (e.g., added parameters)

**Resolution**:
1. Check that app.py exists in repository root
2. Search for "render_overview_clean_tab" in app.py
3. If renamed, update validation script or restore original name
4. If deleted, restore from PR #361

### Issue: "Tab label found only X times"

**Symptom**:
```
[1/6] Validating tab label consistency...
  ✗ Tab label found only 2 times (expected at least 3)
```

**Possible Causes**:
1. Tab label was changed or removed from one layout
2. Layout configuration was modified

**Resolution**:
1. Search app.py for all instances of `st.tabs(`
2. Ensure "Institutional Readiness" is in each tabs array
3. Check that all 3 layout modes (safe mode, wave profile, original) exist

### Issue: "Section missing"

**Symptom**:
```
[4/6] Validating executive language patterns...
  ✓ Executive Header section found
  ✗ AI Recommendations section missing
```

**Possible Causes**:
1. Section header text was modified
2. Section was removed from function
3. Typo in section header

**Resolution**:
1. Check that all required sections exist in `render_overview_clean_tab()`
2. Verify exact text of section headers matches expected patterns
3. Common issue: Emoji or punctuation changes break pattern matching

### Issue: "Diagnostics expander missing or not collapsed"

**Symptom**:
```
[5/6] Validating diagnostics collapse state...
  ✗ System Diagnostics expander missing or not collapsed
```

**Possible Causes**:
1. Expander was removed
2. `expanded=False` was changed to `expanded=True`
3. Expander text was modified

**Resolution**:
1. Search for "System Diagnostics" in `render_overview_clean_tab()`
2. Ensure it's wrapped in `st.expander(..., expanded=False)`
3. Verify exact pattern: `st.expander("🔧 System Diagnostics & Technical Details", expanded=False)`

### Issue: "Constant missing"

**Symptom**:
```
[6/6] Validating threshold constants...
  ✗ Constant 'CONFIDENCE_HIGH_COVERAGE_PCT' missing
```

**Possible Causes**:
1. Constant was renamed or deleted
2. Constant is defined but pattern doesn't match (e.g., wrong spacing)

**Resolution**:
1. Search app.py for the constant name
2. Ensure it follows pattern: `CONSTANT_NAME = value`
3. Check that it's defined in the global scope (not inside a function)
4. Verify no extra spaces: `CONSTANT=value` won't match

### Issue: "File not found" or "Permission denied"

**Symptom**:
```
[1/6] Validating tab label consistency...
  ✗ Error: [Errno 2] No such file or directory: 'app.py'
```

**Possible Causes**:
1. Script is not run from repository root
2. app.py was moved or renamed
3. File permissions issue

**Resolution**:
1. Ensure you're in repository root: `cd /path/to/Waves-Simple`
2. Verify app.py exists: `ls -l app.py`
3. Check permissions: `chmod +r app.py`

---

## Advanced Usage

### Running in CI/CD

The validation script is designed to run in automated environments:

```bash
# Example GitHub Actions step
- name: Validate Institutional Readiness
  run: |
    python3 validation/validate_institutional_readiness.py
```

**Key Features**:
- No interactive prompts
- Clean exit codes (0/1)
- Structured output for log parsing
- No external dependencies

### Verbose Mode (Manual Debugging)

For deeper investigation, you can add debug output:

```python
# Modify validation script temporarily
import traceback

try:
    # ... validation code ...
except Exception as e:
    print(f"  ✗ Error: {e}")
    traceback.print_exc()  # Add this line
    return False
```

### Partial Validation

To test individual checks during development:

```python
# Comment out checks in main()
results = []
results.append(("Tab Label Consistency", validate_tab_label()))
# results.append(("Function Structure", validate_function_exists()))  # Skip
# results.append(("Tab Ordering", validate_tab_ordering()))  # Skip
# ...
```

---

## Integration with Development Workflow

### Pre-Commit Hook

Add validation to Git pre-commit hook:

```bash
# .git/hooks/pre-commit
#!/bin/bash
python3 validation/validate_institutional_readiness.py
if [ $? -ne 0 ]; then
    echo "Institutional Readiness validation failed. Commit aborted."
    exit 1
fi
```

### Pull Request Checklist

Before merging changes to app.py:
1. ✓ Run validation script locally
2. ✓ Verify 6/6 checks pass
3. ✓ Review any changes to Institutional Readiness tab
4. ✓ Confirm CI validation passes

---

## Maintenance

### When to Update Validation Script

Update the validation script if:
1. **Tab label changes**: Update `validate_tab_label()` pattern
2. **Section headers change**: Update `validate_executive_language()` patterns
3. **New constants added**: Add to `validate_threshold_constants()` list
4. **Function renamed**: Update `validate_function_exists()` pattern

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-04 | Initial release with 6 validation checks |

---

## Support

### Troubleshooting Checklist

- [ ] Running from repository root directory?
- [ ] app.py file exists and is readable?
- [ ] Python 3.7+ installed?
- [ ] No recent changes to `render_overview_clean_tab()`?
- [ ] All threshold constants still defined in app.py?

### Additional Resources

- **Verification Documentation**: `docs/INSTITUTIONAL_READINESS_VERIFICATION.md`
- **Original Implementation**: PR #361
- **CI Workflow**: `.github/workflows/validate_institutional_readiness.yml`

---

## Appendix: Validation Script Source

The validation script uses standard Python libraries only:
- `ast`: Abstract syntax tree parsing
- `inspect`: Source code inspection
- `re`: Regular expression pattern matching
- `pathlib`: File path operations

**No external dependencies required** - runs on any Python 3.7+ installation.

---

## Conclusion

This runbook should enable any developer or auditor to:
1. Run the validation script successfully
2. Interpret the results correctly
3. Troubleshoot common issues independently
4. Maintain the validation framework over time

For questions or issues not covered here, refer to the main verification documentation or contact the platform team.
