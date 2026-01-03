# App.py Protection Documentation

## Overview

This document describes the protection mechanisms implemented to prevent accidental erasure or major modification of the critical `app.py` file.

## Current Status

### Investigation Results (January 3, 2026)

After thorough investigation of the repository history:

1. **app.py is INTACT**: The file currently has **19,661 lines** and contains all expected functionality
2. **No restoration needed**: PR #336 (which was mentioned as potentially destructive) actually ADDED the full app.py file with all tabs
3. **All tabs present**: Verified presence of 17+ tabs including:
   - Overview (Clean)
   - Overview
   - Console/Executive
   - Details
   - Reports
   - Overlays
   - Attribution
   - Board Pack
   - IC Pack
   - Alpha Capture
   - Wave Monitor
   - Plan B Monitor
   - Wave Intelligence (Plan B)
   - Governance & Audit
   - Diagnostics
   - Wave Overview (New)

### Git History Analysis

- **Before PR #336** (commit f90401e): app.py had 19,650 lines
- **After PR #336** (commit 2e3ded5): app.py has 19,661 lines
- **Change**: +11 lines (minor enhancement, not a destructive change)

## Protection Mechanism

### GitHub Actions Workflow

A new CI workflow (`.github/workflows/app-py-protection.yml`) has been implemented with the following checks:

#### 1. Python Syntax Validation
```bash
python -m py_compile app.py
```
- Ensures the file has no syntax errors
- Runs before any other checks

#### 2. Line Count Regression Detection
- **Threshold**: Fails if app.py shrinks by more than 20%
- **Current baseline**: 19,661 lines on main branch
- **Minimum allowed**: 15,728 lines (80% of baseline)
- **Calculation**: Compares PR branch against main branch

This check prevents scenarios where:
- The file is accidentally replaced with a minimal scaffold
- Large sections of code are deleted
- Tabs or major functionality are removed

#### 3. Critical Tab Labels Verification
Validates presence of 14 essential tab labels:
- Overview
- Console
- Executive
- Details
- Reports
- Overlays
- Attribution
- Board Pack
- IC Pack
- Alpha Capture
- Wave Monitor
- Plan B Monitor
- Governance
- Diagnostics

#### 4. Critical Render Functions Verification
Validates presence of 8 essential render functions:
- `def render_executive_tab`
- `def render_overview_tab`
- `def render_details_tab`
- `def render_reports_tab`
- `def render_overlays_tab`
- `def render_attribution_tab`
- `def render_diagnostics_tab`
- `def main`

### Workflow Triggers

The protection workflow runs on:
- **Pull Requests** to `main` branch that modify `app.py`
- **Direct pushes** to `main` branch that modify `app.py`

### When Checks Fail

If any check fails, the CI workflow will:
1. Display detailed error messages
2. Block the PR from being merged (if using branch protection rules)
3. Show exactly what is missing or incorrect

## Validation Plan Completed

✅ **Python Syntax**: Validated with `python -m py_compile app.py`
✅ **Tab Rendering**: All 14 critical tab labels verified
✅ **Function Presence**: All 8 critical render functions verified
✅ **Line Count**: Current file at 19,661 lines (healthy)

## Future Recommendations

1. **Enable Branch Protection**: Configure GitHub branch protection rules to require the "App.py Protection" workflow to pass before merging
2. **Regular Monitoring**: Periodically review the workflow runs to ensure it's catching potential issues
3. **Update Thresholds**: If legitimate major refactoring is needed, the 20% threshold can be adjusted in the workflow file
4. **Add More Checks**: Consider adding checks for:
   - Import statements for critical modules
   - Streamlit components usage
   - Configuration constants

## Testing the Protection

To test the protection mechanism locally:

```bash
# Test 1: Syntax validation
python -m py_compile app.py

# Test 2: Line count check
CURRENT_LINES=$(wc -l < app.py)
MAIN_LINES=$(git show main:app.py | wc -l)
THRESHOLD=$((MAIN_LINES * 80 / 100))
echo "Current: $CURRENT_LINES, Main: $MAIN_LINES, Threshold: $THRESHOLD"

# Test 3: Tab labels check
grep -E "(Overview|Console|Details|Reports)" app.py

# Test 4: Render functions check
grep "def render_.*_tab" app.py
```

## Maintenance

The workflow file should be reviewed and updated if:
- New critical tabs are added to the application
- Tab names are changed
- Major refactoring requires temporary line count reduction
- New critical functions are introduced

## Contact

For questions about this protection mechanism, refer to:
- Workflow file: `.github/workflows/app-py-protection.yml`
- This documentation: `APP_PROTECTION_DOCUMENTATION.md`
