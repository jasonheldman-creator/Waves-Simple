# Validation Scripts

This directory contains validation scripts for verifying implementation integrity without running the full application.

## Scripts

### `validate_institutional_readiness.py`

**Purpose**: Verify the Institutional Readiness tab implementation (PR #361) using headless validation.

**Type**: Standalone Python script (no Streamlit or UI dependencies)

**Dependencies**: Python 3.7+ standard library only
- `ast` - Abstract syntax tree parsing
- `inspect` - Source code inspection  
- `re` - Regular expression matching
- `pathlib` - Path operations

**Usage**:
```bash
# From repository root
python3 validation/validate_institutional_readiness.py
```

**Exit Codes**:
- `0` - All validations passed
- `1` - One or more validations failed

**Validation Checks** (6 total):
1. Tab label consistency
2. Function structure
3. Tab ordering (first position)
4. Executive language patterns
5. Diagnostics collapse state
6. Threshold constants

**Documentation**:
- Verification Details: `docs/INSTITUTIONAL_READINESS_VERIFICATION.md`
- Runbook: `docs/VALIDATION_RUNBOOK.md`
- Results: `VALIDATION_RESULTS.md`

**CI Integration**: 
- Workflow: `.github/workflows/validate_institutional_readiness.yml`
- Runs automatically on push/PR to main branches
- Posts results as PR comment

## Adding New Validation Scripts

Follow this pattern for new validation scripts:

1. **No UI Dependencies**: Use source inspection, not runtime execution
2. **Clear Exit Codes**: 0 for success, 1 for failure
3. **Structured Output**: Clear headers and check-by-check reporting
4. **Standard Library**: Avoid external dependencies when possible
5. **Documentation**: Add runbook and verification docs
6. **CI Integration**: Create GitHub Actions workflow

## Why Headless Validation?

Headless validation scripts provide several advantages:

- **Fast**: No need to start Streamlit or load UI framework
- **Reliable**: Independent of runtime environment
- **CI-Friendly**: Easy to integrate into automated workflows
- **Audit-Ready**: Clear, reproducible verification logs
- **Portable**: Run anywhere Python is available

## Maintenance

When updating `app.py` or other validated code:

1. Run validation scripts locally before committing
2. Update validation scripts if implementation patterns change
3. Keep documentation in sync with validation logic
4. Review CI workflow results on PRs
