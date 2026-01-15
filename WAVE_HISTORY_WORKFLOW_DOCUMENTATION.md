# Wave History CI/CD Workflow Documentation

## Overview

This document describes the automated CI/CD workflow for building and validating `wave_history.csv`, which is a critical dependency for the Streamlit application.

## Problem Statement

Previously, `wave_history.csv` was not guaranteed to exist or be up-to-date in the repository. This led to:
- Silent fallbacks to stale or missing data in the Streamlit application
- Recent strategy stacking fixes appearing to have no impact in the UI
- Portfolio and wave metrics appearing "frozen" due to the runtime lacking up-to-date data

## Solution

A GitHub Actions workflow (`.github/workflows/build_wave_history.yml`) has been implemented to:
1. Automatically build `wave_history.csv` when relevant files change
2. Validate the generated file meets quality thresholds
3. Commit and push the file back to the main branch
4. Ensure the file is always up-to-date and correct

## Workflow Triggers

The workflow runs automatically on:
- **Push to main**: When any of these files change:
  - `wave_weights.csv`
  - `prices.csv`
  - `build_wave_history_from_prices.py`
  - `waves_engine.py`
  - `.github/workflows/build_wave_history.yml`
- **Manual dispatch**: Can be triggered manually via GitHub Actions UI
  - Supports a `force` parameter to rebuild even if no changes detected

## Workflow Steps

### 1. Environment Setup
- Checks out the repository
- Sets up Python 3.10 (matching other workflows)
- Installs dependencies from `requirements.txt`

### 2. Pre-build Inspection
- Logs current state of `wave_history.csv` if it exists
- Shows file size, row count
- Displays coverage snapshot if available

### 3. Build Wave History
- Runs `build_wave_history_from_prices.py` script
- This script:
  - Loads wave weights and price data
  - Computes returns for each wave
  - Applies full strategy pipeline (momentum, VIX overlays, regime detection, etc.)
  - Generates `wave_history.csv` with complete strategy-adjusted returns
  - Creates `wave_coverage_snapshot.json` with coverage metrics

### 4. Validation (Post-Build)

The workflow validates the generated files against multiple criteria:

#### File Existence and Size
- `wave_history.csv` must exist
- File must be non-empty (size > 0)

#### Row Count Threshold
- Minimum: 10,000 rows
- Ensures sufficient historical data

#### Wave Count Threshold
- Minimum: 5 unique waves
- Ensures portfolio diversity

#### Required Columns
- `date`: Date of the observation
- `wave`: Wave name
- `portfolio_return`: Wave return for that date
- `benchmark_return`: Benchmark return for that date

#### Data Integrity
- No all-NaN columns
- Valid date range (min_date < max_date)
- Date and wave columns must not have NaN values

#### Coverage Snapshot
- Validates `wave_coverage_snapshot.json` exists (optional)
- Checks metadata:
  - Total waves processed
  - Waves meeting coverage threshold
  - Waves below threshold
  - Missing ticker information

### 5. Change Detection
- Compares generated files with current repository state
- Detects if files are new or modified
- Handles both new builds and updates

### 6. Commit and Push
- Only runs on main branch for push/manual dispatch events
- Fetches latest changes from origin/main
- Rebases to ensure clean history
- Stages `wave_history.csv` and `wave_coverage_snapshot.json`
- Commits with appropriate message:
  - "Update wave_history.csv (auto)" for push events
  - "Update wave_history.csv (manual run)" for manual dispatch
- Pushes changes back to main branch

## Validation Thresholds

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Row count | ≥ 10,000 | Ensure sufficient historical data |
| Wave count | ≥ 5 | Ensure portfolio diversity |
| Coverage | ≥ 90% | Ensure wave weights are adequately covered by price data |

## Concurrency Control

The workflow uses concurrency control to prevent overlapping runs:
```yaml
concurrency:
  group: build-wave-history
  cancel-in-progress: true
```

This ensures only one build runs at a time, preventing conflicts during commits.

## Testing

A comprehensive test suite (`test_build_wave_history_workflow.py`) validates:
1. File exists and is non-empty
2. Row count meets threshold
3. Wave count meets threshold
4. Required columns are present
5. Date range is valid
6. Data integrity (no all-NaN columns)
7. Coverage snapshot is valid
8. Strategy overlay fields are present (optional)

Run tests with:
```bash
python3 test_build_wave_history_workflow.py
```

## Expected Outcomes

1. **Availability Guarantee**: `wave_history.csv` will always exist and be up-to-date on main branch
2. **Improved Transparency**: Streamlit application will accurately reflect updates to:
   - Strategy stacking
   - VIX/momentum overlays
   - Alpha calculations
   - Regime detection
3. **Data Integrity**: File is validated via CI, eliminating silent fallbacks or stale data usage

## Monitoring

Check workflow status:
- Navigate to: https://github.com/jasonheldman-creator/Waves-Simple/actions
- Look for "Build Wave History" workflow runs

Workflow logs include:
- Pre-build file state
- Build progress (wave processing)
- Post-build validation results
- Coverage metrics
- Commit metadata

## Troubleshooting

### Workflow Fails: File Not Found
- Ensure `wave_weights.csv` and `prices.csv` exist in repository
- Check that `build_wave_history_from_prices.py` is present

### Workflow Fails: Row Count Below Threshold
- Check if price data is sufficient (date range, tickers)
- Review `wave_coverage_snapshot.json` for missing tickers
- Adjust `MIN_ROW_THRESHOLD` if needed (in workflow file)

### Workflow Fails: Wave Count Below Threshold
- Check `wave_weights.csv` for valid wave definitions
- Ensure benchmarks are mapped in `BENCHMARK_BY_WAVE` (in build script)
- Review coverage snapshot for waves below threshold

### Workflow Fails: Validation Errors
- Check workflow logs for specific error messages
- Run `python3 build_wave_history_from_prices.py` locally to reproduce
- Run `python3 test_build_wave_history_workflow.py` to validate

### Workflow Succeeds but No Commit
- This is normal if no changes detected after rebase
- Another workflow may have already updated the files
- For manual runs, a warning message will be logged

## Manual Workflow Dispatch

To manually trigger the workflow:
1. Navigate to: https://github.com/jasonheldman-creator/Waves-Simple/actions
2. Select "Build Wave History" workflow
3. Click "Run workflow"
4. Select branch (usually `main`)
5. Optionally check "Force rebuild" to rebuild even if no changes
6. Click "Run workflow"

## Files Modified by Workflow

- `wave_history.csv`: Main output file with wave returns
- `wave_coverage_snapshot.json`: Coverage metrics and diagnostics

## Permissions

The workflow requires `contents: write` permission to commit and push changes back to the repository.

## Integration with Streamlit App

The Streamlit application (`app.py`) uses `wave_history.csv` to:
- Display portfolio metrics
- Calculate wave performance
- Show alpha attribution
- Render wave diagnostics
- Display strategy overlay effects

With this workflow, the app will always have access to current, validated data.
