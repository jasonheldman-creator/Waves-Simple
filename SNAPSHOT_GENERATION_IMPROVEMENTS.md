# Snapshot Generation Improvements

## Overview
This document describes the improvements made to the snapshot generation system to enforce required tickers, implement the new schema, and ensure wave_id validation.

## Changes Made

### 1. GitHub Actions Workflow (`.github/workflows/rebuild_snapshot.yml`)
- Modified permissions to `contents: write` to allow automatic commits
- Added conditional logic to only commit/push when `data/live_snapshot.csv` changes
- Workflow now uses `git diff --exit-code` to detect changes
- Added proper git configuration for automated commits

### 2. Analytics Truth Module (`analytics_truth.py`)

#### Required Tickers Validation
- Added `validate_required_tickers()` function
- Validates presence of: SPY, QQQ, IWM (if in universe)
- Validates at least one VIX proxy: ^VIX, VIXY, or VXX (if any in universe)
- Raises `AssertionError` with detailed diagnostics if validation fails

#### Enhanced Diagnostics
- Added comprehensive diagnostics output showing:
  - Count of OK vs NO DATA rows
  - List of waves with NO DATA status
  - Missing tickers for each failed wave
  - Required symbols check results
- Diagnostics are printed during workflow execution for visibility

#### BRK-B Ticker Handling
- Updated documentation to clarify BRK-B handling
- yfinance expects "BRK-B" format (not "BRK.B")
- No transformation needed - ticker should already be in correct format

#### wave_id Validation
- Existing validation ensures wave_id is non-empty (lines 592-654)
- Raises `AssertionError` if any wave_id is null or blank
- Validates unique count matches expected waves from wave_weights.csv

## New Schema
The snapshot now uses the following columns:
- wave_id: Canonical wave identifier (snake_case, validated non-empty)
- wave: Display name
- return_1d, return_30d, return_60d, return_365d: Returns for different periods
- status: 'OK' or 'NO DATA'
- coverage_pct: Percentage of tickers successfully fetched (0-100)
- missing_tickers: Comma-separated list of failed tickers
- tickers_ok: Count of successfully fetched tickers
- tickers_total: Total number of tickers for the wave
- asof_utc: ISO timestamp of snapshot generation
- mode: Operating mode (e.g., "STANDARD")
- date: Date in YYYY-MM-DD format

## Testing
The implementation was tested locally and correctly:
- Validates required tickers (raises error when SPY/QQQ missing)
- Displays diagnostics output
- Validates wave_id non-empty constraint
- Handles all tickers appropriately (including BRK-B)
