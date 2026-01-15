#!/usr/bin/env python3
"""
Test for build wave history workflow.

This test validates that the build_wave_history_from_prices.py script generates
wave_history.csv with all required fields and meets minimum quality thresholds.
"""

import sys
import os
import pandas as pd
import json

# Constants matching the workflow
MIN_ROW_THRESHOLD = 10000
MIN_WAVE_THRESHOLD = 5
REQUIRED_COLUMNS = ['date', 'wave', 'portfolio_return', 'benchmark_return']


def test_wave_history_file_exists():
    """Test that wave_history.csv exists."""
    print("\n" + "=" * 80)
    print("TEST: wave_history.csv file exists")
    print("=" * 80)
    
    wave_history_file = "wave_history.csv"
    
    assert os.path.exists(wave_history_file), f"wave_history.csv not found: {wave_history_file}"
    print(f"✓ wave_history.csv exists: {wave_history_file}")
    
    # Check file size
    file_size = os.path.getsize(wave_history_file)
    assert file_size > 0, "wave_history.csv is empty"
    print(f"✓ File size: {file_size:,} bytes")
    
    print("=" * 80)
    print("✓ PASSED: File exists and is non-empty\n")


def test_wave_history_row_count():
    """Test that wave_history.csv has sufficient rows."""
    print("\n" + "=" * 80)
    print("TEST: wave_history.csv row count threshold")
    print("=" * 80)
    
    df = pd.read_csv("wave_history.csv")
    row_count = len(df)
    
    print(f"Row count: {row_count:,}")
    print(f"Minimum threshold: {MIN_ROW_THRESHOLD:,}")
    
    assert row_count >= MIN_ROW_THRESHOLD, \
        f"Row count {row_count} is below minimum threshold {MIN_ROW_THRESHOLD}"
    
    print(f"✓ Row count meets threshold")
    print("=" * 80)
    print("✓ PASSED: Sufficient rows\n")


def test_wave_history_wave_count():
    """Test that wave_history.csv has sufficient unique waves."""
    print("\n" + "=" * 80)
    print("TEST: wave_history.csv wave count threshold")
    print("=" * 80)
    
    df = pd.read_csv("wave_history.csv")
    wave_count = df['wave'].nunique()
    
    print(f"Unique waves: {wave_count}")
    print(f"Minimum threshold: {MIN_WAVE_THRESHOLD}")
    
    assert wave_count >= MIN_WAVE_THRESHOLD, \
        f"Wave count {wave_count} is below minimum threshold {MIN_WAVE_THRESHOLD}"
    
    print(f"✓ Wave count meets threshold")
    
    # List waves
    waves = sorted(df['wave'].unique())
    print(f"\nWaves in wave_history.csv ({len(waves)}):")
    for i, wave in enumerate(waves, 1):
        wave_rows = len(df[df['wave'] == wave])
        print(f"  {i:2d}. {wave} ({wave_rows:,} rows)")
    
    print("=" * 80)
    print("✓ PASSED: Sufficient waves\n")


def test_wave_history_required_columns():
    """Test that wave_history.csv has all required columns."""
    print("\n" + "=" * 80)
    print("TEST: wave_history.csv required columns")
    print("=" * 80)
    
    df = pd.read_csv("wave_history.csv")
    
    print(f"Required columns: {REQUIRED_COLUMNS}")
    print(f"Actual columns: {list(df.columns)}")
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    
    print(f"✓ All required columns present")
    print("=" * 80)
    print("✓ PASSED: All required columns exist\n")


def test_wave_history_date_range():
    """Test that wave_history.csv has a valid date range."""
    print("\n" + "=" * 80)
    print("TEST: wave_history.csv date range")
    print("=" * 80)
    
    df = pd.read_csv("wave_history.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = (max_date - min_date).days
    
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Total days: {date_range:,}")
    
    assert date_range > 0, "Date range is invalid (0 days)"
    assert min_date < max_date, "Min date is not before max date"
    
    print(f"✓ Valid date range")
    print("=" * 80)
    print("✓ PASSED: Valid date range\n")


def test_wave_history_data_integrity():
    """Test that wave_history.csv has valid data (no all-NaN columns)."""
    print("\n" + "=" * 80)
    print("TEST: wave_history.csv data integrity")
    print("=" * 80)
    
    df = pd.read_csv("wave_history.csv")
    
    # Check for all-NaN columns in required fields
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = 100 * null_count / len(df)
            print(f"Column '{col}': {null_count:,} NaN values ({null_pct:.1f}%)")
            
            # portfolio_return and benchmark_return can have some NaN, but not 100%
            if col in ['portfolio_return', 'benchmark_return']:
                assert null_pct < 100, f"Column '{col}' is entirely NaN"
            # date and wave should never be NaN
            elif col in ['date', 'wave']:
                assert null_count == 0, f"Column '{col}' has NaN values"
    
    print(f"✓ Data integrity validated")
    print("=" * 80)
    print("✓ PASSED: Data integrity check\n")


def test_wave_coverage_snapshot_exists():
    """Test that wave_coverage_snapshot.json exists and is valid."""
    print("\n" + "=" * 80)
    print("TEST: wave_coverage_snapshot.json")
    print("=" * 80)
    
    snapshot_file = "wave_coverage_snapshot.json"
    
    if not os.path.exists(snapshot_file):
        print(f"⚠ WARNING: {snapshot_file} not found (optional)")
        print("=" * 80)
        print("⚠ SKIPPED: Coverage snapshot not found\n")
        return
    
    print(f"✓ Coverage snapshot exists: {snapshot_file}")
    
    # Load and validate
    with open(snapshot_file, 'r') as f:
        snapshot = json.load(f)
    
    print(f"\nSnapshot metadata:")
    print(f"  Timestamp: {snapshot.get('timestamp', 'N/A')}")
    print(f"  Total waves: {snapshot.get('total_waves', 0)}")
    print(f"  Waves meeting threshold: {snapshot.get('waves_meeting_threshold', 0)}")
    print(f"  Waves below threshold: {snapshot.get('waves_below_threshold', 0)}")
    print(f"  Min coverage threshold: {snapshot.get('min_coverage_threshold', 0):.0%}")
    
    # List waves below threshold if any
    waves_below = [w for w in snapshot.get('waves', []) if not w.get('meets_threshold', True)]
    if waves_below:
        print(f"\nWaves below coverage threshold ({len(waves_below)}):")
        for wave in waves_below[:5]:  # Show first 5
            print(f"  - {wave['wave']}: {wave['coverage_pct']:.2%} coverage")
        if len(waves_below) > 5:
            print(f"  ... and {len(waves_below) - 5} more")
    
    print("=" * 80)
    print("✓ PASSED: Coverage snapshot validated\n")


def test_strategy_overlay_fields():
    """Test that strategy overlay fields are present if applicable."""
    print("\n" + "=" * 80)
    print("TEST: Strategy overlay fields (optional)")
    print("=" * 80)
    
    df = pd.read_csv("wave_history.csv")
    
    overlay_columns = ['vix_level', 'vix_regime', 'exposure_used', 'overlay_active']
    present_overlay_cols = [col for col in overlay_columns if col in df.columns]
    
    if not present_overlay_cols:
        print("⚠ No overlay columns found (this is OK for basic implementation)")
        print("=" * 80)
        print("⚠ SKIPPED: No overlay fields present\n")
        return
    
    print(f"Overlay columns found: {present_overlay_cols}")
    
    if 'overlay_active' in df.columns:
        overlay_count = df['overlay_active'].sum()
        overlay_pct = 100 * overlay_count / len(df)
        print(f"Strategy overlay records: {overlay_count:,} ({overlay_pct:.1f}%)")
    
    print("=" * 80)
    print("✓ PASSED: Strategy overlay fields validated\n")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("BUILD WAVE HISTORY WORKFLOW TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("File exists", test_wave_history_file_exists),
        ("Row count threshold", test_wave_history_row_count),
        ("Wave count threshold", test_wave_history_wave_count),
        ("Required columns", test_wave_history_required_columns),
        ("Date range", test_wave_history_date_range),
        ("Data integrity", test_wave_history_data_integrity),
        ("Coverage snapshot", test_wave_coverage_snapshot_exists),
        ("Strategy overlay", test_strategy_overlay_fields),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1
        except FileNotFoundError as e:
            # Only skip for expected missing optional files
            if "wave_coverage_snapshot.json" in str(e):
                print(f"\n⚠ SKIPPED: {test_name}")
                print(f"  Reason: Optional file not found - {e}")
                skipped += 1
            else:
                print(f"\n✗ FAILED: {test_name}")
                print(f"  Unexpected error: {e}")
                failed += 1
        except Exception as e:
            # All other exceptions are failures
            print(f"\n✗ FAILED: {test_name}")
            print(f"  Unexpected error: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Skipped: {skipped}")
    print("=" * 80)
    
    if failed > 0:
        print("\n✗ OVERALL RESULT: FAILED")
        sys.exit(1)
    else:
        print("\n✓ OVERALL RESULT: PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
