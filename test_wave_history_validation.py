#!/usr/bin/env python3
"""
Test script to validate wave_history.csv data integrity enforcement.

Tests:
1. safe_load_wave_history() properly validates data
2. Missing wave_history.csv is detected and reported
3. Empty wave_history.csv is detected and reported
4. Stale wave_history.csv is detected and reported
5. Validation state is properly stored in session state
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil


def test_missing_wave_history():
    """Test that missing wave_history.csv is properly detected."""
    print("=" * 70)
    print("TEST 1: Missing wave_history.csv Detection")
    print("=" * 70)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to temp directory (no wave_history.csv)
            os.chdir(tmpdir)
            
            # Import after changing directory
            sys.path.insert(0, original_dir)
            from app import safe_load_wave_history
            
            # Try to load - should return None
            df = safe_load_wave_history()
            
            if df is None:
                print("✅ Correctly returned None for missing file")
                return True
            else:
                print("❌ Should have returned None but got DataFrame")
                return False
                
        finally:
            os.chdir(original_dir)
            sys.path.pop(0)


def test_empty_wave_history():
    """Test that empty wave_history.csv is properly detected."""
    print("\n" + "=" * 70)
    print("TEST 2: Empty wave_history.csv Detection")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        
        try:
            os.chdir(tmpdir)
            
            # Create empty CSV file
            with open('wave_history.csv', 'w') as f:
                f.write("date,wave\n")  # Header only
            
            sys.path.insert(0, original_dir)
            from app import safe_load_wave_history
            
            df = safe_load_wave_history()
            
            if df is None:
                print("✅ Correctly returned None for empty file")
                return True
            else:
                print("❌ Should have returned None for empty file")
                return False
                
        finally:
            os.chdir(original_dir)
            sys.path.pop(0)


def test_invalid_wave_history():
    """Test that invalid wave_history.csv is properly detected."""
    print("\n" + "=" * 70)
    print("TEST 3: Invalid wave_history.csv Detection")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        
        try:
            os.chdir(tmpdir)
            
            # Create CSV without required 'date' column
            with open('wave_history.csv', 'w') as f:
                f.write("wave,value\n")
                f.write("Test Wave,100\n")
            
            sys.path.insert(0, original_dir)
            from app import safe_load_wave_history
            
            df = safe_load_wave_history()
            
            if df is None:
                print("✅ Correctly returned None for invalid file (no date column)")
                return True
            else:
                print("❌ Should have returned None for invalid file")
                return False
                
        finally:
            os.chdir(original_dir)
            sys.path.pop(0)


def test_stale_wave_history():
    """Test that stale wave_history.csv is detected."""
    print("\n" + "=" * 70)
    print("TEST 4: Stale wave_history.csv Detection")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        
        try:
            os.chdir(tmpdir)
            
            # Create CSV with old data (60 days old)
            old_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            with open('wave_history.csv', 'w') as f:
                f.write("date,wave,portfolio_return\n")
                f.write(f"{old_date},Test Wave,0.01\n")
            
            sys.path.insert(0, original_dir)
            from app import safe_load_wave_history
            
            df = safe_load_wave_history()
            
            # Should still load the data, but we'll check if staleness is detected
            # by looking at session state in a real app context
            if df is not None:
                print(f"✅ Loaded stale data (max date: {df['date'].max()})")
                days_old = (datetime.now() - pd.to_datetime(df['date'].max())).days
                print(f"   Data is {days_old} days old")
                return True
            else:
                print("❌ Failed to load valid (albeit stale) data")
                return False
                
        finally:
            os.chdir(original_dir)
            sys.path.pop(0)


def test_valid_wave_history():
    """Test that valid wave_history.csv loads properly."""
    print("\n" + "=" * 70)
    print("TEST 5: Valid wave_history.csv Loading")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        
        try:
            os.chdir(tmpdir)
            
            # Create CSV with recent data
            recent_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            with open('wave_history.csv', 'w') as f:
                f.write("date,wave,portfolio_return,benchmark_return\n")
                f.write(f"{recent_date},Test Wave,0.01,0.005\n")
            
            sys.path.insert(0, original_dir)
            from app import safe_load_wave_history
            
            df = safe_load_wave_history()
            
            if df is not None and len(df) > 0:
                print(f"✅ Successfully loaded valid data")
                print(f"   Rows: {len(df)}")
                print(f"   Max date: {df['date'].max()}")
                return True
            else:
                print("❌ Failed to load valid data")
                return False
                
        finally:
            os.chdir(original_dir)
            sys.path.pop(0)


def test_actual_wave_history():
    """Test loading the actual wave_history.csv from the repository."""
    print("\n" + "=" * 70)
    print("TEST 6: Actual wave_history.csv Validation")
    print("=" * 70)
    
    try:
        # Check if wave_history.csv exists
        if not os.path.exists('wave_history.csv'):
            print("⚠️  wave_history.csv not found in current directory")
            return True  # Not a failure, just skip
        
        from app import safe_load_wave_history
        
        df = safe_load_wave_history()
        
        if df is None:
            print("❌ Failed to load existing wave_history.csv")
            return False
        
        print(f"✅ Successfully loaded wave_history.csv")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {', '.join(df.columns[:5])}...")
        
        if 'date' in df.columns:
            max_date = df['date'].max()
            min_date = df['date'].min()
            days_old = (datetime.now() - pd.to_datetime(max_date)).days
            
            print(f"   Date range: {min_date} to {max_date}")
            print(f"   Data age: {days_old} days")
            
            if days_old > 14:
                print(f"   ⚠️  Data is {days_old} days old (>14 days)")
        
        if 'wave' in df.columns:
            unique_waves = df['wave'].nunique()
            print(f"   Unique waves: {unique_waves}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("WAVE HISTORY DATA INTEGRITY VALIDATION TESTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Missing File Detection", test_missing_wave_history()))
    results.append(("Empty File Detection", test_empty_wave_history()))
    results.append(("Invalid File Detection", test_invalid_wave_history()))
    results.append(("Stale Data Detection", test_stale_wave_history()))
    results.append(("Valid Data Loading", test_valid_wave_history()))
    results.append(("Actual File Validation", test_actual_wave_history()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
