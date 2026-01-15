#!/usr/bin/env python3
"""
Test script to validate wave_history.csv data integrity enforcement.

Tests:
1. Actual wave_history.csv file validation
2. Data freshness checks
3. Required columns validation
4. Validation state storage
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta


def test_wave_history_exists():
    """Test that wave_history.csv exists."""
    print("=" * 70)
    print("TEST 1: wave_history.csv File Existence")
    print("=" * 70)
    
    if os.path.exists('wave_history.csv'):
        print("✅ wave_history.csv file exists")
        file_size = os.path.getsize('wave_history.csv')
        print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return True
    else:
        print("❌ wave_history.csv file not found")
        return False


def test_wave_history_readable():
    """Test that wave_history.csv is readable."""
    print("\n" + "=" * 70)
    print("TEST 2: wave_history.csv Readability")
    print("=" * 70)
    
    try:
        df = pd.read_csv('wave_history.csv')
        print(f"✅ Successfully read wave_history.csv")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Failed to read wave_history.csv: {e}")
        return False


def test_required_columns():
    """Test that wave_history.csv has required columns."""
    print("\n" + "=" * 70)
    print("TEST 3: Required Columns Validation")
    print("=" * 70)
    
    try:
        df = pd.read_csv('wave_history.csv')
        
        required_cols = ['date', 'wave']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {', '.join(missing_cols)}")
            print(f"   Available columns: {', '.join(df.columns[:10])}...")
            return False
        else:
            print(f"✅ All required columns present")
            print(f"   Columns: {', '.join(df.columns[:10])}...")
            return True
            
    except Exception as e:
        print(f"❌ Error checking columns: {e}")
        return False


def test_date_column_validity():
    """Test that date column has valid dates."""
    print("\n" + "=" * 70)
    print("TEST 4: Date Column Validity")
    print("=" * 70)
    
    try:
        df = pd.read_csv('wave_history.csv')
        
        if 'date' not in df.columns:
            print("❌ Date column not found")
            return False
        
        # Try to parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Check for invalid dates
        invalid_count = df['date'].isna().sum()
        valid_count = (~df['date'].isna()).sum()
        
        if valid_count == 0:
            print("❌ No valid dates found")
            return False
        
        if invalid_count > 0:
            print(f"⚠️  Found {invalid_count} invalid dates (out of {len(df)} rows)")
        
        print(f"✅ Date column validated")
        print(f"   Valid dates: {valid_count:,}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating dates: {e}")
        return False


def test_data_freshness():
    """Test data freshness - check if data is recent."""
    print("\n" + "=" * 70)
    print("TEST 5: Data Freshness Check")
    print("=" * 70)
    
    try:
        df = pd.read_csv('wave_history.csv')
        
        if 'date' not in df.columns:
            print("❌ Date column not found")
            return False
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            print("❌ No valid dates found")
            return False
        
        max_date = df['date'].max()
        days_old = (datetime.now() - max_date).days
        
        print(f"   Latest data date: {max_date.strftime('%Y-%m-%d')}")
        print(f"   Data age: {days_old} days")
        
        if days_old > 30:
            print(f"❌ Data is critically stale (>30 days old)")
            return False
        elif days_old > 14:
            print(f"⚠️  Data is stale (>14 days old)")
            print("   Recommendation: Refresh wave_history.csv")
        elif days_old > 7:
            print(f"⚠️  Data needs refresh (>7 days old)")
        else:
            print(f"✅ Data is fresh (≤7 days old)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking freshness: {e}")
        return False


def test_data_columns():
    """Test that essential data columns exist."""
    print("\n" + "=" * 70)
    print("TEST 6: Essential Data Columns")
    print("=" * 70)
    
    try:
        df = pd.read_csv('wave_history.csv')
        
        essential_cols = ['portfolio_return', 'benchmark_return']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing optional columns: {', '.join(missing_cols)}")
            print("   These columns are recommended for full analytics")
        else:
            print(f"✅ All essential data columns present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data columns: {e}")
        return False


def test_wave_coverage():
    """Test wave coverage in the data."""
    print("\n" + "=" * 70)
    print("TEST 7: Wave Coverage")
    print("=" * 70)
    
    try:
        df = pd.read_csv('wave_history.csv')
        
        if 'wave' not in df.columns:
            print("⚠️  Wave column not found")
            return True  # Not critical
        
        unique_waves = df['wave'].nunique()
        print(f"✅ Wave coverage validated")
        print(f"   Unique waves: {unique_waves}")
        
        if unique_waves > 0:
            # Show sample waves
            sample_waves = df['wave'].unique()[:5]
            print(f"   Sample waves: {', '.join(str(w) for w in sample_waves)}")
            if unique_waves > 5:
                print(f"   ... and {unique_waves - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking wave coverage: {e}")
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
    results.append(("File Existence", test_wave_history_exists()))
    results.append(("File Readability", test_wave_history_readable()))
    results.append(("Required Columns", test_required_columns()))
    results.append(("Date Validity", test_date_column_validity()))
    results.append(("Data Freshness", test_data_freshness()))
    results.append(("Data Columns", test_data_columns()))
    results.append(("Wave Coverage", test_wave_coverage()))
    
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
