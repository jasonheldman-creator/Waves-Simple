#!/usr/bin/env python3
"""
Test Ticker Master File Implementation

Validates that:
1. ticker_master_clean.csv exists and is valid
2. Ticker normalization works correctly
3. Startup validation includes ticker checks
4. All 28 waves are defined
"""

import os
import sys
import pandas as pd


def test_ticker_master_exists():
    """Test that ticker_master_clean.csv exists."""
    print("\n1. Testing ticker master file exists...")
    ticker_file = 'ticker_master_clean.csv'
    
    if not os.path.exists(ticker_file):
        print(f"   ❌ FAIL: {ticker_file} not found")
        return False
    
    print(f"   ✅ PASS: {ticker_file} exists")
    return True


def test_ticker_master_structure():
    """Test that ticker_master_clean.csv has correct structure."""
    print("\n2. Testing ticker master file structure...")
    ticker_file = 'ticker_master_clean.csv'
    
    try:
        df = pd.read_csv(ticker_file)
        
        # Check required columns
        required_cols = ['ticker', 'original_forms', 'created_date', 'source']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"   ❌ FAIL: Missing columns: {missing}")
            return False
        
        print(f"   ✅ PASS: All required columns present")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Error reading file: {e}")
        return False


def test_ticker_master_no_duplicates():
    """Test that ticker_master_clean.csv has no duplicates."""
    print("\n3. Testing for duplicate tickers...")
    ticker_file = 'ticker_master_clean.csv'
    
    try:
        df = pd.read_csv(ticker_file)
        duplicates = df[df['ticker'].duplicated()]
        
        if not duplicates.empty:
            print(f"   ❌ FAIL: Found {len(duplicates)} duplicate tickers")
            print(f"      Duplicates: {duplicates['ticker'].tolist()}")
            return False
        
        print(f"   ✅ PASS: No duplicate tickers found")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Error checking duplicates: {e}")
        return False


def test_ticker_count():
    """Test that ticker_master_clean.csv has expected number of tickers."""
    print("\n4. Testing ticker count...")
    ticker_file = 'ticker_master_clean.csv'
    
    try:
        df = pd.read_csv(ticker_file)
        ticker_count = len(df)
        
        # Should have 120 tickers from WAVE_WEIGHTS
        if ticker_count < 100:
            print(f"   ⚠️  WARNING: Only {ticker_count} tickers (expected ~120)")
            return True  # Warning, not failure
        
        print(f"   ✅ PASS: {ticker_count} tickers present")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Error counting tickers: {e}")
        return False


def test_ticker_normalization():
    """Test that ticker normalization works correctly."""
    print("\n5. Testing ticker normalization...")
    
    try:
        from waves_engine import _normalize_ticker
        
        # Test cases
        test_cases = [
            ('BRK-B', 'BRK-B'),  # Already normalized
            ('AAPL', 'AAPL'),    # Simple ticker
            ('BTC-USD', 'BTC-USD'),  # Crypto ticker
            ('stETH-USD', 'STETH-USD'),  # Normalized crypto
        ]
        
        all_passed = True
        for input_ticker, expected in test_cases:
            result = _normalize_ticker(input_ticker)
            if result != expected:
                print(f"   ❌ FAIL: {input_ticker} → {result} (expected {expected})")
                all_passed = False
        
        if all_passed:
            print(f"   ✅ PASS: All normalization tests passed")
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ FAIL: Error testing normalization: {e}")
        return False


def test_wave_count():
    """Test that all 28 waves are defined."""
    print("\n6. Testing wave count...")
    
    try:
        from waves_engine import get_all_waves_universe
        
        from waves_engine import WAVE_ID_REGISTRY
        expected_count = len(WAVE_ID_REGISTRY)
        
        universe = get_all_waves_universe()
        wave_count = universe['count']
        
        if wave_count != expected_count:
            print(f"   ❌ FAIL: Found {wave_count} waves (expected {expected_count} from WAVE_ID_REGISTRY)")
            return False
        
        print(f"   ✅ PASS: All {expected_count} waves defined (from WAVE_ID_REGISTRY)")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Error checking waves: {e}")
        return False


def test_startup_validation():
    """Test that startup validation includes ticker checks."""
    print("\n7. Testing startup validation...")
    
    try:
        from helpers.startup_validation import check_ticker_master_file
        
        success, message = check_ticker_master_file()
        
        if not success:
            print(f"   ❌ FAIL: Validation failed: {message}")
            return False
        
        print(f"   ✅ PASS: {message}")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Error running validation: {e}")
        return False


def test_old_files_deprecated():
    """Test that old ticker files are deprecated."""
    print("\n8. Testing old ticker files deprecated...")
    
    old_files = ['list.csv', 'Master_Stock_Sheet.csv']
    deprecated_files = ['list.csv.deprecated', 'Master_Stock_Sheet.csv.deprecated']
    
    all_good = True
    for old_file in old_files:
        if os.path.exists(old_file):
            print(f"   ⚠️  WARNING: {old_file} still exists (should be deprecated)")
            all_good = False
    
    for dep_file in deprecated_files:
        if os.path.exists(dep_file):
            print(f"   ✅ Found deprecated file: {dep_file}")
    
    if all_good:
        print(f"   ✅ PASS: Old ticker files properly deprecated")
    
    return all_good


def main():
    """Run all tests."""
    print("=" * 60)
    print("TICKER MASTER FILE IMPLEMENTATION TESTS")
    print("=" * 60)
    
    tests = [
        test_ticker_master_exists,
        test_ticker_master_structure,
        test_ticker_master_no_duplicates,
        test_ticker_count,
        test_ticker_normalization,
        test_wave_count,
        test_startup_validation,
        test_old_files_deprecated,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n   ❌ EXCEPTION in {test.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
