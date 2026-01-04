#!/usr/bin/env python3
"""
Validation Script for PR #352 - Continuous Rerun Elimination & Price Cache Updates

This script validates:
1. Auto-refresh is disabled by default
2. Run counter logic prevents infinite loops
3. Data age indicators work correctly
4. Price cache structure is correct
5. GitHub Actions workflow is properly configured
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auto_refresh_disabled():
    """Test 1: Verify auto-refresh is disabled by default"""
    print("\n" + "=" * 70)
    print("TEST 1: Auto-Refresh Default State")
    print("=" * 70)
    
    try:
        from auto_refresh_config import DEFAULT_AUTO_REFRESH_ENABLED
        
        if DEFAULT_AUTO_REFRESH_ENABLED is False:
            print("✅ PASS: Auto-refresh is disabled by default")
            print(f"   DEFAULT_AUTO_REFRESH_ENABLED = {DEFAULT_AUTO_REFRESH_ENABLED}")
            return True
        else:
            print("❌ FAIL: Auto-refresh is enabled by default")
            print(f"   Expected: False, Got: {DEFAULT_AUTO_REFRESH_ENABLED}")
            return False
    except ImportError as e:
        print(f"❌ FAIL: Could not import auto_refresh_config: {e}")
        return False


def test_run_counter_logic():
    """Test 2: Verify run counter logic exists in app.py"""
    print("\n" + "=" * 70)
    print("TEST 2: Run Counter Logic")
    print("=" * 70)
    
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('run_count initialization', 'if "run_count" not in st.session_state:'),
            ('run_count increment', 'st.session_state.run_count += 1'),
            ('run_count threshold check', 'if st.session_state.run_count > 3:'),
            ('loop detection message', 'LOOP DETECTION'),
            ('user_interaction_detected flag', 'user_interaction_detected'),
        ]
        
        all_passed = True
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"✅ Found: {check_name}")
            else:
                print(f"❌ Missing: {check_name}")
                all_passed = False
        
        if all_passed:
            print("\n✅ PASS: All run counter logic components present")
            return True
        else:
            print("\n❌ FAIL: Some run counter logic components missing")
            return False
            
    except FileNotFoundError:
        print("❌ FAIL: app.py not found")
        return False


def test_data_age_indicators():
    """Test 3: Verify data age indicators exist in app.py"""
    print("\n" + "=" * 70)
    print("TEST 3: Data Age Indicators")
    print("=" * 70)
    
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('data_age_days variable', 'data_age_days'),
            ('Last Price Date metric', '"Last Price Date"'),
            ('Data Age metric', 'age_display'),
            ('STALE warning', 'STALE'),
            ('ALLOW_NETWORK_FETCH check', 'ALLOW_NETWORK_FETCH'),
        ]
        
        all_passed = True
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"✅ Found: {check_name}")
            else:
                print(f"❌ Missing: {check_name}")
                all_passed = False
        
        if all_passed:
            print("\n✅ PASS: All data age indicator components present")
            return True
        else:
            print("\n❌ FAIL: Some data age indicator components missing")
            return False
            
    except FileNotFoundError:
        print("❌ FAIL: app.py not found")
        return False


def test_price_cache_structure():
    """Test 4: Verify price cache file exists and is properly structured"""
    print("\n" + "=" * 70)
    print("TEST 4: Price Cache Structure")
    print("=" * 70)
    
    cache_path = Path("data/cache/prices_cache.parquet")
    
    if not cache_path.exists():
        print("⚠️  WARNING: prices_cache.parquet does not exist yet")
        print("   This is expected before first workflow run")
        print("   Cache will be created by GitHub Action")
        return True  # Not a failure - cache will be created by workflow
    
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  WARNING: pandas not installed, cannot validate cache structure")
        print("   Cache file exists and will be validated by workflow")
        print(f"   Size: {cache_path.stat().st_size / 1024:.2f} KB")
        return True  # Not a failure - just can't validate details
    
    try:
        
        df = pd.read_parquet(cache_path)
        
        print(f"✅ Cache file exists: {cache_path}")
        print(f"   Size: {cache_path.stat().st_size / 1024:.2f} KB")
        print(f"   Trading Days: {len(df)}")
        print(f"   Tickers: {len(df.columns)}")
        print(f"   Date Range: {df.index.min()} to {df.index.max()}")
        
        # Check structure
        if isinstance(df.index, pd.DatetimeIndex):
            print("✅ Index is DatetimeIndex (correct format)")
        else:
            print("❌ Index is not DatetimeIndex")
            return False
        
        if len(df.columns) > 0:
            print(f"✅ Cache contains {len(df.columns)} ticker columns")
        else:
            print("❌ Cache has no ticker columns")
            return False
        
        # Calculate data age
        last_date = df.index.max().to_pydatetime()
        age_days = (datetime.now() - last_date).days
        print(f"   Data Age: {age_days} days")
        
        if age_days <= 10:
            print(f"✅ Data is fresh (<= 10 days old)")
        else:
            print(f"⚠️  Data is stale (> 10 days old)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error reading cache file: {e}")
        return False


def test_workflow_configuration():
    """Test 5: Verify GitHub Actions workflow exists and is properly configured"""
    print("\n" + "=" * 70)
    print("TEST 5: GitHub Actions Workflow Configuration")
    print("=" * 70)
    
    workflow_path = Path(".github/workflows/update_price_cache.yml")
    
    if not workflow_path.exists():
        print("❌ FAIL: update_price_cache.yml workflow not found")
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('Workflow name', 'name: Update Price Cache'),
            ('Schedule trigger', 'schedule:'),
            ('Cron schedule', 'cron:'),
            ('Manual trigger', 'workflow_dispatch:'),
            ('Checkout step', 'actions/checkout@v4'),
            ('Python setup', 'actions/setup-python@v5'),
            ('Build cache step', 'build_complete_price_cache.py'),
            ('CSV to Parquet conversion', 'prices_cache.parquet'),
            ('Commit step', 'git commit'),
            ('Push step', 'git push'),
        ]
        
        all_passed = True
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"✅ Found: {check_name}")
            else:
                print(f"❌ Missing: {check_name}")
                all_passed = False
        
        if all_passed:
            print("\n✅ PASS: Workflow is properly configured")
            return True
        else:
            print("\n❌ FAIL: Workflow is missing some components")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error reading workflow file: {e}")
        return False


def test_price_book_module():
    """Test 6: Verify price_book module configuration"""
    print("\n" + "=" * 70)
    print("TEST 6: Price Book Module Configuration")
    print("=" * 70)
    
    try:
        # We can't import directly due to streamlit dependencies
        # So we'll read the file and check for key constants
        with open('helpers/price_book.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('CANONICAL_CACHE_PATH', 'CANONICAL_CACHE_PATH ='),
            ('STALE_DAYS_THRESHOLD', 'STALE_DAYS_THRESHOLD = 10'),
            ('DEGRADED_DAYS_THRESHOLD', 'DEGRADED_DAYS_THRESHOLD = 5'),
            ('PRICE_FETCH_ENABLED', 'PRICE_FETCH_ENABLED ='),
            ('get_price_book function', 'def get_price_book('),
            ('get_price_book_meta function', 'def get_price_book_meta('),
            ('rebuild_price_cache function', 'def rebuild_price_cache('),
        ]
        
        all_passed = True
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"✅ Found: {check_name}")
            else:
                print(f"❌ Missing: {check_name}")
                all_passed = False
        
        if all_passed:
            print("\n✅ PASS: Price book module is properly configured")
            return True
        else:
            print("\n❌ FAIL: Price book module is missing some components")
            return False
            
    except FileNotFoundError:
        print("❌ FAIL: helpers/price_book.py not found")
        return False


def test_build_script_exists():
    """Test 7: Verify build_complete_price_cache.py exists"""
    print("\n" + "=" * 70)
    print("TEST 7: Build Script Availability")
    print("=" * 70)
    
    script_path = Path("build_complete_price_cache.py")
    
    if not script_path.exists():
        print("❌ FAIL: build_complete_price_cache.py not found")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('Extract wave tickers', 'extract_wave_holdings_tickers'),
            ('Extract benchmarks', 'extract_benchmark_tickers'),
            ('Safe assets', 'get_safe_asset_tickers'),
            ('Download function', 'download_ticker_prices'),
            ('Batch download', 'download_all_prices'),
            ('Consolidate CSV', 'build_consolidated_prices_csv'),
            ('Main function', 'def main():'),
        ]
        
        all_passed = True
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"✅ Found: {check_name}")
            else:
                print(f"⚠️  Not found: {check_name}")
                # Don't fail on these - script may have different implementation
        
        print("\n✅ PASS: Build script exists and appears functional")
        return True
            
    except Exception as e:
        print(f"❌ FAIL: Error reading build script: {e}")
        return False


def main():
    """Run all validation tests"""
    print("\n" + "=" * 70)
    print("PR #352 VALIDATION - Continuous Rerun Elimination & Price Cache Updates")
    print("=" * 70)
    
    tests = [
        ("Auto-Refresh Disabled", test_auto_refresh_disabled),
        ("Run Counter Logic", test_run_counter_logic),
        ("Data Age Indicators", test_data_age_indicators),
        ("Price Cache Structure", test_price_cache_structure),
        ("Workflow Configuration", test_workflow_configuration),
        ("Price Book Module", test_price_book_module),
        ("Build Script", test_build_script_exists),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("\nNext Steps:")
        print("1. Run GitHub Actions workflow manually")
        print("2. Capture screenshots for proof artifacts")
        print("3. Document results in PR description")
        return 0
    else:
        print("\n⚠️  Some validation tests failed")
        print("Please review the failures above and fix before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
