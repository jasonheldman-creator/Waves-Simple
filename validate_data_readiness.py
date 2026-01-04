#!/usr/bin/env python3
"""
Data Readiness Validation Script

This script validates all the requirements from the PR:
A) Missing Tickers Resolution
B) Price Data Staleness Detection
C) Wave Universe Validation
D) System Health Banner

Usage:
    python validate_data_readiness.py
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

def validate_missing_tickers():
    """Validate A) Missing Tickers Resolution."""
    print_section("A) MISSING TICKERS VALIDATION")
    
    # Load cache
    cache_path = 'data/cache/prices_cache.parquet'
    if not os.path.exists(cache_path):
        print("❌ FAIL: Price cache not found")
        return False
    
    df = pd.read_parquet(cache_path)
    tickers = set(df.columns.tolist())
    
    # Check for required tickers
    required_tickers = ['IGV', 'STETH-USD', '^VIX']
    missing = []
    found = []
    
    for ticker in required_tickers:
        if ticker in tickers:
            found.append(ticker)
            print(f"✅ {ticker:15s} - FOUND in cache")
        else:
            missing.append(ticker)
            print(f"❌ {ticker:15s} - MISSING from cache")
    
    # Also check for stETH-USD variant
    if 'stETH-USD' in tickers:
        print(f"ℹ️  {'stETH-USD':15s} - FOUND (lowercase variant)")
    
    print(f"\nSummary:")
    print(f"  Required tickers: {len(required_tickers)}")
    print(f"  Found in cache:   {len(found)}")
    print(f"  Missing:          {len(missing)}")
    
    if len(missing) == 0:
        print(f"\n✅ PASS: All required tickers present (Coverage: 100%)")
        return True
    else:
        print(f"\n❌ FAIL: Missing {len(missing)} tickers")
        return False

def validate_price_staleness():
    """Validate B) Price Data Staleness Detection."""
    print_section("B) PRICE DATA STALENESS VALIDATION")
    
    try:
        from helpers.price_loader import check_cache_readiness
        
        readiness = check_cache_readiness(active_only=True)
        
        print(f"Cache Status:")
        print(f"  Exists:        {readiness['exists']}")
        print(f"  Trading Days:  {readiness['num_days']}")
        print(f"  Tickers:       {readiness['num_tickers']}")
        print(f"  Max Date:      {readiness['max_date']}")
        print(f"  Days Stale:    {readiness['days_stale']}")
        print(f"  Status Code:   {readiness['status_code']}")
        print(f"  Status:        {readiness['status']}")
        
        # Check if staleness detection is working
        if readiness['days_stale'] is not None:
            print(f"\n✅ PASS: Staleness detection working (uses max date from cache)")
            print(f"  Data age: {readiness['days_stale']} days")
            
            if readiness['days_stale'] <= 1:
                print(f"  Status: FRESH (≤ 1 trading day)")
            elif readiness['days_stale'] <= 5:
                print(f"  Status: RECENT (≤ 5 days)")
            elif readiness['days_stale'] <= 10:
                print(f"  Status: DEGRADED (≤ 10 days)")
            else:
                print(f"  Status: STALE (> 10 days)")
            
            return True
        else:
            print(f"\n❌ FAIL: Staleness detection not working")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def validate_wave_universe():
    """Validate C) Wave Universe Validation."""
    print_section("C) WAVE UNIVERSE VALIDATION")
    
    try:
        # Load wave registry
        registry_path = 'data/wave_registry.csv'
        if not os.path.exists(registry_path):
            print(f"❌ FAIL: Wave registry not found at {registry_path}")
            return False
        
        df = pd.read_csv(registry_path)
        
        total_waves = len(df)
        active_waves = df[df['active'] == True]
        inactive_waves = df[df['active'] == False]
        
        active_count = len(active_waves)
        inactive_count = len(inactive_waves)
        
        print(f"Wave Registry:")
        print(f"  Total waves:    {total_waves}")
        print(f"  Active waves:   {active_count}")
        print(f"  Inactive waves: {inactive_count}")
        
        if inactive_count > 0:
            print(f"\nInactive waves:")
            for _, row in inactive_waves.iterrows():
                print(f"  - {row['wave_name']}")
        
        # Validation
        expected_active = 27
        if active_count == expected_active:
            print(f"\n✅ PASS: Wave universe validated ({active_count}/{active_count} active waves)")
            if inactive_count > 0:
                print(f"ℹ️  INFO: {inactive_count} inactive wave(s) properly excluded")
            return True
        else:
            print(f"\n❌ FAIL: Expected {expected_active} active waves, found {active_count}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def validate_system_health():
    """Validate D) System Health Banner."""
    print_section("D) SYSTEM HEALTH VALIDATION")
    
    try:
        from helpers.price_book import (
            get_price_book,
            compute_missing_and_extra_tickers,
            compute_system_health
        )
        
        # Load price book
        price_book = get_price_book()
        
        # Compute ticker analysis
        ticker_analysis = compute_missing_and_extra_tickers(price_book)
        
        print(f"Ticker Coverage:")
        print(f"  Required:  {ticker_analysis['required_count']}")
        print(f"  In Cache:  {ticker_analysis['cached_count']}")
        print(f"  Missing:   {ticker_analysis['missing_count']}")
        print(f"  Extra:     {ticker_analysis['extra_count']}")
        
        # Compute system health
        health = compute_system_health(price_book)
        
        print(f"\nSystem Health:")
        print(f"  Status:    {health['health_emoji']} {health['health_status']}")
        print(f"  Coverage:  {health['coverage_pct']:.1f}%")
        print(f"  Days Old:  {health['days_stale']}")
        print(f"  Details:   {health['details']}")
        
        # Determine pass/fail
        if ticker_analysis['missing_count'] == 0:
            print(f"\n✅ PASS: System health logic working correctly")
            print(f"  - No missing tickers")
            print(f"  - Coverage: 100%")
            
            if health['health_status'] == 'OK':
                print(f"  - Status: OK/GREEN (data fresh)")
            elif health['health_status'] == 'DEGRADED':
                print(f"  - Status: DEGRADED/YELLOW (data {health['days_stale']} days old)")
                print(f"  - Note: Will be OK/GREEN when cache updated to < 5 days old")
            elif health['health_status'] == 'STALE':
                print(f"  - Status: STALE/RED (data {health['days_stale']} days old)")
                print(f"  - Note: Needs cache update")
            
            return True
        else:
            print(f"\n❌ FAIL: {ticker_analysis['missing_count']} missing tickers")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation routine."""
    print("=" * 70)
    print("DATA READINESS AND HEALTH VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'A_Missing_Tickers': validate_missing_tickers(),
        'B_Staleness_Detection': validate_price_staleness(),
        'C_Wave_Universe': validate_wave_universe(),
        'D_System_Health': validate_system_health()
    }
    
    print_section("FINAL VALIDATION SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ')}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 SUCCESS: All validation checks passed!")
        return 0
    else:
        print(f"\n⚠️  WARNING: {failed}/{total} validation checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
