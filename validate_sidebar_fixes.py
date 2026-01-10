#!/usr/bin/env python3
"""
Manual validation script to verify the sidebar maintenance action fixes.

This script tests:
1. rebuild_price_cache_toolbox() function (from operator_toolbox)
2. rebuild_wave_history() function
3. force_ledger_recompute() function
4. Verifies that price_book.max('date') == wave_history.max('date')
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add helpers to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers'))

print("=" * 80)
print("SIDEBAR MAINTENANCE ACTIONS - MANUAL VALIDATION")
print("=" * 80)
print()

# Test 1: Check that operator_toolbox functions are importable
print("Test 1: Import operator_toolbox functions")
print("-" * 80)
try:
    from helpers.operator_toolbox import (
        rebuild_price_cache as rebuild_price_cache_toolbox,
        rebuild_wave_history,
        force_ledger_recompute
    )
    print("✅ Successfully imported operator_toolbox functions")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)
print()

# Test 2: Check rebuild_wave_history function
print("Test 2: Test rebuild_wave_history() function")
print("-" * 80)
try:
    success, message = rebuild_wave_history()
    
    if success:
        print("✅ rebuild_wave_history() succeeded")
        print(f"   Message: {message[:200]}...")
    else:
        print(f"❌ rebuild_wave_history() failed: {message}")
        # Don't exit, continue with other tests
except Exception as e:
    print(f"❌ rebuild_wave_history() raised exception: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Verify price_book and wave_history dates match
print("Test 3: Verify price_book.max('date') == wave_history.max('date')")
print("-" * 80)
try:
    from helpers import price_book
    
    # Get price_book max date
    price_data = price_book.get_price_book()
    if price_data.empty:
        print("❌ Price data is empty")
    else:
        price_book_max_date = price_data.index[-1].strftime('%Y-%m-%d')
        print(f"   price_book max date: {price_book_max_date}")
        
        # Get wave_history max date
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            print(f"❌ wave_history.csv not found at {wave_history_path}")
        else:
            wave_history = pd.read_csv(wave_history_path)
            if 'date' not in wave_history.columns:
                print("❌ wave_history missing 'date' column")
            else:
                wave_history['date'] = pd.to_datetime(wave_history['date'])
                wave_history_max_date = wave_history['date'].max().strftime('%Y-%m-%d')
                print(f"   wave_history max date: {wave_history_max_date}")
                
                if price_book_max_date == wave_history_max_date:
                    print(f"✅ Dates match! Both at {price_book_max_date}")
                else:
                    print(f"❌ Dates don't match!")
                    print(f"   price_book: {price_book_max_date}")
                    print(f"   wave_history: {wave_history_max_date}")
except Exception as e:
    print(f"❌ Failed to compare dates: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 4: Test force_ledger_recompute function
print("Test 4: Test force_ledger_recompute() function")
print("-" * 80)
try:
    success, message, details = force_ledger_recompute()
    
    if success:
        print("✅ force_ledger_recompute() succeeded")
        print(f"   Message (first 200 chars): {message[:200]}...")
        print(f"   Details keys: {list(details.keys())}")
        
        # Verify critical details
        if 'ledger_max_date' in details:
            print(f"   Ledger max date: {details['ledger_max_date']}")
        if 'price_book_max_date' in details and 'wave_history_max_date' in details:
            if details['price_book_max_date'] == details['wave_history_max_date']:
                print(f"   ✅ price_book and wave_history dates aligned: {details['price_book_max_date']}")
            else:
                print(f"   ⚠️ Dates differ: price_book={details['price_book_max_date']}, wave_history={details['wave_history_max_date']}")
    else:
        print(f"❌ force_ledger_recompute() failed: {message}")
        if 'traceback' in details:
            print("   Full traceback:")
            print(details['traceback'])
except Exception as e:
    print(f"❌ force_ledger_recompute() raised exception: {e}")
    import traceback
    traceback.print_exc()
print()

# Final summary
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
print()
print("Summary:")
print("1. ✅ Functions are importable without name collision")
print("2. Check rebuild_wave_history() output above")
print("3. Check date matching output above")
print("4. Check force_ledger_recompute() output above")
print()
print("If all tests show ✅, the fixes are working correctly!")
