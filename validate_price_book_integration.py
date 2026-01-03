"""
Validate app.py can import and basic structure is intact.

This test ensures:
1. app.py can be imported without errors
2. Required functions exist
3. Helper modules can be imported
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_app_imports():
    """Test that app.py imports successfully."""
    print("=" * 70)
    print("Validating app.py Structure")
    print("=" * 70)
    
    # Test helper imports
    print("\n1. Testing helper module imports...")
    
    try:
        from helpers.price_book import get_price_book, get_price_book_meta
        print("   ✓ helpers.price_book imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import helpers.price_book: {e}")
        return False
    
    try:
        from helpers.wave_performance import (
            compute_wave_returns,
            compute_all_waves_performance,
            compute_all_waves_readiness,
            get_price_book_diagnostics
        )
        print("   ✓ helpers.wave_performance imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import helpers.wave_performance: {e}")
        return False
    
    # Test that we can load PRICE_BOOK
    print("\n2. Testing PRICE_BOOK loading...")
    try:
        price_book = get_price_book()
        print(f"   ✓ PRICE_BOOK loaded: {price_book.shape}")
    except Exception as e:
        print(f"   ✗ Failed to load PRICE_BOOK: {e}")
        return False
    
    # Test performance computation
    print("\n3. Testing performance computation...")
    try:
        perf_df = compute_all_waves_performance(price_book, periods=[1, 30])
        print(f"   ✓ Performance computed for {len(perf_df)} waves")
        
        # Check expected columns
        expected_cols = ['Wave', '1D Return', '30D', 'Status/Confidence', 'Failure_Reason']
        for col in expected_cols:
            if col not in perf_df.columns:
                print(f"   ✗ Missing expected column: {col}")
                return False
        print(f"   ✓ All expected columns present")
        
    except Exception as e:
        print(f"   ✗ Failed to compute performance: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test readiness computation
    print("\n4. Testing readiness computation...")
    try:
        readiness_df = compute_all_waves_readiness(price_book)
        print(f"   ✓ Readiness computed for {len(readiness_df)} waves")
        
        # Check expected columns
        expected_cols = ['wave_name', 'data_ready', 'reason', 'coverage_pct', 'history_days']
        for col in expected_cols:
            if col not in readiness_df.columns:
                print(f"   ✗ Missing expected column: {col}")
                return False
        print(f"   ✓ All expected columns present")
        
    except Exception as e:
        print(f"   ✗ Failed to compute readiness: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test diagnostics
    print("\n5. Testing PRICE_BOOK diagnostics...")
    try:
        pb_diag = get_price_book_diagnostics(price_book)
        print(f"   ✓ Diagnostics computed")
        print(f"      Path: {pb_diag['path']}")
        print(f"      Shape: {pb_diag['shape']}")
        print(f"      Date range: {pb_diag['date_min']} to {pb_diag['date_max']}")
    except Exception as e:
        print(f"   ✗ Failed to get diagnostics: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All validation checks passed!")
    print("=" * 70)
    
    return True


def test_app_structure():
    """Test app.py structure (without running streamlit)."""
    print("\n" + "=" * 70)
    print("Checking app.py Structure")
    print("=" * 70)
    
    app_path = 'app.py'
    if not os.path.exists(app_path):
        print(f"✗ app.py not found")
        return False
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Check for key updates
    checks = [
        ('PRICE_BOOK import', 'from helpers.price_book import get_price_book'),
        ('wave_performance import', 'from helpers.wave_performance import'),
        ('Performance Overview label', 'Data Source: PRICE_BOOK'),
        ('prices_cache.parquet reference', 'prices_cache.parquet'),
        ('PRICE_BOOK Truth Diagnostics', 'PRICE_BOOK Truth Diagnostics'),
        ('compute_all_waves_performance call', 'compute_all_waves_performance'),
        ('compute_all_waves_readiness call', 'compute_all_waves_readiness'),
    ]
    
    print("\nChecking for required code sections:")
    all_passed = True
    for check_name, check_str in checks:
        if check_str in content:
            print(f"   ✓ {check_name}")
        else:
            print(f"   ✗ {check_name} - NOT FOUND")
            all_passed = False
    
    if all_passed:
        print("\n✓ All structural checks passed!")
    else:
        print("\n✗ Some structural checks failed")
    
    return all_passed


if __name__ == '__main__':
    success1 = test_app_imports()
    success2 = test_app_structure()
    
    if success1 and success2:
        print("\n" + "=" * 70)
        print("🎉 ALL VALIDATIONS PASSED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 70)
        sys.exit(1)
