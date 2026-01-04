#!/usr/bin/env python3
"""
Validation script for Executive Dashboard implementation
Tests that the new render_overview_clean_tab() function can execute without errors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dashboard_structure():
    """Test that the dashboard function exists and has correct structure"""
    print("Testing Executive Dashboard structure...")
    
    # Import the app module
    import app
    
    # Check that render_overview_clean_tab exists
    assert hasattr(app, 'render_overview_clean_tab'), "render_overview_clean_tab function not found"
    print("✅ render_overview_clean_tab function exists")
    
    # Check the function has a proper docstring
    func = app.render_overview_clean_tab
    assert func.__doc__ is not None, "Function missing docstring"
    assert "Executive Dashboard" in func.__doc__, "Docstring doesn't mention Executive Dashboard"
    print("✅ Function has proper docstring")
    
    # Check that it mentions all required sections
    required_sections = [
        "Executive Header",
        "KPI Scoreboard",
        "Leaders/Laggards",
        "Alpha Attribution",
        "System Health",
        "Market Context"
    ]
    
    for section in required_sections:
        assert section in func.__doc__, f"Missing section in docstring: {section}"
    print("✅ All required sections mentioned in docstring")
    
    return True


def test_helper_functions():
    """Test that required helper functions are available"""
    print("\nTesting helper function availability...")
    
    try:
        from helpers.price_book import get_price_book, get_price_book_meta
        from helpers.wave_performance import compute_all_waves_performance, get_price_book_diagnostics
        print("✅ All required helper functions available")
        return True
    except ImportError as e:
        print(f"❌ Missing helper function: {e}")
        return False


def test_data_availability():
    """Test that required data sources are available"""
    print("\nTesting data availability...")
    
    try:
        from helpers.price_book import get_price_book, get_price_book_meta
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        assert price_book is not None, "PRICE_BOOK is None"
        assert not price_book.empty, "PRICE_BOOK is empty"
        print(f"✅ PRICE_BOOK available: shape {price_book.shape}")
        
        # Get metadata
        meta = get_price_book_meta(price_book)
        assert meta is not None, "PRICE_BOOK metadata is None"
        print(f"✅ PRICE_BOOK metadata: last_date={meta.get('date_max', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Data availability error: {e}")
        return False


def test_performance_computation():
    """Test that wave performance can be computed"""
    print("\nTesting performance computation...")
    
    try:
        from helpers.price_book import get_price_book
        from helpers.wave_performance import compute_all_waves_performance
        
        price_book = get_price_book()
        
        # Compute performance with multiple periods
        perf_df = compute_all_waves_performance(price_book, periods=[1, 30, 60, 365], only_validated=True)
        
        assert perf_df is not None, "Performance DataFrame is None"
        assert not perf_df.empty, "Performance DataFrame is empty"
        
        print(f"✅ Performance computed: {len(perf_df)} validated waves")
        
        # Check required columns exist
        required_cols = ['Wave', '1D Return', '30D', '60D', '365D', 'Status/Confidence']
        for col in required_cols:
            assert col in perf_df.columns, f"Missing column: {col}"
        
        print("✅ All required columns present")
        
        return True
    except Exception as e:
        print(f"❌ Performance computation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("Executive Dashboard Validation")
    print("=" * 60)
    
    tests = [
        ("Dashboard Structure", test_dashboard_structure),
        ("Helper Functions", test_helper_functions),
        ("Data Availability", test_data_availability),
        ("Performance Computation", test_performance_computation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print("\n⚠️ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
