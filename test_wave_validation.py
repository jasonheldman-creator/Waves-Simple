"""
Test Wave Validation Functions

This test suite validates the wave validation functionality:
1. validate_wave_price_history() - validates individual waves
2. generate_wave_validation_report() - generates comprehensive reports
3. compute_all_waves_performance() with only_validated=True - filters invalid waves
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_price_book(tickers, days=100, start_date='2024-01-01'):
    """Create a test PRICE_BOOK with specified tickers."""
    dates = pd.date_range(start=start_date, periods=days, freq='B')  # Business days
    data = {}
    
    for ticker in tickers:
        # Create random price data with upward trend
        base_price = 100
        trend = np.linspace(0, 20, days)
        noise = np.random.randn(days) * 2
        data[ticker] = base_price + trend + noise
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

def test_validation_with_complete_data():
    """Test validation when all required tickers are present."""
    print("\n" + "=" * 70)
    print("Test 1: Validation with Complete Data")
    print("=" * 70)
    
    # Import directly from module to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from wave_performance import validate_wave_price_history
    
    # Create test PRICE_BOOK with all tickers for S&P 500 Wave
    test_tickers = ['SPY']  # S&P 500 Wave only needs SPY
    price_book = create_test_price_book(test_tickers, days=60)
    
    print(f"\nTest PRICE_BOOK created:")
    print(f"  - Tickers: {list(price_book.columns)}")
    print(f"  - Days: {len(price_book)}")
    print(f"  - Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    
    # Validate S&P 500 Wave (should pass)
    validation = validate_wave_price_history('S&P 500 Wave', price_book, min_history_days=30)
    
    print(f"\nValidation Result:")
    print(f"  - Valid: {validation['valid']}")
    print(f"  - Return Computable: {validation['return_computable']}")
    print(f"  - Validation Reason: {validation['validation_reason']}")
    print(f"  - Required Tickers: {validation['required_tickers']}")
    print(f"  - Found Tickers: {validation['found_tickers']}")
    print(f"  - Missing Tickers: {validation['missing_tickers']}")
    print(f"  - Coverage: {validation['coverage_pct']:.1f}%")
    
    # Assertions
    assert validation['valid'] == True, "Validation should pass with complete data"
    assert validation['return_computable'] == 'Yes', "Returns should be computable"
    assert len(validation['missing_tickers']) == 0, "No tickers should be missing"
    assert validation['coverage_pct'] == 100.0, "Coverage should be 100%"
    
    print("\n✓ Test 1 PASSED: Wave validated successfully with complete data")

def test_validation_with_missing_tickers():
    """Test validation when some required tickers are missing."""
    print("\n" + "=" * 70)
    print("Test 2: Validation with Missing Tickers")
    print("=" * 70)
    
    # Import directly from module to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from wave_performance import validate_wave_price_history
    
    # Create test PRICE_BOOK without all required tickers for AI & Cloud MegaCap Wave
    # AI & Cloud MegaCap Wave requires: ADBE, AMD, AVGO, CRM, GOOGL, INTC, META, MSFT, NVDA, ORCL
    # We'll only include half of them
    test_tickers = ['MSFT', 'NVDA', 'GOOGL', 'META', 'AMD']
    price_book = create_test_price_book(test_tickers, days=60)
    
    print(f"\nTest PRICE_BOOK created:")
    print(f"  - Tickers: {list(price_book.columns)}")
    print(f"  - Days: {len(price_book)}")
    
    # Validate AI & Cloud MegaCap Wave (should fail due to missing tickers)
    validation = validate_wave_price_history('AI & Cloud MegaCap Wave', price_book)
    
    print(f"\nValidation Result:")
    print(f"  - Valid: {validation['valid']}")
    print(f"  - Return Computable: {validation['return_computable']}")
    print(f"  - Validation Reason: {validation['validation_reason']}")
    print(f"  - Required Tickers: {validation['required_tickers']}")
    print(f"  - Found Tickers: {validation['found_tickers']}")
    print(f"  - Missing Tickers: {validation['missing_tickers']}")
    print(f"  - Coverage: {validation['coverage_pct']:.1f}%")
    
    # Assertions
    assert validation['valid'] == False, "Validation should fail with missing tickers"
    assert validation['return_computable'] == 'No', "Returns should not be computable"
    assert len(validation['missing_tickers']) > 0, "Some tickers should be missing"
    assert validation['coverage_pct'] < 100.0, "Coverage should be less than 100%"
    assert 'Insufficient ticker coverage' in validation['validation_reason'], \
        "Validation reason should mention insufficient coverage"
    
    print("\n✓ Test 2 PASSED: Wave correctly failed validation due to missing tickers")

def test_validation_with_insufficient_history():
    """Test validation when date coverage is insufficient."""
    print("\n" + "=" * 70)
    print("Test 3: Validation with Insufficient History")
    print("=" * 70)
    
    # Import directly from module to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from wave_performance import validate_wave_price_history
    
    # Create test PRICE_BOOK with only 10 days of data (less than minimum 30)
    test_tickers = ['SPY']
    price_book = create_test_price_book(test_tickers, days=10)
    
    print(f"\nTest PRICE_BOOK created:")
    print(f"  - Tickers: {list(price_book.columns)}")
    print(f"  - Days: {len(price_book)}")
    print(f"  - Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    
    # Validate S&P 500 Wave with min_history_days=30 (should fail)
    validation = validate_wave_price_history('S&P 500 Wave', price_book, min_history_days=30)
    
    print(f"\nValidation Result:")
    print(f"  - Valid: {validation['valid']}")
    print(f"  - Return Computable: {validation['return_computable']}")
    print(f"  - Validation Reason: {validation['validation_reason']}")
    print(f"  - Date Coverage Days: {validation['date_coverage_days']}")
    
    # Assertions
    assert validation['valid'] == False, "Validation should fail with insufficient history"
    assert validation['return_computable'] == 'No', "Returns should not be computable"
    assert validation['date_coverage_days'] < 30, "Should have less than 30 days"
    assert 'Insufficient date coverage' in validation['validation_reason'], \
        "Validation reason should mention insufficient date coverage"
    
    print("\n✓ Test 3 PASSED: Wave correctly failed validation due to insufficient history")

def test_validation_with_relaxed_criteria():
    """Test validation with relaxed criteria (partial coverage allowed)."""
    print("\n" + "=" * 70)
    print("Test 4: Validation with Relaxed Criteria")
    print("=" * 70)
    
    # Import directly from module to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from wave_performance import validate_wave_price_history
    
    # Create test PRICE_BOOK with 90% of required tickers
    # AI & Cloud MegaCap Wave requires 10 tickers, we'll provide 9
    test_tickers = ['MSFT', 'NVDA', 'GOOGL', 'META', 'AMD', 'ADBE', 'CRM', 'INTC', 'ORCL']
    price_book = create_test_price_book(test_tickers, days=60)
    
    print(f"\nTest PRICE_BOOK created:")
    print(f"  - Tickers: {list(price_book.columns)}")
    print(f"  - Days: {len(price_book)}")
    
    # Validate with min_coverage_pct=90.0 (should pass)
    validation = validate_wave_price_history(
        'AI & Cloud MegaCap Wave',
        price_book,
        min_coverage_pct=90.0,
        min_history_days=30
    )
    
    print(f"\nValidation Result (90% min coverage):")
    print(f"  - Valid: {validation['valid']}")
    print(f"  - Return Computable: {validation['return_computable']}")
    print(f"  - Coverage: {validation['coverage_pct']:.1f}%")
    print(f"  - Missing Tickers: {validation['missing_tickers']}")
    
    # Assertions
    assert validation['valid'] == True, "Validation should pass with 90% coverage requirement"
    assert validation['return_computable'] == 'Yes', "Returns should be computable"
    assert validation['coverage_pct'] >= 90.0, "Coverage should be at least 90%"
    
    print("\n✓ Test 4 PASSED: Wave validated successfully with relaxed criteria")

def test_performance_filtering():
    """Test that compute_all_waves_performance filters invalid waves when only_validated=True."""
    print("\n" + "=" * 70)
    print("Test 5: Performance Table Filtering")
    print("=" * 70)
    
    # Import directly from module to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from wave_performance import compute_all_waves_performance
    
    # Create test PRICE_BOOK with only a few tickers (most waves will fail validation)
    test_tickers = ['SPY', 'QQQ', 'GLD', 'BTC-USD']
    price_book = create_test_price_book(test_tickers, days=60)
    
    print(f"\nTest PRICE_BOOK created:")
    print(f"  - Tickers: {list(price_book.columns)}")
    print(f"  - Days: {len(price_book)}")
    
    # Compute performance with only_validated=True (default)
    print("\nComputing performance with only_validated=True...")
    perf_validated = compute_all_waves_performance(
        price_book,
        periods=[1, 30],
        only_validated=True,
        min_coverage_pct=100.0,
        min_history_days=30
    )
    
    print(f"\nResults with only_validated=True:")
    print(f"  - Waves in table: {len(perf_validated)}")
    if not perf_validated.empty:
        print(f"  - Waves: {list(perf_validated['Wave'])}")
    
    # Compute performance with only_validated=False (show all)
    print("\nComputing performance with only_validated=False...")
    perf_all = compute_all_waves_performance(
        price_book,
        periods=[1, 30],
        only_validated=False
    )
    
    print(f"\nResults with only_validated=False:")
    print(f"  - Waves in table: {len(perf_all)}")
    
    # Count waves with failures
    failed_count = perf_all['Failure_Reason'].notna().sum()
    print(f"  - Waves with failures: {failed_count}")
    
    # Assertions
    assert len(perf_validated) < len(perf_all), \
        "Validated table should have fewer waves than unfiltered table"
    assert len(perf_validated) > 0, \
        "Should have at least some valid waves (e.g., S&P 500 Wave with just SPY)"
    assert 'Failure_Reason' not in perf_validated.columns or perf_validated['Failure_Reason'].isna().all(), \
        "Validated table should not have failure reasons (all waves valid)"
    
    print(f"\n✓ Test 5 PASSED: Performance filtering works correctly")
    print(f"  - Validated table: {len(perf_validated)} waves (only valid)")
    print(f"  - Unfiltered table: {len(perf_all)} waves ({failed_count} with failures)")

def test_validation_report_generation():
    """Test the validation report generation."""
    print("\n" + "=" * 70)
    print("Test 6: Validation Report Generation")
    print("=" * 70)
    
    # Import directly from module to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers')
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from wave_performance import generate_wave_validation_report
    
    # Create test PRICE_BOOK with mixed coverage
    test_tickers = ['SPY', 'QQQ', 'GLD', 'BTC-USD', 'ETH-USD', 'NVDA', 'MSFT']
    price_book = create_test_price_book(test_tickers, days=60)
    
    print(f"\nTest PRICE_BOOK created:")
    print(f"  - Tickers: {list(price_book.columns)}")
    print(f"  - Days: {len(price_book)}")
    
    # Generate validation report
    print("\nGenerating validation report...")
    report = generate_wave_validation_report(
        price_book,
        min_coverage_pct=100.0,
        min_history_days=30,
        output_file=None  # Don't save to file in test
    )
    
    print(f"\nReport Summary:")
    print(f"  - Timestamp: {report['timestamp']}")
    print(f"  - Waves Validated: {report['waves_validated']}")
    print(f"  - Waves Valid: {report['waves_valid']}")
    print(f"  - Waves Invalid: {report['waves_invalid']}")
    print(f"  - Summary: {report['summary']}")
    
    # Check PRICE_BOOK metadata in report
    pb_meta = report['price_book_meta']
    print(f"\nPRICE_BOOK Metadata in Report:")
    print(f"  - Date Range: {pb_meta['date_min']} to {pb_meta['date_max']}")
    print(f"  - Total Tickers: {pb_meta['tickers_count']}")
    print(f"  - Trading Days: {pb_meta['rows']}")
    
    # Assertions
    assert 'timestamp' in report, "Report should have timestamp"
    assert 'waves_validated' in report, "Report should have waves_validated count"
    assert 'validation_results' in report, "Report should have validation_results"
    assert len(report['validation_results']) == report['waves_validated'], \
        "Should have validation result for each wave"
    assert report['waves_valid'] + report['waves_invalid'] == report['waves_validated'], \
        "Valid + Invalid should equal Total"
    
    # Check that some waves passed and some failed
    assert report['waves_valid'] > 0, "Should have at least some valid waves"
    assert report['waves_invalid'] > 0, "Should have at least some invalid waves (with limited tickers)"
    
    print(f"\n✓ Test 6 PASSED: Validation report generated successfully")

def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("RUNNING WAVE VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        test_validation_with_complete_data,
        test_validation_with_missing_tickers,
        test_validation_with_insufficient_history,
        test_validation_with_relaxed_criteria,
        test_performance_filtering,
        test_validation_report_generation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {str(e)}")
        except Exception as e:
            failed += 1
            print(f"\n✗ TEST ERROR: {test_func.__name__}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\nTotal Tests: {len(tests)}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return True
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
