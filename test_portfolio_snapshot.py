#!/usr/bin/env python3
"""
Test suite for portfolio snapshot and alpha attribution.

This test validates the requirements from the problem statement:
1. Portfolio snapshot populates with 1D/30D/60D metrics when 60+ days exist
2. Alpha attribution outputs non-null numeric values
3. Wave-level snapshot and attribution filled for minimum 3 waves
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_portfolio_snapshot_basic():
    """Test basic portfolio snapshot computation."""
    print("\n=== Test: Portfolio Snapshot Basic ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_snapshot
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days × {len(price_book.columns)} tickers")
        
        # Compute portfolio snapshot
        print("\nComputing portfolio snapshot...")
        snapshot = compute_portfolio_snapshot(price_book, mode='Standard', periods=[1, 30, 60, 365])
        
        # Validate results
        if not snapshot['success']:
            print(f"❌ FAIL: Snapshot computation failed: {snapshot['failure_reason']}")
            return False
        
        print(f"✓ Snapshot computation succeeded")
        print(f"  - Wave count: {snapshot['wave_count']}")
        print(f"  - Date range: {snapshot['date_range']}")
        print(f"  - Latest date: {snapshot['latest_date']}")
        print(f"  - Data age: {snapshot['data_age_days']} days")
        
        # Check for required series
        print("\nValidating required series:")
        print(f"  - Portfolio returns series: {snapshot['has_portfolio_returns_series']}")
        print(f"  - Portfolio benchmark series: {snapshot['has_portfolio_benchmark_series']}")
        print(f"  - Overlay alpha series: {snapshot['has_overlay_alpha_series']}")
        
        if not snapshot['has_portfolio_returns_series']:
            print("❌ FAIL: Portfolio returns series missing")
            return False
        
        if not snapshot['has_portfolio_benchmark_series']:
            print("❌ FAIL: Portfolio benchmark series missing")
            return False
        
        # Check returns for each period
        print("\nPortfolio Returns:")
        for period in [1, 30, 60, 365]:
            key = f'{period}D'
            ret = snapshot['portfolio_returns'][key]
            bench = snapshot['benchmark_returns'][key]
            alpha = snapshot['alphas'][key]
            
            if ret is not None:
                print(f"  {key:4s}: {ret:+.2%} (Benchmark: {bench:+.2%}, Alpha: {alpha:+.2%})")
            else:
                print(f"  {key:4s}: N/A (insufficient history)")
        
        # Test requirement: 1D/30D/60D metrics should exist if 60+ days available
        total_days = len(price_book)
        if total_days >= 60:
            if snapshot['portfolio_returns']['1D'] is None:
                print("❌ FAIL: 1D return is None despite having 60+ days")
                return False
            if snapshot['portfolio_returns']['30D'] is None:
                print("❌ FAIL: 30D return is None despite having 60+ days")
                return False
            if snapshot['portfolio_returns']['60D'] is None:
                print("❌ FAIL: 60D return is None despite having 60+ days")
                return False
            
            print("✓ All required metrics (1D/30D/60D) populated with 60+ days of data")
        else:
            print(f"⚠ Warning: Only {total_days} days available (need 60 for full validation)")
        
        print("\n✓ PASS: Portfolio snapshot basic test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_attribution():
    """Test alpha attribution computation."""
    print("\n=== Test: Alpha Attribution ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded")
        
        # Compute alpha attribution
        print("\nComputing alpha attribution...")
        attribution = compute_portfolio_alpha_attribution(price_book, mode='Standard', min_waves=3)
        
        # Validate results
        if not attribution['success']:
            print(f"❌ FAIL: Attribution computation failed: {attribution['failure_reason']}")
            return False
        
        print(f"✓ Attribution computation succeeded")
        print(f"  - Wave count: {attribution['wave_count']}")
        
        # Test requirement: Alpha attribution outputs non-null numeric values
        print("\nAlpha Attribution Values:")
        print(f"  - Cumulative alpha: {attribution['cumulative_alpha']}")
        print(f"  - Selection alpha: {attribution['selection_alpha']}")
        print(f"  - Overlay alpha: {attribution['overlay_alpha']}")
        
        if attribution['cumulative_alpha'] is None:
            print("❌ FAIL: Cumulative alpha is None")
            return False
        
        if attribution['selection_alpha'] is None:
            print("❌ FAIL: Selection alpha is None")
            return False
        
        if attribution['overlay_alpha'] is None:
            print("❌ FAIL: Overlay alpha is None")
            return False
        
        # Verify they are numeric
        try:
            float(attribution['cumulative_alpha'])
            float(attribution['selection_alpha'])
            float(attribution['overlay_alpha'])
        except (TypeError, ValueError):
            print("❌ FAIL: Alpha values are not numeric")
            return False
        
        print("✓ All alpha values are non-null and numeric")
        
        # Test requirement: minimum 3 waves
        if attribution['wave_count'] < 3:
            print(f"❌ FAIL: Wave count {attribution['wave_count']} < 3")
            return False
        
        print(f"✓ Wave count {attribution['wave_count']} >= 3")
        
        print("\n✓ PASS: Alpha attribution test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostics_validation():
    """Test diagnostics validation."""
    print("\n=== Test: Diagnostics Validation ===")
    
    try:
        from helpers.wave_performance import validate_portfolio_diagnostics
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded")
        
        # Run diagnostics
        print("\nRunning diagnostics validation...")
        diagnostics = validate_portfolio_diagnostics(price_book, mode='Standard')
        
        # Display diagnostics
        print(f"\nDiagnostics Results:")
        print(f"  - Latest date: {diagnostics['latest_date']}")
        print(f"  - Data age: {diagnostics['data_age_days']} days")
        print(f"  - Data quality: {diagnostics['data_quality']}")
        print(f"  - Wave count: {diagnostics['wave_count']}")
        print(f"  - Min history days: {diagnostics['min_history_days']}")
        print(f"  - Has portfolio returns: {diagnostics['has_portfolio_returns_series']}")
        print(f"  - Has benchmark: {diagnostics['has_portfolio_benchmark_series']}")
        print(f"  - Has overlay alpha: {diagnostics['has_overlay_alpha_series']}")
        
        if diagnostics['issues']:
            print(f"\n  Issues found:")
            for issue in diagnostics['issues']:
                print(f"    - {issue}")
        else:
            print(f"\n  No issues found")
        
        # Validate that key fields are populated
        if diagnostics['latest_date'] is None:
            print("❌ FAIL: Latest date is None")
            return False
        
        if diagnostics['data_age_days'] is None:
            print("❌ FAIL: Data age is None")
            return False
        
        print("✓ Key diagnostic fields populated")
        
        print("\n✓ PASS: Diagnostics validation test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wave_level_snapshot():
    """Test wave-level snapshot for minimum 3 waves."""
    print("\n=== Test: Wave-Level Snapshot ===")
    
    try:
        from helpers.wave_performance import compute_wave_returns
        from helpers.price_book import get_price_book
        from waves_engine import get_all_waves_universe
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded")
        
        # Get all waves
        print("\nGetting wave universe...")
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
        
        print(f"✓ Found {len(all_waves)} waves")
        
        # Compute returns for first 5 waves
        print("\nComputing returns for first 5 waves...")
        successful_waves = 0
        
        for i, wave_name in enumerate(all_waves[:5]):
            result = compute_wave_returns(wave_name, price_book, periods=[1, 30, 60])
            
            if result['success']:
                successful_waves += 1
                ret_1d = result['returns'].get('1D')
                ret_30d = result['returns'].get('30D')
                ret_60d = result['returns'].get('60D')
                
                print(f"  {i+1}. {wave_name}: ", end='')
                if ret_1d is not None:
                    print(f"1D={ret_1d:+.2%} ", end='')
                if ret_30d is not None:
                    print(f"30D={ret_30d:+.2%} ", end='')
                if ret_60d is not None:
                    print(f"60D={ret_60d:+.2%}", end='')
                print()
            else:
                print(f"  {i+1}. {wave_name}: FAILED - {result['failure_reason']}")
        
        # Test requirement: minimum 3 waves with valid data
        if successful_waves < 3:
            print(f"\n❌ FAIL: Only {successful_waves} waves succeeded (need 3)")
            return False
        
        print(f"\n✓ {successful_waves} waves have valid snapshot data (>= 3)")
        
        print("\n✓ PASS: Wave-level snapshot test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Portfolio Snapshot and Alpha Attribution Test Suite")
    print("=" * 70)
    
    tests = [
        ("Portfolio Snapshot Basic", test_portfolio_snapshot_basic),
        ("Alpha Attribution", test_alpha_attribution),
        ("Diagnostics Validation", test_diagnostics_validation),
        ("Wave-Level Snapshot", test_wave_level_snapshot),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
