#!/usr/bin/env python3
"""
Integration test for TruthFrame-based portfolio snapshot.
Tests that the function is properly imported and callable.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_truthframe_import():
    """Test that TruthFrame portfolio snapshot can be imported."""
    print("\n=== Test: TruthFrame Import ===")
    
    try:
        from analytics_truth import compute_portfolio_snapshot_from_truth
        print("✓ Successfully imported compute_portfolio_snapshot_from_truth")
        
        # Check function signature
        import inspect
        sig = inspect.signature(compute_portfolio_snapshot_from_truth)
        params = list(sig.parameters.keys())
        
        print(f"  - Function parameters: {params}")
        
        if 'mode' not in params:
            print("❌ FAIL: Missing 'mode' parameter")
            return False
        
        if 'periods' not in params:
            print("❌ FAIL: Missing 'periods' parameter")
            return False
        
        print("✓ PASS: TruthFrame import test")
        return True
        
    except ImportError as e:
        print(f"❌ FAIL: Could not import: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_truthframe_function_structure():
    """Test that the function returns proper structure."""
    print("\n=== Test: TruthFrame Function Structure ===")
    
    try:
        from analytics_truth import compute_portfolio_snapshot_from_truth
        
        # Call with minimal parameters - will use Safe Mode fallback if needed
        print("Calling compute_portfolio_snapshot_from_truth...")
        result = compute_portfolio_snapshot_from_truth(mode='Standard', periods=(1, 30))
        
        print(f"  - Result type: {type(result)}")
        
        if not isinstance(result, dict):
            print(f"❌ FAIL: Result is not a dict, got {type(result)}")
            return False
        
        print(f"  - Result keys: {list(result.keys())}")
        
        # Check if it has either error or return metrics
        has_error = 'error' in result
        has_returns = any(k.startswith('return_') for k in result.keys())
        has_alphas = any(k.startswith('alpha_') for k in result.keys())
        
        print(f"  - Has error: {has_error}")
        print(f"  - Has returns: {has_returns}")
        print(f"  - Has alphas: {has_alphas}")
        
        if has_error:
            print(f"  - Error message: {result['error']}")
            print("  ⚠ Function returned error (may be expected if no data)")
        elif not has_returns and not has_alphas:
            print("❌ FAIL: Result has neither error nor metrics")
            return False
        
        # Check for timestamp
        if 'computed_at_utc' in result:
            print(f"  - Timestamp: {result['computed_at_utc']}")
        
        print("✓ PASS: TruthFrame function structure test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_integration():
    """Test that app.py properly imports the TruthFrame function."""
    print("\n=== Test: App Integration ===")
    
    try:
        # Check if app.py has the right imports
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        # Check for new import
        if 'from analytics_truth import compute_portfolio_snapshot_from_truth' not in app_content:
            print("❌ FAIL: app.py missing TruthFrame import")
            return False
        
        print("✓ app.py has TruthFrame import")
        
        # Check for deprecation of legacy function
        if '_legacy_compute_portfolio_snapshot' not in app_content:
            print("⚠ WARNING: app.py doesn't rename legacy function")
        else:
            print("✓ app.py renames legacy function to _legacy_*")
        
        # Check for TRUTHFRAME_PORTFOLIO_AVAILABLE flag
        if 'TRUTHFRAME_PORTFOLIO_AVAILABLE' not in app_content:
            print("❌ FAIL: app.py missing TRUTHFRAME_PORTFOLIO_AVAILABLE flag")
            return False
        
        print("✓ app.py has TRUTHFRAME_PORTFOLIO_AVAILABLE flag")
        
        # Check that the new function is being called
        if 'compute_portfolio_snapshot_from_truth' not in app_content.split('from analytics_truth import')[1]:
            print("⚠ WARNING: TruthFrame function may not be called in app.py")
        else:
            print("✓ TruthFrame function appears to be used in app.py")
        
        # Check for deprecation comments
        if 'DEPRECATED' in app_content and 'PRICE_BOOK' in app_content:
            print("✓ app.py has deprecation comments for legacy code")
        
        print("✓ PASS: App integration test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("TruthFrame Portfolio Snapshot Integration Tests")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("TruthFrame Import", test_truthframe_import()))
    results.append(("TruthFrame Function Structure", test_truthframe_function_structure()))
    results.append(("App Integration", test_app_integration()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if total_passed == len(results) else 1)
