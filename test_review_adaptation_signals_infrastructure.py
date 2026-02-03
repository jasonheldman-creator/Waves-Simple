"""
Test for Review & Adaptation Signals rendering infrastructure

This test validates that the rendering infrastructure is properly configured
and can be called without errors.
"""

import pandas as pd
import numpy as np


def test_diagnostics_review_signals_import():
    """Test that the diagnostics_review_signals helper can be imported."""
    try:
        from helpers import diagnostics_review_signals
        assert diagnostics_review_signals is not None
        assert hasattr(diagnostics_review_signals, 'render_review_and_adaptation_signals')
        print("✓ diagnostics_review_signals module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import diagnostics_review_signals: {e}")
        return False


def test_adaptive_intelligence_tab_function():
    """Test that the render_adaptive_intelligence_tab function exists."""
    try:
        import adaptive_intelligence
        assert hasattr(adaptive_intelligence, 'render_adaptive_intelligence_tab')
        print("✓ render_adaptive_intelligence_tab function exists")
        return True
    except (ImportError, AssertionError) as e:
        print(f"✗ Failed to verify render_adaptive_intelligence_tab: {e}")
        return False


def test_helper_function_signature():
    """Test that the helper function has the expected signature."""
    try:
        from helpers import diagnostics_review_signals
        import inspect
        
        sig = inspect.signature(diagnostics_review_signals.render_review_and_adaptation_signals)
        params = list(sig.parameters.keys())
        
        expected_params = ['snapshot_df', 'attrib_df', 'adaptive_state']
        assert params == expected_params, f"Expected {expected_params}, got {params}"
        
        print(f"✓ Helper function has correct signature: {params}")
        return True
    except Exception as e:
        print(f"✗ Failed to verify function signature: {e}")
        return False


def test_module_structure():
    """Test that the module structure is sound."""
    try:
        import adaptive_intelligence
        
        # Check that both rendering functions exist
        assert hasattr(adaptive_intelligence, 'render_alpha_quality_and_confidence')
        assert hasattr(adaptive_intelligence, 'render_adaptive_intelligence_tab')
        
        print("✓ Adaptive intelligence module structure is correct")
        return True
    except Exception as e:
        print(f"✗ Module structure test failed: {e}")
        return False


if __name__ == '__main__':
    print("Testing Review & Adaptation Signals rendering infrastructure...\n")
    
    results = []
    results.append(("Import diagnostics_review_signals", test_diagnostics_review_signals_import()))
    results.append(("Verify render_adaptive_intelligence_tab", test_adaptive_intelligence_tab_function()))
    results.append(("Check function signature", test_helper_function_signature()))
    results.append(("Verify module structure", test_module_structure()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        exit(1)
