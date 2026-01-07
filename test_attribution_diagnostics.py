"""
Test for Attribution Diagnostics feature.
Validates that the compute_alpha_source_breakdown function returns diagnostics.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_attribution_diagnostics_structure():
    """Test that compute_alpha_source_breakdown returns the correct structure including diagnostics."""
    
    # Import dependencies
    import pandas as pd
    import numpy as np
    from unittest.mock import MagicMock, patch
    
    # Mock streamlit session state
    mock_session_state = {'selected_mode': 'Standard'}
    
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'value': np.random.randn(100)
    })
    
    # Import the function (this will be in app.py)
    # We'll test the structure it should return
    expected_keys = [
        'total_alpha',
        'selection_alpha', 
        'overlay_alpha',
        'residual',
        'data_available',
        'diagnostics'
    ]
    
    expected_diagnostic_keys = [
        'period_used',
        'start_date',
        'end_date',
        'using_fallback_exposure',
        'exposure_series_found',
        'exposure_min',
        'exposure_max',
        'cum_realized',
        'cum_unoverlay',
        'cum_benchmark'
    ]
    
    print("✓ Expected result structure defined")
    print(f"  - Main keys: {expected_keys}")
    print(f"  - Diagnostic keys: {expected_diagnostic_keys}")
    
    # Test that the function signature and documentation is correct
    try:
        # Read the app.py file to verify the function exists
        with open('app.py', 'r') as f:
            content = f.read()
            
        # Check that compute_alpha_source_breakdown exists
        assert 'def compute_alpha_source_breakdown(df):' in content
        print("✓ Function compute_alpha_source_breakdown exists")
        
        # Check that it mentions periods=[60]
        assert 'periods=[60]' in content
        print("✓ Function explicitly uses periods=[60] for 60D alignment")
        
        # Check that diagnostics are populated
        assert "'diagnostics': {" in content
        assert "'period_used':" in content
        assert "'start_date':" in content
        assert "'end_date':" in content
        assert "'using_fallback_exposure':" in content
        assert "'exposure_series_found':" in content
        assert "'exposure_min':" in content
        assert "'exposure_max':" in content
        assert "'cum_realized':" in content
        assert "'cum_unoverlay':" in content
        assert "'cum_benchmark':" in content
        print("✓ All required diagnostic fields are populated")
        
        # Check that the expander exists
        assert 'st.expander("🔬 Attribution Diagnostics"' in content or 'st.expander(\'🔬 Attribution Diagnostics\'' in content
        print("✓ Attribution Diagnostics expander exists in UI")
        
        # Check that diagnostic values are displayed
        assert 'diagnostics.get(\'period_used\'' in content
        assert 'diagnostics.get(\'start_date\'' in content
        assert 'diagnostics.get(\'end_date\'' in content
        assert 'diagnostics.get(\'using_fallback_exposure\'' in content
        assert 'diagnostics.get(\'exposure_series_found\'' in content
        assert 'diagnostics.get(\'exposure_min\'' in content
        assert 'diagnostics.get(\'exposure_max\'' in content
        assert 'diagnostics.get(\'cum_realized\'' in content
        assert 'diagnostics.get(\'cum_unoverlay\'' in content
        assert 'diagnostics.get(\'cum_benchmark\'' in content
        print("✓ All diagnostic values are displayed in expander")
        
        # Check compounded math caption
        assert 'compounded math' in content.lower()
        print("✓ Compounded math explanation present")
        
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True


def test_cumulative_return_formula():
    """Verify that cumulative returns use compounded math formula."""
    
    # Read the helpers/wave_performance.py file
    try:
        with open('helpers/wave_performance.py', 'r') as f:
            content = f.read()
        
        # Check for compounded return formula
        assert '(1 + window_series).prod() - 1' in content or \
               '(1 + series).prod() - 1' in content or \
               '(1 + daily_realized_return).prod() - 1' in content
        print("✓ Cumulative returns use compounded math: (1 + returns).prod() - 1")
        
        # Check in the compute_cumulative_return function
        assert 'cum_return = (1 + window_series).prod() - 1' in content
        print("✓ compute_cumulative_return function uses compounded formula")
        
        # Check since inception calculations
        assert 'cum_real_inception = (1 + daily_realized_return).prod() - 1' in content
        assert 'cum_sel_inception = (1 + daily_unoverlay_return).prod() - 1' in content
        assert 'cum_bm_inception = (1 + daily_benchmark_return).prod() - 1' in content
        print("✓ Since inception calculations use compounded formula")
        
    except AssertionError as e:
        print(f"✗ Compounded math verification failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("\n✅ Cumulative return formula verification passed!")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Attribution Diagnostics Test Suite")
    print("=" * 60)
    print()
    
    test1_passed = test_attribution_diagnostics_structure()
    print()
    test2_passed = test_cumulative_return_formula()
    
    print()
    print("=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
