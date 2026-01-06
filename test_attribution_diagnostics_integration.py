"""
Integration test for Attribution Diagnostics - validates function behavior.
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()

def test_compute_alpha_source_breakdown_diagnostics():
    """Test that compute_alpha_source_breakdown properly extracts diagnostics."""
    
    print("Testing compute_alpha_source_breakdown with diagnostics...")
    
    # Import the function
    from app import compute_alpha_source_breakdown
    
    # Create mock data
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'value': np.random.randn(100)
    })
    
    # Mock the necessary dependencies
    with patch('helpers.price_book.get_price_book') as mock_get_price_book, \
         patch('helpers.wave_performance.compute_portfolio_alpha_attribution') as mock_attribution:
        
        # Setup mock price book
        dates = pd.date_range('2024-01-01', periods=100)
        mock_price_book = pd.DataFrame({
            'SPY': np.random.randn(100).cumsum() + 100,
            'AAPL': np.random.randn(100).cumsum() + 150,
        }, index=dates)
        mock_get_price_book.return_value = mock_price_book
        
        # Setup mock attribution result
        mock_daily_exposure = pd.Series(1.0, index=dates)
        mock_daily_realized = pd.Series(np.random.randn(100) * 0.01, index=dates)
        mock_daily_unoverlay = pd.Series(np.random.randn(100) * 0.01, index=dates)
        mock_daily_benchmark = pd.Series(np.random.randn(100) * 0.01, index=dates)
        
        mock_attribution_result = {
            'success': True,
            'daily_exposure': mock_daily_exposure,
            'daily_realized_return': mock_daily_realized,
            'daily_unoverlay_return': mock_daily_unoverlay,
            'daily_benchmark_return': mock_daily_benchmark,
            'using_fallback_exposure': True,
            'period_summaries': {
                '60D': {
                    'period': 60,
                    'cum_real': 0.05,
                    'cum_sel': 0.04,
                    'cum_bm': 0.03,
                    'total_alpha': 0.02,
                    'selection_alpha': 0.01,
                    'overlay_alpha': 0.01,
                    'residual': 0.0
                }
            }
        }
        mock_attribution.return_value = mock_attribution_result
        
        # Call the function
        result = compute_alpha_source_breakdown(test_df)
        
        # Verify the result structure
        assert 'diagnostics' in result, "Result should contain 'diagnostics' key"
        assert result['data_available'] == True, "Data should be available"
        
        diagnostics = result['diagnostics']
        
        # Verify all diagnostic fields are present
        required_fields = [
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
        
        for field in required_fields:
            assert field in diagnostics, f"Diagnostic field '{field}' should be present"
            print(f"  ✓ {field}: {diagnostics[field]}")
        
        # Verify specific values
        assert diagnostics['period_used'] == '60D', "Should use 60D period"
        assert diagnostics['using_fallback_exposure'] == True, "Should indicate fallback exposure"
        assert diagnostics['exposure_series_found'] == True, "Should find exposure series"
        assert diagnostics['exposure_min'] == 1.0, "Min exposure should be 1.0"
        assert diagnostics['exposure_max'] == 1.0, "Max exposure should be 1.0"
        assert diagnostics['cum_realized'] == 0.05, "Should have cum_realized value"
        assert diagnostics['cum_unoverlay'] == 0.04, "Should have cum_unoverlay value"
        assert diagnostics['cum_benchmark'] == 0.03, "Should have cum_benchmark value"
        
        print("\n✅ All diagnostic fields verified!")
        return True


def test_periods_parameter():
    """Verify that the function calls compute_portfolio_alpha_attribution with periods=[60]."""
    
    print("\nTesting that periods=[60] is used...")
    
    from app import compute_alpha_source_breakdown
    
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'value': np.random.randn(100)
    })
    
    with patch('helpers.price_book.get_price_book') as mock_get_price_book, \
         patch('helpers.wave_performance.compute_portfolio_alpha_attribution') as mock_attribution:
        
        # Setup mocks
        dates = pd.date_range('2024-01-01', periods=100)
        mock_price_book = pd.DataFrame({
            'SPY': np.random.randn(100).cumsum() + 100,
        }, index=dates)
        mock_get_price_book.return_value = mock_price_book
        
        mock_attribution.return_value = {'success': False}
        
        # Call the function
        compute_alpha_source_breakdown(test_df)
        
        # Verify that compute_portfolio_alpha_attribution was called with periods=[60]
        mock_attribution.assert_called_once()
        call_args = mock_attribution.call_args
        
        # Check that periods parameter is [60]
        assert 'periods' in call_args[1], "Should pass 'periods' parameter"
        assert call_args[1]['periods'] == [60], f"Should pass periods=[60], got {call_args[1]['periods']}"
        
        print("  ✓ Function calls compute_portfolio_alpha_attribution with periods=[60]")
        print("\n✅ Period parameter verification passed!")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("Attribution Diagnostics Integration Test")
    print("=" * 60)
    print()
    
    try:
        test1_passed = test_compute_alpha_source_breakdown_diagnostics()
        test2_passed = test_periods_parameter()
        
        print()
        print("=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
