"""
Test script to validate Executive Tab enhancements.

Tests:
1. Exposure Series Fallback - verify fallback flag is set
2. Capital-Weighted Alpha - verify N/A display logic  
3. Executive Intelligence Summary - verify metric extraction
4. Top Performing Strategies - verify ranking logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_exposure_fallback():
    """Test that fallback exposure flag is set correctly."""
    print("\n=== Test 1: Exposure Series Fallback ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Get price book
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("⚠️  SKIPPED: No price book available")
            return
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60]
        )
        
        if result['success']:
            # Check if fallback flag is set
            using_fallback = result.get('using_fallback_exposure', False)
            print(f"✓ Fallback exposure flag: {using_fallback}")
            
            # Check overlay alpha when fallback is used
            summary_30d = result['period_summaries'].get('30D', {})
            overlay_alpha = summary_30d.get('overlay_alpha', None)
            
            if overlay_alpha is not None:
                print(f"✓ Overlay alpha (30D): {overlay_alpha*100:.2f}%")
                
                # When fallback (exposure=1.0), overlay alpha should be ~0
                if using_fallback:
                    if abs(overlay_alpha) < 0.001:  # Within 0.1%
                        print("✓ PASS: Overlay alpha is ~0.00% when using fallback exposure")
                    else:
                        print(f"⚠️  WARNING: Overlay alpha is {overlay_alpha*100:.2f}% (expected ~0% with fallback)")
            else:
                print("⚠️  WARNING: No overlay alpha computed")
                
            # Check that warnings are not present (fallback is expected)
            warnings = result.get('warnings', [])
            exposure_warnings = [w for w in warnings if 'exposure' in w.lower()]
            
            if using_fallback and len(exposure_warnings) == 0:
                print("✓ PASS: No exposure warnings when using expected fallback")
            elif using_fallback and len(exposure_warnings) > 0:
                print(f"⚠️  WARNING: Found {len(exposure_warnings)} exposure warnings despite fallback")
                for w in exposure_warnings:
                    print(f"   - {w}")
            
        else:
            print(f"⚠️  FAILED: Attribution computation failed - {result.get('failure_reason')}")
            
    except Exception as e:
        print(f"⚠️  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def test_capital_weighted_alpha_logic():
    """Test capital-weighted alpha display logic."""
    print("\n=== Test 2: Capital-Weighted Alpha N/A Logic ===")
    
    # Test data without capital weights (should trigger N/A)
    test_df_no_capital = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'portfolio_return': np.random.randn(30) * 0.01,
        'benchmark_return': np.random.randn(30) * 0.01
    })
    
    # Test data with capital weights
    test_df_with_capital = test_df_no_capital.copy()
    test_df_with_capital['capital_weight'] = np.random.dirichlet(np.ones(30))
    
    try:
        # Import the function
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import compute_capital_weighted_alpha
        
        # Test without capital weights
        result_no_capital = compute_capital_weighted_alpha(test_df_no_capital)
        
        if result_no_capital['data_available']:
            method = result_no_capital['weighting_method']
            print(f"✓ Method without capital inputs: {method}")
            
            if method == 'equal-weight':
                print("✓ PASS: Falls back to equal-weight when no capital inputs")
                print("   → UI should display 'N/A' and helper text")
            else:
                print(f"⚠️  WARNING: Expected 'equal-weight', got '{method}'")
        
        # Test with capital weights
        result_with_capital = compute_capital_weighted_alpha(test_df_with_capital)
        
        if result_with_capital['data_available']:
            method = result_with_capital['weighting_method']
            value = result_with_capital['capital_weighted_alpha']
            print(f"✓ Method with capital inputs: {method}")
            print(f"✓ Value: {value*100:.4f}%")
            
            if method == 'capital':
                print("✓ PASS: Uses capital-weighted method when capital inputs available")
            else:
                print(f"⚠️  WARNING: Expected 'capital', got '{method}'")
        
    except Exception as e:
        print(f"⚠️  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def test_top_performers_ranking():
    """Test top performers ranking logic."""
    print("\n=== Test 3: Top Performing Strategies Ranking ===")
    
    # Create test performance data
    waves = ['Wave A', 'Wave B', 'Wave C', 'Wave D', 'Wave E', 'Wave F']
    test_df = pd.DataFrame({
        'Wave': waves,
        '30D': ['+5%', '+3%', '+8%', '-2%', '+1%', '+6%'],
        '60D': ['+10%', '+7%', '+12%', '-3%', '+2%', '+9%'],
        '1D Return': ['+0.5%', '-0.2%', '+0.8%', '-0.5%', '+0.1%', '+0.3%']
    })
    
    def parse_return_value(val):
        """Helper to parse return strings."""
        if pd.isna(val) or val == "N/A":
            return None
        try:
            return float(str(val).replace('%', '').replace('+', ''))
        except:
            return None
    
    # Test 30D ranking
    test_df['30D_Numeric'] = test_df['30D'].apply(parse_return_value)
    top_30d = test_df.nlargest(5, '30D_Numeric')
    
    print("✓ Top 5 by 30D Alpha:")
    for rank, (_, row) in enumerate(top_30d.iterrows(), 1):
        print(f"   #{rank}: {row['Wave']} - {row['30D']}")
    
    # Verify ranking is correct
    expected_order = ['Wave C', 'Wave F', 'Wave A', 'Wave B', 'Wave E']
    actual_order = top_30d['Wave'].tolist()
    
    if actual_order == expected_order:
        print("✓ PASS: 30D ranking is correct")
    else:
        print(f"⚠️  WARNING: Expected {expected_order}, got {actual_order}")
    
    # Test 60D ranking
    test_df['60D_Numeric'] = test_df['60D'].apply(parse_return_value)
    top_60d = test_df.nlargest(5, '60D_Numeric')
    
    print("\n✓ Top 5 by 60D Alpha:")
    for rank, (_, row) in enumerate(top_60d.iterrows(), 1):
        print(f"   #{rank}: {row['Wave']} - {row['60D']}")
    
    expected_order_60d = ['Wave C', 'Wave A', 'Wave F', 'Wave B', 'Wave E']
    actual_order_60d = top_60d['Wave'].tolist()
    
    if actual_order_60d == expected_order_60d:
        print("✓ PASS: 60D ranking is correct")
    else:
        print(f"⚠️  WARNING: Expected {expected_order_60d}, got {actual_order_60d}")


def test_executive_summary_metrics():
    """Test executive summary metric extraction."""
    print("\n=== Test 4: Executive Intelligence Summary Metrics ===")
    
    # Test data
    test_perf_df = pd.DataFrame({
        'Wave': ['Wave A', 'Wave B', 'Wave C'],
        '1D Return': ['+0.5%', '-0.2%', '+0.3%'],
        '30D': ['+5%', '+3%', '+4%'],
        '60D': ['+10%', '+7%', '+8%'],
        '365D': ['+25%', '+20%', '+22%']
    })
    
    def parse_return(val):
        """Helper to parse return strings."""
        if pd.isna(val) or val == "N/A":
            return None
        try:
            return float(str(val).replace('%', '').replace('+', ''))
        except:
            return None
    
    # Extract metrics
    returns_30d = test_perf_df['30D'].apply(parse_return).dropna()
    returns_60d = test_perf_df['60D'].apply(parse_return).dropna()
    returns_365d = test_perf_df['365D'].apply(parse_return).dropna()
    
    avg_30d = returns_30d.mean() if len(returns_30d) > 0 else None
    avg_60d = returns_60d.mean() if len(returns_60d) > 0 else None
    avg_365d = returns_365d.mean() if len(returns_365d) > 0 else None
    
    print(f"✓ Portfolio 30D Return: {avg_30d:+.1f}%")
    print(f"✓ Portfolio 60D Return: {avg_60d:+.1f}%")
    print(f"✓ Portfolio 365D Return: {avg_365d:+.1f}%")
    
    # Test expected values
    if abs(avg_30d - 4.0) < 0.1:
        print("✓ PASS: 30D average calculated correctly")
    else:
        print(f"⚠️  WARNING: Expected 30D avg ~4.0%, got {avg_30d:.1f}%")
    
    if abs(avg_60d - 8.33) < 0.1:
        print("✓ PASS: 60D average calculated correctly")
    else:
        print(f"⚠️  WARNING: Expected 60D avg ~8.33%, got {avg_60d:.1f}%")
    
    # Test market context formatting
    mock_market_data = {
        'SPY': 0.5,  # +0.5%
        'QQQ': -0.3,  # -0.3%
        'IWM': 0.1,   # +0.1%
        'TLT': 0.2    # +0.2%
    }
    
    market_parts = [f"{ticker} {ret:+.1f}%" for ticker, ret in mock_market_data.items()]
    market_context = "Market: " + ", ".join(market_parts)
    
    print(f"\n✓ Market Context: {market_context}")
    
    expected_context = "Market: SPY +0.5%, QQQ -0.3%, IWM +0.1%, TLT +0.2%"
    if market_context == expected_context:
        print("✓ PASS: Market context formatted correctly")
    else:
        print(f"⚠️  WARNING: Expected '{expected_context}', got '{market_context}'")


if __name__ == "__main__":
    print("=" * 60)
    print("Executive Tab Enhancements - Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_exposure_fallback()
    test_capital_weighted_alpha_logic()
    test_top_performers_ranking()
    test_executive_summary_metrics()
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
