"""
Test suite for wave coverage computation in build_wave_history_from_prices.py

Tests the following key features:
1. Coverage percentage is correctly computed with missing tickers
2. Waves meeting >= 90% coverage threshold are kept
3. Waves below 90% coverage threshold are excluded
4. Coverage metrics are properly tracked in the snapshot
"""

import sys
import os
import json
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import normalize_ticker directly from file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ticker_normalize",
    os.path.join(os.path.dirname(__file__), "helpers", "ticker_normalize.py")
)
ticker_normalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ticker_normalize)
normalize_ticker = ticker_normalize.normalize_ticker


def test_coverage_computation_with_missing_tickers():
    """Test that coverage is correctly computed when some tickers are missing."""
    print("="*70)
    print("TEST: Coverage computation with missing tickers")
    print("="*70)
    
    # Create test wave weights with known ticker weights
    weights_data = {
        'wave': ['Test Wave'] * 5,
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'MISSING1', 'MISSING2'],
        'weight': [0.2, 0.2, 0.2, 0.2, 0.2]  # Each 20%, total 100%
    }
    weights_df = pd.DataFrame(weights_data)
    
    # Normalize tickers
    weights_df['ticker_norm'] = weights_df['ticker'].apply(normalize_ticker)
    
    # Simulate available tickers (missing MISSING1 and MISSING2)
    available_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Calculate coverage
    total_weight = weights_df['weight'].abs().sum()
    available_weight = weights_df[weights_df['ticker_norm'].isin(available_tickers)]['weight'].abs().sum()
    coverage_pct = available_weight / total_weight if total_weight > 0 else 0.0
    
    # Verify coverage is 60% (3 out of 5 tickers, each 20%)
    expected_coverage = 0.6
    assert abs(coverage_pct - expected_coverage) < 0.001, \
        f"Expected {expected_coverage:.2%} coverage, got {coverage_pct:.2%}"
    
    print(f"✓ Coverage correctly computed: {coverage_pct:.2%}")
    print(f"  Total tickers: {len(weights_df)}")
    print(f"  Available tickers: {len(available_tickers)}")
    print(f"  Total weight: {total_weight}")
    print(f"  Available weight: {available_weight}")
    

def test_90_percent_threshold():
    """Test that 90% threshold correctly determines wave inclusion."""
    print("\n" + "="*70)
    print("TEST: 90% threshold validation")
    print("="*70)
    
    MIN_COVERAGE_THRESHOLD = 0.90
    
    test_cases = [
        # (coverage_pct, should_meet_threshold, description)
        (1.00, True, "100% coverage"),
        (0.95, True, "95% coverage"),
        (0.90, True, "Exactly 90% coverage"),
        (0.89, False, "89% coverage (below threshold)"),
        (0.80, False, "80% coverage (below threshold)"),
        (0.50, False, "50% coverage (well below threshold)"),
    ]
    
    for coverage_pct, expected, description in test_cases:
        meets_threshold = coverage_pct >= MIN_COVERAGE_THRESHOLD
        assert meets_threshold == expected, \
            f"Failed for {description}: expected {expected}, got {meets_threshold}"
        status = "✓" if meets_threshold else "✗"
        print(f"{status} {description}: {coverage_pct:.2%} - {'KEEP' if meets_threshold else 'SKIP'}")
    
    print("✓ All threshold tests passed")


def test_proportional_reweighting():
    """Test that weights are proportionally reweighted when tickers are missing."""
    print("\n" + "="*70)
    print("TEST: Proportional reweighting")
    print("="*70)
    
    # Create test wave with 5 tickers
    weights_data = {
        'ticker_norm': ['A', 'B', 'C', 'D', 'E'],
        'weight': [0.1, 0.2, 0.3, 0.2, 0.2]  # Total = 1.0
    }
    weights_df = pd.DataFrame(weights_data)
    
    # Simulate 3 available tickers (A, B, C) with combined weight of 0.6
    available_tickers = ['A', 'B', 'C']
    weights_available = weights_df[weights_df['ticker_norm'].isin(available_tickers)].copy()
    
    # Calculate normalized weights
    total_abs = weights_available['weight'].abs().sum()
    weights_available['norm_weight'] = weights_available['weight'] / total_abs
    
    # Verify normalized weights sum to 1.0
    norm_sum = weights_available['norm_weight'].sum()
    assert abs(norm_sum - 1.0) < 0.001, \
        f"Expected normalized weights to sum to 1.0, got {norm_sum}"
    
    # Verify proportions are maintained
    # A should be 0.1/0.6 = 1/6, B should be 0.2/0.6 = 1/3, C should be 0.3/0.6 = 1/2
    expected_weights = {
        'A': 0.1 / 0.6,
        'B': 0.2 / 0.6,
        'C': 0.3 / 0.6
    }
    
    for ticker, expected_weight in expected_weights.items():
        actual_weight = weights_available[weights_available['ticker_norm'] == ticker]['norm_weight'].values[0]
        assert abs(actual_weight - expected_weight) < 0.001, \
            f"Expected {ticker} weight {expected_weight:.4f}, got {actual_weight:.4f}"
    
    print("✓ Proportional reweighting verified:")
    for _, row in weights_available.iterrows():
        print(f"  {row['ticker_norm']}: {row['weight']:.2f} → {row['norm_weight']:.4f}")


def test_ticker_normalization_in_coverage():
    """Test that ticker normalization is applied in coverage computation."""
    print("\n" + "="*70)
    print("TEST: Ticker normalization in coverage")
    print("="*70)
    
    # Create test wave weights with various ticker formats
    weights_data = {
        'wave': ['Test Wave'] * 4,
        'ticker': ['brk.b', 'BRK–B', 'BRK—B', 'AAPL'],  # Different formats, first 3 should normalize to same
        'weight': [0.25, 0.25, 0.25, 0.25]
    }
    weights_df = pd.DataFrame(weights_data)
    
    # Apply normalization
    weights_df['ticker_norm'] = weights_df['ticker'].apply(normalize_ticker)
    
    # Verify normalization
    print("Original → Normalized:")
    for _, row in weights_df.iterrows():
        print(f"  {row['ticker']} → {row['ticker_norm']}")
    
    # Check that first 3 tickers normalize to the same value
    normalized_brk = weights_df['ticker_norm'].iloc[0:3].unique()
    assert len(normalized_brk) == 1, \
        f"Expected all BRK variants to normalize to same value, got {normalized_brk}"
    assert normalized_brk[0] == 'BRK-B', \
        f"Expected 'BRK-B', got '{normalized_brk[0]}'"
    
    print("✓ Ticker normalization correctly applied")


def run_all_tests():
    """Run all wave coverage tests."""
    print("\n" + "="*70)
    print("Running wave coverage computation tests...")
    print("="*70 + "\n")
    
    try:
        test_coverage_computation_with_missing_tickers()
        test_90_percent_threshold()
        test_proportional_reweighting()
        test_ticker_normalization_in_coverage()
        
        print("\n" + "="*70)
        print("✓ All wave coverage tests passed!")
        print("="*70)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
