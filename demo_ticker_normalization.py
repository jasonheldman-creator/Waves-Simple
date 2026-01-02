#!/usr/bin/env python3
"""
Demo script showing ticker normalization and wave coverage implementation.

This script demonstrates:
1. Ticker normalization with various input formats
2. Coverage computation with missing tickers
3. Wave validation with 90% threshold
"""

import importlib.util
import os
import sys

# Import normalize_ticker directly from file
spec = importlib.util.spec_from_file_location(
    "ticker_normalize",
    os.path.join(os.path.dirname(__file__), "helpers", "ticker_normalize.py")
)
ticker_normalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ticker_normalize)
normalize_ticker = ticker_normalize.normalize_ticker


def demo_ticker_normalization():
    """Demonstrate ticker normalization with various formats."""
    print("="*70)
    print("DEMO 1: Ticker Normalization")
    print("="*70)
    print("\nNormalizing various ticker formats:")
    
    test_cases = [
        ("brk.b", "BRK-B (dot to hyphen)"),
        ("BRK–B", "BRK-B (en-dash to hyphen)"),
        ("BRK—B", "BRK-B (em-dash to hyphen)"),
        ("BRK‐B", "BRK-B (Unicode hyphen to hyphen)"),
        ("BRK−B", "BRK-B (minus sign to hyphen)"),
        ("  aapl  ", "AAPL (whitespace trimmed)"),
        ("msft", "MSFT (lowercase to uppercase)"),
        (None, "'' (None to empty string)"),
    ]
    
    for original, description in test_cases:
        normalized = normalize_ticker(original)
        print(f"  '{original}' → '{normalized}' ({description})")


def demo_coverage_computation():
    """Demonstrate coverage computation with missing tickers."""
    print("\n" + "="*70)
    print("DEMO 2: Coverage Computation with Missing Tickers")
    print("="*70)
    
    # Simulate a wave with 5 tickers
    wave_data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'MISSING1', 'MISSING2'],
        'weight': [0.2, 0.2, 0.2, 0.2, 0.2]
    }
    
    available_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Calculate coverage
    total_weight = sum(wave_data['weight'])
    available_weight = sum(
        w for t, w in zip(wave_data['ticker'], wave_data['weight'])
        if t in available_tickers
    )
    coverage_pct = available_weight / total_weight if total_weight > 0 else 0.0
    
    print(f"\nWave Configuration:")
    print(f"  Total tickers: {len(wave_data['ticker'])}")
    for ticker, weight in zip(wave_data['ticker'], wave_data['weight']):
        status = "✓" if ticker in available_tickers else "✗"
        print(f"    {status} {ticker}: {weight:.2f} weight")
    
    print(f"\nCoverage Metrics:")
    print(f"  Total weight: {total_weight:.2f}")
    print(f"  Available weight: {available_weight:.2f}")
    print(f"  Coverage: {coverage_pct:.2%}")


def demo_threshold_validation():
    """Demonstrate wave validation with 90% threshold."""
    print("\n" + "="*70)
    print("DEMO 3: Wave Validation with 90% Threshold")
    print("="*70)
    
    MIN_COVERAGE_THRESHOLD = 0.90
    
    test_scenarios = [
        (1.00, "Perfect coverage - all tickers available"),
        (0.95, "Excellent coverage - one small ticker missing"),
        (0.90, "Exactly at threshold - wave kept"),
        (0.85, "Below threshold - wave excluded"),
        (0.50, "Poor coverage - wave excluded"),
    ]
    
    print(f"\nMinimum coverage threshold: {MIN_COVERAGE_THRESHOLD:.0%}\n")
    
    for coverage, scenario in test_scenarios:
        meets_threshold = coverage >= MIN_COVERAGE_THRESHOLD
        status = "✓ KEEP" if meets_threshold else "✗ SKIP"
        print(f"  {status} | {coverage:.2%} coverage | {scenario}")


def demo_proportional_reweighting():
    """Demonstrate proportional reweighting when tickers are missing."""
    print("\n" + "="*70)
    print("DEMO 4: Proportional Reweighting")
    print("="*70)
    
    print("\nOriginal wave weights:")
    original_weights = {
        'AAPL': 0.10,
        'MSFT': 0.20,
        'GOOGL': 0.30,
        'MISSING1': 0.20,
        'MISSING2': 0.20
    }
    
    for ticker, weight in original_weights.items():
        status = "✓" if ticker not in ['MISSING1', 'MISSING2'] else "✗"
        print(f"  {status} {ticker}: {weight:.2f}")
    
    print(f"\nTotal original weight: {sum(original_weights.values()):.2f}")
    
    # Reweight available tickers
    available = {k: v for k, v in original_weights.items() if k not in ['MISSING1', 'MISSING2']}
    total_available = sum(available.values())
    
    print(f"\nAfter removing missing tickers:")
    print(f"  Available weight: {total_available:.2f}")
    
    print(f"\nProportionally reweighted (to sum to 1.0):")
    for ticker, weight in available.items():
        new_weight = weight / total_available
        print(f"  ✓ {ticker}: {weight:.2f} → {new_weight:.4f} ({new_weight/weight:.2f}x)")
    
    reweighted_sum = sum(w / total_available for w in available.values())
    print(f"\nReweighted sum: {reweighted_sum:.4f}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*10 + "TICKER NORMALIZATION & WAVE COVERAGE DEMO" + " "*17 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    demo_ticker_normalization()
    demo_coverage_computation()
    demo_threshold_validation()
    demo_proportional_reweighting()
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
