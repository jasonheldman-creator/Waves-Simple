#!/usr/bin/env python3
"""
Demonstration of the new crypto wave_id handling in generate_live_snapshot_csv.

This script demonstrates:
1. Equity waves require strict unique wave_ids (will raise AssertionError if not)
2. Crypto waves only log warnings for missing or duplicate wave_ids
3. The function still produces exactly 28 rows
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_truth import _is_crypto_wave, _convert_wave_name_to_id


def demonstrate_crypto_equity_separation():
    """Demonstrate how waves are separated into crypto and equity"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Crypto vs Equity Wave Separation")
    print("=" * 80)
    
    # Load actual waves
    from analytics_truth import load_weights, expected_waves
    
    weights_df = load_weights('wave_weights.csv')
    waves = expected_waves(weights_df)
    
    crypto_waves = [w for w in waves if _is_crypto_wave(w)]
    equity_waves = [w for w in waves if not _is_crypto_wave(w)]
    
    print(f"\nTotal waves: {len(waves)}")
    print(f"Crypto waves: {len(crypto_waves)}")
    print(f"Equity waves: {len(equity_waves)}")
    
    print("\nCrypto waves:")
    for w in crypto_waves:
        print(f"  - {w}")
    
    print(f"\nEquity waves (showing first 10):")
    for w in equity_waves[:10]:
        print(f"  - {w}")
    
    print("\n" + "=" * 80)


def demonstrate_validation_rules():
    """Demonstrate the different validation rules for crypto vs equity"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Validation Rules")
    print("=" * 80)
    
    print("\nEQUITY WAVES (Strict Validation):")
    print("  ✓ All equity waves MUST have unique wave_ids")
    print("  ✓ AssertionError is raised if any duplicates found")
    print("  ✓ AssertionError is raised if any wave_ids are missing")
    
    print("\nCRYPTO WAVES (Lenient Validation):")
    print("  ✓ Crypto waves only generate warnings for issues")
    print("  ✓ Missing wave_ids are logged as warnings")
    print("  ✓ Duplicate wave_ids are logged as warnings")
    print("  ✓ No AssertionError is raised")
    
    print("\nCOMMON VALIDATION:")
    print("  ✓ Total row count must be exactly 28")
    
    print("\n" + "=" * 80)


def demonstrate_wave_id_conversion():
    """Demonstrate wave_id conversion examples"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Wave ID Conversion Examples")
    print("=" * 80)
    
    examples = [
        "S&P 500 Wave",
        "AI & Cloud MegaCap Wave",
        "Crypto AI Growth Wave",
        "Crypto DeFi Growth Wave",
        "Russell 3000 Wave",
        "Income Wave",
    ]
    
    for wave_name in examples:
        wave_id = _convert_wave_name_to_id(wave_name)
        wave_type = "Crypto" if _is_crypto_wave(wave_name) else "Equity"
        print(f"  {wave_type:7} | {wave_name:35} -> {wave_id}")
    
    print("\n" + "=" * 80)


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print("CRYPTO WAVE_ID HANDLING DEMONSTRATION")
    print("=" * 80)
    
    demonstrate_crypto_equity_separation()
    demonstrate_validation_rules()
    demonstrate_wave_id_conversion()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe generate_live_snapshot_csv function now:")
    print("  1. Identifies crypto waves by checking if name starts with 'Crypto'")
    print("  2. Applies strict validation to equity waves (assertions)")
    print("  3. Applies lenient validation to crypto waves (warnings only)")
    print("  4. Ensures exactly 28 rows in the output")
    print("\nThis allows crypto wave_id discrepancies to be handled gracefully")
    print("while maintaining strict requirements for equity waves.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
