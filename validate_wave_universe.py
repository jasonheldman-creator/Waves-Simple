"""
Wave Universe Validation

This module validates that all Wave definitions resolve to tickers in universal_universe.csv.
It is called during app startup to ensure data governance.
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_waves_against_universe(verbose: bool = False) -> Dict[str, any]:
    """
    Validate that all Wave definitions resolve to tickers in the universal universe.
    
    This function:
    1. Loads all Wave definitions from waves_engine.WAVE_WEIGHTS
    2. Checks each ticker against universal_universe.csv
    3. Logs missing tickers (with graceful degradation)
    4. Returns validation report
    
    Args:
        verbose: If True, print detailed validation output
    
    Returns:
        Dict with validation results:
        {
            'total_waves': int,
            'total_tickers_referenced': int,
            'valid_tickers': int,
            'invalid_tickers': int,
            'waves_with_issues': List[str],
            'missing_ticker_details': Dict[str, List[str]],
            'degraded_waves': List[str],
            'fully_valid_waves': List[str]
        }
    """
    try:
        from waves_engine import WAVE_WEIGHTS
    except ImportError as e:
        return {
            'error': f"Could not import waves_engine: {e}",
            'total_waves': 0,
            'validation_passed': False
        }
    
    try:
        import pandas as pd
        universe_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'universal_universe.csv'
        )
        universe_df = pd.read_csv(universe_path)
        universe_tickers = set(universe_df['ticker'].str.upper().values)
    except Exception as e:
        return {
            'error': f"Could not load universal universe: {e}",
            'total_waves': len(WAVE_WEIGHTS) if 'WAVE_WEIGHTS' in locals() else 0,
            'validation_passed': False
        }
    
    # Validation results
    all_referenced_tickers = set()
    valid_tickers = set()
    invalid_tickers = set()
    
    # Per-wave tracking
    waves_with_issues = []
    fully_valid_waves = []
    degraded_waves = []
    missing_ticker_details = defaultdict(list)
    
    if verbose:
        print("=" * 70)
        print("WAVE UNIVERSE VALIDATION")
        print("=" * 70)
        print(f"Validating {len(WAVE_WEIGHTS)} Wave definitions against universal universe...")
        print()
    
    for wave_name, holdings in WAVE_WEIGHTS.items():
        wave_tickers = [holding.ticker for holding in holdings]
        wave_valid = []
        wave_invalid = []
        
        for ticker in wave_tickers:
            ticker_upper = ticker.strip().upper()
            all_referenced_tickers.add(ticker_upper)
            
            if ticker_upper in universe_tickers:
                valid_tickers.add(ticker_upper)
                wave_valid.append(ticker)
            else:
                invalid_tickers.add(ticker_upper)
                wave_invalid.append(ticker)
                missing_ticker_details[wave_name].append(ticker)
        
        # Categorize wave
        if wave_invalid:
            waves_with_issues.append(wave_name)
            
            # Determine if degraded (some valid) or fully invalid
            if wave_valid:
                degraded_waves.append(wave_name)
                status = "DEGRADED"
                status_icon = "🟡"
            else:
                status = "INVALID"
                status_icon = "🔴"
            
            if verbose:
                print(f"{status_icon} {wave_name} - {status}")
                print(f"   Valid: {len(wave_valid)}/{len(wave_tickers)}")
                print(f"   Missing: {wave_invalid[:3]}")
                if len(wave_invalid) > 3:
                    print(f"   ... and {len(wave_invalid) - 3} more")
                print()
        else:
            fully_valid_waves.append(wave_name)
            if verbose:
                print(f"✓ {wave_name} - All {len(wave_tickers)} tickers valid")
    
    # Generate report
    report = {
        'total_waves': len(WAVE_WEIGHTS),
        'total_tickers_referenced': len(all_referenced_tickers),
        'valid_tickers': len(valid_tickers),
        'invalid_tickers': len(invalid_tickers),
        'waves_with_issues': waves_with_issues,
        'degraded_waves': degraded_waves,
        'fully_valid_waves': fully_valid_waves,
        'missing_ticker_details': dict(missing_ticker_details),
        'validation_passed': len(invalid_tickers) == 0,
        'graceful_degradation_enabled': True  # Always true - we don't block rendering
    }
    
    if verbose:
        print()
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Waves: {report['total_waves']}")
        print(f"Fully Valid Waves: {len(fully_valid_waves)}")
        print(f"Degraded Waves: {len(degraded_waves)} (some tickers missing)")
        print(f"Waves with Issues: {len(waves_with_issues)}")
        print()
        print(f"Total Tickers Referenced: {report['total_tickers_referenced']}")
        print(f"Valid Tickers: {report['valid_tickers']}")
        print(f"Invalid Tickers: {report['invalid_tickers']}")
        print()
        
        if invalid_tickers:
            print("⚠ GRACEFUL DEGRADATION ACTIVE:")
            print("  - All Waves will render (no blocking)")
            print("  - Degraded Waves will use available data only")
            print("  - Analytics will be limited for affected Waves")
            print()
            print(f"Missing tickers to add to universe: {sorted(list(invalid_tickers))[:10]}")
            if len(invalid_tickers) > 10:
                print(f"  ... and {len(invalid_tickers) - 10} more")
        else:
            print("✓ ALL TICKERS VALIDATED - Full analytics available for all Waves")
        print("=" * 70)
    
    return report


def check_wave_universe_alignment() -> Tuple[bool, str]:
    """
    Startup validation check for Wave-Universe alignment.
    
    Returns:
        (success: bool, message: str)
    """
    report = validate_waves_against_universe(verbose=False)
    
    if 'error' in report:
        return False, report['error']
    
    # Always return True (graceful degradation)
    # But provide informative message
    total_waves = report['total_waves']
    valid_waves = len(report['fully_valid_waves'])
    degraded_waves = len(report['degraded_waves'])
    
    if degraded_waves > 0:
        return True, f"{valid_waves}/{total_waves} Waves fully valid, {degraded_waves} degraded (graceful)"
    else:
        return True, f"All {total_waves} Waves validated against universe"


if __name__ == "__main__":
    """Run validation as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Waves against universal universe')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    report = validate_waves_against_universe(verbose=args.verbose)
    
    if args.json:
        import json
        print(json.dumps(report, indent=2))
    
    # Exit code: 0 if validation passed (or gracefully degraded), 1 if error
    if 'error' in report:
        sys.exit(1)
    else:
        sys.exit(0)
