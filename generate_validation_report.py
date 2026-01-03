#!/usr/bin/env python3
"""
Generate Wave Validation Report

This script generates a comprehensive validation report for all active waves,
showing which waves have valid, continuous price history from PRICE_BOOK.

The report includes:
- Required tickers for each wave
- Found tickers in PRICE_BOOK
- Missing tickers (if any)
- Date coverage (start/end dates, number of days)
- Return computable status (Yes/No with reason)

This is the deterministic validation required before rendering performance tables.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)

def print_validation_result(validation, index, total):
    """Print a single wave validation result in a readable format."""
    wave_name = validation['wave_name']
    valid = validation['valid']
    
    # Status indicator
    status_icon = "✓" if valid else "✗"
    status_color = "\033[92m" if valid else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    
    print(f"\n{status_color}[{index}/{total}] {status_icon} {wave_name}{reset_color}")
    print(f"  Return Computable: {validation['return_computable']}")
    print(f"  Validation: {validation['validation_reason']}")
    
    # Required tickers
    required_count = len(validation['required_tickers'])
    found_count = len(validation['found_tickers'])
    missing_count = len(validation['missing_tickers'])
    
    print(f"  Required Tickers: {required_count} ({found_count} found, {missing_count} missing)")
    
    if validation['required_tickers']:
        print(f"    - Required: {', '.join(validation['required_tickers'][:10])}")
        if len(validation['required_tickers']) > 10:
            print(f"      ... and {len(validation['required_tickers']) - 10} more")
    
    if validation['missing_tickers']:
        print(f"    - Missing: {', '.join(validation['missing_tickers'])}")
    
    # Date coverage
    if validation['date_coverage_start'] and validation['date_coverage_end']:
        print(f"  Date Coverage: {validation['date_coverage_start']} to {validation['date_coverage_end']}")
        print(f"  Trading Days: {validation['date_coverage_days']}")
    else:
        print(f"  Date Coverage: N/A")
    
    print(f"  Coverage: {validation['coverage_pct']:.1f}%")

def generate_and_display_report(min_coverage_pct=100.0, min_history_days=30, save_to_file=True):
    """Generate and display the validation report."""
    print_separator()
    print("WAVE VALIDATION REPORT GENERATOR")
    print_separator()
    print(f"\nValidation Criteria:")
    print(f"  - Minimum Ticker Coverage: {min_coverage_pct}%")
    print(f"  - Minimum History Days: {min_history_days}")
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import required modules
    print("\nLoading modules...")
    # Import directly from modules to avoid streamlit dependency in helpers/__init__.py
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helpers'))
    from price_book import get_price_book
    from wave_performance import generate_wave_validation_report
    
    # Load PRICE_BOOK
    print("Loading PRICE_BOOK...")
    price_book = get_price_book()
    
    if price_book.empty:
        print("\n❌ ERROR: PRICE_BOOK is empty!")
        print("Please run build_price_cache.py to populate the cache.")
        return
    
    print(f"  ✓ Loaded PRICE_BOOK: {price_book.shape[0]} days × {price_book.shape[1]} tickers")
    print(f"  Date range: {price_book.index[0].date()} to {price_book.index[-1].date()}")
    
    # Generate validation report
    print("\nValidating all waves...")
    output_file = 'wave_validation_report.json' if save_to_file else None
    
    report = generate_wave_validation_report(
        price_book,
        min_coverage_pct=min_coverage_pct,
        min_history_days=min_history_days,
        output_file=output_file
    )
    
    # Display summary
    print_separator()
    print("VALIDATION SUMMARY")
    print_separator()
    print(f"\nTotal Waves: {report['waves_validated']}")
    print(f"  ✓ Valid (Passed):   {report['waves_valid']}")
    print(f"  ✗ Invalid (Failed): {report['waves_invalid']}")
    print(f"\nSuccess Rate: {report['waves_valid']/report['waves_validated']*100:.1f}%")
    
    # Display PRICE_BOOK metadata
    pb_meta = report.get('price_book_meta', {})
    if pb_meta and not pb_meta.get('is_empty', True):
        print(f"\nPRICE_BOOK Metadata:")
        print(f"  Path: {pb_meta['cache_path']}")
        print(f"  Date Range: {pb_meta['date_min']} to {pb_meta['date_max']}")
        print(f"  Trading Days: {pb_meta['rows']}")
        print(f"  Total Tickers: {pb_meta['tickers_count']}")
    
    # Display detailed results
    print_separator()
    print("DETAILED VALIDATION RESULTS")
    print_separator()
    
    validation_results = report.get('validation_results', [])
    
    # Separate valid and invalid results
    valid_waves = [v for v in validation_results if v['valid']]
    invalid_waves = [v for v in validation_results if not v['valid']]
    
    # Show invalid waves first (these are the problems)
    if invalid_waves:
        print(f"\n{len(invalid_waves)} INVALID WAVES (Failed Validation):")
        print_separator('-')
        for i, validation in enumerate(invalid_waves, 1):
            print_validation_result(validation, i, len(invalid_waves))
    
    # Show valid waves
    if valid_waves:
        print(f"\n\n{len(valid_waves)} VALID WAVES (Passed Validation):")
        print_separator('-')
        for i, validation in enumerate(valid_waves, 1):
            print_validation_result(validation, i, len(valid_waves))
    
    # Final summary
    print_separator()
    print("REPORT COMPLETE")
    print_separator()
    print(f"\nSummary: {report['summary']}")
    
    if output_file:
        print(f"\n✓ Full report saved to: {output_file}")
        print(f"  Use 'cat {output_file} | python -m json.tool' for formatted JSON")
    
    print(f"\nNext Steps:")
    if report['waves_invalid'] > 0:
        print(f"  1. Review the {report['waves_invalid']} invalid waves above")
        print(f"  2. Check missing tickers and date coverage")
        print(f"  3. Run build_price_cache.py to fetch missing data (if PRICE_FETCH_ENABLED=true)")
        print(f"  4. Re-run this script to verify all waves pass validation")
    else:
        print(f"  ✓ All waves passed validation!")
        print(f"  ✓ Performance tables can safely render all {report['waves_valid']} waves")
    
    print()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate wave validation report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report with default criteria (100% coverage, 30 days history)
  python generate_validation_report.py
  
  # Allow partial coverage (90%) and shorter history (10 days)
  python generate_validation_report.py --min-coverage 90 --min-days 10
  
  # Don't save to file, just display
  python generate_validation_report.py --no-save
        """
    )
    
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=100.0,
        help='Minimum ticker coverage percentage (default: 100.0)'
    )
    
    parser.add_argument(
        '--min-days',
        type=int,
        default=30,
        help='Minimum number of trading days required (default: 30)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save report to file (just display)'
    )
    
    args = parser.parse_args()
    
    try:
        generate_and_display_report(
            min_coverage_pct=args.min_coverage,
            min_history_days=args.min_days,
            save_to_file=not args.no_save
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Report generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
