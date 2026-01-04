#!/usr/bin/env python3
"""
Price Coverage Analysis Tool

Analyzes the current price data cache (prices.csv) and generates comprehensive
diagnostics about Wave coverage, missing tickers, and data completeness.

This tool:
1. Loads existing prices.csv
2. Compares against all Wave requirements (wave_weights.csv, wave_config.csv)
3. Calculates coverage percentage per Wave
4. Identifies missing tickers with categorization
5. Generates actionable diagnostic report

Usage:
    python analyze_price_coverage.py [--output OUTPUT_DIR]
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import pandas as pd


# Configuration
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def log_message(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol."""
    return ticker.strip().upper()


def load_existing_price_data() -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load existing prices.csv and extract metadata.
    
    Returns:
        (dataframe, set of tickers)
    """
    prices_path = os.path.join(REPO_ROOT, "prices.csv")
    
    if not os.path.exists(prices_path):
        log_message(f"prices.csv not found at {prices_path}", "ERROR")
        return pd.DataFrame(), set()
    
    try:
        df = pd.read_csv(prices_path)
        tickers = set(df['ticker'].unique())
        
        log_message(f"Loaded prices.csv: {len(df)} rows, {len(tickers)} tickers")
        log_message(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df, tickers
    except Exception as e:
        log_message(f"Error loading prices.csv: {e}", "ERROR")
        return pd.DataFrame(), set()


def load_wave_requirements() -> Dict[str, Dict]:
    """
    Load Wave requirements from wave_weights.csv and wave_config.csv.
    
    Returns:
        Dict mapping wave name to {holdings: [...], benchmark: str}
    """
    waves = {}
    
    # Load holdings from wave_weights.csv
    weights_path = os.path.join(REPO_ROOT, "wave_weights.csv")
    if os.path.exists(weights_path):
        try:
            df = pd.read_csv(weights_path)
            for wave_name in df['wave'].unique():
                wave_data = df[df['wave'] == wave_name]
                tickers = [normalize_ticker(t) for t in wave_data['ticker'].tolist()]
                waves[wave_name] = {'holdings': tickers, 'benchmark': None}
        except Exception as e:
            log_message(f"Error loading wave_weights.csv: {e}", "ERROR")
    
    # Load benchmarks from wave_config.csv
    config_path = os.path.join(REPO_ROOT, "wave_config.csv")
    if os.path.exists(config_path):
        try:
            df = pd.read_csv(config_path)
            for _, row in df.iterrows():
                wave_name = row['Wave']
                benchmark = normalize_ticker(row['Benchmark']) if pd.notna(row['Benchmark']) else None
                
                if wave_name in waves:
                    waves[wave_name]['benchmark'] = benchmark
                else:
                    waves[wave_name] = {'holdings': [], 'benchmark': benchmark}
        except Exception as e:
            log_message(f"Error loading wave_config.csv: {e}", "ERROR")
    
    log_message(f"Loaded {len(waves)} Wave definitions")
    return waves


def calculate_wave_coverage(
    wave_name: str,
    wave_data: Dict,
    available_tickers: Set[str],
    price_df: pd.DataFrame
) -> Dict:
    """
    Calculate coverage statistics for a single Wave.
    
    Returns:
        Dict with coverage metrics
    """
    all_required = set(wave_data['holdings'])
    if wave_data['benchmark']:
        all_required.add(wave_data['benchmark'])
    
    available = all_required.intersection(available_tickers)
    missing = all_required - available_tickers
    
    coverage_pct = (len(available) / len(all_required) * 100) if all_required else 0
    
    # Calculate history length for available tickers
    history_days = 0
    if available and not price_df.empty:
        # Get min and max dates for any ticker in this wave
        wave_prices = price_df[price_df['ticker'].isin(available)]
        if not wave_prices.empty:
            min_date = pd.to_datetime(wave_prices['date'].min())
            max_date = pd.to_datetime(wave_prices['date'].max())
            history_days = (max_date - min_date).days
    
    return {
        'total_tickers_required': len(all_required),
        'tickers_available': len(available),
        'tickers_missing': len(missing),
        'coverage_percentage': round(coverage_pct, 1),
        'history_days': history_days,
        'missing_ticker_list': sorted(list(missing)),
        'available_ticker_list': sorted(list(available)),
        'benchmark': wave_data['benchmark'],
        'benchmark_available': wave_data['benchmark'] in available_tickers if wave_data['benchmark'] else None
    }


def categorize_missing_tickers(missing: Set[str]) -> Dict[str, List[str]]:
    """
    Categorize missing tickers by type.
    
    Returns:
        Dict mapping category to list of tickers
    """
    categories = {
        'crypto': [],
        'equity': [],
        'etf': [],
        'other': []
    }
    
    for ticker in missing:
        if '-USD' in ticker or ticker.startswith('BTC') or ticker.startswith('ETH'):
            categories['crypto'].append(ticker)
        elif ticker.isupper() and len(ticker) <= 5 and '-' not in ticker:
            # Likely an equity
            if ticker in ['SPY', 'QQQ', 'IWM', 'IWV', 'DIA', 'AGG', 'TLT', 'GLD', 
                          'ICLN', 'XLE', 'XLI', 'BIL', 'SHY', 'IEF', 'MUB', 'SUB',
                          'SGOV', 'SMH', 'TAN', 'PAVE', 'ARKK', 'IJH', 'IWO', 'IWP',
                          'MDY', 'DVY', 'VBK', 'VGT', 'VTV', 'VTWO', 'VYM', 'HDV',
                          'SCHD', 'NOBL', 'SHM', 'HYD', 'HYG', 'LQD', 'BND']:
                categories['etf'].append(ticker)
            else:
                categories['equity'].append(ticker)
        else:
            categories['other'].append(ticker)
    
    # Sort each category
    for cat in categories:
        categories[cat] = sorted(categories[cat])
    
    return categories


def generate_coverage_report(
    waves: Dict[str, Dict],
    available_tickers: Set[str],
    price_df: pd.DataFrame
) -> Dict:
    """Generate comprehensive coverage report."""
    
    wave_reports = {}
    all_missing = set()
    total_waves = len(waves)
    waves_with_full_coverage = 0
    waves_with_partial_coverage = 0
    waves_with_no_coverage = 0
    
    for wave_name, wave_data in waves.items():
        coverage = calculate_wave_coverage(wave_name, wave_data, available_tickers, price_df)
        wave_reports[wave_name] = coverage
        all_missing.update(coverage['missing_ticker_list'])
        
        if coverage['coverage_percentage'] == 100:
            waves_with_full_coverage += 1
        elif coverage['coverage_percentage'] > 0:
            waves_with_partial_coverage += 1
        else:
            waves_with_no_coverage += 1
    
    # Categorize all missing tickers
    missing_categorized = categorize_missing_tickers(all_missing)
    
    # Overall statistics
    total_unique_required = set()
    for wave_data in waves.values():
        total_unique_required.update(wave_data['holdings'])
        if wave_data['benchmark']:
            total_unique_required.add(wave_data['benchmark'])
    
    overall_coverage_pct = (len(available_tickers.intersection(total_unique_required)) / 
                           len(total_unique_required) * 100) if total_unique_required else 0
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_waves': total_waves,
            'waves_with_full_coverage': waves_with_full_coverage,
            'waves_with_partial_coverage': waves_with_partial_coverage,
            'waves_with_no_coverage': waves_with_no_coverage,
            'total_unique_tickers_required': len(total_unique_required),
            'tickers_available': len(available_tickers.intersection(total_unique_required)),
            'tickers_missing': len(all_missing),
            'overall_coverage_percentage': round(overall_coverage_pct, 1)
        },
        'missing_tickers_by_category': missing_categorized,
        'wave_coverage': wave_reports
    }
    
    return report


def print_report_summary(report: Dict):
    """Print human-readable summary of the report."""
    print("\n" + "=" * 80)
    print("PRICE DATA COVERAGE ANALYSIS")
    print("=" * 80)
    
    summary = report['summary']
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total Waves: {summary['total_waves']}")
    print(f"  Waves with 100% coverage: {summary['waves_with_full_coverage']}")
    print(f"  Waves with partial coverage: {summary['waves_with_partial_coverage']}")
    print(f"  Waves with 0% coverage: {summary['waves_with_no_coverage']}")
    print(f"\n  Total unique tickers required: {summary['total_unique_tickers_required']}")
    print(f"  Tickers available: {summary['tickers_available']}")
    print(f"  Tickers missing: {summary['tickers_missing']}")
    print(f"  Overall coverage: {summary['overall_coverage_percentage']}%")
    
    print(f"\n📋 MISSING TICKERS BY CATEGORY")
    missing_cat = report['missing_tickers_by_category']
    for category, tickers in missing_cat.items():
        if tickers:
            print(f"  {category.upper()} ({len(tickers)}): {', '.join(tickers[:10])}")
            if len(tickers) > 10:
                print(f"    ... and {len(tickers) - 10} more")
    
    print(f"\n🔍 WAVE-BY-WAVE COVERAGE")
    print(f"{'Wave Name':<40} {'Coverage':<10} {'Available':<12} {'Missing':<10} {'History'}")
    print("-" * 80)
    
    for wave_name, coverage in sorted(report['wave_coverage'].items(), 
                                     key=lambda x: x[1]['coverage_percentage']):
        coverage_str = f"{coverage['coverage_percentage']:.1f}%"
        available_str = f"{coverage['tickers_available']}/{coverage['total_tickers_required']}"
        missing_str = str(coverage['tickers_missing'])
        history_str = f"{coverage['history_days']} days" if coverage['history_days'] > 0 else "N/A"
        
        print(f"{wave_name:<40} {coverage_str:<10} {available_str:<12} {missing_str:<10} {history_str}")
    
    print("\n" + "=" * 80)


def save_report_json(report: Dict, output_dir: str):
    """Save detailed report to JSON file."""
    output_file = os.path.join(output_dir, "price_coverage_analysis.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        log_message(f"Saved detailed report to {output_file}")
    except Exception as e:
        log_message(f"Error saving report: {e}", "ERROR")


def save_missing_tickers_csv(report: Dict, output_dir: str):
    """Save missing tickers list to CSV for easy reference."""
    output_file = os.path.join(output_dir, "missing_tickers.csv")
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker', 'category', 'affected_waves'])
            
            # Build reverse map: ticker -> list of waves that need it
            ticker_to_waves = defaultdict(list)
            for wave_name, coverage in report['wave_coverage'].items():
                for ticker in coverage['missing_ticker_list']:
                    ticker_to_waves[ticker].append(wave_name)
            
            # Write rows
            for category, tickers in report['missing_tickers_by_category'].items():
                for ticker in tickers:
                    waves = '; '.join(ticker_to_waves[ticker])
                    writer.writerow([ticker, category, waves])
        
        log_message(f"Saved missing tickers list to {output_file}")
    except Exception as e:
        log_message(f"Error saving missing tickers CSV: {e}", "ERROR")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze price data coverage for all Waves'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=REPO_ROOT,
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    log_message("=" * 80)
    log_message("PRICE DATA COVERAGE ANALYSIS")
    log_message("=" * 80)
    
    # Step 1: Load existing price data
    log_message("\nStep 1: Loading existing price data...")
    price_df, available_tickers = load_existing_price_data()
    
    if price_df.empty:
        log_message("No price data available - cannot generate report", "ERROR")
        return 1
    
    # Step 2: Load Wave requirements
    log_message("\nStep 2: Loading Wave requirements...")
    waves = load_wave_requirements()
    
    if not waves:
        log_message("No Wave definitions found - cannot generate report", "ERROR")
        return 1
    
    # Step 3: Calculate coverage
    log_message("\nStep 3: Calculating coverage for each Wave...")
    report = generate_coverage_report(waves, available_tickers, price_df)
    
    # Step 4: Display summary
    print_report_summary(report)
    
    # Step 5: Save detailed reports
    log_message("\nStep 4: Saving detailed reports...")
    save_report_json(report, args.output)
    save_missing_tickers_csv(report, args.output)
    
    log_message("\n" + "=" * 80)
    log_message(f"Analysis complete. Reports saved to {args.output}")
    log_message("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
