"""
Enhanced Ticker Diagnostics Module

Provides comprehensive diagnostics for ticker validation, wave readiness,
and system health related to the ticker master file implementation.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any


def get_ticker_master_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive diagnostics for the ticker master file.
    
    Returns:
        Dict with diagnostics including:
        - ticker_count: Total number of tickers
        - validation_timestamp: When tickers were last validated
        - file_exists: Whether ticker_master_clean.csv exists
        - has_duplicates: Whether duplicates exist
        - status: Overall status (healthy/degraded/error)
        - issues: List of any issues found
    """
    diagnostics = {
        'ticker_count': 0,
        'validation_timestamp': None,
        'file_exists': False,
        'has_duplicates': False,
        'status': 'unknown',
        'issues': [],
        'tickers_by_source': {}
    }
    
    try:
        ticker_file = 'ticker_master_clean.csv'
        
        # Check file exists
        if not os.path.exists(ticker_file):
            diagnostics['status'] = 'error'
            diagnostics['issues'].append('ticker_master_clean.csv not found')
            return diagnostics
        
        diagnostics['file_exists'] = True
        
        # Read file
        df = pd.read_csv(ticker_file)
        diagnostics['ticker_count'] = len(df)
        
        # Get validation timestamp
        if 'created_date' in df.columns and not df['created_date'].empty:
            diagnostics['validation_timestamp'] = df['created_date'].iloc[0]
        
        # Check for duplicates
        duplicates = df[df['ticker'].duplicated()]
        if not duplicates.empty:
            diagnostics['has_duplicates'] = True
            diagnostics['issues'].append(f'Found {len(duplicates)} duplicate tickers')
        
        # Count by source
        if 'source' in df.columns:
            source_counts = df['source'].value_counts().to_dict()
            diagnostics['tickers_by_source'] = source_counts
        
        # Determine status
        if diagnostics['issues']:
            diagnostics['status'] = 'degraded'
        else:
            diagnostics['status'] = 'healthy'
        
        return diagnostics
        
    except Exception as e:
        diagnostics['status'] = 'error'
        diagnostics['issues'].append(f'Error reading ticker file: {str(e)}')
        return diagnostics


def get_wave_ticker_coverage() -> Dict[str, Dict[str, Any]]:
    """
    Get ticker coverage statistics for each wave.
    
    Returns:
        Dict mapping wave_id to coverage stats:
        - total_tickers: Number of tickers in wave
        - ticker_list: List of tickers
        - validated: Number validated in ticker master
        - coverage_pct: Percentage validated
    """
    try:
        from waves_engine import WAVE_WEIGHTS, get_all_wave_ids, get_display_name_from_wave_id, _normalize_ticker
        
        # Load ticker master
        ticker_file = 'ticker_master_clean.csv'
        if not os.path.exists(ticker_file):
            return {}
        
        master_df = pd.read_csv(ticker_file)
        validated_tickers = set(master_df['ticker'].tolist())
        
        # Analyze each wave
        coverage = {}
        
        for wave_id in get_all_wave_ids():
            display_name = get_display_name_from_wave_id(wave_id)
            if display_name not in WAVE_WEIGHTS:
                continue
            
            holdings = WAVE_WEIGHTS[display_name]
            wave_tickers = [h.ticker for h in holdings]
            # IMPORTANT: Normalize tickers before checking against master
            normalized_tickers = [_normalize_ticker(t) for t in wave_tickers]
            validated_count = sum(1 for t in normalized_tickers if t in validated_tickers)
            
            coverage[wave_id] = {
                'display_name': display_name,
                'total_tickers': len(wave_tickers),
                'ticker_list': wave_tickers,
                'validated': validated_count,
                'coverage_pct': (validated_count / len(wave_tickers) * 100) if wave_tickers else 0
            }
        
        return coverage
        
    except Exception as e:
        print(f"Error getting wave ticker coverage: {e}")
        return {}


def get_degraded_waves() -> List[Dict[str, Any]]:
    """
    Get list of degraded waves with causes.
    
    A wave is considered degraded if:
    - It has failed ticker validations
    - It has insufficient data coverage
    
    Returns:
        List of dicts with wave info and degradation causes
    """
    degraded_waves = []
    
    try:
        from helpers.ticker_diagnostics import get_diagnostics_tracker
        
        tracker = get_diagnostics_tracker()
        
        # Get failures by wave
        failures = tracker.get_all_failures()
        
        # Group by wave
        wave_failures = {}
        for failure in failures:
            wave_id = failure.wave_id or 'unknown'
            if wave_id not in wave_failures:
                wave_failures[wave_id] = []
            wave_failures[wave_id].append(failure)
        
        # Build degraded wave list
        for wave_id, wave_failures_list in wave_failures.items():
            if wave_failures_list:
                degraded_waves.append({
                    'wave_id': wave_id,
                    'wave_name': wave_failures_list[0].wave_name if wave_failures_list else wave_id,
                    'failed_ticker_count': len(wave_failures_list),
                    'failed_tickers': [f.ticker_original for f in wave_failures_list[:5]],  # First 5
                    'causes': list(set(f.failure_type.value for f in wave_failures_list))
                })
        
        return degraded_waves
        
    except Exception as e:
        print(f"Error getting degraded waves: {e}")
        return []


def generate_ticker_diagnostics_report() -> str:
    """
    Generate a comprehensive diagnostics report for tickers.
    
    Returns:
        Formatted string report
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TICKER MASTER FILE DIAGNOSTICS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Ticker Master File Status
    report_lines.append("1. TICKER MASTER FILE STATUS")
    report_lines.append("-" * 70)
    
    master_diag = get_ticker_master_diagnostics()
    report_lines.append(f"   Status: {master_diag['status'].upper()}")
    report_lines.append(f"   File Exists: {'✅' if master_diag['file_exists'] else '❌'}")
    report_lines.append(f"   Total Tickers: {master_diag['ticker_count']}")
    report_lines.append(f"   Validation Date: {master_diag['validation_timestamp']}")
    report_lines.append(f"   Duplicates: {'❌ YES' if master_diag['has_duplicates'] else '✅ NO'}")
    
    if master_diag['issues']:
        report_lines.append("   Issues:")
        for issue in master_diag['issues']:
            report_lines.append(f"      - {issue}")
    
    if master_diag['tickers_by_source']:
        report_lines.append("   Sources:")
        for source, count in master_diag['tickers_by_source'].items():
            report_lines.append(f"      - {source}: {count} tickers")
    
    report_lines.append("")
    
    # Wave Ticker Coverage
    report_lines.append("2. WAVE TICKER COVERAGE")
    report_lines.append("-" * 70)
    
    coverage = get_wave_ticker_coverage()
    if coverage:
        total_waves = len(coverage)
        full_coverage_waves = sum(1 for v in coverage.values() if v['coverage_pct'] == 100)
        
        report_lines.append(f"   Total Waves: {total_waves}")
        report_lines.append(f"   Full Coverage: {full_coverage_waves}/{total_waves} waves")
        report_lines.append("")
        report_lines.append("   Per-Wave Coverage:")
        
        # Sort by coverage percentage
        sorted_coverage = sorted(coverage.items(), key=lambda x: x[1]['coverage_pct'], reverse=True)
        
        for wave_id, stats in sorted_coverage:
            status_icon = "✅" if stats['coverage_pct'] == 100 else "⚠️" if stats['coverage_pct'] >= 80 else "❌"
            report_lines.append(
                f"      {status_icon} {stats['display_name']}: "
                f"{stats['validated']}/{stats['total_tickers']} "
                f"({stats['coverage_pct']:.0f}%)"
            )
    else:
        report_lines.append("   No coverage data available")
    
    report_lines.append("")
    
    # Degraded Waves
    report_lines.append("3. DEGRADED WAVES")
    report_lines.append("-" * 70)
    
    degraded = get_degraded_waves()
    if degraded:
        report_lines.append(f"   Found {len(degraded)} degraded waves:")
        for wave in degraded:
            report_lines.append(f"   - {wave['wave_name']}:")
            report_lines.append(f"      Failed Tickers: {wave['failed_ticker_count']}")
            report_lines.append(f"      Examples: {', '.join(wave['failed_tickers'])}")
            report_lines.append(f"      Causes: {', '.join(wave['causes'])}")
    else:
        report_lines.append("   ✅ No degraded waves detected")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    
    return "\n".join(report_lines)


def export_diagnostics_to_file(filename: str = None) -> str:
    """
    Export diagnostics report to a file.
    
    Args:
        filename: Optional filename. If None, generates timestamp-based name.
        
    Returns:
        Path to the exported file
    """
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ticker_diagnostics_report_{timestamp}.txt'
    
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    filepath = os.path.join('reports', filename)
    
    # Generate and write report
    report = generate_ticker_diagnostics_report()
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    return filepath


if __name__ == "__main__":
    # Generate and print report when run directly
    report = generate_ticker_diagnostics_report()
    print(report)
    
    # Also export to file
    filepath = export_diagnostics_to_file()
    print(f"\nReport exported to: {filepath}")
