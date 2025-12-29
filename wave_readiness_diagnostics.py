"""
Wave Readiness Diagnostics Module

Comprehensive forensic diagnostics for wave data readiness issues.
Implements the requirements from the wave readiness diagnosis problem statement.

This module provides:
1. Ground truth wave universe verification
2. Per-wave detailed diagnostic data with failure reasons
3. Root cause identification and reporting
4. Mismatch detection between configuration sources
5. Ticker failure impact analysis
6. Actionable remediation suggestions
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd

# Import from waves_engine
from waves_engine import (
    get_all_wave_ids,
    get_all_waves_universe,
    get_display_name_from_wave_id,
    WAVE_WEIGHTS,
    WAVE_ID_REGISTRY,
    BENCHMARK_WEIGHTS_STATIC,
    DISPLAY_NAME_TO_WAVE_ID,
)

# Import analytics pipeline
from analytics_pipeline import (
    compute_data_ready_status,
    get_wave_analytics_dir,
    resolve_wave_tickers,
    resolve_wave_benchmarks,
    MIN_DAYS_OPERATIONAL,
    MIN_DAYS_PARTIAL,
    MIN_DAYS_FULL,
    MIN_COVERAGE_OPERATIONAL,
    MIN_COVERAGE_PARTIAL,
    MIN_COVERAGE_FULL,
)


@dataclass
class WaveDiagnostic:
    """Comprehensive diagnostic data for a single wave."""
    wave_id: str
    display_name: str
    mode: str
    readiness_status: str  # 'full', 'partial', 'operational', 'unavailable'
    readiness_grade: str  # A, B, C, D, F
    is_ready: bool
    
    # Holdings and weights
    holdings_count: int
    holdings_tickers: List[str]
    weights_source: str  # 'WAVE_WEIGHTS'
    weights_loaded: bool
    
    # Benchmark
    benchmark_defined: bool
    benchmark_tickers: List[str]
    benchmark_loaded: bool
    
    # Price data
    prices_exist: bool
    prices_path: str
    prices_ticker_count: int
    prices_missing_tickers: List[str]
    coverage_pct: float
    
    # Benchmark prices
    benchmark_prices_exist: bool
    benchmark_prices_path: str
    benchmark_missing_tickers: List[str]
    
    # NAV
    nav_exist: bool
    nav_path: str
    nav_days: int
    
    # History and freshness
    history_start: Optional[str]
    history_end: Optional[str]
    history_days: int
    data_age_days: int
    is_fresh: bool
    
    # Readiness thresholds
    min_days_operational: int
    min_days_partial: int
    min_days_full: int
    min_coverage_operational: float
    min_coverage_partial: float
    min_coverage_full: float
    
    # Status
    blocking_issues: List[str]
    informational_issues: List[str]
    failure_stage: str  # e.g., 'weights_resolution', 'price_download', 'nav_computation'
    primary_failure_reason: str
    reason_codes: List[str]
    
    # Suggested actions
    suggested_actions: List[str]
    
    # Analytics capabilities
    allowed_analytics: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_readiness_grade(self) -> str:
        """Get letter grade for readiness."""
        if self.readiness_status == 'full' and self.coverage_pct >= 95:
            return 'A'
        elif self.readiness_status == 'full' or (self.readiness_status == 'partial' and self.coverage_pct >= 85):
            return 'B'
        elif self.readiness_status == 'partial' or (self.readiness_status == 'operational' and self.coverage_pct >= 70):
            return 'C'
        elif self.readiness_status == 'operational':
            return 'D'
        else:
            return 'F'


@dataclass
class WaveUniverseReport:
    """Report on wave universe integrity and consistency."""
    total_waves_engine: int
    total_waves_registry: int
    total_waves_weights: int
    waves_consistent: bool
    
    engine_waves: List[str]
    registry_waves: List[str]
    weights_waves: List[str]
    
    missing_in_registry: List[str]
    missing_in_weights: List[str]
    extra_in_weights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def get_wave_mode(wave_id: str) -> str:
    """
    Get the mode for a wave from wave_config.csv.
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        Mode string (e.g., 'Standard', 'Alpha-Minus-Beta', 'Private Logic')
    """
    try:
        display_name = get_display_name_from_wave_id(wave_id)
        config_path = 'wave_config.csv'
        
        if os.path.exists(config_path):
            config_df = pd.read_csv(config_path)
            wave_row = config_df[config_df['Wave'] == display_name]
            if not wave_row.empty:
                return wave_row.iloc[0].get('Mode', 'Standard')
    except Exception:
        pass
    
    return 'Standard'


def verify_wave_universe() -> WaveUniverseReport:
    """
    Verify consistency and completeness of wave universe across all sources.
    
    This is the ground truth verification - Step 1 of the diagnostic process.
    
    Returns:
        WaveUniverseReport with consistency analysis
    """
    # Get waves from all sources
    universe = get_all_waves_universe()
    wave_ids = set(get_all_wave_ids())
    registry_ids = set(WAVE_ID_REGISTRY.keys())
    weights_names = set(WAVE_WEIGHTS.keys())
    
    # Convert weights display names to wave_ids for comparison
    weights_ids = set()
    for display_name in weights_names:
        wave_id = DISPLAY_NAME_TO_WAVE_ID.get(display_name)
        if wave_id:
            weights_ids.add(wave_id)
    
    # Check consistency
    consistent = (wave_ids == registry_ids == weights_ids)
    
    # Find mismatches
    missing_in_registry = list(wave_ids - registry_ids)
    missing_in_weights = list(wave_ids - weights_ids)
    extra_in_weights = list(weights_ids - wave_ids)
    
    return WaveUniverseReport(
        total_waves_engine=len(wave_ids),
        total_waves_registry=len(registry_ids),
        total_waves_weights=len(weights_ids),
        waves_consistent=consistent,
        engine_waves=sorted(list(wave_ids)),
        registry_waves=sorted(list(registry_ids)),
        weights_waves=sorted(list(weights_ids)),
        missing_in_registry=missing_in_registry,
        missing_in_weights=missing_in_weights,
        extra_in_weights=extra_in_weights
    )


def diagnose_wave(wave_id: str) -> WaveDiagnostic:
    """
    Perform comprehensive diagnostic analysis for a single wave.
    
    This implements Step 2 of the diagnostic process - detailed per-wave tracing.
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        WaveDiagnostic with complete diagnostic data
    """
    # Get display name and mode
    display_name = get_display_name_from_wave_id(wave_id) or wave_id
    mode = get_wave_mode(wave_id)
    
    # Get readiness status from analytics pipeline
    status = compute_data_ready_status(wave_id)
    
    # Resolve holdings
    holdings_tickers = resolve_wave_tickers(wave_id)
    holdings_count = len(holdings_tickers) if holdings_tickers else 0
    weights_loaded = holdings_count > 0
    
    # Resolve benchmark
    benchmark_specs = resolve_wave_benchmarks(wave_id)
    benchmark_tickers = [ticker for ticker, _ in benchmark_specs]
    benchmark_defined = len(benchmark_tickers) > 0
    
    # Get file paths
    wave_dir = get_wave_analytics_dir(wave_id)
    prices_path = os.path.join(wave_dir, 'prices.csv')
    benchmark_path = os.path.join(wave_dir, 'benchmark_prices.csv')
    nav_path = os.path.join(wave_dir, 'nav.csv')
    
    # Check file existence
    prices_exist = os.path.exists(prices_path)
    benchmark_prices_exist = os.path.exists(benchmark_path)
    nav_exist = os.path.exists(nav_path)
    
    # Analyze price data if it exists
    prices_ticker_count = 0
    prices_missing_tickers = []
    coverage_pct = 0.0
    history_start = None
    history_end = None
    history_days = 0
    data_age_days = 999
    is_fresh = False
    
    if prices_exist:
        try:
            prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            if not prices_df.empty:
                prices_ticker_count = len(prices_df.columns)
                
                # Check for missing tickers
                available_tickers = set(prices_df.columns)
                expected_tickers = set(holdings_tickers)
                prices_missing_tickers = list(expected_tickers - available_tickers)
                
                # Calculate coverage
                if holdings_count > 0:
                    coverage_pct = ((holdings_count - len(prices_missing_tickers)) / holdings_count) * 100.0
                
                # Get history range
                history_start = prices_df.index[0].strftime('%Y-%m-%d')
                history_end = prices_df.index[-1].strftime('%Y-%m-%d')
                history_days = len(prices_df)
                
                # Calculate data age
                last_date = prices_df.index[-1]
                now = datetime.now()
                data_age_days = (now - last_date).days
                is_fresh = data_age_days <= 7
        except Exception:
            pass
    else:
        prices_missing_tickers = holdings_tickers
    
    # Analyze benchmark data
    benchmark_missing_tickers = []
    benchmark_loaded = False
    
    if benchmark_prices_exist:
        try:
            benchmark_df = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
            if not benchmark_df.empty:
                available_benchmark = set(benchmark_df.columns)
                expected_benchmark = set(benchmark_tickers)
                benchmark_missing_tickers = list(expected_benchmark - available_benchmark)
                benchmark_loaded = len(benchmark_missing_tickers) == 0
        except Exception:
            benchmark_missing_tickers = benchmark_tickers
    else:
        benchmark_missing_tickers = benchmark_tickers
    
    # Analyze NAV data
    nav_days = 0
    if nav_exist:
        try:
            nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
            nav_days = len(nav_df)
        except Exception:
            pass
    
    # Determine failure stage and primary reason
    failure_stage = 'unknown'
    primary_failure_reason = 'UNKNOWN'
    
    if not weights_loaded:
        failure_stage = 'weights_resolution'
        primary_failure_reason = 'MISSING_WEIGHTS'
    elif not prices_exist:
        failure_stage = 'price_download'
        primary_failure_reason = 'MISSING_PRICES'
    elif coverage_pct < (MIN_COVERAGE_OPERATIONAL * 100):
        failure_stage = 'price_download'
        primary_failure_reason = 'LOW_COVERAGE'
    elif history_days < MIN_DAYS_OPERATIONAL:
        failure_stage = 'price_download'
        primary_failure_reason = 'INSUFFICIENT_HISTORY'
    elif not is_fresh:
        failure_stage = 'data_freshness'
        primary_failure_reason = 'STALE_DATA'
    elif not nav_exist:
        failure_stage = 'nav_computation'
        primary_failure_reason = 'MISSING_NAV'
    else:
        failure_stage = 'ready'
        primary_failure_reason = 'READY'
    
    # Create diagnostic object
    diagnostic = WaveDiagnostic(
        wave_id=wave_id,
        display_name=display_name,
        mode=mode,
        readiness_status=status.get('readiness_status', 'unavailable'),
        readiness_grade='',  # Will be set below
        is_ready=status.get('is_ready', False),
        holdings_count=holdings_count,
        holdings_tickers=holdings_tickers,
        weights_source='WAVE_WEIGHTS',
        weights_loaded=weights_loaded,
        benchmark_defined=benchmark_defined,
        benchmark_tickers=benchmark_tickers,
        benchmark_loaded=benchmark_loaded,
        prices_exist=prices_exist,
        prices_path=prices_path,
        prices_ticker_count=prices_ticker_count,
        prices_missing_tickers=prices_missing_tickers,
        coverage_pct=round(coverage_pct, 2),
        benchmark_prices_exist=benchmark_prices_exist,
        benchmark_prices_path=benchmark_path,
        benchmark_missing_tickers=benchmark_missing_tickers,
        nav_exist=nav_exist,
        nav_path=nav_path,
        nav_days=nav_days,
        history_start=history_start,
        history_end=history_end,
        history_days=history_days,
        data_age_days=data_age_days,
        is_fresh=is_fresh,
        min_days_operational=MIN_DAYS_OPERATIONAL,
        min_days_partial=MIN_DAYS_PARTIAL,
        min_days_full=MIN_DAYS_FULL,
        min_coverage_operational=MIN_COVERAGE_OPERATIONAL,
        min_coverage_partial=MIN_COVERAGE_PARTIAL,
        min_coverage_full=MIN_COVERAGE_FULL,
        blocking_issues=status.get('blocking_issues', []),
        informational_issues=status.get('informational_issues', []),
        failure_stage=failure_stage,
        primary_failure_reason=primary_failure_reason,
        reason_codes=status.get('reason_codes', []),
        suggested_actions=status.get('suggested_actions', []),
        allowed_analytics=status.get('allowed_analytics', {}),
    )
    
    # Set readiness grade
    diagnostic.readiness_grade = diagnostic.get_readiness_grade()
    
    return diagnostic


def diagnose_all_waves() -> List[WaveDiagnostic]:
    """
    Diagnose all waves in the system.
    
    Returns:
        List of WaveDiagnostic objects for all waves
    """
    wave_ids = get_all_wave_ids()
    diagnostics = []
    
    for wave_id in sorted(wave_ids):
        diagnostic = diagnose_wave(wave_id)
        diagnostics.append(diagnostic)
    
    return diagnostics


def generate_readiness_report(output_format: str = 'text') -> str:
    """
    Generate comprehensive wave readiness report.
    
    Args:
        output_format: 'text', 'json', or 'markdown'
        
    Returns:
        Formatted report string
    """
    # Step 1: Verify wave universe
    universe = verify_wave_universe()
    
    # Step 2: Diagnose all waves
    diagnostics = diagnose_all_waves()
    
    # Generate report based on format
    if output_format == 'json':
        return json.dumps({
            'universe': universe.to_dict(),
            'diagnostics': [d.to_dict() for d in diagnostics],
            'generated_at': datetime.now().isoformat()
        }, indent=2)
    
    elif output_format == 'markdown':
        return _generate_markdown_report(universe, diagnostics)
    
    else:  # text
        return _generate_text_report(universe, diagnostics)


def _generate_text_report(universe: WaveUniverseReport, diagnostics: List[WaveDiagnostic]) -> str:
    """Generate text format report."""
    lines = []
    lines.append("=" * 100)
    lines.append("WAVES INTELLIGENCE™ - COMPREHENSIVE READINESS DIAGNOSTIC REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Section 1: Ground Truth (Wave Universe)
    lines.append("-" * 100)
    lines.append("SECTION 1: GROUND TRUTH - WAVE UNIVERSE VERIFICATION")
    lines.append("-" * 100)
    lines.append(f"Total Waves (Engine):    {universe.total_waves_engine}")
    lines.append(f"Total Waves (Registry):  {universe.total_waves_registry}")
    lines.append(f"Total Waves (Weights):   {universe.total_waves_weights}")
    lines.append(f"Consistency Status:      {'✓ CONSISTENT' if universe.waves_consistent else '✗ INCONSISTENT'}")
    
    if not universe.waves_consistent:
        if universe.missing_in_registry:
            lines.append(f"\nMissing in Registry: {', '.join(universe.missing_in_registry)}")
        if universe.missing_in_weights:
            lines.append(f"Missing in Weights:  {', '.join(universe.missing_in_weights)}")
        if universe.extra_in_weights:
            lines.append(f"Extra in Weights:    {', '.join(universe.extra_in_weights)}")
    
    lines.append("")
    
    # Section 2: Readiness Summary
    lines.append("-" * 100)
    lines.append("SECTION 2: WAVE READINESS SUMMARY")
    lines.append("-" * 100)
    
    status_counts = {'full': 0, 'partial': 0, 'operational': 0, 'unavailable': 0}
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    
    for d in diagnostics:
        status_counts[d.readiness_status] += 1
        grade_counts[d.readiness_grade] += 1
    
    total = len(diagnostics)
    lines.append(f"Total Waves:         {total}")
    lines.append(f"  Full Ready:        {status_counts['full']} ({status_counts['full']/total*100:.1f}%)")
    lines.append(f"  Partial Ready:     {status_counts['partial']} ({status_counts['partial']/total*100:.1f}%)")
    lines.append(f"  Operational:       {status_counts['operational']} ({status_counts['operational']/total*100:.1f}%)")
    lines.append(f"  Unavailable:       {status_counts['unavailable']} ({status_counts['unavailable']/total*100:.1f}%)")
    lines.append("")
    lines.append(f"Readiness Grades:")
    lines.append(f"  A (Excellent):     {grade_counts['A']}")
    lines.append(f"  B (Good):          {grade_counts['B']}")
    lines.append(f"  C (Acceptable):    {grade_counts['C']}")
    lines.append(f"  D (Poor):          {grade_counts['D']}")
    lines.append(f"  F (Failing):       {grade_counts['F']}")
    lines.append("")
    
    # Section 3: Per-Wave Diagnostics
    lines.append("-" * 100)
    lines.append("SECTION 3: PER-WAVE DIAGNOSTIC DETAILS")
    lines.append("-" * 100)
    lines.append(f"{'Wave':<35} {'Status':<12} {'Grade':<6} {'Cov%':<6} {'Days':<6} {'Primary Issue'}")
    lines.append("-" * 100)
    
    for d in sorted(diagnostics, key=lambda x: (x.readiness_status, x.display_name)):
        display_name = d.display_name[:33] + "..." if len(d.display_name) > 34 else d.display_name
        issue = d.primary_failure_reason if d.primary_failure_reason != 'READY' else 'OK'
        
        lines.append(
            f"{display_name:<35} {d.readiness_status:<12} {d.readiness_grade:<6} "
            f"{d.coverage_pct:>5.1f}% {d.history_days:>5} {issue}"
        )
    
    lines.append("")
    
    # Section 4: Root Cause Analysis
    lines.append("-" * 100)
    lines.append("SECTION 4: ROOT CAUSE ANALYSIS")
    lines.append("-" * 100)
    
    # Count failure reasons
    failure_counts = {}
    for d in diagnostics:
        reason = d.primary_failure_reason
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    
    lines.append("Failure Distribution:")
    for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"  {reason:<30} {count:>3} waves ({pct:>5.1f}%)")
    
    lines.append("")
    
    # Section 5: Recommended Actions
    lines.append("-" * 100)
    lines.append("SECTION 5: RECOMMENDED ACTIONS")
    lines.append("-" * 100)
    
    if failure_counts.get('MISSING_PRICES', 0) > 0:
        lines.append(f"1. CRITICAL: Run analytics pipeline for {failure_counts['MISSING_PRICES']} waves with MISSING_PRICES")
        lines.append(f"   Command: python analytics_pipeline.py --all-waves --lookback=14")
        lines.append("")
    
    if failure_counts.get('LOW_COVERAGE', 0) > 0:
        lines.append(f"2. HIGH: Investigate ticker download failures for {failure_counts['LOW_COVERAGE']} waves with LOW_COVERAGE")
        lines.append(f"   Review ticker diagnostics and circuit breaker logs")
        lines.append("")
    
    if failure_counts.get('INSUFFICIENT_HISTORY', 0) > 0:
        lines.append(f"3. MEDIUM: Extend history for {failure_counts['INSUFFICIENT_HISTORY']} waves")
        lines.append(f"   Command: python analytics_pipeline.py --all-waves --lookback=365")
        lines.append("")
    
    if failure_counts.get('MISSING_NAV', 0) > 0:
        lines.append(f"4. LOW: Generate NAV data for {failure_counts['MISSING_NAV']} waves")
        lines.append(f"   NAV generation should happen automatically in analytics pipeline")
        lines.append("")
    
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def _generate_markdown_report(universe: WaveUniverseReport, diagnostics: List[WaveDiagnostic]) -> str:
    """Generate markdown format report."""
    lines = []
    lines.append("# WAVES Intelligence™ - Comprehensive Readiness Diagnostic Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Section 1
    lines.append("## 1. Ground Truth - Wave Universe Verification")
    lines.append("")
    lines.append(f"- **Total Waves (Engine):** {universe.total_waves_engine}")
    lines.append(f"- **Total Waves (Registry):** {universe.total_waves_registry}")
    lines.append(f"- **Total Waves (Weights):** {universe.total_waves_weights}")
    lines.append(f"- **Consistency:** {'✓ CONSISTENT' if universe.waves_consistent else '✗ INCONSISTENT'}")
    lines.append("")
    
    # Section 2
    lines.append("## 2. Wave Readiness Summary")
    lines.append("")
    
    status_counts = {'full': 0, 'partial': 0, 'operational': 0, 'unavailable': 0}
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    
    for d in diagnostics:
        status_counts[d.readiness_status] += 1
        grade_counts[d.readiness_grade] += 1
    
    total = len(diagnostics)
    lines.append(f"**Total Waves:** {total}")
    lines.append("")
    lines.append("### By Status")
    lines.append(f"- Full Ready: {status_counts['full']} ({status_counts['full']/total*100:.1f}%)")
    lines.append(f"- Partial Ready: {status_counts['partial']} ({status_counts['partial']/total*100:.1f}%)")
    lines.append(f"- Operational: {status_counts['operational']} ({status_counts['operational']/total*100:.1f}%)")
    lines.append(f"- Unavailable: {status_counts['unavailable']} ({status_counts['unavailable']/total*100:.1f}%)")
    lines.append("")
    
    lines.append("### By Grade")
    lines.append(f"- A (Excellent): {grade_counts['A']}")
    lines.append(f"- B (Good): {grade_counts['B']}")
    lines.append(f"- C (Acceptable): {grade_counts['C']}")
    lines.append(f"- D (Poor): {grade_counts['D']}")
    lines.append(f"- F (Failing): {grade_counts['F']}")
    lines.append("")
    
    # Section 3
    lines.append("## 3. Per-Wave Diagnostic Details")
    lines.append("")
    lines.append("| Wave | Status | Grade | Coverage | Days | Primary Issue |")
    lines.append("|------|--------|-------|----------|------|---------------|")
    
    for d in sorted(diagnostics, key=lambda x: (x.readiness_status, x.display_name)):
        issue = d.primary_failure_reason if d.primary_failure_reason != 'READY' else 'OK'
        lines.append(
            f"| {d.display_name} | {d.readiness_status} | {d.readiness_grade} | "
            f"{d.coverage_pct:.1f}% | {d.history_days} | {issue} |"
        )
    
    lines.append("")
    
    # Section 4
    lines.append("## 4. Root Cause Analysis")
    lines.append("")
    
    failure_counts = {}
    for d in diagnostics:
        reason = d.primary_failure_reason
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    
    lines.append("### Failure Distribution")
    lines.append("")
    for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"- **{reason}**: {count} waves ({pct:.1f}%)")
    
    lines.append("")
    
    # Section 5
    lines.append("## 5. Recommended Actions")
    lines.append("")
    
    action_num = 1
    if failure_counts.get('MISSING_PRICES', 0) > 0:
        lines.append(f"{action_num}. **CRITICAL**: Run analytics pipeline for {failure_counts['MISSING_PRICES']} waves with MISSING_PRICES")
        lines.append(f"   ```bash")
        lines.append(f"   python analytics_pipeline.py --all-waves --lookback=14")
        lines.append(f"   ```")
        lines.append("")
        action_num += 1
    
    if failure_counts.get('LOW_COVERAGE', 0) > 0:
        lines.append(f"{action_num}. **HIGH**: Investigate ticker download failures for {failure_counts['LOW_COVERAGE']} waves with LOW_COVERAGE")
        lines.append(f"   - Review ticker diagnostics and circuit breaker logs")
        lines.append("")
        action_num += 1
    
    return "\n".join(lines)


if __name__ == '__main__':
    """Generate and print comprehensive readiness report."""
    import sys
    
    # Determine output format from command line
    output_format = 'text'
    if len(sys.argv) > 1:
        format_arg = sys.argv[1].lower()
        if format_arg in ['json', 'markdown', 'md', 'text', 'txt']:
            output_format = 'markdown' if format_arg == 'md' else format_arg
            output_format = 'text' if format_arg == 'txt' else output_format
    
    # Generate and print report
    report = generate_readiness_report(output_format=output_format)
    print(report)
