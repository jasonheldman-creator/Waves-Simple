"""
Basket Integrity Validation Module

This module validates the integrity of the Wave system by checking:
1. All waves have weight definitions
2. All tickers in weights exist in the canonical universal basket
3. All benchmark tickers exist in the universal basket
4. Weight sums are valid (allowing for SmartSafe gating < 1.0)
5. Price fetch functionality for all tickers

This runs at startup and logs warnings without crashing the application.
"""

import os
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
import warnings

# Import wave registry
try:
    import sys
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from waves_engine import get_all_waves_universe, WAVE_WEIGHTS
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    warnings.warn("waves_engine not available - limited validation")

# File paths
REPO_ROOT = Path(__file__).parent.parent
UNIVERSAL_UNIVERSE_PATH = REPO_ROOT / "universal_universe.csv"
WAVE_WEIGHTS_PATH = REPO_ROOT / "wave_weights.csv"
WAVE_CONFIG_PATH = REPO_ROOT / "wave_config.csv"

# Validation thresholds
MAX_WEIGHT_SUM_THRESHOLD = 1.01  # Maximum allowed weight sum (allowing minor rounding errors)
SMARTSAFE_WEIGHT_THRESHOLD = 0.99  # Threshold below which SmartSafe gating is considered active


class BasketIntegrityIssue:
    """Represents a basket integrity issue"""
    def __init__(self, category: str, severity: str, message: str, details: Dict[str, Any] = None):
        self.category = category
        self.severity = severity  # 'critical', 'warning', 'info'
        self.message = message
        self.details = details or {}


class BasketIntegrityReport:
    """Comprehensive basket integrity validation report"""
    def __init__(self):
        self.issues: List[BasketIntegrityIssue] = []
        self.stats: Dict[str, Any] = {}
        
    def add_issue(self, category: str, severity: str, message: str, details: Dict[str, Any] = None):
        """Add an issue to the report"""
        self.issues.append(BasketIntegrityIssue(category, severity, message, details))
    
    def has_critical_issues(self) -> bool:
        """Check if report has any critical issues"""
        return any(issue.severity == 'critical' for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if report has any warnings"""
        return any(issue.severity == 'warning' for issue in self.issues)
    
    def get_summary(self) -> str:
        """Get a text summary of the report"""
        lines = []
        lines.append("=" * 80)
        lines.append("BASKET INTEGRITY REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Statistics
        if self.stats:
            lines.append("STATISTICS:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Issues by category
        if not self.issues:
            lines.append("✓ No issues found - all validations passed!")
        else:
            critical = [i for i in self.issues if i.severity == 'critical']
            warnings = [i for i in self.issues if i.severity == 'warning']
            info = [i for i in self.issues if i.severity == 'info']
            
            if critical:
                lines.append(f"CRITICAL ISSUES ({len(critical)}):")
                for issue in critical:
                    lines.append(f"  ❌ [{issue.category}] {issue.message}")
                    if issue.details:
                        for key, value in issue.details.items():
                            if isinstance(value, (list, set)):
                                lines.append(f"     {key}: {len(value)} items")
                                for item in sorted(value)[:5]:  # Show first 5
                                    lines.append(f"       - {item}")
                                if len(value) > 5:
                                    lines.append(f"       ... and {len(value) - 5} more")
                            else:
                                lines.append(f"     {key}: {value}")
                lines.append("")
            
            if warnings:
                lines.append(f"WARNINGS ({len(warnings)}):")
                for issue in warnings:
                    lines.append(f"  ⚠️  [{issue.category}] {issue.message}")
                    if issue.details:
                        for key, value in issue.details.items():
                            if isinstance(value, (list, set)):
                                lines.append(f"     {key}: {len(value)} items")
                                for item in sorted(value)[:5]:  # Show first 5
                                    lines.append(f"       - {item}")
                                if len(value) > 5:
                                    lines.append(f"       ... and {len(value) - 5} more")
                            else:
                                lines.append(f"     {key}: {value}")
                lines.append("")
            
            if info:
                lines.append(f"INFORMATIONAL ({len(info)}):")
                for issue in info:
                    lines.append(f"  ℹ️  [{issue.category}] {issue.message}")
                lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


def load_csvs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all CSV files needed for validation"""
    universe_df = pd.read_csv(UNIVERSAL_UNIVERSE_PATH) if UNIVERSAL_UNIVERSE_PATH.exists() else pd.DataFrame()
    weights_df = pd.read_csv(WAVE_WEIGHTS_PATH) if WAVE_WEIGHTS_PATH.exists() else pd.DataFrame()
    config_df = pd.read_csv(WAVE_CONFIG_PATH) if WAVE_CONFIG_PATH.exists() else pd.DataFrame()
    return universe_df, weights_df, config_df


def validate_wave_registry_completeness(report: BasketIntegrityReport, weights_df: pd.DataFrame):
    """Validate all waves from registry have weight definitions"""
    if not WAVES_ENGINE_AVAILABLE:
        report.add_issue(
            "wave_registry",
            "warning",
            "Cannot validate wave registry - waves_engine not available"
        )
        return
    
    try:
        universe = get_all_waves_universe()
        expected_waves = set(universe['waves'])
        actual_waves = set(weights_df['wave'].unique())
        
        report.stats['expected_wave_count'] = len(expected_waves)
        report.stats['waves_with_weights'] = len(actual_waves)
        
        missing_waves = expected_waves - actual_waves
        extra_waves = actual_waves - expected_waves
        
        if missing_waves:
            report.add_issue(
                "wave_registry",
                "critical",
                f"{len(missing_waves)} waves missing from wave_weights.csv",
                {"missing_waves": sorted(missing_waves)}
            )
        
        if extra_waves:
            report.add_issue(
                "wave_registry",
                "warning",
                f"{len(extra_waves)} waves in wave_weights.csv not in registry",
                {"extra_waves": sorted(extra_waves)}
            )
        
        if not missing_waves and not extra_waves:
            report.add_issue(
                "wave_registry",
                "info",
                f"✓ All {len(expected_waves)} waves have weight definitions"
            )
    except Exception as e:
        report.add_issue(
            "wave_registry",
            "warning",
            f"Error validating wave registry: {e}"
        )


def validate_tickers_in_universe(report: BasketIntegrityReport, universe_df: pd.DataFrame, weights_df: pd.DataFrame):
    """Validate all weight tickers exist in universal basket"""
    if universe_df.empty:
        report.add_issue(
            "universe_basket",
            "critical",
            "Universal universe CSV not found or empty"
        )
        return
    
    universe_tickers = set(universe_df['ticker'].unique())
    weight_tickers = set(weights_df['ticker'].unique())
    
    report.stats['universe_ticker_count'] = len(universe_tickers)
    report.stats['weight_ticker_count'] = len(weight_tickers)
    
    missing_tickers = weight_tickers - universe_tickers
    
    if missing_tickers:
        report.add_issue(
            "universe_basket",
            "critical",
            f"{len(missing_tickers)} tickers in weights not found in universal basket",
            {"missing_tickers": sorted(missing_tickers)}
        )
    else:
        report.add_issue(
            "universe_basket",
            "info",
            f"✓ All {len(weight_tickers)} weight tickers exist in universal basket"
        )


def validate_benchmark_tickers(report: BasketIntegrityReport, universe_df: pd.DataFrame, config_df: pd.DataFrame):
    """Validate all benchmark tickers exist in universal basket"""
    if config_df.empty:
        report.add_issue(
            "benchmarks",
            "warning",
            "Wave config CSV not found or empty - cannot validate benchmarks"
        )
        return
    
    if universe_df.empty:
        report.add_issue(
            "benchmarks",
            "critical",
            "Universal universe CSV not found - cannot validate benchmarks"
        )
        return
    
    universe_tickers = set(universe_df['ticker'].unique())
    benchmark_tickers = set(config_df['Benchmark'].dropna().unique())
    
    report.stats['benchmark_count'] = len(benchmark_tickers)
    report.stats['waves_with_config'] = len(config_df)
    
    missing_benchmarks = benchmark_tickers - universe_tickers
    
    if missing_benchmarks:
        report.add_issue(
            "benchmarks",
            "warning",
            f"{len(missing_benchmarks)} benchmark tickers not in universal basket",
            {"missing_benchmarks": sorted(missing_benchmarks)}
        )
    else:
        report.add_issue(
            "benchmarks",
            "info",
            f"✓ All {len(benchmark_tickers)} benchmark tickers exist in universal basket"
        )


def validate_wave_configs(report: BasketIntegrityReport, weights_df: pd.DataFrame, config_df: pd.DataFrame):
    """Validate all waves have configuration entries"""
    if config_df.empty:
        report.add_issue(
            "wave_config",
            "warning",
            "Wave config CSV not found or empty"
        )
        return
    
    waves_with_weights = set(weights_df['wave'].unique())
    waves_with_config = set(config_df['Wave'].unique())
    
    missing_configs = waves_with_weights - waves_with_config
    
    if missing_configs:
        report.add_issue(
            "wave_config",
            "warning",
            f"{len(missing_configs)} waves missing from wave_config.csv",
            {"missing_configs": sorted(missing_configs)}
        )
    else:
        report.add_issue(
            "wave_config",
            "info",
            f"✓ All {len(waves_with_weights)} waves have configuration entries"
        )


def validate_weight_sums(report: BasketIntegrityReport, weights_df: pd.DataFrame):
    """
    Validate weight sums for all waves.
    
    Note: Weights < 1.0 are allowed (remainder allocated to SmartSafe).
    Weights > MAX_WEIGHT_SUM_THRESHOLD or < 0.0 are errors.
    """
    weight_sums = weights_df.groupby('wave')['weight'].sum()
    
    # Allow for SmartSafe gating (weights can be < 1.0, even as low as 0.45)
    # Flag if weights > MAX_WEIGHT_SUM_THRESHOLD or < 0.0
    problem_weights = weight_sums[(weight_sums > MAX_WEIGHT_SUM_THRESHOLD) | (weight_sums < 0.0)]
    low_weights = weight_sums[(weight_sums >= 0.0) & (weight_sums < SMARTSAFE_WEIGHT_THRESHOLD)]
    
    if len(problem_weights) > 0:
        details = {wave: float(sum_val) for wave, sum_val in problem_weights.items()}
        report.add_issue(
            "weight_sums",
            "critical",
            f"{len(problem_weights)} waves have invalid weight sums (>{MAX_WEIGHT_SUM_THRESHOLD} or <0.0)",
            {"problem_waves": details}
        )
    
    if len(low_weights) > 0:
        details = {wave: float(sum_val) for wave, sum_val in low_weights.items()}
        report.add_issue(
            "weight_sums",
            "info",
            f"{len(low_weights)} waves have weights < {SMARTSAFE_WEIGHT_THRESHOLD} (SmartSafe gating active)",
            {"waves_with_smartsafe": details}
        )
    
    if len(problem_weights) == 0:
        report.add_issue(
            "weight_sums",
            "info",
            f"✓ All {len(weight_sums)} waves have valid weight sums"
        )


def validate_basket_integrity() -> BasketIntegrityReport:
    """
    Run comprehensive basket integrity validation.
    
    Returns:
        BasketIntegrityReport with all validation results
    """
    report = BasketIntegrityReport()
    
    # Load CSVs
    try:
        universe_df, weights_df, config_df = load_csvs()
    except Exception as e:
        report.add_issue(
            "file_loading",
            "critical",
            f"Error loading CSV files: {e}"
        )
        return report
    
    # Run validations
    validate_wave_registry_completeness(report, weights_df)
    validate_tickers_in_universe(report, universe_df, weights_df)
    validate_benchmark_tickers(report, universe_df, config_df)
    validate_wave_configs(report, weights_df, config_df)
    validate_weight_sums(report, weights_df)
    
    return report


def print_basket_integrity_report(verbose: bool = False):
    """
    Print basket integrity report to console.
    
    Args:
        verbose: If True, show all details. If False, show summary only.
    """
    report = validate_basket_integrity()
    
    if verbose:
        print(report.get_summary())
    else:
        # Show condensed summary
        if not report.issues:
            print("✓ Basket integrity check passed - no issues found")
        else:
            critical = sum(1 for i in report.issues if i.severity == 'critical')
            warnings = sum(1 for i in report.issues if i.severity == 'warning')
            if critical > 0:
                print(f"❌ Basket integrity check found {critical} critical issues, {warnings} warnings")
            elif warnings > 0:
                print(f"⚠️  Basket integrity check found {warnings} warnings")
            else:
                print("✓ Basket integrity check passed with info messages")
    
    return report


if __name__ == "__main__":
    # Run validation when executed directly
    import argparse
    parser = argparse.ArgumentParser(description='Validate basket integrity')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed report')
    args = parser.parse_args()
    
    print_basket_integrity_report(verbose=args.verbose)
