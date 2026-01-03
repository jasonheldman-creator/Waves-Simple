"""
Ticker Diagnostics Module

Provides detailed diagnostics for failed tickers in the data pipeline.
Includes failure categorization, structured reporting, and remediation suggestions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
import csv
import os
from pathlib import Path


class FailureType(Enum):
    """Categorization of ticker failure types."""
    SYMBOL_INVALID = "SYMBOL_INVALID"  # Invalid or delisted ticker
    SYMBOL_NEEDS_NORMALIZATION = "SYMBOL_NEEDS_NORMALIZATION"  # Formatting issues
    RATE_LIMIT = "RATE_LIMIT"  # API rate limit exceeded
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"  # Network/connection timeout
    PROVIDER_EMPTY = "PROVIDER_EMPTY"  # Empty response from provider
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"  # Not enough historical data
    UNKNOWN_ERROR = "UNKNOWN_ERROR"  # Uncategorized error


@dataclass
class FailedTickerReport:
    """Structured report for a failed ticker."""
    ticker_original: str
    ticker_normalized: str
    wave_id: Optional[str] = None
    wave_name: Optional[str] = None
    source: str = "yfinance"
    failure_type: FailureType = FailureType.UNKNOWN_ERROR
    error_message: str = ""
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    is_fatal: bool = True
    suggested_fix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            'ticker_original': self.ticker_original,
            'ticker_normalized': self.ticker_normalized,
            'wave_id': self.wave_id or '',
            'wave_name': self.wave_name or '',
            'source': self.source,
            'failure_type': self.failure_type.value,
            'error_message': self.error_message,
            'first_seen': self.first_seen.isoformat() if self.first_seen else '',
            'last_seen': self.last_seen.isoformat() if self.last_seen else '',
            'is_fatal': self.is_fatal,
            'suggested_fix': self.suggested_fix,
        }


class TickerDiagnosticsTracker:
    """Tracks and aggregates ticker failures across the system."""
    
    def __init__(self):
        self.failed_tickers: Dict[str, FailedTickerReport] = {}
        self.reports_dir = "./reports"
        
    def record_failure(self, report: FailedTickerReport) -> None:
        """
        Record a ticker failure.
        
        Args:
            report: FailedTickerReport instance
        """
        key = f"{report.ticker_original}:{report.wave_id or 'global'}"
        
        if key in self.failed_tickers:
            # Update existing record
            existing = self.failed_tickers[key]
            existing.last_seen = report.last_seen or datetime.now()
            existing.error_message = report.error_message  # Update with latest error
        else:
            # New failure
            if not report.first_seen:
                report.first_seen = datetime.now()
            if not report.last_seen:
                report.last_seen = datetime.now()
            self.failed_tickers[key] = report
    
    def get_all_failures(self) -> List[FailedTickerReport]:
        """Get all recorded failures."""
        return list(self.failed_tickers.values())
    
    def get_failures_by_wave(self, wave_id: str) -> List[FailedTickerReport]:
        """Get failures for a specific wave."""
        return [r for r in self.failed_tickers.values() if r.wave_id == wave_id]
    
    def get_failures_by_type(self, failure_type: FailureType) -> List[FailedTickerReport]:
        """Get failures of a specific type."""
        return [r for r in self.failed_tickers.values() if r.failure_type == failure_type]
    
    def clear(self) -> None:
        """Clear all recorded failures."""
        self.failed_tickers.clear()
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export failed ticker reports to CSV.
        
        Args:
            filename: Optional custom filename. If None, generates timestamp-based name.
            
        Returns:
            Path to the created CSV file
        """
        # Ensure reports directory exists
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"failed_tickers_report_{timestamp}.csv"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        # Write CSV
        if not self.failed_tickers:
            # Empty report
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'ticker_original', 'ticker_normalized', 'wave_id', 'wave_name',
                    'source', 'failure_type', 'error_message', 'first_seen', 
                    'last_seen', 'is_fatal', 'suggested_fix'
                ])
            return filepath
        
        # Get all reports and convert to dicts
        reports = [r.to_dict() for r in self.failed_tickers.values()]
        
        # Write to CSV
        with open(filepath, 'w', newline='') as f:
            fieldnames = [
                'ticker_original', 'ticker_normalized', 'wave_id', 'wave_name',
                'source', 'failure_type', 'error_message', 'first_seen', 
                'last_seen', 'is_fatal', 'suggested_fix'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reports)
        
        return filepath
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of failures."""
        total = len(self.failed_tickers)
        
        if total == 0:
            return {
                'total_failures': 0,
                'by_type': {},
                'fatal_count': 0,
                'non_fatal_count': 0,
                'unique_tickers': 0,
            }
        
        by_type = {}
        for failure_type in FailureType:
            count = len(self.get_failures_by_type(failure_type))
            if count > 0:
                by_type[failure_type.value] = count
        
        fatal_count = sum(1 for r in self.failed_tickers.values() if r.is_fatal)
        non_fatal_count = total - fatal_count
        
        unique_tickers = len(set(r.ticker_original for r in self.failed_tickers.values()))
        
        return {
            'total_failures': total,
            'by_type': by_type,
            'fatal_count': fatal_count,
            'non_fatal_count': non_fatal_count,
            'unique_tickers': unique_tickers,
        }


def categorize_error(error_message: str, ticker: str = "") -> tuple[FailureType, str]:
    """
    Categorize an error message into a failure type and suggest a fix.
    
    Args:
        error_message: The error message to categorize
        ticker: The ticker symbol (for context)
        
    Returns:
        Tuple of (FailureType, suggested_fix)
    """
    error_lower = error_message.lower()
    
    # Check for rate limiting
    if any(keyword in error_lower for keyword in ['rate limit', 'too many requests', '429', 'quota']):
        return (
            FailureType.RATE_LIMIT,
            "Implement exponential backoff and retry logic. Consider batching requests with delays."
        )
    
    # Check for network issues
    if any(keyword in error_lower for keyword in ['timeout', 'connection', 'network', 'timed out']):
        return (
            FailureType.NETWORK_TIMEOUT,
            "Retry with exponential backoff. Check network connectivity and API availability."
        )
    
    # Check for empty/no data
    if any(keyword in error_lower for keyword in ['empty', 'no data', 'not found', 'no close column']):
        # Could be delisted or invalid ticker
        if ticker and '.' in ticker:
            return (
                FailureType.SYMBOL_NEEDS_NORMALIZATION,
                f"Try normalizing ticker symbol: replace '.' with '-' (e.g., {ticker} → {ticker.replace('.', '-')})"
            )
        else:
            return (
                FailureType.PROVIDER_EMPTY,
                "Verify ticker symbol is valid and not delisted. Check if ticker is available on the data provider."
            )
    
    # Check for insufficient history
    if any(keyword in error_lower for keyword in ['insufficient', 'not enough', 'limited history']):
        return (
            FailureType.INSUFFICIENT_HISTORY,
            "Ticker may be newly listed. Reduce lookback period or wait for more historical data to accumulate."
        )
    
    # Check for symbol normalization issues
    if '.' in ticker and 'close' not in error_lower:
        return (
            FailureType.SYMBOL_NEEDS_NORMALIZATION,
            f"Try normalizing ticker symbol: replace '.' with '-' (e.g., {ticker} → {ticker.replace('.', '-')})"
        )
    
    # Default to unknown
    return (
        FailureType.UNKNOWN_ERROR,
        "Review error message and check ticker validity. Consult data provider documentation."
    )


# Global singleton instance
_global_tracker: Optional[TickerDiagnosticsTracker] = None


def get_diagnostics_tracker() -> TickerDiagnosticsTracker:
    """Get the global diagnostics tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TickerDiagnosticsTracker()
    return _global_tracker


def load_broken_tickers_from_csv(csv_path: str = "data/broken_tickers.csv") -> List[Dict[str, Any]]:
    """
    Load broken tickers from the standard broken_tickers.csv file.
    
    Args:
        csv_path: Path to the broken tickers CSV file
        
    Returns:
        List of dictionaries with ticker information including:
        - ticker_original: Original ticker symbol
        - ticker_normalized: Normalized ticker symbol
        - failure_type: Type of failure
        - error_message: Error message
        - impacted_waves: Comma-separated list of impacted waves
        - suggested_fix: Suggested fix for the issue
        - first_seen: First seen timestamp
        - last_seen: Last seen timestamp
        - is_fatal: Whether the failure is fatal
        - failure_count: Number of waves this ticker fails in
    """
    import pandas as pd
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        
        # Sanitize and validate data
        broken_tickers = []
        for _, row in df.iterrows():
            # Validate required fields
            ticker = str(row.get('ticker_original', '')).strip()
            if not ticker:
                continue
            
            # Parse impacted waves
            impacted_waves_str = str(row.get('impacted_waves', '')).strip()
            impacted_waves = [w.strip() for w in impacted_waves_str.split(',') if w.strip()] if impacted_waves_str else []
            
            # Create entry
            entry = {
                'ticker_original': ticker,
                'ticker_normalized': str(row.get('ticker_normalized', ticker)).strip(),
                'failure_type': str(row.get('failure_type', 'UNKNOWN_ERROR')).strip(),
                'error_message': str(row.get('error_message', '')).strip(),
                'impacted_waves': impacted_waves,
                'impacted_waves_str': impacted_waves_str,
                'suggested_fix': str(row.get('suggested_fix', '')).strip(),
                'first_seen': str(row.get('first_seen', '')).strip(),
                'last_seen': str(row.get('last_seen', '')).strip(),
                'is_fatal': bool(row.get('is_fatal', True)),
                'failure_count': len(impacted_waves)
            }
            broken_tickers.append(entry)
        
        # Sort by failure count descending
        broken_tickers.sort(key=lambda x: x['failure_count'], reverse=True)
        
        return broken_tickers
        
    except Exception as e:
        print(f"Warning: Failed to load broken tickers from {csv_path}: {e}")
        return []


def export_failed_tickers_to_cache(broken_tickers: List[Dict[str, Any]], 
                                    output_path: str = "data/cache/failed_tickers.csv") -> bool:
    """
    Export failed tickers to a CSV file in the cache directory.
    
    Args:
        broken_tickers: List of broken ticker dictionaries
        output_path: Path to the output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            if not broken_tickers:
                # Write header only
                writer = csv.writer(f)
                writer.writerow([
                    'ticker_original', 'ticker_normalized', 'failure_type', 
                    'error_message', 'failure_count', 'impacted_waves',
                    'suggested_fix', 'first_seen', 'last_seen', 'is_fatal'
                ])
            else:
                # Write data
                fieldnames = [
                    'ticker_original', 'ticker_normalized', 'failure_type', 
                    'error_message', 'failure_count', 'impacted_waves',
                    'suggested_fix', 'first_seen', 'last_seen', 'is_fatal'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                for ticker in broken_tickers:
                    # Create a copy and convert impacted_waves list to string
                    row = ticker.copy()
                    row['impacted_waves'] = row.get('impacted_waves_str', 
                                                    ', '.join(row.get('impacted_waves', [])))
                    writer.writerow(row)
        
        return True
        
    except Exception as e:
        print(f"Error exporting failed tickers to {output_path}: {e}")
        return False
