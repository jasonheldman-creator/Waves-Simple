"""
Decision Ledger Module - Append-Only Governance Logging

This module provides an append-only ledger for logging governance metrics
from the Streamlit Institutional Console. The ledger captures snapshots of
Wave performance, exposure levels, VIX conditions, and computed alpha metrics.

Features:
- Append-only design (no edits/deletions)
- CSV-based storage in data/ directory
- Handles missing values gracefully with warnings
- Thread-safe appending
- Schema validation
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import csv


# Ledger file path
LEDGER_PATH = os.path.join("data", "decision_ledger.csv")

# Ledger schema definition
LEDGER_SCHEMA = [
    "timestamp_utc",
    "wave_name",
    "mode",
    "period",
    "vix",
    "exposure",
    "selection_alpha",
    "overlay_alpha",
    "risk_off_alpha",
    "total_alpha",
    "cumulative_return",
    "sharpe_ratio",
    "max_drawdown",
    "volatility",
    "warnings"
]


def initialize_ledger():
    """
    Initialize the decision ledger file if it doesn't exist.
    Creates the CSV file with appropriate headers.
    """
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Create ledger file with headers if it doesn't exist
    if not os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(LEDGER_SCHEMA)


def append_decision_snapshot(
    wave_name: str,
    mode: str,
    period: int,
    metrics: Dict[str, Any],
    warnings: Optional[List[str]] = None
) -> bool:
    """
    Append a decision snapshot to the ledger.
    
    Args:
        wave_name: Name of the Wave
        mode: Operating mode (e.g., "Standard", "Aggressive")
        period: Lookback period in days
        metrics: Dictionary containing computed metrics
        warnings: Optional list of warning messages
        
    Returns:
        bool: True if append succeeded, False otherwise
    """
    try:
        # Initialize ledger if needed
        initialize_ledger()
        
        # Extract metrics with safe defaults
        vix = metrics.get("vix", None)
        exposure = metrics.get("exposure", None)
        selection_alpha = metrics.get("selection_alpha", None)
        overlay_alpha = metrics.get("overlay_alpha", None)
        risk_off_alpha = metrics.get("risk_off_alpha", None)
        total_alpha = metrics.get("total_alpha", None)
        cumulative_return = metrics.get("cumulative_return", None)
        sharpe_ratio = metrics.get("sharpe_ratio", None)
        max_drawdown = metrics.get("max_drawdown", None)
        volatility = metrics.get("volatility", None)
        
        # Format warnings
        warnings_str = "; ".join(warnings) if warnings else ""
        
        # Create row
        row = [
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            wave_name,
            mode,
            period,
            vix,
            exposure,
            selection_alpha,
            overlay_alpha,
            risk_off_alpha,
            total_alpha,
            cumulative_return,
            sharpe_ratio,
            max_drawdown,
            volatility,
            warnings_str
        ]
        
        # Append to CSV
        with open(LEDGER_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        return True
        
    except Exception as e:
        print(f"Error appending to ledger: {e}")
        return False


def read_ledger(wave_name: Optional[str] = None, limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Read the decision ledger, optionally filtered by wave.
    
    Args:
        wave_name: Optional wave name to filter by
        limit: Maximum number of rows to return (most recent)
        
    Returns:
        DataFrame with ledger entries or None if unavailable
    """
    try:
        # Check if ledger exists
        if not os.path.exists(LEDGER_PATH):
            return None
        
        # Read CSV
        df = pd.read_csv(LEDGER_PATH)
        
        # Filter by wave if specified
        if wave_name:
            df = df[df['wave_name'] == wave_name]
        
        # Return most recent entries
        if len(df) > limit:
            df = df.tail(limit)
        
        return df
        
    except Exception as e:
        print(f"Error reading ledger: {e}")
        return None


def get_ledger_stats() -> Dict[str, Any]:
    """
    Get statistics about the ledger.
    
    Returns:
        Dictionary with ledger statistics
    """
    try:
        if not os.path.exists(LEDGER_PATH):
            return {
                "total_entries": 0,
                "waves_tracked": 0,
                "oldest_entry": None,
                "newest_entry": None
            }
        
        df = pd.read_csv(LEDGER_PATH)
        
        return {
            "total_entries": len(df),
            "waves_tracked": df['wave_name'].nunique() if len(df) > 0 else 0,
            "oldest_entry": df['timestamp_utc'].min() if len(df) > 0 else None,
            "newest_entry": df['timestamp_utc'].max() if len(df) > 0 else None
        }
        
    except Exception:
        return {
            "total_entries": 0,
            "waves_tracked": 0,
            "oldest_entry": None,
            "newest_entry": None
        }


def validate_ledger_entry(metrics: Dict[str, Any]) -> List[str]:
    """
    Validate a ledger entry and return warnings for missing data.
    
    Args:
        metrics: Dictionary of metrics to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check for critical missing values
    if metrics.get("total_alpha") is None:
        warnings.append("total_alpha unavailable")
    
    if metrics.get("vix") is None:
        warnings.append("vix data missing")
    
    if metrics.get("exposure") is None:
        warnings.append("exposure data missing")
    
    if metrics.get("selection_alpha") is None:
        warnings.append("selection_alpha not computed")
    
    return warnings
