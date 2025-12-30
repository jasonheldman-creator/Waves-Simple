"""
Diagnostics Artifact Generator

Generates diagnostics_run.json on each snapshot build with:
- Timestamp
- Count of waves processed
- Success/Degraded/Unavailable counts
- Top failure reasons
- List of broken tickers
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter


def generate_diagnostics_artifact(
    snapshot_df=None,
    broken_tickers: Optional[List[str]] = None,
    failure_reasons: Optional[Dict[str, int]] = None,
    output_path: str = "data/diagnostics_run.json"
) -> Dict[str, Any]:
    """
    Generate diagnostics artifact from snapshot data.
    
    Args:
        snapshot_df: DataFrame with snapshot data (must have Data_Regime_Tag column)
        broken_tickers: List of tickers that failed to fetch
        failure_reasons: Dictionary mapping failure reason to count
        output_path: Where to save the diagnostics JSON
        
    Returns:
        Diagnostics dictionary
    """
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "snapshot_build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "waves_processed": 0,
        "status_counts": {
            "Full": 0,
            "Partial": 0,
            "Operational": 0,
            "Unavailable": 0
        },
        "top_failure_reasons": [],
        "broken_tickers": [],
        "summary": ""
    }
    
    # Process snapshot data
    if snapshot_df is not None and not snapshot_df.empty:
        diagnostics["waves_processed"] = len(snapshot_df)
        
        # Count status levels
        if "Data_Regime_Tag" in snapshot_df.columns:
            status_counts = snapshot_df["Data_Regime_Tag"].value_counts().to_dict()
            for status in ["Full", "Partial", "Operational", "Unavailable"]:
                diagnostics["status_counts"][status] = status_counts.get(status, 0)
    
    # Add broken tickers
    if broken_tickers:
        diagnostics["broken_tickers"] = sorted(list(set(broken_tickers)))
    
    # Add top failure reasons
    if failure_reasons:
        # Sort by count descending
        sorted_reasons = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)
        diagnostics["top_failure_reasons"] = [
            {"reason": reason, "count": count}
            for reason, count in sorted_reasons[:10]  # Top 10
        ]
    
    # Generate summary
    total_waves = diagnostics["waves_processed"]
    full_count = diagnostics["status_counts"]["Full"]
    partial_count = diagnostics["status_counts"]["Partial"]
    operational_count = diagnostics["status_counts"]["Operational"]
    unavailable_count = diagnostics["status_counts"]["Unavailable"]
    
    summary_parts = []
    if total_waves > 0:
        summary_parts.append(f"{total_waves} waves processed")
        summary_parts.append(f"{full_count} Full ({full_count/total_waves*100:.0f}%)")
        summary_parts.append(f"{partial_count} Partial ({partial_count/total_waves*100:.0f}%)")
        summary_parts.append(f"{operational_count} Operational ({operational_count/total_waves*100:.0f}%)")
        summary_parts.append(f"{unavailable_count} Unavailable ({unavailable_count/total_waves*100:.0f}%)")
    
    if broken_tickers:
        summary_parts.append(f"{len(diagnostics['broken_tickers'])} broken tickers")
    
    diagnostics["summary"] = ", ".join(summary_parts)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to file
    try:
        with open(output_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save diagnostics artifact: {e}")
    
    return diagnostics


def load_diagnostics_artifact(path: str = "data/diagnostics_run.json") -> Optional[Dict[str, Any]]:
    """
    Load the most recent diagnostics artifact.
    
    Args:
        path: Path to diagnostics file
        
    Returns:
        Diagnostics dictionary or None if not found
    """
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load diagnostics artifact: {e}")
        return None


def extract_diagnostics_from_snapshot(snapshot_df, broken_tickers_path: str = "data/broken_tickers.csv"):
    """
    Extract diagnostics information from snapshot DataFrame and broken tickers file.
    
    Args:
        snapshot_df: Snapshot DataFrame with Data_Regime_Tag column
        broken_tickers_path: Path to broken tickers CSV
        
    Returns:
        Tuple of (broken_tickers list, failure_reasons dict)
    """
    broken_tickers = []
    failure_reasons = {}
    
    # Load broken tickers if file exists
    if os.path.exists(broken_tickers_path):
        try:
            import pandas as pd
            broken_df = pd.read_csv(broken_tickers_path)
            if "Ticker" in broken_df.columns:
                broken_tickers = broken_df["Ticker"].dropna().unique().tolist()
            
            # Extract failure reasons if available
            if "Reason" in broken_df.columns:
                reason_counts = broken_df["Reason"].value_counts().to_dict()
                failure_reasons = reason_counts
        except Exception as e:
            print(f"Warning: Failed to load broken tickers: {e}")
    
    # If snapshot has Missing_Data_Reasons, extract those too
    if snapshot_df is not None and "Missing_Data_Reasons" in snapshot_df.columns:
        for reasons_str in snapshot_df["Missing_Data_Reasons"].dropna():
            if isinstance(reasons_str, str) and reasons_str:
                # Parse reasons (comma-separated)
                for reason in reasons_str.split(","):
                    reason = reason.strip()
                    if reason:
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    return broken_tickers, failure_reasons


if __name__ == "__main__":
    # Test diagnostics generation
    import pandas as pd
    
    # Create sample snapshot data
    sample_data = {
        "Wave_ID": ["sp500_wave", "russell_3000_wave", "crypto_l1_growth_wave"],
        "Display_Name": ["S&P 500 Wave", "Russell 3000 Wave", "Crypto L1 Growth Wave"],
        "Data_Regime_Tag": ["Full", "Partial", "Operational"],
    }
    
    df = pd.DataFrame(sample_data)
    
    diagnostics = generate_diagnostics_artifact(
        snapshot_df=df,
        broken_tickers=["INVALID-TICKER", "BAD-DATA"],
        failure_reasons={"No data available": 5, "API timeout": 3, "Invalid ticker": 2}
    )
    
    print("Generated diagnostics artifact:")
    print(json.dumps(diagnostics, indent=2))
