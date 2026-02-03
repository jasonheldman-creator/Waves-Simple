"""
diagnostics_review_signals.py

Provides observational diagnostic signals for human review and adaptation.
Read-only module that surfaces persistent diagnostic indicators without 
changing execution, parameters, or behavior.

All signals are advisory/observational only.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def get_review_signals(
    snapshot_df: Optional[pd.DataFrame] = None,
    attrib_df: Optional[pd.DataFrame] = None,
    adaptive_state: Optional[Dict] = None
) -> List[Dict]:
    """
    Compute review and adaptation signals from current system state.
    
    Args:
        snapshot_df: Portfolio snapshot DataFrame (optional)
        attrib_df: Attribution DataFrame (optional)
        adaptive_state: Adaptive learning state dictionary (optional)
    
    Returns:
        List of signal dictionaries with:
            - title: Signal title
            - status: "Monitoring", "Review Recommended", or "Stable"
            - observation: One-line observation text
            - scope: "Portfolio-level" or "Wave-level"
            - actionability: Always "Observational only — human evaluation recommended"
    """
    signals = []
    
    # Signal 1: Data Availability Status
    signals.append(_compute_data_availability_signal(snapshot_df, attrib_df))
    
    # Signal 2: Wave Count Stability
    if snapshot_df is not None and not snapshot_df.empty:
        signals.append(_compute_wave_count_signal(snapshot_df))
    
    # Signal 3: Attribution Coverage
    if attrib_df is not None and not attrib_df.empty:
        signals.append(_compute_attribution_coverage_signal(attrib_df, snapshot_df))
    
    # Signal 4: Alpha Consistency Across Horizons
    if snapshot_df is not None and not snapshot_df.empty:
        signals.append(_compute_alpha_consistency_signal(snapshot_df))
    
    # Signal 5: Adaptive State Health
    if adaptive_state is not None:
        signals.append(_compute_adaptive_state_signal(adaptive_state))
    
    # Filter out None signals (graceful degradation)
    signals = [s for s in signals if s is not None]
    
    return signals


def _compute_data_availability_signal(
    snapshot_df: Optional[pd.DataFrame],
    attrib_df: Optional[pd.DataFrame]
) -> Dict:
    """Compute signal for data availability status."""
    has_snapshot = snapshot_df is not None and not snapshot_df.empty
    has_attrib = attrib_df is not None and not attrib_df.empty
    
    if has_snapshot and has_attrib:
        status = "Stable"
        observation = "All primary data sources available for analysis"
    elif has_snapshot or has_attrib:
        status = "Review Recommended"
        observation = "Partial data availability — some diagnostic capabilities limited"
    else:
        status = "Monitoring"
        observation = "Awaiting data initialization — diagnostics pending"
    
    return {
        "title": "Data Availability",
        "status": status,
        "observation": observation,
        "scope": "Portfolio-level",
        "actionability": "Observational only — human evaluation recommended"
    }


def _compute_wave_count_signal(snapshot_df: pd.DataFrame) -> Optional[Dict]:
    """Compute signal for wave count stability."""
    try:
        wave_count = len(snapshot_df["wave_name"].unique())
        
        if wave_count >= 5:
            status = "Stable"
            observation = f"Portfolio maintains {wave_count} active waves"
        elif wave_count >= 3:
            status = "Monitoring"
            observation = f"Portfolio has {wave_count} active waves — below typical range"
        else:
            status = "Review Recommended"
            observation = f"Limited wave diversification detected ({wave_count} waves)"
        
        return {
            "title": "Wave Portfolio Composition",
            "status": status,
            "observation": observation,
            "scope": "Portfolio-level",
            "actionability": "Observational only — human evaluation recommended"
        }
    except Exception:
        return None


def _compute_attribution_coverage_signal(
    attrib_df: pd.DataFrame,
    snapshot_df: Optional[pd.DataFrame]
) -> Optional[Dict]:
    """Compute signal for attribution data coverage."""
    try:
        # Count waves with attribution data
        waves_with_attrib = len(attrib_df["wave"].unique()) if "wave" in attrib_df.columns else 0
        total_waves = len(snapshot_df["wave_name"].unique()) if snapshot_df is not None and not snapshot_df.empty else waves_with_attrib
        
        if total_waves == 0:
            return None
        
        coverage_pct = (waves_with_attrib / total_waves) * 100
        
        if coverage_pct >= 90:
            status = "Stable"
            observation = f"Attribution coverage at {coverage_pct:.0f}% ({waves_with_attrib}/{total_waves} waves)"
        elif coverage_pct >= 70:
            status = "Monitoring"
            observation = f"Attribution coverage at {coverage_pct:.0f}% — some gaps present"
        else:
            status = "Review Recommended"
            observation = f"Attribution coverage at {coverage_pct:.0f}% — significant gaps detected"
        
        return {
            "title": "Attribution Data Coverage",
            "status": status,
            "observation": observation,
            "scope": "Portfolio-level",
            "actionability": "Observational only — human evaluation recommended"
        }
    except Exception:
        return None


def _compute_alpha_consistency_signal(snapshot_df: pd.DataFrame) -> Optional[Dict]:
    """Compute signal for alpha consistency across time horizons."""
    try:
        # Check for required columns
        required_cols = ["alpha_30d", "alpha_60d", "alpha_365d"]
        if not all(col in snapshot_df.columns for col in required_cols):
            return None
        
        # Compute cross-horizon alpha agreement
        positive_30d = (snapshot_df["alpha_30d"] > 0).sum()
        positive_60d = (snapshot_df["alpha_60d"] > 0).sum()
        positive_365d = (snapshot_df["alpha_365d"] > 0).sum()
        
        total_waves = len(snapshot_df)
        if total_waves == 0:
            return None
        
        # Agreement: waves positive across all horizons
        all_positive = (
            (snapshot_df["alpha_30d"] > 0) & 
            (snapshot_df["alpha_60d"] > 0) & 
            (snapshot_df["alpha_365d"] > 0)
        ).sum()
        
        agreement_pct = (all_positive / total_waves) * 100
        
        if agreement_pct >= 60:
            status = "Stable"
            observation = f"{agreement_pct:.0f}% of waves show positive alpha across all horizons"
        elif agreement_pct >= 40:
            status = "Monitoring"
            observation = f"{agreement_pct:.0f}% horizon agreement — mixed signal environment"
        else:
            status = "Review Recommended"
            observation = f"Low cross-horizon agreement at {agreement_pct:.0f}% — divergent signals present"
        
        return {
            "title": "Cross-Horizon Alpha Consistency",
            "status": status,
            "observation": observation,
            "scope": "Portfolio-level",
            "actionability": "Observational only — human evaluation recommended"
        }
    except Exception:
        return None


def _compute_adaptive_state_signal(adaptive_state: Dict) -> Optional[Dict]:
    """Compute signal for adaptive learning state health."""
    try:
        has_learning_history = bool(adaptive_state.get("learning_history", []))
        has_scenario_results = bool(adaptive_state.get("scenario_results", {}))
        is_initialized = adaptive_state.get("initialized", False)
        
        if is_initialized and (has_learning_history or has_scenario_results):
            status = "Stable"
            observation = "Adaptive learning state active with historical context"
        elif is_initialized:
            status = "Monitoring"
            observation = "Adaptive state initialized — accumulating learning data"
        else:
            status = "Review Recommended"
            observation = "Adaptive learning state not yet initialized"
        
        return {
            "title": "Adaptive Learning State",
            "status": status,
            "observation": observation,
            "scope": "Portfolio-level",
            "actionability": "Observational only — human evaluation recommended"
        }
    except Exception:
        return None
