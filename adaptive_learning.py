"""
adaptive_learning.py

Stub module for WAVES adaptive learning capabilities.
Provides minimal implementations to support app_min.py.
"""

import json
from pathlib import Path
import pandas as pd


def load_adaptive_state():
    """
    Load adaptive state from persistent storage.
    Returns empty state if file doesn't exist.
    """
    state_file = Path("data/adaptive_state.json")
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    
    # Default empty state
    return {
        "initialized": True,
        "last_update": None,
        "learning_history": [],
        "tilt_proposals": [],
        "scenario_results": {}
    }


def update_adaptive_state(snapshot_df, attrib_df, adaptive_state):
    """
    Update adaptive state based on current snapshot and attribution data.
    Returns updated state and learning messages.
    """
    messages = []
    
    # Minimal update logic - just record that an update occurred
    if adaptive_state is None:
        adaptive_state = load_adaptive_state()
    
    # Save updated state
    state_file = Path("data/adaptive_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(state_file, "w") as f:
            json.dump(adaptive_state, f, indent=2)
        messages.append("Adaptive state updated and persisted")
    except Exception as e:
        messages.append(f"Warning: Could not persist adaptive state: {e}")
    
    return adaptive_state, messages


def compute_scenario_simulation(scenario, snapshot_df, attrib_df):
    """
    Compute scenario simulation results.
    Returns simulation results dictionary.
    """
    return {
        "scenario": scenario,
        "status": "simulated",
        "impact": 0.0,
        "confidence": 0.0,
        "notes": "Scenario simulation placeholder"
    }


def compute_cross_horizon_agreement(snapshot_df, attrib_df):
    """
    Compute agreement across different time horizons.
    Returns list of analysis dictionaries for cross-horizon comparison.
    """
    # Return empty list if insufficient data
    if snapshot_df is None or snapshot_df.empty:
        return []
    
    # Check for required alpha columns
    required_cols = ["alpha_30d", "alpha_365d"]
    if not all(col in snapshot_df.columns for col in required_cols):
        return []
    
    # Compute mean values for 30D and 365D horizons
    alpha_30d = snapshot_df["alpha_30d"].dropna()
    alpha_365d = snapshot_df["alpha_365d"].dropna()
    
    if len(alpha_30d) == 0 or len(alpha_365d) == 0:
        return []
    
    mean_30d = alpha_30d.mean()
    mean_365d = alpha_365d.mean()
    
    # Determine agreement status
    sign_30d = 1 if mean_30d > 0 else -1
    sign_365d = 1 if mean_365d > 0 else -1
    
    if sign_30d == sign_365d:
        if sign_30d > 0:
            agreement = "Aligned Positive"
        else:
            agreement = "Aligned Negative"
        suppress_action = False
    else:
        agreement = "Divergent"
        suppress_action = True
    
    # Prepare interpretation
    if agreement == "Aligned Positive":
        interpretation = "Short-term and long-term signals both show positive momentum."
    elif agreement == "Aligned Negative":
        interpretation = "Short-term and long-term signals both show negative momentum."
    else:
        interpretation = "Short-term and long-term signals diverge. Exercise caution."
    
    # Return as list containing single analysis dict
    return [{
        "comparison": "30D vs 365D Alpha",
        "agreement": agreement,
        "short_term": mean_30d,
        "long_term": mean_365d,
        "suppress_action": suppress_action,
        "interpretation": interpretation
    }]


def compute_derived_signals(snapshot_df, attrib_df):
    """
    Compute derived signals from snapshot and attribution data.
    Returns dictionary of derived signals.
    """
    signals = {}
    
    if snapshot_df is not None and not snapshot_df.empty:
        # Compute basic signals from snapshot data
        if "alpha_30d" in snapshot_df.columns:
            alpha_30d = snapshot_df["alpha_30d"].dropna()
            if len(alpha_30d) > 0:
                signals["mean_alpha_30d"] = alpha_30d.mean()
        
        if "alpha_365d" in snapshot_df.columns:
            alpha_365d = snapshot_df["alpha_365d"].dropna()
            if len(alpha_365d) > 0:
                signals["mean_alpha_365d"] = alpha_365d.mean()
    
    return signals


def generate_adaptive_tilt_proposals(signals, adaptive_state, cross_horizon_agreements):
    """
    Generate adaptive tilt proposals based on signals and state.
    Returns list of tilt proposal dictionaries.
    """
    return []


def save_adaptive_state(adaptive_state):
    """
    Save adaptive state to persistent storage.
    """
    if adaptive_state is None:
        return
    
    state_file = Path("data/adaptive_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(state_file, "w") as f:
            json.dump(adaptive_state, f, indent=2)
    except Exception as e:
        # Silently fail - this is not critical
        pass
