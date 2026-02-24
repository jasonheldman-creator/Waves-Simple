"""
runtime_bootstrap.py
Runtime stabilization helpers for the Waves Intelligence Console.

Ensures session_state keys are initialized to empty DataFrames before
any tab rendering, and provides safe_df() to guarantee compute outputs
are always pandas DataFrames (never None).
"""

import pandas as pd


def safe_df(df):
    """Return df if it is a pandas DataFrame, otherwise return an empty DataFrame."""
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def initialize_intelligence_state():
    """Initialize all intelligence session_state keys to empty DataFrames if not already set."""
    import streamlit as st

    defaults = {
        "alpha_quality_df": pd.DataFrame(),
        "capital_pressure_df": pd.DataFrame(),
        "rotation_velocity_df": pd.DataFrame(),
        "alpha_ignition_df": pd.DataFrame(),
        "adaptive_stability_df": pd.DataFrame(),
        "adaptive_learning_df": pd.DataFrame(),
        "cross_horizon_df": pd.DataFrame(),
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
