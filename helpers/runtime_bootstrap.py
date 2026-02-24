"""
runtime_bootstrap.py
Runtime stabilization helpers for the Waves Intelligence Console.

Ensures session_state keys are initialized to empty DataFrames before
any tab rendering, and provides safe_df() to guarantee compute outputs
are always pandas DataFrames (never None).
"""

import logging
import pandas as pd


def safe_df(df, context: str = ""):
    """Return df if it is a non-empty pandas DataFrame; otherwise log a warning and return empty.

    Validation contract:
    - Always returns a pandas DataFrame (never None or other type).
    - Logs a warning when the input is not a DataFrame or is empty so that
      silent data loss is surfaced in the application log.
    - Returns an empty DataFrame (not None) when no valid data is available
      so that downstream callers can detect emptiness without AttributeError.
    """
    if not isinstance(df, pd.DataFrame):
        logging.warning(
            "[safe_df] Expected DataFrame but received %s%s — returning empty DataFrame.",
            type(df).__name__,
            f" ({context})" if context else "",
        )
        return pd.DataFrame()
    if df.empty:
        logging.warning(
            "[safe_df] Received empty DataFrame%s — returning empty DataFrame.",
            f" ({context})" if context else "",
        )
    return df


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
