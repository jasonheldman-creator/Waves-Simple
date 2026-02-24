"""
helpers/adaptive_pipeline.py
Adaptive Intelligence data pipeline.

Loads canonical attribution data and returns a non-empty DataFrame.
Falls back to synthetic bootstrap data when the canonical source is
unavailable so that Adaptive Intelligence panels always have data to render.
"""

import os

import pandas as pd


def run_adaptive_pipeline() -> pd.DataFrame:
    """Load or generate attribution data for the Adaptive Intelligence panels.

    Execution order:
    1. Try to load ``data/alpha_attribution_summary.csv``.
    2. If unavailable or empty, generate synthetic data via
       :func:`helpers.bootstrap_data.generate_adaptive_data`.

    Also recomputes cross-horizon stability data and updates the
    ``cross_horizon_df`` session_state key so that the Cross-Horizon
    Stability panel renders populated data.

    Returns
    -------
    pd.DataFrame
        Non-empty attribution DataFrame with schema:
        wave, horizon, total_alpha, selection_alpha, momentum_alpha,
        volatility_alpha, regime_alpha, exposure_alpha, residual_alpha.
    """
    import streamlit as st
    import adaptive_learning as al
    from helpers.runtime_bootstrap import safe_df
    from helpers.bootstrap_data import generate_adaptive_data

    attrib_path = os.path.join("data", "alpha_attribution_summary.csv")
    attrib_df: pd.DataFrame = pd.DataFrame()

    if os.path.exists(attrib_path):
        try:
            attrib_df = pd.read_csv(attrib_path)
            attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
        except Exception:
            attrib_df = pd.DataFrame()

    if attrib_df.empty:
        attrib_df = generate_adaptive_data()

    # Update cross_horizon_df session_state key
    try:
        if "wave" in attrib_df.columns and "horizon" in attrib_df.columns:
            cross_horizon = safe_df(
                attrib_df.groupby(["wave", "horizon"]).mean(numeric_only=True).reset_index()
            )
            st.session_state["cross_horizon_df"] = cross_horizon
    except Exception:
        pass

    return attrib_df
