"""
helpers/alpha_pipeline.py
Alpha Intelligence data pipeline.

Loads canonical attribution data and returns a non-empty DataFrame.
Falls back to synthetic bootstrap data when the canonical source is
unavailable so that Alpha Intelligence panels always have data to render.
"""

import os

import pandas as pd


def run_alpha_pipeline() -> pd.DataFrame:
    """Load or generate attribution data for the Alpha Intelligence panels.

    Execution order:
    1. Try to load ``data/alpha_attribution_summary.csv``.
    2. If unavailable or empty, generate synthetic data via
       :func:`helpers.bootstrap_data.generate_alpha_data`.

    Also populates the individual Streamlit session_state keys
    (``alpha_quality_df``, ``capital_pressure_df``, ``rotation_velocity_df``,
    ``alpha_ignition_df``) so that panel renderers which read directly from
    session_state see populated DataFrames.

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
    from helpers.bootstrap_data import generate_alpha_data

    attrib_path = os.path.join("data", "alpha_attribution_summary.csv")
    attrib_df: pd.DataFrame = pd.DataFrame()

    if os.path.exists(attrib_path):
        try:
            attrib_df = pd.read_csv(attrib_path)
            attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
        except Exception:
            attrib_df = pd.DataFrame()

    if attrib_df.empty:
        attrib_df = generate_alpha_data()

    # Populate individual session_state keys for panel renderers
    try:
        st.session_state["alpha_quality_df"] = safe_df(al.alpha_quality_df(attrib_df))
        st.session_state["capital_pressure_df"] = safe_df(al.capital_pressure_df(attrib_df))
        st.session_state["rotation_velocity_df"] = safe_df(al.rotation_velocity_df(attrib_df))
        st.session_state["alpha_ignition_df"] = safe_df(al.alpha_ignition_df(attrib_df))
    except Exception:
        pass

    return attrib_df
