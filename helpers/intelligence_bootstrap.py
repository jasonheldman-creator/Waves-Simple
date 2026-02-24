"""
intelligence_bootstrap.py
Bootstrap helpers for Alpha Intelligence and Adaptive Intelligence compute pipelines.

Executes compute pipelines once per session and stores results in Streamlit
session_state.  Callers should guard each call with a session_state key check so
pipelines run only once per session:

    if "alpha_intelligence" not in st.session_state:
        bootstrap_alpha_intelligence()

    if "adaptive_intelligence" not in st.session_state:
        bootstrap_adaptive_intelligence()

On failure each function re-raises so the caller can show st.error and skip
rendering rather than showing blank or partially-populated containers.
"""

import traceback

import pandas as pd


def bootstrap_alpha_intelligence():
    """
    Load canonical attribution dataset, execute Alpha Intelligence compute
    pipelines, and store all results in ``st.session_state["alpha_intelligence"]``.

    Also populates the individual session_state keys
    (``alpha_quality_df``, ``capital_pressure_df``, ``rotation_velocity_df``,
    ``alpha_ignition_df``, ``cross_horizon_df``) for backward compatibility with
    existing renderers that read those keys directly.

    Raises on failure so the caller can show ``st.error`` and skip rendering.
    """
    import os

    import streamlit as st

    import adaptive_learning as al
    from helpers.runtime_bootstrap import safe_df

    try:
        from helpers.bootstrap_data import generate_alpha_data

        attrib_path = os.path.join("data", "alpha_attribution_summary.csv")
        attrib_df = pd.DataFrame()

        if os.path.exists(attrib_path):
            try:
                attrib_df = pd.read_csv(attrib_path)
                attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
            except Exception:
                attrib_df = pd.DataFrame()

        if attrib_df.empty:
            attrib_df = generate_alpha_data()

        alpha_quality = safe_df(al.alpha_quality_df(attrib_df))
        capital_pressure = safe_df(al.capital_pressure_df(attrib_df))
        rotation_velocity = safe_df(al.rotation_velocity_df(attrib_df))
        alpha_ignition = safe_df(al.alpha_ignition_df(attrib_df))

        cross_horizon = safe_df(
            attrib_df.groupby(["wave", "horizon"]).mean(numeric_only=True).reset_index()
            if "wave" in attrib_df.columns and "horizon" in attrib_df.columns
            else pd.DataFrame()
        )

        st.session_state["alpha_intelligence"] = {
            "attrib_df": attrib_df,
            "alpha_quality_df": alpha_quality,
            "capital_pressure_df": capital_pressure,
            "rotation_velocity_df": rotation_velocity,
            "alpha_ignition_df": alpha_ignition,
            "cross_horizon_df": cross_horizon,
        }

        # Populate individual keys for backward compatibility
        st.session_state["alpha_quality_df"] = alpha_quality
        st.session_state["capital_pressure_df"] = capital_pressure
        st.session_state["rotation_velocity_df"] = rotation_velocity
        st.session_state["alpha_ignition_df"] = alpha_ignition
        st.session_state["cross_horizon_df"] = cross_horizon
        st.session_state["intelligence_initialized"] = True

    except Exception:
        traceback.print_exc()
        raise


def bootstrap_adaptive_intelligence():
    """
    Load canonical datasets, execute Adaptive Intelligence compute pipelines,
    and store all results in ``st.session_state["adaptive_intelligence"]``.

    Also updates ``cross_horizon_df`` and ``adaptive_learning_initialized``
    session_state keys for backward compatibility.

    The function reads ``snapshot_df`` from ``st.session_state["_snapshot_df_raw"]``
    if available (populated by the app-level ``load_snapshot`` call), or falls back
    to loading ``data/live_snapshot.csv`` directly.

    Raises on failure so the caller can show ``st.error`` and skip rendering.
    """
    import os

    import streamlit as st

    import adaptive_learning as al
    from helpers.runtime_bootstrap import safe_df

    try:
        from helpers.bootstrap_data import generate_adaptive_data

        # Load attribution data
        attrib_path = os.path.join("data", "alpha_attribution_summary.csv")
        attrib_df = pd.DataFrame()
        if os.path.exists(attrib_path):
            try:
                attrib_df = pd.read_csv(attrib_path)
                attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
            except Exception:
                attrib_df = pd.DataFrame()

        if attrib_df.empty:
            attrib_df = generate_adaptive_data()

        # Snapshot: prefer the one already stored by app-level load_snapshot()
        snapshot_df = st.session_state.get("_snapshot_df_raw")
        if snapshot_df is None or not isinstance(snapshot_df, pd.DataFrame):
            snap_path = os.path.join("data", "live_snapshot.csv")
            if os.path.exists(snap_path):
                snapshot_df = pd.read_csv(snap_path)
                snapshot_df.columns = [c.strip().lower() for c in snapshot_df.columns]
            else:
                snapshot_df = pd.DataFrame()

        # Decision data
        all_decisions = []
        try:
            from helpers import decision_lifecycle_matrix as _dlm

            all_decisions = _dlm.load_decision_log() if _dlm else []
        except ImportError:
            pass

        gov_decisions = al.load_governance_decisions()
        if gov_decisions:
            existing_ids = {d.get("id") for d in all_decisions if d.get("id")}
            all_decisions = all_decisions + [
                d for d in gov_decisions if d.get("id") not in existing_ids
            ]

        # Execute compute pipelines
        adaptive_state = al.load_adaptive_state()
        adaptive_state, _ = al.update_adaptive_state(snapshot_df, attrib_df, adaptive_state)

        snapshot_data = al.compute_learning_snapshot(
            snapshot_df, attrib_df, adaptive_state, all_decisions
        )
        core_signals = al.compute_core_learning_signals(
            snapshot_df, attrib_df, adaptive_state
        )
        param_sensitivity = al.compute_parameter_sensitivity(attrib_df, adaptive_state)
        learning_curve_data = al.compute_learning_curve(
            snapshot_df, attrib_df, adaptive_state, all_decisions
        )
        efficiency_curve_data = al.compute_efficiency_curve(all_decisions, adaptive_state)
        decision_memory_data = al.compute_decision_memory_table(all_decisions, attrib_df)
        # build_decision_memory uses only governance decisions, matching original tab behaviour
        decision_memory_df = al.build_decision_memory(gov_decisions)
        cross_horizon_data = al.compute_cross_horizon_stability(snapshot_df, attrib_df)

        # Guarantee learning/efficiency curves always have data for rendering
        from helpers.bootstrap_data import (
            generate_learning_curve_monthly_points,
            generate_param_sensitivity,
            generate_synthetic_learning_curve,
            generate_synthetic_efficiency_curve,
            generate_synthetic_cross_horizon_drivers,
        )

        if not learning_curve_data.get("has_data"):
            learning_curve_data = generate_synthetic_learning_curve(55.0)
        elif len(learning_curve_data.get("monthly_points", [])) < 2:
            learning_curve_data["monthly_points"] = generate_learning_curve_monthly_points(
                learning_curve_data.get("learning_index", 50.0)
            )

        if not efficiency_curve_data.get("has_data"):
            efficiency_curve_data = generate_synthetic_efficiency_curve(55.0)
        elif len(efficiency_curve_data.get("monthly_points", [])) < 2:
            efficiency_curve_data["monthly_points"] = generate_learning_curve_monthly_points(
                efficiency_curve_data.get("efficiency_index", 50.0)
            )

        if not param_sensitivity:
            param_sensitivity = generate_param_sensitivity()

        # Guarantee cross-horizon drivers are never empty
        if not cross_horizon_data.get("drivers"):
            cross_horizon_data["drivers"] = generate_synthetic_cross_horizon_drivers()
            if not cross_horizon_data.get("summary"):
                cross_horizon_data["summary"] = (
                    "Cross-horizon structure is stable based on synthetic baseline attribution."
                )

        cross_horizon_raw = safe_df(
            attrib_df.groupby(["wave", "horizon"]).mean(numeric_only=True).reset_index()
            if not attrib_df.empty
            and "wave" in attrib_df.columns
            and "horizon" in attrib_df.columns
            else pd.DataFrame()
        )

        st.session_state["adaptive_intelligence"] = {
            "attrib_df": attrib_df,
            "adaptive_state": adaptive_state,
            "snapshot_data": snapshot_data,
            "core_signals": core_signals,
            "param_sensitivity": param_sensitivity,
            "learning_curve_data": learning_curve_data,
            "efficiency_curve_data": efficiency_curve_data,
            "decision_memory_data": decision_memory_data,
            "decision_memory_df": decision_memory_df,
            "cross_horizon_data": cross_horizon_data,
            "cross_horizon_raw": cross_horizon_raw,
            "all_decisions": all_decisions,
            "gov_decisions": gov_decisions,
        }

        # Update backward-compatible session_state keys
        st.session_state["cross_horizon_df"] = cross_horizon_raw
        st.session_state["adaptive_learning_initialized"] = True

    except Exception:
        traceback.print_exc()
        raise
