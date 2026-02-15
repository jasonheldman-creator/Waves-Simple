"""
Adaptive Intelligence — Alpha Quality & Confidence Module

Provides IC-grade Alpha Quality and Confidence diagnostics.
Safe for Streamlit rendering. No trading logic.

This module serves as the centralized rendering layer for all Adaptive Intelligence
features, ensuring app_min.py remains clean and focused on orchestration.
"""

# Ensure reliable imports across Streamlit Cloud and local development
import runtime_path_resolver

import pandas as pd
import numpy as np
import streamlit as st

# Import helper modules for specialized rendering
try:
    from helpers import diagnostics_review_signals
except ImportError:
    diagnostics_review_signals = None

try:
    import adaptive_learning as al
except ImportError:
    al = None


def render_alpha_quality_and_confidence(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    st.subheader("Alpha Quality & Confidence")

    wave_row = snapshot_df[snapshot_df["display_name"] == selected_wave]

    if wave_row.empty:
        st.warning("Wave data not available.")
        return

    wave_row = wave_row.iloc[0]

    # ---------------------------
    # Horizon Alpha
    # ---------------------------
    horizons = ["30d", "60d", "365d"]
    alpha_vals = [
        wave_row[return_cols[h]] - wave_row[benchmark_cols[h]]
        for h in horizons
    ]
    alpha_series = pd.Series(alpha_vals, index=horizons)

    # ---------------------------
    # Residual & Dominant Driver
    # ---------------------------
    if source_df is not None:
        residual = source_df.loc[
            source_df["Alpha Source"] == "Residual Alpha", "Contribution"
        ].values[0]
        dominant_driver = (
            source_df.sort_values("Contribution", ascending=False)
            .iloc[0]["Alpha Source"]
        )
        explained = 1 - abs(residual)
    else:
        residual = None
        dominant_driver = "Not Available"
        explained = None

    # ---------------------------
    # Consistency Score
    # ---------------------------
    consistency = (
        1 - alpha_series.std()
        if alpha_series.notna().all()
        else 0.3
    )

    # ---------------------------
    # Alpha Confidence Index
    # ---------------------------
    if explained is not None:
        aci = int(
            np.clip(
                (explained * 0.5 + consistency * 0.5) * 100,
                0,
                100,
            )
        )
        if aci >= 75:
            aci_label = "High Confidence"
        elif aci >= 50:
            aci_label = "Moderate Confidence"
        else:
            aci_label = "Fragile Alpha"
    else:
        aci = "Not Available"
        aci_label = "Not Available"

    # ---------------------------
    # Summary Table
    # ---------------------------
    summary_df = pd.DataFrame({
        "Metric": [
            "Dominant Driver",
            "Residual Alpha Share",
            "Horizon Consistency",
            "Alpha Confidence Index",
        ],
        "Assessment": [
            dominant_driver,
            f"{residual:.3f}" if residual is not None else "Not Available",
            "Stable" if consistency > 0.7 else "Variable",
            f"{aci} ({aci_label})" if isinstance(aci, int) else aci,
        ],
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ---------------------------
    # IC Narrative
    # ---------------------------
    st.markdown(
        f"""
        **Interpretation**

        • Alpha is primarily driven by **{summary_df.iloc[0]['Assessment']}**  
        • Residual alpha is **{summary_df.iloc[1]['Assessment']}**, indicating disciplined signal structure  
        • Alpha behavior across horizons is **{summary_df.iloc[2]['Assessment']}**  
        • Overall confidence in alpha persistence is **{aci_label}**
        """
    )


def render_adaptive_intelligence_tab(snapshot_df, attrib_df):
    """
    Main rendering function for the Adaptive Intelligence tab.
    
    This function serves as the entry point for all Adaptive Intelligence rendering,
    ensuring the logic lives in adaptive_intelligence.py rather than inline in app_min.py.
    
    Following the existing Adaptive Intelligence render flow pattern, this function
    orchestrates all sub-sections including Review & Adaptation Signals.
    
    Args:
        snapshot_df: Portfolio snapshot DataFrame (can be None)
        attrib_df: Attribution data DataFrame (can be None)
    """
    # ALWAYS render the header - ensures visibility in UI
    st.header("Adaptive Intelligence Center")
    st.caption("Decision support layer · System-learned insights and recommendations · LIVE learning enabled")
    st.markdown("")
    
    # Defensive: Handle None inputs gracefully
    if snapshot_df is None:
        st.warning("Portfolio snapshot data is not available. Some features may be limited.")
    if attrib_df is None:
        st.warning("Attribution data is not available. Some features may be limited.")
    
    # Load adaptive state if available
    adaptive_state = {}
    learning_messages = []
    
    if al is not None:
        try:
            adaptive_state = al.load_adaptive_state()
            # Only update if we have data
            if snapshot_df is not None or attrib_df is not None:
                adaptive_state, learning_messages = al.update_adaptive_state(
                    snapshot_df, attrib_df, adaptive_state
                )
            
            # Show learning updates if any
            if learning_messages:
                with st.expander("Live Learning Updates", expanded=False):
                    for msg in learning_messages:
                        st.caption(msg)
                    st.caption("Adaptive state persisted to data/adaptive_state.json")
        except Exception as e:
            st.warning(f"Could not load adaptive learning state: {e}")
            # Continue rendering even if adaptive learning fails
    
    # Review & Adaptation Signals section - ALWAYS attempt to render
    st.divider()
    
    # Ensure this section is ALWAYS visible, even if there are issues
    if diagnostics_review_signals is not None:
        try:
            # Call the helper with defensive parameters
            diagnostics_review_signals.render_review_and_adaptation_signals(
                snapshot_df, attrib_df, adaptive_state
            )
        except Exception as e:
            # Show error but still render section header
            st.subheader("Review & Adaptation Signals")
            st.error(f"Error rendering Review & Adaptation Signals: {e}")
            st.info("The system encountered an error while generating adaptive signals. Please check the logs or contact support.")
    else:
        # Fallback if module not imported - still show section header
        st.subheader("Review & Adaptation Signals")
        st.info(
            "Review & Adaptation Signals rendering module not found. "
            "Please ensure helpers/diagnostics_review_signals.py is available in the deployment environment."
        )
    
    # Additional sections can be added here following the same pattern
    st.divider()
    st.caption("End of Adaptive Intelligence sections.")