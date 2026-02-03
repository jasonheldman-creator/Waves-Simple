"""
Adaptive Intelligence — Alpha Quality & Confidence Module

Provides IC-grade Alpha Quality and Confidence diagnostics.
Safe for Streamlit rendering. No trading logic.
"""

import pandas as pd
import numpy as np
import streamlit as st
from helpers.diagnostics_review_signals import get_review_signals


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


def render_review_and_adaptation_signals(
    snapshot_df=None,
    attrib_df=None,
    adaptive_state=None
):
    """
    Render Review & Adaptation Signals section.
    
    This is a read-only, observational section that surfaces persistent
    diagnostic signals without changing execution, parameters, or behavior.
    
    Args:
        snapshot_df: Portfolio snapshot DataFrame (optional)
        attrib_df: Attribution DataFrame (optional)
        adaptive_state: Adaptive learning state dictionary (optional)
    """
    st.subheader("Review & Adaptation Signals")
    st.caption("Persistent diagnostic indicators · Observational only · No execution changes")
    st.markdown("")
    
    # Get diagnostic signals
    signals = get_review_signals(snapshot_df, attrib_df, adaptive_state)
    
    if not signals:
        st.info("Accumulating data for review signals. Signals will appear as system data becomes available.")
        return
    
    # Render each signal
    for signal in signals:
        with st.expander(f"**{signal['title']}** — {signal['status']}", expanded=False):
            cols = st.columns([1, 1])
            
            with cols[0]:
                st.markdown(f"**Status:** {signal['status']}")
                st.markdown(f"**Scope:** {signal['scope']}")
            
            with cols[1]:
                # Status indicator color
                if signal['status'] == "Stable":
                    status_color = "🟢"
                elif signal['status'] == "Review Recommended":
                    status_color = "🟡"
                else:  # Monitoring
                    status_color = "🔵"
                
                st.markdown(f"{status_color} **Observation:** {signal['observation']}")
            
            st.caption(f"ℹ️ {signal['actionability']}")
