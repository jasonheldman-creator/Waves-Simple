"""
Review & Adaptation Signals Helper

This module provides helper functions for rendering the Review & Adaptation Signals
section in the Adaptive Intelligence tab. It ensures rendering logic stays out of
app_min.py and lives in proper helper modules.
"""

import pandas as pd
import streamlit as st


def render_review_and_adaptation_signals(snapshot_df, attrib_df, adaptive_state):
    """
    Render the Review & Adaptation Signals section.
    
    This function encapsulates all rendering logic for adaptive signals review,
    keeping app_min.py clean and focused on orchestration.
    
    Args:
        snapshot_df: Portfolio snapshot DataFrame
        attrib_df: Attribution data DataFrame  
        adaptive_state: Current adaptive learning state dictionary
    """
    st.subheader("Review & Adaptation Signals")
    st.caption("System-learned insights for human review · Advisory only · No execution")
    
    # Defensive check: Handle missing data gracefully
    if snapshot_df is None and attrib_df is None:
        st.info(
            "📊 No portfolio or attribution data available at this time. "
            "Review & Adaptation Signals will be displayed once data is loaded."
        )
        return
    
    # Show partial data warning if only one dataset is available
    data_status = []
    if snapshot_df is None:
        data_status.append("portfolio snapshot")
    if attrib_df is None:
        data_status.append("attribution data")
    
    if data_status:
        st.warning(
            f"⚠️ Partial data available: Missing {', '.join(data_status)}. "
            "Signals shown below may be limited."
        )
    
    # Display basic signal information
    st.markdown("**Signal Categories**")
    
    # Compute basic signals from available data
    signal_rows = []
    
    if snapshot_df is not None and not snapshot_df.empty:
        # Count active waves
        wave_count = len(snapshot_df)
        signal_rows.append({
            "Category": "Portfolio Coverage",
            "Signal": f"{wave_count} active waves in portfolio",
            "Status": "✓ Available" if wave_count > 0 else "⚠️ No waves"
        })
        
        # Check for alpha data
        alpha_cols = [col for col in snapshot_df.columns if 'alpha' in col.lower()]
        if alpha_cols:
            avg_alpha_30d = snapshot_df.get('alpha_30d', pd.Series([0])).mean()
            if pd.notna(avg_alpha_30d):
                alpha_status = "Positive" if avg_alpha_30d > 0 else "Negative"
                signal_rows.append({
                    "Category": "Portfolio Alpha (30D)",
                    "Signal": f"Average alpha: {avg_alpha_30d*100:.2f}%",
                    "Status": f"✓ {alpha_status}"
                })
    
    if attrib_df is not None and not attrib_df.empty:
        # Count attribution records
        attrib_count = len(attrib_df)
        signal_rows.append({
            "Category": "Attribution Coverage",
            "Signal": f"{attrib_count} attribution records available",
            "Status": "✓ Available"
        })
        
        # Check for dominant drivers
        if 'selection_alpha' in attrib_df.columns:
            dominant_driver = attrib_df.nlargest(1, 'selection_alpha')
            if not dominant_driver.empty:
                wave_name = dominant_driver.iloc[0].get('wave', 'Unknown')
                signal_rows.append({
                    "Category": "Top Contributor",
                    "Signal": f"Wave: {wave_name}",
                    "Status": "✓ Identified"
                })
    
    # Display signals in a table
    if signal_rows:
        signals_df = pd.DataFrame(signal_rows)
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "✨ Adaptive signals are being computed from available data. "
            "This section will display actionable insights as the system learns."
        )
    
    # Display adaptive state summary if available
    if adaptive_state and isinstance(adaptive_state, dict) and adaptive_state:
        with st.expander("Adaptive State Overview", expanded=False):
            st.caption("Current adaptive learning state from data/adaptive_state.json")
            st.json(adaptive_state)
    
    st.caption(
        "ℹ️ All signals are derived from historical data and portfolio snapshots. "
        "No automated execution occurs from this panel."
    )
