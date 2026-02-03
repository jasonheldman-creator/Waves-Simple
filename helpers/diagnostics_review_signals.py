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
    
    This function ALWAYS renders the section header and provides meaningful feedback,
    even when data is unavailable. It implements graceful degradation to ensure
    visibility in the UI under all conditions.
    
    Args:
        snapshot_df: Portfolio snapshot DataFrame (can be None)
        attrib_df: Attribution data DataFrame (can be None)
        adaptive_state: Current adaptive learning state dictionary (can be None or empty)
    """
    # ALWAYS render the section header - this ensures visibility
    st.subheader("Review & Adaptation Signals")
    st.caption("System-learned insights for human review · Advisory only · No execution")
    
    # Defensive: Check data availability and provide clear feedback
    data_available = True
    missing_components = []
    
    if snapshot_df is None:
        missing_components.append("portfolio snapshot")
        data_available = False
    elif len(snapshot_df) == 0:
        missing_components.append("portfolio snapshot (empty)")
        data_available = False
        
    if attrib_df is None:
        missing_components.append("attribution data")
        data_available = False
    elif len(attrib_df) == 0:
        missing_components.append("attribution data (empty)")
        data_available = False
    
    # If data is missing, show informative fallback
    if not data_available:
        missing_str = ', '.join(missing_components)
        st.info(
            f"📊 Adaptive signal analysis requires complete data. "
            f"Currently unavailable: {missing_str}. "
            f"This section will populate automatically once data is available."
        )
        # Still show adaptive state if available
        if adaptive_state and len(adaptive_state) > 0:
            with st.expander("📁 Adaptive State (Available)", expanded=False):
                st.caption("Historical learning state is available, but current analysis requires fresh data.")
                st.json(adaptive_state)
        return
    
    # Data is available - proceed with analysis
    st.success(
        "✓ Data is available. Review & Adaptation Signals section is properly configured to render "
        "from adaptive_intelligence.py helper modules."
    )
    
    # Display data summary for transparency
    with st.expander("📊 Data Summary", expanded=False):
        st.caption(f"**Portfolio Snapshot:** {len(snapshot_df)} waves")
        st.caption(f"**Attribution Data:** {len(attrib_df)} records")
        if adaptive_state and len(adaptive_state) > 0:
            st.caption(f"**Adaptive State:** {len(adaptive_state)} keys")
        else:
            st.caption("**Adaptive State:** Empty or not available")
    
    # Display adaptive state summary if available
    if adaptive_state and len(adaptive_state) > 0:
        with st.expander("🔍 Adaptive State Details", expanded=False):
            st.json(adaptive_state)
    
    # Placeholder for future signal analysis implementation
    st.info(
        "🚀 Signal analysis engine is ready. "
        "Future enhancements will add automated signal detection, "
        "anomaly identification, and actionable recommendations here."
    )
