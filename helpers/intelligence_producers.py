from __future__ import annotations
import streamlit as st
from helpers.intelligence_boot import register_intelligence


@register_intelligence
def bootstrap_marker():
    """
    Minimal producer used only to verify registry stability.
    """
    if "executive_summary" not in st.session_state:
        st.session_state["executive_summary"] = (
            "System initialized successfully."
        )