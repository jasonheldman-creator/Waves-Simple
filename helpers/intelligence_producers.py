"""
WAVES Intelligence Producers
----------------------------

Safe baseline intelligence producers.

Rules:
- No UI output
- Fail-open behavior
- Only writes to st.session_state
- Registered automatically by intelligence_boot
"""

from __future__ import annotations

import streamlit as st
from helpers.intelligence_boot import register_intelligence


# ============================================================
# SAFE HELPER
# ============================================================

def _safe_setdefault(key: str, value):
    """Safely set session_state defaults."""
    try:
        if key not in st.session_state:
            st.session_state[key] = value
    except Exception:
        # Never allow producer failure to break app
        pass


# ============================================================
# BASELINE PRODUCER
# ============================================================

@register_intelligence
def build_executive_summary():
    """
    Minimal producer used to confirm registry execution.
    """
    _safe_setdefault(
        "executive_summary",
        "System initialized. Intelligence boot successful."
    )